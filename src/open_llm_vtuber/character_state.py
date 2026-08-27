"""Character-level persistent state shared across all conversations of one conf.

Relationship and long-term memory belong to the character (``conf_uid``), not to
a single chat history. Conversation transcripts, rolling summaries and recent
context stay per history; this module only owns cross-chat character state.

Storage is deliberately simple: one JSON file per character under
``character_state/<conf_uid>.json`` with atomic replace writes. No database, no
vector index, no embeddings, no autonomous memory agent.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from .agent.context_window import estimate_tokens
from .agent.relationship_context import (
    RelationshipStatus,
    normalize_relationship_status,
)
from .chat_history_manager import _sanitize_path_component

# Conservative target for the character-memory block injected into the prompt.
# Kept inside the 500-1000 estimated token range from the v2 spec.
CHARACTER_MEMORY_MAX_TOKENS = 600

_RELATIONSHIP_RANK: Dict[str, int] = {
    "stranger": 0,
    "familiar": 1,
    "close": 2,
    "dating": 3,
}

_state_locks: Dict[str, threading.RLock] = {}
_state_locks_guard = threading.Lock()


def _get_state_lock(filepath: str) -> threading.RLock:
    with _state_locks_guard:
        return _state_locks.setdefault(filepath, threading.RLock())


def _write_state_atomic(filepath: str, data: Dict[str, Any]) -> None:
    temporary_path = f"{filepath}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, filepath)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


@dataclass
class CharacterState:
    relationship_status: RelationshipStatus = "stranger"
    relationship_updated_at: Optional[str] = None
    relationship_reason: str = "default"
    relationship_migrated: bool = False
    memories: List[Dict[str, Any]] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_character_state_path(conf_uid: str) -> str:
    """Return the on-disk path for a character state file (safe component)."""
    if not conf_uid:
        raise ValueError("conf_uid cannot be empty")
    safe_conf_uid = _sanitize_path_component(conf_uid)
    return os.path.join("character_state", f"{safe_conf_uid}.json")


def load_character_state(conf_uid: str) -> CharacterState:
    """Load character state; missing/corrupt files yield a fresh default state."""
    filepath = get_character_state_path(conf_uid)
    if not os.path.exists(filepath):
        return CharacterState()
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        memories = [
            {
                "text": str(item.get("text", "")).strip(),
                "added_at": str(item.get("added_at", "")),
                "explicit": bool(item.get("explicit", False)),
            }
            for item in data.get("memories", [])
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]
        return CharacterState(
            relationship_status=normalize_relationship_status(
                data.get("relationship_status", "stranger")
            ),
            relationship_updated_at=data.get("relationship_updated_at"),
            relationship_reason=str(data.get("relationship_reason", "default")),
            relationship_migrated=bool(data.get("relationship_migrated", False)),
            memories=memories,
        )
    except Exception as error:
        logger.error(
            "Failed to load character state: error_type={}", type(error).__name__
        )
        return CharacterState()


def save_character_state(conf_uid: str, state: CharacterState) -> bool:
    """Atomically persist character state; returns success."""
    filepath = get_character_state_path(conf_uid)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        lock = _get_state_lock(filepath)
        with lock:
            data = {
                "relationship_status": state.relationship_status,
                "relationship_updated_at": state.relationship_updated_at,
                "relationship_reason": state.relationship_reason,
                "relationship_migrated": state.relationship_migrated,
                "memories": state.memories,
            }
            _write_state_atomic(filepath, data)
        return True
    except Exception as error:
        logger.error(
            "Failed to save character state: error_type={}", type(error).__name__
        )
        return False


def migrate_relationship_if_needed(conf_uid: str, state: CharacterState) -> CharacterState:
    """Backward-compatible migration: per-conversation metadata -> character state.

    Only runs once per character. Existing explicit relationship metadata from
    older conversations is the source: the strongest, most recently updated
    non-stranger status wins. ``stranger`` defaults never migrate into a guess,
    and explicit ``dating`` recorded in metadata is preserved as-is.
    """
    if state.relationship_migrated:
        return state
    if state.relationship_status != "stranger":
        # Already established at character level; just remember that migration
        # ran so we never rescan conversations again.
        state.relationship_migrated = True
        save_character_state(conf_uid, state)
        return state

    best: Optional[tuple[int, str, RelationshipStatus, str]] = None
    conf_dir = os.path.join("chat_history", _sanitize_path_component(conf_uid))
    if os.path.isdir(conf_dir):
        for filename in os.listdir(conf_dir):
            if not filename.endswith(".json"):
                continue
            try:
                with open(
                    os.path.join(conf_dir, filename), "r", encoding="utf-8"
                ) as file:
                    data = json.load(file)
                metadata = (
                    data[0]
                    if data and isinstance(data[0], dict)
                    and data[0].get("role") == "metadata"
                    else {}
                )
                status = normalize_relationship_status(
                    metadata.get("relationship_status", "stranger")
                )
                if status == "stranger":
                    continue
                rank = _RELATIONSHIP_RANK[status]
                updated_at = str(metadata.get("relationship_updated_at", "") or "")
                candidate = (
                    rank,
                    updated_at,
                    status,
                    str(metadata.get("relationship_reason", "migrated")),
                )
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            except Exception:
                continue

    if best is not None:
        state = CharacterState(
            relationship_status=best[2],
            relationship_updated_at=best[1] or _now_iso(),
            relationship_reason=best[3],
            relationship_migrated=True,
            memories=state.memories,
        )
        save_character_state(conf_uid, state)
        logger.info(
            "Relationship migration stats: migrated=True, "
            "relationship_status={}, relationship_update_trigger=legacy_migration",
            state.relationship_status,
        )
    else:
        state.relationship_migrated = True
        save_character_state(conf_uid, state)
        logger.info(
            "Relationship migration stats: migrated=False, "
            "relationship_update_trigger=legacy_migration"
        )
    return state


def set_character_relationship(
    conf_uid: str,
    status: RelationshipStatus,
    trigger: str,
    *,
    updated_at: Optional[str] = None,
) -> Optional[CharacterState]:
    """Persist a character-level relationship update; None on write failure."""
    state = load_character_state(conf_uid)
    state.relationship_status = normalize_relationship_status(status)
    state.relationship_updated_at = updated_at or _now_iso()
    state.relationship_reason = trigger
    state.relationship_migrated = True
    if not save_character_state(conf_uid, state):
        return None
    return state


def _normalize_memory_text(text: str) -> str:
    return " ".join((text or "").lower().split()).strip(" .,!?;:，。！？；：")


def add_character_memory(
    conf_uid: str,
    text: str,
    *,
    explicit: bool = True,
) -> Optional[CharacterState]:
    """Append one long-term fact (deduplicated); None on write failure."""
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return None
    state = load_character_state(conf_uid)
    normalized = _normalize_memory_text(cleaned)
    if any(
        _normalize_memory_text(str(item.get("text", ""))) == normalized
        for item in state.memories
    ):
        return state
    state.memories.append(
        {
            "text": cleaned,
            "added_at": _now_iso(),
            "explicit": bool(explicit),
        }
    )
    if not save_character_state(conf_uid, state):
        return None
    return state


def remove_character_memory(conf_uid: str, text: str) -> Optional[CharacterState]:
    """Remove stored facts overlapping the given text; None on write failure."""
    target = _normalize_memory_text(text)
    if not target:
        return load_character_state(conf_uid)
    state = load_character_state(conf_uid)
    remaining = [
        item
        for item in state.memories
        if not (
            target in _normalize_memory_text(str(item.get("text", "")))
            or _normalize_memory_text(str(item.get("text", ""))) in target
        )
    ]
    if len(remaining) == len(state.memories):
        return state
    state.memories = remaining
    if not save_character_state(conf_uid, state):
        return None
    return state


def reset_character_memory(conf_uid: str) -> Optional[CharacterState]:
    """Clear all long-term memory for a character; None on write failure."""
    state = load_character_state(conf_uid)
    if not state.memories:
        return state
    state.memories = []
    if not save_character_state(conf_uid, state):
        return None
    return state


def reset_character_state(conf_uid: str) -> Optional[CharacterState]:
    """Reset relationship to stranger and clear memory; None on write failure."""
    state = load_character_state(conf_uid)
    state.relationship_status = "stranger"
    state.relationship_updated_at = _now_iso()
    state.relationship_reason = "manual_reset"
    state.relationship_migrated = True
    state.memories = []
    if not save_character_state(conf_uid, state):
        return None
    return state


def build_character_memory_context(
    state: CharacterState,
    *,
    max_tokens: int = CHARACTER_MEMORY_MAX_TOKENS,
) -> str:
    """Return a compact, bounded character-memory block for the system prompt.

    Explicit (manual) memories are prioritized, then the most recent facts.
    Memory is never dumped wholesale; the block stays well under the budget.
    """
    if not state.memories:
        return ""
    ordered = sorted(
        state.memories,
        key=lambda item: (
            not bool(item.get("explicit", False)),
            str(item.get("added_at", "")),
        ),
    )
    lines: List[str] = []
    used_tokens = 0
    for item in ordered:
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        line = f"- {text}"
        line_tokens = estimate_tokens(line) + 4
        if used_tokens + line_tokens > max_tokens:
            break
        lines.append(line)
        used_tokens += line_tokens
    if not lines:
        return ""
    header = "Known long-term context (character memory, shared across all chats):"
    return f"{header}\n" + "\n".join(lines)
