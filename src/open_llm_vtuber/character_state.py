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
from typing import Any, Dict, List, Literal, Optional

from loguru import logger

from .agent.context_window import estimate_tokens
from .agent.relationship_context import (
    RelationshipStatus,
    normalize_relationship_status,
)
from .chat_history_manager import _sanitize_path_component
from .interaction_preferences import (
    apply_preference,
    detect_interaction_preference,
    normalize_stored_preference,
)
from .world_state import memory_age_label

# Conservative target for the character-memory block injected into the prompt.
# Kept inside the 500-1000 estimated token range from the v2 spec.
CHARACTER_MEMORY_MAX_TOKENS = 600

_RELATIONSHIP_RANK: Dict[str, int] = {
    "stranger": 0,
    "familiar": 1,
    "close": 2,
    "dating": 3,
    # Terminal tier above dating. Old state files never contain it, so every
    # previously stored status keeps its exact rank and migration behaviour.
    "married": 4,
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
    goals: List[Dict[str, Any]] = field(default_factory=list)
    # Durable, user-stated constraints on HOW Mili interacts (tone,
    # formality, humor, address, directness). Separate from persona YAML
    # (identity) and from memories (facts): these change the way Mili talks
    # to this user across sessions. See interaction_preferences.py.
    interaction_preferences: List[Dict[str, Any]] = field(default_factory=list)
    # Last known user timezone (IANA name, e.g. "Asia/Jakarta"). Session
    # scope by design: refreshed whenever the frontend sends a valid value,
    # used as fallback when a turn carries no timezone (restart/proactive).
    user_timezone: Optional[str] = None
    # Explicit reminder requests with absolute UTC due times. Same file, same
    # atomic store as memories/goals/preferences — not a new memory system.
    # Old state files without this key load as empty (restart/old-state safe).
    future_intentions: List[Dict[str, Any]] = field(default_factory=list)


GoalStatus = Literal["seed", "active", "done"]
GoalSource = Literal["seed"]

_VALID_GOAL_STATUSES = frozenset({"seed", "active", "done"})


@dataclass(frozen=True)
class Goal:
    """Finite, completable self-model goal (foundation data only).

    Ongoing aspirations stay as static tendencies in self_model.py; a Goal
    always has a verifiable completion criterion (documented per seed in
    ``default_seed_goals``). Transitions are explicit only; there is no
    auto-complete, no reactivation, no planner.

    ``last_evidence_id`` / ``last_evidence_at`` are the dedup anchor for the
    Autonomous Decision Layer: they record which deterministic piece of
    evidence already produced a decision for this goal, so the same evidence
    can never trigger the same goal twice. Both are optional and purely
    observational -- old state files without them load unchanged.
    """

    id: str
    text: str
    status: GoalStatus = "seed"
    created_at: Optional[str] = None
    activated_at: Optional[str] = None
    completed_at: Optional[str] = None
    source: GoalSource = "seed"
    last_evidence_id: Optional[str] = None
    last_evidence_at: Optional[str] = None


def goal_to_dict(goal: Goal) -> Dict[str, Any]:
    data = {
        "id": goal.id,
        "text": goal.text,
        "status": goal.status,
        "created_at": goal.created_at,
        "activated_at": goal.activated_at,
        "completed_at": goal.completed_at,
        "source": goal.source,
    }
    # The evidence anchor is written only once it exists, so a goal that has
    # never produced a decision serialises exactly as it did before the anchor
    # existed and an old state file stays byte-identical after a load/save
    # cycle. Round-trip still holds once an anchor is present.
    if goal.last_evidence_id:
        data["last_evidence_id"] = goal.last_evidence_id
    if goal.last_evidence_at:
        data["last_evidence_at"] = goal.last_evidence_at
    return data


def goal_from_dict(item: Any) -> Optional[Goal]:
    """Tolerant parse; None for entries without id/text (fail-soft)."""
    if not isinstance(item, dict):
        return None
    goal_id = str(item.get("id", "") or "").strip()
    text = str(item.get("text", "") or "").strip()
    if not goal_id or not text:
        return None
    status = str(item.get("status", "seed") or "seed")
    if status not in _VALID_GOAL_STATUSES:
        status = "seed"
    source = str(item.get("source", "seed") or "seed")
    if source != "seed":
        source = "seed"
    return Goal(
        id=goal_id,
        text=text,
        status=status,  # type: ignore[arg-type]
        created_at=item.get("created_at"),
        activated_at=item.get("activated_at"),
        completed_at=item.get("completed_at"),
        source=source,  # type: ignore[arg-type]
        last_evidence_id=item.get("last_evidence_id"),
        last_evidence_at=item.get("last_evidence_at"),
    )


def default_seed_goals(now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """The 3 approved finite seed goals with creation stamp (pure).

    Completion criteria (verified explicitly, never auto-completed):
    - morning-reading-week: 7 consecutive user-local days with a reading entry.
    - try-three-dishes: 3 distinct eating-related evidence items.
    - finish-one-book: explicit user statement that the book is finished.
    """
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    seeds = [
        ("morning-reading-week", "membaca pagi 7 hari berturut-turut"),
        ("try-three-dishes", "mencoba 3 makanan berbeda"),
        ("finish-one-book", "menyelesaikan satu buku"),
    ]
    return [
        {
            "id": goal_id,
            "text": text,
            "status": "seed",
            "created_at": stamp,
            "activated_at": None,
            "completed_at": None,
            "source": "seed",
        }
        for goal_id, text in seeds
    ]


def _with_goal_status(
    goals: List[Dict[str, Any]],
    goal_id: str,
    from_status: str,
    to_status: GoalStatus,
    stamp_field: str,
    now: Optional[datetime],
) -> tuple:
    """Shared explicit-transition helper; (new_list, ok). Never mutates input."""
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp = moment.isoformat(timespec="seconds")
    changed = False
    out: List[Dict[str, Any]] = []
    for item in goals or []:
        if (
            isinstance(item, dict)
            and str(item.get("id", "")) == goal_id
            and str(item.get("status", "seed")) == from_status
            and not changed
        ):
            updated = dict(item)
            updated["status"] = to_status
            updated[stamp_field] = stamp
            out.append(updated)
            changed = True
        else:
            out.append(item)
    return out, changed


def activate_goal(
    goals: List[Dict[str, Any]], goal_id: str, now: Optional[datetime] = None
) -> tuple:
    """seed -> active with activation stamp (pure). Unknown/non-seed: no-op."""
    return _with_goal_status(goals, goal_id, "seed", "active", "activated_at", now)


def complete_goal(
    goals: List[Dict[str, Any]], goal_id: str, now: Optional[datetime] = None
) -> tuple:
    """active -> done with completion stamp (pure). No auto-complete."""
    return _with_goal_status(goals, goal_id, "active", "done", "completed_at", now)


def ensure_seed_goals(
    goals: Any, now: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Return the seeded goal list, seeding only when there is nothing yet.

    Idempotent and backward compatible:

    - an empty/absent/corrupt ``goals`` value is seeded once with the three
      approved finite seeds, so a character always has a goal surface;
    - any non-empty list is returned normalised *as-is*: no re-seeding, no
      status rewrite, no reactivation, no evidence-anchor reset. A character
      that already carries goals keeps exactly those.

    Pure: never mutates ``goals`` and never performs I/O. The caller persists.
    """
    try:
        existing = list(goals or [])
    except Exception:
        existing = []
    normalised: List[Dict[str, Any]] = []
    for item in existing:
        parsed = goal_from_dict(item)
        if parsed is not None:
            normalised.append(goal_to_dict(parsed))
    if normalised:
        return normalised
    return default_seed_goals(now)


def goal_status_counts(goals: Any) -> Dict[str, int]:
    """``{"seed": n, "active": n, "done": n}`` (pure, fail-soft to zeros).

    This is how the decision layer distinguishes "no goal relevant", "a seed
    not yet activated", "actively working on it" and "already finished"
    without ever inferring a transition.
    """
    counts = {"seed": 0, "active": 0, "done": 0}
    try:
        items = list(goals or [])
    except Exception:
        return counts
    for item in items:
        parsed = goal_from_dict(item)
        if parsed is None:
            continue
        counts[parsed.status] = counts.get(parsed.status, 0) + 1
    return counts


def active_goal_ids(goals: Any) -> tuple:
    """Ids of goals in ``active`` status only (pure, fail-soft)."""
    out: List[str] = []
    try:
        items = list(goals or [])
    except Exception:
        return tuple()
    for item in items:
        parsed = goal_from_dict(item)
        if parsed is not None and parsed.status == "active":
            out.append(parsed.id)
    return tuple(out)


def record_goal_evidence(
    goals: List[Dict[str, Any]],
    goal_id: str,
    evidence_id: str,
    now: Optional[datetime] = None,
) -> tuple:
    """Pin the evidence that already produced a decision for ``goal_id``.

    Pure ``(new_list, changed)``. The anchor only moves when the evidence
    identity actually differs, so replaying the same event cannot retrigger
    the same goal. A blank ``evidence_id`` is a no-op rather than a silent
    reset (which would re-open an already-consumed goal).
    """
    marker = str(evidence_id or "").strip()
    if not marker:
        return (list(goals or []), False)
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp = moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    changed = False
    out: List[Dict[str, Any]] = []
    for item in goals or []:
        if (
            isinstance(item, dict)
            and str(item.get("id", "")) == str(goal_id)
            and str(item.get("status", "seed")) == "active"
            and not changed
        ):
            if str(item.get("last_evidence_id", "") or "") != marker:
                updated = dict(item)
                updated["last_evidence_id"] = marker
                updated["last_evidence_at"] = stamp
                out.append(updated)
                changed = True
            else:
                out.append(item)
        else:
            out.append(item)
    return out, changed


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
                "kind": str(item.get("kind", "") or ""),
            }
            for item in data.get("memories", [])
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]
        goals: List[Dict[str, Any]] = []
        preferences: List[Dict[str, Any]] = []
        for item in data.get("interaction_preferences", []) or []:
            normalized = normalize_stored_preference(item)
            if normalized is not None:
                preferences.append(normalized)
        raw_goals = data.get("goals", [])
        if isinstance(raw_goals, list):
            for item in raw_goals:
                parsed = goal_from_dict(item)
                if parsed is not None:
                    goals.append(goal_to_dict(parsed))
        intentions: List[Dict[str, Any]] = []
        try:
            from .future_intentions import normalize_stored_intention

            raw_intentions = data.get("future_intentions", [])
            if isinstance(raw_intentions, list):
                for item in raw_intentions:
                    parsed_intention = normalize_stored_intention(item)
                    if parsed_intention is not None:
                        intentions.append(parsed_intention)
        except Exception:
            intentions = []
        return CharacterState(
            relationship_status=normalize_relationship_status(
                data.get("relationship_status", "stranger")
            ),
            relationship_updated_at=data.get("relationship_updated_at"),
            relationship_reason=str(data.get("relationship_reason", "default")),
            relationship_migrated=bool(data.get("relationship_migrated", False)),
            memories=memories,
            goals=goals,
            interaction_preferences=preferences,
            user_timezone=str(data.get("user_timezone") or "") or None,
            future_intentions=intentions,
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
                "goals": state.goals,
                "interaction_preferences": state.interaction_preferences,
                "user_timezone": state.user_timezone,
                "future_intentions": state.future_intentions,
            }
            _write_state_atomic(filepath, data)
        return True
    except Exception as error:
        logger.error(
            "Failed to save character state: error_type={}", type(error).__name__
        )
        return False


def migrate_relationship_if_needed(
    conf_uid: str, state: CharacterState
) -> CharacterState:
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
                    if data
                    and isinstance(data[0], dict)
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
            interaction_preferences=state.interaction_preferences,
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


def set_character_timezone(
    conf_uid: str, tz: Optional[str]
) -> Optional[CharacterState]:
    """Persist the last known user timezone; None on write failure.

    Fail-soft by design: blank/unknown values never wipe a stored one.
    """
    name = str(tz or "").strip()
    if not name:
        return None
    state = load_character_state(conf_uid)
    if state.user_timezone == name:
        return state
    state.user_timezone = name
    if not save_character_state(conf_uid, state):
        return None
    return state


def _normalize_memory_text(text: str) -> str:
    return " ".join((text or "").lower().split()).strip(" .,!?;:，。！？；：")


# Words that carry no identity in a stored fact. Dropping them lets the same
# sentence survive being written twice in different surface forms.
_MEMORY_STOPWORDS = frozenset(
    {
        "aku", "gw", "gue", "gua", "ane", "saya", "user", "yang", "dengan",
        "untuk", "dari", "pada", "itu", "ini", "ya", "sih", "deh", "dong",
        "kok", "banget", "suka", "biasa", "selalu",
    }
)


def _memory_significant_tokens(text: str) -> frozenset:
    """Lowercase content words of a stored fact, ignoring voice/filler."""
    raw = _normalize_memory_text(text)
    tokens = [
        token
        for token in "".join(
            ch if ch.isalnum() else " " for ch in raw
        ).split()
        if token and token not in _MEMORY_STOPWORDS
    ]
    return frozenset(tokens)


def _memory_is_same_fact(existing: str, incoming: str) -> bool:
    """True when two phrasings state the same durable fact.

    Automatic capture drops the first-person subject ("suka kopi susu gula
    aren") while an explicit command keeps it ("aku suka minum kopi susu gula
    aren"), so a plain string comparison stores the same fact twice. A token
    subset in EITHER direction, with at least two shared content words, means
    one phrasing is contained in the other and they must not coexist.

    Deliberately conservative: genuinely different facts ("suka kopi" vs "suka
    teh") share few tokens and stay separate.
    """
    left = _memory_significant_tokens(existing)
    right = _memory_significant_tokens(incoming)
    if not left or not right:
        return False
    shared = left & right
    if len(shared) < 2:
        return False
    return left <= right or right <= left


def add_character_memory(
    conf_uid: str,
    text: str,
    *,
    explicit: bool = True,
    kind: str = "",
) -> Optional[CharacterState]:
    """Append one long-term fact (deduplicated); None on write failure.

    Deduplication is semantic as well as literal: a fact already stored in a
    different phrasing ("aku suka minum kopi susu" vs "suka kopi susu") is not
    written a second time.
    """
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return None
    state = load_character_state(conf_uid)
    normalized = _normalize_memory_text(cleaned)
    for item in state.memories:
        stored_text = str(item.get("text", ""))
        if _normalize_memory_text(stored_text) == normalized:
            return state
        if _memory_is_same_fact(stored_text, cleaned):
            return state
    state.memories.append(
        {
            "text": cleaned,
            "added_at": _now_iso(),
            "explicit": bool(explicit),
            "kind": str(kind or ""),
        }
    )
    if not save_character_state(conf_uid, state):
        return None
    return state


def record_interaction_preference(
    conf_uid: str,
    user_text: str,
    *,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Detect and persist one durable interaction preference (pure-ish I/O).

    Returns the stored entry, or None when the turn carried no durable
    preference. Conflicting preferences of the same category supersede the
    older active entry rather than coexisting. Fail-soft: never raises.
    """
    detected = detect_interaction_preference(user_text, now=now)
    if detected is None:
        return None
    try:
        state = load_character_state(conf_uid)
        preference_id = uuid.uuid4().hex
        updated = apply_preference(
            state.interaction_preferences,
            detected,
            preference_id=preference_id,
            now=now,
        )
        state.interaction_preferences = updated
        if not save_character_state(conf_uid, state):
            return None
        logger.info(
            "Interaction preference stored: category={} polarity={} frame={}",
            detected.category,
            detected.polarity,
            detected.frame,
        )
        for item in reversed(updated):
            if item.get("id") == preference_id:
                return dict(item)
        return None
    except Exception as error:
        logger.warning(
            "Interaction preference not stored: type={}", type(error).__name__
        )
        return None


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
    """Reset relationship to stranger and clear character-level state.

    Clears memories, interaction preferences and future intentions as well:
    this action is a full character-state reset, so no character-level state
    may survive it hidden.
    """
    state = load_character_state(conf_uid)
    state.relationship_status = "stranger"
    state.relationship_updated_at = _now_iso()
    state.relationship_reason = "manual_reset"
    state.relationship_migrated = True
    state.memories = []
    state.interaction_preferences = []
    state.future_intentions = []
    if not save_character_state(conf_uid, state):
        return None
    return state


def build_character_memory_context(
    state: CharacterState,
    *,
    max_tokens: int = CHARACTER_MEMORY_MAX_TOKENS,
    tz: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    """Return a compact, bounded character-memory block for the system prompt.

    Explicit (manual) memories are prioritized, then the most recent facts.
    Memory is never dumped wholesale; the block stays well under the budget.

    Each bullet carries a render-time age tag (e.g. ``[2 days ago | Sep 29]``)
    computed from the persisted ``added_at`` in the user timezone. Stored
    text and ordering are unchanged; unparseable timestamps render untagged.
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
        age = memory_age_label(item.get("added_at", ""), now, tz)
        line = f"- {age} {text}" if age else f"- {text}"
        line_tokens = estimate_tokens(line) + 4
        if used_tokens + line_tokens > max_tokens:
            break
        lines.append(line)
        used_tokens += line_tokens
    if not lines:
        return ""
    header = "Known long-term context (character memory, shared across all chats):"
    return f"{header}\n" + "\n".join(lines)


def record_future_intention(
    conf_uid: str,
    user_text: str,
    *,
    now: Optional[datetime] = None,
    tz: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Detect and persist one explicit reminder request (fail-soft).

    Tries the narrow reminder grammar first (category A), then the natural
    future-plan grammar (category B). Returns the stored entry, or None when
    the turn carried neither. Same-turn duplicates are not stored twice.
    Never raises.
    """
    try:
        from .future_intentions import add_future_intention, detect_any_future_intention
    except Exception:
        return None
    try:
        detected = detect_any_future_intention(user_text, now=now, tz=tz)
    except Exception:
        return None
    if detected is None:
        return None
    try:
        import uuid as _uuid

        state = load_character_state(conf_uid)
        intention_id = _uuid.uuid4().hex
        updated = add_future_intention(
            state.future_intentions, detected, intention_id=intention_id
        )
        state.future_intentions = updated
        if not save_character_state(conf_uid, state):
            return None
        logger.info(
            "Future intention stored: kind={} due_at={} text_chars={}",
            getattr(detected, "kind", "reminder"),
            detected.due_at,
            len(detected.text),
        )
        for item in reversed(updated):
            if item.get("id") == intention_id:
                return dict(item)
        return None
    except Exception as error:
        logger.warning(
            "Future intention not persisted: type={}", type(error).__name__
        )
        return None


def complete_future_intention(
    conf_uid: str, intention_id: str
) -> bool:
    """Mark one reminder request done; False when missing or unwritable."""
    try:
        from .future_intentions import complete_future_intention as _complete

        state = load_character_state(conf_uid)
        updated, changed = _complete(state.future_intentions, intention_id)
        if not changed:
            return False
        state.future_intentions = updated
        return bool(save_character_state(conf_uid, state))
    except Exception:
        return False


def consume_due_future_intentions(
    conf_uid: str,
    *,
    now: Optional[datetime] = None,
) -> int:
    """Mark all currently-due reminders done; returns count consumed."""
    try:
        from .future_intentions import complete_future_intention as _complete
        from .future_intentions import due_intentions

        state = load_character_state(conf_uid)
        due = due_intentions(state.future_intentions, now=now)
        if not due:
            return 0
        rows = list(state.future_intentions)
        count = 0
        for item in due:
            updated, changed = _complete(rows, str(item.get("id", "")))
            if changed:
                rows = updated
                count += 1
        if count:
            state.future_intentions = rows
            if not save_character_state(conf_uid, state):
                return 0
        return count
    except Exception:
        return 0
