"""Behavior / interaction preferences — deterministic capture and rendering.

A persistent, user-stated constraint on HOW Mili interacts with this user
("don't be too harsh", "be more casual", "don't joke when I'm serious").

Layer separation (deliberate, one source of truth each):

- persona YAML: who Mili is. Never written by this module.
- relationship state: relational tier. Owned by ``relationship_context``.
- explicit memory: facts the user asked to remember (``character_memory_commands``).
- episodic memory: something that happened at a point in time.
- interaction preferences (this module): how the user wants to be spoken to.

Capture is deterministic and local: a narrow interaction grammar requires a
frame (durability or first-person evaluation) AND a recognised interaction
dimension AND a polarity. A bare keyword search is explicitly not enough,
and temporary/one-off requests are suppressed so they never become
durable state. No LLM call, no scheduler, no background work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

PreferenceCategory = str  # "tone" | "formality" | "humor" | "address" | "directness"

PREFERENCE_STATUS_ACTIVE = "active"
PREFERENCE_STATUS_SUPERSEDED = "superseded"

# Durable-frames: the user is asking for an ongoing change, not this turn.
_DURABLE_FRAME = re.compile(
    r"\b(?:mulai\s+(?:sekarang\b|dari\s+sekarang\b)|"
    r"mulai\s+sekarang\s+(?:dan\s+)?(?:terus\b|setiap\s+kali\b)?|"
    r"dari\s+sekarang\b|ke\s*depannya\b|selama\s+ini\b|terus\s+(?:begini|begitu)\b|"
    r"going\s+forward\b|from\s+now\s+on\b|permanently\b)\b",
    re.IGNORECASE,
)

# First-person evaluation frame: the user states a standing like/dislike
# about how Mili talks to them.
_EVALUATION_FRAME = re.compile(
    r"\b(?:aku|gw|gue|saya)\s+(?:lebih\s+)?(?:suka|biasanya|selalu|umon)\b"
    r"|\baku\s+nggak\s+suka\b|\baku\s+(?:ga|gak|nggak|ngga)\s+suka\b",
    re.IGNORECASE,
)

# Temporary suppression: valid conversational requests that must NOT become
# durable state ("don't be harsh today").
_TEMPORARY_FRAME = re.compile(
    r"\b(?:hari\s+ini\b|sekarang\s+(?:aja|saja|dulu)\b|sementara\b|"
    r"for\s+(?:now|today)\b|just\s+(?:today|now)\b|"
    r"lagi(?:\s+(?:aja|saja))?\s+(?:bikin|buat)\s+(?:situasi|suasana)\b|"
    r"nanti\s+(?:aja\s+)?(?:kita|bisa)\s+(?:bicara|ngobrol)\b)",
    re.IGNORECASE,
)

# Questions are always conversation, never a durable instruction.
_QUESTION = re.compile(r"[?？]\s*$")

# Interaction dimensions. Each entry maps a category to the phrases that
# name that dimension, plus whether the phrase carries a negative polarity.
_DIMENSIONS: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = [
    (
        "tone",
        ("galak", "kasar", "sengit", "tajam", "sindir", "nyindir", "bentar"),
        ("lembut", "gentle", "halus", "soft", "hangat", "ramah", "santai"),
    ),
    (
        "formality",
        ("formal", "resmi", "kaku", "beradab"),
        ("santai", "casual", "relaxed", "gamblang", "bebas"),
    ),
    (
        "humor",
        ("bercanda", "canda", "lelucon", "humor", "godaan"),
        ("serius", "serious"),
    ),
    (
        "address",
        ("panggil", "sapaan", "nama"),
        (),
    ),
    (
        "directness",
        ("muter", "bypass", "gurat", "roas", "sengaja"),
        ("langsung", "jujur", "open", "to the point"),
    ),
]

# A directive needs a softener so ordinary chat does not trip the detector.
_DIRECTIVE_FRAME = re.compile(
    r"(?:^|[\s,.])(?:tolong|jangan|jg)\s+|"
    r"\b(?:please|don'?t|do\s+not|stop)\b",
    re.IGNORECASE,
)

# Guard: long, descriptive messages are narration, not an interaction rule.
_MAX_PREFERENCE_CHARS = 220

_CATEGORY_LABELS = {
    "tone": "tone of voice",
    "formality": "formality",
    "humor": "humor",
    "address": "how to address the user",
    "directness": "directness",
}


@dataclass(frozen=True)
class InteractionPreference:
    """One detected, user-stated interaction preference (pure data)."""

    category: str
    polarity: str  # "avoid" | "prefer"
    text: str
    frame: str  # "durability" | "evaluation" | "directive"
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "polarity": self.polarity,
            "text": self.text,
            "frame": self.frame,
            "created_at": self.created_at,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _match_dimension(text: str) -> Optional[Tuple[str, str]]:
    """Return ``(category, polarity)`` for the first dimension phrase found."""
    lowered = text.lower()
    for category, negative_phrases, positive_phrases in _DIMENSIONS:
        for phrase in negative_phrases:
            if phrase in lowered:
                return category, "avoid"
        for phrase in positive_phrases:
            if phrase in lowered:
                return category, "prefer"
    return None


def detect_interaction_preference(
    user_text: Any, now: Optional[datetime] = None
) -> Optional[InteractionPreference]:
    """Detect ONE durable interaction preference in a single user turn.

    Returns ``None`` for ordinary conversation, questions, temporary
    requests, and anything without a frame + dimension + polarity. Pure and
    deterministic: same input always yields the same result.
    """
    text = _normalize(str(user_text or ""))
    if not text or len(text) > _MAX_PREFERENCE_CHARS:
        return None
    if _QUESTION.search(text):
        return None
    if _TEMPORARY_FRAME.search(text):
        # A temporary request is a legitimate turn-level instruction, but it
        # must never be promoted to durable state.
        return None
    dimension = _match_dimension(text)
    if dimension is None:
        return None

    if _DURABLE_FRAME.search(text):
        frame = "durability"
    elif _EVALUATION_FRAME.search(text):
        frame = "evaluation"
    elif _DIRECTIVE_FRAME.search(text):
        frame = "directive"
    else:
        # No frame: this is ordinary conversation about the topic.
        return None

    category, polarity = dimension
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return InteractionPreference(
        category=category,
        polarity=polarity,
        text=text,
        frame=frame,
        created_at=moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
    )


def normalize_stored_preference(item: Any) -> Optional[Dict[str, Any]]:
    """Tolerant parse of one stored preference; None when unusable."""
    if not isinstance(item, dict):
        return None
    category = _normalize(str(item.get("category", ""))).lower()
    polarity = _normalize(str(item.get("polarity", ""))).lower()
    text = _normalize(str(item.get("text", "")))
    if category not in _CATEGORY_LABELS or polarity not in ("avoid", "prefer"):
        return None
    if not text:
        return None
    status = _normalize(str(item.get("status", ""))) or PREFERENCE_STATUS_ACTIVE
    if status not in (PREFERENCE_STATUS_ACTIVE, PREFERENCE_STATUS_SUPERSEDED):
        status = PREFERENCE_STATUS_ACTIVE
    return {
        "id": _normalize(str(item.get("id", ""))),
        "category": category,
        "polarity": polarity,
        "text": text,
        "frame": _normalize(str(item.get("frame", ""))),
        "created_at": _normalize(str(item.get("created_at", ""))),
        "updated_at": _normalize(str(item.get("updated_at", "")))
        or _normalize(str(item.get("created_at", ""))),
        "source": _normalize(str(item.get("source", ""))) or "conversation",
        "status": status,
        "superseded_by": _normalize(str(item.get("superseded_by", ""))),
    }


def active_preferences(
    stored: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """At most one active preference per category, newest first."""
    parsed = [p for p in (normalize_stored_preference(i) for i in stored or []) if p]
    active = [p for p in parsed if p["status"] == PREFERENCE_STATUS_ACTIVE]
    active.sort(key=lambda p: (p["updated_at"] or p["created_at"]), reverse=True)
    return active


def apply_preference(
    existing: Optional[List[Dict[str, Any]]],
    detected: InteractionPreference,
    *,
    preference_id: str,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Insert one preference and supersede any conflicting active entry.

    One active preference per category: a newer statement of the same
    interaction dimension replaces the older one instead of contradicting
    it. Pure: returns a new list, never mutates the input.
    """
    moment = now if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp = moment.astimezone(timezone.utc).isoformat(timespec="seconds")

    parsed = [p for p in (normalize_stored_preference(i) for i in existing or []) if p]
    kept: List[Dict[str, Any]] = []
    for item in parsed:
        if item["category"] == detected.category:
            superseded = dict(item)
            superseded["status"] = PREFERENCE_STATUS_SUPERSEDED
            superseded["superseded_by"] = preference_id
            superseded["updated_at"] = stamp
            kept.append(superseded)
        else:
            kept.append(item)
    kept.append(
        {
            "id": preference_id,
            "category": detected.category,
            "polarity": detected.polarity,
            "text": detected.text,
            "frame": detected.frame,
            "created_at": detected.created_at or stamp,
            "updated_at": stamp,
            "source": "conversation",
            "status": PREFERENCE_STATUS_ACTIVE,
            "superseded_by": "",
        }
    )
    return kept


def build_interaction_preference_context(
    stored: Optional[List[Dict[str, Any]]],
    *,
    tz: Optional[str] = None,
    max_tokens: int = 240,
) -> str:
    """Render active preferences as a bounded prompt block (pure, no I/O).

    Empty string when nothing is active, so the persona prompt stays
    untouched: preferences constrain interaction, they never replace it.
    """
    active = active_preferences(stored)
    if not active:
        return ""
    try:
        from .agent.context_window import estimate_tokens

        lines: List[str] = []
        used = 0
        for item in active:
            label = _CATEGORY_LABELS.get(item["category"], item["category"])
            if item["polarity"] == "avoid":
                line = f"- {label}: the user asked Mili not to do this. (user said: {item['text']})"
            else:
                line = f"- {label}: the user asked Mili to do more of this. (user said: {item['text']})"
            cost = estimate_tokens(line) + 4
            if used + cost > max_tokens:
                break
            lines.append(line)
            used += cost
        if not lines:
            return ""
        header = (
            "Standing interaction preferences from the user (persistent; apply "
            "every turn, but never replace the persona):"
        )
        return f"{header}\n" + "\n".join(lines)
    except Exception as error:
        logger.warning(
            "Interaction preference context skipped: type={}",
            type(error).__name__,
        )
        return ""
