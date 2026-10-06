"""Mili Self Model v1 — static identity + read-only live references.

Architecture: STATIC SELF MODEL + READ-ONLY LIVE REFERENCES. This module
owns ONLY facts that currently have no authoritative owner (identity
nature, reality boundary, capabilities, limitations, seed tendencies).
Everything dynamic stays owned by its system and is read here at render
time: WorldState (activity/location/energy/mood), RelationshipState,
character memory, temporal anchor.

Explicitly NOT a store: there is no self_model.json and no new
persistence. No LLM calls, no scheduler, no background work. One pure
composer consumed by the existing Context Builder. The block is kept
compact (~130 tokens) so it fits production prompts and tight test
budgets alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .world_state import user_local_datetime

# Token budget for the rendered SELF block (measured with the repo's
# byte//3 estimator; typical full renders stay well under it).
SELF_CONTEXT_MAX_TOKENS = 220

# --- Static identity (no other owner exists for these facts) ---------------

SELF_IDENTITY_NAME = "Mili"

# Stable reality boundary. Simulated life is persistent internal character
# state — NOT roleplay. Roleplay happens only when the user explicitly
# starts it.
SELF_REALITY_BOUNDS: List[str] = [
    "no physical human body",
    "no physical real-world presence",
    "no physical house or address",
    "no physical meetings",
]

SELF_CAPABILITIES: List[str] = [
    "chat/voice",
    "memory",
]

SELF_LIMITATIONS: List[str] = [
    "real action",
]

# Seed tendencies: initial directions only, not missions. They do not
# generate goals, do not drive the Decision Layer, and evolve only via
# future work on top of memories + activity history.
SELF_SEED_TENDENCIES: List[str] = [
    "hubungan bermakna",
    "rutinitas",
    "minat hal baru",
]

SELF_USAGE_RULE = (
    "Identity/home: truthful first, then in character. "
    'No jokes to deflect; no "As an AI..." openers.'
)


# ---------------------------------------------------------------------------
# Preference derivation (Phase 2A): READ-ONLY aggregate over persisted
# evidence. No store, no LLM, no generation.
#
# Evidence (all persisted, UTC, reload-safe):
# - world recent_activity_history entries: {from, to, at, location, by?}
# - long-term memories: {text, added_at, explicit}
# A preference is ESTABLISHED only with >=3 evidence items spread over
# >=2 distinct calendar days (user tz when given): a single mention or a
# single-session burst never qualifies. Day labels are always derived at
# render/query time; nothing relative is ever stored.
# ---------------------------------------------------------------------------

# Deterministic activity keyword stems (Indonesian + English, substring
# match on lowered text: "main" also matches "bermain"/"mainan").
PREFERENCE_ACTIVITY_KEYWORDS = {
    "reading": ("baca", "read", "buku", "book", "novel", "komik", "cerita"),
    "playing": ("main", "play", "game", "gim"),
    "eating": ("makan", "eat", "bakso", "snack", "jajan", "minum", "kopi"),
    "sleeping": ("tidur", "sleep", "bobok"),
    "resting": ("istirahat", "rest", "rebahan"),
}

# idle is not a preference; everything else valid may qualify.
PREFERENCE_ELIGIBLE_ACTIVITIES = tuple(a for a in PREFERENCE_ACTIVITY_KEYWORDS)

PREFERENCE_MIN_EVIDENCE = 3
PREFERENCE_MIN_DAYS = 2
# Cap rendered preference items so the SELF block stays compact.
PREFERENCE_RENDER_LIMIT = 2


@dataclass(frozen=True)
class PreferenceCandidate:
    """One derived activity preference (data only, never stored)."""

    activity: str
    evidence_count: int
    distinct_days: int
    first_at: str
    last_at: str
    established: bool


def _memory_activity(text: Any) -> Optional[str]:
    lowered = str(text or "").lower()
    if not lowered.strip():
        return None
    for activity in PREFERENCE_ELIGIBLE_ACTIVITIES:
        for stem in PREFERENCE_ACTIVITY_KEYWORDS[activity]:
            if stem in lowered:
                return activity
    return None


def _local_day(iso_value: Any, tz: Optional[str]) -> Optional[Tuple[Any, str]]:
    """(user-local date, original string) or None when unparseable."""
    try:
        parsed = datetime.fromisoformat(str(iso_value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    local = user_local_datetime(parsed, tz)
    return local.date(), str(iso_value)


def derive_activity_preferences(
    world_history: Optional[Sequence[Dict[str, Any]]] = None,
    memories: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    tz: Optional[str] = None,
) -> List[PreferenceCandidate]:
    """Aggregate persisted evidence into preference candidates (pure).

    Reads world activity history + long-term memories only; writes
    nothing, calls no LLM. Deterministic: sorted by (-evidence, activity).
    Unparseable timestamps are skipped (evidence must stay traceable).
    """
    dated: Dict[str, List[Tuple[Any, str]]] = {}
    for entry in world_history or []:
        if not isinstance(entry, dict):
            continue
        activity = str(entry.get("to", "")).strip().lower()
        if activity not in PREFERENCE_ELIGIBLE_ACTIVITIES:
            continue
        day = _local_day(entry.get("at"), tz)
        if day is None:
            continue
        dated.setdefault(activity, []).append(day)
    for memo in memories or []:
        if not isinstance(memo, dict):
            continue
        activity = _memory_activity(memo.get("text"))
        if activity is None:
            continue
        day = _local_day(memo.get("added_at"), tz)
        if day is None:
            continue
        dated.setdefault(activity, []).append(day)
    candidates = []
    for activity, hits in dated.items():
        days = {day for day, _ in hits}
        ordered = sorted(hits, key=lambda h: (str(h[0]), h[1]))
        count = len(hits)
        candidates.append(
            PreferenceCandidate(
                activity=activity,
                evidence_count=count,
                distinct_days=len(days),
                first_at=ordered[0][1],
                last_at=ordered[-1][1],
                established=(
                    count >= PREFERENCE_MIN_EVIDENCE
                    and len(days) >= PREFERENCE_MIN_DAYS
                ),
            )
        )
    candidates.sort(key=lambda c: (-c.evidence_count, c.activity))
    return candidates


def format_preference_line(
    candidates: Sequence[PreferenceCandidate],
    *,
    limit: int = PREFERENCE_RENDER_LIMIT,
) -> str:
    """Compact render of established preferences only (pure)."""
    shown = [c for c in candidates if c.established][: max(0, limit)]
    if not shown:
        return ""
    items = ", ".join(f"{c.activity} (evidence: {c.evidence_count})" for c in shown)
    noun = "preference" if len(shown) == 1 else "preferences"
    return f"Emerging {noun}: {items}."


def build_self_context(
    *,
    character_name: str = SELF_IDENTITY_NAME,
    avatar_present: bool = False,
    live2d_model_name: Optional[str] = None,
    activity: Optional[str] = None,
    location: Optional[str] = None,
    relationship_status: Optional[str] = None,
    memory_count: int = 0,
    preferences: Sequence[PreferenceCandidate] = (),
    max_tokens: int = SELF_CONTEXT_MAX_TOKENS,
) -> str:
    """Assemble the compact SELF block for the system prompt (pure).

    All live arguments are read-only references rendered at call time;
    nothing is stored here. ``preferences`` carries the derived activity
    candidates (established ones render, the rest is ignored); the agent
    passes them from persisted world history + memories every turn.
    Output stays under ``max_tokens``; overlong seed lists are truncated,
    never the identity/boundary lines.
    """
    name = (character_name or SELF_IDENTITY_NAME).strip() or SELF_IDENTITY_NAME
    if live2d_model_name:
        avatar_bit = f", with avatar '{live2d_model_name}'"
    elif avatar_present:
        avatar_bit = ", with avatar shown"
    else:
        avatar_bit = ""
    lines = [
        "SELF:",
        f"- {name}: I am an AI, not human, in this app{avatar_bit}.",
        "- No body/house; never invent addresses/past. Simulated room, "
        "not physical; sim-life, not roleplay.",
    ]
    if activity:
        where = f" in {location}" if location else ""
        lines.append(f"- Now: {activity}{where}.")
    rel_bits = []
    if relationship_status:
        rel_bits.append(f"Relationship: {relationship_status}")
    if memory_count > 0:
        rel_bits.append(f"Facts: {memory_count}")
    if rel_bits:
        lines.append("- " + ". ".join(rel_bits) + ".")
    lines.append(
        "- Can: "
        + "; ".join(SELF_CAPABILITIES)
        + ". Cannot: "
        + "; ".join(SELF_LIMITATIONS)
        + "."
    )
    lines.append("- Tendencies: " + "; ".join(SELF_SEED_TENDENCIES) + ".")
    preference_line = format_preference_line(preferences)
    if preference_line:
        lines.append(f"- {preference_line}")
    lines.append(f"- {SELF_USAGE_RULE}")
    kept: List[str] = []
    used = 0
    for line in lines:
        cost = len(line.encode("utf-8")) // 3
        if used + cost > max_tokens:
            break
        kept.append(line)
        used += cost
    # Identity/boundary lines come first, so truncation only ever drops
    # later detail lines; guarantee a non-empty block regardless.
    return "\n".join(kept if kept else lines[:4])
