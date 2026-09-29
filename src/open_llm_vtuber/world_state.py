"""Stage 7 — Simulated Life: character-scoped World/Life State.

This module is the SINGLE WRITER for the World/Life State fields::

    location, activity, energy, mood, time_context,
    activity_started_at, last_update_at, recent_activity_history

All other subsystems (conversation, proactive, emotion/avatar, TTS, voice)
get READ-ONLY snapshots via :func:`load_and_reconcile_world_state` or
:func:`build_world_state_context`. Nothing else may mutate these fields.

Design rules (see Phase 2 scope):

* Pure deterministic core: :func:`reconcile` / :func:`transition` contain
  NO file I/O, NO websocket logic, NO LLM/provider calls, NO scheduler,
  NO frontend logic. Persistence and orchestration live outside the core.
* Lazy reconciliation only: state is reconciled when existing lifecycle
  events already occur (conversation trigger, proactive check, reconnect /
  history switch, character load / server restart). There is deliberately
  NO background loop, NO setInterval, NO second scheduler.
* Fail-soft: corrupt files yield a safe default; save failures return
  ``False`` and never raise into the conversation path.
* Mood isolation: ``mood`` is slow persistent life state. It MUST NOT write
  ``Actions.expressions`` / ``Actions.emotions`` and never bypasses
  ``agent/transformers.py``. Semantic Emotion stays authoritative for
  per-response avatar expression.
* Reactive layer: backend semantic-emotion labels observed on a completed
  turn are INPUT to ``apply_reactive`` (pure): mapped labels arm a
  temporary mood with a turn ttl (shy/flustered ladder included);
  neutral turns decay it stepwise back to baseline; charged turns may
  interrupt reading/playing and nudge energy -1. No LLM, no scheduler;
  ttl also expires on a wall-clock backstop during lazy reconciliation.
* Clock injection: every public function accepts ``now`` so tests can use a
  fake clock (+5m / +2h / +9h / +2d) without waiting. Default is real
  server-side wall-clock time (UTC, matching ``character_state``).
* User timezone: pure functions accept an optional ``tz`` IANA name. Hour
  derivation (time_context, night rules, playing location) then uses the
  user-local hour; persisted timestamps always stay canonical UTC.
  ``tz=None`` (or unknown) keeps the original server-UTC behavior.

Initial transition model (intentionally small, documented here):

* ``time_context`` derives from wall-clock hour (server-side convention):
  night 22-05, morning 05-11, afternoon 11-17, evening 17-22.
* Energy rates per hour (deterministic, bounded 0-100):
  sleeping +15, resting +8, eating +5, idle -1, reading -2, playing -6.
* Activity durations (time since ``activity_started_at``):
  sleeping >= 8h -> idle; eating >= 45m -> idle; playing >= 2h -> idle;
  reading >= 3h -> idle; resting >= 2h -> idle.
  Early exits: sleeping with energy >= 100 and age >= 90m -> idle;
  resting with energy >= 95 and age >= 30m -> idle.
  Low-energy: idle at night (22-05) with energy <= 30 -> sleeping;
  idle with energy <= 10 -> resting; playing with energy <= 5 -> resting.
* Location consistency (enforced on every reconcile, no history entry
  unless the activity itself changed):
  sleeping/resting/reading -> room; eating -> kitchen;
  playing -> outside by day (06-18) else room; idle keeps its location.
* Mood derivation (slow state, recomputed on every reconcile):
  sleeping -> sleepy; energy >= 70 -> content; 30-70 -> calm;
  10-30 -> tired; below 10 -> exhausted.
* ``recent_activity_history`` records WORLD ACTIVITY TRANSITIONS only
  (never chat messages), capped at 10 entries.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger

from .chat_history_manager import _sanitize_path_component

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLD_STATE_VERSION = 1
WORLD_STATE_DIR = "world_state"

VALID_LOCATIONS = ("room", "kitchen", "outside")
VALID_ACTIVITIES = ("idle", "resting", "reading", "eating", "playing", "sleeping")
# Baseline moods come from derive_mood; reactive moods are set only by
# apply_reactive from backend semantic-emotion labels (never by the clock).
VALID_MOODS = (
    "calm",
    "content",
    "tired",
    "sleepy",
    "exhausted",
    "happy",
    "sad",
    "shy",
    "flustered",
    "irritated",
    "angry",
)

DEFAULT_LOCATION = "room"
DEFAULT_ACTIVITY = "idle"
DEFAULT_ENERGY = 80
DEFAULT_MOOD = "calm"

MIN_ENERGY = 0
MAX_ENERGY = 100

HISTORY_CAP = 10

# Energy delta per elapsed hour, by activity. Positive recovers, negative
# spends. Deliberately simple linear rates; no physiology simulation.
ENERGY_RATE_PER_HOUR: Dict[str, float] = {
    "sleeping": 15.0,
    "resting": 8.0,
    "eating": 5.0,
    "idle": -1.0,
    "reading": -2.0,
    "playing": -6.0,
}

# Activity age thresholds (seconds) after which an activity ends -> idle.
ACTIVITY_DURATION_LIMIT_S: Dict[str, float] = {
    "sleeping": 8 * 3600,
    "eating": 45 * 60,
    "playing": 2 * 3600,
    "reading": 3 * 3600,
    "resting": 2 * 3600,
}

NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 5
DAY_OUTSIDE_START_HOUR = 6
DAY_OUTSIDE_END_HOUR = 18

# ---------------------------------------------------------------------------
# Reactive layer (interaction -> life state). Deterministic, event-driven,
# no LLM. Backend semantic-emotion labels (Live2D emo_map keys, lowercased)
# act as INPUT; World mood stays a separate persistent field.
# ---------------------------------------------------------------------------

# Backend emotion label -> (reactive mood, ttl in visible turns).
EMOTION_MOOD_MAP: Dict[str, tuple] = {
    "anger_strong": ("angry", 3),
    "anger": ("irritated", 3),
    "sadness": ("sad", 3),
    "embarrassed": ("shy", 3),
    "joy": ("happy", 2),
}

# Priority when one turn carries several labels (first hit wins).
EMOTION_PRIORITY = ("anger_strong", "anger", "sadness", "embarrassed", "joy")

# Labels that mean "no emotional charge" for decay purposes.
NEUTRAL_EMOTION_LABELS = frozenset({"neutral", ""})

# One rung down per exhausted ttl: (next mood or None=baseline, next ttl).
# flustered -> shy -> baseline; angry -> irritated -> baseline.
DECAY_NEXT: Dict[str, tuple] = {
    "flustered": ("shy", 2),
    "angry": ("irritated", 2),
}

# Wall-clock backstop: a reactive mood older than this with no new turns
# falls back to baseline on the next lazy reconcile (no scheduler).
REACTIVE_MOOD_MAX_AGE_S = 30 * 60

# Charged interaction interrupts only light activities; sleep/rest/meals
# are never broken by chatting (continuity first).
REACTIVE_INTERRUPT_ACTIVITIES = frozenset({"reading", "playing"})

# Per-turn energy nudge applies only outside rest states (their time
# rules own recovery); small by design, never a 65 -> 50 drop.
REACTIVE_ENERGY_ACTIVITIES = frozenset({"idle", "reading", "playing"})
REACTIVE_ENERGY_DELTA = -1

_state_locks: Dict[str, threading.RLock] = {}
_state_locks_guard = threading.Lock()

_TZ_CACHE: Dict[str, Optional[ZoneInfo]] = {}
_TZ_WARNED: set = set()


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


# ---------------------------------------------------------------------------
# Clock + parsing helpers (pure)
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    """Default real wall-clock (server-side UTC convention)."""
    return datetime.now(timezone.utc)


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _parse_iso(value: Any, fallback: datetime) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return fallback
    return _ensure_aware(parsed)


def _to_iso(value: datetime) -> str:
    return _ensure_aware(value).isoformat(timespec="seconds")


def resolve_tz(tz: Optional[str]) -> Optional[ZoneInfo]:
    """Return a ZoneInfo for a user/session timezone name, or None.

    ``None`` (or blank/invalid) means "no user timezone known" and callers
    fall back to server-side UTC, preserving pre-timezone behavior.
    Persistence always stays canonical UTC; only hour derivation converts.
    Results (including misses) are cached so an unknown name warns once.
    """
    name = str(tz or "").strip()
    if not name:
        return None
    if name in _TZ_CACHE:
        return _TZ_CACHE[name]
    try:
        zone: Optional[ZoneInfo] = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        zone = None
        if name not in _TZ_WARNED:
            _TZ_WARNED.add(name)
            logger.warning(
                "Unknown user timezone, falling back to UTC: tz={}", name[:64]
            )
    _TZ_CACHE[name] = zone
    return zone


def local_hour(moment: datetime, tz: Optional[str]) -> int:
    """Wall-clock hour of ``moment`` in the user timezone (pure)."""
    aware = _ensure_aware(moment)
    zone = resolve_tz(tz)
    if zone is None:
        return aware.hour
    try:
        return aware.astimezone(zone).hour
    except Exception as error:
        logger.warning(
            "Timezone conversion failed, using UTC hour: type={}",
            type(error).__name__,
        )
        return aware.hour


# ---------------------------------------------------------------------------
# State shape
# ---------------------------------------------------------------------------


@dataclass
class WorldState:
    """Character-scoped simulated-life state (single-writer: this module)."""

    location: str = DEFAULT_LOCATION
    activity: str = DEFAULT_ACTIVITY
    energy: int = DEFAULT_ENERGY
    mood: str = DEFAULT_MOOD
    time_context: str = "evening"
    activity_started_at: Optional[str] = None
    last_update_at: Optional[str] = None
    recent_activity_history: List[Dict[str, Any]] = field(default_factory=list)
    # Reactive layer: remaining visible turns for a reactive mood (0 means
    # the mood is baseline-derived), plus when it was set (wall backstop).
    mood_ttl_turns: int = 0
    mood_set_at: Optional[str] = None
    # Autonomous Decision Layer v1: wall-clock stamp of the last autonomous
    # activity pick. Old files lack it (None = no cooldown). Never a ticker.
    last_autonomous_decision_at: Optional[str] = None


def normalize_location(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in VALID_LOCATIONS else DEFAULT_LOCATION


def normalize_activity(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in VALID_ACTIVITIES else DEFAULT_ACTIVITY


def normalize_mood(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in VALID_MOODS else DEFAULT_MOOD


def clamp_energy(value: Any) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return DEFAULT_ENERGY
    return max(MIN_ENERGY, min(MAX_ENERGY, number))


# ---------------------------------------------------------------------------
# Pure deterministic core (no I/O)
# ---------------------------------------------------------------------------


def derive_time_context(moment: datetime, tz: Optional[str] = None) -> str:
    """Map wall-clock hour to a coarse time context (pure).

    The hour is taken in the user/session timezone when ``tz`` (IANA name)
    is provided, otherwise server-side UTC. Buckets are unchanged.
    """
    hour = local_hour(moment, tz)
    if hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR:
        return "night"
    if hour < 11:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


def derive_mood(activity: str, energy: int) -> str:
    """Derive slow life-state mood from activity + energy (pure)."""
    if activity == "sleeping":
        return "sleepy"
    if energy >= 70:
        return "content"
    if energy >= 30:
        return "calm"
    if energy >= 10:
        return "tired"
    return "exhausted"


# Temporal anchoring for the LLM context (pure display helpers).
# These do NOT change transition/reconciliation logic; they only render
# the existing UTC clock and stored timestamps in the user/session
# timezone so relative words ("today", "yesterday") stay truthful.
# ---------------------------------------------------------------------------


def user_local_datetime(
    moment: Optional[datetime] = None, tz: Optional[str] = None
) -> datetime:
    """Return ``moment`` (default real clock) in the user timezone (pure).

    Falls back to UTC when ``tz`` is missing or invalid. Naive inputs are
    read as UTC, matching the persistence convention.
    """
    aware = _ensure_aware(moment if moment is not None else utcnow())
    zone = resolve_tz(tz)
    return aware.astimezone(zone) if zone is not None else aware


def format_temporal_anchor(
    moment: Optional[datetime] = None, tz: Optional[str] = None
) -> str:
    """Compact "today + now" anchor for the system prompt (pure).

    Gives the LLM a reliable current-date/weekday/clock-time/timezone
    reference, rebuilt at every request from the runtime system clock, so
    "what time is it" is answered from real time, never guessed or stale.
    """
    local = user_local_datetime(moment, tz)
    date_str = f"{local:%B} {local.day}, {local.year}"
    weekday = f"{local:%A}"
    clock_str = f"{local:%H:%M}"
    zone = resolve_tz(tz)
    tz_label = tz if zone is not None else "UTC"
    return (
        f"Current date: {date_str} ({weekday})\n"
        f"Current time: {clock_str}\n"
        f"Timezone: {tz_label}"
    )


def relative_day_parts(
    added_at: Any,
    moment: Optional[datetime] = None,
    tz: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """(label, date) age of a stored ISO timestamp in the user timezone.

    Pure. Day boundaries use user-local calendar dates. Returns None when
    ``added_at`` is unparseable so callers render the original text as-is.
    Future timestamps (clock skew) read as today, never negative.
    """
    try:
        added = _ensure_aware(datetime.fromisoformat(str(added_at)))
    except (TypeError, ValueError):
        return None
    local_now = user_local_datetime(moment, tz)
    zone = resolve_tz(tz)
    local_added = added.astimezone(zone) if zone is not None else added
    delta_days = (local_now.date() - local_added.date()).days
    if delta_days < 0:
        delta_days = 0
    if delta_days == 0:
        label = "Today"
    elif delta_days == 1:
        label = "Yesterday"
    else:
        label = f"{delta_days} days ago"
    date_str = f"{local_added:%b} {local_added.day}"
    return label, date_str


def memory_age_label(
    added_at: Any,
    moment: Optional[datetime] = None,
    tz: Optional[str] = None,
) -> str:
    """Compact render-time age tag like ``[2 days ago | Sep 29]`` (pure).

    Calculated at context-build time from the persisted timestamp; the
    stored memory text is never rewritten. Empty string when unparseable.
    """
    parts = relative_day_parts(added_at, moment, tz)
    if parts is None:
        return ""
    label, date_str = parts
    return f"[{label} | {date_str}]"


def location_for(
    activity: str,
    moment: datetime,
    current_location: str,
    tz: Optional[str] = None,
) -> str:
    """Enforce activity/location consistency (pure).

    ``idle`` keeps its location; every other activity has a home so the
    world cannot drift into impossible states (e.g. sleeping outside).
    Day/night for ``playing`` uses the user-local hour when ``tz`` is set.
    """
    if activity in ("sleeping", "resting", "reading"):
        return "room"
    if activity == "eating":
        return "kitchen"
    if activity == "playing":
        hour = local_hour(moment, tz)
        if DAY_OUTSIDE_START_HOUR <= hour < DAY_OUTSIDE_END_HOUR:
            return "outside"
        return "room"
    return normalize_location(current_location)


def default_state(
    now: Optional[datetime] = None, tz: Optional[str] = None
) -> WorldState:
    """Build the initial world state for a character (pure)."""
    moment = _ensure_aware(now) if now is not None else utcnow()
    stamp = _to_iso(moment)
    return WorldState(
        location=DEFAULT_LOCATION,
        activity=DEFAULT_ACTIVITY,
        energy=DEFAULT_ENERGY,
        mood=derive_mood(DEFAULT_ACTIVITY, DEFAULT_ENERGY),
        time_context=derive_time_context(moment, tz),
        activity_started_at=stamp,
        last_update_at=stamp,
        recent_activity_history=[],
    )


def _is_night(moment: datetime, tz: Optional[str] = None) -> bool:
    hour = local_hour(moment, tz)
    return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR


# ---------------------------------------------------------------------------
# Autonomous Decision Layer v1 (deterministic policy; no LLM, no scheduler).
#
# Thin policy between the reconciled WorldState and the existing
# transition/persistence mechanics: decide_activity() chooses WHAT should
# happen; transition() stays authoritative for validity, location,
# energy, history and timestamps. Evaluated lazily inside reconcile();
# read-only callers pass decide=False.
# ---------------------------------------------------------------------------

# Idle this long (user-local activity age) before an autonomous pick.
# Above the 36-minute pinned no-change window in the existing suite.
IDLE_STALE_AFTER_S = 60 * 60
# Minimum gap between autonomous picks (persisted wall-clock).
DECISION_COOLDOWN_S = 15 * 60
# Minimum energy for an active pick (reading/playing/eating).
DECISION_ACTIVE_MIN_ENERGY = 40
# Below this at night, idle picks sleeping instead of holding.
DECISION_NIGHT_REST_ENERGY = 50
# Morning eating window (user-local hour); eating is a scheduled activity
# here, never biological hunger (no hunger state exists).
DECISION_MORNING_EAT_START_HOUR = 6
DECISION_MORNING_EAT_END_HOUR = 9

DecisionReason = Literal[
    "low_energy",
    "stale_idle",
    "duration_limit",
    "night_rest",
    "cooldown_active",
    "no_change",
]


@dataclass(frozen=True)
class DecisionResult:
    """Minimum explainable autonomous decision (pure data, no I/O)."""

    action: Optional[str]  # target activity, or None = hold current state
    reason: DecisionReason
    decided_at: str  # ISO UTC wall-clock of this evaluation
    state_version: Optional[str]  # baseline last_update_at being decided on


def _decision_hold(
    state: WorldState, moment: datetime, reason: DecisionReason
) -> DecisionResult:
    return DecisionResult(
        action=None,
        reason=reason,
        decided_at=_to_iso(moment),
        state_version=state.last_update_at,
    )


def _decision_cooldown_active(state: WorldState, moment: datetime) -> bool:
    raw = getattr(state, "last_autonomous_decision_at", None)
    if not raw:
        return False
    try:
        stamp = _ensure_aware(datetime.fromisoformat(str(raw)))
    except (TypeError, ValueError):
        return False
    delta_s = (moment - stamp).total_seconds()
    if delta_s < 0:
        return False
    return delta_s < DECISION_COOLDOWN_S


def _activity_age_s(state: WorldState, moment: datetime) -> float:
    return max(
        0.0,
        (moment - _parse_iso(state.activity_started_at, moment)).total_seconds(),
    )


def _pick_stale_idle_activity(
    state: WorldState, moment: datetime, tz: Optional[str]
) -> Optional[Tuple[str, DecisionReason]]:
    """Deterministic pick for long-idle; None = hold. No personality logic."""
    energy = clamp_energy(state.energy)
    if energy < DECISION_ACTIVE_MIN_ENERGY:
        return "resting", "low_energy"
    context = derive_time_context(moment, tz)
    recent_to = [
        str(item.get("to", ""))
        for item in (state.recent_activity_history or [])[-3:]
        if isinstance(item, dict)
    ]
    if context == "night":
        if energy < DECISION_NIGHT_REST_ENERGY:
            return "sleeping", "night_rest"
        return None
    if context == "morning":
        hour = local_hour(moment, tz)
        if (
            DECISION_MORNING_EAT_START_HOUR <= hour < DECISION_MORNING_EAT_END_HOUR
            and "eating" not in recent_to
        ):
            return "eating", "stale_idle"
    candidates = ["reading", "playing"]
    if recent_to and recent_to[-1] in candidates:
        candidates.remove(recent_to[-1])
    return candidates[0], "stale_idle"


def decide_activity(
    state: WorldState, moment: datetime, tz: Optional[str] = None
) -> DecisionResult:
    """Pure deterministic life-activity policy (no I/O, no LLM).

    Priority: sleeping holds; duration-limit mirror; stale-idle pick
    (cooldown-gated, energy-banded); otherwise hold. Extreme low-energy
    cases are already resolved by transition() before this runs (idle<=10,
    playing<=5, night<=30); the decision layer only acts on stale idle,
    so fresh states and pinned transition behavior are never overridden.
    Returned actions are always valid activities; the existing transition
    mechanics stay authoritative at apply time.
    """
    aware = _ensure_aware(moment)
    activity = normalize_activity(state.activity)

    if activity == "sleeping":
        return _decision_hold(state, aware, "no_change")

    limit = ACTIVITY_DURATION_LIMIT_S.get(activity)
    if (
        limit is not None
        and activity != "idle"
        and _activity_age_s(state, aware) >= limit
    ):
        return DecisionResult(
            "idle", "duration_limit", _to_iso(aware), state.last_update_at
        )

    if activity == "idle" and _activity_age_s(state, aware) >= IDLE_STALE_AFTER_S:
        if _decision_cooldown_active(state, aware):
            return _decision_hold(state, aware, "cooldown_active")
        pick = _pick_stale_idle_activity(state, aware, tz)
        if pick is None:
            return _decision_hold(state, aware, "no_change")
        target, reason = pick
        return DecisionResult(target, reason, _to_iso(aware), state.last_update_at)

    return _decision_hold(state, aware, "no_change")


def _apply_decision(
    state: WorldState, decision: DecisionResult, moment: datetime, tz: Optional[str]
) -> WorldState:
    """Execute a decision through existing transition mechanics (pure).

    Validity, location, mood-ttl respect, history bounding and timestamps
    follow the same rules as transition(); the record carries by="decision".
    Unknown targets are ignored (transition wins).
    """
    if decision.action is None:
        return state
    target = normalize_activity(decision.action)
    if target not in VALID_ACTIVITIES or target == state.activity:
        return state
    aware = _ensure_aware(moment)
    new_location = location_for(target, aware, state.location, tz)
    record: Dict[str, Any] = {
        "from": state.activity,
        "to": target,
        "at": _to_iso(aware),
        "location": new_location,
        "by": "decision",
    }
    history = [*state.recent_activity_history, record][-HISTORY_CAP:]
    if state.mood_ttl_turns > 0:
        new_mood, new_ttl, new_mood_set_at = (
            state.mood,
            state.mood_ttl_turns,
            state.mood_set_at,
        )
    else:
        new_mood, new_ttl, new_mood_set_at = derive_mood(target, state.energy), 0, None
    return WorldState(
        location=new_location,
        activity=target,
        energy=state.energy,
        mood=new_mood,
        time_context=derive_time_context(aware, tz),
        activity_started_at=_to_iso(aware),
        last_update_at=state.last_update_at,
        recent_activity_history=history,
        mood_ttl_turns=new_ttl,
        mood_set_at=new_mood_set_at,
        last_autonomous_decision_at=_to_iso(aware),
    )


def transition(
    state: WorldState,
    elapsed_s: float,
    now: datetime,
    tz: Optional[str] = None,
) -> Tuple[WorldState, bool]:
    """Apply one deterministic transition step (pure, no I/O).

    Returns ``(new_state, activity_changed)``. Energy, mood, location and
    timestamps are updated in place on a copy; history is appended only
    when the activity itself changes. Day/night-dependent rules use the
    user-local hour when ``tz`` (IANA name) is provided, else server UTC.
    """
    moment = _ensure_aware(now)
    elapsed = max(0.0, float(elapsed_s or 0.0))

    current = WorldState(
        location=normalize_location(state.location),
        activity=normalize_activity(state.activity),
        energy=clamp_energy(state.energy),
        mood=normalize_mood(state.mood),
        time_context=state.time_context,
        activity_started_at=state.activity_started_at,
        last_update_at=state.last_update_at,
        recent_activity_history=[
            dict(item) for item in (state.recent_activity_history or [])
        ],
        mood_ttl_turns=max(0, int(getattr(state, "mood_ttl_turns", 0) or 0)),
        mood_set_at=getattr(state, "mood_set_at", None),
        last_autonomous_decision_at=getattr(state, "last_autonomous_decision_at", None),
    )

    # Energy drifts linearly with elapsed time at the activity rate.
    rate = ENERGY_RATE_PER_HOUR[current.activity]
    new_energy = clamp_energy(current.energy + (elapsed / 3600.0) * rate)

    activity_age_s = max(
        0.0,
        (moment - _parse_iso(current.activity_started_at, moment)).total_seconds(),
    )

    new_activity = current.activity
    limit = ACTIVITY_DURATION_LIMIT_S.get(current.activity)
    if limit is not None and activity_age_s >= limit:
        new_activity = "idle"
    elif current.activity == "sleeping" and (
        new_energy >= MAX_ENERGY and activity_age_s >= 90 * 60
    ):
        new_activity = "idle"
    elif current.activity == "resting" and (
        new_energy >= 95 and activity_age_s >= 30 * 60
    ):
        new_activity = "idle"
    elif current.activity == "idle":
        if _is_night(moment, tz) and new_energy <= 30:
            new_activity = "sleeping"
        elif new_energy <= 10:
            new_activity = "resting"
    elif current.activity == "playing" and new_energy <= 5:
        new_activity = "resting"

    new_location = location_for(new_activity, moment, current.location, tz)
    # Reactive moods survive the clock: only the wall backstop (stale
    # reactive mood) or an exhausted ttl handled elsewhere clears them.
    # Baseline derivation applies when no reactive mood is armed.
    new_mood = current.mood
    new_ttl = current.mood_ttl_turns
    new_mood_set_at = current.mood_set_at
    if new_ttl > 0:
        age_s = (moment - _parse_iso(current.mood_set_at, moment)).total_seconds()
        if current.mood_set_at is None or age_s > REACTIVE_MOOD_MAX_AGE_S:
            new_mood = derive_mood(new_activity, new_energy)
            new_ttl = 0
            new_mood_set_at = None
    else:
        new_mood = derive_mood(new_activity, new_energy)
        new_mood_set_at = None
    new_time_context = derive_time_context(moment, tz)

    activity_changed = new_activity != current.activity
    if activity_changed:
        record = {
            "from": current.activity,
            "to": new_activity,
            "at": _to_iso(moment),
            "location": new_location,
        }
        history = [*current.recent_activity_history, record][-HISTORY_CAP:]
    else:
        history = current.recent_activity_history[-HISTORY_CAP:]

    updated = WorldState(
        location=new_location,
        activity=new_activity,
        energy=new_energy,
        mood=new_mood,
        time_context=new_time_context,
        activity_started_at=(
            _to_iso(moment) if activity_changed else current.activity_started_at
        ),
        last_update_at=current.last_update_at,
        recent_activity_history=history,
        mood_ttl_turns=new_ttl,
        mood_set_at=new_mood_set_at,
        last_autonomous_decision_at=current.last_autonomous_decision_at,
    )
    return updated, activity_changed


def reconcile(
    state: WorldState,
    now: Optional[datetime] = None,
    tz: Optional[str] = None,
    decide: bool = True,
) -> Tuple[WorldState, bool]:
    """Lazily reconcile state against wall-clock time (pure, no I/O).

    Computes ``elapsed = now - last_update_at`` and runs one bounded
    deterministic transition. Never replays every missed minute/hour:
    long offline gaps collapse into a single step (e.g. sleeping through
    a 2-day gap wakes to idle with full energy, not 48 hourly ticks).

    ``tz`` is the user/session IANA timezone name. Day/night-dependent
    rules and ``time_context`` use the user-local hour; persistence
    timestamps stay canonical UTC. ``None`` keeps server-UTC behavior.

    Returns ``(new_state, changed)`` where ``changed`` covers any field
    difference, including timestamp / time_context / energy-only updates.
    A second call with the same ``now`` is always a no-op.

    Only material field changes advance ``last_update_at``. Timestamp-only
    touches are deliberately NOT persisted: integer energy cannot represent
    sub-unit drift, so moving the baseline on every read would freeze
    slow drains (e.g. idle -1/hour) under frequent refreshes. Keeping the
    baseline at the last material change lets fractional elapsed time
    accumulate correctly across sparse AND frequent reconciles.

    ``decide`` runs the Autonomous Decision Layer v1 policy on the
    post-transition state and applies the pick through the same
    mechanics. Read-only callers (widget fetch) pass ``decide=False``.
    Decision evaluation is fail-soft: any error keeps the transitioned
    state intact.
    """
    moment = _ensure_aware(now) if now is not None else utcnow()
    baseline = WorldState(
        location=normalize_location(state.location),
        activity=normalize_activity(state.activity),
        energy=clamp_energy(state.energy),
        mood=normalize_mood(state.mood),
        time_context=state.time_context,
        activity_started_at=state.activity_started_at,
        last_update_at=state.last_update_at,
        recent_activity_history=[
            dict(item) for item in (state.recent_activity_history or [])
        ],
        mood_ttl_turns=max(0, int(getattr(state, "mood_ttl_turns", 0) or 0)),
        mood_set_at=getattr(state, "mood_set_at", None),
        last_autonomous_decision_at=getattr(state, "last_autonomous_decision_at", None),
    )
    last = _parse_iso(baseline.last_update_at, moment)
    if moment < last:
        # Clock skew / backwards fake clock: fail-soft, never move backwards.
        return baseline, False

    elapsed = (moment - last).total_seconds()
    updated, _ = transition(baseline, elapsed, moment, tz)

    if decide:
        try:
            decision = decide_activity(updated, moment, tz)
            applied = _apply_decision(updated, decision, moment, tz)
            if applied is not updated:
                logger.debug(
                    "Autonomous decision: action={} reason={}",
                    decision.action,
                    decision.reason,
                )
            updated = applied
        except Exception as error:
            logger.warning(
                "Autonomous decision skipped: error_type={}",
                type(error).__name__,
            )

    changed = (
        updated.location != baseline.location
        or updated.activity != baseline.activity
        or updated.energy != baseline.energy
        or updated.mood != baseline.mood
        or updated.time_context != baseline.time_context
        or updated.activity_started_at != baseline.activity_started_at
        or updated.recent_activity_history != baseline.recent_activity_history
        or updated.mood_ttl_turns != baseline.mood_ttl_turns
        or updated.mood_set_at != baseline.mood_set_at
        or updated.last_autonomous_decision_at != baseline.last_autonomous_decision_at
    )
    if not changed:
        # No material difference: keep the old baseline (including
        # last_update_at) so sub-unit energy drift can still accumulate
        # on later reconciles. Still a strict no-op for identical ``now``.
        return baseline, False
    updated.last_update_at = _to_iso(moment)
    return updated, True


# Maximum recent transitions exposed to character context (selective).
# The store keeps HISTORY_CAP; the prompt carries only the latest few so
# the model gets continuity ("tadi ngapain") without a giant history block.
CONTEXT_RECENT_LIFE_LIMIT = 2


def _normalize_emotion_label(value: Any) -> str:
    return str(value or "").strip().lower()


def apply_reactive(
    state: WorldState, emotion_keys: List[str], now: datetime
) -> Tuple[WorldState, bool]:
    """Apply one deterministic interaction-driven transition (pure, no I/O).

    ``emotion_keys`` are backend semantic-emotion labels observed on the
    just-completed turn (e.g. ``["joy"]``). They are INPUT only: the
    semantic-emotion pipeline is untouched, and no new emotion system is
    created here.

    Rules (all deterministic, strongest mapped label wins):
    - mapped label -> reactive mood (+shy ladder: embarrassed while shy
      deepens to flustered), ttl armed, mood_set_at = now;
    - no mapped label but ttl armed -> ttl decays one turn; at zero, step
      one DECAY_NEXT rung (flustered->shy, angry->irritated) or fall back
      to baseline derive_mood;
    - neutral/empty turn with no armed ttl -> no change at all;
    - charged turn (any non-neutral label): reading/playing -> idle
      (sleep/rest/meals are never interrupted by chatting);
    - charged turn: energy -1 while previously idle/reading/playing
      (rest states keep their time-rule recovery only).

    Returns ``(new_state, changed)``. Callers persist when changed.
    """
    moment = _ensure_aware(now)
    labels = [_normalize_emotion_label(k) for k in (emotion_keys or [])]
    labels = [label for label in labels if label]

    mapped: Optional[str] = None
    for candidate in EMOTION_PRIORITY:
        if candidate in labels:
            mapped = candidate
            break
    charged = mapped is not None or any(
        label not in NEUTRAL_EMOTION_LABELS for label in labels
    )

    current = WorldState(
        location=normalize_location(state.location),
        activity=normalize_activity(state.activity),
        energy=clamp_energy(state.energy),
        mood=normalize_mood(state.mood),
        time_context=state.time_context,
        activity_started_at=state.activity_started_at,
        last_update_at=state.last_update_at,
        recent_activity_history=[
            dict(item) for item in (state.recent_activity_history or [])
        ],
        mood_ttl_turns=max(0, int(getattr(state, "mood_ttl_turns", 0) or 0)),
        mood_set_at=getattr(state, "mood_set_at", None),
    )

    new_mood = current.mood
    new_ttl = current.mood_ttl_turns
    new_mood_set_at = current.mood_set_at
    mood_touched = False

    if mapped is not None:
        target, ttl = EMOTION_MOOD_MAP[mapped]
        if mapped == "embarrassed" and current.mood == "shy":
            target, ttl = "flustered", 3
        elif mapped == "embarrassed" and current.mood == "flustered":
            target, ttl = "flustered", 3
        new_mood, new_ttl = target, ttl
        new_mood_set_at = _to_iso(moment)
        mood_touched = True
    elif new_ttl > 0:
        new_ttl -= 1
        mood_touched = True
        if new_ttl <= 0:
            step = DECAY_NEXT.get(current.mood)
            if step is not None:
                new_mood, new_ttl = step
                new_mood_set_at = _to_iso(moment)
            else:
                new_mood = derive_mood(current.activity, current.energy)
                new_ttl = 0
                new_mood_set_at = None

    new_activity = current.activity
    activity_changed = False
    if charged and current.activity in REACTIVE_INTERRUPT_ACTIVITIES:
        new_activity = "idle"
        activity_changed = True

    new_energy = current.energy
    if charged and current.activity in REACTIVE_ENERGY_ACTIVITIES:
        new_energy = clamp_energy(current.energy + REACTIVE_ENERGY_DELTA)

    history = list(current.recent_activity_history)
    if activity_changed:
        history = [
            *history,
            {
                "from": current.activity,
                "to": new_activity,
                "at": _to_iso(moment),
                "location": location_for(new_activity, moment, current.location),
            },
        ][-HISTORY_CAP:]

    changed = (
        mood_touched
        or activity_changed
        or new_energy != current.energy
        or history != current.recent_activity_history
    )
    if not changed:
        return current, False
    return (
        WorldState(
            location=location_for(new_activity, moment, current.location)
            if activity_changed
            else current.location,
            activity=new_activity,
            energy=new_energy,
            mood=new_mood,
            time_context=current.time_context,
            activity_started_at=(
                _to_iso(moment) if activity_changed else current.activity_started_at
            ),
            last_update_at=_to_iso(moment),
            recent_activity_history=history,
            mood_ttl_turns=new_ttl,
            mood_set_at=new_mood_set_at,
        ),
        True,
    )


def build_world_state_context(state: WorldState) -> str:
    """Render the VERY COMPACT world context injected into the system prompt.

    Current life (activity/location/energy/mood/time) plus at most the last
    two activity transitions for continuity. No chat text, no full history.
    """
    lines = [
        "[Mili World State]",
        f"location={normalize_location(state.location)}; "
        f"activity={normalize_activity(state.activity)}; "
        f"energy={clamp_energy(state.energy)}; "
        f"mood={normalize_mood(state.mood)}; "
        f"time_context={state.time_context or derive_time_context(utcnow())}",
    ]
    recent = list(state.recent_activity_history or [])[-CONTEXT_RECENT_LIFE_LIMIT:]
    moves = [
        f"{normalize_activity(item.get('from'))} → {normalize_activity(item.get('to'))}"
        for item in recent
        if isinstance(item, dict)
    ]
    if moves:
        lines.append("Recent life: " + ", ".join(moves))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence (sibling store mirroring character_state.py patterns)
# ---------------------------------------------------------------------------


def get_world_state_path(conf_uid: str, base_dir: str = WORLD_STATE_DIR) -> str:
    """Return the on-disk path for a character world-state file."""
    if not conf_uid:
        raise ValueError("conf_uid cannot be empty")
    safe_conf_uid = _sanitize_path_component(conf_uid)
    return os.path.join(base_dir, f"{safe_conf_uid}.json")


def _state_to_dict(state: WorldState) -> Dict[str, Any]:
    return {
        "version": WORLD_STATE_VERSION,
        "location": normalize_location(state.location),
        "activity": normalize_activity(state.activity),
        "energy": clamp_energy(state.energy),
        "mood": normalize_mood(state.mood),
        "time_context": state.time_context,
        "activity_started_at": state.activity_started_at,
        "last_update_at": state.last_update_at,
        "recent_activity_history": list(state.recent_activity_history or [])[
            -HISTORY_CAP:
        ],
        "mood_ttl_turns": max(0, int(getattr(state, "mood_ttl_turns", 0) or 0)),
        "mood_set_at": getattr(state, "mood_set_at", None),
        "last_autonomous_decision_at": getattr(
            state, "last_autonomous_decision_at", None
        ),
    }


def _state_from_dict(data: Any, now: datetime, tz: Optional[str] = None) -> WorldState:
    fallback = default_state(now, tz)
    if not isinstance(data, dict):
        return fallback
    history: List[Dict[str, Any]] = []
    raw_history = data.get("recent_activity_history", [])
    if isinstance(raw_history, list):
        for item in raw_history[-HISTORY_CAP:]:
            if not isinstance(item, dict):
                continue
            entry: Dict[str, Any] = {
                "from": normalize_activity(item.get("from")),
                "to": normalize_activity(item.get("to")),
                "at": str(item.get("at", "")),
                "location": normalize_location(item.get("location", DEFAULT_LOCATION)),
            }
            # Additive decision tag only (old entries simply lack it).
            by_tag = str(item.get("by") or "")
            if by_tag:
                entry["by"] = by_tag
            history.append(entry)
    try:
        ttl_raw = data.get("mood_ttl_turns", 0)
        ttl = max(0, int(ttl_raw or 0))
    except (TypeError, ValueError):
        ttl = 0
    mood_set_at = data.get("mood_set_at")
    if mood_set_at is not None:
        mood_set_at = str(mood_set_at)
    decision_at = data.get("last_autonomous_decision_at")
    if decision_at is not None:
        decision_at = str(decision_at)
    return WorldState(
        location=normalize_location(data.get("location", fallback.location)),
        activity=normalize_activity(data.get("activity", fallback.activity)),
        energy=clamp_energy(data.get("energy", fallback.energy)),
        mood=normalize_mood(data.get("mood", fallback.mood)),
        time_context=str(
            data.get("time_context", fallback.time_context) or fallback.time_context
        ),
        activity_started_at=data.get("activity_started_at")
        or fallback.activity_started_at,
        last_update_at=data.get("last_update_at") or fallback.last_update_at,
        recent_activity_history=history,
        mood_ttl_turns=ttl,
        mood_set_at=mood_set_at,
        last_autonomous_decision_at=decision_at,
    )


def load_world_state(
    conf_uid: str,
    now: Optional[datetime] = None,
    base_dir: str = WORLD_STATE_DIR,
    tz: Optional[str] = None,
) -> WorldState:
    """Load world state without reconciling; missing/corrupt -> safe default."""
    moment = _ensure_aware(now) if now is not None else utcnow()
    filepath = get_world_state_path(conf_uid, base_dir)
    if not os.path.exists(filepath):
        return default_state(moment, tz)
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        return _state_from_dict(data, moment, tz)
    except Exception as error:
        logger.error("Failed to load world state: error_type={}", type(error).__name__)
        return default_state(moment, tz)


def save_world_state(
    conf_uid: str,
    state: WorldState,
    base_dir: str = WORLD_STATE_DIR,
) -> bool:
    """Atomically persist world state; returns success (fail-soft)."""
    filepath = get_world_state_path(conf_uid, base_dir)
    try:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        lock = _get_state_lock(os.path.abspath(filepath))
        with lock:
            _write_state_atomic(filepath, _state_to_dict(state))
        return True
    except Exception as error:
        logger.error("Failed to save world state: error_type={}", type(error).__name__)
        return False


def load_and_reconcile_world_state(
    conf_uid: str,
    now: Optional[datetime] = None,
    base_dir: str = WORLD_STATE_DIR,
    tz: Optional[str] = None,
    decide: bool = True,
) -> WorldState:
    """Load, lazily reconcile against ``now``, persist only if changed.

    This is the single entry point for conversation triggers, proactive
    checks, reconnect/history-switch and character load paths. It never
    raises: on any failure the caller gets a usable in-memory state and
    the conversation path continues unchanged. ``tz`` selects the
    user-local hour for time rules; stored timestamps stay UTC.
    ``decide=False`` keeps read-only callers (widget fetch) decision-free.
    """
    try:
        moment = _ensure_aware(now) if now is not None else utcnow()
        filepath = get_world_state_path(conf_uid, base_dir)
        existed = os.path.exists(filepath)
        state = load_world_state(conf_uid, moment, base_dir, tz)
        reconciled, changed = reconcile(state, moment, tz, decide)
        if changed or not existed:
            save_world_state(conf_uid, reconciled, base_dir)
        return reconciled
    except Exception as error:
        logger.error(
            "World state reconcile skipped: error_type={}", type(error).__name__
        )
        try:
            moment = _ensure_aware(now) if now is not None else utcnow()
            return default_state(moment)
        except Exception:
            return WorldState()
