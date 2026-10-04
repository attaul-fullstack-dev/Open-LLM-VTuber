"""Proactive V2 — deterministic gate, trigger priority, and budget.

Separates the LLM-free concerns from the LLM call itself:

* **Life clock** stays where it is: ``world_state.reconcile`` is pure
  arithmetic over elapsed time and never calls a model.
* **Cognition clock** (this module) decides *whether* a proactive cognition is
  warranted, using only deterministic signals. The LLM is invoked by the
  caller **after** :func:`evaluate_gate` returns ``allowed=True``.

Hard rules honoured here:

* No model call, no timer thread, no scheduler of its own. Everything is a
  pure function of persisted state + an injected clock.
* Every gate is evaluated BEFORE the caller marks a generation in flight, so a
  suppressed turn never reaches the provider.
* Proactive LLM calls are capped per user-local day. User-initiated chat is
  never counted against that budget.
* Absolute UTC timestamps only. No relative labels are ever persisted.
* Fail-soft: corrupt state degrades to the safest (most suppressed) reading.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from loguru import logger

STATE_DIR = "proactive_state"
STATE_VERSION = 2

# Trigger priority levels (technical execution order, not a quality ranking).
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"
PRIORITY_ORDER = (PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW)

GATE_OK = "ok"
GATE_QUIET_HOURS = "quiet_hours"
GATE_DAILY_LIMIT = "daily_hard_limit"
GATE_HOURLY_BUDGET = "hourly_budget"
GATE_MIN_GAP = "minimum_gap"
GATE_BACKOFF = "backoff_active"
GATE_DORMANT = "dormant_requires_meaningful"
GATE_UNANSWERED = "unanswered_requires_meaningful"
GATE_NO_REASON = "no_meaningful_reason"
GATE_IDLE_BUDGET = "idle_budget_exhausted"
GATE_CONNECTION = "connection_or_session"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return _aware(value).isoformat(timespec="seconds") if value else None


def _parse(value: Any) -> Optional[datetime]:
    """Parse an ISO stamp; invalid/absent -> None (never 'now')."""
    if not value:
        return None
    try:
        return _aware(datetime.fromisoformat(str(value)))
    except (TypeError, ValueError):
        return None


def _sanitize(conf_uid: str) -> str:
    from .chat_history_manager import _sanitize_path_component

    return _sanitize_path_component(conf_uid)


# Daily proactive LLM ceiling. 35 was the default shipped with the first
# Proactive V2 build; it was never a user-chosen value. A config still
# carrying it predates the revision to 60 and would otherwise silently keep the
# old ceiling, so it migrates. Every other explicit value - including 0 or a
# custom ceiling - is a deliberate override and is preserved as-is.
DEFAULT_DAILY_HARD_LIMIT = 60
LEGACY_DAILY_HARD_LIMIT = 35

_legacy_limit_notice_done = False


def _log_legacy_limit_migration() -> None:
    """Tell the operator once that the legacy ceiling was migrated."""
    global _legacy_limit_notice_done
    if _legacy_limit_notice_done:
        return
    _legacy_limit_notice_done = True
    logger.info(
        "Proactive daily hard limit migrated: legacy value {} -> {}. "
        "Update the config to remove the stale override.",
        LEGACY_DAILY_HARD_LIMIT,
        DEFAULT_DAILY_HARD_LIMIT,
    )


def normalize_daily_hard_limit(value: Any) -> int:
    """Resolve the effective daily ceiling, migrating the legacy default.

    Minimal and safe: exactly one sentinel value migrates. Users can still
    choose any other ceiling, now or later, and it is honoured.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return DEFAULT_DAILY_HARD_LIMIT
    if parsed == LEGACY_DAILY_HARD_LIMIT:
        _log_legacy_limit_migration()
        return DEFAULT_DAILY_HARD_LIMIT
    return parsed


@dataclass(frozen=True)
class ProactiveGateConfig:
    """Proactive V2 limits. Values are the agreed baseline, not suggestions."""

    minimum_proactive_gap_seconds: int = 900  # 15 minutes
    proactive_daily_hard_limit: int = DEFAULT_DAILY_HARD_LIMIT  # hard ceiling
    maximum_unanswered_consecutive: int = 2
    ignored_threshold_before_backoff: int = 2
    backoff_multiplier: float = 2.0
    max_backoff_seconds: int = 21600  # 6 hours
    quiet_hours_start_hour: int = 23
    quiet_hours_end_hour: int = 7
    meaningful_trigger_budget_per_hour: int = 2
    idle_trigger_budget_per_hour: int = 0
    idle_trigger_budget_per_day: int = 0
    behavior_on_budget_exhausted: str = "degrade_to_silent"

    def __post_init__(self) -> None:
        """Normalise the legacy ceiling on EVERY construction path.

        All three runtime construction sites plus ``from_dict`` funnel through
        this dataclass, so normalising here is the single place that stops an
        old config file from silently restoring the previous ceiling.
        """
        effective = normalize_daily_hard_limit(self.proactive_daily_hard_limit)
        if effective != self.proactive_daily_hard_limit:
            object.__setattr__(self, "proactive_daily_hard_limit", effective)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "minimum_proactive_gap_seconds": int(self.minimum_proactive_gap_seconds),
            "proactive_daily_hard_limit": int(self.proactive_daily_hard_limit),
            "maximum_unanswered_consecutive": int(self.maximum_unanswered_consecutive),
            "ignored_threshold_before_backoff": int(
                self.ignored_threshold_before_backoff
            ),
            "backoff_multiplier": float(self.backoff_multiplier),
            "max_backoff_seconds": int(self.max_backoff_seconds),
            "quiet_hours_start_hour": int(self.quiet_hours_start_hour),
            "quiet_hours_end_hour": int(self.quiet_hours_end_hour),
            "meaningful_trigger_budget_per_hour": int(
                self.meaningful_trigger_budget_per_hour
            ),
            "idle_trigger_budget_per_hour": int(self.idle_trigger_budget_per_hour),
            "idle_trigger_budget_per_day": int(self.idle_trigger_budget_per_day),
            "behavior_on_budget_exhausted": str(self.behavior_on_budget_exhausted),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ProactiveGateConfig":
        base = cls()
        if not isinstance(data, dict):
            return base
        out: Dict[str, Any] = {}
        for key, default in base.to_dict().items():
            raw = data.get(key, default)
            try:
                out[key] = (
                    type(default)(raw) if not isinstance(default, bool) else bool(raw)
                )
            except (TypeError, ValueError):
                out[key] = default
        return cls(**out)


@dataclass
class ProactiveBudgetState:
    """Persisted proactive accounting. Absolute UTC stamps only."""

    last_proactive_at: Optional[str] = None
    consecutive_unanswered: int = 0
    backoff_until: Optional[str] = None
    dormant_until: Optional[str] = None
    daily_request_count: int = 0
    daily_count_date: Optional[str] = None  # YYYY-MM-DD in the USER timezone
    hourly_meaningful_count: int = 0
    hourly_count_hour: Optional[str] = None  # YYYY-MM-DDTHH in the USER timezone
    total_proactive_count: int = 0
    version: int = STATE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "last_proactive_at": self.last_proactive_at,
            "consecutive_unanswered": int(self.consecutive_unanswered),
            "backoff_until": self.backoff_until,
            "dormant_until": self.dormant_until,
            "daily_request_count": int(self.daily_request_count),
            "daily_count_date": self.daily_count_date,
            "hourly_meaningful_count": int(self.hourly_meaningful_count),
            "hourly_count_hour": self.hourly_count_hour,
            "total_proactive_count": int(self.total_proactive_count),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ProactiveBudgetState":
        if not isinstance(data, dict):
            return cls()
        state = cls()
        state.last_proactive_at = _parse(data.get("last_proactive_at")) and _iso(
            _parse(data.get("last_proactive_at"))
        )
        state.backoff_until = _iso(_parse(data.get("backoff_until")))
        state.dormant_until = _iso(_parse(data.get("dormant_until")))
        for key in (
            "consecutive_unanswered",
            "daily_request_count",
            "hourly_meaningful_count",
            "total_proactive_count",
        ):
            try:
                setattr(state, key, max(0, int(data.get(key, 0) or 0)))
            except (TypeError, ValueError):
                setattr(state, key, 0)
        for key in ("daily_count_date", "hourly_count_hour"):
            raw = data.get(key)
            setattr(
                state, key, str(raw) if isinstance(raw, str) and raw.strip() else None
            )
        return state

    def copy(self) -> "ProactiveBudgetState":
        return ProactiveBudgetState.from_dict(self.to_dict())


def state_path(conf_uid: str, base_dir: str = STATE_DIR) -> str:
    if not conf_uid:
        raise ValueError("conf_uid cannot be empty")
    return os.path.join(base_dir, f"{_sanitize(conf_uid)}.json")


def load_proactive_state(
    conf_uid: str, base_dir: str = STATE_DIR
) -> ProactiveBudgetState:
    """Load persisted accounting; missing/corrupt -> zeroed state (fail-soft)."""
    try:
        path = state_path(conf_uid, base_dir)
        if not os.path.exists(path):
            return ProactiveBudgetState()
        with open(path, "r", encoding="utf-8") as handle:
            return ProactiveBudgetState.from_dict(json.load(handle))
    except Exception as error:
        logger.warning(
            "Proactive state unreadable (starting safe): type={}",
            type(error).__name__,
        )
        return ProactiveBudgetState()


def save_proactive_state(
    conf_uid: str, state: ProactiveBudgetState, base_dir: str = STATE_DIR
) -> bool:
    """Persist atomically. Never raises into the conversation path."""
    try:
        path = state_path(conf_uid, base_dir)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=1)
        os.replace(temporary, path)
        return True
    except Exception as error:
        logger.warning("Proactive state not persisted: type={}", type(error).__name__)
        return False


# ---------------------------------------------------------------------------
# Timezone helpers — always the USER timezone, never the server's.
# ---------------------------------------------------------------------------


def resolve_user_tz(tz: Optional[str]) -> ZoneInfo:
    try:
        if tz:
            return ZoneInfo(str(tz))
    except Exception:
        logger.debug("Unknown user timezone; falling back to UTC: tz={}", tz)
    return ZoneInfo("UTC")


def local_day_key(moment: datetime, tz: ZoneInfo) -> str:
    return _aware(moment).astimezone(tz).strftime("%Y-%m-%d")


def local_hour_key(moment: datetime, tz: ZoneInfo) -> str:
    return _aware(moment).astimezone(tz).strftime("%Y-%m-%dT%H")


def is_quiet_hours(moment: datetime, tz: ZoneInfo, config: ProactiveGateConfig) -> bool:
    """True inside the user-local quiet window, cross-midnight safe."""
    hour = _aware(moment).astimezone(tz).hour
    start = int(config.quiet_hours_start_hour) % 24
    end = int(config.quiet_hours_end_hour) % 24
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end  # window wraps midnight


def roll_counters(state: ProactiveBudgetState, moment: datetime, tz: ZoneInfo) -> None:
    """Reset the daily counter at user-local midnight and the hourly one on the hour.

    Deliberately NOT called on reconnect/reconnect of the websocket: only the
    passage of local time moves these counters.
    """
    day = local_day_key(moment, tz)
    if state.daily_count_date != day:
        state.daily_count_date = day
        state.daily_request_count = 0
    hour = local_hour_key(moment, tz)
    if state.hourly_count_hour != hour:
        state.hourly_count_hour = hour
        state.hourly_meaningful_count = 0


def current_gap_seconds(
    state: ProactiveBudgetState, config: ProactiveGateConfig
) -> float:
    """Escalating gap: base gap, doubled per extra ignored proactive turn.

    Genuinely multiplicative (not a two-value switch like the previous
    implementation) and hard-capped at ``max_backoff_seconds``.
    """
    base = max(0.0, float(config.minimum_proactive_gap_seconds))
    ignored = max(0, int(state.consecutive_unanswered))
    threshold = max(1, int(config.ignored_threshold_before_backoff))
    if ignored < threshold:
        return base
    multiplier = max(1.0, float(config.backoff_multiplier))
    escalated = base * (multiplier ** (ignored - threshold + 1))
    return float(min(escalated, float(config.max_backoff_seconds)))


# ---------------------------------------------------------------------------
# Trigger priority — deterministic only, never a model call.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerReason:
    """Why the cognition clock believes a proactive turn is warranted."""

    priority: str = PRIORITY_LOW
    reason: str = "generic_idle"
    detail: str = ""

    @property
    def is_meaningful(self) -> bool:
        return self.priority in (PRIORITY_HIGH, PRIORITY_MEDIUM)


def classify_trigger(
    *,
    user_question_pending: bool = False,
    unfinished_topic: bool = False,
    has_useful_memory: bool = False,
    memory_relevance_score: float = 0.0,
    memory_relevant_threshold: float = 0.6,
    meaningful_life_event: bool = False,
    relationship_event: bool = False,
    explicit_reminder: bool = False,
    goal_evidence: bool = False,
) -> TriggerReason:
    """Map existing deterministic signals onto a priority. Pure, no I/O.

    LOW means "generic idle thought". With ``idle_trigger_budget_* = 0`` the
    gate suppresses LOW entirely, so the model is never called merely to look
    for a reason to speak.

    ``goal_evidence`` is the Autonomous Decision Layer's single contribution:
    it is raised only from a stored, explicitly *active* goal matched with
    fresh deterministic evidence (see
    ``autonomous_decision.classify_goal_evidence``). It ranks HIGH because a
    verified, self-declared objective is a real reason to speak, ahead of the
    reactive MEDIUM sources, and it still has to clear every budget, gap,
    quiet-hour and backoff rule below -- a priority is never permission.
    """
    if explicit_reminder:
        return TriggerReason(
            PRIORITY_HIGH, "explicit_reminder", "user-defined reminder"
        )
    if goal_evidence:
        return TriggerReason(
            PRIORITY_HIGH, "goal_evidence", "active goal with fresh evidence"
        )
    if user_question_pending or unfinished_topic:
        return TriggerReason(PRIORITY_HIGH, "unfinished_topic", "unresolved user topic")
    if relationship_event:
        return TriggerReason(
            PRIORITY_MEDIUM, "relationship_event", "relationship change"
        )
    if meaningful_life_event:
        return TriggerReason(
            PRIORITY_MEDIUM, "life_event", "meaningful life transition"
        )
    if has_useful_memory and float(memory_relevance_score) >= float(
        memory_relevant_threshold
    ):
        return TriggerReason(
            PRIORITY_MEDIUM, "relevant_memory", "relevant episodic recall"
        )
    return TriggerReason(PRIORITY_LOW, "generic_idle", "no meaningful reason available")


@dataclass(frozen=True)
class GateDecision:
    """Result of the deterministic gate. ``allowed`` is the only LLM trigger."""

    allowed: bool
    reason: str
    priority: str
    daily_remaining: int = 0
    gap_seconds: float = 0.0
    backoff_until: Optional[str] = None
    dormant_until: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": bool(self.allowed),
            "reason": self.reason,
            "priority": self.priority,
            "daily_remaining": int(self.daily_remaining),
            "gap_seconds": round(float(self.gap_seconds), 3),
            "backoff_until": self.backoff_until,
            "dormant_until": self.dormant_until,
        }


def evaluate_gate(
    state: ProactiveBudgetState,
    config: ProactiveGateConfig,
    trigger: TriggerReason,
    *,
    now: Optional[datetime] = None,
    tz: Optional[str] = None,
    connection_valid: bool = True,
    generation_in_progress: bool = False,
) -> GateDecision:
    """Single deterministic entry point, evaluated before any provider call.

    Check order is deliberate and fail-safe: the cheapest absolute vetoes run
    first, and every ambiguous condition resolves to *suppress*.
    """
    moment = _aware(now) if now is not None else _utcnow()
    zone = resolve_user_tz(tz)
    try:
        roll_counters(state, moment, zone)

        def decide(allowed: bool, reason: str) -> GateDecision:
            return GateDecision(
                allowed=allowed,
                reason=reason,
                priority=trigger.priority,
                daily_remaining=max(
                    0,
                    int(config.proactive_daily_hard_limit)
                    - int(state.daily_request_count),
                ),
                gap_seconds=current_gap_seconds(state, config),
                backoff_until=state.backoff_until,
                dormant_until=state.dormant_until,
            )

        if not connection_valid:
            return decide(False, GATE_CONNECTION)
        if generation_in_progress:
            return decide(False, GATE_CONNECTION)

        # LOW idle thoughts are budgeted at zero by default: a generic idle
        # timer must never be able to buy a model call.
        if not trigger.is_meaningful:
            if (
                int(config.idle_trigger_budget_per_hour) <= 0
                and int(config.idle_trigger_budget_per_day) <= 0
            ):
                return decide(False, GATE_IDLE_BUDGET)
            return decide(False, GATE_NO_REASON)

        if is_quiet_hours(moment, zone, config):
            return decide(False, GATE_QUIET_HOURS)

        if int(state.daily_request_count) >= int(config.proactive_daily_hard_limit):
            return decide(False, GATE_DAILY_LIMIT)

        if int(config.meaningful_trigger_budget_per_hour) > 0 and int(
            state.hourly_meaningful_count
        ) >= int(config.meaningful_trigger_budget_per_hour):
            # No HIGH bypass exists in the current architecture, so the safe
            # choice is defer, not invent a new exemption.
            return decide(False, GATE_HOURLY_BUDGET)

        dormant_until = _parse(state.dormant_until)
        if dormant_until is not None and moment < dormant_until:
            # DORMANT: only a meaningful trigger may wake, and only after the
            # window expires. Meaningfulness was already required above.
            return decide(False, GATE_DORMANT)

        backoff_until = _parse(state.backoff_until)
        if backoff_until is not None and moment < backoff_until:
            return decide(False, GATE_BACKOFF)

        last_proactive = _parse(state.last_proactive_at)
        if last_proactive is not None:
            elapsed = (moment - last_proactive).total_seconds()
            if elapsed < float(config.minimum_proactive_gap_seconds):
                return decide(False, GATE_MIN_GAP)

        if (
            int(state.consecutive_unanswered)
            >= int(config.maximum_unanswered_consecutive)
            and not trigger.is_meaningful
        ):
            return decide(False, GATE_UNANSWERED)

        return decide(True, GATE_OK)
    except Exception as error:  # pragma: no cover - defensive
        logger.warning(
            "Proactive gate failed safe (suppressed): type={}", type(error).__name__
        )
        return GateDecision(
            allowed=False,
            reason="gate_error_suppressed",
            priority=trigger.priority,
        )


def record_proactive_dispatch(
    state: ProactiveBudgetState,
    config: ProactiveGateConfig,
    trigger: TriggerReason,
    *,
    now: Optional[datetime] = None,
    tz: Optional[str] = None,
) -> ProactiveBudgetState:
    """Account for ONE proactive LLM call. Call before/with the dispatch.

    Increments the daily hard-limit counter and, for meaningful turns, the
    hourly counter. User-initiated chat never routes through here.
    """
    moment = _aware(now) if now is not None else _utcnow()
    zone = resolve_user_tz(tz)
    roll_counters(state, moment, zone)
    state.last_proactive_at = _iso(moment)
    state.daily_request_count = int(state.daily_request_count) + 1
    state.total_proactive_count = int(state.total_proactive_count) + 1
    if trigger.is_meaningful:
        state.hourly_meaningful_count = int(state.hourly_meaningful_count) + 1
    # The gap only escalates once the previous proactive went unanswered.
    state.consecutive_unanswered = int(state.consecutive_unanswered) + 1
    gap = current_gap_seconds(state, config)
    state.backoff_until = _iso(moment + timedelta(seconds=gap))
    if int(state.consecutive_unanswered) >= int(config.maximum_unanswered_consecutive):
        state.dormant_until = _iso(
            moment + timedelta(seconds=int(config.max_backoff_seconds))
        )
    return state


def record_proactive_answered(
    state: ProactiveBudgetState, *, now: Optional[datetime] = None
) -> ProactiveBudgetState:
    """User replied: clear the unanswered/dormant/backoff pressure at once."""
    state.consecutive_unanswered = 0
    state.backoff_until = None
    state.dormant_until = None
    return state


def record_suppressed(
    state: ProactiveBudgetState,
    config: ProactiveGateConfig,
    *,
    now: Optional[datetime] = None,
    tz: Optional[str] = None,
) -> ProactiveBudgetState:
    """A turn the user ignored without us speaking: grow the gap only.

    Does NOT touch the daily budget — a suppressed turn costs no model call.
    """
    moment = _aware(now) if now is not None else _utcnow()
    zone = resolve_user_tz(tz)
    roll_counters(state, moment, zone)
    state.consecutive_unanswered = int(state.consecutive_unanswered) + 1
    gap = current_gap_seconds(state, config)
    state.backoff_until = _iso(moment + timedelta(seconds=gap))
    if int(state.consecutive_unanswered) >= int(config.maximum_unanswered_consecutive):
        state.dormant_until = _iso(
            moment + timedelta(seconds=int(config.max_backoff_seconds))
        )
    return state


__all__ = [
    "DEFAULT_DAILY_HARD_LIMIT",
    "GATE_BACKOFF",
    "GATE_CONNECTION",
    "GATE_DAILY_LIMIT",
    "GATE_DORMANT",
    "GATE_HOURLY_BUDGET",
    "GATE_IDLE_BUDGET",
    "GATE_MIN_GAP",
    "GATE_NO_REASON",
    "GATE_OK",
    "GATE_QUIET_HOURS",
    "GATE_UNANSWERED",
    "PRIORITY_HIGH",
    "PRIORITY_LOW",
    "PRIORITY_MEDIUM",
    "LEGACY_DAILY_HARD_LIMIT",
    "PRIORITY_ORDER",
    "GateDecision",
    "ProactiveBudgetState",
    "ProactiveGateConfig",
    "TriggerReason",
    "classify_trigger",
    "current_gap_seconds",
    "evaluate_gate",
    "is_quiet_hours",
    "load_proactive_state",
    "normalize_daily_hard_limit",
    "local_day_key",
    "local_hour_key",
    "record_proactive_answered",
    "record_proactive_dispatch",
    "record_suppressed",
    "resolve_user_tz",
    "roll_counters",
    "save_proactive_state",
    "state_path",
]
