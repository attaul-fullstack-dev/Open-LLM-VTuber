"""Autonomous Decision Layer — one typed vocabulary over the existing decision makers.

This module is a *decision layer*, not a second chatbot and not a second
scheduler. It owns no timers, no loops and no I/O.

What already existed and is reused unchanged:

- ``world_state.decide_activity`` (+ ``DecisionResult`` / ``DecisionInputs``)
  decides what Mili does in-world while idle. Committed in ``2acac8f``.
- ``proactive_chat.ProactiveStateMachine`` / ``ProactiveIntentDecision``
  decide whether and what Mili says proactively. Pre-existing, still the only
  proactive execution pipeline.

What this module adds, and only this:

- one typed outcome vocabulary shared by both (``DecisionOutcome``),
- a typed decision record with a UTC cooldown anchor,
- pure adapters that classify an existing decision into that vocabulary.

Design rules honoured here:

- deterministic: same input -> same output, no randomness, no clock reads
  unless the caller passes ``now``;
- fail-soft: corrupt or missing inputs degrade to ``no_decision``;
- timezone-aware: everything is UTC ISO-8601 with offset;
- persistence-safe: the only persisted anchor is the existing
  ``last_autonomous_decision_at`` field in the world state file;
- no LLM call, no scheduler, no autonomous loop, no behaviour change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional

from loguru import logger

from .world_state import (
    DECISION_COOLDOWN_S,
    EPISODIC_CONTEXT_MAX_AGE_H,
    EPISODIC_CONTEXT_MAX_EVENTS,
    DecisionContextSignals,
    DecisionInputs,
    DecisionResult,
)

# Cooldown anchors are shared with the existing world-state policy so the two
# decision makers can never disagree about "did we just decide something?".
DEFAULT_COOLDOWN_S = DECISION_COOLDOWN_S

# Evidence beyond this age cannot justify a goal-driven move.
GOAL_EVIDENCE_MAX_AGE_H = 72

DecisionOutcome = str
"""What kind of autonomous move was decided.

- ``no_decision``            nothing to do (also the common, valid outcome)
- ``idle_behavior``          do something in-world while idle
- ``proactive_interaction``  speak to the user unprompted
- ``goal_related_behavior``  the pick was driven by an active goal
- ``relationship_related_behavior`` the pick was driven by relationship state
"""

OUTCOME_NO_DECISION: DecisionOutcome = "no_decision"
OUTCOME_IDLE_BEHAVIOR: DecisionOutcome = "idle_behavior"
OUTCOME_PROACTIVE_INTERACTION: DecisionOutcome = "proactive_interaction"
OUTCOME_GOAL_BEHAVIOR: DecisionOutcome = "goal_related_behavior"
OUTCOME_RELATIONSHIP_BEHAVIOR: DecisionOutcome = "relationship_related_behavior"

ALL_OUTCOMES = (
    OUTCOME_NO_DECISION,
    OUTCOME_IDLE_BEHAVIOR,
    OUTCOME_PROACTIVE_INTERACTION,
    OUTCOME_GOAL_BEHAVIOR,
    OUTCOME_RELATIONSHIP_BEHAVIOR,
)

# World-reason -> outcome. This is a classification of reasons that already
# exist in world_state; it introduces no new decision rule.
_WORLD_REASON_OUTCOMES: Dict[str, DecisionOutcome] = {
    "goal_related": OUTCOME_GOAL_BEHAVIOR,
    "relationship_bias": OUTCOME_RELATIONSHIP_BEHAVIOR,
    # Preference- and mood-driven picks are still idle behaviour; only goals
    # and relationship get their own outcome so the taxonomy stays meaningful.
    "preference": OUTCOME_IDLE_BEHAVIOR,
    "mood_bias": OUTCOME_IDLE_BEHAVIOR,
    "stale_idle": OUTCOME_IDLE_BEHAVIOR,
    "duration_limit": OUTCOME_IDLE_BEHAVIOR,
    "low_energy": OUTCOME_IDLE_BEHAVIOR,
    "night_rest": OUTCOME_IDLE_BEHAVIOR,
    "no_change": OUTCOME_NO_DECISION,
    "cooldown_active": OUTCOME_NO_DECISION,
}


@dataclass(frozen=True)
class AutonomousDecision:
    """Typed, explainable autonomous decision (pure data, no I/O).

    ``cooldown_until`` is an absolute UTC instant, never a relative label, so
    it stays correct across midnight, timezone changes and restarts.
    """

    outcome: DecisionOutcome
    reason: str
    decided_at: str  # ISO-8601 UTC
    cooldown_until: Optional[str] = None  # ISO-8601 UTC or None
    state_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def acts(self) -> bool:
        """True when this decision means Mili does something."""
        return self.outcome != OUTCOME_NO_DECISION


def _utc_iso(moment: datetime) -> str:
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def cooldown_until(moment: datetime, cooldown_s: float = DEFAULT_COOLDOWN_S) -> str:
    """Absolute UTC cooldown anchor ``cooldown_s`` after ``moment``."""
    return _utc_iso(moment + timedelta(seconds=max(0.0, float(cooldown_s))))


def is_cooldown_active(
    last_decision_at: Any,
    moment: datetime,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
) -> bool:
    """Same rule as ``world_state._decision_cooldown_active``, fail-soft.

    A missing or unparseable anchor means "not on cooldown" so a corrupt
    timestamp can never permanently mute Mili.
    """
    anchor = _parse_iso(last_decision_at)
    if anchor is None:
        return False
    delta_s = (moment - anchor).total_seconds()
    if delta_s < 0:
        # Clock skew / backwards fake clock: never punish the future.
        return False
    return delta_s < max(0.0, float(cooldown_s))


def no_decision(
    reason: str,
    moment: datetime,
    *,
    cooldown_until_iso: Optional[str] = None,
    state_version: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AutonomousDecision:
    """Build the valid 'do nothing' decision."""
    return AutonomousDecision(
        outcome=OUTCOME_NO_DECISION,
        reason=str(reason or "no_decision"),
        decided_at=_utc_iso(moment),
        cooldown_until=cooldown_until_iso,
        state_version=state_version,
        metadata=dict(metadata or {}),
    )


def classify_world_decision(
    result: Optional[DecisionResult],
    moment: datetime,
    *,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
) -> AutonomousDecision:
    """Adapt an existing ``decide_activity`` result into the shared vocabulary.

    Pure classification: the policy already ran, this only labels it. A
    ``None`` or malformed result becomes ``no_decision``.
    """
    try:
        if result is None:
            return no_decision("no_result", moment)
        reason = str(getattr(result, "reason", "") or "no_change")
        action = getattr(result, "action", None)
        state_version = getattr(result, "state_version", None)
        if not action:
            return AutonomousDecision(
                outcome=_WORLD_REASON_OUTCOMES.get(reason, OUTCOME_NO_DECISION),
                reason=reason,
                decided_at=_utc_iso(moment),
                cooldown_until=None,
                state_version=state_version,
                metadata={"activity": None},
            )
        return AutonomousDecision(
            outcome=_WORLD_REASON_OUTCOMES.get(reason, OUTCOME_IDLE_BEHAVIOR),
            reason=reason,
            decided_at=_utc_iso(moment),
            cooldown_until=cooldown_until(moment, cooldown_s),
            state_version=state_version,
            metadata={"activity": str(action)},
        )
    except Exception as error:  # pragma: no cover - defensive
        logger.warning(
            "Autonomous decision classification failed (no_decision): type={}",
            type(error).__name__,
        )
        return no_decision("classification_failed", moment)


def classify_proactive_decision(
    *,
    eligible: bool,
    reason: str,
    moment: datetime,
    strategy: Optional[str] = None,
    intent: Optional[str] = None,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
    state_version: Optional[str] = None,
) -> AutonomousDecision:
    """Adapt the existing proactive machine's verdict into the vocabulary.

    ``eligible`` is exactly ``ProactiveStateMachine.is_eligible``; this adds no
    gating of its own, it only labels the outcome.
    """
    try:
        if not eligible:
            return AutonomousDecision(
                outcome=OUTCOME_NO_DECISION,
                reason=str(reason or "not_eligible"),
                decided_at=_utc_iso(moment),
                cooldown_until=None,
                state_version=state_version,
                metadata={"strategy": strategy, "intent": intent},
            )
        return AutonomousDecision(
            outcome=OUTCOME_PROACTIVE_INTERACTION,
            reason=str(reason or "eligible"),
            decided_at=_utc_iso(moment),
            cooldown_until=cooldown_until(moment, cooldown_s),
            state_version=state_version,
            metadata={"strategy": strategy, "intent": intent},
        )
    except Exception as error:  # pragma: no cover - defensive
        logger.warning(
            "Proactive decision classification failed (no_decision): type={}",
            type(error).__name__,
        )
        return no_decision("classification_failed", moment)


def build_decision_inputs(
    *,
    relationship_status: Any = None,
    goal_activities: Any = (),
    preferred_activities: Any = (),
    interaction_preferences: Any = (),
    episodic_events: Any = (),
    query: Any = "",
    moment: Optional[datetime] = None,
    context: Any = None,
    continuity_event_ids: Any = (),
    mood_bias: bool = False,
) -> DecisionInputs:
    """Assemble the existing ``DecisionInputs`` from available character state.

    Reuses ``world_state.DecisionInputs`` verbatim — this function only maps
    caller state onto it, so there is still exactly one inputs type.

    Context (ADL v2): pass either a prebuilt ``context`` (``DecisionContext-
    Signals``) or the raw sources — ``interaction_preferences`` plus
    ``episodic_events`` with a ``query`` — and the compact signals are derived
    here. Raw preference text and event text never enter ``DecisionInputs``;
    only ``category:polarity`` labels and counts/timestamps do. With neither,
    ``context`` stays ``None`` and behaviour is exactly pre-v2.
    """
    resolved = context
    if resolved is None and (interaction_preferences or episodic_events):
        reference = moment if moment is not None else datetime.now(timezone.utc)
        candidate = build_context_signals(
            interaction_preferences=interaction_preferences,
            episodic_events=episodic_events,
            query=query,
            moment=reference,
            continuity_event_ids=continuity_event_ids,
        )
        # Same predicate as the decision rule: only a verified continuity
        # candidate may make the context relevant. Preferences and unverified
        # episodic recall stay observable but behaviourally inert.
        if candidate.has_continuity(EPISODIC_CONTEXT_MAX_AGE_H):
            resolved = candidate
    return DecisionInputs(
        relationship_status=(
            str(relationship_status).strip().lower() if relationship_status else None
        ),
        goal_activities=tuple(goal_activities or ()),
        preferred_activities=tuple(preferred_activities or ()),
        mood_bias=bool(mood_bias),
        context=resolved,
    )


def _safe_count(value: Any) -> int:
    """Length of an input collection, fail-soft to 0 (never raises)."""
    try:
        return len(tuple(value or ()))
    except Exception:
        return 0


def _safe_status(value: Any) -> Optional[str]:
    """Lowercased relationship status, fail-soft to None (never raises)."""
    try:
        text = str(value or "").strip().lower()
        return text or None
    except Exception:
        return None


def _safe_nonneg_int(value: Any) -> int:
    """Non-negative integer, fail-soft to 0 (never raises)."""
    try:
        return max(0, int(value))
    except Exception:
        return 0


def decision_inputs_summary(
    *,
    relationship_status: Any = None,
    goal_activities: Any = (),
    preferred_activities: Any = (),
    interaction_preferences: Any = (),
    episodic_event_count: int = 0,
) -> Dict[str, Any]:
    """Counts-only view of every decision input, safe to log.

    Each field is normalised independently: one corrupt input degrades that
    field only, never the whole summary. Carries no message text, no memory
    content, no episodic content.
    """
    try:
        return {
            "relationship_status": _safe_status(relationship_status),
            "goal_activities": _safe_count(goal_activities),
            "preferred_activities": _safe_count(preferred_activities),
            "interaction_preferences": _safe_count(interaction_preferences),
            "episodic_events": _safe_nonneg_int(episodic_event_count),
        }
    except Exception as error:  # pragma: no cover - defensive
        logger.warning(
            "Decision inputs summary unavailable: type={}", type(error).__name__
        )
        return {}


def _recency_bounded_candidates(events: list, limit: int) -> list:
    """Newest-by-``occurred_at`` slice, bounded, timestamps read only.

    Timestamp-only pass: no event text is scored or inspected here, so this
    cannot become a back door for the whole store into the policy. Its only
    job is to guarantee that recency cannot be starved by relevance ranking.
    """
    rows: list = []
    for item in events:
        try:
            if not isinstance(item, dict):
                continue
            stamp = _parse_iso(item.get("occurred_at"))
            if stamp is None:
                # No event time -> recency is unknowable, so it cannot be a
                # recency candidate (occurred_at stays the source of truth).
                continue
            rows.append((stamp, str(item.get("created_at", "") or ""), item))
        except Exception:
            continue
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [row[2] for row in rows[: max(1, int(limit))]]


def _merge_bounded_candidates(
    relevance_bounded: list, recency_bounded: list, limit: int
) -> list:
    """Union of both bounded sets, capped, newest relevant always retained.

    Pure relevance top-N can hide the most recent relevant event behind older
    events that merely score higher (an extra shared keyword). Keeping the
    newest candidate first and filling the remaining slots in the retriever's
    own deterministic order removes that starvation without enlarging the
    output: the result never exceeds ``limit`` entries.
    """
    cap = max(1, int(limit))
    merged: list = []
    seen: set = set()

    def _add(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        key = id(item)
        if key in seen:
            return False
        seen.add(key)
        merged.append(item)
        return len(merged) < cap

    for group in (recency_bounded, relevance_bounded):
        for item in group or []:
            if not _add(item):
                return merged[:cap]
    return merged[:cap]


def build_context_signals(
    *,
    interaction_preferences: Any = (),
    episodic_events: Any = (),
    query: Any = "",
    moment: datetime,
    max_events: int = EPISODIC_CONTEXT_MAX_EVENTS,
    continuity_event_ids: Any = (),
    prefetched_relevant: Any = None,
) -> DecisionContextSignals:
    """Build the compact contextual signals used by the decision stage.

    Both sources already exist and are read through their own helpers:

    - interaction preferences -> ``interaction_preferences.active_preferences``
      (active rows only). Only counts and ``category:polarity`` labels cross
      the boundary; preference text never does. They carry NO semantic weight
      in the decision yet and never make the context look relevant.
    - episodic memory -> ``episodic_memory.retrieve_episodic_events`` (the
      existing deterministic retriever). Candidate selection is the union of a
      relevance-bounded pass and a recency-bounded pass, capped at
      ``max_events``, so the newest relevant event can never be starved by
      older higher-scoring ones. The whole store is never handed over and no
      event text is carried either.

    Semantic continuity (ADL v2): ``continuity_event_ids`` is the set of event
    ids a caller has VERIFIED as Mili's own or a shared activity. The episodic
    schema stores no actor/participant field, so nothing may infer this from
    event text: a user-only memory is a user experience, never Mili world
    state. With no verified ids the continuity signal stays False and no
    recent user memory can hold Mili's activity.

    Time correctness: recency is measured from ``occurred_at`` (event time),
    never ``created_at``, and a missing/invalid stamp drops the event. Fail-
    soft throughout: any broken input yields fewer signals, never an
    exception and never a permanently muted decision layer.
    """
    signals = DecisionContextSignals()
    try:
        # --- interaction preferences (durable relational signal) ---
        categories: list = []
        try:
            from .interaction_preferences import active_preferences

            for item in active_preferences(list(interaction_preferences or [])):
                category = str(item.get("category", "") or "").strip().lower()
                polarity = str(item.get("polarity", "") or "").strip().lower()
                if category and polarity:
                    categories.append(f"{category}:{polarity}")
        except Exception as error:
            logger.debug(
                "Interaction preference signals unavailable: type={}",
                type(error).__name__,
            )
        signals = _with_preferences(signals, tuple(sorted(set(categories))))

        # --- episodic recall (bounded, deterministic) ---
        try:
            from .episodic_memory import retrieve_episodic_events

            events = list(episodic_events or [])
            if not events or not str(query or "").strip():
                return signals
            limit = max(1, int(max_events))
            # A prefetched relevance selection (already produced once per turn
            # for the prompt block) is reused as-is: same store, same query,
            # same deterministic ranking, one pass instead of two.
            if prefetched_relevant is not None:
                relevance_bounded = list(prefetched_relevant)
            else:
                relevance_bounded = retrieve_episodic_events(
                    events, query, now=moment, top_n=limit
                )
            # The recency pass stays: it is what guarantees the newest relevant
            # event can never be starved by older higher-scoring ones.
            recency_bounded = retrieve_episodic_events(
                _recency_bounded_candidates(events, limit),
                query,
                now=moment,
                top_n=limit,
            )
            candidates = _merge_bounded_candidates(
                relevance_bounded, recency_bounded, limit
            )
        except Exception as error:
            logger.debug(
                "Episodic decision signals unavailable: type={}",
                type(error).__name__,
            )
            return signals

        stamps: list = []
        continuity = False
        try:
            verified = {str(item) for item in list(continuity_event_ids or [])}
        except Exception:
            verified = set()
        for item in candidates or []:
            try:
                if not isinstance(item, dict):
                    continue
                stamp = _parse_iso(item.get("occurred_at"))
                if stamp is None:
                    # No event time -> cannot judge recency; skip (fail-soft).
                    continue
                stamps.append(stamp)
                if verified and str(item.get("id", "") or "") in verified:
                    continuity = True
            except Exception:
                continue
        if not stamps:
            return signals
        newest = max(stamps)
        # Whole hours elapsed, rounded UP, so the 24h window is exact: an
        # event 24h05m old reports 25 and no longer holds. A clock skew that
        # puts an event slightly in the future clamps to 0, never negative.
        elapsed_s = (moment - newest).total_seconds()
        age_h = -(-int(elapsed_s) // 3600) if elapsed_s > 0 else 0
        return _with_episodic(
            signals,
            relevant_count=len(stamps),
            latest_occurred_at=_utc_iso(newest),
            age_hours=age_h,
            continuity_candidate=continuity,
        )
    except Exception as error:  # pragma: no cover - defensive
        logger.warning(
            "Context signals unavailable (decision continues): type={}",
            type(error).__name__,
        )
        return DecisionContextSignals()


def _with_preferences(
    signals: DecisionContextSignals, categories: tuple
) -> DecisionContextSignals:
    return DecisionContextSignals(
        episodic_relevant_count=signals.episodic_relevant_count,
        episodic_latest_occurred_at=signals.episodic_latest_occurred_at,
        episodic_age_hours=signals.episodic_age_hours,
        preference_signals=categories,
    )


def _with_episodic(
    signals: DecisionContextSignals,
    *,
    relevant_count: int,
    latest_occurred_at: str,
    age_hours: int,
    continuity_candidate: bool = False,
) -> DecisionContextSignals:
    return DecisionContextSignals(
        episodic_relevant_count=relevant_count,
        episodic_latest_occurred_at=latest_occurred_at,
        episodic_age_hours=age_hours,
        preference_signals=signals.preference_signals,
        continuity_candidate=bool(continuity_candidate),
    )


# ---------------------------------------------------------------------------
# Goal evidence — the one place a goal can become a *reason* to act.
# ---------------------------------------------------------------------------

# Deterministic evidence vocabulary, one entry per seeded goal id.
#
# These are literal substring markers over already-stored episodic event text.
# They are NOT inferred, NOT generated and NOT expanded at runtime: an event
# that contains none of them simply produces no goal evidence. Keeping them
# explicit is what makes the decision explainable ("why did Mili bring this
# up?") and what stops the layer from hallucinating progress toward a goal.
# An unknown goal id has no vocabulary and therefore can never match.
GOAL_EVIDENCE_KEYWORDS: Dict[str, tuple] = {
    "morning-reading-week": ("bacain", "baca buku", "reading", "bacaan"),
    "finish-one-book": ("selesai baca", "selesai buku", "tuntas baca", "finish the book"),
    "try-three-dishes": ("masak", "coba makan", "makan", "cobaatile", "try cook"),
}

def goal_evidence_summary(goals: Any) -> Dict[str, Any]:
    """Compact, deterministic goal state for logs and decisions.

    Shape: ``{"seed": n, "active": n, "done": n, "active_ids": (...)}``.
    Pure, bounded, fail-soft. No goal text leaves this function.
    """
    try:
        from .character_state import active_goal_ids, goal_status_counts

        counts = goal_status_counts(goals)
        ids = active_goal_ids(goals)
        return {
            "seed": int(counts.get("seed", 0)),
            "active": int(counts.get("active", 0)),
            "done": int(counts.get("done", 0)),
            "active_ids": tuple(ids),
        }
    except Exception as error:  # pragma: no cover - defensive
        logger.debug(
            "Goal evidence summary unavailable: type={}", type(error).__name__
        )
        return {"seed": 0, "active": 0, "done": 0, "active_ids": ()}


def _event_matches_goal(text: str, keywords: tuple) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in keywords)


def classify_goal_evidence(
    *,
    goals: Any,
    episodic_events: Any = (),
    moment: datetime,
    max_age_h: int = GOAL_EVIDENCE_MAX_AGE_H,
    max_events: int = EPISODIC_CONTEXT_MAX_EVENTS,
) -> AutonomousDecision:
    """Decide whether a stored goal justifies an autonomous move, right now.

    Deterministic end to end: no LLM, no clock read beyond the passed
    ``moment``, no randomness, no I/O. The four states the layer must be able
    to tell apart are all explicit outcomes:

    - no goals at all                    -> ``no_decision`` / ``no_goal``
    - goals exist but none activated     -> ``no_decision`` / ``goal_not_active``
    - an active goal already done-evidence -> ``no_decision`` / ``goal_evidence_consumed``
    - active goal + fresh unconsumed evidence -> ``goal_related_behavior``

    Evidence freshness is measured from ``occurred_at`` (event time), never
    ``created_at``, and never from a relative label. The same piece of evidence
    cannot produce a second decision: the goal carries the id of the evidence
    it already acted on, and a repeat is reported as ``goal_evidence_consumed``
    rather than re-firing.

    Fail-soft throughout: any malformed input yields ``no_decision``, never an
    exception, so a corrupt goal or event file can never mute the runtime.
    """
    try:
        summary = goal_evidence_summary(goals)
        active_ids = summary["active_ids"]
        if summary["seed"] == 0 and summary["active"] == 0 and summary["done"] == 0:
            return AutonomousDecision(
                outcome=OUTCOME_NO_DECISION,
                reason="no_goal",
                decided_at=_utc_iso(moment),
                cooldown_until=None,
                metadata={"goals": summary},
            )
        if not active_ids:
            return AutonomousDecision(
                outcome=OUTCOME_NO_DECISION,
                reason="goal_not_active",
                decided_at=_utc_iso(moment),
                cooldown_until=None,
                metadata={"goals": summary},
            )

        # Goal id -> (text, last consumed evidence id), read once, fail-soft.
        wanted: Dict[str, Dict[str, Optional[str]]] = {}
        for item in list(goals or []):
            try:
                if not isinstance(item, dict):
                    continue
                goal_id = str(item.get("id", "") or "").strip()
                if goal_id not in active_ids:
                    continue
                wanted[goal_id] = {
                    "consumed": str(item.get("last_evidence_id", "") or "").strip()
                    or None
                }
            except Exception:
                continue
        if not wanted:
            return AutonomousDecision(
                outcome=OUTCOME_NO_DECISION,
                reason="goal_not_active",
                decided_at=_utc_iso(moment),
                cooldown_until=None,
                metadata={"goals": summary},
            )

        # Bounded scan over already-stored events, newest-first by occurred_at.
        try:
            candidates = list(episodic_events or [])
        except Exception:
            candidates = []
        limit = max(1, int(max_events))
        horizon_h = max(1, int(max_age_h))
        stamps: list = []
        for item in candidates:
            try:
                if not isinstance(item, dict):
                    continue
                stamp = _parse_iso(item.get("occurred_at"))
                if stamp is None:
                    continue
                age_h = (moment - stamp).total_seconds() / 3600.0
                if age_h < 0 or age_h > horizon_h:
                    continue
                stamps.append((stamp, item))
            except Exception:
                continue
        stamps.sort(key=lambda pair: pair[0], reverse=True)
        stamps = stamps[:limit]

        consumed_seen = False
        for stamp, item in stamps:
            text = item.get("event_text") or item.get("content") or ""
            event_id = str(item.get("id", "") or "").strip()
            for goal_id, info in wanted.items():
                keywords = GOAL_EVIDENCE_KEYWORDS.get(goal_id)
                if not keywords:
                    continue
                if not _event_matches_goal(text, keywords):
                    continue
                consumed = info.get("consumed")
                if consumed and event_id and consumed == event_id:
                    consumed_seen = True
                    continue
                return AutonomousDecision(
                    outcome=OUTCOME_GOAL_BEHAVIOR,
                    reason="goal_evidence",
                    decided_at=_utc_iso(moment),
                    cooldown_until=cooldown_until(moment, DEFAULT_COOLDOWN_S),
                    metadata={
                        "goals": summary,
                        "goal_id": goal_id,
                        "evidence_id": event_id or None,
                        "evidence_occurred_at": _utc_iso(stamp),
                    },
                )
        return AutonomousDecision(
            outcome=OUTCOME_NO_DECISION,
            reason="goal_evidence_consumed" if consumed_seen else "goal_no_fresh_evidence",
            decided_at=_utc_iso(moment),
            cooldown_until=None,
            metadata={"goals": summary},
        )
    except Exception as error:  # pragma: no cover - defensive
        logger.warning(
            "Goal evidence classification failed (no_decision): type={}",
            type(error).__name__,
        )
        return no_decision("classification_failed", moment)


__all__ = [
    "ALL_OUTCOMES",
    "AutonomousDecision",
    "build_context_signals",
    "classify_goal_evidence",
    "DEFAULT_COOLDOWN_S",
    "GOAL_EVIDENCE_KEYWORDS",
    "GOAL_EVIDENCE_MAX_AGE_H",
    "OUTCOME_GOAL_BEHAVIOR",
    "OUTCOME_IDLE_BEHAVIOR",
    "OUTCOME_NO_DECISION",
    "OUTCOME_PROACTIVE_INTERACTION",
    "OUTCOME_RELATIONSHIP_BEHAVIOR",
    "build_decision_inputs",
    "classify_proactive_decision",
    "classify_world_decision",
    "cooldown_until",
    "decision_inputs_summary",
    "goal_evidence_summary",
    "is_cooldown_active",
    "no_decision",
]
