"""ADL v2 — context-aware decision inputs, post-semantic-review contract.

Groups:
  * PreferenceRegressionTest  (Step 1A: preference-only must be inert)
  * CandidateSelectionTest    (Step 1B: newest relevant event cannot starve)
  * ContinuitySemanticsTest   (Step 2: user-only memory != Mili world state)
  * AntiStasisTest            (Step 3: one hold, then normal behaviour)
  * InvariantTest             (unchanged guarantees + architecture limits)

Deterministic: fixed clock, no LLM, no scheduler, no background loop.
Corrupt inputs must degrade, never mute Mili.
"""

import ast
import json
import os
import pathlib
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber import autonomous_decision as ad
from src.open_llm_vtuber.autonomous_decision import (
    DecisionContextSignals,
    build_context_signals,
    build_decision_inputs,
    classify_proactive_decision,
    classify_world_decision,
)
from src.open_llm_vtuber.episodic_memory import (
    append_episodic_event,
    load_episodic_events,
)
from src.open_llm_vtuber.world_state import (
    EPISODIC_CONTEXT_MAX_AGE_H,
    EPISODIC_CONTEXT_MAX_EVENTS,
    DecisionInputs,
    WorldState,
    decide_activity,
    reconcile,
    save_world_state,
    load_world_state,
)

UTC = timezone.utc
JKT = "Asia/Jakarta"
NOW = datetime(2026, 10, 2, 5, 0, tzinfo=UTC)  # 12:00 JKT: outside the eating window
STALE_H = 5


def idle_state(**over):
    base = {
        "activity": "idle",
        "energy": 80,
        "mood": "calm",
        "activity_started_at": (NOW - timedelta(hours=STALE_H)).isoformat(),
        "last_update_at": (NOW - timedelta(hours=STALE_H)).isoformat(),
    }
    base.update(over)
    return WorldState(**base)


def pref(category="tone", polarity="avoid", status="active"):
    stamp = "2026-10-01T00:00:00+00:00"
    return {
        "id": f"p-{category}-{polarity}",
        "category": category,
        "polarity": polarity,
        "text": "jangan terlalu galak",
        "frame": "durability",
        "created_at": stamp,
        "updated_at": stamp,
        "source": "conversation",
        "status": status,
        "superseded_by": "",
    }


def event(text, age_h, event_id="e1", moment=NOW):
    """One stored-shaped episodic event (timestamp fields only, no text leak)."""
    stamp = (moment - timedelta(hours=age_h)).isoformat()
    return {
        "id": event_id,
        "event_text": text,
        "occurred_at": stamp,
        "created_at": stamp,
        "session_uid": "s",
        "source": "conversation",
        "tz": JKT,
    }


# Text used across the continuity tests. Both share keywords with the query, so
# only the semantic verification can tell them apart.
USER_ONLY = "Gw baru baca buku Zen"
SHARED = "Tadi kita baca buku Zen bareng"
QUERY = "buku zen"


# --------------------------------------------------------------------------
# Step 1A — preference-only must be behaviourally inert
# --------------------------------------------------------------------------
class PreferenceRegressionTest(unittest.TestCase):
    def test_preference_only_context_is_not_relevant(self):
        signals = build_context_signals(interaction_preferences=[pref()], moment=NOW)
        self.assertEqual(signals.preference_signals, ("tone:avoid",))
        self.assertFalse(signals.continuity_candidate)
        self.assertFalse(signals.has_continuity(EPISODIC_CONTEXT_MAX_AGE_H))

    def test_preference_only_does_not_change_decision(self):
        baseline = decide_activity(
            idle_state(), NOW, JKT, DecisionInputs(mood_bias=True)
        )
        with_prefs = decide_activity(
            idle_state(),
            NOW,
            JKT,
            DecisionInputs(
                mood_bias=True,
                context=build_context_signals(
                    interaction_preferences=[pref()], moment=NOW
                ),
            ),
        )
        self.assertEqual(
            (baseline.action, baseline.reason), (with_prefs.action, with_prefs.reason)
        )

    def test_preference_only_does_not_flip_activity(self):
        # Regression from the semantic review. Like-for-like: identical
        # relationship status, only the presence of a stored preference
        # differs. A preference must never move Mili to another activity.
        for status in ("stranger", "close", None):
            for mood in ("calm", "content", "tired"):
                with self.subTest(status=status, mood=mood):
                    state = idle_state(mood=mood)
                    without = build_decision_inputs(
                        relationship_status=status, mood_bias=True
                    )
                    with_pref = build_decision_inputs(
                        relationship_status=status,
                        mood_bias=True,
                        interaction_preferences=[pref()],
                    )
                    self.assertIsNone(with_pref.context)
                    first = decide_activity(state, NOW, JKT, without)
                    second = decide_activity(state, NOW, JKT, with_pref)
                    self.assertEqual(
                        (first.action, first.reason), (second.action, second.reason)
                    )

    def test_preference_only_keeps_the_mood_bias_activity(self):
        # The exact case from the review: mood="content" -> playing/mood_bias,
        # reached through the agent early-return shape.
        state = idle_state(mood="content")
        early_return_shape = DecisionInputs(mood_bias=True)
        self.assertEqual(
            decide_activity(state, NOW, JKT, early_return_shape).action, "playing"
        )
        source = pathlib.Path(
            pathlib.Path(ad.__file__).parent
            / "agent"
            / "agents"
            / "basic_memory_agent.py"
        ).read_text(encoding="utf-8")
        # The guard must not be widened by context (that was the regression).
        self.assertIn("if not goal_activities and not preferred_activities:", source)
        self.assertIn("return DecisionInputs(mood_bias=True, context=context)", source)

    def test_superseded_preference_is_not_a_signal(self):
        signals = build_context_signals(
            interaction_preferences=[pref(status="superseded")], moment=NOW
        )
        self.assertEqual(signals.preference_signals, ())

    def test_preference_signal_carries_no_text(self):
        signals = build_context_signals(interaction_preferences=[pref()], moment=NOW)
        self.assertNotIn("galak", repr(signals.preference_signals))


# --------------------------------------------------------------------------
# Step 1B — newest relevant event must never starve
# --------------------------------------------------------------------------
class CandidateSelectionTest(unittest.TestCase):
    def _starved_pattern(self):
        """Newest relevant event scores lower than three older ones."""
        return [
            event("buku", 1, "new"),  # overlap 1  -> score ~1.49
            event("buku novel zen", 30, "old1"),  # overlap 2 -> score 2.00
            event("buku novel zenzt", 31, "old2"),
            event("buku novel zenwoo", 32, "old3"),
        ]

    def test_newest_relevant_event_is_not_starved(self):
        signals = build_context_signals(
            episodic_events=self._starved_pattern(),
            query="buku novel",
            moment=NOW,
        )
        self.assertEqual(
            signals.episodic_latest_occurred_at, (NOW - timedelta(hours=1)).isoformat()
        )
        self.assertEqual(signals.episodic_age_hours, 1)
        self.assertGreater(signals.episodic_age_hours, 0)

    def test_candidate_set_stays_bounded(self):
        signals = build_context_signals(
            episodic_events=self._starved_pattern() * 5,
            query="buku novel",
            moment=NOW,
        )
        self.assertLessEqual(
            signals.episodic_relevant_count, EPISODIC_CONTEXT_MAX_EVENTS
        )

    def test_candidate_count_never_exceeds_cap_even_with_many_events(self):
        events = [
            event(f"buku novel zen {chr(97 + i)}", i + 1, f"e{i}") for i in range(12)
        ]
        signals = build_context_signals(
            episodic_events=events, query="buku novel", moment=NOW
        )
        self.assertLessEqual(
            signals.episodic_relevant_count, EPISODIC_CONTEXT_MAX_EVENTS
        )

    def test_selection_is_deterministic(self):
        events = self._starved_pattern()
        first = build_context_signals(
            episodic_events=events, query="buku novel", moment=NOW
        )
        second = build_context_signals(
            episodic_events=list(events), query="buku novel", moment=NOW
        )
        self.assertEqual(first, second)

    def test_recency_pass_reads_timestamps_only(self):
        # The recency-bounded slice must not smuggle text into the policy.
        rows = ad._recency_bounded_candidates(
            [event("a", 5, "a"), event("b", 1, "b"), "not-a-dict", {}], 3
        )
        self.assertEqual([row.get("id") for row in rows], ["b", "a"])

    def test_recency_pass_skips_events_without_occurred_at(self):
        rows = ad._recency_bounded_candidates(
            [{"id": "x", "event_text": "no stamp"}, event("stamped", 2, "y")], 3
        )
        self.assertEqual([row.get("id") for row in rows], ["y"])

    def test_relevance_is_still_required(self):
        signals = build_context_signals(
            episodic_events=[event("Gw Beli buku cerita", 1, "u")],
            query="server timeout webhook",
            moment=NOW,
        )
        self.assertEqual(signals.episodic_relevant_count, 0)
        self.assertIsNone(signals.episodic_latest_occurred_at)


# --------------------------------------------------------------------------
# Step 2 — episodic continuity semantics
# --------------------------------------------------------------------------
class ContinuitySemanticsTest(unittest.TestCase):
    def _signals(self, events, **kwargs):
        return build_context_signals(
            episodic_events=events, query=QUERY, moment=NOW, **kwargs
        )

    def test_1_user_only_memory_has_no_influence(self):
        """'Tadi aku baca buku Zen.' must not change Mili's activity."""
        signals = self._signals([event(USER_ONLY, 1, "u1")])
        self.assertEqual(signals.episodic_relevant_count, 1)
        self.assertEqual(signals.episodic_age_hours, 1)
        self.assertFalse(signals.continuity_candidate)
        baseline = decide_activity(
            idle_state(), NOW, JKT, DecisionInputs(mood_bias=True)
        )
        actual = decide_activity(
            idle_state(), NOW, JKT, DecisionInputs(mood_bias=True, context=signals)
        )
        self.assertEqual(
            (baseline.action, baseline.reason), (actual.action, actual.reason)
        )
        self.assertNotEqual(actual.reason, "episodic_continuity")

    def test_2_recent_user_only_event_does_not_hold(self):
        for text in (
            "Tadi aku baca buku Zen",
            "Tadi aku makan nasi goreng",
            "Tadi aku debugging WebSocket",
            "Tadi aku nonton film",
        ):
            with self.subTest(text=text):
                signals = self._signals(
                    [event(text, 1, "u")],
                )
                self.assertFalse(signals.continuity_candidate)
                result = decide_activity(
                    idle_state(),
                    NOW,
                    JKT,
                    DecisionInputs(mood_bias=True, context=signals),
                )
                self.assertNotEqual(result.reason, "episodic_continuity")

    def test_3_verified_shared_event_can_hold(self):
        signals = self._signals([event(SHARED, 1, "s1")], continuity_event_ids=["s1"])
        self.assertTrue(signals.continuity_candidate)
        self.assertTrue(signals.has_continuity(EPISODIC_CONTEXT_MAX_AGE_H))
        result = decide_activity(
            idle_state(), NOW, JKT, DecisionInputs(context=signals)
        )
        self.assertIsNone(result.action)
        self.assertEqual(result.reason, "episodic_continuity")

    def test_3b_verification_is_scoped_to_the_bounded_candidates(self):
        events = [event(USER_ONLY, 1, "u1"), event(SHARED, 1, "s1")]
        self.assertTrue(
            self._signals(events, continuity_event_ids=["s1"]).continuity_candidate
        )
        # No verification supplied -> nothing qualifies, whatever the text says.
        self.assertFalse(self._signals(events).continuity_candidate)
        # A verified id that the bounded selection drops grants nothing: only
        # candidates actually considered can carry the continuity flag.
        crowded = [
            event("buku zen alpha", 1, "e1"),
            event("buku zen beta", 1, "e2"),
            event("buku zen gamma", 1, "e3"),
            event("buku", 1, "e9"),  # relevant, but lowest score -> capped out
        ]
        excluded = build_context_signals(
            episodic_events=crowded,
            query="buku zen",
            moment=NOW,
            continuity_event_ids=["e9"],
        )
        self.assertLessEqual(
            excluded.episodic_relevant_count, EPISODIC_CONTEXT_MAX_EVENTS
        )
        self.assertFalse(excluded.continuity_candidate)
        # The same event qualifies once it is inside the bounded set.
        included = build_context_signals(
            episodic_events=crowded[:1] + crowded[3:],
            query="buku zen",
            moment=NOW,
            continuity_event_ids=["e9"],
        )
        self.assertTrue(included.continuity_candidate)

    def test_3c_unknown_verification_id_grants_nothing(self):
        signals = self._signals([event(SHARED, 1, "s1")], continuity_event_ids=["nope"])
        self.assertFalse(signals.continuity_candidate)

    def test_4_goal_still_beats_continuity(self):
        signals = self._signals([event(SHARED, 1, "s1")], continuity_event_ids=["s1"])
        result = decide_activity(
            idle_state(),
            NOW,
            JKT,
            DecisionInputs(context=signals, goal_activities=("playing",)),
        )
        self.assertEqual((result.action, result.reason), ("playing", "goal_related"))

    def test_5_relationship_preference_mood_intact_without_continuity(self):
        state = idle_state(mood="content")
        for inputs in (
            DecisionInputs(relationship_status="close"),
            DecisionInputs(relationship_status="close", mood_bias=True),
            DecisionInputs(preferred_activities=("reading",), mood_bias=True),
            DecisionInputs(mood_bias=True),
        ):
            with self.subTest(inputs=inputs):
                result = decide_activity(state, NOW, JKT, inputs)
                self.assertIsNotNone(result.action)
                self.assertNotEqual(result.reason, "episodic_continuity")

    def test_5b_continuity_still_outranks_soft_biases(self):
        signals = self._signals([event(SHARED, 1, "s1")], continuity_event_ids=["s1"])
        result = decide_activity(
            idle_state(mood="content"),
            NOW,
            JKT,
            DecisionInputs(
                relationship_status="close", mood_bias=True, context=signals
            ),
        )
        self.assertEqual(result.reason, "episodic_continuity")

    def test_hold_never_selects_an_activity(self):
        signals = self._signals([event(SHARED, 1, "s1")], continuity_event_ids=["s1"])
        for activity in ("idle", "reading", "playing", "eating"):
            with self.subTest(activity=activity):
                state = WorldState(
                    activity=activity,
                    energy=80,
                    mood="calm",
                    activity_started_at=(NOW - timedelta(hours=STALE_H)).isoformat(),
                    last_update_at=(NOW - timedelta(hours=STALE_H)).isoformat(),
                )
                result = decide_activity(
                    state, NOW, JKT, DecisionInputs(context=signals)
                )
                if result.reason == "episodic_continuity":
                    self.assertIsNone(result.action)
                else:
                    # A busy activity is released by the pre-existing duration
                    # limit, which runs before the stale-idle stage.
                    self.assertEqual(result.reason, "duration_limit")
                    self.assertEqual(result.action, "idle")

    def test_continuity_never_introduces_a_new_activity(self):
        signals = self._signals([event(SHARED, 1, "s1")], continuity_event_ids=["s1"])
        for activity in ("idle", "reading", "playing", "eating"):
            with self.subTest(activity=activity):
                state = WorldState(
                    activity=activity,
                    energy=80,
                    mood="calm",
                    activity_started_at=(NOW - timedelta(hours=STALE_H)).isoformat(),
                    last_update_at=(NOW - timedelta(hours=STALE_H)).isoformat(),
                )
                after, _ = reconcile(
                    state, now=NOW, tz=JKT, inputs=DecisionInputs(context=signals)
                )
                self.assertIn(after.activity, (activity, "idle"))

    def test_agent_producer_suppresses_unverified_context(self):
        # The agent has no way to verify involvement, so it must never hand an
        # unverified memory to the decision as continuity.
        source = pathlib.Path(
            pathlib.Path(ad.__file__).parent
            / "agent"
            / "agents"
            / "basic_memory_agent.py"
        ).read_text(encoding="utf-8")
        self.assertIn("continuity_event_ids=()", source)
        self.assertIn("if not signals.continuity_candidate:", source)


# --------------------------------------------------------------------------
# Step 3 — anti-stasis
# --------------------------------------------------------------------------
class AntiStasisTest(unittest.TestCase):
    TIMELINE_MIN = (0, 30, 60, 90, 120, 180, 360, 720, 1080, 1440)

    def _timeline(self, continuity_ids=("s1",), query=QUERY):
        """Replay ONE fixed memory across the window.

        The event keeps its original occurred_at; only the evaluation clock
        moves. Re-deriving signals per step is correct (age changes with now)
        and must not be mistaken for a new experience.
        """
        rows = []
        state = idle_state()
        fixed = [event(SHARED, 1, "s1", moment=NOW)]
        for minutes in self.TIMELINE_MIN:
            moment = NOW + timedelta(minutes=minutes)
            signals = build_context_signals(
                episodic_events=fixed,
                query=query,
                moment=moment,
                continuity_event_ids=continuity_ids,
            )
            inputs = DecisionInputs(
                relationship_status="close", mood_bias=True, context=signals
            )
            decision = decide_activity(state, moment, JKT, inputs)
            state, _ = reconcile(state, now=moment, tz=JKT, inputs=inputs)
            rows.append((minutes, decision, state))
        return rows

    def test_same_hold_is_not_repeated_without_bound(self):
        reasons = [decision.reason for _, decision, _ in self._timeline()]
        holds = [r for r in reasons if r == "episodic_continuity"]
        self.assertEqual(len(holds), 1, f"hold repeated: {reasons}")
        self.assertEqual(reasons[0], "episodic_continuity")

    def test_mili_is_not_frozen_for_the_whole_window(self):
        actions = [decision.action for _, decision, _ in self._timeline()]
        self.assertIsNone(actions[0])
        self.assertTrue(
            any(action is not None for action in actions[1:]),
            f"never resumed normal activity: {actions}",
        )

    def test_hold_sets_the_decision_anchor(self):
        _, _, after_hold = self._timeline()[0]
        self.assertEqual(
            after_hold.last_autonomous_decision_at, NOW.isoformat(timespec="seconds")
        )

    def test_cooldown_engages_right_after_a_hold(self):
        _, _, state = self._timeline()[0]
        moment = NOW + timedelta(minutes=5)
        signals = build_context_signals(
            episodic_events=[event(SHARED, 1, "s1", moment=moment)],
            query=QUERY,
            moment=moment,
            continuity_event_ids=["s1"],
        )
        result = decide_activity(state, moment, JKT, DecisionInputs(context=signals))
        self.assertEqual(result.reason, "cooldown_active")

    def test_a_genuinely_new_experience_can_hold_again(self):
        state = idle_state()
        first = build_context_signals(
            episodic_events=[event(SHARED, 1, "s1")],
            query=QUERY,
            moment=NOW,
            continuity_event_ids=["s1"],
        )
        state, _ = reconcile(
            state, now=NOW, tz=JKT, inputs=DecisionInputs(context=first)
        )
        later = NOW + timedelta(hours=2)
        second = build_context_signals(
            episodic_events=[
                event(SHARED, 1, "s1"),
                event(SHARED, 0, "s2", moment=later),
            ],
            query=QUERY,
            moment=later,
            continuity_event_ids=["s1", "s2"],
        )
        result = decide_activity(state, later, JKT, DecisionInputs(context=second))
        self.assertEqual(result.reason, "episodic_continuity")

    def test_anchor_survives_restart_and_decisions_return_to_normal(self):
        state = idle_state()
        signals = build_context_signals(
            episodic_events=[event(SHARED, 1, "s1")],
            query=QUERY,
            moment=NOW,
            continuity_event_ids=["s1"],
        )
        held, _ = reconcile(
            state, now=NOW, tz=JKT, inputs=DecisionInputs(context=signals)
        )
        self.assertEqual(held.activity, "idle")
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                self.assertTrue(save_world_state("c", held))
                loaded = load_world_state("c")
                self.assertEqual(loaded.activity, "idle")
                self.assertEqual(
                    loaded.last_autonomous_decision_at,
                    NOW.isoformat(timespec="seconds"),
                )
                # 24h later the anchor is long expired and normal decisions work.
                much_later = NOW + timedelta(hours=24)
                resumed, _ = reconcile(
                    loaded,
                    now=much_later,
                    tz=JKT,
                    inputs=DecisionInputs(relationship_status="close", mood_bias=True),
                )
                self.assertIsNotNone(resumed.last_autonomous_decision_at)
            finally:
                os.chdir(previous)


# --------------------------------------------------------------------------
# Unchanged guarantees
# --------------------------------------------------------------------------
class InvariantTest(unittest.TestCase):
    def test_24h_contract_unchanged(self):
        inside = build_context_signals(
            episodic_events=[event(SHARED, 23, "s1")],
            query=QUERY,
            moment=NOW,
            continuity_event_ids=["s1"],
        )
        self.assertEqual(inside.episodic_age_hours, 23)  # exactly 23h
        self.assertTrue(inside.has_continuity(EPISODIC_CONTEXT_MAX_AGE_H))
        almost = build_context_signals(
            episodic_events=[event(SHARED, 23.9, "s1")],
            query=QUERY,
            moment=NOW,
            continuity_event_ids=["s1"],
        )
        self.assertEqual(almost.episodic_age_hours, 24)  # 23h54m rounds up
        self.assertTrue(almost.has_continuity(EPISODIC_CONTEXT_MAX_AGE_H))
        outside = build_context_signals(
            episodic_events=[event(SHARED, 24.2, "s1")],
            query=QUERY,
            moment=NOW,
            continuity_event_ids=["s1"],
        )
        self.assertEqual(outside.episodic_age_hours, 25)
        self.assertFalse(outside.has_continuity(EPISODIC_CONTEXT_MAX_AGE_H))
        self.assertNotEqual(
            decide_activity(
                idle_state(), NOW, JKT, DecisionInputs(context=outside)
            ).reason,
            "episodic_continuity",
        )

    def test_exact_24h_is_eligible(self):
        signals = build_context_signals(
            episodic_events=[event(SHARED, 24, "s1", moment=NOW + timedelta(hours=24))],
            query=QUERY,
            moment=NOW + timedelta(hours=24),
            continuity_event_ids=["s1"],
        )
        self.assertEqual(signals.episodic_age_hours, 24)
        self.assertTrue(signals.has_continuity(EPISODIC_CONTEXT_MAX_AGE_H))

    def test_future_timestamp_is_clamped_not_negative(self):
        signals = build_context_signals(
            episodic_events=[event(SHARED, -4, "s1")],
            query=QUERY,
            moment=NOW,
            continuity_event_ids=["s1"],
        )
        self.assertEqual(signals.episodic_age_hours, 0)

    def test_non_utc_occurred_at_is_normalised(self):
        raw = event(SHARED, 1, "s1")
        raw["occurred_at"] = "2026-10-02T12:00:00+07:00"  # == 05:00Z == NOW
        signals = build_context_signals(
            episodic_events=[raw], query=QUERY, moment=NOW, continuity_event_ids=["s1"]
        )
        self.assertEqual(
            signals.episodic_latest_occurred_at, "2026-10-02T05:00:00+00:00"
        )
        self.assertEqual(signals.episodic_age_hours, 0)

    def test_invalid_occurred_at_never_falls_back_to_created_at(self):
        raw = event(SHARED, 1, "s1")
        raw["occurred_at"] = "not-a-time"
        signals = build_context_signals(
            episodic_events=[raw], query=QUERY, moment=NOW, continuity_event_ids=["s1"]
        )
        self.assertEqual(signals.episodic_relevant_count, 0)
        self.assertFalse(signals.continuity_candidate)

    def test_missing_occurred_at_is_dropped(self):
        raw = {"id": "x", "event_text": SHARED, "created_at": NOW.isoformat()}
        signals = build_context_signals(
            episodic_events=[raw], query=QUERY, moment=NOW, continuity_event_ids=["x"]
        )
        self.assertEqual(signals.episodic_relevant_count, 0)

    def test_corrupt_inputs_fail_soft(self):
        for kwargs in (
            {"interaction_preferences": [{"nonsense": True}, "x", None, 42]},
            {"episodic_events": ["not-a-dict", 7]},
            {"episodic_events": object(), "interaction_preferences": object()},
            {"query": object()},
        ):
            with self.subTest(kwargs=kwargs):
                signals = build_context_signals(moment=NOW, **kwargs)
                self.assertIsInstance(signals, DecisionContextSignals)
                result = decide_activity(
                    idle_state(),
                    NOW,
                    JKT,
                    DecisionInputs(mood_bias=True, context=signals),
                )
                self.assertIsNotNone(result.action)
                self.assertNotEqual(result.reason, "episodic_continuity")

    def test_corrupt_episodic_store_does_not_mute_mili(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                os.makedirs("episodic", exist_ok=True)
                with open(
                    os.path.join("episodic", "ctx.json"), "w", encoding="utf-8"
                ) as handle:
                    handle.write("{broken json")
                signals = build_context_signals(
                    episodic_events=load_episodic_events("ctx"),
                    query=QUERY,
                    moment=NOW,
                )
                self.assertEqual(signals.episodic_relevant_count, 0)
                self.assertFalse(signals.continuity_candidate)
                result = decide_activity(
                    idle_state(),
                    NOW,
                    JKT,
                    DecisionInputs(mood_bias=True, context=signals),
                )
                self.assertEqual(result.reason, "stale_idle")
            finally:
                os.chdir(previous)

    def test_missing_context_object_is_pre_v2(self):
        baseline = decide_activity(
            idle_state(), NOW, JKT, DecisionInputs(mood_bias=True)
        )
        explicit = decide_activity(
            idle_state(), NOW, JKT, DecisionInputs(mood_bias=True, context=None)
        )
        self.assertEqual(
            (baseline.action, baseline.reason), (explicit.action, explicit.reason)
        )

    def test_same_input_same_decision(self):
        signals = build_context_signals(
            episodic_events=[event(SHARED, 1, "s1")],
            query=QUERY,
            moment=NOW,
            continuity_event_ids=["s1"],
        )
        inputs = DecisionInputs(context=signals)
        first = decide_activity(idle_state(), NOW, JKT, inputs)
        second = decide_activity(idle_state(), NOW, JKT, inputs)
        self.assertEqual(first, second)

    def test_signals_rebuilt_identically_after_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                append_episodic_event(
                    "ctx",
                    {
                        "event_text": SHARED,
                        "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                        "session_uid": "s",
                        "source": "conversation",
                        "tz": JKT,
                    },
                )
                stored = load_episodic_events("ctx")
                event_id = stored[0]["id"]
                first = build_context_signals(
                    episodic_events=stored,
                    query=QUERY,
                    moment=NOW,
                    continuity_event_ids=[event_id],
                )
                second = build_context_signals(
                    episodic_events=load_episodic_events("ctx"),
                    query=QUERY,
                    moment=NOW,
                    continuity_event_ids=[event_id],
                )
                self.assertEqual(first, second)
                self.assertEqual(
                    first.episodic_latest_occurred_at,
                    (NOW - timedelta(hours=1)).isoformat(timespec="seconds"),
                )
            finally:
                os.chdir(previous)

    def test_no_context_text_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                append_episodic_event(
                    "ctx",
                    {
                        "event_text": SHARED,
                        "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                        "session_uid": "s",
                        "source": "conversation",
                        "tz": JKT,
                    },
                )
                stored = load_episodic_events("ctx")
                signals = build_context_signals(
                    episodic_events=stored,
                    query=QUERY,
                    moment=NOW,
                    continuity_event_ids=[stored[0]["id"]],
                )
                held, _ = reconcile(
                    idle_state(),
                    now=NOW,
                    tz=JKT,
                    inputs=DecisionInputs(context=signals),
                )
                self.assertTrue(save_world_state("ctx", held))
                from src.open_llm_vtuber.world_state import get_world_state_path

                raw = open(get_world_state_path("ctx"), encoding="utf-8").read()
                self.assertNotIn("Zen", raw)
                self.assertNotIn("buku", raw)
                self.assertNotIn(QUERY, raw)
            finally:
                os.chdir(previous)

    def test_signals_only_carry_counts_and_timestamps(self):
        signals = build_context_signals(
            interaction_preferences=[pref()],
            episodic_events=[event(SHARED, 1, "s1")],
            query=QUERY,
            moment=NOW,
            continuity_event_ids=["s1"],
        )
        self.assertNotIn("Zen", repr(signals))
        self.assertNotIn("galak", repr(signals))

    def test_signals_field_contract(self):
        self.assertEqual(
            sorted(DecisionContextSignals.__dataclass_fields__),
            [
                "continuity_candidate",
                "episodic_age_hours",
                "episodic_latest_occurred_at",
                "episodic_relevant_count",
                "preference_signals",
            ],
        )

    def test_episodic_store_loader_is_not_used_by_the_layer(self):
        source = pathlib.Path(ad.__file__).read_text(encoding="utf-8")
        self.assertIn("retrieve_episodic_events", source)
        self.assertNotIn("load_episodic_events", source)

    def test_no_llm_call_in_signal_path(self):
        source = pathlib.Path(ad.__file__).read_text(encoding="utf-8")
        self.assertNotIn("chat_completion", source)
        self.assertNotIn("openai", source)

    def test_no_new_scheduler_loop_or_thread(self):
        world_path = pathlib.Path(ad.__file__).parent / "world_state.py"
        world = world_path.read_text(encoding="utf-8")
        # NOTE: world_state legitimately imports threading for its per-file
        # state lock. What must not exist is scheduler/worker machinery.
        for banned in (
            "asyncio",
            "Timer(",
            "create_task",
            "while True",
            "threading.Thread",
            "ThreadPoolExecutor",
            "APScheduler",
        ):
            self.assertNotIn(banned, world)
        layer = pathlib.Path(ad.__file__).read_text(encoding="utf-8")
        for banned in ("asyncio", "create_task", "chat_completion", "openai"):
            self.assertNotIn(banned, layer)
        tree = ast.parse(world)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                target = getattr(node, "module", "") or ""
                names = [alias.name for alias in node.names]
                self.assertNotIn("autonomous_decision", target)
                self.assertFalse(any("autonomous_decision" in n for n in names))
        self.assertNotIn("build_context_signals(", world)

    def test_user_turn_path_untouched(self):
        convo = (
            pathlib.Path(ad.__file__).parent
            / "conversations"
            / "single_conversation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("DecisionContextSignals", convo)
        self.assertNotIn("continuity_candidate", convo)
        self.assertNotIn("build_context_signals", convo)

    def test_proactive_gate_still_owned_by_state_machine(self):
        from src.open_llm_vtuber.proactive_chat import (
            ProactiveChatConfig,
            ProactiveRuntimeState,
            ProactiveStateMachine,
        )

        machine = ProactiveStateMachine(ProactiveChatConfig(enabled=True))
        state = ProactiveRuntimeState(
            history_uid="h",
            last_user_activity_monotonic=time.monotonic(),
            next_proactive_eligible_at=time.monotonic() - 1,
        )
        self.assertTrue(machine.is_eligible(state))
        self.assertEqual(
            classify_proactive_decision(eligible=True, reason="", moment=NOW).outcome,
            "proactive_interaction",
        )
        state.next_proactive_eligible_at = time.monotonic() + 900
        self.assertFalse(machine.is_eligible(state))
        self.assertEqual(
            classify_proactive_decision(eligible=False, reason="", moment=NOW).outcome,
            "no_decision",
        )

    def test_typed_records_still_classify_holds(self):
        signals = build_context_signals(
            episodic_events=[event(SHARED, 1, "s1")],
            query=QUERY,
            moment=NOW,
            continuity_event_ids=["s1"],
        )
        decision = decide_activity(
            idle_state(), NOW, JKT, DecisionInputs(context=signals)
        )
        typed = classify_world_decision(decision, NOW)
        self.assertEqual(typed.reason, "episodic_continuity")
        self.assertFalse(typed.acts)

    def test_json_roundtrip_of_stored_events_is_still_supported(self):
        # Guards the loader against the shape build_context_signals consumes.
        raw = event(SHARED, 1, "s1")
        self.assertEqual(json.loads(json.dumps(raw))["occurred_at"], raw["occurred_at"])


if __name__ == "__main__":
    unittest.main()
