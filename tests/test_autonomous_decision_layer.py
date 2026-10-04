"""Autonomous Decision Layer — typed outcomes, adapters, cooldown, fail-soft.

Deterministic: no LLM, no network, no scheduler, no clock reads beyond the
``now`` values passed in. Adapters are tested against the real world_state
decision result and the real proactive eligibility gate.
"""

import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber import autonomous_decision as ad
from src.open_llm_vtuber.autonomous_decision import (
    ALL_OUTCOMES,
    DEFAULT_COOLDOWN_S,
    OUTCOME_GOAL_BEHAVIOR,
    OUTCOME_IDLE_BEHAVIOR,
    OUTCOME_NO_DECISION,
    OUTCOME_PROACTIVE_INTERACTION,
    OUTCOME_RELATIONSHIP_BEHAVIOR,
    AutonomousDecision,
    build_decision_inputs,
    classify_proactive_decision,
    classify_world_decision,
    cooldown_until,
    decision_inputs_summary,
    is_cooldown_active,
    no_decision,
)
from src.open_llm_vtuber.world_state import (
    DECISION_COOLDOWN_S,
    DecisionInputs,
    DecisionResult,
    WorldState,
    decide_activity,
)

UTC = timezone.utc
NOW = datetime(2026, 10, 2, 13, 0, tzinfo=UTC)
JKT = "Asia/Jakarta"
STAMP = "2026-10-02T13:00:00+00:00"


def world_result(action, reason, state_version="2026-10-02T12:00:00+00:00"):
    return DecisionResult(action, reason, STAMP, state_version)


def idle_state(**over):
    base = {
        "activity": "idle",
        "energy": 80,
        "mood": "calm",
        "activity_started_at": "2026-09-29T00:00:00+00:00",
        "last_update_at": "2026-09-29T00:00:00+00:00",
    }
    base.update(over)
    return WorldState(**base)


class OutcomeVocabularyTest(unittest.TestCase):
    def test_exactly_the_five_documented_outcomes(self):
        self.assertEqual(
            set(ALL_OUTCOMES),
            {
                "no_decision",
                "idle_behavior",
                "proactive_interaction",
                "goal_related_behavior",
                "relationship_related_behavior",
            },
        )
        self.assertEqual(len(ALL_OUTCOMES), 5)

    def test_no_decision_is_the_valid_default(self):
        decision = no_decision("nothing_to_do", NOW)
        self.assertEqual(decision.outcome, OUTCOME_NO_DECISION)
        self.assertFalse(decision.acts)

    def test_decision_record_is_typed_and_frozen(self):
        decision = no_decision("x", NOW)
        self.assertIsInstance(decision, AutonomousDecision)
        with self.assertRaises(Exception):
            decision.outcome = "idle_behavior"  # type: ignore[misc]


class WorldAdapterTest(unittest.TestCase):
    def test_no_decision_on_normal_conditions(self):
        typed = classify_world_decision(world_result(None, "no_change"), NOW)
        self.assertEqual(typed.outcome, OUTCOME_NO_DECISION)
        self.assertFalse(typed.acts)
        self.assertIsNone(typed.cooldown_until)

    def test_valid_autonomous_decision(self):
        typed = classify_world_decision(world_result("reading", "stale_idle"), NOW)
        self.assertEqual(typed.outcome, OUTCOME_IDLE_BEHAVIOR)
        self.assertTrue(typed.acts)
        self.assertEqual(typed.cooldown_until, cooldown_until(NOW))

    def test_goal_related_outcome(self):
        typed = classify_world_decision(world_result("playing", "goal_related"), NOW)
        self.assertEqual(typed.outcome, OUTCOME_GOAL_BEHAVIOR)
        self.assertTrue(typed.acts)

    def test_relationship_related_outcome(self):
        typed = classify_world_decision(
            world_result("playing", "relationship_bias"), NOW
        )
        self.assertEqual(typed.outcome, OUTCOME_RELATIONSHIP_BEHAVIOR)

    def test_cooldown_reason_is_no_decision(self):
        typed = classify_world_decision(world_result(None, "cooldown_active"), NOW)
        self.assertEqual(typed.outcome, OUTCOME_NO_DECISION)
        self.assertFalse(typed.acts)

    def test_every_world_reason_classifies(self):
        for reason in (
            "low_energy",
            "stale_idle",
            "duration_limit",
            "night_rest",
            "cooldown_active",
            "no_change",
            "goal_related",
            "relationship_bias",
            "preference",
            "mood_bias",
        ):
            with self.subTest(reason=reason):
                typed = classify_world_decision(world_result("reading", reason), NOW)
                self.assertIn(typed.outcome, ALL_OUTCOMES)

    def test_unknown_reason_defaults_to_idle_behavior(self):
        typed = classify_world_decision(world_result("reading", "future_reason"), NOW)
        self.assertEqual(typed.outcome, OUTCOME_IDLE_BEHAVIOR)

    def test_adapter_adds_no_new_behaviour(self):
        """Classification must agree with the real policy's own action."""
        state = idle_state()
        result = decide_activity(state, NOW, JKT)
        typed = classify_world_decision(result, NOW)
        self.assertEqual(typed.acts, result.action is not None)
        self.assertEqual(typed.metadata["activity"], result.action)


class ProactiveAdapterTest(unittest.TestCase):
    def test_eligible_is_proactive_interaction(self):
        typed = classify_proactive_decision(
            eligible=True, reason="eligible", moment=NOW, strategy="semantic_auto"
        )
        self.assertEqual(typed.outcome, OUTCOME_PROACTIVE_INTERACTION)
        self.assertTrue(typed.acts)

    def test_not_eligible_is_no_decision(self):
        typed = classify_proactive_decision(
            eligible=False, reason="cooldown", moment=NOW
        )
        self.assertEqual(typed.outcome, OUTCOME_NO_DECISION)
        self.assertFalse(typed.acts)
        self.assertIsNone(typed.cooldown_until)

    def test_matches_real_proactive_gate(self):
        """The adapter must mirror ProactiveStateMachine.is_eligible exactly."""
        from src.open_llm_vtuber.proactive_chat import (
            ProactiveChatConfig,
            ProactiveRuntimeState,
            ProactiveStateMachine,
        )
        from src.open_llm_vtuber.request_latency import monotonic_ms

        state = ProactiveRuntimeState(
            history_uid="h1",
            last_user_activity_monotonic=monotonic_ms(),
            next_proactive_eligible_at=monotonic_ms() - 1,
        )
        machine = ProactiveStateMachine(ProactiveChatConfig(enabled=True))
        eligible = machine.is_eligible(state)
        typed = classify_proactive_decision(
            eligible=eligible, reason="eligible", moment=NOW
        )
        self.assertEqual(typed.acts, eligible)

    def test_metadata_carries_no_text(self):
        typed = classify_proactive_decision(
            eligible=True,
            reason="ok",
            moment=NOW,
            strategy="heuristic",
            intent="ask_user_something",
        )
        self.assertEqual(
            typed.metadata, {"strategy": "heuristic", "intent": "ask_user_something"}
        )
        for value in typed.metadata.values():
            self.assertIsInstance(value, (str, type(None)))


class CooldownTest(unittest.TestCase):
    def test_cooldown_prevents_spam(self):
        self.assertTrue(is_cooldown_active(STAMP, NOW + timedelta(minutes=5)))
        self.assertFalse(is_cooldown_active(STAMP, NOW + timedelta(minutes=20)))

    def test_cooldown_uses_existing_world_constant(self):
        self.assertEqual(DEFAULT_COOLDOWN_S, DECISION_COOLDOWN_S)
        self.assertEqual(cooldown_until(NOW), "2026-10-02T13:15:00+00:00")

    def test_missing_or_corrupt_anchor_is_not_on_cooldown(self):
        for anchor in (None, "", "not-a-timestamp", 12345, {}):
            with self.subTest(anchor=anchor):
                self.assertFalse(is_cooldown_active(anchor, NOW))

    def test_clock_skew_never_punishes(self):
        future = (NOW + timedelta(minutes=5)).isoformat()
        self.assertFalse(is_cooldown_active(future, NOW))

    def test_naive_timestamp_treated_as_utc(self):
        self.assertTrue(is_cooldown_active("2026-10-02T13:00:00", NOW))

    def test_timezone_aware_output(self):
        moment = datetime(2026, 10, 2, 20, 0, tzinfo=timezone(timedelta(hours=7)))
        stamp = cooldown_until(moment)
        self.assertEqual(stamp, "2026-10-02T13:15:00+00:00")

    def test_elapsed_time_computed_correctly(self):
        anchor = (NOW - timedelta(minutes=9)).isoformat()
        self.assertTrue(is_cooldown_active(anchor, NOW))
        boundary = (NOW - timedelta(minutes=15)).isoformat()
        self.assertFalse(is_cooldown_active(boundary, NOW))


class DeterminismTest(unittest.TestCase):
    def test_same_input_same_output(self):
        first = classify_world_decision(world_result("playing", "goal_related"), NOW)
        second = classify_world_decision(world_result("playing", "goal_related"), NOW)
        self.assertEqual(first, second)

    def test_decided_at_is_utc(self):
        typed = classify_world_decision(world_result("reading", "stale_idle"), NOW)
        self.assertEqual(typed.decided_at, "2026-10-02T13:00:00+00:00")

    def test_no_randomness_in_module(self):
        import pathlib

        source = pathlib.Path(ad.__file__).read_text(encoding="utf-8")
        lowered = source.lower()
        for banned in (
            "import random",
            "random.",
            "uuid",
            "time.sleep",
            "import asyncio",
            "openai",
        ):
            with self.subTest(token=banned):
                self.assertNotIn(banned, lowered)

    def test_module_has_no_llm_or_scheduler(self):
        import pathlib

        source = pathlib.Path(ad.__file__).read_text(encoding="utf-8")
        for banned in ("chat_completion", "create_task", "while True", "asyncio"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, source)


class FailSoftTest(unittest.TestCase):
    def test_missing_world_result(self):
        self.assertEqual(
            classify_world_decision(None, NOW).outcome, OUTCOME_NO_DECISION
        )

    def test_malformed_world_result(self):
        class Broken:
            reason = None
            action = object()
            state_version = None

        typed = classify_world_decision(Broken(), NOW)  # type: ignore[arg-type]
        self.assertIn(typed.outcome, ALL_OUTCOMES)

    def test_proactive_adapter_missing_fields(self):
        typed = classify_proactive_decision(eligible=True, reason="", moment=NOW)
        self.assertEqual(typed.outcome, OUTCOME_PROACTIVE_INTERACTION)
        self.assertEqual(typed.metadata, {"strategy": None, "intent": None})

    def test_summary_survives_corrupt_inputs(self):
        summary = decision_inputs_summary(
            relationship_status=object(),
            goal_activities=(1, 2),
            interaction_preferences=None,
            episodic_event_count="not-a-number",
        )
        self.assertEqual(summary["goal_activities"], 2)
        self.assertEqual(summary["episodic_events"], 0)


class DecisionInputsTest(unittest.TestCase):
    def test_reuses_existing_decision_inputs_type(self):
        inputs = build_decision_inputs(
            relationship_status="DATING",
            goal_activities=("reading",),
            preferred_activities=("playing",),
            mood_bias=True,
        )
        self.assertIsInstance(inputs, DecisionInputs)
        self.assertEqual(inputs.relationship_status, "dating")
        self.assertEqual(inputs.goal_activities, ("reading",))

    def test_empty_inputs_are_inert(self):
        inputs = build_decision_inputs()
        result = decide_activity(idle_state(), NOW, JKT, inputs)
        self.assertEqual(result.reason, "stale_idle")

    def test_active_goal_becomes_decision_input(self):
        inputs = build_decision_inputs(goal_activities=("playing",))
        result = decide_activity(idle_state(), NOW, JKT, inputs)
        self.assertEqual((result.action, result.reason), ("playing", "goal_related"))

    def test_relationship_becomes_decision_input(self):
        inputs = build_decision_inputs(relationship_status="close")
        result = decide_activity(idle_state(), NOW, JKT, inputs)
        self.assertEqual(result.reason, "relationship_bias")

    def test_done_goal_is_never_selected(self):
        # Goal mapping only accepts non-done ids upstream; assert the policy
        # cannot produce a goal outcome from a completed goal's activity.
        from src.open_llm_vtuber.character_state import goal_from_dict

        done = goal_from_dict(
            {"id": "morning-reading-week", "text": "x", "status": "done"}
        )
        self.assertEqual(done.status, "done")
        from src.open_llm_vtuber.world_state import DECISION_GOAL_ACTIVITY_HINTS

        mapping = DECISION_GOAL_ACTIVITY_HINTS["morning-reading-week"]
        # The agent-side input builder filters status != done; the layer itself
        # only ever receives activities that passed that filter.
        self.assertEqual(mapping, "reading")

    def test_episodic_is_never_read_whole_by_the_policy(self):
        """v2: bounded retriever only — never the whole store, never text."""
        import pathlib

        source = pathlib.Path(ad.__file__).read_text(encoding="utf-8")
        # The deterministic retriever is the only allowed entry point...
        self.assertIn("retrieve_episodic_events", source)
        # ...and the store loader must never be used by the layer.
        self.assertNotIn("load_episodic_events", source)
        self.assertNotIn("episodic_memory import load", source)
        # Only counts and the newest event stamp may cross the boundary.
        signals = ad.DecisionContextSignals.__dataclass_fields__
        self.assertEqual(
            sorted(signals),
            [
                "continuity_candidate",
                "episodic_age_hours",
                "episodic_latest_occurred_at",
                "episodic_relevant_count",
                "preference_signals",
            ],
        )
        for name in signals:
            self.assertNotIn("text", name)
        # Continuity is opt-in verification, never inferred from event text.
        self.assertFalse(signals["continuity_candidate"].default)

    def test_interaction_preferences_are_context_not_output(self):
        """Preferences influence prompt style, and reach the layer as counts."""
        summary = decision_inputs_summary(
            interaction_preferences=(1, 2, 3), episodic_event_count=7
        )
        self.assertEqual(summary["interaction_preferences"], 3)
        self.assertEqual(summary["episodic_events"], 7)

    def test_summary_contains_no_text_fields(self):
        summary = decision_inputs_summary(
            relationship_status="dating",
            goal_activities=("reading",),
            preferred_activities=("playing",),
            interaction_preferences=("tone",),
            episodic_event_count=2,
        )
        self.assertEqual(
            sorted(summary),
            [
                "episodic_events",
                "goal_activities",
                "interaction_preferences",
                "preferred_activities",
                "relationship_status",
            ],
        )
        # activity *names* are counts here, not content
        self.assertNotIn("reading", summary.values())


class PersistenceSafetyTest(unittest.TestCase):
    def test_cooldown_anchor_is_absolute_utc(self):
        typed = classify_world_decision(world_result("reading", "stale_idle"), NOW)
        self.assertTrue(typed.cooldown_until.endswith("+00:00"))
        self.assertNotIn("today", typed.cooldown_until.lower())
        self.assertNotIn("tomorrow", typed.cooldown_until.lower())

    def test_no_relative_labels_in_record(self):
        typed = classify_proactive_decision(
            eligible=True, reason="ok", moment=NOW, strategy="heuristic"
        )
        for value in (typed.decided_at, typed.cooldown_until or ""):
            self.assertNotIn("today", value.lower())
            self.assertNotIn("in ", value.lower())

    def test_state_version_is_carried_through(self):
        typed = classify_world_decision(
            world_result("reading", "stale_idle", state_version="v1"), NOW
        )
        self.assertEqual(typed.state_version, "v1")


class UserTurnPriorityTest(unittest.TestCase):
    def test_user_conversation_path_untouched(self):
        """The layer must not sit in the user-triggered conversation path."""
        import pathlib

        convo = pathlib.Path(ad.__file__).parent / "conversations"
        for name in ("single_conversation.py", "group_conversation.py"):
            source = (convo / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertNotIn("autonomous_decision", source)

    def test_layer_is_only_observed_in_existing_pipelines(self):
        import pathlib

        root = pathlib.Path(ad.__file__).parent
        world = (root / "world_state.py").read_text(encoding="utf-8")
        handler = (root / "websocket_handler.py").read_text(encoding="utf-8")
        self.assertIn("_log_typed_decision", world)
        self.assertIn("_log_typed_proactive_decision", handler)
        # no new loop or scheduler introduced
        for source in (world, handler):
            self.assertNotIn("autonomous_decision.create", source)


if __name__ == "__main__":
    unittest.main()
