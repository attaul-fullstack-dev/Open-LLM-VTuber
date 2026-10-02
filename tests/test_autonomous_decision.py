"""Autonomous Decision Layer v1 — deterministic tests (fake clock, no I/O outside tmp).

No LLM calls, no scheduler, no background loops. Covers A-N.
"""

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber import world_state as ws_mod
from src.open_llm_vtuber.world_state import (
    DecisionInputs,
    WorldState,
    decide_activity,
    load_and_reconcile_world_state,
    reconcile,
    save_world_state,
)

JKT = "Asia/Jakarta"
UTC = timezone.utc
# Monday Sep 29 2026 02:00Z == 09:00 Jakarta (morning, outside eating window).
MON_09_JKT = datetime(2026, 9, 29, 2, 0, tzinfo=UTC)
# Same day 16:00Z == 23:00 Jakarta (night).
MON_23_JKT = datetime(2026, 9, 29, 16, 0, tzinfo=UTC)
# 05:00Z == 12:00 Jakarta (afternoon, outside eating window).
NOON_JKT = datetime(2026, 9, 29, 5, 0, tzinfo=UTC)
# 01:00Z == 08:00 Jakarta (inside the eating window).
EIGHT_JKT = datetime(2026, 9, 29, 1, 0, tzinfo=UTC)


def history_to(*targets):
    """Minimal recent-activity history ending with the given targets."""
    return [
        {
            "from": "idle",
            "to": target,
            "at": "2026-09-29T00:30:00+00:00",
            "location": "room",
        }
        for target in targets
    ]


def idle_state(**over):
    base = {
        "activity": "idle",
        "energy": 80,
        "activity_started_at": "2026-09-29T00:00:00+00:00",
        "last_update_at": "2026-09-29T00:00:00+00:00",
    }
    base.update(over)
    return WorldState(**base)


class DecisionRulesTest(unittest.TestCase):
    def test_a_low_energy_idle_selects_resting(self):
        res = decide_activity(idle_state(energy=28), MON_09_JKT, JKT)
        self.assertEqual(res.action, "resting")
        self.assertEqual(res.reason, "low_energy")

    def test_b_morning_stale_idle_selects_active(self):
        res = decide_activity(idle_state(), MON_09_JKT, JKT)
        self.assertIn(res.action, ("reading", "eating", "playing"))
        self.assertEqual(res.reason, "stale_idle")

    def test_c_sleeping_holds(self):
        state = idle_state(activity="sleeping", energy=100)
        res = decide_activity(state, MON_23_JKT, JKT)
        self.assertIsNone(res.action)
        self.assertEqual(res.reason, "no_change")

    def test_d_fresh_activity_holds(self):
        state = idle_state(
            activity="reading",
            activity_started_at="2026-09-29T01:50:00+00:00",
        )
        res = decide_activity(state, MON_09_JKT, JKT)
        self.assertIsNone(res.action)

    def test_d_fresh_idle_holds(self):
        state = idle_state(activity_started_at="2026-09-29T01:50:00+00:00")
        res = decide_activity(state, MON_09_JKT, JKT)
        self.assertIsNone(res.action)

    def test_e_recent_activity_not_repeated(self):
        eight_jkt = datetime(2026, 9, 29, 1, 0, tzinfo=UTC)  # 08:00 Jakarta
        history = [
            {
                "from": "idle",
                "to": "reading",
                "at": "2026-09-29T00:30:00+00:00",
                "location": "room",
            }
        ]
        state = idle_state(recent_activity_history=history)
        res = decide_activity(state, eight_jkt, JKT)
        # 08:00 Jakarta is inside the eating window and eating is not recent.
        self.assertEqual(res.action, "eating")
        state2 = idle_state(
            recent_activity_history=[
                {
                    "from": "idle",
                    "to": "eating",
                    "at": "2026-09-29T00:30:00+00:00",
                    "location": "kitchen",
                }
            ]
        )
        res2 = decide_activity(state2, eight_jkt, JKT)
        # Eating was just done: do not repeat it immediately.
        self.assertEqual(res2.action, "reading")

    def test_e_alternation_outside_eat_window(self):
        noon = datetime(2026, 9, 29, 5, 0, tzinfo=UTC)  # 12:00 Jakarta
        history = [
            {
                "from": "idle",
                "to": "reading",
                "at": "2026-09-29T03:00:00+00:00",
                "location": "room",
            }
        ]
        res = decide_activity(idle_state(recent_activity_history=history), noon, JKT)
        self.assertEqual(res.action, "playing")

    def test_f_morning_vs_night_differ(self):
        morning = decide_activity(idle_state(), MON_09_JKT, JKT)
        night = decide_activity(idle_state(energy=60), MON_23_JKT, JKT)
        self.assertIsNotNone(morning.action)
        self.assertIsNone(night.action)
        night_low = decide_activity(idle_state(energy=40), MON_23_JKT, JKT)
        self.assertEqual(night_low.action, "sleeping")
        self.assertEqual(night_low.reason, "night_rest")

    def test_duration_limit_mirror(self):
        state = idle_state(
            activity="reading",
            activity_started_at="2026-09-28T20:00:00+00:00",
        )
        res = decide_activity(state, MON_09_JKT, JKT)
        self.assertEqual(res.action, "idle")
        self.assertEqual(res.reason, "duration_limit")

    def test_result_carries_timestamp_and_version(self):
        res = decide_activity(idle_state(), MON_09_JKT, JKT)
        self.assertEqual(res.decided_at, "2026-09-29T02:00:00+00:00")
        self.assertEqual(res.state_version, "2026-09-29T00:00:00+00:00")

    def test_legacy_state_without_new_field(self):
        # Old persisted files have activity_started_at but no decision stamp:
        # they stay fully eligible (fail-soft default, never a crash).
        state = WorldState(
            activity="idle",
            energy=80,
            activity_started_at="2026-09-29T00:00:00+00:00",
            last_update_at="2026-09-29T00:00:00+00:00",
        )
        state.last_autonomous_decision_at = None
        res = decide_activity(state, MON_09_JKT, JKT)
        self.assertIsNotNone(res.action)


class ReconcileIntegrationTest(unittest.TestCase):
    def test_g_autonomous_transition_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = idle_state()
            self.assertTrue(save_world_state("mili", seed, tmp))
            revived = load_and_reconcile_world_state(
                "mili", now=MON_09_JKT, base_dir=tmp, tz=JKT
            )
            self.assertIn(revived.activity, ("reading", "eating", "playing"))
            self.assertIsNotNone(revived.last_autonomous_decision_at)
            self.assertEqual(revived.recent_activity_history[-1].get("by"), "decision")
            # Reload from disk: the pick survived.
            again = load_and_reconcile_world_state(
                "mili", now=MON_09_JKT, base_dir=tmp, tz=JKT
            )
            self.assertEqual(again.activity, revived.activity)

    def test_h_restart_45h_reconciles_from_persisted_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            thu = datetime(2026, 10, 1, 5, 0, tzinfo=UTC)  # Thu 12:00 JKT
            seed = idle_state(
                activity_started_at="2026-09-29T08:00:00+00:00",
                last_update_at="2026-09-29T08:00:00+00:00",
            )
            self.assertTrue(save_world_state("mili", seed, tmp))
            del seed
            revived = load_and_reconcile_world_state(
                "mili", now=thu, base_dir=tmp, tz=JKT
            )
            self.assertEqual(revived.last_update_at, thu.isoformat(timespec="seconds"))
            # 45h idle decayed energy; decision layer consumed reconciled state.
            self.assertLess(revived.energy, 80)

    def test_i_forced_exception_keeps_state_intact(self):
        original = ws_mod.decide_activity
        ws_mod.decide_activity = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        try:
            state = idle_state()
            out, _ = reconcile(state, MON_09_JKT, JKT, decide=True)
            self.assertEqual(out.activity, "idle")
            self.assertEqual(out.energy, 78)  # 2h idle drift applied
        finally:
            ws_mod.decide_activity = original

    def test_l_cooldown_blocks_repeat(self):
        five_min_ago = (MON_09_JKT - timedelta(minutes=5)).isoformat(timespec="seconds")
        state = idle_state(last_autonomous_decision_at=five_min_ago)
        res = decide_activity(state, MON_09_JKT, JKT)
        self.assertIsNone(res.action)
        self.assertEqual(res.reason, "cooldown_active")
        # After the window, the same state becomes eligible again.
        old = (MON_09_JKT - timedelta(minutes=16)).isoformat(timespec="seconds")
        res2 = decide_activity(
            idle_state(last_autonomous_decision_at=old), MON_09_JKT, JKT
        )
        self.assertIsNotNone(res2.action)

    def test_l_second_reconcile_same_moment_is_quiet(self):
        state = idle_state()
        first, changed1 = reconcile(state, MON_09_JKT, JKT, decide=True)
        self.assertTrue(changed1)
        second, changed2 = reconcile(first, MON_09_JKT, JKT, decide=True)
        self.assertFalse(changed2)
        self.assertEqual(
            len(second.recent_activity_history),
            len(first.recent_activity_history),
        )

    def test_m_old_json_without_new_field_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mili.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": 1,
                        "location": "room",
                        "activity": "idle",
                        "energy": 80,
                        "mood": "calm",
                        "time_context": "morning",
                        "activity_started_at": "2026-09-29T00:00:00+00:00",
                        "last_update_at": "2026-09-29T00:00:00+00:00",
                        "recent_activity_history": [],
                        "mood_ttl_turns": 0,
                        "mood_set_at": None,
                    },
                    f,
                )
            revived = load_and_reconcile_world_state(
                "mili", now=MON_09_JKT, base_dir=tmp, tz=JKT
            )
            self.assertIsNotNone(revived.last_autonomous_decision_at)
            self.assertIn(revived.activity, ("reading", "eating", "playing"))

    def test_n_widget_read_only_writes_nothing_autonomous(self):
        # decide=False still reconciles wall-clock time (existing behavior:
        # energy/mood drift persists) but must never write an autonomous
        # decision: no decision stamp, no by="decision" history, and the
        # activity itself is left for the existing transition rules only.
        with tempfile.TemporaryDirectory() as tmp:
            seed = idle_state()
            self.assertTrue(save_world_state("mili", seed, tmp))
            revived = load_and_reconcile_world_state(
                "mili", now=MON_09_JKT, base_dir=tmp, tz=JKT, decide=False
            )
            self.assertIsNone(revived.last_autonomous_decision_at)
            hist = revived.recent_activity_history
            self.assertTrue(all(h.get("by") != "decision" for h in hist))
            self.assertEqual(revived.activity, "idle")
            on_disk = json.load(open(os.path.join(tmp, "mili.json"), encoding="utf-8"))
            self.assertIsNone(on_disk.get("last_autonomous_decision_at"))


class NoSidecarTest(unittest.TestCase):
    def test_j_no_llm_call(self):
        for fn in (decide_activity, reconcile):
            self.assertFalse(asyncio.iscoroutinefunction(fn))

    def test_k_no_scheduler_or_loop(self):
        import inspect
        import re

        src = inspect.getsource(ws_mod.decide_activity)
        src += inspect.getsource(ws_mod._apply_decision)
        # Whole-word match so "sleeping" (an activity) never trips "sleep".
        for token in (
            "asyncio",
            "create_task",
            "Timer",
            "Scheduler",
            "threading",
            "chat_completion",
            "sleep",
            "generate",
        ):
            self.assertIsNone(re.search(r"\b" + re.escape(token) + r"\b", src), token)


class InfluenceStageTest(unittest.TestCase):
    """Goal / relationship / preference / mood may change the pick."""

    def test_baseline_without_inputs_is_unchanged(self):
        res = decide_activity(idle_state(), NOON_JKT, JKT)
        self.assertEqual(res.action, "reading")
        self.assertEqual(res.reason, "stale_idle")
        empty = decide_activity(idle_state(), NOON_JKT, JKT, DecisionInputs())
        self.assertEqual((empty.action, empty.reason), ("reading", "stale_idle"))

    def test_goal_selects_mapped_activity(self):
        # Default alternation would pick reading here; the goal moves it.
        res = decide_activity(
            idle_state(),
            NOON_JKT,
            JKT,
            DecisionInputs(goal_activities=("playing",)),
        )
        self.assertEqual(res.action, "playing")
        self.assertEqual(res.reason, "goal_related")

    def test_goal_eating_requires_eating_window(self):
        outside = decide_activity(
            idle_state(), NOON_JKT, JKT, DecisionInputs(goal_activities=("eating",))
        )
        self.assertNotEqual(outside.action, "eating")
        self.assertEqual(outside.reason, "stale_idle")
        inside = decide_activity(
            idle_state(), EIGHT_JKT, JKT, DecisionInputs(goal_activities=("eating",))
        )
        self.assertEqual(inside.action, "eating")
        self.assertEqual(inside.reason, "goal_related")

    def test_relationship_bias_selects_class(self):
        close = decide_activity(
            idle_state(), NOON_JKT, JKT, DecisionInputs(relationship_status="close")
        )
        self.assertEqual((close.action, close.reason), ("playing", "relationship_bias"))
        stranger = decide_activity(
            idle_state(),
            NOON_JKT,
            JKT,
            DecisionInputs(relationship_status="stranger"),
        )
        self.assertEqual(
            (stranger.action, stranger.reason), ("reading", "relationship_bias")
        )

    def test_unknown_relationship_is_no_signal(self):
        res = decide_activity(
            idle_state(), NOON_JKT, JKT, DecisionInputs(relationship_status="???")
        )
        self.assertEqual((res.action, res.reason), ("reading", "stale_idle"))

    def test_established_preference_selects_activity(self):
        state = idle_state(recent_activity_history=history_to("reading"))
        res = decide_activity(
            state,
            NOON_JKT,
            JKT,
            DecisionInputs(preferred_activities=("playing",)),
        )
        self.assertEqual((res.action, res.reason), ("playing", "preference"))

    def test_preference_to_repeat_is_filtered(self):
        state = idle_state(recent_activity_history=history_to("reading"))
        res = decide_activity(
            state,
            NOON_JKT,
            JKT,
            DecisionInputs(preferred_activities=("reading",)),
        )
        self.assertEqual(res.action, "playing")
        self.assertEqual(res.reason, "stale_idle")

    def test_mood_bias_is_the_weakest_tiebreak(self):
        content = decide_activity(
            idle_state(mood="content"), NOON_JKT, JKT, DecisionInputs(mood_bias=True)
        )
        self.assertEqual((content.action, content.reason), ("playing", "mood_bias"))
        tired = decide_activity(
            idle_state(mood="tired"), NOON_JKT, JKT, DecisionInputs(mood_bias=True)
        )
        self.assertEqual((tired.action, tired.reason), ("reading", "mood_bias"))
        calm = decide_activity(
            idle_state(mood="calm"), NOON_JKT, JKT, DecisionInputs(mood_bias=True)
        )
        self.assertEqual(calm.reason, "stale_idle")

    def test_invalid_activity_hint_is_ignored(self):
        res = decide_activity(
            idle_state(), NOON_JKT, JKT, DecisionInputs(goal_activities=("dancing",))
        )
        self.assertEqual((res.action, res.reason), ("reading", "stale_idle"))


class InfluencePrecedenceTest(unittest.TestCase):
    """Documented precedence: goal > relationship > preference > mood."""

    def test_energy_guard_beats_every_factor(self):
        state = idle_state(energy=28)
        res = decide_activity(
            state,
            NOON_JKT,
            JKT,
            DecisionInputs(
                relationship_status="close",
                goal_activities=("reading",),
                preferred_activities=("playing",),
            ),
        )
        self.assertEqual((res.action, res.reason), ("resting", "low_energy"))

    def test_cooldown_beats_every_factor(self):
        state = idle_state(last_autonomous_decision_at="2026-09-29T04:55:00+00:00")
        res = decide_activity(
            state,
            NOON_JKT,
            JKT,
            DecisionInputs(
                relationship_status="close",
                goal_activities=("reading",),
                preferred_activities=("playing",),
            ),
        )
        self.assertIsNone(res.action)
        self.assertEqual(res.reason, "cooldown_active")

    def test_night_beats_every_factor(self):
        res = decide_activity(
            idle_state(energy=60),
            MON_23_JKT,
            JKT,
            DecisionInputs(relationship_status="close", goal_activities=("reading",)),
        )
        self.assertIsNone(res.action)
        self.assertEqual(res.reason, "no_change")

    def test_recent_activity_beats_goal(self):
        state = idle_state(recent_activity_history=history_to("reading"))
        res = decide_activity(
            state, NOON_JKT, JKT, DecisionInputs(goal_activities=("reading",))
        )
        self.assertEqual((res.action, res.reason), ("playing", "stale_idle"))

    def test_goal_beats_relationship(self):
        # Both factors point at "playing"; the reason proves which one won.
        res = decide_activity(
            idle_state(),
            NOON_JKT,
            JKT,
            DecisionInputs(relationship_status="close", goal_activities=("playing",)),
        )
        self.assertEqual((res.action, res.reason), ("playing", "goal_related"))

    def test_relationship_beats_preference_and_mood(self):
        res = decide_activity(
            idle_state(mood="tired"),
            NOON_JKT,
            JKT,
            DecisionInputs(
                relationship_status="close", preferred_activities=("reading",)
            ),
        )
        self.assertEqual((res.action, res.reason), ("playing", "relationship_bias"))

    def test_preference_beats_mood(self):
        res = decide_activity(
            idle_state(mood="tired"),
            NOON_JKT,
            JKT,
            DecisionInputs(preferred_activities=("playing",), mood_bias=True),
        )
        self.assertEqual((res.action, res.reason), ("playing", "preference"))


class InfluenceIsolationTest(unittest.TestCase):
    """No new dependency, no lifecycle mutation, no I/O from the influence stage."""

    def test_inputs_are_frozen(self):
        inputs = DecisionInputs(relationship_status="close")
        with self.assertRaises(Exception):
            inputs.relationship_status = "dating"  # type: ignore[misc]

    def test_world_state_does_not_import_goal_or_relationship_owners(self):
        source = (ws_mod.__file__ or "").endswith("world_state.py")
        self.assertTrue(source)
        with open(ws_mod.__file__, "r", encoding="utf-8") as handle:
            text = handle.read()
        self.assertNotIn("import character_state", text)
        self.assertNotIn("from .character_state", text)
        self.assertNotIn("from .agent.relationship_context", text)
        self.assertNotIn("from ..agent.relationship_context", text)

    def test_goal_hint_map_is_read_only_and_seeded_only(self):
        self.assertEqual(
            ws_mod.DECISION_GOAL_ACTIVITY_HINTS,
            {
                "morning-reading-week": "reading",
                "finish-one-book": "reading",
                "try-three-dishes": "eating",
            },
        )
        for target in ws_mod.DECISION_GOAL_ACTIVITY_HINTS.values():
            self.assertIn(target, ws_mod.VALID_ACTIVITIES)

    def test_influence_stage_stays_inside_existing_activities(self):
        for status in ("stranger", "familiar", "close", "dating"):
            res = decide_activity(
                idle_state(),
                NOON_JKT,
                JKT,
                DecisionInputs(
                    relationship_status=status, goal_activities=("reading", "playing")
                ),
            )
            if res.action is not None:
                self.assertIn(res.action, ws_mod.VALID_ACTIVITIES)

    def test_agent_helper_is_pure_and_llm_free(self):
        import pathlib

        agent = pathlib.Path(ws_mod.__file__).parent / "agent" / "agents"
        source = (agent / "basic_memory_agent.py").read_text(encoding="utf-8")
        body = source[
            source.index("def _decision_inputs(self)") : source.index(
                "def _episodic_context_for_prompt(self)"
            )
        ]
        for token in ("await ", "chat_completion", "async def", "create_task"):
            self.assertNotIn(token, body, token)
        self.assertIn("inputs=self._decision_inputs()", source)


class DecisionStampSurvivesReactiveTest(unittest.TestCase):
    """Regression: apply_reactive() must not drop the decision stamp.

    Dropping it wiped the cooldown anchor, so a decision could repeat within
    the 15-minute window whenever an emotion/reactive mood landed on the same
    turn.
    """

    STAMP = "2026-10-02T12:55:00+00:00"
    MOMENT = datetime(2026, 10, 2, 13, 0, tzinfo=UTC)

    def _state(self, **over):
        base = {
            "activity": "playing",
            "energy": 70,
            "mood": "calm",
            "activity_started_at": "2026-10-02T12:00:00+00:00",
            "last_update_at": "2026-10-02T12:59:00+00:00",
            "last_autonomous_decision_at": self.STAMP,
        }
        base.update(over)
        return WorldState(**base)

    def test_stamp_survives_mapped_emotion(self):
        out, changed = ws_mod.apply_reactive(self._state(), ["joy"], self.MOMENT, JKT)
        self.assertTrue(changed)
        self.assertEqual(out.last_autonomous_decision_at, self.STAMP)

    def test_stamp_survives_negative_emotion(self):
        out, changed = ws_mod.apply_reactive(self._state(), ["anger"], self.MOMENT, JKT)
        self.assertTrue(changed)
        self.assertEqual(out.last_autonomous_decision_at, self.STAMP)

    def test_stamp_survives_neutral_turn(self):
        out, changed = ws_mod.apply_reactive(self._state(), [], self.MOMENT, JKT)
        self.assertFalse(changed)
        self.assertEqual(out.last_autonomous_decision_at, self.STAMP)

    def test_stamp_survives_armed_ttl_decay(self):
        out, changed = ws_mod.apply_reactive(
            self._state(
                mood="happy", mood_ttl_turns=2, mood_set_at="2026-10-02T12:59:00+00:00"
            ),
            [],
            self.MOMENT,
            JKT,
        )
        self.assertTrue(changed)
        self.assertEqual(out.last_autonomous_decision_at, self.STAMP)

    def test_legacy_state_without_field_stays_none(self):
        state = WorldState(
            activity="playing",
            energy=70,
            mood="calm",
            activity_started_at="2026-10-02T12:00:00+00:00",
            last_update_at="2026-10-02T12:59:00+00:00",
        )
        out, _ = ws_mod.apply_reactive(state, ["joy"], self.MOMENT, JKT)
        self.assertIsNone(out.last_autonomous_decision_at)

    def test_cooldown_survives_emotion_turn(self):
        # Decision at 13:00, emotion turn 30s later: cooldown must still hold.
        moment = datetime(2026, 10, 2, 13, 0, tzinfo=UTC)
        state = idle_state(
            activity_started_at="2026-09-29T00:00:00+00:00",
            last_update_at="2026-09-29T00:00:00+00:00",
        )
        decided = decide_activity(
            state, moment, JKT, DecisionInputs(relationship_status="dating")
        )
        self.assertEqual(decided.action, "playing")
        applied = ws_mod._apply_decision(state, decided, moment, JKT)
        self.assertEqual(
            applied.last_autonomous_decision_at, "2026-10-02T13:00:00+00:00"
        )
        emoted, _changed = ws_mod.apply_reactive(
            applied, ["joy"], moment + timedelta(seconds=30), JKT
        )
        self.assertEqual(
            emoted.last_autonomous_decision_at, "2026-10-02T13:00:00+00:00"
        )
        # The emotion turn moved playing -> idle; a fresh idle holds before
        # the cooldown branch is ever reached (existing precedence).
        fresh = decide_activity(
            emoted,
            moment + timedelta(seconds=30),
            JKT,
            DecisionInputs(relationship_status="dating"),
        )
        self.assertIsNone(fresh.action)
        self.assertEqual(fresh.reason, "no_change")
        # Once that idle goes stale again, the survived stamp must still gate.
        aged = WorldState(
            location=emoted.location,
            activity="idle",
            energy=emoted.energy,
            mood=emoted.mood,
            time_context=emoted.time_context,
            activity_started_at="2026-10-02T11:59:00+00:00",
            last_update_at="2026-10-02T12:59:00+00:00",
            recent_activity_history=emoted.recent_activity_history,
            mood_ttl_turns=emoted.mood_ttl_turns,
            mood_set_at=emoted.mood_set_at,
            last_autonomous_decision_at=emoted.last_autonomous_decision_at,
        )
        blocked = decide_activity(
            aged,
            moment + timedelta(seconds=60),
            JKT,
            DecisionInputs(relationship_status="dating"),
        )
        self.assertIsNone(blocked.action)
        self.assertEqual(blocked.reason, "cooldown_active")
        # After the cooldown window the pick may fire again.
        later = decide_activity(
            aged,
            moment + timedelta(seconds=ws_mod.DECISION_COOLDOWN_S + 120),
            JKT,
            DecisionInputs(relationship_status="dating"),
        )
        self.assertIsNotNone(later.action)
        self.assertNotEqual(later.reason, "cooldown_active")


if __name__ == "__main__":
    unittest.main()
