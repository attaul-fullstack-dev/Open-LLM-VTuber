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
    WorldState,
    decide_activity,
    load_and_reconcile_world_state,
    reconcile,
    save_world_state,
)

JKT = "Asia/Jakarta"
UTC = timezone.utc
# Monday Sep 29 2026 02:00Z == 09:00 Jakarta (morning).
MON_09_JKT = datetime(2026, 9, 29, 2, 0, tzinfo=UTC)
# Same day 16:00Z == 23:00 Jakarta (night).
MON_23_JKT = datetime(2026, 9, 29, 16, 0, tzinfo=UTC)


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


if __name__ == "__main__":
    unittest.main()
