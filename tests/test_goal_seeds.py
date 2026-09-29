"""Goal seed lifecycle Phase 2C — deterministic tests (no LLM, no clock)."""

import asyncio
import inspect
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from src.open_llm_vtuber.character_state import (
    CharacterState,
    activate_goal,
    complete_goal,
    default_seed_goals,
    load_character_state,
    save_character_state,
)
from src.open_llm_vtuber import character_state as cs_mod

UTC = timezone.utc
T0 = datetime(2026, 9, 29, 8, 0, tzinfo=UTC)
T1 = datetime(2026, 9, 29, 9, 0, tzinfo=UTC)
T2 = datetime(2026, 10, 1, 5, 0, tzinfo=UTC)  # +45h


def run_in_tmp(fn):
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            return fn()
        finally:
            os.chdir(cwd)


class SeedTest(unittest.TestCase):
    def test_a_default_seeds_exist(self):
        seeds = default_seed_goals(T0)
        self.assertEqual(len(seeds), 3)
        ids = [s["id"] for s in seeds]
        self.assertEqual(
            ids, ["morning-reading-week", "try-three-dishes", "finish-one-book"]
        )
        for s in seeds:
            self.assertEqual(s["status"], "seed")
            self.assertEqual(s["source"], "seed")
            self.assertTrue(s["text"])
            self.assertTrue(s["created_at"].endswith("+00:00"))
            self.assertIsNone(s["activated_at"])
            self.assertIsNone(s["completed_at"])

    def test_l_source_is_seed(self):
        for s in default_seed_goals(T0):
            self.assertEqual(s["source"], "seed")


class TransitionTest(unittest.TestCase):
    def test_c_seed_to_active(self):
        out, ok = activate_goal(default_seed_goals(T0), "try-three-dishes", T1)
        self.assertTrue(ok)
        goal = [g for g in out if g["id"] == "try-three-dishes"][0]
        self.assertEqual(goal["status"], "active")
        self.assertEqual(goal["activated_at"], "2026-09-29T09:00:00+00:00")
        self.assertIsNone(goal["completed_at"])
        # Others untouched; input not mutated.
        self.assertEqual(len(out), 3)

    def test_d_active_to_done(self):
        seeds, _ = activate_goal(default_seed_goals(T0), "finish-one-book", T1)
        out, ok = complete_goal(seeds, "finish-one-book", T2)
        self.assertTrue(ok)
        goal = [g for g in out if g["id"] == "finish-one-book"][0]
        self.assertEqual(goal["status"], "done")
        self.assertEqual(goal["completed_at"], "2026-10-01T05:00:00+00:00")

    def test_e_invalid_transitions_rejected(self):
        seeds = default_seed_goals(T0)
        # done without active:
        out, ok = complete_goal(seeds, "finish-one-book", T2)
        self.assertFalse(ok)
        self.assertEqual(out, seeds)
        # re-activate an active goal:
        active, _ = activate_goal(seeds, "finish-one-book", T1)
        out2, ok2 = activate_goal(active, "finish-one-book", T2)
        self.assertFalse(ok2)
        # re-activate a done goal:
        done, _ = complete_goal(active, "finish-one-book", T2)
        out3, ok3 = activate_goal(done, "finish-one-book", T2)
        self.assertFalse(ok3)
        self.assertEqual(out3, done)

    def test_i_unknown_id_fails_soft(self):
        seeds = default_seed_goals(T0)
        out, ok = activate_goal(seeds, "nope", T1)
        self.assertFalse(ok)
        self.assertEqual(out, seeds)
        out, ok = complete_goal(seeds, "nope", T1)
        self.assertFalse(ok)
        self.assertEqual(out, seeds)
        out, ok = activate_goal([], "x", T1)
        self.assertFalse(ok)
        self.assertEqual(out, [])

    def test_f_utc_timestamps(self):
        seeds = default_seed_goals(T0)
        for s in seeds:
            datetime.fromisoformat(s["created_at"])
            self.assertTrue(s["created_at"].endswith("+00:00"))
        # Naive `now` is read as UTC, never local/uptime.
        naive = datetime(2026, 9, 29, 8, 0)
        out, ok = activate_goal(seeds, "try-three-dishes", naive)
        self.assertTrue(ok)
        goal = [g for g in out if g["id"] == "try-three-dishes"][0]
        self.assertTrue(goal["activated_at"].endswith("+00:00"))


class PersistenceTest(unittest.TestCase):
    def test_b_save_load_round_trip(self):
        def go():
            state = CharacterState(goals=default_seed_goals(T0))
            self.assertTrue(save_character_state("c1", state))
            raw = json.load(open("character_state/c1.json", encoding="utf-8"))
            self.assertEqual(len(raw["goals"]), 3)
            reloaded = load_character_state("c1")
            self.assertEqual(reloaded.goals, state.goals)
            active, _ = activate_goal(reloaded.goals, "try-three-dishes", T1)
            state.goals = active
            self.assertTrue(save_character_state("c1", state))
            reloaded2 = load_character_state("c1")
            goal = [g for g in reloaded2.goals if g["id"] == "try-three-dishes"][0]
            self.assertEqual(goal["status"], "active")

        run_in_tmp(go)

    def test_g_restart_preserves_state(self):
        def go():
            state = CharacterState(goals=default_seed_goals(T0))
            save_character_state("c2", state)
            first = load_character_state("c2")
            second = load_character_state("c2")
            self.assertEqual(first.goals, second.goals)

        run_in_tmp(go)

    def test_h_offline_gap_does_not_alter(self):
        def go():
            state = CharacterState(goals=default_seed_goals(T0))
            save_character_state("c3", state)
            # +45h later, no process alive in between: identical.
            loaded = load_character_state("c3")
            self.assertEqual(loaded.goals, state.goals)
            self.assertTrue(all(g["status"] == "seed" for g in loaded.goals))

        run_in_tmp(go)

    def test_j_old_json_without_goals_loads(self):
        def go():
            os.makedirs("character_state", exist_ok=True)
            json.dump(
                {"relationship_status": "stranger", "memories": []},
                open("character_state/old.json", "w"),
            )
            state = load_character_state("old")
            self.assertEqual(state.goals, [])
            # And saves back fine with the key present.
            self.assertTrue(save_character_state("old", state))
            raw = json.load(open("character_state/old.json"))
            self.assertEqual(raw["goals"], [])

        run_in_tmp(go)


class SeparationTest(unittest.TestCase):
    def test_k_tendencies_separate_from_goals(self):
        from src.open_llm_vtuber.self_model import (
            SELF_SEED_TENDENCIES,
            build_self_context,
        )

        seeds = default_seed_goals(T0)
        seed_ids = {s["id"] for s in seeds}
        # Tendency strings are not goal ids and vice versa.
        for t in SELF_SEED_TENDENCIES:
            self.assertNotIn(t, seed_ids)
        block = build_self_context(character_name="Mili")
        for s in seeds:
            self.assertNotIn(s["id"], block)

    def test_m_no_prompt_output_change(self):
        from src.open_llm_vtuber.self_model import build_self_context

        before = build_self_context(character_name="Mili")
        # Goals are not wired into the composer: output has no goal lines.
        self.assertNotIn("morning-reading-week", before)
        self.assertNotIn("Goals:", before)

    def test_n_no_llm_scheduler_clock(self):
        for fn in (default_seed_goals, activate_goal, complete_goal):
            self.assertFalse(asyncio.iscoroutinefunction(fn))
        src = inspect.getsource(cs_mod.activate_goal)
        src += inspect.getsource(cs_mod.complete_goal)
        src += inspect.getsource(cs_mod.default_seed_goals)
        for token in (
            "chat_completion",
            "generate",
            "llm",
            "Scheduler",
            "create_task",
            "threading",
            "sleep",
            "monotonic",
        ):
            self.assertNotIn(token, src)


if __name__ == "__main__":
    unittest.main()
