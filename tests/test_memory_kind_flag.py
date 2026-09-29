"""Phase 2B — typed memory flag `kind` (no behavior change, no LLM)."""

import json
import os
import tempfile
import unittest

from src.open_llm_vtuber.character_state import (
    add_character_memory,
    build_character_memory_context,
    load_character_state,
)
from src.open_llm_vtuber.self_model import (
    build_self_context,
    derive_activity_preferences,
)


def run_in_tmp(testcase, fn):
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            return fn()
        finally:
            os.chdir(cwd)


class KindFlagTest(unittest.TestCase):
    def test_a_old_json_without_kind_loads_as_empty(self):
        def go():
            path = os.path.join("character_state", "c1.json")
            os.makedirs("character_state", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "relationship_status": "stranger",
                        "memories": [
                            {
                                "text": "legacy fact",
                                "added_at": "2026-09-20T00:00:00+00:00",
                                "explicit": True,
                            }
                        ],
                    },
                    f,
                )
            state = load_character_state("c1")
            self.assertEqual(state.memories[0]["kind"], "")
            return True

        self.assertTrue(run_in_tmp(self, go))

    def test_b_preference_kind_round_trip(self):
        def go():
            state = add_character_memory("c2", "suka baca buku", kind="preference")
            self.assertIsNotNone(state)
            raw = json.load(
                open(os.path.join("character_state", "c2.json"), encoding="utf-8")
            )
            self.assertEqual(raw["memories"][0]["kind"], "preference")
            reloaded = load_character_state("c2")
            self.assertEqual(reloaded.memories[0]["kind"], "preference")

        run_in_tmp(self, go)

    def test_c_other_kind_round_trip(self):
        def go():
            add_character_memory("c3", "ulang tahun user", kind="event")
            reloaded = load_character_state("c3")
            self.assertEqual(reloaded.memories[0]["kind"], "event")

        run_in_tmp(self, go)

    def test_d_no_kind_behaves_as_before(self):
        def go():
            state = add_character_memory("c4", "fakta biasa")
            mem = state.memories[0]
            self.assertEqual(mem["kind"], "")
            self.assertTrue(mem["explicit"])
            from datetime import datetime, timezone

            block = build_character_memory_context(
                state,
                tz="Asia/Jakarta",
                now=datetime(2026, 10, 1, tzinfo=timezone.utc),
            )
            self.assertIn("fakta biasa", block)
            self.assertNotIn("kind", block)
            self.assertNotIn("preference", block)

        run_in_tmp(self, go)


class Phase2AUnchangedTest(unittest.TestCase):
    def _ev(self):
        h = [
            {
                "from": "idle",
                "to": "reading",
                "at": f"2026-09-2{d}T08:00:00+00:00",
                "location": "room",
            }
            for d in (7, 8, 9)
        ]
        m = [
            {
                "text": "baca buku",
                "added_at": "2026-09-29T08:00:00+00:00",
                "explicit": True,
            },
        ]
        return h, m

    def test_e_phase2a_absent_kind(self):
        h, m = self._ev()
        out = derive_activity_preferences(h, m, tz="Asia/Jakarta")
        self.assertTrue(out[0].established)
        self.assertEqual(out[0].evidence_count, 4)

    def test_f_phase2a_with_kind(self):
        h, m = self._ev()
        m = [dict(item, kind="preference") for item in m]
        h = [dict(item, kind="preference") for item in h]
        out = derive_activity_preferences(h, m, tz="Asia/Jakarta")
        self.assertTrue(out[0].established)
        self.assertEqual(out[0].evidence_count, 4)

    def test_g_composer_byte_identical(self):
        plain = build_self_context(character_name="Mili")
        self.assertNotIn("preference", plain)
        self.assertNotIn("kind", plain)


if __name__ == "__main__":
    unittest.main()
