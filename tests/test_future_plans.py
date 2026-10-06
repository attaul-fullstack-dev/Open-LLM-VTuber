"""Natural future plans — deterministic first-person grammar (no LLM).

NOW = Friday 2026-10-02 12:00 Jakarta. Covers the 20 Phase 7 cases:
dated/relative/weekday/undated capture, hypothetical/question/general
rejection, dedup, distinct rows, timezone, midnight, restart, malformed
state, context retrieval without hallucination, completion, cancellation,
reminder regression, proactive budget, bounded storage.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber.character_state import (
    complete_future_intention,
    load_character_state,
    record_future_intention,
)
from src.open_llm_vtuber.future_intentions import (
    KIND_PLAN,
    KIND_REMINDER,
    build_future_intention_context,
    detect_any_future_intention,
    detect_future_intention,
    detect_future_plan,
    due_intentions,
    pending_intentions,
)

JKT = "Asia/Jakarta"
# Friday 2026-10-02 12:00 Jakarta.
NOW = datetime(2026, 10, 2, 5, 0, tzinfo=timezone.utc)
CONF = "planchar"


class CaptureTest(unittest.TestCase):
    def test_1_plan_with_explicit_jam(self):
        found = detect_future_plan("besok jam 9 aku ada ujian", now=NOW, tz=JKT)
        self.assertIsNotNone(found)
        self.assertEqual(found.kind, KIND_PLAN)
        stamp = datetime.fromisoformat(found.due_at)
        local = stamp.astimezone(__import__("zoneinfo").ZoneInfo(JKT))
        self.assertEqual((local.day, local.hour), (3, 9))

    def test_2_relative_day(self):
        found = detect_future_plan("besok aku ada ujian", now=NOW, tz=JKT)
        self.assertIsNotNone(found)
        stamp = datetime.fromisoformat(found.due_at)
        local = stamp.astimezone(__import__("zoneinfo").ZoneInfo(JKT))
        self.assertEqual((local.day, local.hour), (3, 7))

    def test_3_weekday(self):
        found = detect_future_plan(
            "Selasa aku bakal nanya lagi soal bug websocket", now=NOW, tz=JKT
        )
        self.assertIsNotNone(found)
        stamp = datetime.fromisoformat(found.due_at)
        local = stamp.astimezone(__import__("zoneinfo").ZoneInfo(JKT))
        self.assertEqual(local.weekday(), 1)  # Tuesday
        self.assertEqual(local.hour, 7)

    def test_4_no_exact_date_stored_without_due(self):
        found = detect_future_plan("aku mau mulai project baru", now=NOW, tz=JKT)
        self.assertIsNotNone(found)
        self.assertEqual(found.kind, KIND_PLAN)
        self.assertIsNone(found.due_at)

    def test_5_hypothetical_rejected(self):
        self.assertIsNone(
            detect_future_plan("kalau besok hujan mungkin gw gak keluar", now=NOW, tz=JKT)
        )
        self.assertIsNone(
            detect_any_future_intention("kalau besok hujan mungkin gw gak keluar", now=NOW, tz=JKT)
        )

    def test_6_question_rejected(self):
        self.assertIsNone(detect_future_plan("besok hujan gak?", now=NOW, tz=JKT))
        self.assertIsNone(detect_future_plan("besok aku ada ujian?", now=NOW, tz=JKT))

    def test_7_general_statement_rejected(self):
        self.assertIsNone(
            detect_future_plan("orang biasanya kerja besok", now=NOW, tz=JKT)
        )
        self.assertIsNone(
            detect_any_future_intention("orang biasanya kerja besok", now=NOW, tz=JKT)
        )

    def test_reminder_still_wins_over_plan(self):
        found = detect_any_future_intention(
            "tolong ingetin aku besok jam 7 berangkat", now=NOW, tz=JKT
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.kind, KIND_REMINDER)


class StorageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_8_duplicate_dedup(self):
        record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        self.assertEqual(len(load_character_state(CONF).future_intentions), 1)

    def test_9_distinct_intentions_coexist(self):
        record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        record_future_intention(
            CONF, "Selasa aku bakal nanya lagi soal bug websocket", now=NOW, tz=JKT
        )
        self.assertEqual(len(load_character_state(CONF).future_intentions), 2)

    def test_10_timezone_changes_utc_due(self):
        a = record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        b = record_future_intention("planz", "besok aku ada ujian", now=NOW, tz="UTC")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertNotEqual(a["due_at"], b["due_at"])
        # Same wall clock 07:00, different zones → 7h apart.
        da = datetime.fromisoformat(a["due_at"])
        db = datetime.fromisoformat(b["due_at"])
        self.assertEqual(abs((da - db).total_seconds()), 7 * 3600)

    def test_11_midnight_boundary(self):
        # 23:55 Jakarta Friday → "besok" is Saturday 07:00 local.
        late = datetime(2026, 10, 2, 16, 55, tzinfo=timezone.utc)
        found = detect_future_plan("besok aku ada ujian", now=late, tz=JKT)
        local = datetime.fromisoformat(found.due_at).astimezone(
            __import__("zoneinfo").ZoneInfo(JKT)
        )
        self.assertEqual((local.day, local.hour), (3, 7))

    def test_12_restart_persistence(self):
        record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        again = load_character_state(CONF)
        self.assertEqual(len(again.future_intentions), 1)
        self.assertEqual(again.future_intentions[0]["kind"], KIND_PLAN)

    def test_13_malformed_state_tolerated(self):
        os.makedirs("character_state", exist_ok=True)
        with open(os.path.join("character_state", f"{CONF}.json"), "w") as handle:
            json.dump(
                {
                    "future_intentions": [
                        {"text": "besok aku ada ujian", "status": "pending"},
                        {"text": "", "status": "pending"},
                        "garbage",
                        {"text": "ingatkan aku besok minum obat", "kind": "weird"},
                    ]
                },
                handle,
            )
        rows = load_character_state(CONF).future_intentions
        self.assertEqual(len(rows), 2)
        # Missing kind and unknown kind both default to reminder.
        kinds = {row["kind"] for row in rows}
        self.assertEqual(kinds, {KIND_REMINDER})
        # Row without kind defaults to reminder; unknown kind maps to reminder.
        by_text = {row["text"]: row["kind"] for row in rows}
        self.assertEqual(by_text["ingatkan aku besok minum obat"], KIND_REMINDER)

    def test_16_completion(self):
        stored = record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        self.assertTrue(complete_future_intention(CONF, stored["id"]))
        rows = load_character_state(CONF).future_intentions
        self.assertEqual(pending_intentions(rows), [])
        self.assertEqual(due_intentions(rows, now=NOW + timedelta(days=5)), [])

    def test_17_cancellation(self):
        from src.open_llm_vtuber.character_state import complete_future_intention as _  # noqa
        from src.open_llm_vtuber.future_intentions import cancel_future_intention

        stored = record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        state_rows = load_character_state(CONF).future_intentions
        updated, changed = cancel_future_intention(state_rows, stored["id"])
        self.assertTrue(changed)
        self.assertEqual(pending_intentions(updated), [])

    def test_20_bounded_storage(self):
        for i in range(25):
            record_future_intention(CONF, f"besok aku ada acara nomor {i}", now=NOW, tz=JKT)
        pending = pending_intentions(load_character_state(CONF).future_intentions)
        self.assertLessEqual(len(pending), 20)


class ContextTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_14_context_retrieval(self):
        record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        block = build_future_intention_context(
            load_character_state(CONF).future_intentions, tz=JKT
        )
        self.assertIn("upcoming", block)
        self.assertIn("besok aku ada ujian", block)

    def test_15_no_hallucination(self):
        record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        block = build_future_intention_context(
            load_character_state(CONF).future_intentions, tz=JKT
        )
        for invented in ("Matematika", "08:00", "ruang", "pak Budi", "exam"):
            self.assertNotIn(invented, block)

    def test_18_reminder_regression(self):
        stored = record_future_intention(
            CONF, "tolong ingetin aku besok jam 7 berangkat", now=NOW, tz=JKT
        )
        self.assertIsNotNone(stored)
        self.assertEqual(stored["kind"], KIND_REMINDER)
        self.assertIsNotNone(stored["due_at"])
        # Old detector path unchanged: verb + anchor still required.
        self.assertIsNone(detect_future_intention("besok aku ada ujian", now=NOW, tz=JKT))

    def test_19_proactive_budget_still_binds(self):
        from src.open_llm_vtuber.proactive_gate import (
            ProactiveBudgetState,
            ProactiveGateConfig,
            classify_trigger,
            evaluate_gate,
            local_day_key,
            local_hour_key,
            resolve_user_tz,
        )

        record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        later = NOW + timedelta(days=2)
        rows = load_character_state(CONF).future_intentions
        self.assertEqual(len(due_intentions(rows, now=later)), 1)
        zone = resolve_user_tz(JKT)
        budget = ProactiveBudgetState(
            daily_request_count=0,
            daily_count_date=local_day_key(later, zone),
            hourly_meaningful_count=0,
            hourly_count_hour=local_hour_key(later, zone),
        )
        trigger = classify_trigger(explicit_reminder=True)
        cfg = ProactiveGateConfig()
        self.assertTrue(evaluate_gate(budget, cfg, trigger, now=later, tz=JKT).allowed)
        full = ProactiveBudgetState(
            daily_request_count=60,
            daily_count_date=local_day_key(later, zone),
            hourly_meaningful_count=0,
            hourly_count_hour=local_hour_key(later, zone),
        )
        self.assertFalse(evaluate_gate(full, cfg, trigger, now=later, tz=JKT).allowed)


if __name__ == "__main__":
    unittest.main()
