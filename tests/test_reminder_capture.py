"""Reminder capture — explicit, referential, ambiguous, persistence, receipts.

Regression target (mission issues 1 and 2):

* "Nanti jam 11.51 ingetin aku dengan bilang 'TES PENGINGAT BERHASIL'." must be
  stored with a resolved due time and be visible in the Memory Manager.
* "Nanti jam 11.54 bilang kaya tadi lagi ya." must be stored by resolving the
  referenced payload from the conversation, not because the user said the word
  "reminder".
* An unresolvable reference must NOT be stored and must produce a clarify
  receipt, so the model cannot silently invent content or claim success.
* A failed write must never be reported as a stored reminder.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from src.open_llm_vtuber.character_state import (
    capture_reminder,
    claim_due_future_intentions,
    complete_claimed_future_intentions,
    complete_future_intention,
    load_character_state,
    record_future_intention,
    save_character_state,
)
from src.open_llm_vtuber.future_intentions import (
    detect_contextual_reminder,
    detect_reminder_request,
    due_intentions,
    next_due_at,
    pending_intentions,
)
from src.open_llm_vtuber.reminder_receipt import (
    build_reminder_receipt_block,
    process_reminder_turn,
)

JKT = "Asia/Jakarta"
CONF = "capturechar"
# 11:40 Jakarta, ten minutes before the first example.
NOW = datetime(2026, 10, 21, 4, 40, tzinfo=timezone.utc)

PRIOR_TURNS = [
    {"role": "user", "content": "Nanti jam 11.51 ingetin aku dengan bilang 'TES PENGINGAT BERHASIL'."},
    {"role": "assistant", "content": "Oke, nanti jam 11.51 aku bilang 'TES PENGINGAT BERHASIL' ya."},
]

# The turn that "kaya tadi" points at offers TWO different payloads, so the
# referent cannot be known without guessing.
AMBIGUOUS_TURNS = [
    {"role": "user", "content": "ingetin aku besok jam 7 bilang 'minum obat' ya"},
    {"role": "assistant", "content": "Oke, 'minum obat' atau 'minum vitamin'?"},
]


class DetectionTest(unittest.TestCase):
    def test_explicit_request_captured_with_payload(self):
        detection = detect_reminder_request(
            "Nanti jam 11.51 ingetin aku dengan bilang 'TES PENGINGAT BERHASIL'.",
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(detection.status, "captured")
        self.assertEqual(detection.intention.delivery_text, "TES PENGINGAT BERHASIL")
        due = datetime.fromisoformat(detection.intention.due_at)
        self.assertEqual(due.astimezone(timezone.utc), datetime(2026, 10, 21, 4, 51, tzinfo=timezone.utc))

    def test_referential_request_resolves_payload_from_context(self):
        detection = detect_contextual_reminder(
            "Nanti jam 11.54 bilang kaya tadi lagi ya.",
            PRIOR_TURNS,
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(detection.status, "captured")
        self.assertEqual(detection.intention.delivery_text, "TES PENGINGAT BERHASIL")
        due = datetime.fromisoformat(detection.intention.due_at)
        self.assertEqual(due.astimezone(timezone.utc), datetime(2026, 10, 21, 4, 54, tzinfo=timezone.utc))

    def test_explicit_request_with_reference_resolves_payload(self):
        detection = detect_reminder_request(
            "Nanti jam 11.54 ingetin aku bilang kaya tadi lagi ya.",
            PRIOR_TURNS,
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(detection.status, "captured")
        self.assertIsNotNone(detection.intention)
        self.assertEqual(detection.intention.delivery_text, "TES PENGINGAT BERHASIL")

    def test_explicit_request_without_reference_context_needs_clarification(self):
        detection = detect_reminder_request(
            "Nanti jam 11.54 ingetin aku bilang kaya tadi lagi ya.",
            [],
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(detection.status, "clarify")
        self.assertIsNone(detection.intention)

    def test_ingetin_with_kayak_tadi_alone_needs_clarification(self):
        # Exact audit example: explicit verb + reference, no resolvable payload.
        detection = detect_reminder_request(
            "Nanti jam 11.54 ingetin aku kayak tadi lagi ya.",
            [],
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(detection.status, "clarify")
        self.assertEqual(detection.reason, "unresolved_reference")
        self.assertIsNone(detection.intention)

    def test_two_distinct_referenced_payloads_need_clarification(self):
        detection = detect_contextual_reminder(
            "nanti jam 9 bilang kaya tadi lagi ya",
            AMBIGUOUS_TURNS,
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(detection.status, "clarify")
        self.assertIsNone(detection.intention)

    def test_quoted_reference_is_never_used_as_the_payload(self):
        # The nearest turn only quotes the reference phrase itself, so the
        # resolver must skip it and use the actual earlier payload.
        history = [
            {"role": "user", "content": "ingetin aku besok jam 7 bilang 'minum obat' ya"},
            {"role": "assistant", "content": "Oke, nanti aku bilang 'kaya tadi lagi' ya"},
        ]
        detection = detect_contextual_reminder(
            "nanti jam 9 bilang kaya tadi lagi ya", history, now=NOW, tz=JKT
        )
        self.assertEqual(detection.status, "captured")
        self.assertEqual(detection.intention.delivery_text, "minum obat")

    def test_two_distinct_quoted_payloads_need_clarification(self):
        # One turn naming two different payloads: never take the first quote.
        detection = detect_reminder_request(
            "Besok jam 7 ingetin aku bilang 'minum obat' atau 'minum vitamin' ya.",
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(detection.status, "clarify")
        self.assertEqual(detection.reason, "ambiguous_payload")
        self.assertIsNone(detection.intention)

    def test_two_distinct_quoted_payloads_need_clarification_via_reference(self):
        detection = detect_contextual_reminder(
            "nanti jam 9 bilang 'minum obat' atau 'minum vitamin' kaya tadi lagi ya",
            PRIOR_TURNS,
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(detection.status, "clarify")
        self.assertIsNone(detection.intention)

    def test_repeated_identical_quotes_count_as_one_payload(self):
        detection = detect_reminder_request(
            "Besok jam 7 ingetin aku bilang 'minum obat', 'minum obat' ya.",
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(detection.status, "captured")
        self.assertEqual(detection.intention.delivery_text, "minum obat")

    def test_single_valid_quoted_payload_still_captures(self):
        detection = detect_reminder_request(
            "Besok jam 7 ingetin aku bilang 'minum obat' ya.",
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(detection.status, "captured")
        self.assertEqual(detection.intention.delivery_text, "minum obat")
        self.assertIsNotNone(detection.intention.due_at)

    def test_quoted_reference_is_not_treated_as_payload(self):
        detection = detect_reminder_request(
            "Nanti jam 11.54 ingetin aku bilang 'kaya tadi lagi' ya.",
            PRIOR_TURNS,
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(detection.status, "clarify")
        self.assertIsNone(detection.intention)

    def test_user_speaking_is_not_a_referential_reminder(self):
        for text in (
            "nanti jam 7 aku bilang kaya tadi lagi ya",
            "nanti jam 7 aku akan bilang kaya tadi lagi ya",
        ):
            with self.subTest(text=text):
                detection = detect_contextual_reminder(
                    text, PRIOR_TURNS, now=NOW, tz=JKT
                )
                self.assertEqual(detection.status, "none")

    def test_referential_without_context_needs_clarification(self):
        detection = detect_contextual_reminder(
            "Nanti jam 11.54 bilang kaya tadi lagi ya.", [], now=NOW, tz=JKT
        )
        self.assertEqual(detection.status, "clarify")
        self.assertIsNone(detection.intention)

    def test_unquoted_earlier_payload_is_resolved(self):
        history = [
            {"role": "user", "content": "ingetin aku jam 11.51 bilang minum obat ya"},
            {"role": "assistant", "content": "siap"},
        ]
        detection = detect_contextual_reminder(
            "nanti jam 11.54 ulangi ya", history, now=NOW, tz=JKT
        )
        self.assertEqual(detection.status, "captured")
        self.assertEqual(detection.intention.delivery_text, "minum obat")

    def test_ordinary_mention_is_not_a_reminder(self):
        for text in (
            "nanti jam 7 aku bilang ke ibu mau pulang",
            "jam 8 kita ketemu ya",
            "nanti jam 9 kamu bilang apa?",
            "kalau nanti jam 9 bilang kaya tadi, gimana?",
        ):
            self.assertEqual(
                detect_contextual_reminder(text, PRIOR_TURNS, now=NOW, tz=JKT).status,
                "none",
                msg=text,
            )

    def test_plan_grammar_still_works(self):
        detection = detect_reminder_request(
            "besok aku ada ujian jam 7 pagi", now=NOW, tz=JKT
        )
        self.assertEqual(detection.status, "none")

    def test_next_due_at_reports_earliest_pending(self):
        rows = [
            {"id": "a", "text": "x", "status": "pending", "due_at": (NOW + timedelta(minutes=5)).isoformat()},
            {"id": "b", "text": "y", "status": "pending", "due_at": (NOW + timedelta(minutes=1)).isoformat()},
        ]
        self.assertEqual(next_due_at(rows, now=NOW), NOW + timedelta(minutes=1))


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_referential_request_is_persisted_and_visible(self):
        status, row = capture_reminder(
            CONF,
            "Nanti jam 11.54 bilang kaya tadi lagi ya.",
            now=NOW,
            tz=JKT,
            recent_messages=PRIOR_TURNS,
        )
        self.assertEqual(status, "captured")
        self.assertIsNotNone(row)
        stored = load_character_state(CONF).future_intentions
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["status"], "pending")
        self.assertEqual(stored[0]["delivery_text"], "TES PENGINGAT BERHASIL")
        # Visible through the same reader the Memory Manager uses.
        self.assertEqual([item["id"] for item in pending_intentions(stored)], [row["id"]])

    def test_user_reminding_mili_is_not_persisted_as_reminder(self):
        # Speaker direction: the user is the actor, Mili is the addressee.
        for text in (
            "besok aku akan ingatkan kamu minum obat",
            "nanti aku ingetin kamu ya",
        ):
            with self.subTest(text=text):
                status, row = capture_reminder(CONF, text, now=NOW, tz=JKT)
                self.assertEqual(status, "none")
                self.assertIsNone(row)
                self.assertEqual(load_character_state(CONF).future_intentions, [])

    def test_ambiguous_quoted_payload_is_not_persisted(self):
        status, row = capture_reminder(
            CONF,
            "Besok jam 7 ingetin aku bilang 'minum obat' atau 'minum vitamin' ya.",
            now=NOW,
            tz=JKT,
        )
        self.assertEqual(status, "clarify")
        self.assertIsNone(row)
        self.assertEqual(load_character_state(CONF).future_intentions, [])

    def test_undated_reminder_is_stored_but_never_due_or_claimed(self):
        status, row = capture_reminder(
            CONF, "ingetin aku hari ini minum obat", now=NOW, tz=JKT
        )
        self.assertEqual(status, "captured")
        self.assertIsNotNone(row)
        self.assertIsNone(row["due_at"])
        later = NOW + timedelta(days=3)
        stored = load_character_state(CONF).future_intentions
        # Recallable and visible, but never due, never schedulable, never
        # claimable by the proactive dispatcher.
        self.assertEqual([item["id"] for item in pending_intentions(stored)], [row["id"]])
        self.assertEqual(due_intentions(stored, now=later), [])
        self.assertIsNone(next_due_at(stored, now=later))
        self.assertEqual(
            claim_due_future_intentions(CONF, owner="owner-a", now=later), []
        )
        self.assertNotIn("claimed_by", load_character_state(CONF).future_intentions[0])

    def test_ambiguous_reference_is_not_persisted(self):
        status, row = capture_reminder(
            CONF, "Nanti jam 11.54 bilang kaya tadi lagi ya.", now=NOW, tz=JKT
        )
        self.assertEqual(status, "clarify")
        self.assertIsNone(row)
        self.assertEqual(load_character_state(CONF).future_intentions, [])

    def test_reference_with_two_candidates_is_not_persisted(self):
        status, row = capture_reminder(
            CONF,
            "nanti jam 9 ingetin aku bilang kaya tadi lagi ya",
            now=NOW,
            tz=JKT,
            recent_messages=AMBIGUOUS_TURNS,
        )
        self.assertEqual(status, "clarify")
        self.assertIsNone(row)
        self.assertEqual(load_character_state(CONF).future_intentions, [])

    def test_same_request_is_not_stored_twice(self):
        record_future_intention(
            CONF, "ingetin aku jam 15 minum obat", now=NOW, tz=JKT
        )
        record_future_intention(
            CONF, "ingetin aku jam 15 minum obat", now=NOW, tz=JKT
        )
        self.assertEqual(len(load_character_state(CONF).future_intentions), 1)

    def test_completed_reminder_leaves_active_list(self):
        status, row = capture_reminder(
            CONF, "ingetin aku jam 15 minum obat", now=NOW, tz=JKT
        )
        self.assertEqual(status, "captured")
        self.assertTrue(complete_future_intention(CONF, row["id"]))
        stored = load_character_state(CONF).future_intentions
        self.assertEqual(pending_intentions(stored), [])
        # Historical row is retained, not deleted.
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["status"], "done")

    def test_failed_write_is_reported_as_failed(self):
        with patch(
            "src.open_llm_vtuber.character_state.save_character_state",
            return_value=False,
        ):
            status, row = capture_reminder(
                CONF, "ingetin aku jam 15 minum obat", now=NOW, tz=JKT
            )
        self.assertEqual(status, "failed")
        self.assertIsNone(row)
        self.assertEqual(load_character_state(CONF).future_intentions, [])

    def test_capture_cannot_clobber_a_concurrent_claim(self):
        """Capture's load-modify-save must share the claim/completion lock."""
        state = load_character_state(CONF)
        state.future_intentions = [
            {
                "id": "existing-reminder",
                "text": "ingetin aku minum obat",
                "due_at": (NOW - timedelta(minutes=1)).isoformat(),
                "created_at": NOW.isoformat(),
                "status": "pending",
                "source": "conversation",
                "kind": "reminder",
            }
        ]
        self.assertTrue(save_character_state(CONF, state))

        from src.open_llm_vtuber.future_intentions import (
            add_future_intention as original_add,
        )

        entered = threading.Event()
        release = threading.Event()

        def paused_add(existing, detected, **kwargs):
            result = original_add(existing, detected, **kwargs)
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release capture")
            return result

        capture_result = {}
        claim_result = {}

        def run_capture():
            with patch(
                "src.open_llm_vtuber.future_intentions.add_future_intention",
                side_effect=paused_add,
            ):
                capture_result["value"] = capture_reminder(
                    CONF, "ingetin aku jam 15 minum obat", now=NOW, tz=JKT
                )

        def run_claim():
            claim_result["rows"] = claim_due_future_intentions(
                CONF, owner="owner-a", now=NOW
            )

        capture_thread = threading.Thread(target=run_capture, daemon=True)
        claim_thread = threading.Thread(target=run_claim, daemon=True)
        capture_thread.start()
        try:
            self.assertTrue(
                entered.wait(5), "capture did not reach its save section"
            )
            claim_thread.start()
            # Give the claim thread time to hit the same state lock. Without
            # the lock it would finish here and be overwritten by the paused
            # capture snapshot.
            time.sleep(0.2)
        finally:
            release.set()
        capture_thread.join(5)
        claim_thread.join(5)
        self.assertFalse(capture_thread.is_alive())
        self.assertFalse(claim_thread.is_alive())

        status, row = capture_result["value"]
        self.assertEqual(status, "captured")
        self.assertIsNotNone(row)
        self.assertEqual(
            [item["id"] for item in claim_result["rows"]],
            ["existing-reminder"],
        )
        stored = {
            item["id"]: item
            for item in load_character_state(CONF).future_intentions
        }
        self.assertEqual(stored["existing-reminder"]["claimed_by"], "owner-a")
        self.assertEqual(stored[row["id"]]["status"], "pending")

    def test_capture_cannot_resurrect_a_concurrent_completion(self):
        """A capture snapshot must not undo a completion written by another thread."""
        state = load_character_state(CONF)
        state.future_intentions = [
            {
                "id": "existing-reminder",
                "text": "ingetin aku minum obat",
                "due_at": (NOW - timedelta(minutes=1)).isoformat(),
                "created_at": NOW.isoformat(),
                "status": "pending",
                "source": "conversation",
                "kind": "reminder",
            }
        ]
        self.assertTrue(save_character_state(CONF, state))
        claimed = claim_due_future_intentions(
            CONF, owner="owner-a", now=NOW
        )
        self.assertEqual([item["id"] for item in claimed], ["existing-reminder"])

        from src.open_llm_vtuber.future_intentions import (
            add_future_intention as original_add,
        )

        entered = threading.Event()
        release = threading.Event()

        def paused_add(existing, detected, **kwargs):
            result = original_add(existing, detected, **kwargs)
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release capture")
            return result

        capture_result = {}
        completion_result = {}

        def run_capture():
            with patch(
                "src.open_llm_vtuber.future_intentions.add_future_intention",
                side_effect=paused_add,
            ):
                capture_result["value"] = capture_reminder(
                    CONF, "ingetin aku jam 15 minum obat", now=NOW, tz=JKT
                )

        def run_completion():
            completion_result["count"] = complete_claimed_future_intentions(
                CONF,
                owner="owner-a",
                intention_ids=["existing-reminder"],
            )

        capture_thread = threading.Thread(target=run_capture, daemon=True)
        completion_thread = threading.Thread(target=run_completion, daemon=True)
        capture_thread.start()
        try:
            self.assertTrue(
                entered.wait(5), "capture did not reach its save section"
            )
            completion_thread.start()
            time.sleep(0.2)
        finally:
            release.set()
        capture_thread.join(5)
        completion_thread.join(5)
        self.assertFalse(capture_thread.is_alive())
        self.assertFalse(completion_thread.is_alive())

        status, row = capture_result["value"]
        self.assertEqual(status, "captured")
        self.assertIsNotNone(row)
        self.assertEqual(completion_result["count"], 1)
        stored = {
            item["id"]: item
            for item in load_character_state(CONF).future_intentions
        }
        self.assertEqual(stored["existing-reminder"]["status"], "done")
        self.assertNotIn("claimed_by", stored["existing-reminder"])
        self.assertEqual(stored[row["id"]]["status"], "pending")


class ReceiptTest(unittest.TestCase):
    def test_captured_block_allows_confirmation(self):
        row = {
            "text": "Nanti jam 11.54 bilang kaya tadi lagi ya.",
            "due_at": "2026-10-21T04:54:00+00:00",
            "delivery_text": "TES PENGINGAT BERHASIL",
        }
        block = build_reminder_receipt_block("captured", row, tz=JKT)
        self.assertIn("IS stored", block)
        self.assertIn("11:54", block)
        self.assertIn("TES PENGINGAT BERHASIL", block)

    def test_captured_without_due_does_not_promise_delivery(self):
        row = {
            "text": "ingetin aku minum obat",
            "due_at": None,
            "delivery_text": "minum obat",
        }
        block = build_reminder_receipt_block("captured", row, tz=JKT)
        self.assertIn("IS stored", block)
        self.assertIn("NO valid due time", block)
        self.assertIn("NOT be delivered automatically", block)
        self.assertIn("Do NOT claim it is scheduled", block)
        self.assertNotIn("due 20", block)
        self.assertNotIn("deliver it at that time", block)

    def test_unparseable_due_is_not_reported_as_scheduled(self):
        row = {"text": "ingetin aku minum obat", "due_at": "besok pagi"}
        block = build_reminder_receipt_block("captured", row, tz=JKT)
        self.assertIn("NO valid due time", block)
        self.assertNotIn("deliver it at that time", block)

    def test_captured_with_due_allows_confirmation(self):
        row = {"text": "ingetin aku minum obat", "due_at": "2026-10-21T04:54:00+00:00"}
        block = build_reminder_receipt_block("captured", row, tz=JKT)
        self.assertIn("due 2026-10-21 11:54", block)
        self.assertIn("deliver it at that time", block)
        self.assertNotIn("NO valid due time", block)

    def test_clarify_block_forbids_claiming_success(self):
        block = build_reminder_receipt_block("clarify")
        self.assertIn("NOTHING was stored", block)
        self.assertIn("clarifying question", block)

    def test_failed_block_forbids_claiming_success(self):
        block = build_reminder_receipt_block("failed")
        self.assertIn("NOT stored", block)

    def test_no_request_renders_empty(self):
        self.assertEqual(build_reminder_receipt_block("none"), "")

    def test_capture_exception_becomes_failed_receipt(self):
        def boom():
            raise RuntimeError("disk full")

        block = process_reminder_turn("ingetin aku jam 5", capture_fn=boom)
        self.assertIn("FAILED", block)


if __name__ == "__main__":
    unittest.main()
