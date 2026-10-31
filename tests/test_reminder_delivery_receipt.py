"""Durable reminder delivery receipts — duplicate-send protection.

The boundary under test is between a *delivered* proactive turn and the
state completion write:

* failed delivery: no receipt is written, the reminder stays retryable;
* delivered + completion save failure: the durable receipt makes the next
  claim reconcile the row to ``done`` instead of sending it again, even after
  the claim lease expires;
* crash/restart after the receipt but before completion: same protection,
  driven only by on-disk state;
* crash before the receipt: documented at-least-once window (no receipt, so a
  resend is possible) — this module does NOT claim exactly-once delivery;
* two dispatchers: the claim still admits exactly one owner.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from src.open_llm_vtuber.character_state import (
    claim_due_future_intentions,
    complete_claimed_future_intentions,
    load_character_state,
    release_claimed_future_intentions,
    renew_claimed_future_intentions,
    save_character_state,
)
from src.open_llm_vtuber.future_intentions import (
    STATUS_DONE,
    STATUS_PENDING,
    complete_delivered_intentions,
    due_intentions,
)
from src.open_llm_vtuber.reminder_delivery import (
    delivered_ids,
    get_delivery_journal_path,
    record_delivery,
)

CONF = "receipt-char"
NOW = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)


class _Base(unittest.TestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    def _seed(self, due_delta=-3600, text="ingetin aku minum obat ya"):
        state = load_character_state(CONF)
        row = {
            "id": uuid.uuid4().hex,
            "text": text,
            "due_at": (NOW + timedelta(seconds=due_delta)).isoformat(),
            "created_at": (NOW - timedelta(days=1)).isoformat(),
            "status": STATUS_PENDING,
            "source": "conversation",
            "kind": "reminder",
            "tz": "Asia/Jakarta",
        }
        state.future_intentions = [*(state.future_intentions or []), row]
        self.assertTrue(save_character_state(CONF, state))
        return row

    def _row(self, intention_id):
        return next(
            row
            for row in load_character_state(CONF).future_intentions
            if row.get("id") == intention_id
        )


class ReceiptJournalTest(_Base):
    def test_recorded_receipt_is_durable_and_deduplicated(self):
        self.assertTrue(record_delivery(CONF, ["row-1", "row-1"], owner="o", now=NOW))
        self.assertEqual(delivered_ids(CONF), {"row-1"})
        path = get_delivery_journal_path(CONF)
        with open(path, "r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["id"], "row-1")
        self.assertEqual(entry["owner"], "o")
        self.assertTrue(entry["delivered_at"].startswith("2026-10-20"))

    def test_empty_receipt_is_not_written(self):
        self.assertFalse(record_delivery(CONF, [], owner="o", now=NOW))
        self.assertEqual(delivered_ids(CONF), set())

    def test_corrupt_lines_are_ignored(self):
        self.assertTrue(record_delivery(CONF, ["row-1"], owner="o", now=NOW))
        with open(get_delivery_journal_path(CONF), "a", encoding="utf-8") as handle:
            handle.write("{not json\n")
            handle.write("\n")
        self.assertEqual(delivered_ids(CONF), {"row-1"})

    def test_receipt_failure_is_reported_not_raised(self):
        with patch(
            "src.open_llm_vtuber.reminder_delivery.os.open",
            side_effect=OSError("read-only storage"),
        ):
            self.assertFalse(record_delivery(CONF, ["row-1"], owner="o", now=NOW))
        self.assertEqual(delivered_ids(CONF), set())


class DeliveryAttemptTest(_Base):
    def test_failed_delivery_writes_no_receipt_and_allows_retry(self):
        row = self._seed()
        claimed = claim_due_future_intentions(CONF, owner="owner-a", now=NOW)
        self.assertEqual([item["id"] for item in claimed], [row["id"]])
        # The dispatcher releases an empty/failed turn and writes no receipt.
        self.assertEqual(
            release_claimed_future_intentions(
                CONF, owner="owner-a", intention_ids=[row["id"]]
            ),
            1,
        )
        self.assertEqual(delivered_ids(CONF), set())
        retry = claim_due_future_intentions(CONF, owner="owner-b", now=NOW)
        self.assertEqual([item["id"] for item in retry], [row["id"]])

    def test_delivered_receipt_blocks_resend_after_completion_failure(self):
        row = self._seed()
        claimed = claim_due_future_intentions(CONF, owner="owner-a", now=NOW)
        self.assertEqual([item["id"] for item in claimed], [row["id"]])

        # Delivery succeeded, so the dispatcher records the receipt first...
        self.assertTrue(
            record_delivery(CONF, [row["id"]], owner="owner-a", now=NOW)
        )
        # ...and the completion write fails.
        with patch(
            "src.open_llm_vtuber.character_state.save_character_state",
            return_value=False,
        ):
            self.assertEqual(
                complete_claimed_future_intentions(
                    CONF, owner="owner-a", intention_ids=[row["id"]]
                ),
                0,
            )
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertEqual(stored["claimed_by"], "owner-a")

        # Lease expired: the next claim reconciles instead of resending.
        later = NOW + timedelta(seconds=700)
        self.assertEqual(
            claim_due_future_intentions(CONF, owner="owner-b", now=later), []
        )
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_DONE)
        self.assertNotIn("claimed_by", stored)
        self.assertEqual(due_intentions([stored], now=later), [])

    def test_restart_after_receipt_does_not_resend(self):
        """Crash between delivery and completion: receipt is the only state."""
        row = self._seed()
        claim_due_future_intentions(CONF, owner="owner-a", now=NOW)
        self.assertTrue(record_delivery(CONF, [row["id"]], owner="owner-a", now=NOW))

        # "Restart": only files on disk survive. A fresh claim must not send.
        later = NOW + timedelta(seconds=700)
        self.assertEqual(
            claim_due_future_intentions(CONF, owner="owner-restarted", now=later),
            [],
        )
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_DONE)
        self.assertEqual(delivered_ids(CONF), {row["id"]})

    def test_crash_before_receipt_is_at_least_once(self):
        """DOCUMENTED BOUNDARY: no receipt yet -> a resend is possible."""
        row = self._seed()
        claim_due_future_intentions(CONF, owner="owner-a", now=NOW)
        # No receipt: the process died between streaming and the append.
        self.assertEqual(delivered_ids(CONF), set())
        later = NOW + timedelta(seconds=700)
        again = claim_due_future_intentions(CONF, owner="owner-restarted", now=later)
        self.assertEqual([item["id"] for item in again], [row["id"]])

    def test_two_concurrent_dispatchers_claim_once(self):
        self._seed()
        results = {}
        barrier = threading.Barrier(2)

        def run(owner):
            barrier.wait(timeout=5)
            results[owner] = claim_due_future_intentions(
                CONF, owner=owner, now=NOW
            )

        threads = [
            threading.Thread(target=run, args=(owner,), daemon=True)
            for owner in ("owner-a", "owner-b")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())

        winners = [owner for owner, rows in results.items() if rows]
        self.assertEqual(len(winners), 1)
        stored = load_character_state(CONF).future_intentions[0]
        self.assertEqual(stored["claimed_by"], winners[0])

        # The loser cannot claim again while the lease is live, and the
        # receipt (if the winner delivered) blocks it after expiry too.
        self.assertTrue(
            record_delivery(CONF, [stored["id"]], owner=winners[0], now=NOW)
        )
        self.assertEqual(
            claim_due_future_intentions(
                CONF, owner="owner-b", now=NOW + timedelta(seconds=700)
            ),
            [],
        )


class ClaimRenewalTest(_Base):
    """Lease renewal is what keeps a live turn exclusive across processes."""

    def test_renewal_extends_only_the_owners_pending_row(self):
        row = self._seed()
        claimed = claim_due_future_intentions(
            CONF, owner="owner-a", now=NOW, lease_seconds=60
        )
        self.assertEqual([item["id"] for item in claimed], [row["id"]])
        original_until = datetime.fromisoformat(self._row(row["id"])["claim_until"])

        # Another owner cannot extend or read a lease as its own.
        self.assertEqual(
            renew_claimed_future_intentions(
                CONF,
                owner="owner-b",
                intention_ids=[row["id"]],
                now=NOW + timedelta(seconds=50),
                lease_seconds=60,
            ),
            0,
        )
        self.assertEqual(
            datetime.fromisoformat(self._row(row["id"])["claim_until"]),
            original_until,
        )

        self.assertEqual(
            renew_claimed_future_intentions(
                CONF,
                owner="owner-a",
                intention_ids=[row["id"]],
                now=NOW + timedelta(seconds=50),
                lease_seconds=60,
            ),
            1,
        )
        stored = self._row(row["id"])
        self.assertEqual(
            datetime.fromisoformat(stored["claim_until"]),
            NOW + timedelta(seconds=110),
        )
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertEqual(stored["claimed_by"], "owner-a")
        # Renewal must not prevent the real completion of the same claim.
        self.assertEqual(
            complete_claimed_future_intentions(
                CONF, owner="owner-a", intention_ids=[row["id"]]
            ),
            1,
        )

    def test_renewal_never_revives_a_done_or_released_row(self):
        row = self._seed()
        claim_due_future_intentions(CONF, owner="owner-a", now=NOW, lease_seconds=60)
        complete_claimed_future_intentions(
            CONF, owner="owner-a", intention_ids=[row["id"]]
        )
        self.assertEqual(
            renew_claimed_future_intentions(
                CONF,
                owner="owner-a",
                intention_ids=[row["id"]],
                now=NOW + timedelta(seconds=10),
            ),
            0,
        )
        self.assertEqual(self._row(row["id"])["status"], STATUS_DONE)


class ReconciliationTest(_Base):
    def test_complete_delivered_intentions_only_touches_pending_rows(self):
        delivered = {
            "id": "delivered",
            "text": "ingetin aku minum obat",
            "due_at": NOW.isoformat(),
            "created_at": NOW.isoformat(),
            "status": STATUS_PENDING,
            "kind": "reminder",
        }
        done = {
            "id": "already-done",
            "text": "x",
            "due_at": NOW.isoformat(),
            "created_at": NOW.isoformat(),
            "status": STATUS_DONE,
            "kind": "reminder",
        }
        untouched = {
            "id": "other",
            "text": "y",
            "due_at": NOW.isoformat(),
            "created_at": NOW.isoformat(),
            "status": STATUS_PENDING,
            "kind": "reminder",
        }
        rows, count = complete_delivered_intentions(
            [delivered, done, untouched], {"delivered", "already-done"}
        )
        self.assertEqual(count, 1)
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(by_id["delivered"]["status"], STATUS_DONE)
        self.assertEqual(by_id["already-done"]["status"], STATUS_DONE)
        self.assertEqual(by_id["other"]["status"], STATUS_PENDING)

    def test_receipt_for_done_row_does_not_block_a_new_pending_row(self):
        done = self._seed()
        claim_due_future_intentions(CONF, owner="owner-a", now=NOW)
        self.assertTrue(record_delivery(CONF, [done["id"]], owner="owner-a", now=NOW))
        self.assertEqual(
            complete_claimed_future_intentions(
                CONF, owner="owner-a", intention_ids=[done["id"]]
            ),
            1,
        )
        other = self._seed(text="ingetin aku minum air")
        claimed = claim_due_future_intentions(CONF, owner="owner-b", now=NOW)
        self.assertEqual([item["id"] for item in claimed], [other["id"]])


if __name__ == "__main__":
    unittest.main()
