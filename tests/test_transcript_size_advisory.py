"""Transcript size advisory: a signal, never a mutation.

Stage 4.4 asked whether one conversation's transcript needs a size bound. The
audit found that every append rewrites the whole file with ``fsync`` while the
pre-send guard holds the event loop, so per-frame cost follows total rows; but
the transcript is additive history: removing rows would invalidate the
positional ``summary_through_message_index`` boundary, turn a msg_id upsert of
an older row into a duplicate append, and hide rows from the UI, the Memory
Manager and the backup readers. So no rotation/compaction shipped.

What shipped instead is one advisory log line at a measured threshold: it
touches no row, no file and no return value. These tests pin that boundary --
when it fires, what it says, and everything it must leave exactly as it was.
The last test is a negative control: it proves the "exactly one line"
assertions are sensitive to the once-per-file guard rather than passing by
accident.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from loguru import logger

import src.open_llm_vtuber.chat_history_manager as history_store
from src.open_llm_vtuber.chat_history_manager import (
    TRANSCRIPT_SIZE_WARN_ROWS,
    get_history,
    store_message,
)

CONF = "sizechar"
HIST = "h-size-advisory"
MARKER = "Transcript size advisory:"


def _path() -> str:
    return os.path.join("chat_history", CONF, f"{HIST}.json")


def _seed(messages: int, *, include_metadata: bool = True) -> None:
    """Write a transcript of ``messages`` rows directly (no per-row fsync)."""
    rows = []
    if include_metadata:
        rows.append({"role": "metadata", "timestamp": "2026-10-28T00:00:00+00:00"})
    for index in range(messages):
        rows.append(
            {
                "role": "human" if index % 2 == 0 else "ai",
                "timestamp": "2026-10-28T00:00:01+00:00",
                "content": f"pesan-{index}",
                "msg_id": f"seed-{index}",
            }
        )
    with open(_path(), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


class TranscriptSizeAdvisoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mili-size-advisory-")
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs(os.path.join("chat_history", CONF), exist_ok=True)
        # Module-level "already warned" state outlives one test method, exactly
        # like it outlives one session inside a running server: reset it so
        # each test starts from a fresh process' point of view.
        history_store._transcript_size_warned.clear()
        self.lines: list[str] = []
        self._handler = logger.add(
            lambda message: self.lines.append(str(message)),
            format="{message}",
            level="WARNING",
        )

    def tearDown(self) -> None:
        logger.remove(self._handler)
        history_store._transcript_size_warned.clear()
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def _advisories(self) -> list[str]:
        return [line for line in self.lines if MARKER in line]

    # -- threshold boundary ------------------------------------------------

    def test_no_advisory_below_threshold_and_one_when_it_is_reached(self):
        _seed(TRANSCRIPT_SIZE_WARN_ROWS - 2)

        self.assertTrue(store_message(CONF, HIST, "ai", "satu lagi"))
        self.assertEqual(len(get_history(CONF, HIST)), TRANSCRIPT_SIZE_WARN_ROWS - 1)
        self.assertEqual(self._advisories(), [], "one row short: still silent")

        self.assertTrue(store_message(CONF, HIST, "human", "tepat ambang"))
        self.assertEqual(len(get_history(CONF, HIST)), TRANSCRIPT_SIZE_WARN_ROWS)
        advisories = self._advisories()
        self.assertEqual(len(advisories), 1)
        self.assertIn(f"rows={TRANSCRIPT_SIZE_WARN_ROWS}", advisories[0])
        self.assertIn(f"threshold={TRANSCRIPT_SIZE_WARN_ROWS}", advisories[0])
        self.assertIn(f"history_uid={HIST}", advisories[0])

    def test_advisory_names_no_message_content(self):
        _seed(TRANSCRIPT_SIZE_WARN_ROWS - 1)

        store_message(CONF, HIST, "ai", "isi rahasia yang tidak boleh dicatat")

        advisories = self._advisories()
        self.assertEqual(len(advisories), 1)
        self.assertNotIn("isi rahasia", advisories[0])
        self.assertNotIn("pesan-0", advisories[0])

    def test_metadata_row_is_not_counted_as_a_message(self):
        _seed(TRANSCRIPT_SIZE_WARN_ROWS - 1, include_metadata=True)

        store_message(CONF, HIST, "ai", "satu pesan terakhir")

        rows = get_history(CONF, HIST)
        self.assertEqual(len(rows), TRANSCRIPT_SIZE_WARN_ROWS)
        self.assertEqual(
            len(self._advisories()),
            1,
            "the metadata row must not push the file over the row threshold",
        )

    # -- once per transcript per process ----------------------------------

    def test_advisory_is_reported_once_per_transcript(self):
        _seed(TRANSCRIPT_SIZE_WARN_ROWS - 1)

        for index in range(4):
            self.assertTrue(store_message(CONF, HIST, "ai", f"tambahan-{index}"))

        self.assertEqual(len(get_history(CONF, HIST)), TRANSCRIPT_SIZE_WARN_ROWS + 3)
        self.assertEqual(len(self._advisories()), 1)

    def test_advisory_is_reported_again_after_a_restart(self):
        _seed(TRANSCRIPT_SIZE_WARN_ROWS - 1)
        store_message(CONF, HIST, "ai", "melewati ambang")
        self.assertEqual(len(self._advisories()), 1)

        # A restarted server has no memory of having said this.
        history_store._transcript_size_warned.clear()
        self.assertTrue(store_message(CONF, HIST, "human", "sesi berikutnya"))

        self.assertEqual(len(self._advisories()), 2)

    # -- nothing else changes ---------------------------------------------

    def test_advisory_does_not_change_what_is_stored(self):
        _seed(TRANSCRIPT_SIZE_WARN_ROWS - 1)
        before = get_history(CONF, HIST)

        self.assertTrue(store_message(CONF, HIST, "ai", "jawaban baru", msg_id="m-1"))

        after = get_history(CONF, HIST)
        self.assertEqual(len(after), len(before) + 1)
        self.assertEqual(after[: len(before)], before, "existing rows untouched")
        self.assertEqual(after[-1]["content"], "jawaban baru")
        self.assertEqual(after[-1]["msg_id"], "m-1")
        self.assertTrue(after[-1]["timestamp"])
        # Same metadata row, still first: readers depend on that position.
        with open(_path(), encoding="utf-8") as handle:
            raw = json.load(handle)
        self.assertEqual(raw[0]["role"], "metadata")

    def test_retry_upsert_of_an_existing_row_adds_no_row_and_no_advisory(self):
        _seed(TRANSCRIPT_SIZE_WARN_ROWS)
        with open(_path(), encoding="utf-8") as handle:
            raw = json.load(handle)
        raw.append(
            {
                "role": "ai",
                "timestamp": "2026-10-28T00:00:02+00:00",
                "content": "versi pertama",
                "msg_id": "retry-me",
            }
        )
        with open(_path(), "w", encoding="utf-8") as handle:
            json.dump(raw, handle, ensure_ascii=False, indent=2)

        self.assertTrue(
            store_message(CONF, HIST, "ai", "versi kedua", msg_id="retry-me")
        )

        rows = get_history(CONF, HIST)
        self.assertEqual(len(rows), TRANSCRIPT_SIZE_WARN_ROWS + 1)
        self.assertEqual([row["content"] for row in rows].count("versi kedua"), 1)
        self.assertEqual(
            rows[-1]["timestamp"],
            "2026-10-28T00:00:02+00:00",
            "an in-place update keeps the first write's position and stamp",
        )
        self.assertEqual(
            self._advisories(),
            [],
            "updating a row is not growth; the advisory belongs to appends",
        )

    def test_failed_write_reports_no_advisory(self):
        _seed(TRANSCRIPT_SIZE_WARN_ROWS - 1)
        with open(_path(), "w", encoding="utf-8") as handle:
            handle.write("{rusak")

        self.assertFalse(store_message(CONF, HIST, "ai", "tidak tersimpan"))

        self.assertEqual(
            self._advisories(),
            [],
            "a refused write must not report a size milestone it never reached",
        )

    # -- negative control --------------------------------------------------

    def test_negative_control_without_the_once_guard_logs_again(self):
        """Prove the 'exactly one line' assertions are sensitive to the guard.

        With the once-per-file registry bypassed (an implementation that logs
        on every append), the identical sequence reports twice. If it still
        reported once, the assertions above would be passing for the wrong
        reason -- they would not be testing the dedupe at all.
        """

        class _NeverRemembers(set):
            def __contains__(self, item):  # noqa: D105 - deliberately broken
                return False

            def add(self, item):  # noqa: D105 - deliberately broken
                return None

        _seed(TRANSCRIPT_SIZE_WARN_ROWS - 1)
        with patch.object(
            history_store, "_transcript_size_warned", _NeverRemembers()
        ):
            store_message(CONF, HIST, "ai", "melewati ambang")
            store_message(CONF, HIST, "human", "dan lagi")

        self.assertEqual(
            len(self._advisories()),
            2,
            "the detector must notice duplicate advisories",
        )


if __name__ == "__main__":
    unittest.main()
