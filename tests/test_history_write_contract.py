"""The measured write/ordering contract behind persist-before-send.

Stage 4.2 measured what the Stage 4.1 guard actually costs: one full
read-modify-write of the transcript, with ``fsync``, per content frame
(``tests/bench_history_persist.py``). The decision was to keep it, because the
cost is a few milliseconds at the transcript sizes this deployment produces and
every cheaper design moves risk onto the read path or delays durability.

These tests pin the two properties that decision rests on, so a later "make it
faster" change cannot quietly buy speed by weakening them:

* exactly one durable write per content frame -- no coalescing, no debounce, no
  batching that would let the second frame out before its own row exists;
* the row on DISK already contains the text being sent, read back through the
  public store (not through the guard's own bookkeeping).

They assert counts and content, never wall-clock time: timing belongs in the
benchmark, where it can vary without failing a build.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import src.open_llm_vtuber.chat_history_manager as history_store
from src.open_llm_vtuber.chat_history_manager import get_history, load_message_snapshot
from src.open_llm_vtuber.conversations.single_conversation import _DurableAIContent


class DurableWriteContractTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._previous_cwd = os.getcwd()
        self._workdir = tempfile.TemporaryDirectory(prefix="mili-write-contract-")
        os.chdir(self._workdir.name)
        self.conf_uid = "mili_conf"
        self.history_uid = "bench-history"
        conf_dir = os.path.join("chat_history", self.conf_uid)
        os.makedirs(conf_dir, exist_ok=True)
        self.path = os.path.join(conf_dir, f"{self.history_uid}.json")
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(
                [{"role": "metadata", "timestamp": "2026-10-28T00:00:00+00:00"}],
                handle,
            )

        self.writes: list[str] = []
        real_write = history_store._write_history_atomic

        def counting_write(filepath, history_data):
            self.writes.append(filepath)
            return real_write(filepath, history_data)

        self._patcher = patch.object(
            history_store, "_write_history_atomic", counting_write
        )
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        os.chdir(self._previous_cwd)
        self._workdir.cleanup()

    def _guard(self, msg_id: str = "msg-contract") -> _DurableAIContent:
        return _DurableAIContent(
            enabled=True,
            conf_uid=self.conf_uid,
            history_uid=self.history_uid,
            msg_id=msg_id,
            name="Mili",
            avatar="mili.png",
        )

    def _durable_rows(self):
        return get_history(self.conf_uid, self.history_uid)

    async def test_one_durable_write_per_content_frame(self):
        frames = ["Satu. ", "Dua. ", "Tiga. ", "Empat. ", "Lima. "]
        guard = self._guard()
        expected = ""
        for frame in frames:
            self.assertTrue(await guard.before_content(frame))
            expected += frame
            rows = self._durable_rows()
            # One logical message: the row is upserted by msg_id, never appended
            # a second time, and it holds every accepted frame plus this one.
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["msg_id"], "msg-contract")
            self.assertEqual(rows[0]["content"], expected)
            await guard.after_content(frame, True)
        self.assertEqual(len(self.writes), len(frames))

    async def test_finalize_does_not_rewrite_an_already_durable_row(self):
        frames = ["Satu. ", "Dua. ", "Tiga. "]
        guard = self._guard()
        for frame in frames:
            self.assertTrue(await guard.before_content(frame))
            await guard.after_content(frame, True)
        writes_before_final = len(self.writes)

        stored, text = await guard.finalize("".join(frames))

        self.assertTrue(stored)
        self.assertEqual(text, "".join(frames))
        # The accepted frames already put exactly this text on disk, so the
        # turn end costs no extra write. This is why a turn is N writes, not
        # N+1, and why 4.1 did not add a second end-of-turn rewrite.
        self.assertEqual(len(self.writes), writes_before_final)
        self.assertEqual(self._durable_rows()[0]["content"], "".join(frames))

    async def test_refused_frame_leaves_the_accepted_prefix_on_disk(self):
        guard = self._guard()
        self.assertTrue(await guard.before_content("Satu. "))
        await guard.after_content("Satu. ", True)
        self.assertTrue(await guard.before_content("Dua. "))
        # Persistence necessarily happens first; the transport then refuses the
        # frame, so the row must be trimmed back to what really left.
        await guard.after_content("Dua. ", False)

        rows = self._durable_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "Satu. ")
        self.assertFalse(guard.failed)

        stored, text = await guard.finalize("Satu. ")
        self.assertTrue(stored)
        self.assertEqual(text, "Satu. ")
        self.assertEqual(self._durable_rows()[0]["content"], "Satu. ")

    async def test_abort_after_cancel_keeps_only_the_accepted_prefix(self):
        """An interrupted turn must not leave text the transport never took.

        ``before_content`` persists first, so at the moment of interruption the
        row can already carry a frame the transport has not accepted yet. The
        cancellation path calls ``abort()``, and that rollback has to be the
        last word: the row may not keep the unconfirmed frame.

        This is the invariant that rules out moving the store off the event
        loop as a latency quick-fix (measured in ``bench_loop_blocking.py``):
        once the write runs on a worker thread, cancellation can land in the
        middle of it, the orphaned write finishes after the rollback and the
        unconfirmed sentence survives on disk.
        """
        guard = self._guard("msg-cancel")
        self.assertTrue(await guard.before_content("Satu. "))
        await guard.after_content("Satu. ", True)
        # Second frame is durable (necessarily first), transport never confirms
        # it: the user interrupts here.
        self.assertTrue(await guard.before_content("Dua. "))
        await guard.abort()

        rows = self._durable_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "Satu. ")
        self.assertFalse(guard.failed)
        # Reading again (what a reload does) must see the same rollback.
        self.assertEqual(self._durable_rows()[0]["content"], "Satu. ")

    async def test_abort_before_any_accepted_frame_removes_the_row(self):
        guard = self._guard("msg-cancel-empty")
        self.assertTrue(await guard.before_content("Satu. "))
        await guard.abort()

        # Nothing ever left the process, so the transcript gains no row -- the
        # provisional one written for the frame is removed, not kept as an
        # empty assistant turn.
        self.assertEqual(self._durable_rows(), [])

    async def test_legacy_rows_without_msg_id_stay_intact(self):
        legacy = {
            "role": "human",
            "timestamp": "2026-10-01T00:00:00+00:00",
            "content": "pesan lama tanpa identitas",
        }
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data.append(legacy)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

        # A legacy row can never be selected by identity ...
        readable, snapshot = load_message_snapshot(
            self.conf_uid, self.history_uid, "human", "legacy"
        )
        self.assertTrue(readable)
        self.assertIsNone(snapshot)

        # ... and appending a guarded AI row leaves it untouched and in order.
        guard = self._guard()
        self.assertTrue(await guard.before_content("Halo lagi. "))
        await guard.after_content("Halo lagi. ", True)
        rows = self._durable_rows()
        self.assertEqual([row["role"] for row in rows], ["human", "ai"])
        self.assertEqual(rows[0], legacy)
        self.assertEqual(rows[1]["content"], "Halo lagi. ")


if __name__ == "__main__":
    unittest.main()
