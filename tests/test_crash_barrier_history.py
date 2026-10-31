"""Stage 4.1 crash barriers: the AI row is durable before the first frame.

The window this pins is not reachable in-process: the AI text (and its audio)
used to reach the transport BEFORE the history row was durable, so a process
death in that window left a reply nobody could replay after a reload -- the
bubble existed on screen, the canonical row did not exist at all.

Every test here runs the REAL turn (``process_single_conversation``, the real
ordered TTS sender, the real persist-before-send guard, the real history store
and its atomic ``os.replace``) inside a CHILD PROCESS, and the child dies with
``os._exit`` exactly at the boundary under test -- no ``finally``, no event-loop
shutdown, no flush. The parent then inspects the on-disk transcript, which is
the only thing that survives a real crash.

Scenarios (A-G of the Stage 4.1 brief):

A  row persisted, process dies before the first content frame   -> row exists
B  persist fails                                                 -> nothing sent,
                                                                     nothing claimed
C  first frame sent, process dies before final                   -> only the
                                                                     accepted prefix
D  restart + retry with the same msg_id                          -> one row
E  multi-segment audio                                           -> one order,
                                                                     one row
F  repeated final / repeated retry                               -> no duplicate
G  interrupt and error paths                                     -> no false final,
                                                                     no unproduced text

Explicitly not claimed anywhere: the transport is not transactional. A crash
between the persist and the send can leave one frame of text in the row that
the user never heard. That is the deliberate side of persist-before-send; the
tests below assert the row never claims MORE than one frame ahead, and never
claims text from a refused frame in a live turn.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid

from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHILD = os.path.join(ROOT, "tests", "crash_barrier_child.py")
CONF = "mili-crash-barrier"
SENTENCES = ["Aku ingetin minum obat ya. ", "Jam delapan malam ya. "]
FULL_TEXT = "".join(SENTENCES)
CRASH_EXIT = 9


class _CrashCase(unittest.TestCase):
    """One turn per child process, one cwd per test (both stores are local)."""

    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.history_uid = create_new_history(CONF)
        self.assertTrue(self.history_uid)
        self.msg_id = uuid.uuid4().hex
        self.frames_path = os.path.join(self._temp.name, "frames.jsonl")
        self.result_path = os.path.join(self._temp.name, "result.json")

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    # -- child process ------------------------------------------------------

    def run_turn(self, mode, sentences=None, msg_id=None, expect=(0,)):
        argv = [
            sys.executable,
            CHILD,
            mode,
            CONF,
            self.history_uid,
            msg_id or self.msg_id,
            self.frames_path,
            self.result_path,
            json.dumps(list(SENTENCES if sentences is None else sentences)),
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = ROOT
        process = subprocess.run(
            argv,
            cwd=os.getcwd(),
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertIn(
            process.returncode,
            expect,
            msg=(
                f"child exited {process.returncode}, expected one of {expect}\n"
                f"--- stderr ---\n{process.stderr[-4000:]}"
            ),
        )
        return process

    # -- what survived ------------------------------------------------------

    def frames(self):
        if not os.path.exists(self.frames_path):
            return []
        with open(self.frames_path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def content_frames(self):
        return [frame for frame in self.frames() if frame.get("type") == "audio"]

    def frames_of_type(self, frame_type):
        return [frame for frame in self.frames() if frame.get("type") == frame_type]

    def delivered_text(self):
        return "".join(
            str((frame.get("display_text") or {}).get("text") or "")
            for frame in self.content_frames()
        )

    def result(self):
        with open(self.result_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def rows(self):
        return get_history(CONF, self.history_uid)

    def ai_rows(self):
        return [row for row in self.rows() if row.get("role") == "ai"]

    def persist_errors(self):
        return [
            frame
            for frame in self.frames_of_type("error")
            if frame.get("code") == "history-persist-failed"
        ]


# --------------------------------------------------------------------------- #
# A. Durable row, process dies before the first content frame
# --------------------------------------------------------------------------- #


class PersistBeforeFirstFrameTest(_CrashCase):
    def test_row_survives_a_crash_before_the_first_frame(self):
        self.run_turn("crash_before_first", expect=(CRASH_EXIT,))

        # Nothing reached the transport...
        self.assertEqual(self.content_frames(), [])
        self.assertEqual(self.frames_of_type("ai-final"), [])
        # ...but the canonical text a reload renders is already on disk.
        rows = self.ai_rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["content"], SENTENCES[0])
        self.assertEqual(rows[0]["msg_id"], self.msg_id)
        # The user's line was stored first, in order.
        self.assertEqual([row["role"] for row in self.rows()], ["human", "ai"])

    def test_crash_before_first_frame_is_ahead_of_at_most_one_frame(self):
        # Guard against the opposite failure mode: persisting the WHOLE reply
        # before sending anything would also "survive" this crash but would
        # claim text the user never received. The row may only carry the frame
        # whose send was already under way.
        self.run_turn("crash_before_first", expect=(CRASH_EXIT,))
        rows = self.ai_rows()
        self.assertEqual(len(rows), 1)
        self.assertNotIn(SENTENCES[1], rows[0]["content"])


# --------------------------------------------------------------------------- #
# B. Persistence fails: nothing is sent, nothing is claimed
# --------------------------------------------------------------------------- #


class PersistFailureTest(_CrashCase):
    def test_failed_row_withholds_the_frame_and_the_final(self):
        self.run_turn("persist_fail", expect=(0,))

        self.assertEqual(self.content_frames(), [], "no unrecorded content")
        self.assertEqual(self.ai_rows(), [], "no row may be invented")
        self.assertEqual(self.frames_of_type("ai-final"), [])
        errors = self.persist_errors()
        self.assertEqual(len(errors), 1, self.frames())
        self.assertIn("could not be saved", errors[0]["message"])
        # The turn still closed its lifecycle: no hang, no silent stop.
        self.assertEqual(
            [
                frame
                for frame in self.frames_of_type("control")
                if frame.get("text") == "conversation-chain-end"
            ],
            [{"type": "control", "text": "conversation-chain-end"}],
        )
        self.assertEqual(self.result()["returned"], FULL_TEXT)

    def test_no_temp_file_survives_the_failed_write(self):
        self.run_turn("persist_fail", expect=(0,))
        directory = os.path.dirname(
            os.path.join("chat_history", CONF, f"{self.history_uid}.json")
        )
        self.assertEqual(
            [name for name in os.listdir(directory) if name.endswith(".tmp")], []
        )


# --------------------------------------------------------------------------- #
# C. First frame sent, crash before the final frame
# --------------------------------------------------------------------------- #


class CrashAfterFirstFrameTest(_CrashCase):
    def test_only_the_delivered_prefix_is_canonical(self):
        self.run_turn("crash_after_first", expect=(CRASH_EXIT,))

        self.assertEqual(
            [(frame["display_text"]["text"]) for frame in self.content_frames()],
            [SENTENCES[0]],
        )
        rows = self.ai_rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["content"], SENTENCES[0])
        self.assertNotIn(SENTENCES[1], rows[0]["content"])
        self.assertEqual(self.frames_of_type("ai-final"), [], "no false final")


# --------------------------------------------------------------------------- #
# D. Restart after a crash: same msg_id, one canonical row
# --------------------------------------------------------------------------- #


class RestartRetryTest(_CrashCase):
    def test_retry_after_restart_reuses_the_id_and_writes_one_row(self):
        self.run_turn("crash_after_first", expect=(CRASH_EXIT,))
        crashed = self.ai_rows()
        self.assertEqual(len(crashed), 1)
        first_timestamp = crashed[0]["timestamp"]

        # "Restart": a brand-new process, the same logical message, a new
        # attempt. Nothing of the crashed process is reused in memory.
        self.run_turn("complete")

        rows = self.ai_rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["msg_id"], self.msg_id)
        self.assertEqual(rows[0]["content"], FULL_TEXT)
        self.assertEqual(rows[0]["timestamp"], first_timestamp)
        finals = self.frames_of_type("ai-final")
        # The crash left no final; the retry closes the message exactly once.
        self.assertEqual(len(finals), 1, finals)
        self.assertEqual(finals[0]["msg_id"], self.msg_id)
        self.assertEqual(finals[0]["text"], FULL_TEXT)


# --------------------------------------------------------------------------- #
# E. Multi-segment audio: order preserved, one logical row
# --------------------------------------------------------------------------- #


class MultiSegmentTest(_CrashCase):
    def test_multi_segment_turn_keeps_order_and_one_row(self):
        self.run_turn("complete")

        self.assertEqual(
            [frame["display_text"]["text"] for frame in self.content_frames()],
            SENTENCES,
            "streaming/audio order must be untouched",
        )
        self.assertEqual({frame["msg_id"] for frame in self.content_frames()}, {self.msg_id})
        rows = self.ai_rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["content"], FULL_TEXT)
        self.assertEqual(len(self.frames_of_type("ai-final")), 1)


# --------------------------------------------------------------------------- #
# F. Repeated final / repeated retry: no duplication
# --------------------------------------------------------------------------- #


class NoDuplicateTest(_CrashCase):
    def test_second_final_for_the_same_message_does_not_duplicate(self):
        self.run_turn("complete")
        first = self.ai_rows()
        self.assertEqual(len(first), 1)

        # A redelivery of the same logical message (reminder retry, resync,
        # reconnect): same msg_id, another complete turn.
        self.run_turn("complete")

        rows = self.ai_rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["msg_id"], self.msg_id)
        self.assertEqual(rows[0]["timestamp"], first[0]["timestamp"])
        self.assertEqual(rows[0]["content"], FULL_TEXT)
        # Each attempt still closes with its own final frame for the SAME
        # message; the transcript, however, holds exactly one entry.
        finals = self.frames_of_type("ai-final")
        self.assertEqual(len(finals), 2, finals)
        self.assertEqual({frame["msg_id"] for frame in finals}, {self.msg_id})
        self.assertEqual({frame["text"] for frame in finals}, {FULL_TEXT})


# --------------------------------------------------------------------------- #
# G. Interrupt and error paths: no false final, no unproduced text
# --------------------------------------------------------------------------- #


class InterruptAndErrorTest(_CrashCase):
    def test_interrupt_never_claims_the_ungenerated_sentence(self):
        self.run_turn("interrupt")

        self.assertEqual(self.result()["error"], "CancelledError")
        self.assertEqual(self.frames_of_type("ai-final"), [], "no false final")
        rows = self.ai_rows()
        self.assertLessEqual(len(rows), 1)
        accepted = self.delivered_text()
        for row in rows:
            self.assertNotIn(SENTENCES[1], row["content"])
            self.assertEqual(row["content"], accepted)

    def test_provider_error_keeps_the_prefix_and_suppresses_the_final(self):
        self.run_turn("provider_error")

        self.assertEqual(self.result()["returned"], "")
        self.assertEqual(self.frames_of_type("ai-final"), [], "no false final")
        rows = self.ai_rows()
        self.assertLessEqual(len(rows), 1)
        accepted = self.delivered_text()
        for row in rows:
            self.assertEqual(row["content"], accepted)
            self.assertNotIn(SENTENCES[1], row["content"])
        # The failure is always reported, never silently swallowed.
        self.assertTrue(
            any(
                "provider" in str(frame.get("message", "")).lower()
                for frame in self.frames_of_type("error")
            ),
            self.frames(),
        )

    def test_stream_error_finalizes_only_the_text_that_was_delivered(self):
        self.run_turn("raise_error")

        accepted = self.delivered_text()
        self.assertTrue(accepted)
        rows = self.ai_rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["content"], accepted)
        self.assertNotIn(SENTENCES[1], rows[0]["content"])
        finals = self.frames_of_type("ai-final")
        self.assertEqual(len(finals), 1, finals)
        self.assertEqual(finals[0]["text"], accepted)


if __name__ == "__main__":
    unittest.main()
