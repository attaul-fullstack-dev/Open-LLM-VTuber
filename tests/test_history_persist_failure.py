"""Stage 1: a failed history write must never be reported as a success.

``store_message`` used to return ``None`` on every path, including the ones
where NOTHING was written: missing identifiers, an unreadable transcript, a
failed atomic write. ``process_single_conversation`` ignored that and always
logged ``TURN_PERSISTED`` and emitted ``ai-final`` -- the frame the frontend
treats as the authoritative persisted row. A reload then showed a transcript
that did not contain the reply the UI had just "finalized".

Policy pinned here (real ``process_single_conversation``, real history store,
real ``os.replace``; only the provider stream and the transport are faked):

* durable write -> ``store_message`` returns True, ``ai-final`` is emitted with
  the persisted text and identity (unchanged success behaviour);
* unreadable transcript (a real corrupt file) -> the send guard refuses the
  first content frame, no audio/text is sent, no ``ai-final``, an explicit
  ``error`` frame, and the file is left byte-identical;
* real ``OSError`` at the atomic-write boundary -> frames whose history could
  not be written are withheld; an already-accepted prefix remains as the
  durable row, no ``ai-final`` is emitted, and the turn still closes cleanly;
* a retry of the same logical message (same msg_id) then persists exactly one
  row -- never a duplicate, never a moved/rewritten first timestamp;
* transcripts written before msg_id existed stay compatible.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.open_llm_vtuber.agent.output_types import Actions, DisplayText, SentenceOutput
from src.open_llm_vtuber.chat_history_manager import (
    _write_history_atomic,
    create_new_history,
    get_history,
    store_message,
)
from src.open_llm_vtuber.conversations import conversation_utils
from src.open_llm_vtuber.conversations.single_conversation import (
    process_single_conversation,
)

CONF = "mili-persist-failure"
SENTENCES = ["Aku ingetin minum obat ya. ", "Jam delapan malam ya. "]
FULL_TEXT = "".join(SENTENCES)
CORRUPT = "{not valid json!!!"


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class _FakeAgent:
    def __init__(self, sentences=None):
        self.sentences = list(sentences or SENTENCES)
        self._llm = None

    async def chat(self, _batch_input):
        for text in self.sentences:
            yield SentenceOutput(
                display_text=DisplayText(text=text, name="Mili"),
                tts_text=text,
                actions=Actions(),
            )


class _CorruptingAgent(_FakeAgent):
    """Corrupts the transcript while the response is being produced.

    The write happens when the consumer asks for the item AFTER the last
    sentence, i.e. after the user message was stored and before the assistant
    row is written. Deterministic: no sleeps, no timing assumption.
    """

    def __init__(self, history_path, sentences=None):
        super().__init__(sentences)
        self.history_path = history_path

    async def chat(self, batch_input):
        last = len(self.sentences) - 1
        for index, text in enumerate(self.sentences):
            yield SentenceOutput(
                display_text=DisplayText(text=text, name="Mili"),
                tts_text=text,
                actions=Actions(),
            )
            if index == last:
                with open(self.history_path, "w", encoding="utf-8") as handle:
                    handle.write(CORRUPT)


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class _Transport:
    def __init__(self):
        self.frames = []

    async def __call__(self, payload):
        self.frames.append(json.loads(payload))

    def of_type(self, frame_type):
        return [frame for frame in self.frames if frame.get("type") == frame_type]

    @property
    def audio_frames(self):
        return self.of_type("audio")

    @property
    def canonical_finals(self):
        return self.of_type("ai-final")

    @property
    def persist_errors(self):
        return [
            frame
            for frame in self.of_type("error")
            if frame.get("code") == "history-persist-failed"
        ]

    @property
    def chain_ends(self):
        return [
            frame
            for frame in self.of_type("control")
            if frame.get("text") == "conversation-chain-end"
        ]


def _context(history_uid, sentences=None):
    return SimpleNamespace(
        history_uid=history_uid,
        agent_engine=_FakeAgent(sentences),
        user_timezone="Asia/Jakarta",
        asr_engine=None,
        live2d_model=_FakeLive2D(),
        tts_engine=None,
        translate_engine=None,
        voice_output_enabled=False,
        character_config=SimpleNamespace(
            conf_uid=CONF,
            human_name="User",
            character_name="Mili",
            avatar="mili.png",
        ),
    )


async def _finalize_after_flush(tts_manager=None, websocket_send=None, client_uid=None, **_):
    """Flush the ordered payload queue, then run the REAL lifecycle tail.

    ``queue.join()`` is deterministic (it completes only after the sender task
    called ``task_done``, i.e. after every queued frame was handed to the
    transport). The playback wait is mocked out because no browser is present.
    """
    queue = getattr(tts_manager, "_payload_queue", None)
    if queue is not None:
        await asyncio.wait_for(queue.join(), timeout=5)
    with patch.object(
        conversation_utils.message_handler,
        "wait_for_response",
        new=AsyncMock(return_value={"type": "frontend-playback-complete"}),
    ):
        await conversation_utils.finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )


async def _run_turn(transport, history_uid, metadata=None, context=None):
    context = context or _context(history_uid)
    with patch(
        "src.open_llm_vtuber.conversations.single_conversation."
        "finalize_conversation_turn",
        new=_finalize_after_flush,
    ):
        return await process_single_conversation(
            context=context,
            websocket_send=transport,
            client_uid="client",
            user_input="Halo Mili.",
            metadata=metadata or {},
        )


def _failing_replace_for(text: str):
    """os.replace that raises a REAL OSError when the row is ``text``."""
    original = os.replace

    def replace(src, dst):
        try:
            with open(src, encoding="utf-8") as handle:
                payload = handle.read()
        except OSError:
            payload = ""
        if text in payload:
            raise OSError(errno.ENOSPC, "No space left on device")
        return original(src, dst)

    return replace


class _Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.history_uid = create_new_history(CONF)
        self.assertTrue(self.history_uid)
        self.assertTrue(
            store_message(CONF, self.history_uid, "human", "Halo Mili.")
        )

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    @property
    def path(self):
        return os.path.join("chat_history", CONF, f"{self.history_uid}.json")

    def rows(self):
        return get_history(CONF, self.history_uid)

    def ai_rows(self):
        return [row for row in self.rows() if row["role"] == "ai"]

    def raw(self):
        with open(self.path, encoding="utf-8") as handle:
            return handle.read()

    def leftovers(self):
        directory = os.path.dirname(self.path)
        return [name for name in os.listdir(directory) if name.endswith(".tmp")]


# --------------------------------------------------------------------------- #
# A. Durable write: success behaviour is unchanged
# --------------------------------------------------------------------------- #


class SuccessfulPersistTest(_Base):
    async def test_success_still_emits_ai_final_with_persisted_text(self):
        transport = _Transport()
        returned = await _run_turn(transport, self.history_uid)

        self.assertEqual(returned, FULL_TEXT)
        self.assertEqual(len(transport.canonical_finals), 1)
        final = transport.canonical_finals[0]
        self.assertEqual(final["text"], FULL_TEXT)
        rows = self.ai_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["msg_id"], final["msg_id"])
        self.assertEqual(transport.persist_errors, [])
        # The lifecycle tail is unaffected by the persist check.
        self.assertEqual(len(transport.chain_ends), 1)

    async def test_multi_segment_turn_persists_once(self):
        transport = _Transport()
        await _run_turn(transport, self.history_uid)

        self.assertEqual(
            [frame["display_text"]["text"] for frame in transport.audio_frames],
            SENTENCES,
            "streaming/audio order must be untouched",
        )
        ids = {frame["msg_id"] for frame in transport.audio_frames}
        self.assertEqual(len(ids), 1, "one logical message, one identity")
        self.assertEqual(len(transport.canonical_finals), 1)
        rows = self.ai_rows()
        self.assertEqual(len(rows), 1, "several frames, one row")
        self.assertEqual(rows[0]["content"], FULL_TEXT)


# --------------------------------------------------------------------------- #
# B. Unreadable transcript: a real corrupt file, no false success
# --------------------------------------------------------------------------- #


class UnreadableTranscriptTest(_Base):
    async def test_corrupt_transcript_suppresses_ai_final(self):
        context = _context(self.history_uid)
        context.agent_engine = _CorruptingAgent(self.path)
        transport = _Transport()

        returned = await _run_turn(transport, self.history_uid, context=context)

        # The transcript is unreadable, so the first content frame is
        # withheld: no unrecorded audio/text may leave the process.
        self.assertEqual(returned, FULL_TEXT)
        self.assertEqual(len(transport.audio_frames), 0)
        # And nothing may claim it was stored.
        self.assertEqual(transport.canonical_finals, [])
        self.assertEqual(len(transport.persist_errors), 1)
        self.assertIn("could not be saved", transport.persist_errors[0]["message"])
        # The corrupt transcript is preserved byte for byte.
        self.assertEqual(self.raw(), CORRUPT)
        # And the turn closed its lifecycle normally.
        self.assertEqual(len(transport.chain_ends), 1)

    async def test_corrupt_from_the_start_reports_both_messages(self):
        # The user's own row cannot be written either. Both failures are
        # reported; neither is silently upgraded to success.
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(CORRUPT)
        transport = _Transport()
        await _run_turn(transport, self.history_uid)

        self.assertEqual(len(transport.persist_errors), 2)
        self.assertEqual(transport.canonical_finals, [])
        self.assertEqual(self.raw(), CORRUPT)


# --------------------------------------------------------------------------- #
# C. Real I/O failure at the atomic-write boundary
# --------------------------------------------------------------------------- #


class WriteErrorTest(_Base):
    async def test_oserror_withholds_the_unrecorded_frame_and_leaves_prefix(self):
        transport = _Transport()
        with patch("os.replace", new=_failing_replace_for(FULL_TEXT)):
            returned = await _run_turn(transport, self.history_uid)

        self.assertEqual(returned, FULL_TEXT, "the turn must not raise or hang")
        self.assertEqual(transport.canonical_finals, [])
        self.assertEqual(len(transport.persist_errors), 1)
        # The first sentence was written before it was sent, so it survives as
        # the accepted durable prefix. The second frame was withheld because
        # its own write failed; history may not claim it.
        self.assertEqual(
            [frame["display_text"]["text"] for frame in transport.audio_frames],
            [SENTENCES[0]],
        )
        rows = self.rows()
        self.assertEqual([row["role"] for row in rows], ["human", "human", "ai"])
        self.assertEqual([row["content"] for row in self.ai_rows()], [SENTENCES[0]])
        self.assertEqual(self.leftovers(), [], "no temp file may survive a failure")

    async def test_store_message_reports_failure_without_raising(self):
        with patch("os.replace", new=_failing_replace_for("tidak tersimpan")):
            result = store_message(
                CONF, self.history_uid, "ai", "tidak tersimpan", msg_id="m-fail"
            )
        self.assertIs(result, False)
        self.assertEqual(self.ai_rows(), [])
        self.assertEqual(self.leftovers(), [])


# --------------------------------------------------------------------------- #
# D. Retry after a failure: same msg_id, one row, no duplicate
# --------------------------------------------------------------------------- #


class RetryAfterFailureTest(_Base):
    async def test_retry_with_same_msg_id_persists_exactly_once(self):
        stable = uuid.uuid4().hex

        failed = _Transport()
        with patch("os.replace", new=_failing_replace_for(FULL_TEXT)):
            await _run_turn(failed, self.history_uid, metadata={"msg_id": stable})
        self.assertEqual(failed.canonical_finals, [])
        # The first frame was durably recorded before its send; only the
        # unrecorded second frame was withheld.
        rows = self.ai_rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["content"], SENTENCES[0])
        self.assertEqual(rows[0]["msg_id"], stable)

        retried = _Transport()
        await _run_turn(retried, self.history_uid, metadata={"msg_id": stable})

        rows = self.ai_rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["msg_id"], stable)
        self.assertEqual(rows[0]["content"], FULL_TEXT)
        self.assertEqual(len(retried.canonical_finals), 1)
        self.assertEqual(retried.canonical_finals[0]["msg_id"], stable)
        self.assertNotEqual(
            failed.frames[0].get("request_id"),
            None,
            "each attempt still has its own request_id",
        )

    async def test_retry_does_not_duplicate_an_already_persisted_row(self):
        stable = uuid.uuid4().hex
        await _run_turn(_Transport(), self.history_uid, metadata={"msg_id": stable})
        first = self.ai_rows()[0]

        # A redelivery of the SAME message (e.g. a reminder retry) reuses the id.
        second = _Transport()
        await _run_turn(
            second,
            self.history_uid,
            metadata={"msg_id": stable},
            context=_context(self.history_uid, sentences=["Versi kedua. "]),
        )

        rows = self.ai_rows()
        self.assertEqual(len(rows), 1, "one logical message, one row, always")
        self.assertEqual(rows[0]["msg_id"], stable)
        self.assertEqual(rows[0]["content"], "Versi kedua. ")
        self.assertEqual(rows[0]["timestamp"], first["timestamp"])


# --------------------------------------------------------------------------- #
# E. Legacy transcripts remain compatible
# --------------------------------------------------------------------------- #


class LegacyTranscriptTest(_Base):
    async def test_turn_on_a_legacy_transcript_succeeds(self):
        legacy = [
            {"role": "metadata", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"role": "human", "timestamp": "2026-01-01T00:00:01+00:00", "content": "lama"},
            {"role": "ai", "timestamp": "2026-01-01T00:00:02+00:00", "content": "balasan"},
        ]
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(legacy, handle)

        transport = _Transport()
        await _run_turn(transport, self.history_uid)

        self.assertEqual(len(transport.canonical_finals), 1)
        rows = self.rows()
        self.assertNotIn("msg_id", rows[0])
        self.assertNotIn("msg_id", rows[1])
        self.assertEqual([row["content"] for row in rows[0:2]], ["lama", "balasan"])
        self.assertTrue(rows[-1].get("msg_id"))


# --------------------------------------------------------------------------- #
# F. store_message status contract (fast, no turn)
# --------------------------------------------------------------------------- #


class StoreMessageContractTest(_Base):
    def test_returns_true_only_when_the_row_is_durable(self):
        self.assertIs(
            store_message(CONF, self.history_uid, "ai", "tersimpan", msg_id="m-1"),
            True,
        )
        self.assertEqual(self.ai_rows()[-1]["content"], "tersimpan")

    def test_missing_identifiers_are_a_failure_not_a_silent_noop(self):
        self.assertIs(store_message("", self.history_uid, "ai", "x"), False)
        self.assertIs(store_message(CONF, "", "ai", "x"), False)
        self.assertEqual(self.ai_rows(), [])

    def test_unreadable_transcript_is_a_failure(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(CORRUPT)
        self.assertIs(
            store_message(CONF, self.history_uid, "ai", "apa pun"), False
        )
        self.assertEqual(self.raw(), CORRUPT)

    def test_write_error_returns_false_and_keeps_the_file(self):
        before = self.raw()
        with patch("os.replace", new=_failing_replace_for("gagal disimpan")):
            self.assertIs(
                store_message(
                    CONF, self.history_uid, "ai", "gagal disimpan", msg_id="m-x"
                ),
                False,
            )
        self.assertEqual(self.raw(), before)
        self.assertEqual(self.ai_rows(), [])
        self.assertEqual(self.leftovers(), [])

    def test_atomic_writer_still_cleans_up_its_temp_file(self):
        # Guard for the assertion above: a successful write leaves no temp file
        # either, so "leftovers() == []" is a real invariant, not a tautology.
        _write_history_atomic(self.path, [{"role": "ai", "content": "ok"}])
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(self.rows()[0]["content"], "ok")


if __name__ == "__main__":
    unittest.main()
