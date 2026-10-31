"""Stage 0: one logical message = one stable identity (`msg_id`).

A turn's frames used to carry only ``request_id``, which names an ATTEMPT. A
redelivery therefore looked like a brand-new message: new attempt id, new
history row, new bubble. ``msg_id`` names the LOGICAL message: it is minted
once, travels on every frame of the turn, is persisted on the history row (so a
reload renders the same identity) and is reused verbatim by a retry.

Pinned here (real ``process_single_conversation``, real history store, only the
provider stream + transport faked):

* every frame of one turn shares one stable ``msg_id``, which is exactly the
  value persisted on that turn's AI row;
* a retry that supplies the same ``msg_id`` updates the row in place: one row,
  one identity, a different ``request_id``;
* without a supplied ``msg_id`` the same retry WOULD create a second row --
  the negative control proving the mechanism is what prevents the duplicate;
* two different messages with identical text stay two rows with distinct ids;
* history written before this field existed is preserved untouched.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from src.open_llm_vtuber.agent.output_types import Actions, DisplayText, SentenceOutput
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    modify_latest_message,
    store_message,
)
from src.open_llm_vtuber.conversations.single_conversation import (
    process_single_conversation,
)

CONF = "mili-message-identity"
SENTENCES = ["Aku ingetin minum obat ya. ", "Jam delapan malam ya. "]
FULL_TEXT = "".join(SENTENCES)
MSG_ID_HEX = re.compile(r"^[0-9a-f]{32}$")


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class _FakeAgent:
    """Minimal agent engine: one streamed sentence per configured text."""

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

    def handle_interrupt(self, _heard):  # pragma: no cover - not used here
        return None


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class _Transport:
    def __init__(self):
        self.frames = []

    async def __call__(self, payload):
        self.frames.append(json.loads(payload))

    # ---- views --------------------------------------------------------- #

    def of_type(self, frame_type):
        return [frame for frame in self.frames if frame.get("type") == frame_type]

    @property
    def audio_frames(self):
        return self.of_type("audio")

    @property
    def chain_start(self):
        starts = [
            frame
            for frame in self.of_type("control")
            if frame.get("text") == "conversation-chain-start"
        ]
        assert len(starts) == 1, starts
        return starts[0]

    @property
    def canonical_finals(self):
        return self.of_type("ai-final")


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


async def _drain(tts_manager=None, **_kwargs):
    """Stand-in for finalize_conversation_turn: flush, never wait on ACKs."""
    for _ in range(200):
        queue = getattr(tts_manager, "_payload_queue", None)
        if queue is None or queue.empty():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.02)


async def _run_turn(transport, history_uid, metadata=None, sentences=None):
    context = _context(history_uid, sentences)
    with patch(
        "src.open_llm_vtuber.conversations.single_conversation."
        "finalize_conversation_turn",
        new=_drain,
    ):
        await process_single_conversation(
            context=context,
            websocket_send=transport,
            client_uid="client",
            user_input="Halo Mili.",
            metadata=metadata or {},
        )


class _Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.history_uid = create_new_history(CONF)
        self.assertTrue(self.history_uid)
        store_message(CONF, self.history_uid, "human", "Halo Mili.")

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    # ---- views --------------------------------------------------------- #

    def _rows(self):
        return get_history(CONF, self.history_uid)

    def _ai_rows(self):
        return [row for row in self._rows() if row["role"] == "ai"]

    def _path(self):
        return os.path.join("chat_history", CONF, f"{self.history_uid}.json")


# --------------------------------------------------------------------------- #
# 1. Identity of a single turn
# --------------------------------------------------------------------------- #


class TurnIdentityTest(_Base):
    async def test_every_frame_of_a_turn_shares_one_stable_msg_id(self):
        transport = _Transport()
        await _run_turn(transport, self.history_uid)

        started = transport.chain_start["msg_id"]
        self.assertRegex(started, MSG_ID_HEX)
        self.assertEqual(len(transport.audio_frames), len(SENTENCES))
        for frame in transport.audio_frames:
            self.assertEqual(frame["msg_id"], started)
            # The attempt id is still there, unchanged behaviour.
            self.assertTrue(frame["request_id"])
        self.assertEqual(transport.canonical_finals[0]["msg_id"], started)

        rows = self._ai_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["msg_id"], started)
        self.assertEqual(rows[0]["content"], FULL_TEXT)

    async def test_persisted_row_is_what_a_reload_reads(self):
        transport = _Transport()
        await _run_turn(transport, self.history_uid)

        # `history-data` is a raw dump of get_history(); the identity must be
        # in it, otherwise the UI cannot name the row it just rendered.
        dumped = json.loads(json.dumps(self._rows()))
        ai_rows = [row for row in dumped if row["role"] == "ai"]
        self.assertEqual(ai_rows[0]["msg_id"], transport.chain_start["msg_id"])

    async def test_two_turns_in_the_same_second_get_distinct_ids(self):
        first = _Transport()
        second = _Transport()
        await _run_turn(first, self.history_uid)
        await _run_turn(second, self.history_uid)
        self.assertNotEqual(
            first.chain_start["msg_id"], second.chain_start["msg_id"]
        )
        # ids are identity, not timestamps: a 32-hex uuid, never a clock value.
        for frame in (first.chain_start, second.chain_start):
            self.assertRegex(frame["msg_id"], MSG_ID_HEX)


# --------------------------------------------------------------------------- #
# 2. Retry with the same msg_id
# --------------------------------------------------------------------------- #


class RetryIdentityTest(_Base):
    async def test_retry_with_same_msg_id_keeps_one_row(self):
        stable = uuid.uuid4().hex

        first = _Transport()
        await _run_turn(first, self.history_uid, metadata={"msg_id": stable})
        self.assertEqual(first.chain_start["msg_id"], stable)

        # Second attempt: same logical message, different wording.
        second = _Transport()
        await _run_turn(
            second,
            self.history_uid,
            metadata={"msg_id": stable},
            sentences=["Minum obatnya jam delapan ya. "],
        )

        self.assertEqual(second.chain_start["msg_id"], stable)
        # A different ATTEMPT, the same MESSAGE.
        self.assertNotEqual(
            first.chain_start["request_id"], second.chain_start["request_id"]
        )

        rows = self._ai_rows()
        self.assertEqual(len(rows), 1, rows)
        self.assertEqual(rows[0]["msg_id"], stable)
        self.assertEqual(rows[0]["content"], "Minum obatnya jam delapan ya. ")

    async def test_negative_control_without_supplied_msg_id_the_retry_duplicates(self):
        # Same flow, no stable id passed: the two attempts are two messages.
        first = _Transport()
        await _run_turn(first, self.history_uid)
        second = _Transport()
        await _run_turn(second, self.history_uid)

        self.assertNotEqual(
            first.chain_start["msg_id"], second.chain_start["msg_id"]
        )
        rows = self._ai_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["msg_id"] for row in rows}, {
            first.chain_start["msg_id"],
            second.chain_start["msg_id"],
        })

    async def test_two_different_messages_with_identical_text_stay_two_rows(self):
        first = _Transport()
        await _run_turn(first, self.history_uid, metadata={"msg_id": "id-one"})
        second = _Transport()
        await _run_turn(second, self.history_uid, metadata={"msg_id": "id-two"})

        self.assertNotEqual(first.chain_start["msg_id"], second.chain_start["msg_id"])
        rows = self._ai_rows()
        self.assertEqual([row["msg_id"] for row in rows], ["id-one", "id-two"])
        self.assertEqual([row["content"] for row in rows], [FULL_TEXT, FULL_TEXT])


# --------------------------------------------------------------------------- #
# 3. Store-level guarantees
# --------------------------------------------------------------------------- #


class StoreIdentityTest(_Base):
    def test_upsert_keeps_position_and_first_timestamp(self):
        store_message(CONF, self.history_uid, "ai", "versi awal", msg_id="m-1")
        store_message(CONF, self.history_uid, "human", "pesan setelahnya")
        before = self._rows()
        msg_id_rows = [row for row in before if row.get("msg_id") == "m-1"]
        self.assertEqual(len(msg_id_rows), 1)
        original_timestamp = msg_id_rows[0]["timestamp"]
        index_before = before.index(msg_id_rows[0])

        store_message(CONF, self.history_uid, "ai", "versi retry", msg_id="m-1")

        after = self._rows()
        self.assertEqual(len(after), len(before), "a retry must not append a row")
        self.assertEqual(after.index(after[index_before]), index_before)
        self.assertEqual(after[index_before]["content"], "versi retry")
        self.assertEqual(after[index_before]["timestamp"], original_timestamp)
        self.assertEqual(after[-1]["content"], "pesan setelahnya")

    def test_every_persisted_message_gets_an_id(self):
        before = len([row for row in self._rows() if row["role"] == "human"])
        store_message(CONF, self.history_uid, "human", "pesan sama")
        store_message(CONF, self.history_uid, "human", "pesan sama")
        rows = [row for row in self._rows() if row["role"] == "human"]
        self.assertEqual(
            len(rows), before + 2, "identical text is still two messages"
        )
        self.assertTrue(all(row.get("msg_id") for row in rows))
        self.assertEqual(len({row["msg_id"] for row in rows}), len(rows))

    def test_legacy_history_without_ids_is_preserved(self):
        # A transcript written before this field existed (e.g. the user's own
        # existing chat_history files).
        legacy = [
            {"role": "metadata", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"role": "human", "timestamp": "2026-01-01T00:00:01+00:00", "content": "lama"},
            {"role": "ai", "timestamp": "2026-01-01T00:00:02+00:00", "content": "balasan lama"},
        ]
        with open(self._path(), "w", encoding="utf-8") as handle:
            json.dump(legacy, handle)

        store_message(
            CONF, self.history_uid, "ai", "balasan baru", msg_id="m-new"
        )

        rows = self._rows()
        self.assertEqual([row["content"] for row in rows], [
            "lama",
            "balasan lama",
            "balasan baru",
        ])
        self.assertNotIn("msg_id", rows[0])
        self.assertNotIn("msg_id", rows[1])
        self.assertEqual(rows[2]["msg_id"], "m-new")

    def test_legacy_history_ignores_identity_matching_without_ids(self):
        # Two legacy rows with identical text stay two rows: absent identity is
        # never guessed from content.
        legacy = [
            {"role": "metadata", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"role": "ai", "timestamp": "2026-01-01T00:00:01+00:00", "content": "sama"},
            {"role": "ai", "timestamp": "2026-01-01T00:00:02+00:00", "content": "sama"},
        ]
        with open(self._path(), "w", encoding="utf-8") as handle:
            json.dump(legacy, handle)

        store_message(CONF, self.history_uid, "ai", "sama")

        self.assertEqual(len(self._rows()), 3)

    def test_modify_latest_message_keeps_the_identity(self):
        store_message(CONF, self.history_uid, "ai", "awal", msg_id="m-keep")
        self.assertTrue(
            modify_latest_message(CONF, self.history_uid, "ai", "diubah")
        )
        rows = self._rows()
        self.assertEqual(rows[-1]["content"], "diubah")
        self.assertEqual(rows[-1]["msg_id"], "m-keep")
