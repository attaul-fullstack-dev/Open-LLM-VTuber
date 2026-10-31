"""Streamed text, persisted history and reminder status must agree.

Three records exist for one assistant turn, and they used to diverge:

* the frames the transport ACCEPTED (``_ProactiveDeliveryProbe``: the only
  delivery evidence this backend can observe);
* the AI row written to history, which a reload/resync renders and which
  becomes Mili's conversation context;
* the reminder row plus its durable delivery receipt.

The observed divergence: history was written from the GENERATED text
(``full_response``) even when the transport refused every content frame. So a
reminder whose delivery failed was still recorded as spoken, and because the
reminder row correctly stayed pending for a retry, the retry appended the
reminder to the transcript a second time -- one message the user heard, two
bubbles after reconnect. A mid-message socket death had the milder version of
the same problem: history claimed the sentence that was never handed off.

Policy pinned here (all with real ``process_single_conversation``, real store,
real claim/settle/receipt; only the provider stream and the transport faked):

    the transcript may only claim text the transport accepted.

* nothing accepted (every content frame refused) -> no AI row, reminder
  released, the retry writes the row exactly once;
* some accepted, then refused -> the accepted prefix only, and the canonical
  ``ai-final`` text equals that same prefix;
* everything accepted -> the full text, unchanged;
* a turn cut short by a provider error keeps exactly its accepted frames.

There is still no client ACK: "accepted" means handed to the ASGI server, never
"rendered by the user".
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.stateless_llm.stateless_llm_interface import (
    LLMProviderErrorText,
)
from src.open_llm_vtuber.character_state import (
    claim_due_future_intentions,
    load_character_state,
    save_character_state,
)
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    store_message,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.future_intentions import (
    KIND_REMINDER,
    STATUS_DONE,
    STATUS_PENDING,
)
from src.open_llm_vtuber.proactive_chat import (
    ProactiveChatConfig,
    ProactiveIntentSignals,
    ProactiveStateMachine,
)
from src.open_llm_vtuber.proactive_gate import (
    ProactiveBudgetState,
    ProactiveGateConfig,
)
from src.open_llm_vtuber.reminder_delivery import delivered_ids
from src.open_llm_vtuber.websocket_handler import WebSocketHandler

JKT = "Asia/Jakarta"
CONF = "mili-history-consistency"
REMINDER_TEXT = "ingetin aku minum obat ya"
PAYLOAD = "WAKTUNYA MINUM OBAT"
NOW = datetime.now(timezone.utc).replace(microsecond=0)

SENTENCE_ONE = "Aku ingetin minum obat ya. "
SENTENCE_TWO = "Jam delapan malam ya. "


# --------------------------------------------------------------------------- #
# Provider streams
# --------------------------------------------------------------------------- #


class _ProviderBase:
    model = "history-consistency-test"
    max_tokens = 200

    def __init__(self):
        self.calls = []
        self.called = asyncio.Event()

    def _record(self, messages, system):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "system": system or "",
            }
        )
        self.called.set()


class _TwoSentences(_ProviderBase):
    async def chat_completion(self, messages, system=None, tools=None):
        self._record(messages, system)
        yield SENTENCE_ONE
        yield SENTENCE_TWO


class _OneSentenceThenProviderError(_ProviderBase):
    async def chat_completion(self, messages, system=None, tools=None):
        self._record(messages, system)
        yield SENTENCE_ONE
        yield LLMProviderErrorText(
            "Error calling the chat endpoint: stream broke.", category="api_error"
        )


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


# --------------------------------------------------------------------------- #
# Transport double
# --------------------------------------------------------------------------- #


class _Transport:
    """Accepts frames; can refuse content frames (socket dies mid-message)."""

    def __init__(self, refuse_after=None):
        self.accepted = []
        self.refused = []
        self.refuse_after = refuse_after

    @staticmethod
    def _display_text(frame):
        display = frame.get("display_text")
        if isinstance(display, dict):
            return str(display.get("text") or "")
        return ""

    @staticmethod
    def _is_content(frame):
        if frame.get("type") not in {"audio", "full-text", "text", "subtitle"}:
            return False
        if _Transport._display_text(frame).strip():
            return True
        for key in ("text", "transcript"):
            value = frame.get(key)
            if isinstance(value, str) and value.strip():
                return True
        return bool(frame.get("audio"))

    async def __call__(self, payload):
        frame = json.loads(payload)
        if (
            self.refuse_after is not None
            and self._is_content(frame)
            and len([f for f in self.accepted if self._is_content(f)])
            >= self.refuse_after
        ):
            self.refused.append(frame)
            raise RuntimeError("connection closed")
        self.accepted.append(frame)

    # ---- views -----------------------------------------------------------

    @property
    def content(self):
        return [frame for frame in self.accepted if frame["type"] == "audio"]

    @property
    def content_text(self):
        return "".join(
            self._display_text(frame) for frame in self.accepted if self._is_content(frame)
        )

    @property
    def canonical_finals(self):
        return [frame for frame in self.accepted if frame["type"] == "ai-final"]


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def _make_agent(conf_uid, history_uid, llm_cls):
    llm = llm_cls()
    agent = BasicMemoryAgent(
        llm=llm,
        system="persona Mili tetap aktif",
        live2d_model=_FakeLive2D(),
        tts_preprocessor_config=TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={"translate_audio": False, "translate_provider": "deeplx"},
        ),
        context_window_override=5000,
    )
    agent.set_memory_from_history(conf_uid, history_uid)
    return agent, llm


def _make_handler(agent, conf_uid, history_uid, transport):
    handler = WebSocketHandler.__new__(WebSocketHandler)
    handler.client_contexts = {
        "client": SimpleNamespace(
            history_uid=history_uid,
            agent_engine=agent,
            user_timezone=JKT,
            asr_engine=None,
            live2d_model=_FakeLive2D(),
            tts_engine=None,
            translate_engine=None,
            voice_output_enabled=False,
            character_config=SimpleNamespace(
                conf_uid=conf_uid,
                human_name="User",
                character_name="Mili",
                avatar="mili.png",
            ),
        )
    }
    handler.client_connections = {"client": SimpleNamespace(send_text=transport)}
    handler._proactive_maintenance = set()
    handler._proactive_timer_tasks = {}
    handler.current_conversation_tasks = {}
    handler._proactive_states = {}
    handler._proactive_machines = {}
    handler._reminder_row_retry_not_before = {}
    handler.chat_group_manager = SimpleNamespace(get_client_group=lambda _uid: None)
    handler._proactive_gate_config = ProactiveGateConfig(
        quiet_hours_start_hour=0,
        quiet_hours_end_hour=23,
        minimum_proactive_gap_seconds=3600,
    )
    handler._budget_state = lambda _uid, _conf: ProactiveBudgetState(
        dormant_until=(datetime.now(timezone.utc) + timedelta(hours=5)).isoformat(),
        backoff_until=(datetime.now(timezone.utc) + timedelta(hours=5)).isoformat(),
        consecutive_unanswered=3,
    )
    handler._persist_budget = lambda _uid, _conf: True
    handler._proactive_intent_signals = staticmethod(
        lambda _context, _state: ProactiveIntentSignals()
    )
    handler._meaningful_life_event = staticmethod(lambda _context, _since: False)
    return handler


def _machine(initial_idle=1):
    return ProactiveStateMachine(
        ProactiveChatConfig(
            initial_idle_min_seconds=initial_idle,
            initial_idle_max_seconds=initial_idle,
            followup_idle_min_seconds=100,
            followup_idle_max_seconds=100,
        ),
        randint=lambda minimum, _maximum: minimum,
    )


async def _drain_finalize(tts_manager=None, **_kwargs):
    for task in list(getattr(tts_manager, "task_list", None) or []):
        await asyncio.gather(task, return_exceptions=True)
    for _ in range(100):
        queue = getattr(tts_manager, "_payload_queue", None)
        if queue is None or queue.empty():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)


class _Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.history_uid = create_new_history(CONF)
        store_message(CONF, self.history_uid, "human", "Halo Mili.")
        store_message(CONF, self.history_uid, "ai", "Halo sayang.")

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    def _seed(self):
        row = {
            "id": uuid.uuid4().hex,
            "text": REMINDER_TEXT,
            "due_at": (NOW - timedelta(seconds=30)).isoformat(),
            "created_at": (NOW - timedelta(hours=3)).isoformat(),
            "status": STATUS_PENDING,
            "source": "conversation",
            "kind": KIND_REMINDER,
            "tz": JKT,
            "delivery_text": PAYLOAD,
        }
        state = load_character_state(CONF)
        state.future_intentions = [*(state.future_intentions or []), row]
        self.assertTrue(save_character_state(CONF, state))
        return row

    def _row(self, intention_id):
        return next(
            row
            for row in load_character_state(CONF).future_intentions
            if row.get("id") == intention_id
        )

    def _ai_contents(self):
        return [
            message["content"]
            for message in get_history(CONF, self.history_uid)
            if message["role"] == "ai"
        ]

    async def _run(self, handler, ready, *, timeout=10):
        machine = _machine()
        state = machine.new_state(self.history_uid)
        with patch(
            "src.open_llm_vtuber.conversations.single_conversation."
            "finalize_conversation_turn",
            new=_drain_finalize,
        ):
            task = asyncio.create_task(
                handler._run_proactive_timer("client", self.history_uid, state, machine)
            )
            deadline = asyncio.get_running_loop().time() + timeout
            while not ready() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.15)
            if task.done():
                await task
            else:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task


# --------------------------------------------------------------------------- #
# 1. Everything refused: no transcript row, and the retry writes it once
# --------------------------------------------------------------------------- #


class RefusedDeliveryHistoryTest(_Base):
    async def test_refused_reminder_turn_writes_no_transcript_row(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid, _TwoSentences)
        transport = _Transport(refuse_after=0)
        handler = _make_handler(agent, CONF, self.history_uid, transport)
        before = self._ai_contents()

        await self._run(
            handler,
            lambda: bool(handler._reminder_row_retry_not_before.get("client")),
        )

        self.assertTrue(transport.refused)
        self.assertEqual(transport.content_text, "")
        # Nothing was handed to the transport, so the transcript must not
        # claim Mili said anything.
        self.assertEqual(self._ai_contents(), before)
        self.assertEqual(transport.canonical_finals, [])
        # Reminder status stays truthfully pending for a retry.
        self.assertEqual(self._row(row["id"])["status"], STATUS_PENDING)
        self.assertEqual(delivered_ids(CONF), set())

    async def test_retry_after_refused_delivery_leaves_one_transcript_row(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid, _TwoSentences)
        transport = _Transport(refuse_after=0)
        handler = _make_handler(agent, CONF, self.history_uid, transport)
        handler.REMINDER_RETRY_DELAY_SECONDS = 0.05
        before = self._ai_contents()

        await self._run(
            handler,
            lambda: bool(handler._reminder_row_retry_not_before.get("client")),
        )
        self.assertEqual(self._ai_contents(), before)

        # The socket is back: the retry delivers and persists exactly once.
        transport.refuse_after = None
        await asyncio.sleep(0.1)
        await self._run(handler, lambda: self._row(row["id"])["status"] == STATUS_DONE)

        after = self._ai_contents()
        self.assertEqual(len(after), len(before) + 1, after)
        self.assertIn("ingetin minum obat", after[-1])
        # After a reconnect the user sees ONE reminder bubble, not two.
        reminder_rows = [
            content for content in after if "ingetin minum obat" in content
        ]
        self.assertEqual(len(reminder_rows), 1, after)
        self.assertEqual(delivered_ids(CONF), {row["id"]})


# --------------------------------------------------------------------------- #
# 2. Partially accepted: transcript claims only what went out
# --------------------------------------------------------------------------- #


class PartialAcceptanceHistoryTest(_Base):
    async def test_refused_second_sentence_is_not_recorded_as_spoken(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid, _TwoSentences)
        transport = _Transport(refuse_after=1)
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run(handler, lambda: bool(delivered_ids(CONF)))

        self.assertEqual(len(transport.content), 1)
        self.assertEqual(len(transport.refused), 1)
        accepted = transport.content_text
        self.assertIn("ingetin minum obat", accepted)

        stored = self._ai_contents()[-1]
        # History == exactly the frames the transport took: the refused
        # sentence is not presented as something Mili said.
        self.assertEqual(stored, accepted)
        self.assertNotIn("delapan", stored.lower())

        # The canonical final agrees with the persisted row, so a live bubble
        # is never rewritten with text that was dropped.
        finals = transport.canonical_finals
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["text"], stored)

        # Reminder state and transcript agree: delivered once, not resent.
        self.assertEqual(self._row(row["id"])["status"], STATUS_DONE)
        self.assertEqual(delivered_ids(CONF), {row["id"]})

    async def test_full_delivery_persists_the_streamed_text_unchanged(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid, _TwoSentences)
        transport = _Transport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run(handler, lambda: bool(delivered_ids(CONF)))

        self.assertEqual(len(transport.content), 2)
        stored = self._ai_contents()[-1]
        self.assertEqual(stored, transport.content_text)
        self.assertEqual(transport.canonical_finals[0]["text"], stored)
        self.assertEqual(self._row(row["id"])["status"], STATUS_DONE)


# --------------------------------------------------------------------------- #
# 3. Turn cut short by a provider error: accepted frames only
# --------------------------------------------------------------------------- #


class CutShortHistoryTest(_Base):
    async def test_provider_error_history_equals_the_accepted_frame(self):
        row = self._seed()
        agent, _llm = _make_agent(
            CONF, self.history_uid, _OneSentenceThenProviderError
        )
        transport = _Transport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run(handler, lambda: bool(delivered_ids(CONF)))

        self.assertEqual(len(transport.content), 1)
        stored = self._ai_contents()[-1]
        self.assertEqual(stored, transport.content_text)
        self.assertEqual(transport.canonical_finals[0]["text"], stored)
        self.assertEqual(self._row(row["id"])["status"], STATUS_DONE)

    async def test_every_frame_refused_on_cut_short_turn_records_nothing(self):
        row = self._seed()
        agent, _llm = _make_agent(
            CONF, self.history_uid, _OneSentenceThenProviderError
        )
        transport = _Transport(refuse_after=0)
        handler = _make_handler(agent, CONF, self.history_uid, transport)
        before = self._ai_contents()

        await self._run(
            handler,
            lambda: bool(handler._reminder_row_retry_not_before.get("client")),
        )

        self.assertEqual(self._ai_contents(), before)
        self.assertEqual(self._row(row["id"])["status"], STATUS_PENDING)
        self.assertEqual(delivered_ids(CONF), set())


# --------------------------------------------------------------------------- #
# 4. Cross-record invariant, expressed directly
# --------------------------------------------------------------------------- #


class CrossRecordInvariantTest(_Base):
    async def test_delivered_reminder_always_has_matching_transcript_text(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid, _TwoSentences)
        transport = _Transport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run(handler, lambda: bool(delivered_ids(CONF)))

        self.assertEqual(delivered_ids(CONF), {row["id"]})
        stored = self._ai_contents()[-1]
        # Every streamed frame is a prefix of the persisted row (in order),
        # so a reload can neither invent nor lose words relative to the wire.
        streamed = ""
        for frame in transport.content:
            streamed += _frame_text(frame)
            self.assertTrue(stored.startswith(streamed), (streamed, stored))
        self.assertEqual(stored, streamed)

    async def test_undelivered_reminder_never_appears_in_the_transcript(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid, _TwoSentences)
        transport = _Transport(refuse_after=0)
        handler = _make_handler(agent, CONF, self.history_uid, transport)
        before = self._ai_contents()

        await self._run(
            handler,
            lambda: bool(handler._reminder_row_retry_not_before.get("client")),
        )

        self.assertEqual(delivered_ids(CONF), set())
        self.assertEqual(self._ai_contents(), before)
        self.assertEqual(
            [
                item["id"]
                for item in claim_due_future_intentions(
                    CONF, owner="retry", now=NOW + timedelta(seconds=700)
                )
            ],
            [row["id"]],
        )


def _frame_text(frame):
    display = frame.get("display_text")
    return str((display or {}).get("text") or "") if isinstance(display, dict) else ""


if __name__ == "__main__":
    unittest.main()
