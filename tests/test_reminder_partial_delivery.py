"""Partial reminder delivery: what reaches the transport, and what survives.

A reminder turn can stop in the middle. Two different things can be cut:

* the transport dies while the ordered TTS sender is still flushing frames
  (``TTSTaskManager._process_payload_queue`` returns on the first refusal, so
  later payloads are dropped);
* the provider stream ends/breaks after some sentences, so the remainder the
  model would have said never exists anywhere.

Neither case has an acknowledgment: the backend only knows that a frame was
ACCEPTED by the ASGI server, not that a client rendered it. These tests pin the
chosen policy with explicit event ordering:

1. Anything accepted is never sent again (no duplicate phrase, no replay of
   the whole reminder).
2. Nothing accepted stays retryable (no silent loss, no completion claim).
3. The text the user actually saw is kept in history, so a reconnect/reload
   shows the fragment instead of erasing it, and Mili's memory does not
   contradict what she said.
4. A remainder that was never generated is never fabricated, and the next
   scheduler cycle does not speak the reminder again.

Only the websocket transport, the provider stream and the playback wait are
faked. The reminder store, claim/release, receipt journal, settlement,
history write and the real ``process_single_conversation`` are the production
implementations.
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
from src.open_llm_vtuber.reminder_delivery import (
    delivered_ids,
    get_delivery_journal_path,
)
from src.open_llm_vtuber.websocket_handler import WebSocketHandler

JKT = "Asia/Jakarta"
CONF = "mili-partial-delivery"
REMINDER_TEXT = "ingetin aku minum obat ya"
PAYLOAD = "WAKTUNYA MINUM OBAT"
NOW = datetime.now(timezone.utc).replace(microsecond=0)

SENTENCE_ONE = "Aku ingetin minum obat ya. "
SENTENCE_TWO = "Jam delapan malam ya. "
FRAGMENT = "Jam delapan"


# --------------------------------------------------------------------------- #
# Provider shapes: exactly what a real stream does at each cut point
# --------------------------------------------------------------------------- #


class _ProviderBase:
    model = "partial-delivery-test"
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


class _OneSentenceThenProviderError(_ProviderBase):
    """One complete sentence, then a typed provider error (no tail buffered)."""

    async def chat_completion(self, messages, system=None, tools=None):
        self._record(messages, system)
        yield SENTENCE_ONE
        yield LLMProviderErrorText(
            "Error calling the chat endpoint: provider stream broke.",
            category="api_error",
        )


class _SentencePlusFragmentThenProviderError(_ProviderBase):
    """A complete sentence plus an unterminated tail, then the stream breaks.

    The sentence divider flushes its buffer when the stream ends normally, so
    the tail IS delivered even though the turn is a provider error. That flush
    is why "partial" here means "words the user saw", not "words the caller
    acknowledged".
    """

    async def chat_completion(self, messages, system=None, tools=None):
        self._record(messages, system)
        yield SENTENCE_ONE
        yield FRAGMENT
        yield LLMProviderErrorText(
            "Error calling the chat endpoint: provider stream broke.",
            category="api_error",
        )


class _SentenceThenRaise(_ProviderBase):
    """One complete sentence, then the provider stream RAISES.

    A raise is not a normal stream end, so the divider never flushes its
    buffer: only the completed sentence can reach the client.
    """

    async def chat_completion(self, messages, system=None, tools=None):
        self._record(messages, system)
        yield SENTENCE_ONE
        raise RuntimeError("provider stream aborted")


class _ProviderErrorBeforeAnyText(_ProviderBase):
    async def chat_completion(self, messages, system=None, tools=None):
        self._record(messages, system)
        yield LLMProviderErrorText(
            "Error calling the chat endpoint: no tokens.", category="api_error"
        )


class _TwoSentences(_ProviderBase):
    async def chat_completion(self, messages, system=None, tools=None):
        self._record(messages, system)
        yield SENTENCE_ONE
        yield SENTENCE_TWO


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


# --------------------------------------------------------------------------- #
# Transport doubles
# --------------------------------------------------------------------------- #


class _Transport:
    """Records accepted frames and can refuse the Nth content frame onward.

    ``refuse_after=None`` accepts everything; ``refuse_after=0`` refuses every
    content frame (dead socket before the first word); ``refuse_after=1``
    accepts one content frame and then refuses (socket dies mid-message).
    Bookkeeping frames (``control``, ``error``, latency events) are accepted
    unless ``refuse_all`` is set, matching a socket that dies while the turn is
    streaming assistant content.
    """

    def __init__(self, refuse_after=None, refuse_all=False):
        self.accepted = []
        self.refused = []
        self.refuse_after = refuse_after
        self.refuse_all = refuse_all

    def _is_content(self, frame):
        display = frame.get("display_text")
        if isinstance(display, dict) and str(display.get("text") or "").strip():
            return True
        return bool(frame.get("audio"))

    async def __call__(self, payload):
        frame = json.loads(payload)
        if self.refuse_all or (
            self.refuse_after is not None
            and self._is_content(frame)
            and len(self.content) >= self.refuse_after
        ):
            self.refused.append(frame)
            raise RuntimeError("connection closed")
        self.accepted.append(frame)

    @property
    def content(self):
        return [frame for frame in self.accepted if frame.get("type") == "audio"]


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
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
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
    handler.client_connections = {
        "client": SimpleNamespace(send_text=transport)
    }
    handler._proactive_maintenance = set()
    handler._proactive_timer_tasks = {}
    handler.current_conversation_tasks = {}
    handler._proactive_states = {}
    handler._proactive_machines = {}
    handler._reminder_row_retry_not_before = {}
    handler.chat_group_manager = SimpleNamespace(get_client_group=lambda _uid: None)
    # Only the reminder (user obligation) lane may produce a turn.
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
    """Stand-in for the playback wait: flush queued payloads, never block."""
    for task in list(getattr(tts_manager, "task_list", None) or []):
        await asyncio.gather(task, return_exceptions=True)
    for _ in range(100):
        queue = getattr(tts_manager, "_payload_queue", None)
        if queue is None or queue.empty():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)


class _PartialBase(unittest.IsolatedAsyncioTestCase):
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

    # ---- store helpers ---------------------------------------------------

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

    def _receipts(self):
        path = get_delivery_journal_path(CONF)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as handle:
            return [line for line in handle if line.strip()]

    def _messages(self):
        return [
            (message["role"], message["content"])
            for message in get_history(CONF, self.history_uid)
        ]

    def _ai_contents(self):
        return [
            content for role, content in self._messages() if role == "ai"
        ]

    # ---- scheduler driver ------------------------------------------------

    async def _run(self, handler, ready, *, timeout=10):
        """Run the real proactive scheduler until ``ready()`` or timeout."""
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

    def _assert_settled_once(self, row):
        self.assertEqual(delivered_ids(CONF), {row["id"]})
        self.assertEqual(len(self._receipts()), 1)
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_DONE)
        self.assertNotIn("claimed_by", stored)
        self.assertEqual(
            claim_due_future_intentions(
                CONF, owner="other-process", now=NOW + timedelta(seconds=700)
            ),
            [],
        )


# --------------------------------------------------------------------------- #
# 1. Single frame, then the provider breaks
# --------------------------------------------------------------------------- #


class SingleFramePartialTest(_PartialBase):
    async def test_fragment_seen_by_the_client_is_kept_and_never_resent(self):
        row = self._seed()
        agent, llm = _make_agent(
            CONF, self.history_uid, _OneSentenceThenProviderError
        )
        transport = _Transport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)
        baseline = len(self._ai_contents())

        await self._run(handler, lambda: bool(delivered_ids(CONF)))

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(transport.content), 1)
        frame_text = transport.content[0]["display_text"]["text"]
        self.assertIn("ingetin minum obat", frame_text)

        # Settled exactly once; the reminder cannot be claimed again.
        self._assert_settled_once(row)

        # The delivered fragment is the record: it is in history even though
        # the turn ended as a provider error, so a reload/resync still shows
        # the words the user saw and Mili's memory agrees with them.
        ai_contents = self._ai_contents()
        self.assertEqual(len(ai_contents), baseline + 1)
        self.assertIn("ingetin minum obat", ai_contents[-1])

        # Nothing is spoken twice: a restart finds no claim, so no turn runs.
        restarted_agent, restarted_llm = _make_agent(
            CONF, self.history_uid, _OneSentenceThenProviderError
        )
        restarted_transport = _Transport()
        restarted = _make_handler(
            restarted_agent, CONF, self.history_uid, restarted_transport
        )
        await self._run(restarted, lambda: False, timeout=2)
        self.assertEqual(restarted_transport.accepted, [])
        self.assertEqual(restarted_llm.calls, [])

    async def test_flushed_tail_is_delivered_and_no_further_text_is_invented(self):
        row = self._seed()
        agent, _llm = _make_agent(
            CONF, self.history_uid, _SentencePlusFragmentThenProviderError
        )
        transport = _Transport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run(handler, lambda: bool(delivered_ids(CONF)))

        texts = [frame["display_text"]["text"] for frame in transport.content]
        self.assertEqual(len(texts), 2, texts)
        # The unterminated tail the divider flushed is what the user saw.
        self.assertIn("ingetin minum obat", texts[0])
        self.assertIn(FRAGMENT.lower(), texts[1].lower())

        self._assert_settled_once(row)
        stored = self._ai_contents()[-1]
        self.assertIn("ingetin minum obat", stored)
        self.assertIn(FRAGMENT.lower(), stored.lower())
        # The remainder after the break never existed: it is neither spoken
        # nor written, and the turn is not presented as a finished sentence.
        self.assertNotIn("malam", stored.lower())
        for frame in transport.content:
            self.assertNotIn("malam", json.dumps(frame).lower())


class RaisedExceptionAfterSpeechTest(_PartialBase):
    async def test_raise_after_a_sentence_keeps_the_spoken_text(self):
        row = self._seed()
        agent, llm = _make_agent(CONF, self.history_uid, _SentenceThenRaise)
        transport = _Transport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run(handler, lambda: bool(delivered_ids(CONF)))

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(transport.content), 1)
        self.assertIn(
            "ingetin minum obat", transport.content[0]["display_text"]["text"]
        )
        self._assert_settled_once(row)

        # The sentence the user heard is the record; the aborted tail does not
        # exist and is never written as if it had been said.
        stored = self._ai_contents()[-1]
        self.assertIn("ingetin minum obat", stored)
        self.assertNotIn("malam", stored.lower())

        restarted_agent, restarted_llm = _make_agent(
            CONF, self.history_uid, _SentenceThenRaise
        )
        restarted_transport = _Transport()
        restarted = _make_handler(
            restarted_agent, CONF, self.history_uid, restarted_transport
        )
        await self._run(restarted, lambda: False, timeout=2)
        self.assertEqual(restarted_transport.accepted, [])
        self.assertEqual(restarted_llm.calls, [])


# --------------------------------------------------------------------------- #
# 2. Several frames, one of them refused mid-message
# --------------------------------------------------------------------------- #


class MidMessageRefusalTest(_PartialBase):
    async def test_second_frame_refused_keeps_the_first_and_does_not_resend(self):
        row = self._seed()
        agent, llm = _make_agent(CONF, self.history_uid, _TwoSentences)
        transport = _Transport(refuse_after=1)
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run(handler, lambda: bool(delivered_ids(CONF)))

        # Exactly one content frame reached the transport; the second was
        # refused and the ordered sender abandoned the rest.
        self.assertEqual(len(transport.content), 1)
        self.assertIn("ingetin minum obat", transport.content[0]["display_text"]["text"])
        self.assertEqual(len(transport.refused), 1)

        # Accepted content means "may already be visible": settle once, never
        # replay the reminder (which would repeat the first sentence).
        self._assert_settled_once(row)
        self.assertEqual(len(llm.calls), 1)

        # The transcript claims only the frame the transport accepted. The
        # refused sentence existed in memory but was never handed off, so
        # recording it would make the reload claim words the client never got
        # (and, for a reminder that is retried, duplicate the reminder).
        stored = self._ai_contents()[-1]
        self.assertIn("ingetin minum obat", stored)
        self.assertNotIn("delapan", stored.lower())

        restarted_agent, restarted_llm = _make_agent(
            CONF, self.history_uid, _TwoSentences
        )
        restarted_transport = _Transport()
        restarted = _make_handler(
            restarted_agent, CONF, self.history_uid, restarted_transport
        )
        await self._run(restarted, lambda: False, timeout=2)
        self.assertEqual(restarted_transport.accepted, [])
        self.assertEqual(restarted_llm.calls, [])

    async def test_two_frames_delivered_completely_persist_the_whole_reply(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid, _TwoSentences)
        transport = _Transport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run(handler, lambda: bool(delivered_ids(CONF)))

        self.assertEqual(len(transport.content), 2)
        self._assert_settled_once(row)
        stored = self._ai_contents()[-1]
        self.assertIn("ingetin minum obat", stored)
        self.assertIn("delapan", stored.lower())


# --------------------------------------------------------------------------- #
# 3. Nothing accepted -> nothing claimed as delivered
# --------------------------------------------------------------------------- #


class NothingAcceptedTest(_PartialBase):
    async def test_provider_error_before_any_frame_discards_and_retries(self):
        row = self._seed()
        agent, _llm = _make_agent(
            CONF, self.history_uid, _ProviderErrorBeforeAnyText
        )
        transport = _Transport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)
        baseline = len(self._ai_contents())

        await self._run(
            handler,
            lambda: bool(handler._reminder_row_retry_not_before.get("client")),
        )

        self.assertEqual(transport.content, [])
        self.assertEqual(delivered_ids(CONF), set())
        self.assertEqual(self._receipts(), [])
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", stored)
        # No words were spoken, so nothing is written as an assistant message.
        self.assertEqual(self._ai_contents(), self._ai_contents()[:baseline])

    async def test_dead_socket_from_the_start_writes_no_receipt(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid, _TwoSentences)
        transport = _Transport(refuse_after=0)
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run(
            handler,
            lambda: bool(handler._reminder_row_retry_not_before.get("client")),
        )

        self.assertEqual(transport.content, [])
        self.assertTrue(transport.refused)
        self.assertEqual(delivered_ids(CONF), set())
        self.assertEqual(self._receipts(), [])
        self.assertEqual(self._row(row["id"])["status"], STATUS_PENDING)
        self.assertEqual(
            [
                item["id"]
                for item in claim_due_future_intentions(
                    CONF, owner="retry", now=NOW + timedelta(seconds=700)
                )
            ],
            [row["id"]],
        )

    async def test_failed_attempt_is_retried_once_and_delivered_without_overlap(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid, _TwoSentences)
        transport = _Transport(refuse_after=0)
        handler = _make_handler(agent, CONF, self.history_uid, transport)
        handler.REMINDER_RETRY_DELAY_SECONDS = 0.05

        await self._run(
            handler,
            lambda: bool(handler._reminder_row_retry_not_before.get("client")),
        )
        self.assertEqual(transport.content, [])
        self.assertEqual(self._row(row["id"])["status"], STATUS_PENDING)

        # The socket is back; the retry delivers the two frames once.
        transport.refuse_after = None
        await asyncio.sleep(0.1)
        await self._run(handler, lambda: self._row(row["id"])["status"] == STATUS_DONE)

        self.assertEqual(len(transport.content), 2)
        texts = [frame["display_text"]["text"] for frame in transport.content]
        self.assertEqual(sum("ingetin minum obat" in text for text in texts), 1)
        self.assertEqual(sum("delapan" in text.lower() for text in texts), 1)
        self._assert_settled_once(row)


# --------------------------------------------------------------------------- #
# 4. Receipt/completion consistency after a partial delivery
# --------------------------------------------------------------------------- #


class PartialSettlementDurabilityTest(_PartialBase):
    async def test_receipt_durable_but_completion_lost_after_partial_reconciles(self):
        row = self._seed()
        agent, _llm = _make_agent(
            CONF, self.history_uid, _OneSentenceThenProviderError
        )
        agent.complete_claimed_future_intentions = lambda _owner, _ids: 0
        transport = _Transport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run(handler, lambda: bool(self._receipts()))

        # Delivered partially, receipt durable, completion write lost.
        self.assertEqual(len(transport.content), 1)
        self.assertEqual(len(llm_calls := _llm.calls), 1)
        self.assertEqual(delivered_ids(CONF), {row["id"]})
        self.assertNotIn("claimed_by", self._row(row["id"]))

        # A restart reconciles the receipt to done and speaks nothing again.
        restarted_agent, restarted_llm = _make_agent(
            CONF, self.history_uid, _OneSentenceThenProviderError
        )
        restarted_transport = _Transport()
        restarted = _make_handler(
            restarted_agent, CONF, self.history_uid, restarted_transport
        )
        await self._run(
            restarted, lambda: self._row(row["id"])["status"] == STATUS_DONE
        )
        self.assertEqual(restarted_transport.accepted, [])
        self.assertEqual(restarted_llm.calls, [])
        self.assertEqual(len(self._receipts()), 1)
        self.assertEqual(len(transport.content), 1, llm_calls)

    async def test_partial_delivery_with_failed_receipt_is_not_resent(self):
        row = self._seed()
        agent, _llm = _make_agent(
            CONF, self.history_uid, _OneSentenceThenProviderError
        )
        agent.record_reminder_delivery = lambda _owner, _ids: False
        transport = _Transport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run(handler, lambda: self._row(row["id"])["status"] == STATUS_DONE)

        # The completion still lands, so the accepted fragment is not repeated
        # even though the journal line could not be written.
        self.assertEqual(delivered_ids(CONF), set())
        self.assertEqual(self._receipts(), [])
        self.assertEqual(len(transport.content), 1)
        self.assertEqual(self._row(row["id"])["status"], STATUS_DONE)
        self.assertEqual(
            claim_due_future_intentions(
                CONF, owner="other-process", now=NOW + timedelta(seconds=700)
            ),
            [],
        )


# --------------------------------------------------------------------------- #
# 5. The policy boundary itself
# --------------------------------------------------------------------------- #


class PartialPolicyBoundaryTest(unittest.IsolatedAsyncioTestCase):
    def test_no_auto_continuation_is_implemented_for_the_unsent_remainder(self):
        """The remainder is never reconstructed by the dispatcher.

        A partial reminder is completed (no resend) and its delivered text is
        kept in history. The backend does not synthesize a follow-up turn for
        the missing part: there is no transport ACK, so it cannot know which
        words the client rendered, and the remainder may not exist at all.
        """
        import inspect

        from src.open_llm_vtuber import websocket_handler

        source = inspect.getsource(websocket_handler)
        for forbidden in (
            "continue_reminder",
            "resume_partial",
            "partial_remainder",
            "replay_remainder",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
