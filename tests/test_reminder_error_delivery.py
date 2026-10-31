"""Reminder settlement on the error/exception path: no resend after delivery.

The cancel/interrupt path settles a reminder from what the transport accepted
(``_ProactiveDeliveryProbe``). The provider/transport failure path used to
release the claim unconditionally, which reopened the duplicate window for
every case where content had *already* left the process:

* a typed provider error after one or more sentences streamed (the turn
  deliberately empties ``full_response``, so ``bool(response)`` was False even
  though the frame was handed to the transport);
* an exception raised after the reply was spoken (history write, turn
  finalisation);
* a turn that returned text while the transport refused every content frame
  (the TTS sender swallows send errors), which used to mark the reminder
  ``done`` even though nothing was delivered.

These tests pin both directions: evidence of a handoff is never resent, and
absence of evidence (including an explicit refusal) keeps the reminder
retryable. Only the websocket transport and, where stated, the provider turn
are faked; there is still no transport ACK, so exactly-once is not claimed.

Ordering used by every test: claim -> turn -> content frame -> failure ->
settlement -> (optional) restart.
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
from src.open_llm_vtuber.chat_history_manager import create_new_history, store_message
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
from src.open_llm_vtuber.websocket_handler import (
    WebSocketHandler,
    _ProactiveDeliveryProbe,
)

JKT = "Asia/Jakarta"
CONF = "mili-error-delivery"
PAYLOAD = "WAKTUNYA MINUM OBAT"
REMINDER_TEXT = "ingetin aku minum obat ya"
NOW = datetime.now(timezone.utc).replace(microsecond=0)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class _SentenceThenProviderErrorLLM:
    """Real-provider shape: a complete sentence, then a typed provider error."""

    model = "error-delivery-test"
    max_tokens = 200

    def __init__(self):
        self.calls = []
        self.called = asyncio.Event()

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "system": system or "",
            }
        )
        self.called.set()
        yield "Aku ingetin minum obat ya. "
        yield LLMProviderErrorText(
            "Error calling the chat endpoint: provider stream broke.",
            category="api_error",
        )


class _SuccessLLM:
    model = "error-delivery-test"
    max_tokens = 200

    def __init__(self):
        self.calls = []
        self.called = asyncio.Event()

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "system": system or "",
            }
        )
        self.called.set()
        yield "Iya, aku ingetin minum obat ya."


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


def _make_agent(conf_uid, history_uid, llm_cls=_SuccessLLM):
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


def _make_handler(agent, conf_uid, history_uid, send_text=None):
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
        "client": SimpleNamespace(send_text=send_text)
    }
    handler._proactive_maintenance = set()
    handler._proactive_timer_tasks = {}
    handler.current_conversation_tasks = {}
    handler._proactive_states = {}
    handler._proactive_machines = {}
    handler._reminder_row_retry_not_before = {}
    handler.chat_group_manager = SimpleNamespace(get_client_group=lambda _uid: None)
    # Only the user obligation (due reminder) lane may produce a turn.
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


def _delivery_frame(text=PAYLOAD):
    """One assistant-content frame, shaped like ``prepare_audio_payload``."""
    return json.dumps(
        {
            "type": "audio",
            "audio": None,
            "volumes": [],
            "slice_length": 20,
            "display_text": {"name": "Mili", "avatar": "mili.png", "text": text},
            "actions": None,
            "forwarded": False,
        }
    )


async def _recording_send(frames, payload):
    frames.append(json.loads(payload))


class _ProbeTransport:
    """Transport double that can refuse only the frames carrying words."""

    def __init__(self, *, refuse_content=False):
        self.frames = []
        self.refused = []
        self.refuse_content = refuse_content

    async def __call__(self, payload):
        frame = json.loads(payload)
        if self.refuse_content and frame.get("type") == "audio":
            self.refused.append(frame)
            raise RuntimeError("transport refused the audio frame")
        self.frames.append(frame)


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

    def _receipt_lines(self):
        path = get_delivery_journal_path(CONF)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as handle:
            return [line for line in handle if line.strip()]

    async def _run_timer(self, handler, turn, ready, *, timeout=8):
        """Run the real scheduler with ``turn`` until ``ready()`` is true."""
        machine = _machine()
        state = machine.new_state(self.history_uid)
        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=turn,
        ):
            task = asyncio.create_task(
                handler._run_proactive_timer("client", self.history_uid, state, machine)
            )
            deadline = asyncio.get_running_loop().time() + timeout
            while not ready() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.02)
            # Let one more loop turn run so a late settle can still happen.
            await asyncio.sleep(0.1)
            if task.done():
                await task
            else:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

    def _assert_no_claim_possible(self, row):
        self.assertEqual(
            claim_due_future_intentions(
                CONF, owner="other-owner", now=NOW + timedelta(seconds=700)
            ),
            [],
        )
        self.assertEqual(self._row(row["id"])["status"], STATUS_DONE)
        self.assertNotIn("claimed_by", self._row(row["id"]))


# --------------------------------------------------------------------------- #
# The probe's two-directional evidence
# --------------------------------------------------------------------------- #


class ProbeEvidenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_refused_content_frame_is_recorded_and_reraised(self):
        async def refusing(_payload):
            raise RuntimeError("socket gone")

        probe = _ProactiveDeliveryProbe(refusing)
        with self.assertRaises(RuntimeError):
            await probe(_delivery_frame())
        self.assertFalse(probe.delivered)
        self.assertTrue(probe.refused)

    async def test_refused_bookkeeping_frame_is_not_delivery_evidence(self):
        async def refusing(_payload):
            raise RuntimeError("socket gone")

        probe = _ProactiveDeliveryProbe(refusing)
        with self.assertRaises(RuntimeError):
            await probe(json.dumps({"type": "control", "text": "chain-start"}))
        self.assertFalse(probe.delivered)
        self.assertFalse(probe.refused)

    async def test_accepted_content_then_refusal_still_counts_as_delivered(self):
        calls = []

        async def first_ok_then_refuse(payload):
            calls.append(payload)
            if len(calls) > 1:
                raise RuntimeError("socket died mid-turn")

        probe = _ProactiveDeliveryProbe(first_ok_then_refuse)
        await probe(_delivery_frame("Aku ingetin"))
        with self.assertRaises(RuntimeError):
            await probe(_delivery_frame(PAYLOAD))
        self.assertTrue(probe.delivered)
        self.assertTrue(probe.refused)


# --------------------------------------------------------------------------- #
# Exception path, explicit ordering
# --------------------------------------------------------------------------- #


class ExceptionDuringDeliveryTest(_Base):
    async def test_exception_before_first_frame_releases_for_retry(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid)
        transport = _ProbeTransport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)
        started = []

        async def failing_turn(**_kwargs):
            started.append("provider failed before speaking")
            raise TimeoutError("provider timed out")

        await self._run_timer(handler, failing_turn, lambda: bool(started))

        self.assertEqual(delivered_ids(CONF), set())
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", stored)
        self.assertIn(row["id"], handler._reminder_row_retry_not_before["client"])
        self.assertEqual(transport.frames, [])
        # Recovery preserved: the row is claimable again after the lease.
        retry = claim_due_future_intentions(
            CONF, owner="retry-owner", now=NOW + timedelta(seconds=700)
        )
        self.assertEqual([item["id"] for item in retry], [row["id"]])

    async def test_exception_after_first_frame_settles_once(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid)
        transport = _ProbeTransport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        async def turn(**kwargs):
            await kwargs["websocket_send"](_delivery_frame())
            raise OSError("history write failed after the reply was spoken")

        await self._run_timer(
            handler, turn, lambda: self._row(row["id"])["status"] == STATUS_DONE
        )

        self.assertEqual(len(transport.frames), 1)
        self.assertEqual(len(self._receipt_lines()), 1)
        self.assertEqual(delivered_ids(CONF), {row["id"]})
        self._assert_no_claim_possible(row)

    async def test_exception_after_partial_text_does_not_repeat_on_restart(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid)
        transport = _ProbeTransport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        async def turn(**kwargs):
            await kwargs["websocket_send"](_delivery_frame("Aku ingetin ya,"))
            await kwargs["websocket_send"](_delivery_frame(PAYLOAD))
            raise RuntimeError("finalisation failed after both sentences")

        await self._run_timer(
            handler, turn, lambda: self._row(row["id"])["status"] == STATUS_DONE
        )
        self.assertEqual(len(transport.frames), 2)
        self.assertEqual(delivered_ids(CONF), {row["id"]})

        # A fresh process must not deliver it again, even though the first
        # attempt never returned a response string.
        restarted_agent, _llm = _make_agent(CONF, self.history_uid)
        restarted_transport = _ProbeTransport()
        restarted = _make_handler(
            restarted_agent, CONF, self.history_uid, restarted_transport
        )
        calls = []

        async def turn_again(**kwargs):
            calls.append(kwargs["metadata"].get("proactive_due_reminders"))
            await kwargs["websocket_send"](_delivery_frame())
            return PAYLOAD

        await self._run_timer(restarted, turn_again, lambda: bool(calls))
        self.assertEqual(calls, [])
        self.assertEqual(restarted_transport.frames, [])

    async def test_exception_after_delivery_with_failed_receipt_still_completes(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid)
        agent.record_reminder_delivery = lambda _owner, _ids: False
        transport = _ProbeTransport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        async def turn(**kwargs):
            await kwargs["websocket_send"](_delivery_frame())
            raise RuntimeError("turn died after the sentence")

        await self._run_timer(
            handler, turn, lambda: self._row(row["id"])["status"] == STATUS_DONE
        )

        self.assertEqual(delivered_ids(CONF), set())  # receipt was not durable
        self._assert_no_claim_possible(row)

    async def test_receipt_durable_but_completion_lost_is_reconciled(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid)
        agent.complete_claimed_future_intentions = lambda _owner, _ids: 0
        transport = _ProbeTransport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        async def turn(**kwargs):
            await kwargs["websocket_send"](_delivery_frame())
            raise RuntimeError("turn died after the sentence")

        await self._run_timer(
            handler, turn, lambda: bool(self._receipt_lines())
        )

        # Receipt durable, completion write lost: still pending on disk.
        self.assertEqual(delivered_ids(CONF), {row["id"]})
        self.assertEqual(self._row(row["id"])["status"], STATUS_PENDING)

        # "Restart": a fresh process reconciles the receipt instead of sending.
        restarted_agent, _llm = _make_agent(CONF, self.history_uid)
        restarted_transport = _ProbeTransport()
        restarted = _make_handler(
            restarted_agent, CONF, self.history_uid, restarted_transport
        )
        calls = []

        async def turn_again(**kwargs):
            calls.append(kwargs["metadata"].get("proactive_due_reminders"))
            await kwargs["websocket_send"](_delivery_frame())
            return PAYLOAD

        await self._run_timer(
            restarted,
            turn_again,
            lambda: self._row(row["id"])["status"] == STATUS_DONE,
        )
        self.assertEqual(calls, [])
        self.assertEqual(restarted_transport.frames, [])
        self._assert_no_claim_possible(row)

    async def test_transport_refusal_before_delivery_keeps_it_retryable(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid)
        transport = _ProbeTransport(refuse_content=True)
        handler = _make_handler(agent, CONF, self.history_uid, transport)
        observed = {}
        attempted = []

        async def turn(**kwargs):
            try:
                await kwargs["websocket_send"](_delivery_frame())
            except RuntimeError:
                observed["delivered"] = kwargs["websocket_send"].delivered
                observed["refused"] = kwargs["websocket_send"].refused
                attempted.append("refused")
                raise

        await self._run_timer(handler, turn, lambda: bool(attempted))

        self.assertIs(observed.get("delivered"), False)
        self.assertIs(observed.get("refused"), True)
        self.assertEqual(delivered_ids(CONF), set())
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", stored)
        retry = claim_due_future_intentions(
            CONF, owner="retry-owner", now=NOW + timedelta(seconds=700)
        )
        self.assertEqual([item["id"] for item in retry], [row["id"]])

    async def test_retry_after_failed_attempt_delivers_exactly_once(self):
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid)
        transport = _ProbeTransport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)
        handler.REMINDER_RETRY_DELAY_SECONDS = 0.05
        attempts = []

        async def first_attempt(**_kwargs):
            attempts.append("failed")
            raise RuntimeError("provider failed before speaking")

        await self._run_timer(handler, first_attempt, lambda: bool(attempts))
        self.assertEqual(self._row(row["id"])["status"], STATUS_PENDING)

        await asyncio.sleep(0.1)
        delivered = []

        async def second_attempt(**kwargs):
            delivered.append(kwargs["metadata"].get("proactive_due_reminders"))
            await kwargs["websocket_send"](_delivery_frame())
            return PAYLOAD

        await self._run_timer(
            handler, second_attempt, lambda: self._row(row["id"])["status"] == STATUS_DONE
        )

        self.assertEqual(delivered, [[PAYLOAD]])
        self.assertEqual(len(transport.frames), 1)
        self.assertEqual(len(self._receipt_lines()), 1)
        self._assert_no_claim_possible(row)


# --------------------------------------------------------------------------- #
# Real turn (real prompt, real TTS sender, faked transport/provider)
# --------------------------------------------------------------------------- #


class RealTurnErrorPathTest(_Base):
    """``process_single_conversation`` is NOT stubbed in this class."""

    async def _drain_finalize(self, tts_manager=None, **_kwargs):
        """Stand-in for the playback wait: flush queued payloads, then return."""
        for task in list(getattr(tts_manager, "task_list", None) or []):
            await asyncio.gather(task, return_exceptions=True)
        for _ in range(100):
            if getattr(tts_manager, "_payload_queue", None) is None:
                break
            if tts_manager._payload_queue.empty():
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)

    async def _run_real(self, handler, ready, *, timeout=10):
        machine = _machine()
        state = machine.new_state(self.history_uid)
        with patch(
            "src.open_llm_vtuber.conversations.single_conversation."
            "finalize_conversation_turn",
            new=self._drain_finalize,
        ):
            task = asyncio.create_task(
                handler._run_proactive_timer("client", self.history_uid, state, machine)
            )
            deadline = asyncio.get_running_loop().time() + timeout
            while not ready() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.1)
            if task.done():
                await task
            else:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

    async def test_provider_error_after_a_sentence_does_not_resend(self):
        """The production duplicate: a sentence went out, then the stream broke.

        ``process_single_conversation`` empties ``full_response`` for a provider
        error, so the old ``bool(response)`` settlement treated a reminder the
        user had already heard as undelivered and released it for a full
        resend.
        """
        row = self._seed()
        agent, llm = _make_agent(
            CONF, self.history_uid, llm_cls=_SentenceThenProviderErrorLLM
        )
        transport = _ProbeTransport()
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run_real(handler, lambda: bool(delivered_ids(CONF)))

        content = [
            frame
            for frame in transport.frames
            if frame.get("type") == "audio"
            and (frame.get("display_text") or {}).get("text")
        ]
        self.assertTrue(content, "no content frame reached the transport")
        self.assertIn("Aku ingetin minum obat ya", content[0]["display_text"]["text"])
        # The reminder is settled exactly once and cannot be claimed again.
        self.assertEqual(delivered_ids(CONF), {row["id"]})
        self.assertEqual(self._row(row["id"])["status"], STATUS_DONE)
        self._assert_no_claim_possible(row)

        # A later scheduler cycle (fresh process) must not speak it again.
        restarted_agent, _llm = _make_agent(CONF, self.history_uid)
        restarted_transport = _ProbeTransport()
        restarted = _make_handler(
            restarted_agent, CONF, self.history_uid, restarted_transport
        )
        await self._run_real(restarted, lambda: False, timeout=2)
        self.assertEqual(restarted_transport.frames, [])

    async def test_dead_socket_turn_does_not_complete_an_undelivered_reminder(self):
        """A truthy turn result is not delivery: the sender swallowed a refusal."""
        row = self._seed()
        agent, _llm = _make_agent(CONF, self.history_uid)
        transport = _ProbeTransport(refuse_content=True)
        handler = _make_handler(agent, CONF, self.history_uid, transport)

        await self._run_real(
            handler,
            lambda: bool(handler._reminder_row_retry_not_before.get("client")),
        )

        self.assertTrue(transport.refused, "the transport never refused a frame")
        self.assertEqual(
            [frame for frame in transport.frames if frame.get("type") == "audio"], []
        )
        self.assertEqual(delivered_ids(CONF), set())
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", stored)
        retry = claim_due_future_intentions(
            CONF, owner="retry-owner", now=NOW + timedelta(seconds=700)
        )
        self.assertEqual([item["id"] for item in retry], [row["id"]])


if __name__ == "__main__":
    unittest.main()
