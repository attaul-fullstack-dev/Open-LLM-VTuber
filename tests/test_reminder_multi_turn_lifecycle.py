"""Multi-turn reminder lifecycle: inter-turn consistency and duplicate sends.

One temporary data directory, a synthetic ``conf_uid``, the real character
state store, the real claim/lease/receipt/completion code, the real scheduler
loop and the real prompt builders. Only the provider turn and the websocket
transport are faked, so these tests prove the *delivery path and its state
transitions across turns*, not audio output or a live LLM.

Turns covered, in order:

1. a reminder captured in a chat turn is stored with its payload and due time;
2. a later chat turn (still before due) keeps it pending and never claims it
   was delivered, and the row is rendered once per turn;
3. when it comes due while a chat turn is in flight, the chat prompt stays
   context-only and the scheduler delivers the payload exactly once afterwards;
4. the next chat turn neither resends it nor claims an outstanding obligation;
5. a failed completion write is reconciled from the durable receipt, so a
   restart does not deliver again;
6. a cancel before any assistant-content frame stays retryable, while a
   cancel after the reminder was handed to the transport settles from a
   durable receipt and is never replayed (a hard process crash before that
   receipt stays at-least-once; exactly-once is not claimed, because the
   transport has no delivery ack);
7. two processes: the second cannot deliver again after the first one's
   completion write failed;
8. future plans stay context and are never handed to the model as reminders.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.character_state import (
    capture_reminder,
    claim_due_future_intentions,
    load_character_state,
    save_character_state,
)
from src.open_llm_vtuber.chat_history_manager import create_new_history, store_message
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.conversations.conversation_utils import create_batch_input
from src.open_llm_vtuber.future_intentions import (
    KIND_PLAN,
    KIND_REMINDER,
    STATUS_DONE,
    STATUS_PENDING,
    build_future_intention_context,
    pending_intentions,
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
CONF = "mili-multi-turn"
PAYLOAD = "WAKTUNYA MINUM OBAT"
REMINDER_TEXT = "ingetin aku minum obat ya"
PLAN_TEXT = "besok aku ada ujian statistik"
PLAN_DETAIL = "ujian statistik jam 9"
SCHEDULER_CUE = "scheduled reminder is due RIGHT NOW"
CHAT_MARKER = "backend scheduler owns delivery"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime.now(timezone.utc).replace(microsecond=0)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class _FakeLLM:
    """Records every system prompt; emits one short assistant reply."""

    model = "multi-turn-test"
    max_tokens = 100

    def __init__(self):
        self.calls = []
        self.reply = "Iya, aku di sini."

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "system": system or "",
            }
        )
        yield self.reply


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


def _make_agent(conf_uid, history_uid):
    llm = _FakeLLM()
    agent = BasicMemoryAgent(
        llm=llm,
        system="persona Mili tetap aktif",
        live2d_model=_FakeLive2D(),
        tts_preprocessor_config=TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={"translate_audio": False, "translate_provider": "deeplx"},
        ),
        context_window_override=2200,
    )
    agent.set_memory_from_history(conf_uid, history_uid)
    return agent, llm


async def _chat_turn(agent, text="Aku baru pulang."):
    """A real user turn: real prompt composition, faked provider."""
    batch = create_batch_input(text, None, "User")
    return [output async for output in agent.chat(batch)]


def _make_handler(agent, conf_uid, history_uid, *, blocked_gate=True):
    """Real WebSocketHandler scheduler surface with a deterministic gate."""
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
    handler.client_connections = {"client": SimpleNamespace(send_text=AsyncMock())}
    handler._proactive_maintenance = set()
    handler._proactive_timer_tasks = {}
    handler.current_conversation_tasks = {}
    handler._proactive_states = {}
    handler._proactive_machines = {}
    handler._reminder_row_retry_not_before = {}
    handler.chat_group_manager = SimpleNamespace(get_client_group=lambda _uid: None)
    if blocked_gate:
        # Spontaneous conversation is fully suppressed; only the user
        # obligation (due reminder) lane may produce a turn.
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
    else:
        # Permissive gate: a due plan (HIGH trigger, no reminder lane) may fire.
        handler._proactive_gate_config = ProactiveGateConfig(
            quiet_hours_start_hour=0,
            quiet_hours_end_hour=0,
            minimum_proactive_gap_seconds=0,
        )
        handler._budget_state = lambda _uid, _conf: ProactiveBudgetState()
    handler._persist_budget = lambda _uid, _conf: True
    handler._proactive_intent_signals = staticmethod(
        lambda _context, _state: ProactiveIntentSignals()
    )
    handler._meaningful_life_event = staticmethod(lambda _context, _since: False)
    return handler


def _machine(initial_idle=30):
    return ProactiveStateMachine(
        ProactiveChatConfig(
            initial_idle_min_seconds=initial_idle,
            initial_idle_max_seconds=initial_idle,
            followup_idle_min_seconds=100,
            followup_idle_max_seconds=100,
        ),
        randint=lambda minimum, _maximum: minimum,
    )


def _due_deviation(agent, llm, sent):
    """Scheduler turn stand-in: real proactive prompt, recorded delivery.

    Mirrors ``process_single_conversation`` for the proactive path: it takes
    the dispatcher's ``proactive_due_reminders`` metadata and feeds it into the
    real ``chat_proactively`` prompt builder, then returns the streamed text
    (the dispatcher's "delivered" boundary).
    """

    async def fake_turn(**kwargs):
        metadata = kwargs.get("metadata") or {}
        due = list(metadata.get("proactive_due_reminders") or [])
        record = {
            "due": due,
            "status_at_prompt": {
                row["id"]: row["status"]
                for row in load_character_state(CONF).future_intentions
            },
            "claimed_at_prompt": {
                row["id"]: row.get("claimed_by")
                for row in load_character_state(CONF).future_intentions
            },
        }
        async for _ in agent.chat_proactively(due_reminders=due or None):
            pass
        record["prompt"] = llm.calls[-1]["system"]
        record["text"] = llm.reply
        sent.append(record)
        return llm.reply

    return fake_turn


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

    # -- store helpers ----------------------------------------------------- #

    def _seed(self, rows):
        state = load_character_state(CONF)
        state.future_intentions = [*(state.future_intentions or []), *rows]
        self.assertTrue(save_character_state(CONF, state))
        return rows

    def _row(self, intention_id):
        return next(
            row
            for row in load_character_state(CONF).future_intentions
            if row.get("id") == intention_id
        )

    def _reminder(self, *, due_delta=7200, text=REMINDER_TEXT, payload=PAYLOAD):
        return {
            "id": uuid.uuid4().hex,
            "text": text,
            "due_at": (NOW + timedelta(seconds=due_delta)).isoformat(),
            "created_at": (NOW - timedelta(hours=3)).isoformat(),
            "status": STATUS_PENDING,
            "source": "conversation",
            "kind": KIND_REMINDER,
            "tz": JKT,
            "delivery_text": payload,
        }

    def _plan(self, *, due_delta=86400, text=PLAN_TEXT, detail=PLAN_DETAIL):
        return {
            "id": uuid.uuid4().hex,
            "text": text,
            "due_at": (NOW + timedelta(seconds=due_delta)).isoformat(),
            "created_at": (NOW - timedelta(hours=3)).isoformat(),
            "status": STATUS_PENDING,
            "source": "conversation",
            "kind": KIND_PLAN,
            "tz": JKT,
            "delivery_text": detail,
        }

    def _receipt_lines(self):
        path = get_delivery_journal_path(CONF)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as handle:
            return [line for line in handle if line.strip()]

    # -- scheduler helpers ------------------------------------------------- #

    async def _run_scheduler(self, handler, sent, *, seconds=2.5, initial_idle=1):
        """Start the real timer, let it run for ``seconds``, cancel it."""
        machine = _machine(initial_idle=initial_idle)
        state = machine.new_state(self.history_uid)
        task = asyncio.create_task(
            handler._run_proactive_timer("client", self.history_uid, state, machine)
        )
        try:
            await asyncio.sleep(seconds)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        return sent


# --------------------------------------------------------------------------- #
# 1 + 2: capture, then a chat turn before due
# --------------------------------------------------------------------------- #


class CaptureThenChatBeforeDueTest(_Base):
    async def test_capture_is_verified_then_chat_keeps_it_pending(self):
        local_due = datetime.now(timezone.utc).astimezone(ZoneInfo(JKT)) + timedelta(
            hours=2
        )
        user_text = (
            f"Nanti jam {local_due:%H.%M} ingetin aku buat bilang '{PAYLOAD}'."
        )
        status, row = capture_reminder(CONF, user_text, now=NOW, tz=JKT)
        self.assertEqual(status, "captured", row)
        self.assertEqual(row["kind"], KIND_REMINDER)
        self.assertEqual(row["status"], STATUS_PENDING)
        self.assertEqual(row["delivery_text"], PAYLOAD)
        due = datetime.fromisoformat(row["due_at"])
        self.assertEqual(due.astimezone(ZoneInfo(JKT)).strftime("%H:%M"),
                         local_due.strftime("%H:%M"))
        self.assertGreater(due, NOW)
        # One capture, one row: no duplicate from the post-turn observer path.
        self.assertEqual(len(pending_intentions(load_character_state(CONF).future_intentions)), 1)

        agent, llm = _make_agent(CONF, self.history_uid)

        # Turn 1: normal chat, before due.
        await _chat_turn(agent, "Aku baru pulang.")
        first_prompt = llm.calls[-1]["system"]
        self.assertIn(row["text"], first_prompt)
        self.assertIn("stored reminder, not yet due", first_prompt)
        # The captured payload is carried by the user's own sentence here, so
        # the renderer does not repeat it as a separate suffix.
        self.assertIn(PAYLOAD, first_prompt)
        self.assertNotIn(SCHEDULER_CUE, first_prompt)
        self.assertNotIn("Deliver it in this message", first_prompt)
        self.assertEqual(self._row(row["id"])["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", self._row(row["id"]))

        # Turn 2: still pending, still rendered exactly once, never claimed.
        await _chat_turn(agent, "Hari ini capek banget.")
        second_prompt = llm.calls[-1]["system"]
        self.assertEqual(second_prompt.count(row["text"]), 1)
        self.assertNotIn(SCHEDULER_CUE, second_prompt)
        self.assertEqual(self._row(row["id"])["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", self._row(row["id"]))
        self.assertEqual(delivered_ids(CONF), set())
        self.assertIsNone(agent._reminder_receipt_block or None)


# --------------------------------------------------------------------------- #
# 3: due while a chat turn is in flight
# --------------------------------------------------------------------------- #


class DueDuringChatTest(_Base):
    async def test_due_reminder_during_chat_is_delivered_once_afterwards(self):
        row = self._reminder(due_delta=-30)
        self._seed([row])
        agent, llm = _make_agent(CONF, self.history_uid)
        handler = _make_handler(agent, CONF, self.history_uid)
        sent = []
        chat_prompts = []
        chat_started = asyncio.Event()
        release_chat = asyncio.Event()

        async def chat_task():
            await release_chat.wait()
            chat_started.set()
            chat_prompts.append(
                [str(item) for item in await _chat_turn(agent, "Aku lagi ngerjain tugas.")]
            )
            chat_prompts.append(llm.calls[-1]["system"])

        inflight = asyncio.create_task(chat_task())
        handler.current_conversation_tasks["client"] = inflight

        machine = _machine(initial_idle=30)
        state = machine.new_state(self.history_uid)
        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=_due_deviation(agent, llm, sent),
        ):
            timer = asyncio.create_task(
                handler._run_proactive_timer("client", self.history_uid, state, machine)
            )
            # 1. The scheduler must not deliver while the turn is in flight.
            await asyncio.sleep(0.6)
            self.assertEqual(sent, [])
            self.assertEqual(self._row(row["id"])["status"], STATUS_PENDING)
            self.assertNotIn("claimed_by", self._row(row["id"]))

            release_chat.set()
            await asyncio.wait_for(chat_started.wait(), 3)
            await inflight

            # 2. The chat prompt is context only: no payload, no delivery cue.
            chat_prompt = chat_prompts[-1]
            self.assertNotIn(PAYLOAD, chat_prompt)
            self.assertNotIn(SCHEDULER_CUE, chat_prompt)
            self.assertIn(CHAT_MARKER, chat_prompt)

            # 3. The scheduler delivers afterwards, exactly once.
            deadline = asyncio.get_running_loop().time() + 6
            while not sent and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.6)
            timer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await timer

        self.assertEqual(len(sent), 1, sent)
        self.assertEqual(sent[0]["due"], [PAYLOAD])
        self.assertEqual(sent[0]["status_at_prompt"][row["id"]], STATUS_PENDING)
        self.assertEqual(sent[0]["prompt"].count(PAYLOAD), 1)
        self.assertIn(SCHEDULER_CUE, sent[0]["prompt"])
        self.assertNotIn(CHAT_MARKER, sent[0]["prompt"])

        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_DONE)
        self.assertNotIn("claimed_by", stored)
        self.assertEqual(delivered_ids(CONF), {row["id"]})
        self.assertEqual(len(self._receipt_lines()), 1)


# --------------------------------------------------------------------------- #
# 4: the turns after a successful delivery
# --------------------------------------------------------------------------- #


class AfterDeliveryTest(_Base):
    async def test_next_turn_does_not_resend_or_claim_an_obligation(self):
        row = self._reminder(due_delta=-30)
        self._seed([row])
        agent, llm = _make_agent(CONF, self.history_uid)
        handler = _make_handler(agent, CONF, self.history_uid)
        sent = []

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=_due_deviation(agent, llm, sent),
        ):
            await self._run_scheduler(handler, sent, seconds=1.2)

        self.assertEqual(len(sent), 1)
        self.assertEqual(self._row(row["id"])["status"], STATUS_DONE)

        # The next ordinary chat turn: nothing about the reminder at all.
        await _chat_turn(agent, "Makasih ya tadi.")
        prompt = llm.calls[-1]["system"]
        self.assertNotIn(row["text"], prompt)
        self.assertNotIn(PAYLOAD, prompt)
        self.assertNotIn(CHAT_MARKER, prompt)
        # No stale pre-turn receipt may survive into a later turn.
        self.assertIsNone(agent._reminder_receipt_block or None)
        # And no later scheduler cycle may resend it.
        await self._run_scheduler(handler, sent, seconds=2.0, initial_idle=1)
        self.assertEqual(len(sent), 1)
        self.assertEqual(
            claim_due_future_intentions(
                CONF, owner="late-owner", now=NOW + timedelta(seconds=700)
            ),
            [],
        )


# --------------------------------------------------------------------------- #
# 5: failed completion write, then process recovery
# --------------------------------------------------------------------------- #


class FailedCompletionRecoveryTest(_Base):
    async def test_failed_completion_is_reconciled_and_never_resent(self):
        row = self._reminder(due_delta=-30)
        self._seed([row])
        agent, llm = _make_agent(CONF, self.history_uid)
        handler = _make_handler(agent, CONF, self.history_uid)
        sent = []
        completion_calls = []

        def failing_completion(owner, intention_ids):
            completion_calls.append((owner, list(intention_ids)))
            return 0  # the completion write did not persist

        agent.complete_claimed_future_intentions = failing_completion

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=_due_deviation(agent, llm, sent),
        ):
            # Long enough for the loop to re-enter: the still-pending row must
            # never be delivered a second time just because its completion
            # write failed.
            await self._run_scheduler(handler, sent, seconds=1.6)

        self.assertEqual(len(sent), 1, "a delivered reminder was delivered twice")
        self.assertEqual(len(completion_calls), 1)
        self.assertEqual(delivered_ids(CONF), {row["id"]})
        self.assertEqual(len(self._receipt_lines()), 1)
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_DONE)
        self.assertNotIn("claimed_by", stored)

        # Reconstruct the exact moment the failure left behind on disk
        # (delivery durable, completion write lost, claim still live) and
        # recover it from a fresh process. The receipt alone must stop the
        # resend here -- nothing in memory is shared with this attempt.
        state = load_character_state(CONF)
        for item in state.future_intentions:
            if item.get("id") == row["id"]:
                item["status"] = STATUS_PENDING
                item["claimed_by"] = "dead-owner"
                item["claim_until"] = (NOW + timedelta(seconds=600)).isoformat()
        self.assertTrue(save_character_state(CONF, state))
        self.assertEqual(self._row(row["id"])["status"], STATUS_PENDING)

        # "Restart": a brand-new agent/handler in the same data directory.
        restarted_agent, restarted_llm = _make_agent(CONF, self.history_uid)
        restarted_handler = _make_handler(restarted_agent, CONF, self.history_uid)
        resent = []
        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=_due_deviation(restarted_agent, restarted_llm, resent),
        ):
            await self._run_scheduler(
                restarted_handler, resent, seconds=2.0, initial_idle=1
            )

        self.assertEqual(resent, [], "a received reminder must not be sent twice")
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_DONE)
        self.assertNotIn("claimed_by", stored)
        self.assertEqual(len(self._receipt_lines()), 1)


# --------------------------------------------------------------------------- #
# 6: cancel/interrupt during delivery (the duplicate-send regression)
# --------------------------------------------------------------------------- #


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


async def _async_noop(_payload):
    return None


async def _raising_send(_payload):
    raise RuntimeError("transport refused the frame")


class DeliveryProbeTest(unittest.TestCase):
    """The probe counts only evidence that the words left this process."""

    def test_control_and_bookkeeping_frames_do_not_count(self):
        for payload in (
            json.dumps({"type": "control", "text": "conversation-chain-start"}),
            json.dumps({"type": "latency-event", "phase": "agent_start"}),
            json.dumps({"type": "tool_call_status", "status": "running"}),
            json.dumps({"type": "error", "message": "boom"}),
            json.dumps({"type": "history-auto-titled", "title": "x"}),
            # A silent display payload with no text and no audio proves nothing.
            json.dumps({"type": "audio", "audio": None, "display_text": {"text": ""}}),
            "not-json",
        ):
            probe = _ProactiveDeliveryProbe(_async_noop)
            asyncio.run(probe(payload))
            self.assertFalse(probe.delivered, payload)
            self.assertEqual(probe.frames, 1)

    def test_content_frame_counts_after_a_successful_handoff(self):
        probe = _ProactiveDeliveryProbe(_async_noop)
        asyncio.run(probe(_delivery_frame()))
        self.assertTrue(probe.delivered)
        self.assertEqual(probe.content_frames, 1)

    def test_failed_handoff_is_not_counted_as_delivered(self):
        probe = _ProactiveDeliveryProbe(_raising_send)
        with self.assertRaises(RuntimeError):
            asyncio.run(probe(_delivery_frame()))
        self.assertFalse(probe.delivered)
        self.assertEqual(probe.content_frames, 0)


class CancelDuringDeliveryTest(_Base):
    """A cancel must not replay a reminder the user already saw."""

    async def _run_cancelled_turn(self, turn):
        row = self._reminder(due_delta=-30)
        self._seed([row])
        agent, _llm = _make_agent(CONF, self.history_uid)
        handler = _make_handler(agent, CONF, self.history_uid)
        started = []

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=turn(agent, started),
        ):
            machine = _machine(initial_idle=1)
            state = machine.new_state(self.history_uid)
            task = asyncio.create_task(
                handler._run_proactive_timer("client", self.history_uid, state, machine)
            )
            deadline = asyncio.get_running_loop().time() + 5
            while not started and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.02)
            with self.assertRaises(asyncio.CancelledError):
                await task
        return row, agent, handler, started

    async def test_cancel_before_any_content_keeps_it_retryable(self):
        """No content frame -> nothing was delivered -> release and retry."""

        def turn(_agent, started):
            async def fake_turn(**_kwargs):
                started.append("cancelled")
                await asyncio.sleep(0)
                raise asyncio.CancelledError("interrupted during prompt build")

            return fake_turn

        row, _agent, handler, _ = await self._run_cancelled_turn(turn)

        self.assertEqual(delivered_ids(CONF), set())
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", stored)
        # Recovery is preserved: the row is immediately claimable again and
        # only this row is backed off.
        self.assertIn(row["id"], handler._reminder_row_retry_not_before["client"])
        retry = claim_due_future_intentions(
            CONF, owner="retry-owner", now=NOW + timedelta(seconds=120)
        )
        self.assertEqual([item["id"] for item in retry], [row["id"]])

    async def test_cancel_after_content_delivers_once_and_never_replays(self):
        """The sentence left the process -> receipt -> no second send."""

        def turn(_agent, started):
            async def fake_turn(**kwargs):
                await kwargs["websocket_send"](_delivery_frame())
                started.append(PAYLOAD)
                await asyncio.sleep(0)
                raise asyncio.CancelledError("user interrupt after the sentence")

            return fake_turn

        row, agent, handler, started = await self._run_cancelled_turn(turn)

        self.assertEqual(started, [PAYLOAD])
        self.assertEqual(delivered_ids(CONF), {row["id"]})
        self.assertEqual(len(self._receipt_lines()), 1)
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_DONE)
        self.assertNotIn("claimed_by", stored)
        # A later cancel/retry cycle -- and even a different owner -- cannot
        # claim it again, so the same reminder never reaches the user twice.
        self.assertEqual(
            claim_due_future_intentions(
                CONF, owner="other-owner", now=NOW + timedelta(seconds=700)
            ),
            [],
        )
        resent = []
        restarted_agent, restarted_llm = _make_agent(CONF, self.history_uid)
        restarted_handler = _make_handler(restarted_agent, CONF, self.history_uid)
        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=_due_deviation(restarted_agent, restarted_llm, resent),
        ):
            await self._run_scheduler(
                restarted_handler, resent, seconds=2.0, initial_idle=1
            )
        self.assertEqual(resent, [])

    async def test_cancel_after_content_with_failed_receipt_still_completes(self):
        """Receipt I/O failure must not reopen the resend window by itself."""

        def turn(agent, started):
            agent.record_reminder_delivery = lambda _owner, _ids: False

            async def fake_turn(**kwargs):
                await kwargs["websocket_send"](_delivery_frame())
                started.append(PAYLOAD)
                await asyncio.sleep(0)
                raise asyncio.CancelledError("user interrupt after the sentence")

            return fake_turn

        row, _agent, _handler, _ = await self._run_cancelled_turn(turn)

        self.assertEqual(delivered_ids(CONF), set())
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_DONE)
        self.assertNotIn("claimed_by", stored)
        self.assertEqual(
            claim_due_future_intentions(
                CONF, owner="other-owner", now=NOW + timedelta(seconds=700)
            ),
            [],
        )

    async def test_transport_failure_before_content_keeps_it_retryable(self):
        """A send that raised never reached the client: release, no receipt."""
        observed = {}
        started = asyncio.Event()

        async def failing_turn(**kwargs):
            probe = kwargs["websocket_send"]
            try:
                await probe(_delivery_frame())
            except RuntimeError:
                observed["probe_delivered"] = probe.delivered
                started.set()
                raise

        row = self._reminder(due_delta=-30)
        self._seed([row])
        agent, _llm = _make_agent(CONF, self.history_uid)
        handler = _make_handler(agent, CONF, self.history_uid)
        handler.client_connections = {
            "client": SimpleNamespace(
                send_text=AsyncMock(side_effect=RuntimeError("socket gone"))
            )
        }

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=failing_turn,
        ):
            machine = _machine(initial_idle=1)
            state = machine.new_state(self.history_uid)
            task = asyncio.create_task(
                handler._run_proactive_timer("client", self.history_uid, state, machine)
            )
            await asyncio.wait_for(started.wait(), 5)
            # A transport exception is handled (not re-raised): the timer's
            # failure path runs and the task ends normally.
            await asyncio.wait_for(task, 5)

        self.assertIs(observed.get("probe_delivered"), False)
        self.assertEqual(delivered_ids(CONF), set())
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", stored)
        self.assertEqual(
            [
                item["id"]
                for item in claim_due_future_intentions(
                    CONF, owner="retry-owner", now=NOW + timedelta(seconds=120)
                )
            ],
            [row["id"]],
        )

    async def test_restart_after_cancelled_delivery_does_not_resend(self):
        """Receipt written, completion lost: a fresh process reconciles."""

        def turn(_agent, started):
            async def fake_turn(**kwargs):
                await kwargs["websocket_send"](_delivery_frame())
                started.append(PAYLOAD)
                await asyncio.sleep(0)
                raise asyncio.CancelledError("killed between receipt and completion")

            return fake_turn

        row, _agent, _handler, _ = await self._run_cancelled_turn(turn)
        self.assertEqual(delivered_ids(CONF), {row["id"]})

        # Rebuild the exact post-crash state: delivered (receipt) but still
        # pending and still claimed by the dead attempt.
        state = load_character_state(CONF)
        for item in state.future_intentions:
            if item.get("id") == row["id"]:
                item["status"] = STATUS_PENDING
                item["claimed_by"] = "dead-owner"
                item["claim_until"] = (NOW + timedelta(seconds=600)).isoformat()
        self.assertTrue(save_character_state(CONF, state))

        restarted_agent, restarted_llm = _make_agent(CONF, self.history_uid)
        restarted_handler = _make_handler(restarted_agent, CONF, self.history_uid)
        resent = []
        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=_due_deviation(restarted_agent, restarted_llm, resent),
        ):
            await self._run_scheduler(
                restarted_handler, resent, seconds=2.0, initial_idle=1
            )

        self.assertEqual(resent, [])
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_DONE)
        self.assertNotIn("claimed_by", stored)
        self.assertEqual(len(self._receipt_lines()), 1)


# --------------------------------------------------------------------------- #
# 7: two processes, second one after a failed completion
# --------------------------------------------------------------------------- #

CHILD_SCRIPT = r"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

import src.open_llm_vtuber.character_state as cs
from src.open_llm_vtuber.reminder_delivery import record_delivery

conf = payload["conf"]
now = datetime.fromisoformat(payload["now"])
owner = payload["owner"]
result = {"owner": owner, "claimed": [], "sent": 0, "completed": 0, "receipt": False}

rows = cs.claim_due_future_intentions(conf, owner=owner, now=now, max_claims=1)
result["claimed"] = [row["id"] for row in rows]
if rows:
    ids = result["claimed"]
    with open(payload["sends_path"], "a", encoding="utf-8") as handle:
        handle.write("\n".join(f"{owner} {row_id}" for row_id in ids) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    result["sent"] = len(ids)
    result["receipt"] = record_delivery(conf, ids, owner=owner, now=now)
    if payload.get("complete") == "fail":
        original_save = cs.save_character_state
        cs.save_character_state = lambda *args, **kwargs: False
        try:
            result["completed"] = cs.complete_claimed_future_intentions(
                conf, owner=owner, intention_ids=ids
            )
        finally:
            cs.save_character_state = original_save
    else:
        result["completed"] = cs.complete_claimed_future_intentions(
            conf, owner=owner, intention_ids=ids
        )

with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"""


class CrossProcessRecoveryTest(_Base):
    def _run_child(self, payload):
        process = subprocess.run(
            [sys.executable, "-c", CHILD_SCRIPT, ROOT, os.getcwd(), json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            process.returncode, 0, f"child failed: {process.stderr[-2000:]}"
        )
        with open(payload["result_path"], "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_second_process_cannot_deliver_after_failed_completion(self):
        row = self._reminder(due_delta=0)
        row["due_at"] = (NOW - timedelta(minutes=1)).isoformat()
        self._seed([row])
        sends = os.path.join(os.getcwd(), "sends.log")

        first = self._run_child(
            {
                "conf": CONF,
                "now": (NOW + timedelta(seconds=5)).isoformat(),
                "owner": "proc-a",
                "sends_path": sends,
                "result_path": os.path.join(os.getcwd(), "a.json"),
                "complete": "fail",
            }
        )
        self.assertEqual(first["claimed"], [row["id"]])
        self.assertEqual(first["sent"], 1)
        self.assertTrue(first["receipt"])
        self.assertEqual(first["completed"], 0)
        self.assertEqual(self._row(row["id"])["status"], STATUS_PENDING)

        # Process B starts later, after the lease, and must not send again.
        second = self._run_child(
            {
                "conf": CONF,
                "now": (NOW + timedelta(seconds=700)).isoformat(),
                "owner": "proc-b",
                "sends_path": sends,
                "result_path": os.path.join(os.getcwd(), "b.json"),
                "complete": "ok",
            }
        )
        self.assertEqual(second["claimed"], [])
        self.assertEqual(second["sent"], 0)
        with open(sends, "r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        self.assertEqual(len(lines), 1, lines)
        self.assertEqual(self._row(row["id"])["status"], STATUS_DONE)
        self.assertEqual(delivered_ids(CONF), {row["id"]})


# --------------------------------------------------------------------------- #
# 8: plans stay context, never reminders
# --------------------------------------------------------------------------- #


class PlanContextTest(_Base):
    async def test_plan_is_context_and_never_delivered_as_a_reminder(self):
        reminder = self._reminder(due_delta=-30)
        plan = self._plan(due_delta=3600)
        self._seed([reminder, plan])
        agent, llm = _make_agent(CONF, self.history_uid)

        # Chat turn: both are context, only the reminder is a reminder.
        await _chat_turn(agent, "Aku mau belajar dulu.")
        prompt = llm.calls[-1]["system"]
        self.assertIn(plan["text"], prompt)
        self.assertIn("upcoming user plan, for context only", prompt)
        self.assertIn(reminder["text"], prompt)
        self.assertNotIn(PAYLOAD, prompt)
        self.assertNotIn(SCHEDULER_CUE, prompt)

        # Scheduler turn for the due reminder: the plan stays context only.
        handler = _make_handler(agent, CONF, self.history_uid)
        sent = []
        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=_due_deviation(agent, llm, sent),
        ):
            await self._run_scheduler(handler, sent, seconds=1.2)

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["due"], [PAYLOAD])
        self.assertNotIn(plan["text"], sent[0]["due"])
        self.assertIn(plan["text"], sent[0]["prompt"])
        self.assertIn("upcoming user plan, for context only", sent[0]["prompt"])
        self.assertEqual(self._row(plan["id"])["status"], STATUS_PENDING)

        # A due plan alone is not a reminder the scheduler may deliver.
        due_plan = self._plan(due_delta=-45, text="rencana aku besok", detail=PLAN_DETAIL)
        self._seed([due_plan])
        claimed = claim_due_future_intentions(
            CONF, owner="plan-owner", now=NOW + timedelta(seconds=5)
        )
        delivered_texts = [
            str(item.get("delivery_text") or item.get("text") or "").strip()
            for item in claimed
            if (item.get("kind") or KIND_REMINDER) == KIND_REMINDER
        ]
        self.assertEqual(delivered_texts, [])
        self.assertNotIn(due_plan["text"], delivered_texts)
        # It is still rendered to the model as plan context, never as an
        # instruction to remind the user.
        block = build_future_intention_context(
            load_character_state(CONF).future_intentions, tz=JKT
        )
        self.assertIn(due_plan["text"], block)
        self.assertIn("upcoming user plan, for context only", block)
        self.assertNotIn(SCHEDULER_CUE, block)
        self.assertNotIn("backend scheduler owns delivery", block)


if __name__ == "__main__":
    unittest.main()
