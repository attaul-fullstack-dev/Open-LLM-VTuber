"""Isolated end-to-end reminder lifecycle (mission section H).

One temp directory, synthetic ``conf_uid``, empty stores, real scheduler loop,
real claim/lease/complete logic and the real gate. Only the provider turn is
faked (no TTS/browser), so this proves the delivery *path*, not audio output.

Covered here: capture -> visible pending -> not before due -> fires with no new
user message while the spontaneous gate denies -> payload reaches the request
-> success completes exactly once -> row leaves the active list; failure keeps
it pending without a hot retry loop; two competing connections deliver once.

Typed provider error / partial-text-then-error are covered through the real
conversation pipeline in ``test_proactive_reminder_delivery``.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.character_state import (
    capture_reminder,
    load_character_state,
    save_character_state,
)
from src.open_llm_vtuber.chat_history_manager import create_new_history, store_message
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.future_intentions import pending_intentions
from src.open_llm_vtuber.proactive_chat import (
    ProactiveChatConfig,
    ProactiveIntentSignals,
    ProactiveStateMachine,
)
from src.open_llm_vtuber.proactive_gate import (
    ProactiveBudgetState,
    ProactiveGateConfig,
    TriggerReason,
    evaluate_gate,
)
from src.open_llm_vtuber.websocket_handler import WebSocketHandler

JKT = "Asia/Jakarta"
CONF = "e2e-reminder-synthetic"


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class _LLM:
    model = "e2e"
    max_tokens = 8

    async def chat_completion(self, *args, **kwargs):
        raise AssertionError("scheduler test must not call the provider directly")
        if False:
            yield None


def _agent(conf_uid, history_uid):
    agent = BasicMemoryAgent(
        llm=_LLM(),
        system="persona",
        live2d_model=_FakeLive2D(),
        tts_preprocessor_config=TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={"translate_audio": False, "translate_provider": "deeplx"},
        ),
    )
    agent.set_memory_from_history(conf_uid, history_uid)
    return agent


class ReminderLifecycleE2E(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self.conf_uid = CONF
        self.history_uid = create_new_history(self.conf_uid)
        store_message(self.conf_uid, self.history_uid, "human", "Halo Mili.")
        store_message(self.conf_uid, self.history_uid, "ai", "Halo juga.")
        self.agent = _agent(self.conf_uid, self.history_uid)
        self.handler = self._handler(self.agent)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _handler(self, agent):
        handler = WebSocketHandler.__new__(WebSocketHandler)
        handler.client_contexts = {
            "client": SimpleNamespace(
                history_uid=self.history_uid,
                agent_engine=agent,
                user_timezone=JKT,
                asr_engine=None,
                live2d_model=_FakeLive2D(),
                tts_engine=None,
                translate_engine=None,
                voice_output_enabled=False,
                character_config=SimpleNamespace(
                    conf_uid=self.conf_uid,
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
        # Spontaneous conversation is fully suppressed in this scenario:
        # quiet hours all day, one-hour gap, active dormancy/backoff. Only the
        # reminder lane may produce a turn.
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

    def _machine(self, initial_idle=30):
        return ProactiveStateMachine(
            ProactiveChatConfig(
                initial_idle_min_seconds=initial_idle,
                initial_idle_max_seconds=initial_idle,
                followup_idle_min_seconds=100,
                followup_idle_max_seconds=100,
            ),
            randint=lambda minimum, _maximum: minimum,
        )

    def _seed(self, due_in_seconds, delivery, text="Nanti jam 11.54 bilang kaya tadi lagi ya."):
        base = datetime.now(timezone.utc)
        state = load_character_state(self.conf_uid)
        row = {
            "id": uuid.uuid4().hex,
            "text": text,
            "due_at": (base + timedelta(seconds=due_in_seconds)).isoformat(),
            "created_at": base.isoformat(),
            "status": "pending",
            "source": "conversation",
            "kind": "reminder",
            "tz": JKT,
            "delivery_text": delivery,
        }
        state.future_intentions = [*(state.future_intentions or []), row]
        self.assertTrue(save_character_state(self.conf_uid, state))
        self.agent._character_state = load_character_state(self.conf_uid)
        return row

    async def test_capture_visibility_and_due_delivery(self):
        # 1. Explicit request, stored through the same pre-turn path.
        explicit_now = datetime.now(timezone.utc)
        status, explicit_row = capture_reminder(
            self.conf_uid,
            "Nanti jam 11.51 ingetin aku dengan bilang 'TES PENGINGAT BERHASIL'.",
            now=explicit_now,
            tz=JKT,
        )
        self.assertEqual(status, "captured")
        # 2. Informal referential request resolved from the conversation.
        status, referential_row = capture_reminder(
            self.conf_uid,
            "Nanti jam 11.54 bilang kaya tadi lagi ya.",
            now=explicit_now,
            tz=JKT,
            recent_messages=[
                {"role": "user", "content": "Nanti jam 11.51 ingetin aku dengan bilang 'TES PENGINGAT BERHASIL'."},
                {"role": "assistant", "content": "Oke, nanti jam 11.51 aku bilang 'TES PENGINGAT BERHASIL'."},
            ],
        )
        self.assertEqual(status, "captured")
        self.assertEqual(referential_row["delivery_text"], "TES PENGINGAT BERHASIL")
        active = pending_intentions(load_character_state(self.conf_uid).future_intentions)
        self.assertEqual(
            {row["id"] for row in active}, {explicit_row["id"], referential_row["id"]}
        )

        # Empty the store and seed a synthetic due time 2s from now, so the
        # real scheduler wake-up (not a fake sleep) is exercised.
        state = load_character_state(self.conf_uid)
        state.future_intentions = []
        save_character_state(self.conf_uid, state)
        self.agent._character_state = load_character_state(self.conf_uid)
        due_row = self._seed(2.0, "TES PENGINGAT BERHASIL")
        later_row = self._seed(600.0, "jangan terkirim dulu")

        machine = self._machine(initial_idle=30)
        state = machine.new_state(self.history_uid)
        calls = []
        delivered = asyncio.Event()

        async def fake_turn(**kwargs):
            calls.append(kwargs)
            row = next(
                item
                for item in load_character_state(self.conf_uid).future_intentions
                if item["id"] == due_row["id"]
            )
            calls[-1]["status_at_prompt"] = row["status"]
            delivered.set()
            return "TES PENGINGAT BERHASIL"

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=fake_turn,
        ):
            timer = asyncio.create_task(
                self.handler._run_proactive_timer("client", self.history_uid, state, machine)
            )
            # 3. Not delivered before due, even though the timer is running.
            await asyncio.sleep(1.0)
            self.assertEqual(calls, [])
            # 4. Fires with no user message, despite a 30s idle window.
            await asyncio.wait_for(delivered.wait(), 6)
            await asyncio.sleep(0.05)
            timer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await timer

        # 5. Payload reached the request; the row was still pending at prompt.
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["status_at_prompt"], "pending")
        self.assertEqual(calls[0]["metadata"]["proactive_due_reminders"], ["TES PENGINGAT BERHASIL"])
        # 6. Success completed exactly once and left the active list.
        rows = {row["id"]: row for row in load_character_state(self.conf_uid).future_intentions}
        self.assertEqual(rows[due_row["id"]]["status"], "done")
        self.assertEqual(rows[later_row["id"]]["status"], "pending")
        self.assertEqual(
            [row["id"] for row in pending_intentions(rows.values())],
            [later_row["id"]],
        )
        # 7. Spontaneous conversation is still gated in the same conditions.
        decision = evaluate_gate(
            ProactiveBudgetState(
                dormant_until=(datetime.now(timezone.utc) + timedelta(hours=5)).isoformat(),
            ),
            self.handler._proactive_gate_config,
            TriggerReason("high", "unfinished_topic", "unresolved user topic"),
            now=datetime.now(timezone.utc),
            tz=JKT,
        )
        self.assertFalse(decision.allowed)

    async def test_empty_response_keeps_pending_and_retries_are_paced(self):
        due_row = self._seed(0.0, "TES PENGINGAT BERHASIL")
        machine = self._machine(initial_idle=1)
        state = machine.new_state(self.history_uid)
        calls = []
        first = asyncio.Event()

        async def fake_turn(**kwargs):
            calls.append(kwargs)
            first.set()
            return ""

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=fake_turn,
        ):
            timer = asyncio.create_task(
                self.handler._run_proactive_timer("client", self.history_uid, state, machine)
            )
            await asyncio.wait_for(first.wait(), 5)
            await asyncio.sleep(0.6)
            timer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await timer

        row = next(
            item
            for item in load_character_state(self.conf_uid).future_intentions
            if item["id"] == due_row["id"]
        )
        self.assertEqual(row["status"], "pending")
        self.assertNotIn("claimed_by", row)
        self.assertEqual(len(calls), 1, "failed reminder must not hot-loop the provider")
        self.assertIn(
            due_row["id"], self.handler._reminder_row_retry_not_before["client"]
        )

    async def test_two_connections_deliver_once(self):
        due_row = self._seed(-30.0, "TES PENGINGAT BERHASIL")
        other_agent = _agent(self.conf_uid, self.history_uid)
        other_handler = self._handler(other_agent)
        other_handler.client_contexts = {
            "client2": SimpleNamespace(**{
                **other_handler.client_contexts["client"].__dict__,
                "agent_engine": other_agent,
            })
        }
        other_handler.client_connections = {"client2": SimpleNamespace(send_text=AsyncMock())}

        deliveries = []
        done = asyncio.Event()

        async def fake_turn(**kwargs):
            deliveries.append(kwargs["metadata"]["proactive_due_reminders"])
            done.set()
            return "TES PENGINGAT BERHASIL"

        machine_a = self._machine(initial_idle=1)
        machine_b = self._machine(initial_idle=1)
        state_a = machine_a.new_state(self.history_uid)
        state_b = machine_b.new_state(self.history_uid)

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=fake_turn,
        ):
            timers = [
                asyncio.create_task(
                    self.handler._run_proactive_timer("client", self.history_uid, state_a, machine_a)
                ),
                asyncio.create_task(
                    other_handler._run_proactive_timer("client2", self.history_uid, state_b, machine_b)
                ),
            ]
            await asyncio.wait_for(done.wait(), 5)
            await asyncio.sleep(0.4)
            for timer in timers:
                timer.cancel()
            for timer in timers:
                with self.assertRaises(asyncio.CancelledError):
                    await timer

        self.assertEqual(len(deliveries), 1)
        row = next(
            item
            for item in load_character_state(self.conf_uid).future_intentions
            if item["id"] == due_row["id"]
        )
        self.assertEqual(row["status"], "done")


if __name__ == "__main__":
    unittest.main()
