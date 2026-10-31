"""Reminder recovery across reconnect / backend restart (isolated fixtures).

What the architecture actually does, and what these tests pin down:

* The reminder store is the authoritative ``character_state`` file, so a
  pending row survives a process restart untouched.
* The proactive scheduler is per WebSocket connection and starts when a
  history is selected (``_activate_proactive_for_history``). A reconnect
  therefore gets a *fresh* timer, not a replay of the old one.
* Catch-up policy (as implemented, no new rule added here): any pending row
  whose ``due_at`` is in the past is delivered on the first scheduler tick
  after the backend/connection returns. There is no maximum age and no
  "missed" state, so a 3-hour-old reminder is still delivered.
* A row claimed by a process that died stays claimed until its lease
  (600 s by default) expires; only then may another attempt take it over.
  While a foreign lease is still live, a reconnected connection must not
  deliver a second copy.
* ``done`` rows are terminal: no restart or reconnect can claim them again.

Everything runs in a temp directory with a synthetic ``conf_uid``; the real
provider is replaced by a fake turn. This proves the recovery path, not real
audio/browser delivery.
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
    complete_claimed_future_intentions,
    load_character_state,
    save_character_state,
)
from src.open_llm_vtuber.chat_history_manager import create_new_history, store_message
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.proactive_chat import (
    ProactiveChatConfig,
    ProactiveIntentSignals,
    ProactiveStateMachine,
)
from src.open_llm_vtuber.proactive_gate import ProactiveBudgetState, ProactiveGateConfig
from src.open_llm_vtuber.websocket_handler import WebSocketHandler

JKT = "Asia/Jakarta"
CONF = "recovery-reminder-synthetic"


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class _LLM:
    model = "recovery"
    max_tokens = 8

    async def chat_completion(self, *args, **kwargs):
        raise AssertionError("scheduler test must not call the provider directly")
        if False:
            yield None


def _make_agent(conf_uid, history_uid):
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


class ReminderRecoveryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self.conf_uid = CONF
        self.history_uid = create_new_history(self.conf_uid)
        store_message(self.conf_uid, self.history_uid, "human", "Halo Mili.")
        store_message(self.conf_uid, self.history_uid, "ai", "Halo juga.")
        self.agent = _make_agent(self.conf_uid, self.history_uid)
        self.handler = self._handler(self.agent, "client")

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    # ---- fixtures -----------------------------------------------------

    def _handler(self, agent, client_uid):
        handler = WebSocketHandler.__new__(WebSocketHandler)
        handler.client_contexts = {
            client_uid: SimpleNamespace(
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
        handler.client_connections = {
            client_uid: SimpleNamespace(send_text=AsyncMock())
        }
        handler._proactive_maintenance = set()
        handler._proactive_timer_tasks = {}
        handler.current_conversation_tasks = {}
        handler._proactive_states = {}
        handler._proactive_machines = {}
        handler._reminder_row_retry_not_before = {}
        handler.chat_group_manager = SimpleNamespace(get_client_group=lambda _uid: None)
        # Reminder lane only: spontaneous conversation is fully suppressed.
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

    def _restart(self, client_uid="client"):
        """Simulate a backend restart: brand-new agent + handler, no memory."""
        agent = _make_agent(self.conf_uid, self.history_uid)
        return agent, self._handler(agent, client_uid)

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

    def _seed(
        self,
        *,
        due_delta,
        status="pending",
        claimed_by=None,
        claim_delta=None,
        text="ingetin aku minum obat ya",
        delivery="minum obat",
    ):
        base = datetime.now(timezone.utc)
        state = load_character_state(self.conf_uid)
        row = {
            "id": uuid.uuid4().hex,
            "text": text,
            "due_at": (base + timedelta(seconds=due_delta)).isoformat(),
            "created_at": base.isoformat(),
            "status": status,
            "source": "conversation",
            "kind": "reminder",
            "tz": JKT,
            "delivery_text": delivery,
        }
        if claimed_by:
            row["claimed_by"] = claimed_by
            row["claim_until"] = (
                base + timedelta(seconds=claim_delta or 0)
            ).isoformat()
        state.future_intentions = [*(state.future_intentions or []), row]
        self.assertTrue(save_character_state(self.conf_uid, state))
        return row

    def _row(self, row_id):
        return next(
            item
            for item in load_character_state(self.conf_uid).future_intentions
            if item["id"] == row_id
        )

    async def _run(self, handler, machine, state, client_uid="client", until=None, seconds=0.0):
        """Run the scheduler, optionally until an event, then cancel it."""
        deliveries = []

        async def fake_turn(**kwargs):
            deliveries.append(kwargs)
            if until is not None:
                until.set()
            return "minum obat"

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=fake_turn,
        ):
            timer = asyncio.create_task(
                handler._run_proactive_timer(client_uid, self.history_uid, state, machine)
            )
            try:
                if until is not None:
                    await asyncio.wait_for(until.wait(), 6)
                if seconds:
                    await asyncio.sleep(seconds)
            finally:
                timer.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await timer
        return deliveries

    # ---- A. pending survives restart, delivered at due -----------------

    async def test_pending_reminder_survives_restart_and_fires_at_due(self):
        row = self._seed(due_delta=2.5)
        machine = self._machine()
        state = machine.new_state(self.history_uid)

        # Old process shuts down before the due time.
        delivered_before = await self._run(
            self.handler, machine, state, seconds=0.4
        )
        self.assertEqual(delivered_before, [])
        self.assertEqual(self._row(row["id"])["status"], "pending")
        self.assertNotIn("claimed_by", self._row(row["id"]))

        # Restart: new agent, new handler, no in-memory state.
        agent, handler = self._restart()
        overview = agent.list_memory_overview()
        self.assertEqual([item["id"] for item in overview["intentions"]], [row["id"]])

        delivered = asyncio.Event()
        machine = self._machine(initial_idle=30)  # due time, not idle, wakes it
        state = machine.new_state(self.history_uid)
        calls = await self._run(handler, machine, state, until=delivered)

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["metadata"]["proactive_due_reminders"], ["minum obat"]
        )
        self.assertEqual(self._row(row["id"])["status"], "done")

    # ---- B. overdue while the backend was down: catch-up ----------------

    async def test_overdue_reminder_is_caught_up_after_restart(self):
        # Documented policy: no max age, no "missed" state. A 3-hour-old
        # pending reminder is delivered on the first tick after recovery.
        row = self._seed(due_delta=-3 * 3600)
        agent, handler = self._restart()
        delivered = asyncio.Event()
        machine = self._machine()
        state = machine.new_state(self.history_uid)

        calls = await self._run(handler, machine, state, until=delivered)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["metadata"]["proactive_due_reminders"], ["minum obat"])
        self.assertEqual(self._row(row["id"])["status"], "done")

    # ---- C. done rows never come back ----------------------------------

    async def test_done_reminder_is_not_redelivered_after_restart(self):
        row = self._seed(due_delta=-3600, status="done")
        agent, handler = self._restart()
        self.assertEqual(agent.list_memory_overview()["intentions"], [])

        machine = self._machine()
        state = machine.new_state(self.history_uid)
        calls = await self._run(handler, machine, state, seconds=1.2)

        self.assertEqual(calls, [])
        self.assertEqual(self._row(row["id"])["status"], "done")
        self.assertEqual(len(load_character_state(self.conf_uid).future_intentions), 1)

    # ---- D. stale claim from a dead process ----------------------------

    async def test_stale_claim_is_recovered_after_lease_expiry(self):
        row = self._seed(
            due_delta=-60, claimed_by="dead-process:abc", claim_delta=-1
        )
        agent, handler = self._restart()
        delivered = asyncio.Event()
        machine = self._machine()
        state = machine.new_state(self.history_uid)

        calls = await self._run(handler, machine, state, until=delivered)

        self.assertEqual(len(calls), 1)
        settled = self._row(row["id"])
        self.assertEqual(settled["status"], "done")
        self.assertNotIn("claimed_by", settled)
        self.assertNotIn("claim_until", settled)
        # The dead owner must not be able to complete after takeover.
        self.assertEqual(
            complete_claimed_future_intentions(
                self.conf_uid, owner="dead-process:abc", intention_ids=[row["id"]]
            ),
            0,
        )

    async def test_live_foreign_claim_blocks_a_reconnected_delivery(self):
        row = self._seed(
            due_delta=-60, claimed_by="other-live:xyz", claim_delta=120
        )
        agent, handler = self._restart()
        machine = self._machine()
        state = machine.new_state(self.history_uid)

        calls = await self._run(handler, machine, state, seconds=1.0)

        self.assertEqual(calls, [])
        still = self._row(row["id"])
        self.assertEqual(still["status"], "pending")
        self.assertEqual(still["claimed_by"], "other-live:xyz")
        self.assertIn("claim_until", still)

    # ---- E. two competing connections after restart ---------------------

    async def test_two_competing_connections_deliver_once(self):
        row = self._seed(due_delta=-30)
        agent_a, handler_a = self._restart("client-a")
        agent_b, handler_b = self._restart("client-b")
        done = asyncio.Event()
        machine_a = self._machine(initial_idle=1)
        machine_b = self._machine(initial_idle=1)
        state_a = machine_a.new_state(self.history_uid)
        state_b = machine_b.new_state(self.history_uid)

        calls = []

        async def fake_turn(**kwargs):
            calls.append(kwargs)
            done.set()
            return "minum obat"

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=fake_turn,
        ):
            timers = [
                asyncio.create_task(
                    handler_a._run_proactive_timer(
                        "client-a", self.history_uid, state_a, machine_a
                    )
                ),
                asyncio.create_task(
                    handler_b._run_proactive_timer(
                        "client-b", self.history_uid, state_b, machine_b
                    )
                ),
            ]
            await asyncio.wait_for(done.wait(), 6)
            await asyncio.sleep(0.4)
            for timer in timers:
                timer.cancel()
            for timer in timers:
                with self.assertRaises(asyncio.CancelledError):
                    await timer

        self.assertEqual(len(calls), 1)
        self.assertEqual(self._row(row["id"])["status"], "done")

    # ---- F. reconnect after success must not replay ---------------------

    async def test_reconnect_after_success_does_not_resend(self):
        row = self._seed(due_delta=-30)
        delivered = asyncio.Event()
        machine = self._machine()
        state = machine.new_state(self.history_uid)
        calls = await self._run(self.handler, machine, state, until=delivered)
        self.assertEqual(len(calls), 1)

        # Reconnect: same store, brand-new scheduler. Nothing to resend.
        agent, handler = self._restart()
        self.assertEqual(agent.list_memory_overview()["intentions"], [])
        machine = self._machine()
        state = machine.new_state(self.history_uid)
        replayed = await self._run(handler, machine, state, seconds=1.2)

        self.assertEqual(replayed, [])
        self.assertEqual(self._row(row["id"])["status"], "done")

    # ---- G. restart must not deliver early ------------------------------

    async def test_restart_does_not_deliver_before_due(self):
        row = self._seed(due_delta=3.0)
        agent, handler = self._restart()
        machine = self._machine()
        state = machine.new_state(self.history_uid)

        calls = await self._run(handler, machine, state, seconds=1.0)

        self.assertEqual(calls, [])
        still = self._row(row["id"])
        self.assertEqual(still["status"], "pending")
        self.assertNotIn("claimed_by", still)


if __name__ == "__main__":
    unittest.main()
