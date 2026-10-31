"""Reminder scheduler — due lane, wake timing, delivery, retry pacing.

Regression target (mission issues 2 and 3):

* A pending reminder is not delivered before ``due_at``.
* A due reminder is dispatched without a new user message.
* The spontaneous-conversation gate (quiet hours, minimum gap, dormancy,
  backoff, daily ceiling) must not swallow a due reminder, while spontaneous
  turns keep obeying it unchanged.
* Success completes the row exactly once; an empty/failed turn keeps it
  pending and paces the retry instead of hot-looping the provider.
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

from src.open_llm_vtuber.character_state import (
    capture_reminder,
    load_character_state,
    save_character_state,
)
from src.open_llm_vtuber.chat_history_manager import create_new_history, store_message
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.future_intentions import next_due_at, pending_intentions
from src.open_llm_vtuber.proactive_chat import (
    ProactiveChatConfig,
    ProactiveIntentSignals,
    ProactiveStateMachine,
)
from src.open_llm_vtuber.proactive_gate import (
    GATE_REMINDER,
    ProactiveBudgetState,
    ProactiveGateConfig,
    TriggerReason,
    evaluate_gate,
    record_reminder_dispatch,
)
from src.open_llm_vtuber.websocket_handler import WebSocketHandler

JKT = "Asia/Jakarta"
CONF = "schedchar"
NOW = datetime(2026, 10, 21, 4, 40, tzinfo=timezone.utc)


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class _LLM:
    model = "sched"
    max_tokens = 8

    async def chat_completion(self, *args, **kwargs):
        raise AssertionError("no provider call expected in this test")
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


class GateLaneTest(unittest.TestCase):
    def _budget(self, now=NOW):
        return ProactiveBudgetState(
            last_proactive_at=now.isoformat(),
            backoff_until=(now + timedelta(hours=2)).isoformat(),
            dormant_until=(now + timedelta(hours=3)).isoformat(),
            consecutive_unanswered=5,
            daily_request_count=60,
            daily_count_date=now.strftime("%Y-%m-%d"),
            hourly_meaningful_count=10,
        )

    def test_due_reminder_bypasses_spontaneous_pacing(self):
        trigger = TriggerReason("high", "explicit_reminder", "user-defined reminder")
        config = ProactiveGateConfig()  # quiet 23-07, gap 300s, daily 60
        decision = evaluate_gate(
            self._budget(),
            config,
            trigger,
            now=NOW,
            tz=JKT,
            reminder_due=True,
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, GATE_REMINDER)

    def test_spontaneous_turn_still_obeys_the_gate(self):
        trigger = TriggerReason("high", "unfinished_topic", "unresolved user topic")
        decision = evaluate_gate(self._budget(), ProactiveGateConfig(), trigger, now=NOW, tz=JKT)
        self.assertFalse(decision.allowed)
        self.assertIn(decision.reason, {"backoff_active", "minimum_gap", "daily_hard_limit"})

    def test_reminder_lane_still_requires_a_live_connection(self):
        trigger = TriggerReason("high", "explicit_reminder", "user-defined reminder")
        decision = evaluate_gate(
            ProactiveBudgetState(),
            ProactiveGateConfig(),
            trigger,
            now=NOW,
            tz=JKT,
            reminder_due=True,
            connection_valid=False,
        )
        self.assertFalse(decision.allowed)

    def test_quiet_hours_block_spontaneous_but_not_reminder(self):
        quiet_now = datetime(2026, 10, 21, 17, 0, tzinfo=timezone.utc)  # 00:00 JKT
        config = ProactiveGateConfig()
        spontaneous = TriggerReason("medium", "life_event", "meaningful life transition")
        self.assertEqual(
            evaluate_gate(
                ProactiveBudgetState(), config, spontaneous, now=quiet_now, tz=JKT
            ).reason,
            "quiet_hours",
        )
        reminder = TriggerReason("high", "explicit_reminder", "user-defined reminder")
        self.assertTrue(
            evaluate_gate(
                ProactiveBudgetState(),
                config,
                reminder,
                now=quiet_now,
                tz=JKT,
                reminder_due=True,
            ).allowed
        )

    def test_reminder_dispatch_does_not_raise_ignored_pressure(self):
        state = ProactiveBudgetState(consecutive_unanswered=0)
        record_reminder_dispatch(state, ProactiveGateConfig(), now=NOW, tz=JKT)
        self.assertEqual(state.consecutive_unanswered, 0)
        self.assertIsNone(state.backoff_until)
        self.assertEqual(state.daily_request_count, 1)


class SchedulerRuntimeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self.conf_uid = CONF
        self.history_uid = create_new_history(self.conf_uid)
        store_message(self.conf_uid, self.history_uid, "human", "Halo.")
        store_message(self.conf_uid, self.history_uid, "ai", "Halo juga.")
        self.agent = _make_agent(self.conf_uid, self.history_uid)
        self.handler = self._handler(self.agent)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _seed(self, due_delta=-60, delivery="TES PENGINGAT BERHASIL"):
        base = datetime.now(timezone.utc)
        state = load_character_state(self.conf_uid)
        row = {
            "id": uuid.uuid4().hex,
            "text": "Nanti jam 11.54 bilang kaya tadi lagi ya.",
            "due_at": (base + timedelta(seconds=due_delta)).isoformat(),
            "created_at": base.isoformat(),
            "status": "pending",
            "source": "conversation",
            "kind": "reminder",
            "tz": JKT,
            "delivery_text": delivery,
        }
        state.future_intentions = [row]
        self.assertTrue(save_character_state(self.conf_uid, state))
        self.agent._character_state = load_character_state(self.conf_uid)
        return row

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
        # Spontaneous claims are fully blocked: quiet hours all day, gap an
        # hour, dormant for hours. Only the reminder lane may dispatch.
        handler._proactive_gate_config = ProactiveGateConfig(
            quiet_hours_start_hour=0,
            quiet_hours_end_hour=23,
            minimum_proactive_gap_seconds=3600,
        )
        handler._budget_state = lambda _uid, _conf: ProactiveBudgetState(
            dormant_until=(NOW + timedelta(hours=5)).isoformat(),
            backoff_until=(NOW + timedelta(hours=5)).isoformat(),
            consecutive_unanswered=3,
        )
        handler._persist_budget = lambda _uid, _conf: True
        handler._proactive_intent_signals = staticmethod(
            lambda _context, _state: ProactiveIntentSignals()
        )
        handler._meaningful_life_event = staticmethod(lambda _context, _since: False)
        return handler

    def _machine(self):
        return ProactiveStateMachine(
            ProactiveChatConfig(
                initial_idle_min_seconds=1,
                initial_idle_max_seconds=1,
                followup_idle_min_seconds=100,
                followup_idle_max_seconds=100,
            ),
            randint=lambda minimum, _maximum: minimum,
        )

    async def test_due_reminder_dispatches_without_user_input(self):
        self._seed(due_delta=-60)
        machine = self._machine()
        state = machine.new_state(self.history_uid)
        calls = []
        done = asyncio.Event()

        async def fake_turn(**kwargs):
            calls.append(kwargs["metadata"])
            # Pending while it is being delivered, so the text is in prompt.
            row = load_character_state(self.conf_uid).future_intentions[0]
            calls[-1]["status_at_prompt"] = row["status"]
            done.set()
            return "TES PENGINGAT BERHASIL"

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=fake_turn,
        ):
            timer = asyncio.create_task(
                self.handler._run_proactive_timer("client", self.history_uid, state, machine)
            )
            await asyncio.wait_for(done.wait(), 5)
            await asyncio.sleep(0.05)
            timer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await timer

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["status_at_prompt"], "pending")
        self.assertEqual(calls[0]["proactive_due_reminders"], ["TES PENGINGAT BERHASIL"])
        row = load_character_state(self.conf_uid).future_intentions[0]
        self.assertEqual(row["status"], "done")

    async def test_not_due_reminder_is_not_dispatched(self):
        self._seed(due_delta=+600)
        machine = self._machine()
        state = machine.new_state(self.history_uid)
        calls = []

        async def fake_turn(**kwargs):
            calls.append(kwargs)
            return "harusnya tidak jalan"

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=fake_turn,
        ):
            timer = asyncio.create_task(
                self.handler._run_proactive_timer("client", self.history_uid, state, machine)
            )
            await asyncio.sleep(1.5)
            timer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await timer

        self.assertEqual(calls, [])
        row = load_character_state(self.conf_uid).future_intentions[0]
        self.assertEqual(row["status"], "pending")
        self.assertNotIn("claimed_by", row)

    async def test_failed_turn_keeps_pending_and_paces_retry(self):
        self._seed(due_delta=-60)
        machine = self._machine()
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

        # Not completed, not lost, and not retried in a hot loop.
        row = load_character_state(self.conf_uid).future_intentions[0]
        self.assertEqual(row["status"], "pending")
        self.assertNotIn("claimed_by", row)
        self.assertEqual(len(calls), 1)
        row_id = row["id"]
        self.assertIn(
            row_id, self.handler._reminder_row_retry_not_before["client"]
        )

    async def test_wake_time_targets_due_at(self):
        state = self._machine().new_state(self.history_uid)
        due = datetime.now(timezone.utc) + timedelta(seconds=5)

        class _Agent:
            def next_future_intention_due(self, _moment=None):
                return due

        class _Machine:
            def seconds_until_eligible(self, _state):
                return 900.0

        context = SimpleNamespace(agent_engine=_Agent())
        wait = self.handler._proactive_sleep_seconds("client", state, _Machine(), context)
        self.assertLessEqual(wait, 6.0)
        self.assertGreater(wait, 0.0)

    async def test_retry_window_extends_sleep(self):
        state = self._machine().new_state(self.history_uid)
        self.handler._reminder_row_retry_not_before["client"] = {
            "row-1": datetime.now(timezone.utc) + timedelta(seconds=30)
        }

        class _Agent:
            def next_future_intention_due(self, _moment=None, exclude_ids=None):
                # The only due row is excluded by its per-row retry window.
                return None

        class _Machine:
            def seconds_until_eligible(self, _state):
                return 900.0

        wait = self.handler._proactive_sleep_seconds(
            "client", state, _Machine(), SimpleNamespace(agent_engine=_Agent())
        )
        self.assertLessEqual(wait, 31.0)
        self.assertGreater(wait, 0.0)

    async def test_pre_turn_receipt_renders_into_system_prompt(self):
        self.agent._user_timezone = JKT
        # Unresolvable reference: clarify, nothing stored, prompt says so.
        self.agent._refresh_reminder_receipt(
            "Nanti jam 11.54 bilang kaya tadi lagi ya."
        )
        self.assertIn("NOTHING was stored", self.agent._reminder_receipt_block)
        prompt = self.agent._relationship_system_prompt("persona")
        self.assertIn("NOTHING was stored", prompt)
        self.assertEqual(load_character_state(self.conf_uid).future_intentions, [])

        # Explicit request: persisted pre-turn and confirmed in the prompt.
        self.agent._memory.append(
            {
                "role": "assistant",
                "content": "Nanti aku bilang 'TES PENGINGAT BERHASIL' ya.",
            }
        )
        self.agent._refresh_reminder_receipt(
            "Nanti jam 11.51 ingetin aku dengan bilang 'TES PENGINGAT BERHASIL'."
        )
        self.assertIn("IS stored", self.agent._reminder_receipt_block)
        self.assertIn(
            "IS stored", self.agent._relationship_system_prompt("persona")
        )
        self.assertEqual(len(load_character_state(self.conf_uid).future_intentions), 1)

    async def test_overview_lists_pending_reminder_and_hides_done(self):
        from src.open_llm_vtuber.character_state import complete_future_intention

        base = datetime.now(timezone.utc)
        status, row = capture_reminder(
            self.conf_uid,
            "Nanti jam 11.54 bilang kaya tadi lagi ya.",
            now=base,
            tz=JKT,
            recent_messages=[
                {
                    "role": "user",
                    "content": "ingetin jam 11.51 bilang 'TES PENGINGAT BERHASIL'",
                }
            ],
        )
        self.assertEqual(status, "captured")
        self.agent._character_state = load_character_state(self.conf_uid)
        overview = self.agent.list_memory_overview()
        self.assertEqual([item["id"] for item in overview["intentions"]], [row["id"]])
        self.assertEqual(overview["intentions"][0]["delivery_text"], "TES PENGINGAT BERHASIL")
        self.assertTrue(complete_future_intention(self.conf_uid, row["id"]))
        self.agent._character_state = load_character_state(self.conf_uid)
        overview = self.agent.list_memory_overview()
        self.assertEqual(overview["intentions"], [])

    async def test_due_plan_does_not_earn_reminder_lane(self):
        base = datetime.now(timezone.utc)
        state = load_character_state(self.conf_uid)
        state.future_intentions = [
            {
                "id": uuid.uuid4().hex,
                "text": "besok aku ada ujian",
                "due_at": (base - timedelta(seconds=30)).isoformat(),
                "created_at": base.isoformat(),
                "status": "pending",
                "source": "conversation",
                "kind": "plan",
                "tz": JKT,
            }
        ]
        self.assertTrue(save_character_state(self.conf_uid, state))
        self.agent._character_state = load_character_state(self.conf_uid)
        self.assertTrue(self.agent.has_due_future_intention())
        self.assertFalse(self.agent.has_due_reminder())
        self.assertFalse(
            self.handler._due_reminder_lane(self.handler.client_contexts["client"])
        )

    async def test_reconnect_sees_overdue_reminder(self):
        row = self._seed(due_delta=-3600)
        # Fresh agent: a restart/reconnect builds new memory from history.
        fresh = _make_agent(self.conf_uid, self.history_uid)
        self.assertTrue(fresh.has_due_future_intention())
        self.assertEqual(next_due_at(fresh._character_state.future_intentions), datetime.fromisoformat(row["due_at"]))
        self.assertEqual(len(pending_intentions(load_character_state(self.conf_uid).future_intentions)), 1)


if __name__ == "__main__":
    unittest.main()
