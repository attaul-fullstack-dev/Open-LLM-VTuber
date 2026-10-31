"""Per-reminder completion and retry/restart interaction (isolated fixtures).

Regression target: ``claim_due_intentions`` used to claim *all* due rows while
the scheduler settled all of them after one non-empty response. Two reminders
that happened to be due together could therefore be marked ``done`` even if
the response only addressed one. The dispatcher now claims at most one row per
turn, so every row is completed only by the turn generated for it.

These tests use a fake turn and a temp character store; they prove the state
transitions, not real browser/TTS delivery.
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
CONF = "batch-completion-synthetic"


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class _LLM:
    model = "batch"
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


class ReminderBatchCompletionTest(unittest.IsolatedAsyncioTestCase):
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

    def _seed(self, *, due_delta, text, delivery, claimed_by=None, claim_delta=600):
        base = datetime.now(timezone.utc)
        state = load_character_state(self.conf_uid)
        row = {
            "id": uuid.uuid4().hex,
            "text": text,
            "due_at": (base + timedelta(seconds=due_delta)).isoformat(),
            "created_at": base.isoformat(),
            "status": "pending",
            "source": "conversation",
            "kind": "reminder",
            "tz": JKT,
            "delivery_text": delivery,
        }
        if claimed_by:
            row["claimed_by"] = claimed_by
            row["claim_until"] = (
                base + timedelta(seconds=claim_delta)
            ).isoformat()
        state.future_intentions = [*(state.future_intentions or []), row]
        self.assertTrue(save_character_state(self.conf_uid, state))
        return row

    def _rows(self):
        return {
            item["id"]: item
            for item in load_character_state(self.conf_uid).future_intentions
        }

    def _snapshot(self):
        return {
            row_id: (
                row.get("status"),
                bool(row.get("claimed_by")),
            )
            for row_id, row in self._rows().items()
        }

    async def _run(self, responses, *, until_count, retry_delay=None):
        """Run the scheduler with a scripted fake turn, then cancel it."""
        calls = []

        async def fake_turn(**kwargs):
            due = list(kwargs["metadata"]["proactive_due_reminders"])
            calls.append({"due": due, "rows": self._snapshot()})
            if len(calls) >= until_count:
                done.set()
            if len(calls) > len(responses):
                return "Aku ingat."
            return responses[len(calls) - 1]

        if retry_delay is not None:
            self.handler.REMINDER_RETRY_DELAY_SECONDS = retry_delay
        done = asyncio.Event()
        machine = self._machine()
        state = machine.new_state(self.history_uid)
        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=fake_turn,
        ):
            timer = asyncio.create_task(
                self.handler._run_proactive_timer(
                    "client", self.history_uid, state, machine
                )
            )
            try:
                await asyncio.wait_for(done.wait(), 8)
                await asyncio.sleep(0.5)  # let the last turn settle its claim
            finally:
                timer.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await timer
        return calls

    # ---- A. one row per turn, one completion per row -------------------

    async def test_each_reminder_completes_only_after_its_own_turn(self):
        first = self._seed(
            due_delta=-60, text="ingetin minum obat", delivery="minum obat"
        )
        second = self._seed(
            due_delta=-30, text="ingetin telepon ibu", delivery="telepon ibu"
        )

        calls = await self._run(
            ["Aku ingat: minum obat.", "Aku ingat: telepon ibu."],
            until_count=2,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual([len(call["due"]) for call in calls], [1, 1])
        self.assertEqual(
            sorted(call["due"][0] for call in calls),
            ["minum obat", "telepon ibu"],
        )

        # First turn: only the first row was claimed; the second was untouched.
        self.assertEqual(calls[0]["rows"][first["id"]], ("pending", True))
        self.assertEqual(calls[0]["rows"][second["id"]], ("pending", False))
        # Second turn: the first row is already done; the second is claimed.
        self.assertEqual(calls[1]["rows"][first["id"]], ("done", False))
        self.assertEqual(calls[1]["rows"][second["id"]], ("pending", True))

        settled = self._rows()
        self.assertEqual(settled[first["id"]]["status"], "done")
        self.assertEqual(settled[second["id"]]["status"], "done")
        self.assertNotIn("claimed_by", settled[first["id"]])
        self.assertNotIn("claimed_by", settled[second["id"]])

    # ---- B. failure retries only the failed row ------------------------

    async def test_provider_failure_completes_nothing_and_retries_without_starving(
        self,
    ):
        first = self._seed(
            due_delta=-60, text="ingetin minum obat", delivery="minum obat"
        )
        second = self._seed(
            due_delta=-30, text="ingetin telepon ibu", delivery="telepon ibu"
        )

        calls = await self._run(
            ["", "Aku ingat: telepon ibu.", "Aku ingat: minum obat."],
            until_count=3,
            retry_delay=0.2,
        )

        self.assertEqual(len(calls), 3)
        # The failed row is backed off per row, so the other due reminder is
        # attempted next instead of waiting behind an endless retry of row one.
        self.assertEqual(
            [call["due"] for call in calls],
            [["minum obat"], ["telepon ibu"], ["minum obat"]],
        )
        # After the failure both rows are still pending; nothing was completed.
        self.assertEqual(calls[1]["rows"][first["id"]], ("pending", False))
        self.assertEqual(calls[1]["rows"][second["id"]], ("pending", True))
        # The second row completes only after its own turn; the first remains
        # pending until its paced retry succeeds.
        self.assertEqual(calls[2]["rows"][first["id"]], ("pending", True))
        self.assertEqual(calls[2]["rows"][second["id"]], ("done", False))

        settled = self._rows()
        self.assertEqual(settled[first["id"]]["status"], "done")
        self.assertEqual(settled[second["id"]]["status"], "done")
        self.assertEqual(self.handler._reminder_row_retry_not_before, {})

    async def test_persistently_failing_row_does_not_starve_another_row(self):
        first = self._seed(
            due_delta=-60, text="ingetin minum obat", delivery="minum obat"
        )
        second = self._seed(
            due_delta=-30, text="ingetin telepon ibu", delivery="telepon ibu"
        )

        calls = await self._run(
            [
                "",
                "Aku ingat: telepon ibu.",
                "",
                "Aku ingat: minum obat.",
            ],
            until_count=4,
            retry_delay=0.2,
        )

        self.assertEqual(
            [call["due"] for call in calls],
            [["minum obat"], ["telepon ibu"], ["minum obat"], ["minum obat"]],
        )
        # B succeeds while A is still pending/retryable; no batch completion.
        self.assertEqual(calls[1]["rows"][first["id"]], ("pending", False))
        self.assertEqual(calls[1]["rows"][second["id"]], ("pending", True))
        self.assertEqual(calls[2]["rows"][first["id"]], ("pending", True))
        self.assertEqual(calls[2]["rows"][second["id"]], ("done", False))

        settled = self._rows()
        self.assertEqual(settled[first["id"]]["status"], "done")
        self.assertEqual(settled[second["id"]]["status"], "done")
        self.assertEqual(self.handler._reminder_row_retry_not_before, {})

    async def test_backoff_row_does_not_delay_another_rows_future_due(self):
        first = self._seed(
            due_delta=-60, text="ingetin minum obat", delivery="minum obat"
        )
        second = self._seed(
            due_delta=1.0, text="ingetin telepon ibu", delivery="telepon ibu"
        )
        calls = []
        done = asyncio.Event()

        async def fake_turn(**kwargs):
            calls.append(list(kwargs["metadata"]["proactive_due_reminders"]))
            if len(calls) == 2:
                done.set()
            if len(calls) == 1:
                return ""
            return "Aku ingat: telepon ibu."

        self.handler.REMINDER_RETRY_DELAY_SECONDS = 60.0
        # Wake for the failed row's short initial idle tick before the second
        # row is due; the claim-unavailable path must then wait for row two's
        # due time, not the failed row's 60s backoff.
        machine = self._machine(initial_idle=0.1)
        state = machine.new_state(self.history_uid)
        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=fake_turn,
        ):
            timer = asyncio.create_task(
                self.handler._run_proactive_timer(
                    "client", self.history_uid, state, machine
                )
            )
            try:
                await asyncio.wait_for(done.wait(), 3)
                await asyncio.sleep(0.2)
            finally:
                timer.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await timer

        self.assertEqual(calls, [["minum obat"], ["telepon ibu"]])
        settled = self._rows()
        self.assertEqual(settled[first["id"]]["status"], "pending")
        self.assertEqual(settled[second["id"]]["status"], "done")

    # ---- C. cancellation releases the claim for a later retry ---------

    async def test_cancelled_provider_call_releases_claim_and_allows_retry(self):
        row = self._seed(
            due_delta=-60, text="ingetin minum obat", delivery="minum obat"
        )
        started = asyncio.Event()

        async def blocking_turn(**_kwargs):
            started.set()
            await asyncio.sleep(3600)
            return "tidak boleh selesai"

        self.handler.REMINDER_RETRY_DELAY_SECONDS = 0.05
        machine = self._machine()
        state = machine.new_state(self.history_uid)
        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=blocking_turn,
        ):
            timer = asyncio.create_task(
                self.handler._run_proactive_timer(
                    "client", self.history_uid, state, machine
                )
            )
            await asyncio.wait_for(started.wait(), 3)
            self.assertTrue(self._rows()[row["id"]].get("claimed_by"))
            timer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await timer

        after = self._rows()[row["id"]]
        self.assertEqual(after["status"], "pending")
        self.assertNotIn("claimed_by", after)
        self.assertNotIn("claim_until", after)
        self.assertIn(
            row["id"], self.handler._reminder_row_retry_not_before["client"]
        )

        calls = []
        done = asyncio.Event()

        async def success_turn(**kwargs):
            calls.append(kwargs["metadata"]["proactive_due_reminders"])
            done.set()
            return "Aku ingat: minum obat."

        retry_machine = self._machine()
        retry_state = retry_machine.new_state(self.history_uid)
        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=success_turn,
        ):
            retry_timer = asyncio.create_task(
                self.handler._run_proactive_timer(
                    "client", self.history_uid, retry_state, retry_machine
                )
            )
            await asyncio.wait_for(done.wait(), 3)
            await asyncio.sleep(0.1)
            retry_timer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await retry_timer

        self.assertEqual(calls, [["minum obat"]])
        settled = self._rows()[row["id"]]
        self.assertEqual(settled["status"], "done")
        self.assertNotIn("claimed_by", settled)

    # ---- D. startup recovery clears a dead claim before the lease ------

    async def test_dead_process_claim_is_recovered_on_activation(self):
        dead_owner = (
            "linux:00000000000000000000000000000000:4026531836:999999:1|old-client:abc"
        )
        first = self._seed(
            due_delta=-60,
            text="ingetin minum obat",
            delivery="minum obat",
            claimed_by=dead_owner,
            claim_delta=600,  # lease still live: only identity can recover it
        )
        second = self._seed(
            due_delta=-30, text="ingetin telepon ibu", delivery="telepon ibu"
        )

        self.handler._recover_stale_reminder_claims(
            self.handler.client_contexts["client"]
        )

        recovered = self._rows()[first["id"]]
        self.assertEqual(recovered["status"], "pending")
        self.assertNotIn("claimed_by", recovered)

        calls = await self._run(
            ["Aku ingat: minum obat.", "Aku ingat: telepon ibu."],
            until_count=2,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            sorted(call["due"][0] for call in calls),
            ["minum obat", "telepon ibu"],
        )
        self.assertEqual(self._rows()[first["id"]]["status"], "done")
        self.assertEqual(self._rows()[second["id"]]["status"], "done")
        # The dead owner must not be able to settle the row after recovery.
        self.assertEqual(
            complete_claimed_future_intentions(
                self.conf_uid,
                owner=dead_owner,
                intention_ids=[first["id"]],
            ),
            0,
        )

    async def test_activation_runs_recovery_once_per_character(self):
        config = ProactiveChatConfig()
        self.handler._proactive_config = lambda _context: config
        self.handler._run_proactive_timer = AsyncMock()

        with patch(
            "src.open_llm_vtuber.websocket_handler.recover_stale_claims",
            return_value=0,
        ) as recover:
            await self.handler._activate_proactive_for_history(
                "client", self.history_uid
            )
            await self.handler._activate_proactive_for_history(
                "client", self.history_uid
            )

        recover.assert_called_once_with(self.conf_uid)

    async def test_failed_recovery_retries_on_next_activation(self):
        config = ProactiveChatConfig()
        self.handler._proactive_config = lambda _context: config
        self.handler._run_proactive_timer = AsyncMock()

        with patch(
            "src.open_llm_vtuber.websocket_handler.recover_stale_claims",
            side_effect=[None, 0],
        ) as recover:
            await self.handler._activate_proactive_for_history(
                "client", self.history_uid
            )
            await self.handler._activate_proactive_for_history(
                "client", self.history_uid
            )
            await self.handler._activate_proactive_for_history(
                "client", self.history_uid
            )

        self.assertEqual(recover.call_count, 2)


if __name__ == "__main__":
    unittest.main()
