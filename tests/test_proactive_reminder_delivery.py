"""Delivery lifecycle for due proactive reminders.

Regression coverage for the bug where a due reminder was consumed BEFORE the
proactive prompt was built: the model never saw the text, a failed turn lost
the reminder permanently, and two connections could both deliver it. These
tests pin the contract: pending until delivered, id-based ownership, retry on
failure, no duplicate send, and request-only prompt steering.
"""

import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.character_state import (
    claim_due_future_intentions,
    complete_claimed_future_intentions,
    load_character_state,
    release_claimed_future_intentions,
    save_character_state,
)
from src.open_llm_vtuber.chat_history_manager import create_new_history, store_message
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.future_intentions import (
    STATUS_DONE,
    STATUS_PENDING,
    build_future_intention_context,
    due_intentions,
    normalize_stored_intention,
)


class _FakeLLM:
    model = "reminder-delivery-test"
    max_tokens = 100

    def __init__(self):
        self.calls = []

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "system": system,
            }
        )
        yield "Iya, aku inget kok."


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


def _make_agent(conf_uid, history_uid, llm_cls=_FakeLLM):
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
        context_window_override=2200,
    )
    agent.set_memory_from_history(conf_uid, history_uid)
    return agent, llm


def _iso(moment):
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


class ReminderDeliveryTests(unittest.TestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.conf_uid = "mili-reminder-delivery"
        self.history_uid = create_new_history(self.conf_uid)
        store_message(self.conf_uid, self.history_uid, "human", "Halo Mili.")
        store_message(self.conf_uid, self.history_uid, "ai", "Halo sayang.")
        self.now = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    def _seed(self, *, text="ingetin aku minum obat ya", due_delta=-3600, **extra):
        state = load_character_state(self.conf_uid)
        row = {
            "id": uuid.uuid4().hex,
            "text": text,
            "due_at": _iso(self.now + timedelta(seconds=due_delta)),
            "created_at": _iso(self.now - timedelta(days=1)),
            "status": STATUS_PENDING,
            "source": "conversation",
            "kind": "reminder",
            "tz": "Asia/Jakarta",
        }
        row.update(extra)
        state.future_intentions = [*(state.future_intentions or []), row]
        self.assertTrue(save_character_state(self.conf_uid, state))
        return row

    def _row(self, intention_id):
        return next(
            row
            for row in load_character_state(self.conf_uid).future_intentions
            if row.get("id") == intention_id
        )

    # A. The due text is in the LLM prompt actually used for the turn.
    def test_due_reminder_text_is_in_the_proactive_prompt(self):
        row = self._seed()
        agent, llm = _make_agent(self.conf_uid, self.history_uid)
        claimed = agent.claim_due_future_intentions("owner-a", self.now)
        self.assertEqual([item["id"] for item in claimed], [row["id"]])

        import asyncio

        outputs = asyncio.run(_collect(agent, claimed))

        self.assertTrue(outputs)
        system = llm.calls[0]["system"]
        self.assertIn(row["text"], system)
        self.assertIn("scheduled reminder is due RIGHT NOW", system)

    # B. Claiming does not consume; completion happens only after success.
    def test_claim_keeps_pending_and_completion_is_explicit(self):
        row = self._seed()
        claimed = claim_due_future_intentions(
            self.conf_uid, owner="owner-a", now=self.now
        )
        self.assertEqual(len(claimed), 1)
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertEqual(stored["claimed_by"], "owner-a")
        self.assertIn(
            row["text"],
            build_future_intention_context(
                load_character_state(self.conf_uid).future_intentions
            ),
        )

        completed = complete_claimed_future_intentions(
            self.conf_uid, owner="owner-a", intention_ids=[row["id"]]
        )
        self.assertEqual(completed, 1)
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_DONE)
        self.assertNotIn("claimed_by", stored)
        self.assertEqual(
            due_intentions(
                load_character_state(self.conf_uid).future_intentions,
                now=self.now,
            ),
            [],
        )

    # C. Empty/failed delivery releases the claim so the reminder retries.
    def test_failed_delivery_releases_and_retries(self):
        row = self._seed()
        claim_due_future_intentions(self.conf_uid, owner="owner-a", now=self.now)
        self.assertEqual(
            release_claimed_future_intentions(
                self.conf_uid, owner="owner-a", intention_ids=[row["id"]]
            ),
            1,
        )
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", stored)
        self.assertEqual(
            len(
                due_intentions(
                    load_character_state(self.conf_uid).future_intentions,
                    now=self.now,
                )
            ),
            1,
        )

        retry = claim_due_future_intentions(
            self.conf_uid, owner="owner-b", now=self.now
        )
        self.assertEqual([item["id"] for item in retry], [row["id"]])
        self.assertEqual(
            complete_claimed_future_intentions(
                self.conf_uid, owner="owner-b", intention_ids=[row["id"]]
            ),
            1,
        )

    # D. Two live attempts: only one owner can claim/deliver.
    def test_concurrent_attempts_cannot_double_deliver(self):
        row = self._seed()
        first = claim_due_future_intentions(
            self.conf_uid, owner="owner-a", now=self.now
        )
        second = claim_due_future_intentions(
            self.conf_uid, owner="owner-b", now=self.now
        )
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])

        self.assertEqual(
            complete_claimed_future_intentions(
                self.conf_uid, owner="owner-a", intention_ids=[row["id"]]
            ),
            1,
        )
        # A stale second attempt can no longer complete anything.
        self.assertEqual(
            complete_claimed_future_intentions(
                self.conf_uid, owner="owner-b", intention_ids=[row["id"]]
            ),
            0,
        )
        self.assertEqual(
            claim_due_future_intentions(self.conf_uid, owner="owner-b", now=self.now),
            [],
        )

    # E. A crashed sender's lease expires; reconnect/retry recovers once.
    def test_expired_lease_allows_single_recovery(self):
        row = self._seed()
        self.assertEqual(
            len(
                claim_due_future_intentions(
                    self.conf_uid, owner="owner-a", now=self.now, lease_seconds=180
                )
            ),
            1,
        )
        self.assertEqual(
            claim_due_future_intentions(
                self.conf_uid,
                owner="owner-b",
                now=self.now + timedelta(seconds=179),
                lease_seconds=180,
            ),
            [],
        )
        recovered = claim_due_future_intentions(
            self.conf_uid,
            owner="owner-b",
            now=self.now + timedelta(seconds=181),
            lease_seconds=180,
        )
        self.assertEqual([item["id"] for item in recovered], [row["id"]])
        # The original owner must not be able to complete after losing lease.
        self.assertEqual(
            complete_claimed_future_intentions(
                self.conf_uid, owner="owner-a", intention_ids=[row["id"]]
            ),
            0,
        )
        self.assertEqual(
            complete_claimed_future_intentions(
                self.conf_uid, owner="owner-b", intention_ids=[row["id"]]
            ),
            1,
        )

    # F. Future-only rows are never claimed; mixed stores claim only due rows.
    def test_future_intention_is_not_claimed(self):
        future = self._seed(text="besok aku ada ujian", due_delta=3600)
        agent, _llm = _make_agent(self.conf_uid, self.history_uid)
        self.assertEqual(
            claim_due_future_intentions(self.conf_uid, owner="owner-a", now=self.now),
            [],
        )
        self.assertFalse(agent.has_due_future_intention(self.now))

        due = self._seed(text="ingetin aku minum obat ya", due_delta=-60)
        claimed = claim_due_future_intentions(
            self.conf_uid, owner="owner-a", now=self.now
        )
        self.assertEqual([item["id"] for item in claimed], [due["id"]])
        self.assertEqual(self._row(future["id"])["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", self._row(future["id"]))

    # G. Steering is request-only and absent from unrelated proactive turns.
    def test_reminder_instruction_is_request_only(self):
        row = self._seed()
        agent, llm = _make_agent(self.conf_uid, self.history_uid)
        claimed = agent.claim_due_future_intentions("owner-a", self.now)

        import asyncio

        outputs = asyncio.run(_collect(agent, claimed))
        self.assertTrue(outputs)
        system = llm.calls[-1]["system"]
        self.assertIn("Deliver it in this message", system)
        self.assertIn(row["text"], system)
        # The instruction is request-only: never written to memory/history.
        self.assertFalse(
            any(
                "scheduled reminder is due RIGHT NOW" in str(message.get("content", ""))
                for message in agent._memory
            )
        )

        # An unrelated proactive turn gets no reminder steering block.
        asyncio.run(_collect(agent, []))
        self.assertNotIn("scheduled reminder is due RIGHT NOW", llm.calls[-1]["system"])

    # H. Per-user isolation and no consumption by a normal turn.
    def test_other_user_and_normal_turn_do_not_consume(self):
        row = self._seed()
        other_conf = "mili-reminder-other"
        create_new_history(other_conf)
        self.assertEqual(
            claim_due_future_intentions(other_conf, owner="owner-b", now=self.now),
            [],
        )

        agent, _llm = _make_agent(self.conf_uid, self.history_uid)
        import asyncio

        asyncio.run(_normal_turn(agent))
        stored = self._row(row["id"])
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", stored)


async def _collect(agent, claimed):
    return [
        output
        async for output in agent.chat_proactively(
            due_reminders=[item["text"] for item in claimed] or None
        )
    ]


async def _normal_turn(agent):
    from src.open_llm_vtuber.conversations.conversation_utils import (
        create_batch_input,
    )

    batch = create_batch_input("Aku baru pulang.", None, "User")
    return [output async for output in agent.chat(batch)]


class NormalizeClaimFieldsTests(unittest.TestCase):
    def test_claim_fields_round_trip_and_partial_claims_are_dropped(self):
        base = {
            "id": "abc",
            "text": "ingetin aku minum obat",
            "status": "pending",
            "claimed_by": "owner-a",
            "claim_until": "2026-10-20T09:03:00+00:00",
        }
        parsed = normalize_stored_intention(base)
        self.assertEqual(parsed["claimed_by"], "owner-a")
        self.assertEqual(parsed["claim_until"], "2026-10-20T09:03:00+00:00")

        partial = {**base}
        partial.pop("claim_until")
        parsed = normalize_stored_intention(partial)
        self.assertNotIn("claimed_by", parsed)
        self.assertNotIn("claim_until", parsed)


if __name__ == "__main__":
    unittest.main()


class ReminderDispatchOrderingTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end ordering proof through the real proactive timer.

    The stubbed turn inspects the authoritative file *while the prompt would
    be built*: it must still see the reminder pending (so the text renders)
    and must receive the exact text in metadata. Only after it returns a
    truthy response may the row become done.
    """

    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.conf_uid = "mili-reminder-order"
        self.history_uid = create_new_history(self.conf_uid)
        store_message(self.conf_uid, self.history_uid, "human", "Halo Mili.")
        store_message(self.conf_uid, self.history_uid, "ai", "Halo sayang.")
        # The timer trigger uses wall-clock time (no injected now), so seed
        # the row relative to the real UTC clock for this end-to-end path.
        self.now = datetime.now(timezone.utc)
        self.row = self._seed()

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    def _seed(self):
        state = load_character_state(self.conf_uid)
        row = {
            "id": uuid.uuid4().hex,
            "text": "ingetin aku minum obat ya",
            "due_at": _iso(self.now - timedelta(hours=1)),
            "created_at": _iso(self.now - timedelta(days=1)),
            "status": STATUS_PENDING,
            "source": "conversation",
            "kind": "reminder",
            "tz": "Asia/Jakarta",
        }
        state.future_intentions = [*(state.future_intentions or []), row]
        self.assertTrue(save_character_state(self.conf_uid, state))
        return row

    def _stored(self):
        return next(
            row
            for row in load_character_state(self.conf_uid).future_intentions
            if row.get("id") == self.row["id"]
        )

    def _handler(self, agent):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from src.open_llm_vtuber.proactive_chat import ProactiveIntentSignals
        from src.open_llm_vtuber.proactive_gate import (
            ProactiveBudgetState,
            ProactiveGateConfig,
        )
        from src.open_llm_vtuber.websocket_handler import WebSocketHandler

        handler = WebSocketHandler.__new__(WebSocketHandler)
        handler.client_contexts = {
            "client": SimpleNamespace(
                history_uid=self.history_uid,
                agent_engine=agent,
                user_timezone="Asia/Jakarta",
            )
        }
        handler.client_connections = {"client": SimpleNamespace(send_text=AsyncMock())}
        handler._proactive_maintenance = set()
        handler._proactive_timer_tasks = {}
        handler.current_conversation_tasks = {}
        handler.chat_group_manager = SimpleNamespace(get_client_group=lambda _uid: None)
        handler._proactive_gate_config = ProactiveGateConfig(
            quiet_hours_start_hour=0, quiet_hours_end_hour=0
        )
        handler._proactive_budget_state = {}
        handler._proactive_intent_signals = staticmethod(
            lambda _context, _state: ProactiveIntentSignals(
                has_useful_memory=True, memory_relevance_score=0.9
            )
        )
        handler._meaningful_life_event = staticmethod(lambda _context, _since: False)
        handler._budget_state = lambda _uid, _conf: ProactiveBudgetState()
        handler._persist_budget = lambda _uid, _conf: True
        return handler

    async def _run(self, response):
        import asyncio

        from src.open_llm_vtuber.proactive_chat import (
            ProactiveChatConfig,
            ProactiveStateMachine,
        )

        agent, _llm = _make_agent(self.conf_uid, self.history_uid)
        handler = self._handler(agent)
        machine = ProactiveStateMachine(
            ProactiveChatConfig(
                # A one-second initial idle keeps the empty-response retry
                # outside this test's observation window (real monotonic
                # clock), so the assertion checks the first attempt only.
                initial_idle_min_seconds=1,
                initial_idle_max_seconds=1,
                followup_idle_min_seconds=100,
                followup_idle_max_seconds=100,
            ),
            randint=lambda minimum, _maximum: minimum,
        )
        state = machine.new_state(self.history_uid)
        captured = {}
        called = asyncio.Event()
        calls = 0

        async def fake_turn(**kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                # The empty-response path is immediately eligible again with
                # a zero idle; block the retry so the assertion is stable.
                await asyncio.Event().wait()
            captured.update(kwargs["metadata"])
            captured["status_at_prompt"] = self._stored()["status"]
            called.set()
            return response

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=fake_turn,
        ):
            timer = asyncio.create_task(
                handler._run_proactive_timer("client", self.history_uid, state, machine)
            )
            handler._proactive_timer_tasks["client"] = timer
            await asyncio.wait_for(called.wait(), 3)
            await asyncio.sleep(0.05)
            timer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await timer
        return captured

    async def test_prompt_sees_pending_text_and_success_completes(self):
        captured = await self._run("Iya, aku ingetin minum obat ya.")
        self.assertEqual(captured["status_at_prompt"], STATUS_PENDING)
        self.assertEqual(captured["proactive_due_reminders"], [self.row["text"]])
        stored = self._stored()
        self.assertEqual(stored["status"], STATUS_DONE)
        self.assertNotIn("claimed_by", stored)

    async def test_empty_response_keeps_pending_for_retry(self):
        captured = await self._run("")
        self.assertEqual(captured["status_at_prompt"], STATUS_PENDING)
        self.assertEqual(captured["proactive_due_reminders"], [self.row["text"]])
        stored = self._stored()
        self.assertEqual(stored["status"], STATUS_PENDING)
        self.assertNotIn("claimed_by", stored)
