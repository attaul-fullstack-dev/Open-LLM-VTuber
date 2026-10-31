"""Channel separation: normal chat context vs scheduler reminder delivery.

Both lanes read the same bounded store, so a due reminder can reach the model
twice: once as ordinary chat context and once as the scheduler's "deliver this
now" instruction. These tests pin the boundary:

* a normal chat turn renders a due reminder as *context only* — no delivery
  instruction and no delivery payload;
* the scheduler turn still asks for the reminder and still carries its text;
* the reminder being delivered is not also rendered as chat context in that
  same turn (no contradictory instructions);
* a reminder with a durable receipt is neither rendered nor claimable again;
* future intentions that are not due reminders stay available as context.
"""

import asyncio
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.character_state import (
    claim_due_future_intentions,
    complete_claimed_future_intentions,
    load_character_state,
    save_character_state,
)
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    store_message,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.future_intentions import (
    STATUS_PENDING,
    build_future_intention_context,
)
from src.open_llm_vtuber.reminder_delivery import record_delivery

# The renderer compares ``due_at`` against the real clock per turn, so the
# fixtures are anchored to the wall clock to keep "due" and "not yet due"
# unambiguous in both directions.
CONF = "mili-channel-separation"
NOW = datetime.now(timezone.utc).replace(microsecond=0)
JKT = "Asia/Jakarta"
DELIVERY_PAYLOAD = "waktunya minum obat ya"
SCHEDULER_CUE = "scheduled reminder is due RIGHT NOW"
CHAT_MARKER = "backend scheduler owns delivery"


def _row(
    *,
    text,
    kind="reminder",
    due_delta=-3600,
    delivery_text=None,
):
    row = {
        "id": uuid.uuid4().hex,
        "text": text,
        "due_at": (NOW + timedelta(seconds=due_delta)).isoformat(),
        "created_at": (NOW - timedelta(days=1)).isoformat(),
        "status": STATUS_PENDING,
        "source": "conversation",
        "kind": kind,
        "tz": JKT,
    }
    if delivery_text:
        row["delivery_text"] = delivery_text
    return row


class ChatContextIsNotDeliveryTest(unittest.TestCase):
    """The pure renderer never asks a normal chat turn to send a reminder."""

    def test_due_reminder_renders_as_context_not_as_a_delivery_instruction(self):
        block = build_future_intention_context(
            [
                _row(
                    text="ingetin aku minum obat",
                    due_delta=-3600,
                    delivery_text=DELIVERY_PAYLOAD,
                )
            ],
            tz=JKT,
        )

        # The topic stays available as context...
        self.assertIn("ingetin aku minum obat", block)
        # ...but the chat prompt must never claim the scheduled send happened.
        self.assertIn(CHAT_MARKER, block)
        self.assertNotIn(DELIVERY_PAYLOAD, block)
        # And it must not carry the scheduler's imperative.
        self.assertNotIn(SCHEDULER_CUE, block)
        self.assertNotIn("Deliver it in this message", block)
        self.assertNotIn("-> stored delivery payload", block)

    def test_not_yet_due_reminder_keeps_its_payload_as_context(self):
        block = build_future_intention_context(
            [
                _row(
                    text="ingetin aku minum obat",
                    due_delta=7200,
                    delivery_text=DELIVERY_PAYLOAD,
                )
            ],
            tz=JKT,
        )

        self.assertIn("stored reminder, not yet due", block)
        self.assertIn("-> stored delivery payload for when it is due", block)
        self.assertIn(DELIVERY_PAYLOAD, block)
        self.assertNotIn(SCHEDULER_CUE, block)

    def test_plan_is_context_only_and_keeps_its_detail(self):
        block = build_future_intention_context(
            [
                _row(
                    text="besok aku ada ujian",
                    kind="plan",
                    due_delta=86400,
                    delivery_text="ujian statistik",
                )
            ],
            tz=JKT,
        )

        self.assertIn("besok aku ada ujian", block)
        self.assertIn("upcoming user plan, for context only", block)
        self.assertIn("-> associated plan detail", block)
        self.assertIn("ujian statistik", block)
        self.assertNotIn(CHAT_MARKER, block)
        self.assertNotIn(SCHEDULER_CUE, block)

    def test_scheduler_turn_does_not_repeat_the_reminder_it_delivers(self):
        due = _row(
            text="ingetin aku minum obat",
            due_delta=-3600,
            delivery_text=DELIVERY_PAYLOAD,
        )
        plan = _row(
            text="besok aku ada ujian",
            kind="plan",
            due_delta=86400,
            delivery_text="ujian statistik",
        )

        # Excluding by either the row text or its delivery payload drops it.
        for excluded in (due["text"], DELIVERY_PAYLOAD):
            block = build_future_intention_context(
                [due, plan], tz=JKT, exclude_texts=[excluded]
            )
            self.assertNotIn(due["text"], block)
            # The other intentions stay available in the same turn.
            self.assertIn(plan["text"], block)


class _FakeLLM:
    model = "channel-separation-test"
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


def _make_agent(conf_uid, history_uid):
    llm = _FakeLLM()
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


async def _normal_turn(agent):
    from src.open_llm_vtuber.conversations.conversation_utils import (
        create_batch_input,
    )

    batch = create_batch_input("Aku baru pulang.", None, "User")
    return [output async for output in agent.chat(batch)]


async def _scheduler_turn(agent, claimed):
    # Mirrors the dispatcher: the payload is what gets delivered when the row
    # carries one, otherwise the stored request text.
    texts = [
        str(item.get("delivery_text") or item.get("text") or "").strip()
        for item in claimed
    ]
    return [
        output
        async for output in agent.chat_proactively(
            due_reminders=[text for text in texts if text] or None
        )
    ]


class SchedulerLaneTest(unittest.TestCase):
    """End-to-end: chat stays context, scheduler still delivers once."""

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

    def _seed(self, rows):
        state = load_character_state(CONF)
        state.future_intentions = [*(state.future_intentions or []), *rows]
        self.assertTrue(save_character_state(CONF, state))

    def _row(self, intention_id):
        return next(
            row
            for row in load_character_state(CONF).future_intentions
            if row.get("id") == intention_id
        )

    def test_normal_chat_prompt_is_context_and_scheduler_still_delivers(self):
        due = _row(text="ingetin aku minum obat", delivery_text=DELIVERY_PAYLOAD)
        plan = _row(
            text="besok aku ada ujian",
            kind="plan",
            due_delta=86400,
            delivery_text="ujian statistik",
        )
        self._seed([due, plan])

        agent, llm = _make_agent(CONF, self.history_uid)
        asyncio.run(_normal_turn(agent))
        chat_prompt = llm.calls[-1]["system"]

        # Normal chat: context only, nothing that claims a scheduled send.
        self.assertIn(due["text"], chat_prompt)
        self.assertIn(CHAT_MARKER, chat_prompt)
        self.assertIn(plan["text"], chat_prompt)
        self.assertNotIn(SCHEDULER_CUE, chat_prompt)
        self.assertNotIn(DELIVERY_PAYLOAD, chat_prompt)
        # The chat turn must not consume the scheduled reminder.
        self.assertEqual(self._row(due["id"])["status"], STATUS_PENDING)

        claimed = agent.claim_due_future_intentions("owner-a", NOW)
        self.assertEqual([item["id"] for item in claimed], [due["id"]])
        outputs = asyncio.run(_scheduler_turn(agent, claimed))
        self.assertTrue(outputs)
        scheduler_prompt = llm.calls[-1]["system"]

        # Scheduler: still delivers on time, exactly once per prompt lane.
        self.assertIn(SCHEDULER_CUE, scheduler_prompt)
        self.assertIn(DELIVERY_PAYLOAD, scheduler_prompt)
        self.assertEqual(scheduler_prompt.count(DELIVERY_PAYLOAD), 1)
        # The delivered row is not repeated as chat context in this turn.
        self.assertNotIn(CHAT_MARKER, scheduler_prompt)
        # A non-reminder intention is still available in the scheduler turn.
        self.assertIn(plan["text"], scheduler_prompt)

    def test_delivered_reminder_is_not_rendered_or_claimed_again(self):
        due = _row(text="ingetin aku minum obat", delivery_text=DELIVERY_PAYLOAD)
        self._seed([due])

        agent, _llm = _make_agent(CONF, self.history_uid)
        claimed = agent.claim_due_future_intentions("owner-a", NOW)
        self.assertEqual([item["id"] for item in claimed], [due["id"]])
        asyncio.run(_scheduler_turn(agent, claimed))

        # Delivery succeeded: durable receipt first, then completion.
        self.assertTrue(record_delivery(CONF, [due["id"]], owner="owner-a", now=NOW))
        self.assertEqual(
            complete_claimed_future_intentions(
                CONF, owner="owner-a", intention_ids=[due["id"]]
            ),
            1,
        )

        # Neither lane renders it again, and no later claim can resend it.
        self.assertEqual(
            build_future_intention_context(
                load_character_state(CONF).future_intentions, tz=JKT
            ),
            "",
        )
        self.assertEqual(
            claim_due_future_intentions(
                CONF, owner="owner-b", now=NOW + timedelta(seconds=700)
            ),
            [],
        )

        asyncio.run(_normal_turn(agent))
        self.assertNotIn(due["text"], _llm.calls[-1]["system"])
        self.assertNotIn(DELIVERY_PAYLOAD, _llm.calls[-1]["system"])

    def test_plan_survives_the_reminder_lifecycle_as_context(self):
        due = _row(text="ingetin aku minum obat", delivery_text=DELIVERY_PAYLOAD)
        plan = _row(
            text="besok aku ada ujian",
            kind="plan",
            due_delta=86400,
            delivery_text="ujian statistik",
        )
        self._seed([due, plan])

        agent, llm = _make_agent(CONF, self.history_uid)
        claimed = agent.claim_due_future_intentions("owner-a", NOW)
        asyncio.run(_scheduler_turn(agent, claimed))
        self.assertTrue(record_delivery(CONF, [due["id"]], owner="owner-a", now=NOW))
        complete_claimed_future_intentions(
            CONF, owner="owner-a", intention_ids=[due["id"]]
        )

        block = build_future_intention_context(
            load_character_state(CONF).future_intentions, tz=JKT
        )
        self.assertNotIn(due["text"], block)
        self.assertIn(plan["text"], block)
        self.assertIn("upcoming user plan, for context only", block)
        self.assertIn("ujian statistik", block)

        asyncio.run(_normal_turn(agent))
        chat_prompt = llm.calls[-1]["system"]
        self.assertIn(plan["text"], chat_prompt)
        self.assertNotIn(due["text"], chat_prompt)


if __name__ == "__main__":
    unittest.main()
