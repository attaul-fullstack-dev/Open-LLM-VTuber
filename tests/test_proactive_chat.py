import asyncio
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.conversation_summary import SummaryState
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    store_message,
)
from src.open_llm_vtuber.config_manager import (
    BasicMemoryAgentConfig,
    TTSPreprocessorConfig,
)
from src.open_llm_vtuber.conversations.conversation_utils import create_batch_input
from src.open_llm_vtuber.conversations.single_conversation import (
    process_single_conversation,
)
from src.open_llm_vtuber.proactive_chat import (
    DEFAULT_INTENT_WEIGHTS,
    INTENT_SELECTION_ORDER,
    ProactiveChatConfig,
    ProactiveFollowupContext,
    ProactiveIntent,
    ProactiveIntentContext,
    ProactiveIntentSignals,
    ProactiveStateMachine,
    format_followup_instruction,
    format_intent_instruction,
    message_expects_response,
    resolve_proactive_intent,
)
from src.open_llm_vtuber.request_latency import RequestLatencyTracker
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


class _Clock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class _FakeLLM:
    model = "proactive-test"
    max_tokens = 100

    def __init__(self):
        self.calls = []
        self.responses = ["Kok jadi sepi, ya.", "Nah, akhirnya ngomong juga."]

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "system": system,
            }
        )
        yield self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


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
        context_window_override=2000,
    )
    agent.set_memory_from_history(conf_uid, history_uid)
    return agent, llm


class ProactiveTimingTests(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.random_values = iter([60, 120, 130, 200, 250, 55])
        self.machine = ProactiveStateMachine(
            ProactiveChatConfig(),
            monotonic=self.clock,
            randint=lambda _minimum, _maximum: next(self.random_values),
        )

    def test_no_proactive_before_minimum_and_randomized_eligibility(self):
        state = self.machine.new_state("chat-a")
        self.assertEqual(state.next_proactive_eligible_at, 160)
        self.clock.advance(59)
        self.assertFalse(self.machine.is_eligible(state))
        self.clock.advance(1)
        self.assertTrue(self.machine.is_eligible(state))

    def test_user_activity_resets_timer_and_ignored_counter(self):
        state = self.machine.new_state("chat-a")
        state.consecutive_ignored_proactive = 2
        self.clock.advance(10)
        self.machine.record_user_activity(state)
        self.assertEqual(state.consecutive_ignored_proactive, 0)
        self.assertEqual(state.next_proactive_eligible_at, 230)
        self.assertEqual(state.activity_revision, 1)

    def test_followup_and_three_ignored_backoff_ranges(self):
        state = self.machine.new_state("chat-a")
        self.machine.record_proactive_sent(state)
        first_delay = state.next_proactive_eligible_at - self.clock()
        self.assertGreaterEqual(first_delay, 90)
        self.assertLessEqual(first_delay, 240)

        self.machine.record_proactive_sent(state)
        second_delay = state.next_proactive_eligible_at - self.clock()
        self.assertGreaterEqual(second_delay, 90)
        self.assertLessEqual(second_delay, 240)

        self.machine.record_proactive_sent(state)
        backoff = state.next_proactive_eligible_at - self.clock()
        self.assertGreaterEqual(backoff, 180)
        self.assertLessEqual(backoff, 360)

    def test_new_chat_runtime_state_is_independent(self):
        state_a = self.machine.new_state("chat-a")
        self.machine.record_proactive_sent(state_a)
        state_b = self.machine.new_state("chat-b")
        self.assertEqual(state_b.consecutive_ignored_proactive, 0)
        self.assertNotEqual(state_a.history_uid, state_b.history_uid)

    def test_reconnect_uses_fresh_nonpersistent_idle_state(self):
        old_state = self.machine.new_state("chat-a")
        self.machine.record_proactive_sent(old_state)
        self.machine.record_proactive_sent(old_state)
        self.assertEqual(old_state.consecutive_ignored_proactive, 2)

        reconnected_state = self.machine.new_state("chat-a")
        self.assertEqual(reconnected_state.consecutive_ignored_proactive, 0)
        self.assertIsNone(reconnected_state.last_proactive_monotonic)
        self.assertGreater(reconnected_state.next_proactive_eligible_at, self.clock())

    def test_configuration_defaults_match_runtime_contract(self):
        settings = BasicMemoryAgentConfig(llm_provider="ollama_llm")
        self.assertTrue(settings.proactive_enabled)
        self.assertEqual(
            (
                settings.initial_idle_min_seconds,
                settings.initial_idle_max_seconds,
                settings.followup_idle_min_seconds,
                settings.followup_idle_max_seconds,
                settings.ignored_before_backoff,
                settings.backoff_min_seconds,
                settings.backoff_max_seconds,
            ),
            (45, 90, 90, 240, 3, 180, 360),
        )


class ProactivePipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.conf_uid = "mili-proactive"
        self.history_uid = create_new_history(self.conf_uid)
        store_message(self.conf_uid, self.history_uid, "human", "Aku lagi belajar.")
        store_message(self.conf_uid, self.history_uid, "ai", "Belajar apa?")

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    async def test_proactive_turn_uses_context_without_fake_user(self):
        agent, llm = _make_agent(self.conf_uid, self.history_uid)
        agent._summary_state = SummaryState(
            text="User sebelumnya membahas pelajaran penting.",
            summarized_through=0,
        )
        before_roles = [message["role"] for message in agent._memory]

        outputs = [output async for output in agent.chat_proactively()]

        self.assertTrue(outputs)
        self.assertEqual(before_roles, ["user", "assistant"])
        self.assertEqual(
            [message["role"] for message in agent._memory],
            ["user", "assistant", "assistant"],
        )
        self.assertTrue(
            llm.calls[0]["messages"][0]["content"].startswith(
                "Conversation context from earlier messages"
            )
        )
        self.assertEqual(llm.calls[0]["messages"][-1]["role"], "assistant")
        self.assertIn("persona Mili tetap aktif", llm.calls[0]["system"])
        self.assertIn("Internal instruction for this turn only", llm.calls[0]["system"])

    async def test_proactive_message_persists_and_next_reply_sees_it(self):
        agent, llm = _make_agent(self.conf_uid, self.history_uid)
        sent_payloads = []

        async def send(payload):
            sent_payloads.append(json.loads(payload))

        async def return_display_text(output, **_kwargs):
            return output.display_text.text

        context = SimpleNamespace(
            agent_engine=agent,
            asr_engine=None,
            live2d_model=_FakeLive2D(),
            tts_engine=None,
            translate_engine=None,
            history_uid=self.history_uid,
            character_config=SimpleNamespace(
                conf_uid=self.conf_uid,
                human_name="User",
                character_name="Mili",
                avatar="mili.png",
            ),
        )
        with (
            patch(
                "src.open_llm_vtuber.conversations.single_conversation.process_agent_output",
                new=return_display_text,
            ),
            patch(
                "src.open_llm_vtuber.conversations.single_conversation.finalize_conversation_turn",
                new=AsyncMock(),
            ),
        ):
            response = await process_single_conversation(
                context=context,
                websocket_send=send,
                client_uid="client-a",
                user_input="",
                metadata={"request_origin": "proactive"},
            )

        self.assertEqual(response, "Kok jadi sepi,ya.")
        persisted = get_history(self.conf_uid, self.history_uid)
        self.assertEqual([item["role"] for item in persisted], ["human", "ai", "ai"])
        self.assertEqual(persisted[-1]["content"], "Kok jadi sepi,ya.")

        batch = create_batch_input("Aku masih di sini.", None, "User")
        _ = [output async for output in agent.chat(batch)]
        next_request = llm.calls[-1]["messages"]
        self.assertTrue(
            any(
                message["role"] == "assistant"
                and "Kok jadi sepi" in str(message["content"])
                for message in next_request
            )
        )
        self.assertTrue(
            any(
                message["role"] == "user"
                and "Aku masih di sini" in str(message["content"])
                for message in next_request
            )
        )
        self.assertTrue(
            any(
                payload.get("type") == "latency-event"
                and payload.get("event") == "response-complete"
                and payload.get("metrics", {}).get("request_origin") == "proactive"
                for payload in sent_payloads
            )
        )

    async def test_relationship_and_character_memory_are_not_changed(self):
        agent, _llm = _make_agent(self.conf_uid, self.history_uid)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")
        agent.add_character_memory("user suka ramen", explicit=True)
        relationship_before = agent.relationship_status
        memories_before = agent.list_character_memories()
        _ = [output async for output in agent.chat_proactively()]
        self.assertEqual(agent.relationship_status, relationship_before)
        self.assertEqual(agent.list_character_memories(), memories_before)
        self.assertIn("dating", _llm.calls[-1]["system"])
        self.assertIn("user suka ramen", _llm.calls[-1]["system"])

    def test_latency_origin_defaults_to_user_and_supports_proactive(self):
        async def send(_payload):
            return None

        self.assertEqual(
            RequestLatencyTracker(send, "fake", "model").metrics()["request_origin"],
            "user",
        )
        self.assertEqual(
            RequestLatencyTracker(
                send, "fake", "model", request_origin="proactive"
            ).metrics()["request_origin"],
            "proactive",
        )


class ProactiveGuardTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _handler_with_active_chat():
        handler = WebSocketHandler.__new__(WebSocketHandler)
        handler.client_contexts = {"client": SimpleNamespace(history_uid="chat")}
        handler.client_connections = {"client": SimpleNamespace(send_text=AsyncMock())}
        handler._proactive_maintenance = set()
        handler._proactive_timer_tasks = {}
        handler.current_conversation_tasks = {}
        handler.chat_group_manager = SimpleNamespace(get_client_group=lambda _uid: None)
        return handler

    async def test_disconnected_client_is_not_eligible_for_generation(self):
        handler = WebSocketHandler.__new__(WebSocketHandler)
        handler.client_contexts = {}
        handler.client_connections = {}
        handler._proactive_maintenance = set()
        handler.chat_group_manager = SimpleNamespace(get_client_group=lambda _uid: None)
        self.assertFalse(handler._proactive_conditions_allow("client", "chat"))

    async def test_normal_llm_task_blocks_proactive_generation(self):
        gate = asyncio.Event()

        async def normal_turn():
            await gate.wait()

        handler = self._handler_with_active_chat()
        active = asyncio.create_task(normal_turn())
        handler.current_conversation_tasks["client"] = active
        clock = _Clock()
        machine = ProactiveStateMachine(
            ProactiveChatConfig(
                initial_idle_min_seconds=0,
                initial_idle_max_seconds=0,
            ),
            monotonic=clock,
            randint=lambda minimum, _maximum: minimum,
        )
        state = machine.new_state("chat")
        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=AsyncMock(return_value="proactive"),
        ) as generate:
            timer = asyncio.create_task(
                handler._run_proactive_timer("client", "chat", state, machine)
            )
            handler._proactive_timer_tasks["client"] = timer
            await asyncio.sleep(0.01)
            generate.assert_not_awaited()
            timer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await timer
        gate.set()
        await active

    async def test_one_timer_generates_only_one_turn_at_a_time(self):
        handler = self._handler_with_active_chat()
        clock = _Clock()
        machine = ProactiveStateMachine(
            ProactiveChatConfig(
                initial_idle_min_seconds=0,
                initial_idle_max_seconds=0,
                followup_idle_min_seconds=100,
                followup_idle_max_seconds=100,
            ),
            monotonic=clock,
            randint=lambda minimum, _maximum: minimum,
        )
        state = machine.new_state("chat")
        generated = asyncio.Event()

        async def generate_once(**_kwargs):
            generated.set()
            return "proactive"

        with patch(
            "src.open_llm_vtuber.websocket_handler.process_single_conversation",
            new=generate_once,
        ):
            timer = asyncio.create_task(
                handler._run_proactive_timer("client", "chat", state, machine)
            )
            handler._proactive_timer_tasks["client"] = timer
            await asyncio.wait_for(generated.wait(), 1)
            self.assertEqual(state.consecutive_ignored_proactive, 1)
            self.assertIs(handler.current_conversation_tasks["client"], timer)
            timer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await timer

    async def test_generation_flag_prevents_overlap(self):
        clock = _Clock()
        machine = ProactiveStateMachine(
            ProactiveChatConfig(), monotonic=clock, randint=lambda _a, _b: 45
        )
        state = machine.new_state("chat")
        clock.advance(45)
        state.proactive_generation_in_progress = True
        self.assertFalse(machine.is_eligible(state))


class ProactiveFollowupStateTests(unittest.TestCase):
    """Deterministic ignored-proactive state; no LLM anywhere in this path."""

    QUESTION = "Sekarang mau apa biar aktingnya berhenti?"
    STATEMENT = "Gak usah senyum-senyum gitu..."

    def _machine_and_state(self):
        clock = _Clock()
        machine = ProactiveStateMachine(
            ProactiveChatConfig(
                initial_idle_min_seconds=0,
                initial_idle_max_seconds=0,
                followup_idle_min_seconds=10,
                followup_idle_max_seconds=10,
            ),
            monotonic=clock,
            randint=lambda minimum, _maximum: minimum,
        )
        return machine, machine.new_state("chat")

    def test_message_expects_response_detector(self):
        self.assertTrue(message_expects_response(self.QUESTION))
        self.assertTrue(message_expects_response("Kok gak jawab?"))
        self.assertTrue(message_expects_response("Gimana sih"))
        self.assertFalse(message_expects_response(self.STATEMENT))
        self.assertFalse(message_expects_response("Lah, aku nanya loh."))
        self.assertFalse(message_expects_response(None))
        self.assertFalse(message_expects_response(""))

    def test_proactive_question_ignored_yields_question_context(self):
        machine, state = self._machine_and_state()
        machine.record_proactive_sent(state, response_text=self.QUESTION)
        context = machine.proactive_followup_context(state)
        self.assertTrue(context.previous_proactive_ignored)
        self.assertTrue(context.previous_proactive_expected_response)
        self.assertEqual(context.consecutive_ignored, 1)

    def test_proactive_statement_ignored_is_not_marked_as_question(self):
        machine, state = self._machine_and_state()
        machine.record_proactive_sent(state, response_text=self.STATEMENT)
        context = machine.proactive_followup_context(state)
        self.assertTrue(context.previous_proactive_ignored)
        self.assertFalse(context.previous_proactive_expected_response)

    def test_user_reply_clears_ignored_question_state(self):
        machine, state = self._machine_and_state()
        machine.record_proactive_sent(state, response_text=self.QUESTION)
        machine.record_user_activity(state)
        context = machine.proactive_followup_context(state)
        self.assertFalse(context.previous_proactive_ignored)
        self.assertFalse(context.previous_proactive_expected_response)
        self.assertEqual(context.consecutive_ignored, 0)
        self.assertIsNone(state.last_proactive_text)

    def test_consecutive_ignored_count_is_exposed(self):
        machine, state = self._machine_and_state()
        for index, text in enumerate(
            [self.QUESTION, self.STATEMENT, "Aku di sini."], start=1
        ):
            machine.record_proactive_sent(state, response_text=text)
            self.assertEqual(
                machine.proactive_followup_context(state).consecutive_ignored, index
            )

    def test_followup_context_dict_round_trip(self):
        machine, state = self._machine_and_state()
        machine.record_proactive_sent(state, response_text=self.QUESTION)
        context = machine.proactive_followup_context(state)
        self.assertEqual(ProactiveFollowupContext.from_dict(context.as_dict()), context)
        self.assertIsNone(ProactiveFollowupContext.from_dict(None))
        self.assertIsNone(ProactiveFollowupContext.from_dict("bogus"))


class ProactiveFollowupPromptTests(unittest.IsolatedAsyncioTestCase):
    """Contract: ignored context reaches the proactive prompt, nothing else."""

    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.conf_uid = "mili-proactive"
        self.history_uid = create_new_history(self.conf_uid)
        store_message(self.conf_uid, self.history_uid, "human", "Aku lagi belajar.")
        store_message(self.conf_uid, self.history_uid, "ai", "Belajar apa?")

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    @staticmethod
    def _ignored_context(question: bool) -> ProactiveFollowupContext:
        return ProactiveFollowupContext(
            previous_proactive_ignored=True,
            consecutive_ignored=1,
            previous_proactive_expected_response=question,
        )

    async def test_ignored_question_reaches_proactive_prompt(self):
        agent, llm = _make_agent(self.conf_uid, self.history_uid)
        _ = [
            output
            async for output in agent.chat_proactively(
                followup_context=self._ignored_context(question=True)
            )
        ]
        system = llm.calls[-1]["system"]
        self.assertIn("Internal follow-up context for this turn only", system)
        self.assertIn("asked the user a direct question", system)
        self.assertIn("unanswered question", system)
        self.assertIn("never repeat the exact same question", system)

    async def test_ignored_statement_does_not_claim_unanswered_question(self):
        agent, llm = _make_agent(self.conf_uid, self.history_uid)
        _ = [
            output
            async for output in agent.chat_proactively(
                followup_context=self._ignored_context(question=False)
            )
        ]
        system = llm.calls[-1]["system"]
        self.assertIn("was a statement, not a question", system)
        self.assertIn("do NOT claim the user failed to answer", system)
        self.assertNotIn("asked the user a direct question", system)

    async def test_no_ignored_context_produces_no_followup_block(self):
        agent, llm = _make_agent(self.conf_uid, self.history_uid)
        _ = [output async for output in agent.chat_proactively()]
        self.assertNotIn(
            "Internal follow-up context for this turn only", llm.calls[-1]["system"]
        )

    async def test_no_extra_llm_call_for_classification(self):
        agent, llm = _make_agent(self.conf_uid, self.history_uid)
        _ = [
            output
            async for output in agent.chat_proactively(
                followup_context=self._ignored_context(question=True)
            )
        ]
        self.assertEqual(len(llm.calls), 1)

    def test_followup_instruction_contract(self):
        question = format_followup_instruction(self._ignored_context(question=True))
        self.assertIn("Consecutive proactive messages ignored: 1", question)
        self.assertIn("mild confusion or teasing", question)
        self.assertIn("more impatient or annoyed", question)
        self.assertIn("resigned, sulking", question)
        self.assertIn("never mention counters, timers", question)
        statement = format_followup_instruction(self._ignored_context(question=False))
        self.assertIn("do NOT claim the user failed to answer", statement)
        answered = format_followup_instruction(
            ProactiveFollowupContext(
                previous_proactive_ignored=False,
                consecutive_ignored=0,
                previous_proactive_expected_response=True,
            )
        )
        self.assertIsNone(answered)
        self.assertIsNone(format_followup_instruction(None))


def _spin_for(weights, target):
    """Deterministic rng value that lands inside ``target``'s wheel bucket."""
    ordered = [
        (key, float(weights[key]))
        for key in INTENT_SELECTION_ORDER
        if float(weights.get(key, 0.0)) > 0.0
    ]
    total = sum(weight for _, weight in ordered)
    cursor = 0.0
    for key, weight in ordered:
        if key == target:
            return (cursor + weight / 2) / total
        cursor += weight
    return None


class ProactiveIntentSelectionTests(unittest.TestCase):
    """Deterministic weighted intent selection (no LLM involved)."""

    FULL_SIGNALS = ProactiveIntentSignals(
        has_useful_memory=True,
        has_recent_context=True,
        unfinished_topic=False,
    )

    def setUp(self):
        self.machine = ProactiveStateMachine(
            ProactiveChatConfig(
                initial_idle_min_seconds=0,
                initial_idle_max_seconds=0,
            ),
            monotonic=_Clock(),
            randint=lambda minimum, _maximum: minimum,
            random=lambda: 0.0,
        )
        self.state = self.machine.new_state("chat")

    def test_unanswered_question_priority_beats_random_selection(self):
        followup = ProactiveFollowupContext(
            previous_proactive_ignored=True,
            consecutive_ignored=2,
            previous_proactive_expected_response=True,
        )
        intent = resolve_proactive_intent(
            followup,
            self.state,
            self.machine,
            self.FULL_SIGNALS,
            random=lambda: 0.99,
        )
        self.assertEqual(intent, ProactiveIntent.REACT_TO_IGNORED_QUESTION)

    def test_ignored_statement_uses_weighted_selection(self):
        followup = ProactiveFollowupContext(
            previous_proactive_ignored=True,
            consecutive_ignored=1,
            previous_proactive_expected_response=False,
        )
        spin = _spin_for(
            self.machine.effective_intent_weights(self.state, self.FULL_SIGNALS),
            ProactiveIntent.START_NEW_TOPIC,
        )
        intent = resolve_proactive_intent(
            followup,
            self.state,
            self.machine,
            self.FULL_SIGNALS,
            random=lambda: spin,
        )
        self.assertEqual(intent, ProactiveIntent.START_NEW_TOPIC)

    def test_every_weighted_intent_can_be_selected(self):
        cases = {
            ProactiveIntent.REACT_TO_SILENCE: self.FULL_SIGNALS,
            ProactiveIntent.CONTINUE_PREVIOUS_TOPIC: self.FULL_SIGNALS,
            ProactiveIntent.START_NEW_TOPIC: self.FULL_SIGNALS,
            ProactiveIntent.ASK_USER_SOMETHING: self.FULL_SIGNALS,
            ProactiveIntent.BRING_UP_MEMORY: self.FULL_SIGNALS,
            ProactiveIntent.CASUAL_OBSERVATION: self.FULL_SIGNALS,
        }
        for expected, signals in cases.items():
            with self.subTest(intent=expected):
                weights = self.machine.effective_intent_weights(self.state, signals)
                spin = _spin_for(weights, expected)
                self.assertIsNotNone(spin, f"{expected} has zero weight")
                self.assertEqual(
                    self.machine.select_proactive_intent(
                        self.state, signals, random=lambda: spin
                    ),
                    expected,
                )

    def test_repeated_intent_is_penalized(self):
        self.state.recent_proactive_intents = [
            ProactiveIntent.START_NEW_TOPIC,
            ProactiveIntent.START_NEW_TOPIC,
            ProactiveIntent.START_NEW_TOPIC,
        ]
        weights = self.machine.effective_intent_weights(self.state, self.FULL_SIGNALS)
        self.assertAlmostEqual(weights[ProactiveIntent.START_NEW_TOPIC], 30 * 0.25**3)
        self.assertEqual(
            weights[ProactiveIntent.ASK_USER_SOMETHING],
            DEFAULT_INTENT_WEIGHTS[ProactiveIntent.ASK_USER_SOMETHING],
        )

    def test_silence_acknowledgment_decays_fast(self):
        self.state.recent_proactive_intents = [ProactiveIntent.REACT_TO_SILENCE]
        weights = self.machine.effective_intent_weights(self.state, self.FULL_SIGNALS)
        self.assertAlmostEqual(weights[ProactiveIntent.REACT_TO_SILENCE], 0.5)

    def test_missing_memory_disables_bring_up_memory(self):
        signals = ProactiveIntentSignals(
            has_useful_memory=False,
            has_recent_context=True,
            unfinished_topic=False,
        )
        weights = self.machine.effective_intent_weights(self.state, signals)
        self.assertEqual(weights[ProactiveIntent.BRING_UP_MEMORY], 0.0)
        self.assertIsNone(_spin_for(weights, ProactiveIntent.BRING_UP_MEMORY))

    def test_little_context_boosts_self_initiated_intents(self):
        signals = ProactiveIntentSignals(
            has_useful_memory=False,
            has_recent_context=False,
            unfinished_topic=False,
        )
        weights = self.machine.effective_intent_weights(self.state, signals)
        self.assertEqual(weights[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC], 0.0)
        self.assertEqual(weights[ProactiveIntent.START_NEW_TOPIC], 45.0)
        self.assertEqual(weights[ProactiveIntent.ASK_USER_SOMETHING], 30.0)
        self.assertEqual(weights[ProactiveIntent.CASUAL_OBSERVATION], 15.0)

    def test_unfinished_topic_boosts_continue_previous_topic(self):
        signals = ProactiveIntentSignals(
            has_useful_memory=False,
            has_recent_context=True,
            unfinished_topic=True,
        )
        weights = self.machine.effective_intent_weights(self.state, signals)
        self.assertEqual(weights[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC], 40.0)

    def test_recent_intents_are_trimmed_to_three(self):
        for index in range(5):
            self.machine.record_proactive_sent(
                self.state,
                response_text=f"pesan {index}",
                intent=ProactiveIntent.START_NEW_TOPIC,
            )
        self.assertEqual(len(self.state.recent_proactive_intents), 3)

    def test_config_weights_override_defaults_with_validation(self):
        machine = ProactiveStateMachine(
            ProactiveChatConfig(
                initial_idle_min_seconds=0,
                initial_idle_max_seconds=0,
                intent_weights={"start_new_topic": 100, "unknown_key": 7},
            ),
            monotonic=_Clock(),
            randint=lambda minimum, _maximum: minimum,
        )
        weights = machine.effective_intent_weights(self.state, self.FULL_SIGNALS)
        self.assertEqual(weights[ProactiveIntent.START_NEW_TOPIC], 100.0)
        self.assertEqual(
            weights[ProactiveIntent.REACT_TO_SILENCE],
            DEFAULT_INTENT_WEIGHTS[ProactiveIntent.REACT_TO_SILENCE],
        )
        with self.assertRaises(ValueError):
            ProactiveChatConfig(intent_weights={"start_new_topic": -1})

    def test_old_config_without_intent_weights_still_loads(self):
        self.assertIsNone(ProactiveChatConfig().intent_weights)
        settings = BasicMemoryAgentConfig(llm_provider="ollama_llm")
        self.assertIsNone(settings.proactive_intent_weights)
        weights = self.machine.effective_intent_weights(self.state, self.FULL_SIGNALS)
        self.assertEqual(weights, DEFAULT_INTENT_WEIGHTS)

    def test_intent_context_round_trip_and_fallback(self):
        context = ProactiveIntentContext(
            intent=ProactiveIntent.BRING_UP_MEMORY,
            user_has_replied_since_last_proactive=False,
            consecutive_ignored=2,
            recent_silence_acknowledgment=True,
        )
        self.assertEqual(ProactiveIntentContext.from_dict(context.as_dict()), context)
        self.assertIsNone(ProactiveIntentContext.from_dict(None))
        fallback = ProactiveIntentContext.from_dict({"intent": "bogus"})
        self.assertEqual(fallback.intent, ProactiveIntent.CASUAL_OBSERVATION)


class ProactiveIntentPromptTests(unittest.IsolatedAsyncioTestCase):
    """Contract: internal intent context reaches the prompt, not history."""

    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.conf_uid = "mili-proactive"
        self.history_uid = create_new_history(self.conf_uid)
        store_message(self.conf_uid, self.history_uid, "human", "Aku lagi belajar.")
        store_message(self.conf_uid, self.history_uid, "ai", "Belajar apa?")

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    @staticmethod
    def _intent_context(intent):
        return ProactiveIntentContext(
            intent=intent,
            user_has_replied_since_last_proactive=True,
            consecutive_ignored=0,
            recent_silence_acknowledgment=False,
        )

    async def _generate(self, **kwargs):
        agent, llm = _make_agent(self.conf_uid, self.history_uid)
        outputs = [output async for output in agent.chat_proactively(**kwargs)]
        self.assertTrue(outputs)
        return agent, llm

    async def test_intent_context_reaches_proactive_prompt(self):
        agent, llm = await self._generate(
            intent_context=self._intent_context(ProactiveIntent.START_NEW_TOPIC)
        )
        system = llm.calls[-1]["system"]
        self.assertIn("Internal proactive context for this turn only", system)
        self.assertIn("intent: start_new_topic", system)
        self.assertIn("never announce a topic", system.lower())

    async def test_ignored_question_intent_reaches_prompt(self):
        agent, llm = await self._generate(
            followup_context=ProactiveFollowupContext(
                previous_proactive_ignored=True,
                consecutive_ignored=1,
                previous_proactive_expected_response=True,
            ),
            intent_context=self._intent_context(
                ProactiveIntent.REACT_TO_IGNORED_QUESTION
            ),
        )
        system = llm.calls[-1]["system"]
        self.assertIn("intent: react_to_ignored_question", system)
        self.assertIn("unanswered question", system)

    async def test_anti_fake_history_contract_in_prompt(self):
        agent, llm = await self._generate(
            intent_context=self._intent_context(ProactiveIntent.START_NEW_TOPIC)
        )
        system = llm.calls[-1]["system"]
        self.assertIn("Never claim", system)
        self.assertIn("specific past personal events", system)

    async def test_intent_metadata_never_enters_history(self):
        agent, llm = await self._generate(
            intent_context=self._intent_context(ProactiveIntent.START_NEW_TOPIC)
        )
        for message in get_history(self.conf_uid, self.history_uid):
            self.assertNotIn("Internal proactive context", str(message))
            self.assertNotIn("intent:", str(message))

    async def test_no_fake_user_message_and_single_llm_call(self):
        agent, llm = await self._generate(
            intent_context=self._intent_context(ProactiveIntent.CASUAL_OBSERVATION)
        )
        self.assertEqual(
            [message["role"] for message in agent._memory],
            ["user", "assistant", "assistant"],
        )
        self.assertEqual(len(llm.calls), 1)

    def test_intent_instruction_contract(self):
        full = format_intent_instruction(
            self._intent_context(ProactiveIntent.START_NEW_TOPIC)
        )
        self.assertIn("intent: start_new_topic", full)
        self.assertIn("consecutive_ignored: 0", full)
        self.assertIn("recent_silence_acknowledgment: false", full)
        self.assertIn("never announce a topic", full)
        compact = format_intent_instruction(
            self._intent_context(ProactiveIntent.START_NEW_TOPIC),
            include_guidance=False,
        )
        self.assertIn("intent: start_new_topic", compact)
        self.assertNotIn("never announce a topic", compact)
        self.assertIsNone(format_intent_instruction(None))


if __name__ == "__main__":
    unittest.main()
