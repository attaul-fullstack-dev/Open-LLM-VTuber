import copy
import unittest

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.context_window import (
    DEFAULT_CONTEXT_LIMIT,
    ContextBudgetExceeded,
    estimate_message_tokens,
    resolve_context_limit,
    select_messages_for_context,
)
from src.open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig


def message(role: str, content: str):
    return {"role": role, "content": content}


class ContextWindowTests(unittest.TestCase):
    def test_short_history_is_unchanged(self):
        messages = [
            message("user", "halo"),
            message("assistant", "iya?"),
            message("user", "lagi apa?"),
        ]
        selection = select_messages_for_context(
            messages=messages,
            system_prompt="persona Mili",
            model="mistral-small-latest",
            reserved_output_tokens=384,
            safety_margin=1024,
            protected_start=2,
        )
        self.assertFalse(selection.stats.trimmed)
        self.assertEqual(selection.messages, messages)
        self.assertEqual(selection.stats.context_limit, 256_000)

    def test_long_history_keeps_recent_complete_turn(self):
        old_turn = [
            message("user", "lama " * 500),
            message("assistant", "jawaban lama " * 500),
        ]
        recent_turn = [
            message("user", "kode saya mangga"),
            message("assistant", "oke"),
        ]
        current = [message("user", "kode saya apa?")]
        messages = [*old_turn, *recent_turn, *current]
        selection = select_messages_for_context(
            messages=messages,
            system_prompt="persona tetap tersedia",
            model="test-small",
            context_window_override=800,
            reserved_output_tokens=100,
            safety_margin=100,
            protected_start=4,
        )
        self.assertTrue(selection.stats.trimmed)
        self.assertEqual(selection.messages, [*recent_turn, *current])
        self.assertEqual(selection.messages[0]["role"], "user")
        self.assertLessEqual(
            selection.stats.estimated_input_tokens,
            selection.stats.maximum_input_budget,
        )

    def test_selection_does_not_mutate_full_transcript(self):
        messages = [
            message("user", "old " * 1000),
            message("assistant", "old answer " * 1000),
            message("user", "current"),
        ]
        before = copy.deepcopy(messages)
        selection = select_messages_for_context(
            messages=messages,
            system_prompt="system",
            model="test",
            context_window_override=500,
            reserved_output_tokens=100,
            safety_margin=100,
            protected_start=2,
        )
        self.assertTrue(selection.stats.trimmed)
        self.assertEqual(messages, before)
        self.assertEqual(len(messages), 3)

    def test_system_and_current_turn_are_mandatory(self):
        system = "persona " * 80
        current = message("user", "current " * 80)
        with self.assertRaises(ContextBudgetExceeded):
            select_messages_for_context(
                messages=[current],
                system_prompt=system,
                model="test",
                context_window_override=180,
                reserved_output_tokens=50,
                safety_margin=30,
                protected_start=0,
            )

    def test_oversized_current_message_is_rejected(self):
        with self.assertRaises(ContextBudgetExceeded):
            select_messages_for_context(
                messages=[message("user", "x" * 6000)],
                system_prompt="required persona",
                model="test",
                context_window_override=500,
                reserved_output_tokens=100,
                safety_margin=100,
                protected_start=0,
            )

    def test_unknown_model_uses_conservative_fallback(self):
        limit, used_fallback = resolve_context_limit("unknown-provider-model")
        self.assertEqual(limit, DEFAULT_CONTEXT_LIMIT)
        self.assertTrue(used_fallback)

    def test_reserved_output_and_safety_margin_reduce_input_budget(self):
        selection = select_messages_for_context(
            messages=[message("user", "halo")],
            system_prompt="system",
            model="test",
            context_window_override=4096,
            reserved_output_tokens=384,
            safety_margin=512,
            protected_start=0,
        )
        self.assertEqual(selection.stats.reserved_output, 384)
        self.assertEqual(selection.stats.maximum_input_budget, 3200)

    def test_tool_schema_is_counted(self):
        tool = {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "look up data",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        selection = select_messages_for_context(
            messages=[message("user", "halo")],
            system_prompt="system",
            model="test",
            context_window_override=4096,
            reserved_output_tokens=384,
            safety_margin=512,
            tools=[tool],
            protected_start=0,
        )
        self.assertGreater(selection.stats.tool_tokens, 0)
        self.assertGreater(
            selection.stats.estimated_input_tokens,
            estimate_message_tokens(message("user", "halo")),
        )


class _FakeLLM:
    model = "test-model"
    max_tokens = 100

    def __init__(self):
        self.calls = []

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls.append(copy.deepcopy(messages))
        yield "oke."


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class BasicMemoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def make_agent(self, llm, context_window_override):
        tts_config = TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
        )
        return BasicMemoryAgent(
            llm=llm,
            system="persona wajib",
            live2d_model=_FakeLive2D(),
            tts_preprocessor_config=tts_config,
            context_window_override=context_window_override,
            context_safety_margin=100,
        )

    async def test_agent_trims_request_but_keeps_full_memory(self):
        llm = _FakeLLM()
        agent = self.make_agent(llm, context_window_override=800)
        agent._memory = [
            message("user", "old " * 600),
            message("assistant", "old answer " * 600),
            message("user", "kode saya mangga"),
            message("assistant", "oke"),
        ]
        stored_before = copy.deepcopy(agent._memory)
        input_data = BatchInput(texts=[TextData(TextSource.INPUT, "kode saya apa?")])

        outputs = [output async for output in agent.chat(input_data)]

        self.assertTrue(outputs)
        self.assertEqual(len(llm.calls), 1)
        sent = llm.calls[0]
        self.assertNotIn(stored_before[0], sent)
        self.assertIn(message("user", "kode saya mangga"), sent)
        self.assertEqual(sent[-1]["role"], "user")
        self.assertEqual(agent._memory[:4], stored_before)
        self.assertEqual(agent._memory[4]["content"], "kode saya apa?")
        self.assertEqual(agent._memory[-1], message("assistant", "oke."))

    async def test_agent_rejects_oversized_user_before_provider_call(self):
        llm = _FakeLLM()
        agent = self.make_agent(llm, context_window_override=350)
        input_data = BatchInput(texts=[TextData(TextSource.INPUT, "x" * 6000)])

        outputs = [output async for output in agent.chat(input_data)]

        self.assertTrue(outputs)
        self.assertEqual(llm.calls, [])
        self.assertEqual(agent._memory[0]["content"], "x" * 6000)


if __name__ == "__main__":
    unittest.main()
