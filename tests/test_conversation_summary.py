import copy
import os
import tempfile
import unittest

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.context_window import (
    estimate_message_tokens,
    estimate_messages_tokens,
)
from src.open_llm_vtuber.agent.conversation_summary import SUMMARY_SYSTEM_PROMPT
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    get_metadata,
    store_message,
    update_metadate,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig


def message(role: str, content: str):
    return {"role": role, "content": content}


class _SummaryFakeLLM:
    model = "summary-test"
    max_tokens = 100

    def __init__(self):
        self.summary_prompts = []
        self.fail_summary = False

    async def chat_completion(self, messages, system=None, tools=None):
        if system == SUMMARY_SYSTEM_PROMPT:
            if self.fail_summary:
                raise TimeoutError("synthetic timeout")
            prompt = messages[0]["content"]
            self.summary_prompts.append(prompt)
            facts = []
            if "suka ramen" in prompt.lower():
                facts.append("User mengatakan paling suka ramen.")
            if "belajar html" in prompt.lower():
                facts.append("User sedang belajar HTML.")
            if "masalah login selesai" in prompt.lower():
                facts.append("Masalah login user sudah selesai.")
            yield " ".join(facts)
            return
        yield "oke"


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class RollingSummaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temporary_directory = tempfile.TemporaryDirectory()
        os.chdir(self._temporary_directory.name)
        self.conf_uid = "test-character"

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temporary_directory.cleanup()

    def make_agent(self, llm, history_uid, *, summary_min_new_messages=2):
        tts_config = TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
        )
        agent = BasicMemoryAgent(
            llm=llm,
            system="persona Mili wajib dipertahankan",
            live2d_model=_FakeLive2D(),
            tts_preprocessor_config=tts_config,
            context_window_override=700,
            context_safety_margin=100,
            summary_target_tokens=80,
            summary_max_tokens=100,
            summary_min_new_messages=summary_min_new_messages,
        )
        agent.set_memory_from_history(self.conf_uid, history_uid)
        return agent

    def create_history(self, messages):
        history_uid = create_new_history(self.conf_uid)
        for item in messages:
            role = "human" if item["role"] == "user" else "ai"
            store_message(
                self.conf_uid,
                history_uid,
                role,
                item["content"],
            )
        return history_uid

    async def test_short_chat_does_not_call_summarizer(self):
        stored = [message("user", "halo"), message("assistant", "iya?")]
        history_uid = self.create_history(stored)
        llm = _SummaryFakeLLM()
        agent = self.make_agent(llm, history_uid)
        messages = [*copy.deepcopy(agent._memory), message("user", "lagi apa?")]

        request = await agent._prepare_context_with_summary(
            messages,
            agent._system,
            protected_start=len(messages) - 1,
        )

        self.assertEqual(request, messages)
        self.assertEqual(llm.summary_prompts, [])
        self.assertNotIn("conversation_summary", get_metadata(self.conf_uid, history_uid))

    async def test_initial_summary_preserves_transcript_and_recent_turn(self):
        stored = [
            message("user", "Aku paling suka ramen."),
            message("assistant", "Oke, aku ingat."),
            message("user", "lama " * 700),
            message("assistant", "jawaban lama " * 700),
            message("user", "kode terbaru mangga"),
            message("assistant", "sip"),
        ]
        history_uid = self.create_history(stored)
        transcript_before = copy.deepcopy(get_history(self.conf_uid, history_uid))
        llm = _SummaryFakeLLM()
        agent = self.make_agent(llm, history_uid)
        messages = [*copy.deepcopy(agent._memory), message("user", "kodenya apa?")]

        request = await agent._prepare_context_with_summary(
            messages,
            agent._system,
            protected_start=len(messages) - 1,
        )

        metadata = get_metadata(self.conf_uid, history_uid)
        self.assertEqual(len(llm.summary_prompts), 1)
        self.assertIn("suka ramen", metadata["conversation_summary"].lower())
        self.assertEqual(metadata["summary_through_message_index"], 4)
        self.assertEqual(get_history(self.conf_uid, history_uid), transcript_before)
        self.assertIn(message("user", "kode terbaru mangga"), request)
        self.assertEqual(request[-1], message("user", "kodenya apa?"))
        self.assertEqual(request[0]["role"], "system")

    async def test_incremental_update_only_sends_newly_evicted_turns(self):
        stored = [
            message("user", "Aku paling suka ramen."),
            message("assistant", "Dicatat."),
            message("user", "lama " * 700),
            message("assistant", "jawaban lama " * 700),
            message("user", "Aku sedang belajar HTML."),
            message("assistant", "Kita latihan."),
        ]
        history_uid = self.create_history(stored)
        llm = _SummaryFakeLLM()
        agent = self.make_agent(llm, history_uid)

        first_messages = [*copy.deepcopy(agent._memory), message("user", "lanjut")]
        await agent._prepare_context_with_summary(
            first_messages,
            agent._system,
            protected_start=len(first_messages) - 1,
        )

        additions = [
            message("user", "padding " * 700),
            message("assistant", "jawaban padding " * 700),
            message("user", "Masalah login selesai."),
            message("assistant", "Bagus."),
        ]
        for item in additions:
            store_message(
                self.conf_uid,
                history_uid,
                "human" if item["role"] == "user" else "ai",
                item["content"],
            )
            agent._memory.append(copy.deepcopy(item))
        second_messages = [*copy.deepcopy(agent._memory), message("user", "sekarang?")]
        await agent._prepare_context_with_summary(
            second_messages,
            agent._system,
            protected_start=len(second_messages) - 1,
        )

        self.assertEqual(len(llm.summary_prompts), 2)
        newly_evicted = llm.summary_prompts[1].split("NEWLY EVICTED TURNS:", 1)[1]
        self.assertNotIn("paling suka ramen", newly_evicted.lower())
        self.assertIn("belajar html", newly_evicted.lower())
        metadata = get_metadata(self.conf_uid, history_uid)
        self.assertIn("suka ramen", metadata["conversation_summary"].lower())
        self.assertIn("belajar html", metadata["conversation_summary"].lower())
        self.assertGreater(metadata["summary_through_message_index"], 4)

    async def test_summary_persists_across_agent_recreation_and_is_isolated(self):
        stored_a = [
            message("user", "Aku paling suka ramen."),
            message("assistant", "Oke."),
            message("user", "lama " * 700),
            message("assistant", "lama " * 700),
            message("user", "recent"),
            message("assistant", "recent answer"),
        ]
        history_a = self.create_history(stored_a)
        history_b = self.create_history(
            [message("user", "halo"), message("assistant", "hai")]
        )
        llm = _SummaryFakeLLM()
        agent_a = self.make_agent(llm, history_a)
        request_a = [*copy.deepcopy(agent_a._memory), message("user", "lanjut")]
        await agent_a._prepare_context_with_summary(
            request_a,
            agent_a._system,
            protected_start=len(request_a) - 1,
        )

        recreated_a = self.make_agent(_SummaryFakeLLM(), history_a)
        agent_b = self.make_agent(_SummaryFakeLLM(), history_b)

        self.assertIn("ramen", recreated_a._summary_state.text.lower())
        self.assertEqual(agent_b._summary_state.text, "")
        self.assertEqual(agent_b._summary_state.summarized_through, 0)

    async def test_filler_produces_no_facts_and_no_hallucination(self):
        stored = [
            message("user", "halo"),
            message("assistant", "hai"),
            message("user", "wkwk iya oke makasih"),
            message("assistant", "sip"),
            message("user", "padding " * 700),
            message("assistant", "padding " * 700),
            message("user", "recent"),
            message("assistant", "recent"),
        ]
        history_uid = self.create_history(stored)
        llm = _SummaryFakeLLM()
        agent = self.make_agent(llm, history_uid)
        messages = [*copy.deepcopy(agent._memory), message("user", "lanjut")]

        await agent._prepare_context_with_summary(
            messages,
            agent._system,
            protected_start=len(messages) - 1,
        )

        metadata = get_metadata(self.conf_uid, history_uid)
        self.assertEqual(metadata["conversation_summary"], "")
        self.assertNotIn("warna biru", str(metadata).lower())
        self.assertGreater(metadata["summary_through_message_index"], 0)

    async def test_three_updates_keep_first_important_fact(self):
        stored = [
            message("user", "Aku paling suka ramen."),
            message("assistant", "Oke."),
            message("user", "padding awal " * 700),
            message("assistant", "padding awal " * 700),
            message("user", "recent awal"),
            message("assistant", "recent awal"),
        ]
        history_uid = self.create_history(stored)
        llm = _SummaryFakeLLM()
        agent = self.make_agent(llm, history_uid)

        for update_number in range(3):
            messages = [*copy.deepcopy(agent._memory), message("user", "current")]
            await agent._prepare_context_with_summary(
                messages,
                agent._system,
                protected_start=len(messages) - 1,
            )
            if update_number < 2:
                additions = [
                    message("user", f"padding {update_number} " * 700),
                    message("assistant", f"jawaban {update_number} " * 700),
                    message("user", f"recent {update_number}"),
                    message("assistant", f"recent answer {update_number}"),
                ]
                for item in additions:
                    store_message(
                        self.conf_uid,
                        history_uid,
                        "human" if item["role"] == "user" else "ai",
                        item["content"],
                    )
                    agent._memory.append(copy.deepcopy(item))

        self.assertEqual(len(llm.summary_prompts), 3)
        self.assertIn(
            "suka ramen",
            get_metadata(self.conf_uid, history_uid)["conversation_summary"].lower(),
        )

    async def test_summarizer_failure_keeps_old_summary_and_chat_context(self):
        stored = [
            message("user", "Aku paling suka ramen."),
            message("assistant", "Oke."),
            message("user", "padding " * 700),
            message("assistant", "padding " * 700),
            message("user", "recent"),
            message("assistant", "recent"),
        ]
        history_uid = self.create_history(stored)
        update_metadate(
            self.conf_uid,
            history_uid,
            {
                "conversation_summary": "User mengatakan paling suka ramen.",
                "summary_through_message_index": 2,
                "summary_updated_at": "2026-01-01T00:00:00+00:00",
            },
        )
        llm = _SummaryFakeLLM()
        llm.fail_summary = True
        agent = self.make_agent(llm, history_uid)
        messages = [*copy.deepcopy(agent._memory), message("user", "lanjut")]

        request = await agent._prepare_context_with_summary(
            messages,
            agent._system,
            protected_start=len(messages) - 1,
        )

        metadata = get_metadata(self.conf_uid, history_uid)
        self.assertEqual(metadata["conversation_summary"], "User mengatakan paling suka ramen.")
        self.assertEqual(metadata["summary_through_message_index"], 2)
        self.assertEqual(request[-1], message("user", "lanjut"))
        self.assertTrue(request[0]["content"].startswith("Conversation context"))

    async def test_summary_and_recent_context_fit_budget(self):
        stored = [
            message("user", "Aku paling suka ramen."),
            message("assistant", "Oke."),
            message("user", "padding " * 700),
            message("assistant", "padding " * 700),
            message("user", "recent fact"),
            message("assistant", "recent answer"),
        ]
        history_uid = self.create_history(stored)
        agent = self.make_agent(_SummaryFakeLLM(), history_uid)
        messages = [*copy.deepcopy(agent._memory), message("user", "current")]

        request = await agent._prepare_context_with_summary(
            messages,
            agent._system,
            protected_start=len(messages) - 1,
        )

        input_tokens = estimate_message_tokens(
            {"role": "system", "content": agent._system}
        ) + estimate_messages_tokens(request)
        self.assertLessEqual(input_tokens, 700 - 100 - 100)
        self.assertIn(message("user", "recent fact"), request)
        self.assertEqual(request[-1], message("user", "current"))


if __name__ == "__main__":
    unittest.main()
