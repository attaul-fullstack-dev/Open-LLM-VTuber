import copy
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.conversation_summary import SUMMARY_SYSTEM_PROMPT
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    get_metadata,
    store_message,
    update_summary_metadata,
)
from src.open_llm_vtuber.character_state import load_character_state
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


def message(role: str, content: str):
    return {"role": role, "content": content}


class _IntegrationFakeLLM:
    model = "integration-test"
    max_tokens = 100

    async def chat_completion(self, messages, system=None, tools=None):
        if system == SUMMARY_SYSTEM_PROMPT:
            yield "Konteks lama yang masih relevan telah diringkas."
            return
        yield "oke"


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class _FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, payload):
        self.messages.append(json.loads(payload))


class FinalIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temporary_directory = tempfile.TemporaryDirectory()
        os.chdir(self._temporary_directory.name)
        self.conf_uid = "mili-final-integration"

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temporary_directory.cleanup()

    def create_history(self, messages):
        history_uid = create_new_history(self.conf_uid)
        for item in messages:
            store_message(
                self.conf_uid,
                history_uid,
                "human" if item["role"] == "user" else "ai",
                item["content"],
            )
        return history_uid

    def make_agent(self, history_uid, *, context_window=900):
        tts_config = TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
        )
        agent = BasicMemoryAgent(
            llm=_IntegrationFakeLLM(),
            system="persona Mili wajib selalu dipertahankan",
            live2d_model=_FakeLive2D(),
            tts_preprocessor_config=tts_config,
            context_window_override=context_window,
            context_safety_margin=100,
            summary_target_tokens=80,
            summary_max_tokens=100,
            summary_min_new_messages=2,
        )
        agent.set_memory_from_history(self.conf_uid, history_uid)
        return agent

    async def test_dating_summary_trimming_and_recreation_integrate(self):
        stored = [
            message("user", "Mau nggak jadi pacar aku?"),
            message("assistant", "...Iya. Mau."),
            message("user", "history lama " * 700),
            message("assistant", "jawaban lama " * 700),
            message("user", "kode terbaru mangga"),
            message("assistant", "Oke, aku ingat."),
        ]
        history_uid = self.create_history(stored)
        transcript_before = copy.deepcopy(get_history(self.conf_uid, history_uid))
        agent = self.make_agent(history_uid)
        self.assertTrue(
            agent.observe_relationship_event(stored[0]["content"], stored[1]["content"])
        )

        current = message("user", "Kode terbaruku apa?")
        system = agent._relationship_system_prompt(agent._system)
        request = await agent._prepare_context_with_summary(
            [*copy.deepcopy(agent._memory), current],
            system,
            protected_start=len(agent._memory),
        )

        recreated = self.make_agent(history_uid)
        recreated_system = recreated._relationship_system_prompt(recreated._system)
        recreated_request = await recreated._prepare_context_with_summary(
            [*copy.deepcopy(recreated._memory), current],
            recreated_system,
            protected_start=len(recreated._memory),
        )
        selection = recreated._select_context(
            recreated_request,
            recreated_system,
            protected_start=len(recreated_request) - 1,
        )

        self.assertEqual(recreated.relationship_status, "dating")
        self.assertTrue(recreated._summary_state.text)
        self.assertIn(message("user", "kode terbaru mangga"), request)
        self.assertIn(message("user", "kode terbaru mangga"), recreated_request)
        self.assertEqual(recreated_request[-1], current)
        self.assertIn("persona Mili", recreated_system)
        self.assertIn("Current state: dating", recreated_system)
        self.assertLessEqual(
            selection.stats.estimated_input_tokens,
            selection.stats.maximum_input_budget,
        )
        self.assertEqual(get_history(self.conf_uid, history_uid), transcript_before)

    async def test_switching_keeps_summary_relationship_and_memory_isolated(self):
        history_a = self.create_history(
            [message("user", "fakta A"), message("assistant", "ingat A")]
        )
        history_b = self.create_history(
            [message("user", "fakta B"), message("assistant", "ingat B")]
        )
        update_summary_metadata(
            self.conf_uid,
            history_a,
            expected_summarized_through=0,
            conversation_summary="summary A",
            summarized_through=1,
            summary_updated_at="2026-01-01T00:00:00+00:00",
        )
        update_summary_metadata(
            self.conf_uid,
            history_b,
            expected_summarized_through=0,
            conversation_summary="summary B",
            summarized_through=1,
            summary_updated_at="2026-01-01T00:00:00+00:00",
        )
        agent = self.make_agent(history_a, context_window=4000)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")
        agent.set_memory_from_history(self.conf_uid, history_b)

        for history_uid, summary, fact in (
            (history_a, "summary A", "fakta A"),
            (history_b, "summary B", "fakta B"),
            (history_a, "summary A", "fakta A"),
            (history_b, "summary B", "fakta B"),
        ):
            agent.set_memory_from_history(self.conf_uid, history_uid)
            # Relationship is character-level now: it stays global across chats.
            self.assertEqual(agent.relationship_status, "dating")
            self.assertEqual(agent._summary_state.text, summary)
            self.assertTrue(any(fact in item["content"] for item in agent._memory))

    def test_relationship_metadata_write_failure_keeps_valid_old_state(self):
        history_uid = self.create_history([])
        agent = self.make_agent(history_uid)
        metadata_before = copy.deepcopy(get_metadata(self.conf_uid, history_uid))

        with patch(
            "src.open_llm_vtuber.character_state.save_character_state",
            return_value=False,
        ):
            self.assertFalse(
                agent.set_relationship_status("close", trigger="synthetic_failure")
            )

        self.assertEqual(agent.relationship_status, "stranger")
        self.assertEqual(get_metadata(self.conf_uid, history_uid), metadata_before)
        self.assertEqual(
            load_character_state(self.conf_uid).relationship_status, "stranger"
        )

    async def test_reconnect_and_restart_preserve_history_summary_relationship(self):
        stored = [
            message("user", "Mau nggak jadi pacar aku?"),
            message("assistant", "...Iya. Mau."),
            message("user", "besok kita nonton ya"),
            message("assistant", "Hmm, kalau aku sempat."),
        ]
        history_uid = self.create_history(stored)
        agent = self.make_agent(history_uid)
        agent.observe_relationship_event(stored[0]["content"], stored[1]["content"])
        self.assertEqual(agent.relationship_status, "dating")

        history_files_before = len(os.listdir("chat_history/mili-final-integration"))

        # Simulate disconnect + reconnect (new session agent, same history)
        reconnected = self.make_agent(history_uid)
        self.assertEqual(reconnected.relationship_status, "dating")
        self.assertEqual(
            [m["content"] for m in reconnected._memory],
            [m["content"] for m in stored],
        )

        # Simulate backend restart: fresh agent, reload from disk
        restarted = self.make_agent(history_uid)
        self.assertEqual(restarted.relationship_status, "dating")
        self.assertEqual(
            load_character_state(self.conf_uid).relationship_status,
            "dating",
        )
        self.assertEqual(len(restarted._memory), len(stored))

        # No new conversation is created just by reconnecting/restarting
        history_files_after = len(os.listdir("chat_history/mili-final-integration"))
        self.assertEqual(history_files_before, history_files_after)

    async def test_context_stress_with_summary_relationship_and_reserved_output(self):
        class _StressLLM:
            model = "mistral-small-latest"
            max_tokens = 384
            summary_calls = 0

            async def chat_completion(self, messages, system=None, tools=None):
                if system == SUMMARY_SYSTEM_PROMPT:
                    self.summary_calls += 1
                    yield "Konteks lama yang relevan telah diringkas."
                    return
                yield "oke"

        stored = []
        for i in range(60):
            stored.append(message("user", f"pesan lama nomor {i} " + "isi" * 40))
            stored.append(
                message("assistant", f"jawaban lama nomor {i} " + "isi" * 40)
            )
        stored.append(message("user", "pesan terbaru pendek"))
        stored.append(message("assistant", "jawaban terbaru pendek"))
        history_uid = self.create_history(stored)
        transcript_before = copy.deepcopy(get_history(self.conf_uid, history_uid))

        llm = _StressLLM()
        tts_config = TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
        )
        agent = BasicMemoryAgent(
            llm=llm,
            system="persona Mili wajib selalu dipertahankan",
            live2d_model=_FakeLive2D(),
            tts_preprocessor_config=tts_config,
            context_window_override=4000,
            context_safety_margin=200,
            summary_target_tokens=200,
            summary_max_tokens=384,
            summary_min_new_messages=2,
        )
        agent.set_memory_from_history(self.conf_uid, history_uid)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")

        current = message("user", "pesan terbaruku apa?")
        system_prompt = agent._relationship_system_prompt(agent._system)
        full_request = [*copy.deepcopy(agent._memory), current]

        # Relationship context is inside the system prompt (counted against budget)
        self.assertIn("Current state: dating", system_prompt)
        # Trimming is actually active on a large synthetic history
        initial_selection = agent._select_context(
            full_request,
            system_prompt,
            protected_start=len(agent._memory),
        )
        self.assertTrue(initial_selection.stats.trimmed)
        self.assertEqual(initial_selection.stats.reserved_output, 384)
        self.assertEqual(initial_selection.stats.safety_margin, 200)
        self.assertLessEqual(
            initial_selection.stats.estimated_input_tokens,
            initial_selection.stats.maximum_input_budget,
        )

        request = await agent._prepare_context_with_summary(
            full_request,
            system_prompt,
            protected_start=len(agent._memory),
        )
        final_selection = agent._select_context(
            request,
            system_prompt,
            protected_start=len(request) - 1,
        )
        # Final request never exceeds the input budget
        self.assertLessEqual(
            final_selection.stats.estimated_input_tokens,
            final_selection.stats.maximum_input_budget,
        )
        # Summary was produced and is included in the final request
        self.assertTrue(agent._summary_state.text)
        self.assertGreater(agent._summary_state.summarized_through, 0)
        self.assertTrue(any("diringkas" in m["content"] for m in request))
        # Recent turn is intact and history is compact after summarization
        self.assertEqual(request[-1], current)
        self.assertIn(message("user", "pesan terbaru pendek"), request)
        self.assertLess(len(request), len(agent._memory))
        # Original transcript is never mutated
        self.assertEqual(get_history(self.conf_uid, history_uid), transcript_before)

    def test_invalid_remembered_history_uid_falls_back_safely(self):
        history_uid = self.create_history([message("user", "halo")])
        agent = self.make_agent(history_uid)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")

        agent.set_memory_from_history(self.conf_uid, "nonexistent-history-uid")

        # Transcript and summary are per conversation (empty here), but the
        # character-level relationship is global and survives an invalid uid.
        self.assertEqual(agent._memory, [])
        self.assertEqual(agent.relationship_status, "dating")
        self.assertEqual(agent._summary_state.text, "")

    async def test_invalid_relationship_reset_is_controlled(self):
        history_uid = self.create_history([])
        agent = self.make_agent(history_uid)
        handler = WebSocketHandler(default_context_cache=None)
        handler.client_contexts["client"] = SimpleNamespace(
            history_uid=None,
            agent_engine=agent,
        )
        websocket = _FakeWebSocket()

        await handler._handle_reset_relationship(websocket, "client", {})

        self.assertEqual(websocket.messages[-1]["type"], "relationship-reset")
        self.assertFalse(websocket.messages[-1]["success"])


if __name__ == "__main__":
    unittest.main()
