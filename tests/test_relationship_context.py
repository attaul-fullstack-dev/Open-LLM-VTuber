import copy
import json
import os
import tempfile
from types import SimpleNamespace
import unittest

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.conversation_summary import SUMMARY_SYSTEM_PROMPT
from src.open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource
from src.open_llm_vtuber.agent.relationship_context import (
    build_relationship_context,
    detect_relationship_update,
)
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    delete_history,
    get_history,
    get_metadata,
    store_message,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


def message(role: str, content: str):
    return {"role": role, "content": content}


class _RelationshipFakeLLM:
    model = "relationship-test"
    max_tokens = 100

    def __init__(self):
        self.chat_system_prompts = []
        self.summary_calls = 0

    async def chat_completion(self, messages, system=None, tools=None):
        if system == SUMMARY_SYSTEM_PROMPT:
            self.summary_calls += 1
            yield "User memiliki konteks lama yang masih relevan."
            return
        self.chat_system_prompts.append(system)
        yield "oke."


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class _FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, payload):
        self.messages.append(json.loads(payload))


class RelationshipContinuityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temporary_directory = tempfile.TemporaryDirectory()
        os.chdir(self._temporary_directory.name)
        self.conf_uid = "mili-test"

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temporary_directory.cleanup()

    def create_history(self, messages=None):
        history_uid = create_new_history(self.conf_uid)
        for item in messages or []:
            store_message(
                self.conf_uid,
                history_uid,
                "human" if item["role"] == "user" else "ai",
                item["content"],
            )
        return history_uid

    def make_agent(self, history_uid, llm=None, *, context_window=1200):
        llm = llm or _RelationshipFakeLLM()
        tts_config = TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
        )
        agent = BasicMemoryAgent(
            llm=llm,
            system="persona Mili tetap menjadi sumber gaya bicara",
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

    def test_new_conversation_defaults_to_stranger(self):
        history_uid = self.create_history()
        metadata = get_metadata(self.conf_uid, history_uid)
        agent = self.make_agent(history_uid)

        self.assertEqual(metadata["relationship_status"], "stranger")
        self.assertEqual(agent.relationship_status, "stranger")

    def test_persistence_across_agent_recreation_and_restart_load(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)
        self.assertTrue(
            agent.set_relationship_status("close", trigger="synthetic_test_event")
        )

        recreated = self.make_agent(history_uid)

        self.assertEqual(recreated.relationship_status, "close")
        self.assertEqual(
            get_metadata(self.conf_uid, history_uid)["relationship_status"],
            "close",
        )

    def test_conversation_switching_keeps_states_isolated(self):
        history_a = self.create_history()
        history_b = self.create_history()
        agent = self.make_agent(history_a)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")
        agent.set_memory_from_history(self.conf_uid, history_b)
        agent.set_relationship_status("familiar", trigger="synthetic_test_event")

        agent.set_memory_from_history(self.conf_uid, history_a)
        self.assertEqual(agent.relationship_status, "dating")
        agent.set_memory_from_history(self.conf_uid, history_b)
        self.assertEqual(agent.relationship_status, "familiar")

    def test_explicit_mutual_dating_event_updates_state(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)

        updated = agent.observe_relationship_event(
            "Kamu mau nggak jadi pacar aku?",
            "[shy] ...Iya. Mau. Udah, jangan suruh aku ngulang.",
        )

        self.assertTrue(updated)
        self.assertEqual(agent.relationship_status, "dating")
        metadata = get_metadata(self.conf_uid, history_uid)
        self.assertEqual(metadata["relationship_reason"], "explicit_relationship_event")

    def test_clear_indonesian_dating_variants_require_acceptance(self):
        proposals = (
            "Mau gak jadi pacarku?",
            "Mau ngga jadi pacar aku?",
            "Jadi pacar aku ya?",
            "Kita pacaran yuk.",
        )
        for proposal in proposals:
            with self.subTest(proposal=proposal):
                update = detect_relationship_update(
                    "stranger", proposal, "...Iya. Mau."
                )
                self.assertIsNotNone(update)
                self.assertEqual(update.new_status, "dating")

    def test_rejected_proposal_does_not_set_dating(self):
        self.assertIsNone(
            detect_relationship_update(
                "close", "Mau jadi pacarku?", "Nggak. Aku belum mau."
            )
        )

    def test_third_party_relationship_question_is_not_a_proposal(self):
        self.assertIsNone(
            detect_relationship_update(
                "stranger",
                "Karakter di anime itu pacaran nggak?",
                "Kayaknya belum dijelasin di ceritanya.",
            )
        )

    def test_one_sided_romantic_message_does_not_set_dating(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)

        updated = agent.observe_relationship_event(
            "Aku suka sama kamu.",
            "Ih... tiba-tiba banget.",
        )

        self.assertFalse(updated)
        self.assertEqual(agent.relationship_status, "stranger")

    def test_ordinary_compliment_is_not_a_relationship_event(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)

        updated = agent.observe_relationship_event(
            "Kamu lucu banget.",
            "Apaan sih. Tapi... makasih.",
        )

        self.assertFalse(updated)
        self.assertEqual(agent.relationship_status, "stranger")

    def test_explicit_familiar_and_close_events_are_conservative(self):
        familiar = detect_relationship_update(
            "stranger",
            "Aku balik lagi. Masih ingat aku?",
            "Ingat kok. Datang lagi juga akhirnya.",
        )
        close = detect_relationship_update(
            "familiar",
            "Aku percaya sama kamu dan nyaman cerita sama kamu.",
            "Aku juga. Senang kamu percaya sama aku.",
        )

        self.assertEqual(familiar.new_status, "familiar")
        self.assertEqual(close.new_status, "close")
        self.assertIsNone(
            detect_relationship_update(
                "familiar",
                "Makasih udah dengerin.",
                "Iya, santai aja.",
            )
        )

    async def test_relationship_survives_long_history_and_summary(self):
        stored = [
            message("user", "old " * 700),
            message("assistant", "old answer " * 700),
            message("user", "recent fact"),
            message("assistant", "recent answer"),
        ]
        history_uid = self.create_history(stored)
        agent = self.make_agent(history_uid, context_window=900)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")
        recreated = self.make_agent(
            history_uid,
            llm=_RelationshipFakeLLM(),
            context_window=900,
        )
        messages = [*copy.deepcopy(recreated._memory), message("user", "lanjut")]
        system_prompt = recreated._relationship_system_prompt(recreated._system)

        request = await recreated._prepare_context_with_summary(
            messages,
            system_prompt,
            protected_start=len(messages) - 1,
        )

        self.assertEqual(recreated.relationship_status, "dating")
        self.assertIn("Current state: dating", system_prompt)
        self.assertEqual(request[-1], message("user", "lanjut"))
        self.assertIn(message("user", "recent fact"), request)
        self.assertTrue(recreated._summary_state.text)
        self.assertEqual(len(get_history(self.conf_uid, history_uid)), len(stored))

    async def test_actual_chat_request_injects_short_internal_context(self):
        history_uid = self.create_history()
        llm = _RelationshipFakeLLM()
        agent = self.make_agent(history_uid, llm=llm, context_window=4000)
        agent.set_relationship_status("close", trigger="synthetic_test_event")
        persona_before = agent._system

        outputs = [
            output
            async for output in agent.chat(
                BatchInput(texts=[TextData(TextSource.INPUT, "Halo")])
            )
        ]

        self.assertTrue(outputs)
        self.assertEqual(len(llm.chat_system_prompts), 1)
        sent_system = llm.chat_system_prompts[0]
        self.assertTrue(sent_system.startswith(persona_before))
        self.assertIn("Current state: close", sent_system)
        self.assertIn("Never mention internal state names", sent_system)
        self.assertEqual(agent._system, persona_before)

    def test_persona_stays_identical_across_all_states(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)
        persona = agent._system

        prompts = []
        for status in ("stranger", "familiar", "close", "dating"):
            agent.set_relationship_status(status, trigger="synthetic_test_event")
            prompts.append(agent._relationship_system_prompt(agent._system))

        self.assertTrue(all(prompt.startswith(persona) for prompt in prompts))
        self.assertEqual(agent._system, persona)
        self.assertEqual(len(set(prompts)), 4)

    def test_context_does_not_encourage_internal_label_leakage(self):
        context = build_relationship_context("dating")

        self.assertIn("not user-visible metadata", context)
        self.assertIn("Never mention internal state names", context)
        self.assertIn("answer naturally", context)

    def test_manual_reset_persists_stranger(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")

        self.assertTrue(agent.reset_relationship())

        recreated = self.make_agent(history_uid)
        metadata = get_metadata(self.conf_uid, history_uid)
        self.assertEqual(recreated.relationship_status, "stranger")
        self.assertEqual(metadata["relationship_status"], "stranger")
        self.assertEqual(metadata["relationship_reason"], "manual_reset")

    async def test_websocket_backend_reset_uses_active_conversation(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")
        handler = WebSocketHandler(default_context_cache=None)
        handler.client_contexts["client"] = SimpleNamespace(
            history_uid=history_uid,
            agent_engine=agent,
        )
        websocket = _FakeWebSocket()

        await handler._handle_reset_relationship(websocket, "client", {})

        self.assertEqual(agent.relationship_status, "stranger")
        self.assertEqual(websocket.messages[-1]["type"], "relationship-reset")
        self.assertTrue(websocket.messages[-1]["success"])
        self.assertIn("reset-relationship", handler._message_handlers)

    def test_delete_conversation_removes_relationship_metadata(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")

        self.assertTrue(delete_history(self.conf_uid, history_uid))
        self.assertEqual(get_metadata(self.conf_uid, history_uid), {})


if __name__ == "__main__":
    unittest.main()
