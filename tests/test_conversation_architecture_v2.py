import copy
import json
import os
import tempfile
from types import SimpleNamespace
import unittest

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.conversation_summary import SUMMARY_SYSTEM_PROMPT
from src.open_llm_vtuber.character_state import (
    CHARACTER_MEMORY_MAX_TOKENS,
    build_character_memory_context,
    load_character_state,
)
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    get_history_list,
    get_metadata,
    store_message,
    update_metadate,
)
from src.open_llm_vtuber.agent.context_window import estimate_tokens
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


def message(role: str, content: str):
    return {"role": role, "content": content}


class _V2FakeLLM:
    model = "v2-test"
    max_tokens = 100

    def __init__(self):
        self.summary_calls = 0
        self.summary_prompts = []
        self.fail_next_summary = False
        self.chat_system_prompts = []

    async def chat_completion(self, messages, system=None, tools=None):
        if system == SUMMARY_SYSTEM_PROMPT:
            self.summary_calls += 1
            self.summary_prompts.append(
                messages[0]["content"] if messages else ""
            )
            if self.fail_next_summary:
                self.fail_next_summary = False
                raise RuntimeError("summarizer boom")
            yield "Ringkasan fakta lama yang masih relevan."
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


class ConversationArchitectureV2Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temporary_directory = tempfile.TemporaryDirectory()
        os.chdir(self._temporary_directory.name)
        self.conf_uid = "mili-v2"

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
        llm = llm or _V2FakeLLM()
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
            context_window_override=context_window,
            context_safety_margin=100,
            summary_target_tokens=80,
            summary_max_tokens=100,
            summary_min_new_messages=2,
        )
        agent.set_memory_from_history(self.conf_uid, history_uid)
        return agent

    # --- Chat isolation -------------------------------------------------

    def test_new_chat_isolated_transcript_and_summary(self):
        history_a = self.create_history(
            [message("user", "fakta A"), message("assistant", "ingat A")]
        )
        history_b = self.create_history()
        agent_b = self.make_agent(history_b)

        self.assertEqual(agent_b._memory, [])
        self.assertEqual(agent_b._summary_state.text, "")
        self.assertEqual(
            [m["content"] for m in get_history(self.conf_uid, history_a)],
            ["fakta A", "ingat A"],
        )
        self.assertEqual(get_history(self.conf_uid, history_b), [])

    def test_new_chat_does_not_leak_summary_or_recent_messages(self):
        history_a = self.create_history(
            [message("user", "kita bahas HTML"), message("assistant", "oke HTML")]
        )
        history_b = self.create_history(
            [message("user", "kita bahas game"), message("assistant", "oke game")]
        )
        agent = self.make_agent(history_a)

        for history_uid, topic in (
            (history_a, "HTML"),
            (history_b, "game"),
            (history_a, "HTML"),
            (history_b, "game"),
        ):
            agent.set_memory_from_history(self.conf_uid, history_uid)
            self.assertTrue(
                any(topic in m["content"] for m in agent._memory)
            )
            self.assertFalse(
                any(
                    ("HTML" in m["content"] and topic == "game")
                    or ("game" in m["content"] and topic == "HTML")
                    for m in agent._memory
                )
            )

    # --- Global relationship --------------------------------------------

    def test_relationship_is_global_across_new_chats(self):
        history_a = self.create_history()
        agent = self.make_agent(history_a)
        self.assertTrue(
            agent.set_relationship_status("dating", trigger="synthetic_test_event")
        )

        history_b = self.create_history()
        agent_b = self.make_agent(history_b)
        self.assertEqual(agent_b.relationship_status, "dating")
        self.assertEqual(agent_b._memory, [])
        self.assertEqual(agent_b._summary_state.text, "")
        self.assertEqual(
            load_character_state(self.conf_uid).relationship_status, "dating"
        )

    def test_relationship_reset_is_global(self):
        history_a = self.create_history()
        history_b = self.create_history()
        agent = self.make_agent(history_a)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")
        agent.set_memory_from_history(self.conf_uid, history_b)
        self.assertEqual(agent.relationship_status, "dating")

        self.assertTrue(agent.reset_relationship())
        agent.set_memory_from_history(self.conf_uid, history_a)
        self.assertEqual(agent.relationship_status, "stranger")
        self.assertEqual(
            load_character_state(self.conf_uid).relationship_status, "stranger"
        )

    def test_delete_conversation_preserves_global_relationship(self):
        history_a = self.create_history()
        agent = self.make_agent(history_a)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")
        agent.add_character_memory("user suka ramen", explicit=True)

        os.remove(os.path.join("chat_history", self.conf_uid, f"{history_a}.json"))

        self.assertEqual(get_metadata(self.conf_uid, history_a), {})
        state = load_character_state(self.conf_uid)
        self.assertEqual(state.relationship_status, "dating")
        self.assertEqual(len(state.memories), 1)

    # --- Character memory -----------------------------------------------

    def test_character_memory_cross_chat(self):
        history_a = self.create_history()
        agent_a = self.make_agent(history_a)
        agent_a.observe_character_events(
            "Ingat ya, makanan favoritku ramen.",
            "Oke, aku catat.",
        )
        self.assertEqual(len(agent_a.list_character_memories()), 1)

        history_b = self.create_history()
        agent_b = self.make_agent(history_b)
        self.assertEqual(agent_b._memory, [])
        system_prompt = agent_b._relationship_system_prompt(agent_b._system)
        self.assertIn("makanan favoritku ramen", system_prompt)

    def test_remember_forget_and_conservative_defaults(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)

        self.assertTrue(
            agent.observe_character_events(
                "Ingat ya, makanan favoritku ramen.", "Oke."
            )
        )
        self.assertEqual(len(agent.list_character_memories()), 1)

        # Explicit forget via chat.
        self.assertTrue(
            agent.observe_character_events(
                "Lupakan kalau makanan favoritku ramen.", "Oke, sudah kulupakan."
            )
        )
        self.assertEqual(agent.list_character_memories(), [])

        # Ordinary small talk must never become permanent memory.
        agent.observe_character_events("Aku lagi makan mie.", "Enak dong.")
        agent.observe_character_events("Aku ngantuk.", "Cepet tidur.")
        agent.observe_character_events("Hari ini hujan.", "Bawa payung.")
        agent.observe_character_events("wkwk", "Kenapa sih.")
        agent.observe_character_events("Makasih.", "Sama-sama.")
        self.assertEqual(agent.list_character_memories(), [])

        # Questions starting with "Ingat" are not memory requests.
        agent.observe_character_events("Ingat nggak kita bahas tadi?", "HTML tadi.")
        self.assertEqual(agent.list_character_memories(), [])

    def test_memory_budget_is_bounded(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)
        for index in range(30):
            agent.add_character_memory(
                f"fakta penting jangka panjang nomor {index} "
                + "yang harus selalu diingat oleh Mili",
                explicit=True,
            )
        block = build_character_memory_context(agent._character_state)
        self.assertLessEqual(estimate_tokens(block), CHARACTER_MEMORY_MAX_TOKENS)

    def test_character_memory_reset_keeps_relationship(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")
        agent.add_character_memory("user suka ramen", explicit=True)

        self.assertTrue(agent.reset_character_memory())
        self.assertEqual(agent.list_character_memories(), [])
        self.assertEqual(agent.relationship_status, "dating")
        self.assertEqual(
            load_character_state(self.conf_uid).relationship_status, "dating"
        )

    def test_reset_character_state_resets_both(self):
        history_uid = self.create_history()
        transcript_before = copy.deepcopy(get_history(self.conf_uid, history_uid))
        agent = self.make_agent(history_uid)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")
        agent.add_character_memory("user suka ramen", explicit=True)

        self.assertTrue(agent.reset_character_state())
        self.assertEqual(agent.relationship_status, "stranger")
        self.assertEqual(agent.list_character_memories(), [])
        state = load_character_state(self.conf_uid)
        self.assertEqual(state.relationship_status, "stranger")
        self.assertEqual(state.memories, [])
        # Transcript untouched.
        self.assertEqual(get_history(self.conf_uid, history_uid), transcript_before)

    # --- Manual compact --------------------------------------------------

    def _long_history(self, pairs: int, prefix: str):
        stored = []
        for index in range(pairs):
            stored.append(
                message("user", f"{prefix} lama nomor {index} " + "isi" * 120)
            )
            stored.append(
                message("assistant", f"{prefix} jawaban nomor {index} " + "isi" * 120)
            )
        return stored

    async def test_manual_compact_updates_summary_and_preserves_transcript(self):
        stored = self._long_history(6, "pesan")
        history_uid = self.create_history(stored)
        transcript_before = copy.deepcopy(get_history(self.conf_uid, history_uid))
        llm = _V2FakeLLM()
        agent = self.make_agent(history_uid, llm=llm)

        success, error = await agent.compact_conversation()

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertEqual(llm.summary_calls, 1)
        self.assertEqual(
            agent._summary_state.summarized_through, len(agent._memory)
        )
        self.assertTrue(agent._summary_state.text)
        self.assertEqual(get_history(self.conf_uid, history_uid), transcript_before)

    async def test_auto_compact_continues_incrementally_after_manual_compact(self):
        stored = self._long_history(6, "pesan")
        history_uid = self.create_history(stored)
        llm = _V2FakeLLM()
        agent = self.make_agent(history_uid, llm=llm)
        self.assertTrue((await agent.compact_conversation())[0])
        first_through = agent._summary_state.summarized_through
        self.assertGreater(first_through, 0)
        self.assertIn("pesan lama nomor 0", llm.summary_prompts[0])

        # Conversation grows a lot after the manual compact.
        for item in self._long_history(10, "pesanbaru"):
            store_message(
                self.conf_uid,
                history_uid,
                "human" if item["role"] == "user" else "ai",
                item["content"],
            )
        agent.set_memory_from_history(self.conf_uid, history_uid)

        current = message("user", "pesan terbaruku apa?")
        system_prompt = agent._relationship_system_prompt(agent._system)
        request = await agent._prepare_context_with_summary(
            [*copy.deepcopy(agent._memory), current],
            system_prompt,
            protected_start=len(agent._memory),
        )

        # Auto summary ran again, starting strictly after the manual boundary.
        self.assertEqual(llm.summary_calls, 2)
        self.assertGreater(
            agent._summary_state.summarized_through, first_through
        )
        # Old messages are never re-summarized.
        self.assertNotIn("pesan lama nomor 0", llm.summary_prompts[1])
        self.assertIn("pesanbaru lama nomor 0", llm.summary_prompts[1])
        self.assertEqual(request[-1], current)

    async def test_manual_compact_failure_is_controlled(self):
        stored = self._long_history(4, "pesan")
        history_uid = self.create_history(stored)
        transcript_before = copy.deepcopy(get_history(self.conf_uid, history_uid))
        llm = _V2FakeLLM()
        agent = self.make_agent(history_uid, llm=llm)
        summary_before = agent._summary_state.text
        through_before = agent._summary_state.summarized_through
        llm.fail_next_summary = True

        success, error = await agent.compact_conversation()

        self.assertFalse(success)
        self.assertIsNotNone(error)
        self.assertEqual(agent._summary_state.text, summary_before)
        self.assertEqual(agent._summary_state.summarized_through, through_before)
        self.assertEqual(get_history(self.conf_uid, history_uid), transcript_before)

        # Chat stays usable after the failed compact.
        current = message("user", "halo")
        system_prompt = agent._relationship_system_prompt(agent._system)
        request = await agent._prepare_context_with_summary(
            [*copy.deepcopy(agent._memory), current],
            system_prompt,
            protected_start=len(agent._memory),
        )
        self.assertEqual(request[-1], current)

    # --- Rename conversation ---------------------------------------------

    def test_rename_conversation_persistence(self):
        history_uid = self.create_history(
            [message("user", "isi chat"), message("assistant", "jawaban")]
        )
        self.assertTrue(
            update_metadate(self.conf_uid, history_uid, {"title": "Belajar HTML"})
        )

        listed = get_history_list(self.conf_uid)
        self.assertEqual(listed[0]["title"], "Belajar HTML")

        # Restart: metadata is the source of truth for titles.
        reloaded = get_history_list(self.conf_uid)
        self.assertEqual(reloaded[0]["title"], "Belajar HTML")
        self.assertEqual(
            get_metadata(self.conf_uid, history_uid)["title"], "Belajar HTML"
        )
        # Renaming must not touch transcript/summary/relationship/memory.
        self.assertEqual(
            [m["content"] for m in get_history(self.conf_uid, history_uid)],
            ["isi chat", "jawaban"],
        )

    async def test_websocket_compact_rename_and_memory_handlers(self):
        history_uid = self.create_history(
            [message("user", "isi " * 300), message("assistant", "jawab " * 300)]
        )
        agent = self.make_agent(history_uid)
        handler = WebSocketHandler(default_context_cache=None)
        handler.client_contexts["client"] = SimpleNamespace(
            history_uid=history_uid,
            agent_engine=agent,
            character_config=SimpleNamespace(conf_uid=self.conf_uid),
        )
        websocket = _FakeWebSocket()

        await handler._handle_compact_conversation(websocket, "client", {})
        self.assertEqual(websocket.messages[-1]["type"], "compact-result")
        self.assertTrue(websocket.messages[-1]["success"])

        await handler._handle_rename_history(
            websocket, "client", {"history_uid": history_uid, "title": "Belajar HTML"}
        )
        self.assertEqual(websocket.messages[-1]["type"], "history-renamed")
        self.assertTrue(websocket.messages[-1]["success"])
        self.assertEqual(
            get_metadata(self.conf_uid, history_uid)["title"], "Belajar HTML"
        )

        agent.add_character_memory("user suka ramen", explicit=True)
        await handler._handle_fetch_character_memory(websocket, "client", {})
        self.assertEqual(websocket.messages[-1]["type"], "character-memory")
        self.assertEqual(len(websocket.messages[-1]["memories"]), 1)

        await handler._handle_delete_character_memory(
            websocket, "client", {"text": "user suka ramen"}
        )
        self.assertEqual(websocket.messages[-1]["type"], "character-memory-deleted")
        self.assertTrue(websocket.messages[-1]["success"])
        self.assertEqual(agent.list_character_memories(), [])

        agent.add_character_memory("fakta satu", explicit=True)
        agent.add_character_memory("fakta dua", explicit=True)
        await handler._handle_reset_character_memory(websocket, "client", {})
        self.assertEqual(websocket.messages[-1]["type"], "character-memory-reset")
        self.assertTrue(websocket.messages[-1]["success"])
        self.assertEqual(agent.list_character_memories(), [])

    # --- Migration --------------------------------------------------------

    def test_legacy_relationship_migrates_to_character_level(self):
        history_uid = self.create_history()
        update_metadate(
            self.conf_uid,
            history_uid,
            {
                "relationship_status": "dating",
                "relationship_updated_at": "2026-01-01T00:00:00+00:00",
                "relationship_reason": "explicit_relationship_event",
            },
        )
        history_uid_b = self.create_history()

        agent = self.make_agent(history_uid)
        self.assertEqual(agent.relationship_status, "dating")
        state = load_character_state(self.conf_uid)
        self.assertEqual(state.relationship_status, "dating")
        self.assertEqual(state.relationship_reason, "explicit_relationship_event")

        # Migration runs once; later loads do not rescan conversations.
        agent.set_memory_from_history(self.conf_uid, history_uid_b)
        self.assertEqual(agent.relationship_status, "dating")
        self.assertTrue(load_character_state(self.conf_uid).relationship_migrated)

    def test_stranger_conversations_never_migrate_into_a_guess(self):
        history_uid = self.create_history()
        update_metadate(
            self.conf_uid,
            history_uid,
            {"relationship_status": "stranger", "relationship_reason": "default"},
        )

        agent = self.make_agent(history_uid)
        self.assertEqual(agent.relationship_status, "stranger")
        self.assertEqual(
            load_character_state(self.conf_uid).relationship_status, "stranger"
        )

    # --- Context composition & budget ------------------------------------

    async def test_context_composition_order_and_budget(self):
        history_a = self.create_history(
            [message("user", "kita bahas HTML"), message("assistant", "oke HTML")]
        )
        agent = self.make_agent(history_a, context_window=3000)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")
        agent.add_character_memory("user suka ramen", explicit=True)

        current = message("user", "kamu masih ingat makanan favoritku?")
        system_prompt = agent._relationship_system_prompt(agent._system)
        request = await agent._prepare_context_with_summary(
            [*copy.deepcopy(agent._memory), current],
            system_prompt,
            protected_start=len(agent._memory),
        )
        selection = agent._select_context(
            request,
            system_prompt,
            protected_start=len(request) - 1,
        )

        # SYSTEM + PERSONA -> RELATIONSHIP -> MEMORY (all inside the prompt).
        self.assertTrue(system_prompt.startswith(agent._system))
        self.assertIn("Current state: dating", system_prompt)
        self.assertIn("Known long-term context", system_prompt)
        self.assertLess(
            system_prompt.index("Current state: dating"),
            system_prompt.index("Known long-term context"),
        )
        # Current user message is last; recent history stays before it.
        self.assertEqual(request[-1], current)
        self.assertIn(message("user", "kita bahas HTML"), request)
        # No other chat's content leaks in.
        self.assertFalse(any("game" in m["content"] for m in request))
        self.assertLessEqual(
            selection.stats.estimated_input_tokens,
            selection.stats.maximum_input_budget,
        )

    async def test_restart_persistence_of_character_state(self):
        history_uid = self.create_history()
        agent = self.make_agent(history_uid)
        agent.set_relationship_status("dating", trigger="synthetic_test_event")
        agent.add_character_memory("user suka ramen", explicit=True)

        # Simulated backend restart: brand-new agent loads from disk.
        restarted = self.make_agent(history_uid)
        self.assertEqual(restarted.relationship_status, "dating")
        self.assertEqual(len(restarted.list_character_memories()), 1)
        self.assertEqual(
            restarted.list_character_memories()[0]["text"], "user suka ramen"
        )
        system_prompt = restarted._relationship_system_prompt(restarted._system)
        self.assertIn("user suka ramen", system_prompt)
        self.assertIn("Current state: dating", system_prompt)


class BackendCompletionSignalDedupTests(unittest.TestCase):
    """One assistant turn → ONE authoritative `backend-synth-complete`.

    Live proof of the duplicate: the frontend received backend-synth-complete
    TWICE per turn (once from the conversation module itself, once from
    finalize_conversation_turn), so it released the contextual face twice and
    re-sent frontend-playback-complete repeatedly. The single authoritative
    emitter is now finalize_conversation_turn; single_conversation and
    group_conversation only keep an explanatory comment. This source-contract
    guard keeps the dedup from regressing.
    """

    ROOT = os.path.join(os.path.dirname(__file__), "..")

    def _read(self, relative_path):
        with open(
            os.path.join(self.ROOT, relative_path), encoding="utf-8"
        ) as f:
            return f.read()

    def test_single_conversation_no_longer_emits_completion(self):
        source = self._read(
            "src/open_llm_vtuber/conversations/single_conversation.py"
        )
        self.assertNotIn(
            'json.dumps({"type": "backend-synth-complete"})', source
        )

    def test_group_conversation_no_longer_emits_completion(self):
        source = self._read(
            "src/open_llm_vtuber/conversations/group_conversation.py"
        )
        self.assertNotIn(
            'json.dumps({"type": "backend-synth-complete"})', source
        )

    def test_finalize_is_the_only_emitter(self):
        source = self._read(
            "src/open_llm_vtuber/conversations/conversation_utils.py"
        )
        self.assertEqual(
            source.count('json.dumps({"type": "backend-synth-complete"})'),
            1,
            "finalize_conversation_turn must be the ONLY backend-synth-complete "
            "emitter",
        )


if __name__ == "__main__":
    unittest.main()
