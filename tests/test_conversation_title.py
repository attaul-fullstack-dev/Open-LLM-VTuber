"""Automatic conversation titles — gate, generation, persistence, safety.

All storage uses a per-test temp directory and a dedicated conf_uid, so
nothing here can read or write real character data or production history.
"""

import asyncio
import os
import tempfile
import unittest

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    get_history_list,
    get_metadata,
    store_message,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.conversation_title import (
    build_title_prompt,
    has_stored_title,
    sanitize_title,
    should_generate_title,
)

JKT = "Asia/Jakarta"
CONF = "titlechar"


class _TitleLLM:
    model = "title-test"
    max_tokens = 64

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_system = None

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls += 1
        self.last_system = system
        yield self.replies[min(self.calls - 1, len(self.replies) - 1)]


class _FailingLLM(_TitleLLM):
    def __init__(self):
        super().__init__(["unused"])
        self.calls = 0

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls += 1
        raise RuntimeError("provider down")
        yield "unreachable"  # pragma: no cover


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


def make_agent(conf_uid, history_uid, llm):
    agent = BasicMemoryAgent(
        llm=llm,
        system="persona",
        live2d_model=_FakeLive2D(),
        tts_preprocessor_config=TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
        ),
    )
    agent._character_conf_uid = conf_uid
    agent.set_memory_from_history(conf_uid, history_uid, user_timezone=JKT)
    return agent


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TitleBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        for folder in ("chat_history", "character_state", "episodic"):
            os.makedirs(folder, exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def say(self, history_uid, user, assistant=None):
        store_message(CONF, history_uid, "human", user)
        if assistant is not None:
            store_message(CONF, history_uid, "ai", assistant)

    def history_messages(self, history_uid):
        return [m for m in get_history(CONF, history_uid) if m["role"] != "metadata"]


class GateTest(TitleBase):
    def test_new_conversation_without_meaningful_context_stays_untitled(self):
        history_uid = create_new_history(CONF)
        self.say(history_uid, "halo")
        self.assertFalse(
            should_generate_title(
                get_metadata(CONF, history_uid),
                self.history_messages(history_uid),
            )
        )

    def test_single_greeting_exchange_never_titles(self):
        history_uid = create_new_history(CONF)
        self.say(history_uid, "hai mili", "Hai juga!")
        self.assertFalse(
            should_generate_title(
                get_metadata(CONF, history_uid),
                self.history_messages(history_uid),
            )
        )

    def test_meaningful_exchange_passes_the_gate(self):
        history_uid = create_new_history(CONF)
        self.say(history_uid, "halo", "Hai!")
        self.say(
            history_uid,
            "gimana cara setting webhook biar tidak timeout?",
            "Coba naikkan timeout ke 30 detik.",
        )
        self.assertTrue(
            should_generate_title(
                get_metadata(CONF, history_uid),
                self.history_messages(history_uid),
            )
        )

    def test_existing_title_never_regenerates(self):
        history_uid = create_new_history(CONF)
        self.say(history_uid, "halo", "Hai!")
        self.say(history_uid, "bahas websocket yuk", "Siap.")
        metadata = dict(get_metadata(CONF, history_uid))
        metadata["title"] = "Manual Judul"
        self.assertFalse(
            should_generate_title(metadata, self.history_messages(history_uid))
        )


class SanitizeTest(unittest.TestCase):
    def test_plain_title_survives(self):
        self.assertEqual(sanitize_title("Redesign UI Mili"), "Redesign UI Mili")

    def test_quotes_prefix_and_markdown_stripped(self):
        self.assertEqual(
            sanitize_title('**Judul: "Debugging WebSocket Mili"**'),
            "Debugging WebSocket Mili",
        )

    def test_long_reply_truncated_to_seven_words(self):
        self.assertEqual(
            sanitize_title("Belajar Bahasa Inggris Dengan Mili Setiap Hari Pagi Ini"),
            "Belajar Bahasa Inggris Dengan Mili Setiap Hari",
        )

    def test_malformed_reply_falls_back_to_empty(self):
        for bad in ("", "!!", "...", "a", '"', "Judul:"):
            with self.subTest(bad=bad):
                self.assertEqual(sanitize_title(bad), "")

    def test_prompt_is_bounded(self):
        messages = [
            {"role": "human" if i % 2 == 0 else "ai", "content": "x" * 2000}
            for i in range(40)
        ]
        prompt = build_title_prompt(messages)
        self.assertLess(len(prompt), 8 * 320)
        self.assertIn("User:", prompt)
        self.assertIn("Mili:", prompt)


class GenerationTest(TitleBase):
    def test_meaningful_conversation_gets_a_title(self):
        history_uid = create_new_history(CONF)
        self.say(history_uid, "halo", "Hai!")
        self.say(
            history_uid,
            "gimana cara setting webhook biar tidak timeout?",
            "Coba naikkan timeout ke 30 detik.",
        )
        llm = _TitleLLM(["Setting Webhook Timeout"])
        agent = make_agent(CONF, history_uid, llm)
        title = run(agent.ensure_conversation_title())
        self.assertEqual(title, "Setting Webhook Timeout")
        self.assertEqual(llm.calls, 1)

    def test_title_persisted_in_history_metadata(self):
        history_uid = create_new_history(CONF)
        self.say(history_uid, "halo", "Hai!")
        self.say(history_uid, "belajar bahasa inggris yuk", "Siap, mulai!")
        agent = make_agent(CONF, history_uid, _TitleLLM(["Belajar Bahasa Inggris"]))
        title = run(agent.ensure_conversation_title())
        self.assertEqual(get_metadata(CONF, history_uid).get("title"), title)

    def test_reload_keeps_title_without_regenerating(self):
        history_uid = create_new_history(CONF)
        self.say(history_uid, "halo", "Hai!")
        self.say(history_uid, "belajar bahasa inggris yuk", "Siap, mulai!")
        llm = _TitleLLM(["Belajar Bahasa Inggris"])
        first = make_agent(CONF, history_uid, llm)
        run(first.ensure_conversation_title())
        self.assertEqual(llm.calls, 1)
        # Simulate reload / backend restart: a fresh agent on the same files.
        second = make_agent(CONF, history_uid, llm)
        self.assertEqual(
            run(second.ensure_conversation_title()), "Belajar Bahasa Inggris"
        )
        self.assertEqual(llm.calls, 1, "must not call the LLM twice")

    def test_greeting_only_conversation_never_calls_llm(self):
        history_uid = create_new_history(CONF)
        self.say(history_uid, "halo", "Hai juga!")
        llm = _TitleLLM(["Should Never Be Used"])
        agent = make_agent(CONF, history_uid, llm)
        self.assertEqual(run(agent.ensure_conversation_title()), "")
        self.assertEqual(llm.calls, 0)
        self.assertFalse(has_stored_title(get_metadata(CONF, history_uid)))

    def test_llm_failure_keeps_chat_untitled(self):
        history_uid = create_new_history(CONF)
        self.say(history_uid, "halo", "Hai!")
        self.say(history_uid, "bahas websocket yuk", "Siap.")
        agent = make_agent(CONF, history_uid, _FailingLLM())
        self.assertEqual(run(agent.ensure_conversation_title()), "")
        self.assertFalse(has_stored_title(get_metadata(CONF, history_uid)))

    def test_malformed_llm_reply_keeps_untitled(self):
        history_uid = create_new_history(CONF)
        self.say(history_uid, "halo", "Hai!")
        self.say(history_uid, "bahas websocket yuk", "Siap.")
        agent = make_agent(CONF, history_uid, _TitleLLM(["..."]))
        self.assertEqual(run(agent.ensure_conversation_title()), "")
        self.assertFalse(has_stored_title(get_metadata(CONF, history_uid)))

    def test_legacy_history_without_title_key_is_compatible(self):
        history_uid = create_new_history(CONF)
        metadata = get_metadata(CONF, history_uid)
        self.assertNotIn("title", metadata)
        self.assertFalse(has_stored_title(metadata))
        # ...and a legacy untitled session still flows into history-list.
        self.say(history_uid, "halo", "Hai!")
        self.say(history_uid, "bahas websocket yuk", "Siap.")
        listed = get_history_list(CONF)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["title"], "")

    def test_titled_session_appears_in_history_list(self):
        history_uid = create_new_history(CONF)
        self.say(history_uid, "halo", "Hai!")
        self.say(history_uid, "belajar bahasa inggris yuk", "Siap, mulai!")
        agent = make_agent(CONF, history_uid, _TitleLLM(["Belajar Bahasa Inggris"]))
        run(agent.ensure_conversation_title())
        listed = get_history_list(CONF)
        self.assertEqual(listed[0]["title"], "Belajar Bahasa Inggris")


if __name__ == "__main__":
    unittest.main()
