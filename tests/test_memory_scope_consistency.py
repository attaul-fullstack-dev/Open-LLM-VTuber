"""Memory scope consistency: what Mili recalls vs what the UI lists.

Reproduces the reported "Mili remembers it but the Memory Manager does not
show it" inconsistency and pins the proven mechanism.

Mili's recall draws from SEVERAL persistent stores (character memories,
episodic events, rolling summaries, interaction preferences, future
intentions, world state), but the Memory Manager UI lists exactly ONE of
them (character_state.memories). A fact recalled from any other store --
most commonly an episodic event -- is therefore invisible in the dialog BY
CONSTRUCTION. Earlier dummy tests only exercised same-store paths, which is
why they passed without closing the bug.

These tests drive the GENUINE production functions (per-turn
``observe_character_events``, the episodic persistence funnel, the real
prompt-block builders, the real UI-list function) in an isolated working
directory. No production memory is ever used.

Scope contract pinned here:
- every character memory (explicit, automatic, legacy-shaped) is persisted,
  recalled through the character block, AND listed;
- an episodic-only fact is persisted, recalled through the episodic block,
  and ABSENT from the UI list by design (it is an event, not a long-term
  fact; merging the stores would destroy deletion/id semantics).
"""

import os
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.character_state import (
    build_character_memory_context,
    load_character_state,
)
from src.open_llm_vtuber.chat_history_manager import create_new_history
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber import episodic_memory as em

CONF = "scope-matrix"
JKT = "Asia/Jakarta"


class FakeLLM:
    model = "scope-matrix-llm"
    max_tokens = 64

    async def chat_completion(self, *args, **kwargs):
        if False:
            yield None


class MemoryScopeMatrixTest(unittest.TestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self.agent = BasicMemoryAgent(
            llm=FakeLLM(),
            system="persona",
            live2d_model=SimpleNamespace(extract_emotion=lambda text: []),
            tts_preprocessor_config=TTSPreprocessorConfig(
                remove_special_char=True,
                translator_config={
                    "translate_audio": False,
                    "translate_provider": "deeplx",
                },
            ),
        )
        history_uid = create_new_history(CONF)
        self.agent.set_memory_from_history(CONF, history_uid)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _disk_character_texts(self):
        return [m["text"] for m in load_character_state(CONF).memories]

    def _disk_episodic_texts(self):
        return [e["event_text"] for e in em.load_episodic_events(CONF)]

    def _ui_list_texts(self):
        return [m["text"] for m in self.agent.list_character_memories()]

    def test_explicit_command_fact_is_persisted_recalled_and_listed(self):
        self.assertTrue(
            self.agent.observe_character_events(
                "Ingat ya, SCOPE sumur biru.", "Baik, dicatat."
            )
        )
        self.assertIn("SCOPE sumur biru", self._disk_character_texts())
        block = build_character_memory_context(load_character_state(CONF), tz=JKT)
        self.assertIn("SCOPE sumur biru", block)
        self.assertIn("SCOPE sumur biru", self._ui_list_texts())

    def test_auto_extracted_fact_is_persisted_recalled_and_listed(self):
        self.assertTrue(
            self.agent.observe_character_events(
                "Aku selalu SCOPE menyiram anggrek setiap pagi.", "Wah rajin."
            )
        )
        stored = self._disk_character_texts()
        self.assertEqual(len(stored), 1)
        self.assertIn("SCOPE", stored[0])
        block = build_character_memory_context(load_character_state(CONF), tz=JKT)
        self.assertIn("SCOPE", block)
        self.assertIn(stored[0], self._ui_list_texts())

    def test_episodic_fact_is_recalled_but_outside_the_ui_scope(self):
        """THE reproduced inconsistency, pinned as a defined boundary.

        An event stored through the normal episodic funnel is genuinely
        recalled (the episodic prompt block carries it), yet the Memory
        Manager list -- which covers character facts only -- does not
        contain it. This is scope, not data loss: the event remains
        governed by the episodic store (own retrieval, own deletion API).
        """
        stored = em.append_episodic_event(
            CONF,
            {
                "event_text": "SCOPE festival layang kemarin sore",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "session_uid": "sess-scope",
                "source": "conversation",
            },
        )
        self.assertIsNotNone(stored)
        self.assertIn("SCOPE festival layang kemarin sore", self._disk_episodic_texts())

        self.agent._episodic_query = "SCOPE festival layang"
        episodic_block = self.agent._episodic_context_for_prompt()
        self.assertIn("SCOPE festival layang", episodic_block)

        ui_texts = self._ui_list_texts()
        self.assertNotIn("SCOPE festival layang kemarin sore", ui_texts)
        self.assertEqual(
            self._disk_character_texts(),
            ui_texts,
            "the UI list must equal the character store exactly: no more, no less",
        )

    def test_character_store_and_ui_list_stay_identical(self):
        self.agent.observe_character_events("Ingat ya, SCOPE sumur biru.", "Baik.")
        self.agent.observe_character_events(
            "Aku selalu SCOPE menyiram anggrek setiap pagi.", "OK."
        )
        self.assertEqual(self._disk_character_texts(), self._ui_list_texts())
        # and identically so for a fresh session (restart/resume simulation)
        fresh = object.__new__(BasicMemoryAgent)
        fresh._character_conf_uid = CONF
        fresh._character_state = load_character_state(CONF)
        self.assertEqual(
            [m["text"] for m in fresh.list_character_memories()],
            self._disk_character_texts(),
        )


if __name__ == "__main__":
    unittest.main()
