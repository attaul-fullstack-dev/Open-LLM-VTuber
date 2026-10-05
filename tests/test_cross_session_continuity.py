"""Cross-session continuity of persistent state (SESSION A -> SESSION B).

Proves that a new session inherits character-level relationship state and
long-term memories without any transcript injection, that the previous
session summary still works as narrative context only, and that
session-local data never leaks into character-level state.

No network, no LLM: the agent is built with a fake LLM and the real
persistence functions, inside a temporary directory.
"""

import os
import tempfile
import unittest

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.relationship_context import (
    _STATE_GUIDANCE,
    build_relationship_context,
)
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_metadata,
    store_message,
    update_metadate,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig

JKT = "Asia/Jakarta"
PERSONA = "PERSONA-TEXT-FROM-YAML"


class _FakeLLM:
    async def chat_completion(self, messages, system=None, **kwargs):
        yield "ok"


class _FakeLive2D:
    live2d_model_name = "model"

    def set_emotion(self, emotion, duration=None):
        pass

    def set_motion(self, group, index, priority=1):
        pass


class ContinuityBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        self.conf_uid = "cont_char"

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def agent_for(self, history_uid):
        agent = BasicMemoryAgent(
            llm=_FakeLLM(),
            system=PERSONA,
            live2d_model=_FakeLive2D(),
            tts_preprocessor_config=TTSPreprocessorConfig(
                remove_special_char=True,
                translator_config={
                    "translate_audio": False,
                    "translate_provider": "deeplx",
                },
            ),
        )
        agent.set_memory_from_history(self.conf_uid, history_uid, user_timezone=JKT)
        return agent

    def prompt_of(self, agent):
        return agent._relationship_system_prompt(PERSONA)

    def session_a(self, relationship="dating", memory=None, summary=None):
        """Build Session A, persist state, and drop the in-memory agent."""
        hist = create_new_history(self.conf_uid)
        store_message(self.conf_uid, hist, "human", "baseline turn")
        store_message(self.conf_uid, hist, "ai", "acknowledged")
        agent = self.agent_for(hist)
        if relationship:
            self.assertTrue(
                agent.set_relationship_status(
                    relationship, trigger="continuity_test_event"
                )
            )
        if memory:
            self.assertTrue(agent.add_character_memory(memory, explicit=True))
        if summary:
            update_metadate(
                self.conf_uid,
                hist,
                {
                    "conversation_summary": summary,
                    "summary_updated_at": "2026-09-30T06:56:21+00:00",
                },
            )
        del agent
        return hist


class SessionBInheritsPersistentStateTest(ContinuityBase):
    def test_a_relationship_state_crosses_sessions(self):
        hist_a = self.session_a(relationship="dating")
        hist_b = create_new_history(self.conf_uid)
        self.assertNotEqual(hist_a, hist_b)
        agent_b = self.agent_for(hist_b)
        self.assertEqual(agent_b.relationship_status, "dating")
        prompt = self.prompt_of(agent_b)
        self.assertIn("Current state: dating", prompt)

    def test_a_relationship_every_supported_tier_crosses(self):
        for tier in ("stranger", "familiar", "close", "dating"):
            with self.subTest(tier=tier):
                # Fresh sandbox per tier so character state never bleeds.
                with tempfile.TemporaryDirectory() as sandbox:
                    outer = os.getcwd()
                    os.chdir(sandbox)
                    try:
                        self.conf_uid = f"tier_{tier}"
                        self.session_a(relationship=tier)
                        agent_b = self.agent_for(create_new_history(self.conf_uid))
                        self.assertEqual(agent_b.relationship_status, tier)
                    finally:
                        os.chdir(outer)

    def test_b_long_term_memory_crosses_sessions(self):
        memory = "User stated a durable fact about how they want to be treated."
        self.session_a(relationship="close", memory=memory)
        agent_b = self.agent_for(create_new_history(self.conf_uid))
        self.assertEqual(len(agent_b.list_character_memories()), 1)
        self.assertIn("durable fact", self.prompt_of(agent_b))

    def test_b_relationship_change_after_memory_still_wins(self):
        """The newest relationship state is what a new session reads."""
        hist = self.session_a(relationship="close")
        agent_a = self.agent_for(hist)
        agent_a.set_relationship_status("dating", trigger="continuity_test_event")
        agent_b = self.agent_for(create_new_history(self.conf_uid))
        self.assertEqual(agent_b.relationship_status, "dating")

    def test_c_previous_session_summary_still_injected(self):
        summary = "Both talked at length about their shared routines last week."
        self.session_a(relationship="dating", summary=summary)
        agent_b = self.agent_for(create_new_history(self.conf_uid))
        prompt = self.prompt_of(agent_b)
        self.assertIn("shared routines", prompt)
        # Summary is narrative context, never the state source.
        self.assertIn("Current state: dating", prompt)

    def test_g_new_session_needs_no_full_transcript(self):
        hist_a = self.session_a(relationship="dating", summary="Narrative only.")
        agent_b = self.agent_for(create_new_history(self.conf_uid))
        self.assertEqual(len(agent_b._memory or []), 0)
        self.assertNotIn("baseline turn", self.prompt_of(agent_b))
        self.assertIn("Current state: dating", self.prompt_of(agent_b))
        del hist_a

    def test_d_no_duplicate_memory_across_sessions(self):
        memory = "User stated a durable fact about how they want to be treated."
        hist_a = self.session_a(relationship="dating", memory=memory)
        agent_b = self.agent_for(create_new_history(self.conf_uid))
        agent_c = self.agent_for(create_new_history(self.conf_uid))
        self.assertEqual(len(agent_b.list_character_memories()), 1)
        self.assertEqual(len(agent_c.list_character_memories()), 1)
        self.assertEqual(
            agent_c.list_character_memories(), agent_b.list_character_memories()
        )
        del hist_a

    def test_e_session_local_state_does_not_leak(self):
        """A session's own summary must not become character-level state."""
        hist_a = self.session_a(
            relationship="stranger",
            summary="This session mentioned a passing joke.",
        )
        agent_b = self.agent_for(create_new_history(self.conf_uid))
        self.assertEqual(agent_b.relationship_status, "stranger")
        self.assertEqual(agent_b.list_character_memories(), [])
        del hist_a

    def test_f_restart_reload_keeps_state(self):
        """A fresh agent object (process restart) reads the same state."""
        self.session_a(relationship="dating", memory="Durable preference recorded.")
        first = self.agent_for(create_new_history(self.conf_uid))
        self.assertEqual(first.relationship_status, "dating")
        del first
        second = self.agent_for(create_new_history(self.conf_uid))
        self.assertEqual(second.relationship_status, "dating")
        self.assertEqual(len(second.list_character_memories()), 1)

    def test_reset_still_clears_to_stranger(self):
        hist = self.session_a(relationship="dating")
        agent = self.agent_for(hist)
        self.assertTrue(agent.reset_relationship())
        self.assertEqual(agent.relationship_status, "stranger")
        self.assertEqual(agent.list_character_memories(), [])
        fresh = self.agent_for(create_new_history(self.conf_uid))
        self.assertEqual(fresh.relationship_status, "stranger")


class SessionMetadataTruthfulnessTest(ContinuityBase):
    def test_new_session_metadata_is_not_left_at_stranger(self):
        """Regression: session metadata kept a hardcoded 'stranger' default.

        New history files are created with relationship_status='stranger'
        and were never refreshed, so the session record permanently
        disagreed with the real character-level state.
        """
        hist_a = self.session_a(relationship="dating")
        meta = get_metadata(self.conf_uid, hist_a)
        self.assertEqual(meta.get("relationship_status"), "dating")
        self.assertEqual(meta.get("relationship_reason"), "continuity_test_event")

    def test_metadata_syncs_on_later_update(self):
        hist = create_new_history(self.conf_uid)
        self.assertEqual(
            get_metadata(self.conf_uid, hist).get("relationship_status"), "stranger"
        )
        agent = self.agent_for(hist)
        agent.set_relationship_status("close", trigger="continuity_test_event")
        meta = get_metadata(self.conf_uid, hist)
        self.assertEqual(meta.get("relationship_status"), "close")
        self.assertEqual(meta.get("relationship_reason"), "continuity_test_event")

    def test_metadata_sync_is_fail_soft(self):
        agent = self.agent_for(create_new_history(self.conf_uid))
        agent._character_conf_uid = ""  # no character context
        self.assertFalse(
            agent.set_relationship_status("dating", trigger="continuity_test_event")
        )


class RelationshipGuidanceTest(unittest.TestCase):
    """The guidance must not dictate personality over the persona prompt."""

    def test_no_status_mandates_a_personality(self):
        for status, text in _STATE_GUIDANCE.items():
            with self.subTest(status=status):
                lowered = text.lower()
                for banned in ("tsundere", "personality"):
                    self.assertNotIn(banned, lowered)

    def test_dating_guidance_still_refuses_amnesia(self):
        rendered = build_relationship_context("dating")
        self.assertIn("never happened", rendered)
        self.assertIn("persona", rendered)

    def test_every_status_has_guidance(self):
        for status in ("stranger", "familiar", "close", "dating"):
            self.assertIn(status, _STATE_GUIDANCE)


if __name__ == "__main__":
    unittest.main()
