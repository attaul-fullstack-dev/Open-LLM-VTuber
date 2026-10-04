"""Behavior / interaction preferences — detection, persistence, cross-session.

Deterministic: no LLM, no network, no scheduler. Uses the real character
state store and the real agent inside a temporary directory.
"""

import os
import tempfile
import unittest
from datetime import datetime, timezone

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.character_state import (
    load_character_state,
    record_interaction_preference,
    reset_character_state,
    set_character_relationship,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.interaction_preferences import (
    PREFERENCE_STATUS_ACTIVE,
    PREFERENCE_STATUS_SUPERSEDED,
    active_preferences,
    build_interaction_preference_context,
    detect_interaction_preference,
)

JKT = "Asia/Jakarta"
PERSONA = "Mili bisa sedikit galak dan tegas."
UTC = timezone.utc
NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

MUST_CAPTURE = [
    "Mulai sekarang jangan terlalu galak ke aku.",
    "Jangan terlalu formal.",
    "Aku lebih suka kalau kamu ngomong santai.",
    "Kalau aku lagi serius, jangan bercanda.",
    "mulai sekarang boleh panggil aku dile",
    "Aku biasanya suka kamu yang santai aja.",
    "going forward be more casual with me",
]

MUST_NOT_CAPTURE = [
    "Hari ini jangan galak dong.",
    "sekarang saja jangan galak",
    "Kamu masih galak?",
    "halo, apa kabar?",
    "aku lagi belajar python",
    "besok kita ngobrol soal itu ya",
    "kamu suka makan apa?",
    "dari tadi kamu nggak bales",
    "",
    None,
]


class _FakeLLM:
    async def chat_completion(self, messages, system=None, **kwargs):
        yield "ok"


class _FakeLive2D:
    live2d_model_name = "model"

    def set_emotion(self, emotion, duration=None):
        pass

    def set_motion(self, group, index, priority=1):
        pass


class DetectionTest(unittest.TestCase):
    def test_durable_phrases_are_captured(self):
        for text in MUST_CAPTURE:
            with self.subTest(text=text):
                self.assertIsNotNone(detect_interaction_preference(text, now=NOW))

    def test_ordinary_and_temporary_turns_are_not_captured(self):
        for text in MUST_NOT_CAPTURE:
            with self.subTest(text=text):
                self.assertIsNone(detect_interaction_preference(text, now=NOW))

    def test_target_example_is_tone_avoid(self):
        found = detect_interaction_preference(
            "Mulai sekarang jangan terlalu galak ke aku.", now=NOW
        )
        assert found is not None
        self.assertEqual((found.category, found.polarity), ("tone", "avoid"))
        self.assertEqual(found.frame, "durability")

    def test_detection_is_deterministic(self):
        first = detect_interaction_preference("Jangan terlalu formal.", now=NOW)
        second = detect_interaction_preference("Jangan terlalu formal.", now=NOW)
        self.assertEqual(first, second)

    def test_question_is_never_a_preference(self):
        self.assertIsNone(detect_interaction_preference("kamu jangan galak?", now=NOW))

    def test_temporary_request_is_not_promoted(self):
        self.assertIsNone(
            detect_interaction_preference("Hari ini jangan galak dong.", now=NOW)
        )

    def test_long_narration_is_not_a_preference(self):
        self.assertIsNone(
            detect_interaction_preference(
                "jadi tadi aku cerita panjang banget soalafi "
                "mohon jangan galak kalau lagi_rule " * 6,
                now=NOW,
            )
        )

    def test_timestamp_is_utc_aware(self):
        found = detect_interaction_preference("Jangan terlalu formal.", now=NOW)
        assert found is not None
        parsed = datetime.fromisoformat(found.created_at)
        self.assertIsNotNone(parsed.tzinfo)
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        self.conf = "pref_char"

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_stored_entry_has_full_schema(self):
        stored = record_interaction_preference(
            self.conf, "Mulai sekarang jangan terlalu galak ke aku."
        )
        self.assertIsNotNone(stored)
        assert stored is not None
        for field in (
            "id",
            "text",
            "category",
            "polarity",
            "created_at",
            "updated_at",
            "source",
            "status",
        ):
            self.assertIn(field, stored)
        self.assertEqual(stored["status"], PREFERENCE_STATUS_ACTIVE)
        self.assertEqual(stored["source"], "conversation")
        for stamp in (stored["created_at"], stored["updated_at"]):
            parsed = datetime.fromisoformat(stamp)
            self.assertIsNotNone(parsed.tzinfo)
            self.assertEqual(parsed.utcoffset().total_seconds(), 0)

    def test_non_preference_is_not_stored(self):
        self.assertIsNone(record_interaction_preference(self.conf, "halo apa kabar"))
        self.assertEqual(load_character_state(self.conf).interaction_preferences, [])

    def test_ordinary_turn_does_not_create_state(self):
        for text in MUST_NOT_CAPTURE:
            if not text:
                continue
            record_interaction_preference(self.conf, text)
        self.assertEqual(load_character_state(self.conf).interaction_preferences, [])

    def test_new_preference_supersedes_same_category(self):
        first = record_interaction_preference(
            self.conf, "Mulai sekarang jangan terlalu galak ke aku."
        )
        second = record_interaction_preference(
            self.conf, "Mulai sekarang aku justru suka kamu sedikit galak."
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        state = load_character_state(self.conf)
        stored = {item["id"]: item for item in state.interaction_preferences}
        self.assertEqual(stored[first["id"]]["status"], PREFERENCE_STATUS_SUPERSEDED)
        self.assertEqual(stored[first["id"]]["superseded_by"], second["id"])
        self.assertEqual(stored[second["id"]]["status"], PREFERENCE_STATUS_ACTIVE)
        active = active_preferences(state.interaction_preferences)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["id"], second["id"])

    def test_distinct_categories_coexist(self):
        record_interaction_preference(self.conf, "Jangan terlalu formal.")
        record_interaction_preference(
            self.conf, "Kalau aku lagi serius, jangan bercanda."
        )
        active = active_preferences(
            load_character_state(self.conf).interaction_preferences
        )
        self.assertEqual({a["category"] for a in active}, {"formality", "humor"})

    def test_corrupt_rows_are_dropped_on_load(self):
        record_interaction_preference(self.conf, "Jangan terlalu formal.")
        path = os.path.join("character_state", f"{self.conf}.json")
        import json

        data = json.load(open(path))
        data["interaction_preferences"].append({"category": "bogus", "text": "x"})
        data["interaction_preferences"].append("not-a-dict")
        json.dump(data, open(path, "w"), indent=2)
        state = load_character_state(self.conf)
        self.assertEqual(len(state.interaction_preferences), 1)

    def test_legacy_state_without_field_loads(self):
        import json

        os.makedirs("character_state", exist_ok=True)
        json.dump(
            {"relationship_status": "stranger", "memories": []},
            open(os.path.join("character_state", f"{self.conf}.json"), "w"),
        )
        state = load_character_state(self.conf)
        self.assertEqual(state.interaction_preferences, [])
        self.assertEqual(
            build_interaction_preference_context(state.interaction_preferences), ""
        )


class RenderingTest(unittest.TestCase):
    def test_empty_state_renders_nothing(self):
        self.assertEqual(build_interaction_preference_context([]), "")
        self.assertEqual(build_interaction_preference_context(None), "")

    def test_active_preference_renders_with_header(self):
        stored = [
            {
                "id": "a",
                "category": "tone",
                "polarity": "avoid",
                "text": "jangan terlalu galak",
                "created_at": "2026-10-01T00:00:00+00:00",
                "updated_at": "2026-10-01T00:00:00+00:00",
                "source": "conversation",
                "status": PREFERENCE_STATUS_ACTIVE,
            }
        ]
        block = build_interaction_preference_context(stored, tz=JKT)
        self.assertIn("Standing interaction preferences", block)
        self.assertIn("tone of voice", block)
        self.assertIn("jangan terlalu galak", block)

    def test_superseded_preference_is_not_rendered(self):
        stored = [
            {
                "id": "a",
                "category": "tone",
                "polarity": "avoid",
                "text": "lama",
                "status": PREFERENCE_STATUS_SUPERSEDED,
                "updated_at": "2026-10-01T00:00:00+00:00",
            },
            {
                "id": "b",
                "category": "tone",
                "polarity": "prefer",
                "text": "baru",
                "status": PREFERENCE_STATUS_ACTIVE,
                "updated_at": "2026-10-02T00:00:00+00:00",
            },
        ]
        block = build_interaction_preference_context(stored, tz=JKT)
        self.assertNotIn("lama", block)
        self.assertIn("baru", block)

    def test_render_respects_token_budget(self):
        stored = [
            {
                "id": str(i),
                "category": c,
                "polarity": "avoid",
                "text": f"preferensi panjang nomor {i} dengan teks tambahan",
                "status": PREFERENCE_STATUS_ACTIVE,
                "updated_at": f"2026-10-0{i + 1}T00:00:00+00:00",
            }
            for i, c in enumerate(
                ["tone", "formality", "humor", "address", "directness"]
            )
        ]
        block = build_interaction_preference_context(stored, tz=JKT, max_tokens=60)
        self.assertTrue(block)
        self.assertLessEqual(len(block), 400)


class AgentContextTest(unittest.TestCase):
    """Session A -> Session B with the real agent and real persistence."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        self.conf = "agent_pref"

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def agent_for(self, history_uid):
        from src.open_llm_vtuber.chat_history_manager import create_new_history

        uid = create_new_history(self.conf)
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
        agent.set_memory_from_history(self.conf, uid, user_timezone=JKT)
        del history_uid
        return agent

    def prompt_of(self, agent):
        return agent._relationship_system_prompt(PERSONA)

    def test_a_preference_crosses_sessions(self):
        hist_a = self.agent_for("")
        updated = hist_a.observe_character_events(
            "Mulai sekarang jangan terlalu galak ke aku.", "Baik, aku pelan."
        )
        self.assertTrue(updated)
        del hist_a
        agent_b = self.agent_for("")
        prompt = self.prompt_of(agent_b)
        self.assertIn("Standing interaction preferences", prompt)
        self.assertIn("galak", prompt)
        self.assertEqual(len(agent_b.list_interaction_preferences()), 1)

    def test_restart_keeps_preference(self):
        agent = self.agent_for("")
        agent.observe_character_events("Jangan terlalu formal.", "Siap.")
        del agent
        again = self.agent_for("")
        self.assertEqual(len(again.list_interaction_preferences()), 1)
        self.assertIn("formal", self.prompt_of(again))

    def test_preference_does_not_change_relationship(self):
        agent = self.agent_for("")
        before = agent.relationship_status
        agent.observe_character_events(
            "Mulai sekarang jangan terlalu galak ke aku.", "Iya."
        )
        self.assertEqual(agent.relationship_status, before)

    def test_preference_does_not_become_memory(self):
        agent = self.agent_for("")
        agent.observe_character_events("Jangan terlalu formal.", "Siap.")
        self.assertEqual(agent.list_character_memories(), [])

    def test_preference_does_not_touch_persona(self):
        agent = self.agent_for("")
        agent.observe_character_events("Jangan terlalu formal.", "Siap.")
        # Persona is preserved, not replaced: it stays present verbatim.
        self.assertIn(PERSONA, agent._system)
        self.assertIn(PERSONA, self.prompt_of(agent))

    def test_preference_survives_relationship_change(self):
        agent = self.agent_for("")
        agent.observe_character_events("Jangan terlalu formal.", "Siap.")
        set_character_relationship(self.conf, "close", "test")
        fresh = self.agent_for("")
        self.assertEqual(len(fresh.list_interaction_preferences()), 1)
        self.assertEqual(fresh.relationship_status, "close")

    def test_reset_clears_preferences(self):
        agent = self.agent_for("")
        agent.observe_character_events("Jangan terlalu formal.", "Siap.")
        self.assertEqual(len(agent.list_interaction_preferences()), 1)
        reset_character_state(self.conf)
        fresh = self.agent_for("")
        self.assertEqual(fresh.list_interaction_preferences(), [])

    def test_ordinary_turn_injects_nothing(self):
        agent = self.agent_for("")
        agent.observe_character_events("halo, apa kabar?", "Baik!")
        self.assertNotIn("Standing interaction preferences", self.prompt_of(agent))

    def test_explicit_memory_flow_still_works(self):
        agent = self.agent_for("")
        agent.observe_character_events("ingat bahwa aku suka bakso", "Bakso! solidly.")
        self.assertEqual(len(agent.list_character_memories()), 1)
        self.assertIn("bakso", self.prompt_of(agent))

    def test_preference_and_memory_are_separate(self):
        agent = self.agent_for("")
        agent.observe_character_events("Jangan terlalu formal.", "Siap.")
        agent.observe_character_events("ingat bahwa aku suka bakso", "Bakso! solidly.")
        self.assertEqual(len(agent.list_interaction_preferences()), 1)
        self.assertEqual(len(agent.list_character_memories()), 1)
        prompt = self.prompt_of(agent)
        self.assertIn("Standing interaction preferences", prompt)
        self.assertIn("bakso", prompt)


if __name__ == "__main__":
    unittest.main()
