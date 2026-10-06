"""Self Model integration — runtime wiring across categories (isolated).

Matrix gaps covered here: 2 (derivation-input persistence), 3 (tendency
render from persisted inputs), 12 (restart recompute), 13 (old-state),
14 (malformed fail-soft), 15 (render idempotence), 17 (bounded WITH prefs),
18 (Context Builder wiring), 19 (_decision_inputs usage), 20 (no category
contamination). Already covered elsewhere and cited, not duplicated:
1 derivation, 4/5/6 goals, 7 episodic, 8/9/10/11 separations, 16 supersede.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.open_llm_vtuber.self_model import (
    SELF_CONTEXT_MAX_TOKENS,
    build_self_context,
    derive_activity_preferences,
)

JKT = "Asia/Jakarta"
NOW = datetime(2026, 10, 2, 5, 0, tzinfo=timezone.utc)  # Fri 12:00 WIB
CONF = "selfmodelchar"


def make_agent(conf_uid=CONF):
    from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
    from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig

    class _LLM:
        model = "selfmodel"
        max_tokens = 8

        async def chat_completion(self, *a, **k):
            raise AssertionError("no LLM in self-model path")
            if False:
                yield None

    agent = BasicMemoryAgent(
        llm=_LLM(),
        system="persona",
        live2d_model=SimpleNamespace(extract_emotion=lambda text: []),
        tts_preprocessor_config=TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={"translate_audio": False, "translate_provider": "deeplx"},
        ),
    )
    agent._character_conf_uid = conf_uid
    agent._user_timezone = JKT
    agent._load_character_state(conf_uid)
    return agent


def seed_reading_evidence(conf_uid=CONF):
    """3 reading mentions over 2 days -> ESTABLISHED reading."""
    from src.open_llm_vtuber.character_state import (
        add_character_memory,
        load_character_state,
        save_character_state,
    )

    add_character_memory(conf_uid, "aku suka baca buku novel", explicit=False)
    add_character_memory(conf_uid, "aku lagi baca buku cerita", explicit=False)
    state = load_character_state(conf_uid)
    yesterday = (NOW - timedelta(days=1)).isoformat()
    state.memories.append(
        {"text": "aku baca komik kemarin", "added_at": yesterday,
         "explicit": False, "kind": ""}
    )
    save_character_state(conf_uid, state)


class SelfModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        for d in ("character_state", "episodic", "world_state"):
            os.makedirs(d, exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_02_derivation_inputs_persist(self):
        from src.open_llm_vtuber.character_state import load_character_state

        seed_reading_evidence()
        texts = [m["text"] for m in load_character_state(CONF).memories]
        self.assertEqual(len(texts), 3)

    def test_03_tendency_renders_from_persisted_inputs(self):
        from src.open_llm_vtuber.character_state import load_character_state

        seed_reading_evidence()
        prefs = derive_activity_preferences(
            memories=load_character_state(CONF).memories, tz=JKT
        )
        established = [c for c in prefs if c.established]
        self.assertEqual([c.activity for c in established], ["reading"])
        block = build_self_context(preferences=prefs)
        self.assertIn("reading", block)

    def test_12_restart_recompute_equal(self):
        from src.open_llm_vtuber.character_state import load_character_state

        seed_reading_evidence()
        before = derive_activity_preferences(
            memories=load_character_state(CONF).memories, tz=JKT
        )
        after = derive_activity_preferences(
            memories=load_character_state(CONF).memories, tz=JKT
        )
        self.assertEqual(
            [(c.activity, c.evidence_count, c.established) for c in before],
            [(c.activity, c.evidence_count, c.established) for c in after],
        )

    def test_13_old_state_without_new_fields(self):
        os.makedirs("character_state", exist_ok=True)
        with open(os.path.join("character_state", f"{CONF}.json"), "w") as h:
            json.dump(
                {"relationship_status": "close",
                 "memories": [{"text": "suka kopi", "added_at": NOW.isoformat(),
                               "explicit": True, "kind": ""}]},
                h,
            )
        agent = make_agent()
        prompt = agent._relationship_system_prompt("persona")
        self.assertIn("SELF:", prompt)
        self.assertIn("close", prompt)

    def test_14_malformed_state_fail_soft(self):
        os.makedirs("character_state", exist_ok=True)
        with open(os.path.join("character_state", f"{CONF}.json"), "w") as h:
            json.dump(
                {"relationship_status": "close",
                 "memories": [123, None, {"text": "", "added_at": "junk"}],
                 "goals": "junk",
                 "interaction_preferences": [{"bogus": True}]},
                h,
            )
        agent = make_agent()
        prompt = agent._relationship_system_prompt("persona")
        self.assertIn("SELF:", prompt)
        self.assertIn("Mili", prompt)

    def test_15_render_idempotent(self):
        seed_reading_evidence()
        from src.open_llm_vtuber.character_state import load_character_state

        prefs = derive_activity_preferences(
            memories=load_character_state(CONF).memories, tz=JKT
        )
        self.assertEqual(build_self_context(preferences=prefs),
                         build_self_context(preferences=prefs))

    def test_17_bounded_with_preferences(self):
        seed_reading_evidence()
        from src.open_llm_vtuber.character_state import load_character_state

        prefs = derive_activity_preferences(
            memories=load_character_state(CONF).memories, tz=JKT
        )
        block = build_self_context(preferences=prefs)
        from src.open_llm_vtuber.agent.context_window import estimate_tokens

        self.assertLessEqual(estimate_tokens(block), SELF_CONTEXT_MAX_TOKENS)

    def test_18_context_builder_wiring(self):
        agent = make_agent()
        plain = agent._relationship_system_prompt("persona")
        self.assertNotIn("Emerging preferences", plain)
        seed_reading_evidence()
        agent._character_state = __import__(
            "src.open_llm_vtuber.character_state", fromlist=["load_character_state"]
        ).load_character_state(CONF)
        wired = agent._relationship_system_prompt("persona")
        self.assertIn("Emerging preference: reading", wired)

    def test_19_decision_inputs_use_established(self):
        agent = make_agent()
        seed_reading_evidence()
        from src.open_llm_vtuber.character_state import load_character_state

        agent._character_state = load_character_state(CONF)
        inputs = agent._decision_inputs()
        self.assertIsNotNone(inputs)
        self.assertIn("reading", tuple(inputs.preferred_activities))

    def test_20_no_category_contamination(self):
        from src.open_llm_vtuber.character_state import load_character_state

        seed_reading_evidence()
        agent = make_agent()
        agent._character_state = load_character_state(CONF)
        agent.set_goal_status("try-three-dishes", "active")
        agent._character_state = load_character_state(CONF)
        prompt = agent._relationship_system_prompt("persona")
        # Each category renders in its own block, exactly once for the
        # derived-preference line; goal ids never leak as prompt text.
        self.assertEqual(prompt.count("Emerging preference"), 1)
        self.assertNotIn("try-three-dishes", prompt)
        self.assertIn("reading", prompt)


if __name__ == "__main__":
    unittest.main()
