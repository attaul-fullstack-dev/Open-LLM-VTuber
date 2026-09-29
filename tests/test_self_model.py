"""Self Model v1 — deterministic tests (no LLM, no I/O, no clock).

Covers A-Q: static identity, reality boundary, live references without
duplication, seeds without generation, token bound, natural-conversation
rule, persona amendment, and existing-suite safety (run separately).
"""

import asyncio
import inspect
import os
import unittest

from src.open_llm_vtuber import self_model
from src.open_llm_vtuber.self_model import (
    SELF_CONTEXT_MAX_TOKENS,
    SELF_SEED_TENDENCIES,
    build_self_context,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def full_block():
    return build_self_context(
        character_name="Mili",
        avatar_present=True,
        live2d_model_name="mao_pro",
        activity="reading",
        location="room",
        relationship_status="close",
        memory_count=2,
    )


class StaticIdentityTest(unittest.TestCase):
    def test_a_static_identity_renders(self):
        block = full_block()
        self.assertIn("Mili", block)

    def test_b_ai_nature_renders(self):
        self.assertIn("AI companion", full_block())

    def test_c_application_environment_renders(self):
        block = full_block()
        self.assertIn("app", block)
        self.assertIn("chat", block)
        self.assertIn("voice", block)
        self.assertIn("avatar", block)

    def test_d_physical_presence_boundary_renders(self):
        block = full_block()
        self.assertIn("No body/house/presence", block)

    def test_e_no_house_invention_instruction(self):
        block = full_block()
        self.assertIn("house", block)
        self.assertIn("never invent", block.lower())

    def test_f_simulated_life_is_not_roleplay(self):
        block = full_block()
        self.assertIn("not roleplay", block)
        self.assertNotIn("simulated location is roleplay", block)


class LiveReferenceTest(unittest.TestCase):
    def test_g_avatar_resolves_present_and_absent(self):
        present = build_self_context(live2d_model_name="mao_pro")
        self.assertIn("mao_pro", present)
        absent = build_self_context()
        self.assertIn("(chat, voice).", absent)
        self.assertNotIn("mao_pro", absent)

    def test_h_world_state_referenced_not_duplicated(self):
        block = build_self_context(activity="reading", location="room")
        self.assertIn("reading", block)
        self.assertIn("room", block)
        # No storage: composer takes scalars only, touches no world store.
        self.assertNotIn("world_state.json", block)
        src = inspect.getsource(self_model)
        self.assertNotIn("json.dump", src)
        self.assertNotIn("save_world_state", src)

    def test_i_relationship_referenced_not_duplicated(self):
        block = build_self_context(relationship_status="dating")
        self.assertIn("dating", block)
        src = inspect.getsource(self_model)
        self.assertNotIn("set_character_relationship", src)
        self.assertNotIn("detect_relationship_update", src)

    def test_j_memory_not_dumped(self):
        secret = "user secret passphrase xyzzy-123"
        block = build_self_context(memory_count=5)
        self.assertNotIn(secret, block)
        self.assertIn("Facts: 5", block)


class SeedsAndGenerationTest(unittest.TestCase):
    def test_k_seed_goals_render(self):
        block = full_block()
        for seed in SELF_SEED_TENDENCIES:
            self.assertIn(seed, block)
        self.assertGreaterEqual(len(SELF_SEED_TENDENCIES), 3)
        self.assertLessEqual(len(SELF_SEED_TENDENCIES), 4)

    def test_l_no_goal_generation(self):
        first = full_block()
        second = full_block()
        self.assertEqual(first, second)
        src = inspect.getsource(self_model)
        self.assertNotIn("random", src)


class SafetyTest(unittest.TestCase):
    def test_m_no_llm_call(self):
        self.assertFalse(asyncio.iscoroutinefunction(build_self_context))
        src = inspect.getsource(build_self_context)
        for token in ("chat_completion", "generate", "llm_client", "provider"):
            self.assertNotIn(token, src)

    def test_n_block_stays_token_bounded(self):
        block = full_block()
        cost = len(block.encode("utf-8")) // 3
        self.assertLessEqual(cost, SELF_CONTEXT_MAX_TOKENS)
        # Nothing essential was truncated away.
        for marker in ("Tendencies:", "Can:", "Cannot:", "As an AI"):
            self.assertIn(marker, block)

    def test_o_no_as_an_ai_opener(self):
        block = full_block()
        self.assertIn('no "As an AI..." openers', block)
        self.assertEqual(block.count("As an AI"), 1)


class PersonaAmendmentTest(unittest.TestCase):
    def _read(self, rel):
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
            return f.read()

    def test_p_relevance_based_identity_rule(self):
        for rel in ("conf.yaml", os.path.join("characters", "id_mili.yaml")):
            text = self._read(rel)
            # Old absolute concealment is gone...
            self.assertNotIn("kecuali pengguna bertanya langsung", text)
            # ...replaced by relevance-based behavior.
            self.assertIn("bila tidak relevan", text)
            self.assertIn("jangan pernah mengaku manusia", text)
            self.assertIn("As an AI", text)


if __name__ == "__main__":
    unittest.main()
