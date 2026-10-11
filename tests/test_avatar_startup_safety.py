"""Avatar-model failure must never abort server startup (deterministic).

When the configured Live2D model cannot be resolved, ``init_live2d``
promises to "proceed without Live2D" but left ``live2d_model = None`` while
``construct_system_prompt`` dereferenced ``.emo_str`` unconditionally. The
resulting ``AttributeError`` propagated out of ``init_agent`` into
``run_server.py``, which exits the process -- a single bad avatar name
bricked the whole launch.
"""

import asyncio
import unittest
from types import SimpleNamespace


def _context(live2d_model):
    from src.open_llm_vtuber.service_context import ServiceContext

    context = ServiceContext.__new__(ServiceContext)
    context.live2d_model = live2d_model
    context.system_config = SimpleNamespace(
        tool_prompts={"live2d_expression_prompt": "live2d_expression_prompt"}
    )
    return context


class MissingAvatarStartupTest(unittest.TestCase):
    def test_prompt_constructs_without_a_model(self):
        context = _context(None)
        prompt = asyncio.run(context.construct_system_prompt("persona dasar "))
        assert isinstance(prompt, str)
        assert "[<insert_emomap_keys>]" not in prompt

    def test_prompt_constructs_with_a_model(self):
        model = SimpleNamespace(emo_str="[joy] [sadness]")
        context = _context(model)
        prompt = asyncio.run(context.construct_system_prompt("persona dasar "))
        assert "[joy] [sadness]" in prompt
        assert "[<insert_emomap_keys>]" not in prompt


if __name__ == "__main__":
    unittest.main()
