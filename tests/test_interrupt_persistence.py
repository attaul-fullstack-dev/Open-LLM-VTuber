"""Interrupt persistence — deterministic tests (no network, no LLM).

Regression: sending a new message while the AI is still working makes the
frontend send `interrupt-signal` with the *partial* spoken text. When the
interrupt lands before the first sentence, that text is empty, and the turn
persisted an empty assistant message followed by the interruption marker. The
empty turn then fed back into the next LLM context and rendered as a blank
bubble.

These tests pin the fixed contract:
  * empty/whitespace heard_response -> only the "[Interrupted by user]" marker
  * non-empty heard_response        -> partial assistant turn + marker
  * the in-flight task is still cancelled and handle_interrupt still called
  * no history_uid -> nothing persisted at all
"""

import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from src.open_llm_vtuber.chat_history_manager import get_history
from src.open_llm_vtuber.conversations.conversation_handler import (
    handle_individual_interrupt,
)


class FakeAgent:
    def __init__(self):
        self.interrupted_with = []

    def handle_interrupt(self, heard):
        self.interrupted_with.append(heard)


def make_context(history_uid="sess-1", conf_uid="xchar"):
    return SimpleNamespace(
        character_config=SimpleNamespace(
            conf_uid=conf_uid, character_name="Mili", avatar="mao_pro"
        ),
        history_uid=history_uid,
        agent_engine=FakeAgent(),
    )


class InterruptPersistenceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs(os.path.join("chat_history", "xchar"), exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _seed_history(self, uid="sess-1"):
        path = os.path.join("chat_history", "xchar", f"{uid}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([{"role": "metadata", "timestamp": "2026-10-03T10:00:00+00:00"}], handle)

    def _roles(self, uid="sess-1"):
        # get_history returns conversation messages only (metadata excluded)
        return [(m.get("role"), m.get("content")) for m in get_history("xchar", uid)]

    async def _run(self, heard, history_uid="sess-1"):
        async def forever():
            await asyncio.sleep(3600)

        task = asyncio.create_task(forever())
        ctx = make_context(history_uid)
        await handle_individual_interrupt(
            client_uid="client-1",
            current_conversation_tasks={"client-1": task},
            context=ctx,
            heard_response=heard,
        )
        await asyncio.sleep(0)
        return ctx, task

    async def test_empty_interrupter_persists_no_assistant_turn(self):
        """The regression: rapid messages interrupted before any speech."""
        self._seed_history()
        ctx, task = await self._run("")

        self.assertTrue(task.cancelled() or task.done())
        self.assertEqual(ctx.agent_engine.interrupted_with, [""])
        roles = self._roles()
        self.assertNotIn(
            "ai",
            [r for r, _ in roles],
            f"empty assistant turn persisted: {roles}",
        )
        self.assertEqual(
            roles,
            [("system", "[Interrupted by user]")],
            f"unexpected transcript: {roles}",
        )

    async def test_whitespace_interrupter_persists_no_assistant_turn(self):
        self._seed_history()
        await self._run("   \n  ")
        roles = self._roles()
        self.assertNotIn("ai", [r for r, _ in roles], roles)
        self.assertIn(("system", "[Interrupted by user]"), roles)

    async def test_partial_interrupter_keeps_the_spoken_prefix(self):
        """Existing behaviour must be preserved for a real partial response."""
        self._seed_history()
        await self._run("aku lagi ngomongHalf sentence")
        roles = self._roles()
        self.assertIn(("ai", "aku lagi ngomongHalf sentence"), roles)
        self.assertIn(("system", "[Interrupted by user]"), roles)

    async def test_marker_is_always_written(self):
        """The transcript must still record that the turn was cut short."""
        self._seed_history()
        for heard in ("", "  ", "ada"):
            await self._run(heard)
            self.assertIn(("system", "[Interrupted by user]"), self._roles())

    async def test_no_history_uid_persists_nothing(self):
        ctx, _ = await self._run("", history_uid="")
        self.assertEqual(ctx.agent_engine.interrupted_with, [""])
        self.assertEqual(os.listdir(os.path.join("chat_history", "xchar")), [])

    async def test_interrupt_with_no_task_entry_is_a_no_op(self):
        """Pre-existing contract: no task entry -> nothing touched.

        The handler is guarded by `if client_uid in current_conversation_tasks`,
        so an interrupt that arrives after the task was already reaped must not
        write a marker for a turn that never ran.
        """
        self._seed_history()
        ctx = make_context()
        await handle_individual_interrupt(
            client_uid="client-1",
            current_conversation_tasks={},
            context=ctx,
            heard_response="",
        )
        self.assertEqual(ctx.agent_engine.interrupted_with, [])
        self.assertEqual(self._roles(), [])

    async def test_interrupt_after_task_finished_still_marks(self):
        """A finished-but-still-registered task must not skip the marker."""
        self._seed_history()

        async def already_done():
            return None

        task = asyncio.create_task(already_done())
        await asyncio.sleep(0)
        ctx = make_context()
        await handle_individual_interrupt(
            client_uid="client-1",
            current_conversation_tasks={"client-1": task},
            context=ctx,
            heard_response="",
        )
        self.assertEqual(ctx.agent_engine.interrupted_with, [""])
        self.assertIn(("system", "[Interrupted by user]"), self._roles())
        self.assertNotIn("ai", [r for r, _ in self._roles()])


if __name__ == "__main__":
    unittest.main()
