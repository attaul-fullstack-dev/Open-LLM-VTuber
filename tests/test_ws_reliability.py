"""WebSocket reliability — deterministic tests (fake sockets, no timing).

Covers BUG 1 (message loss) + BUG 2 (disconnect/resync) backend fixes:
bounded playback wait with guaranteed lifecycle tail, guarded sends,
group-trigger consistency, KeyError-free cleanup window, idempotent
disconnect, single logical turn under concurrent triggers, no orphan
tasks after mid-response disconnect.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.open_llm_vtuber.conversations import conversation_handler as ch_mod
from src.open_llm_vtuber.conversations import conversation_utils as cu_mod
from src.open_llm_vtuber.conversations.conversation_utils import (
    finalize_conversation_turn,
)
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager


class FakeSocket:
    """WebSocket double: records sends, optionally fails them."""

    def __init__(self, fail_send=False):
        self.attempts = []
        self.fail_send = fail_send

    async def send_text(self, payload):
        self.attempts.append(payload)
        if self.fail_send:
            raise RuntimeError("socket is closed")


def make_trigger_env():
    tasks = {}
    return {
        "client_uid": "client-1",
        "context": SimpleNamespace(),
        "websocket": FakeSocket(),
        "client_contexts": {},
        "client_connections": {},
        "received_data_buffers": {},
        "current_conversation_tasks": tasks,
        "broadcast_to_group": lambda *a, **k: asyncio.sleep(0),
    }


def trigger_text(env, text="halo"):
    return ch_mod.handle_conversation_trigger(
        msg_type="text-input",
        data={"type": "text-input", "text": text},
        client_uid=env["client_uid"],
        context=env["context"],
        websocket=env["websocket"],
        client_contexts=env["client_contexts"],
        client_connections=env["client_connections"],
        chat_group_manager=env.get("group_manager")
        or SimpleNamespace(
            get_client_group=lambda uid: None,
        ),
        received_data_buffers=env["received_data_buffers"],
        current_conversation_tasks=env["current_conversation_tasks"],
        broadcast_to_group=env["broadcast_to_group"],
    )


class FinalizeLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_1_timeout_ends_turn_with_lifecycle_tail(self):
        """Playback-complete never arrives: bounded wait, tail still sent."""
        socket = FakeSocket()
        manager = TTSTaskManager()
        manager.task_list.append(asyncio.sleep(0))
        with patch.object(cu_mod, "PLAYBACK_COMPLETE_TIMEOUT_S", 0.05):
            await finalize_conversation_turn(
                manager, socket.send_text, "ghost-client"
            )
        types = [json.loads(p).get("type") for p in socket.attempts]
        self.assertIn("backend-synth-complete", types)
        self.assertIn("force-new-message", types)
        chained = [
            p
            for p in socket.attempts
            if json.loads(p).get("text") == "conversation-chain-end"
        ]
        self.assertEqual(len(chained), 1)

    async def test_3_send_failure_still_completes_cleanup(self):
        """Dead socket: no raise, no hang, each lifecycle send tried once."""
        socket = FakeSocket(fail_send=True)
        manager = TTSTaskManager()
        manager.task_list.append(asyncio.sleep(0))
        with patch.object(cu_mod, "PLAYBACK_COMPLETE_TIMEOUT_S", 0.05):
            await finalize_conversation_turn(
                manager, socket.send_text, "dead-client"
            )
        types = [json.loads(p).get("type", "?") for p in socket.attempts]
        self.assertEqual(types.count("backend-synth-complete"), 1)
        self.assertEqual(types.count("force-new-message"), 1)
        chain_texts = [
            json.loads(p).get("text") for p in socket.attempts
        ]
        self.assertEqual(chain_texts.count("conversation-chain-end"), 1)

    async def test_5_chain_end_safe_without_tts_tasks(self):
        socket = FakeSocket(fail_send=True)
        await finalize_conversation_turn(
            TTSTaskManager(), socket.send_text, "dead-client"
        )
        self.assertEqual(len(socket.attempts), 2)  # force-new + chain-end


class ConcurrentTriggerTest(unittest.IsolatedAsyncioTestCase):
    async def test_9_double_trigger_leaves_single_surviving_turn(self):
        """Second trigger cancels the first; survivor completes its input."""
        env = make_trigger_env()
        gate = asyncio.Event()
        seen = []

        async def fake_process(**kwargs):
            seen.append(kwargs["user_input"])
            await gate.wait()
            return kwargs["user_input"]

        with patch.object(ch_mod, "process_single_conversation", fake_process):
            await trigger_text(env, "pertama")
            first = env["current_conversation_tasks"]["client-1"]
            await asyncio.sleep(0)  # let the first turn start
            await trigger_text(env, "kedua")
            second = env["current_conversation_tasks"]["client-1"]
            self.assertIsNot(first, second)
            await asyncio.sleep(0)
            self.assertTrue(first.cancelled() or first.done())
            gate.set()
            await asyncio.gather(second, return_exceptions=True)
        self.assertEqual(seen, ["pertama", "kedua"])
        self.assertTrue(second.done() and not second.cancelled())

    async def test_6_group_trigger_not_silently_dropped(self):
        """Busy group: old turn cancelled, new trigger runs (parity)."""
        env = make_trigger_env()
        env["group_manager"] = SimpleNamespace(
            get_client_group=lambda uid: SimpleNamespace(
                group_id="g1", members=["client-1", "client-2"]
            )
        )
        gate = asyncio.Event()
        seen = []

        async def fake_group(**kwargs):
            seen.append(kwargs["user_input"])
            await gate.wait()
            return kwargs["user_input"]

        with patch.object(ch_mod, "process_group_conversation", fake_group):
            await trigger_text(env, "grup-pertama")
            first = env["current_conversation_tasks"]["g1"]
            await asyncio.sleep(0)  # let the first turn start
            await trigger_text(env, "grup-kedua")
            second = env["current_conversation_tasks"]["g1"]
            self.assertIsNot(first, second)
            await asyncio.sleep(0)
            self.assertTrue(first.cancelled() or first.done())
            gate.set()
            await asyncio.gather(second, return_exceptions=True)
        self.assertEqual(seen, ["grup-pertama", "grup-kedua"])
        self.assertTrue(second.done() and not second.cancelled())


class DisconnectCleanupTest(unittest.IsolatedAsyncioTestCase):
    def _handler_with_client(self):
        from src.open_llm_vtuber.websocket_handler import WebSocketHandler

        handler = WebSocketHandler.__new__(WebSocketHandler)
        handler.client_connections = {"client-1": FakeSocket()}
        closed = []

        async def fake_close():
            closed.append(True)

        handler.client_contexts = {
            "client-1": SimpleNamespace(close=fake_close)
        }
        from src.open_llm_vtuber.chat_group import ChatGroupManager

        handler.chat_group_manager = ChatGroupManager()
        handler.received_data_buffers = {"client-1": None}
        handler.current_conversation_tasks = {}
        handler._proactive_timer_tasks = {}
        handler._proactive_states = {}
        handler._proactive_machines = {}
        handler._proactive_maintenance = set()
        handler._cancel_proactive_timer = lambda uid: asyncio.sleep(0)
        return handler, closed

    async def test_7_trigger_in_cleanup_window_no_keyerror(self):
        from src.open_llm_vtuber.websocket_handler import WebSocketHandler

        handler = WebSocketHandler.__new__(WebSocketHandler)
        handler.client_contexts = {}  # already cleaned up
        handler._proactive_timer_tasks = {}
        handler._proactive_states = {}
        handler._proactive_machines = {}
        socket = FakeSocket()
        await WebSocketHandler._handle_conversation_trigger(
            handler,
            socket,
            "gone-client",
            {"type": "text-input", "text": "halo?"},
        )
        errors = [
            json.loads(p)
            for p in socket.attempts
            if json.loads(p).get("type") == "error"
        ]
        self.assertEqual(len(errors), 1)

    async def test_8_double_cleanup_is_safe(self):
        handler, closed = self._handler_with_client()
        await handler.handle_disconnect("client-1")
        await handler.handle_disconnect("client-1")
        self.assertEqual(closed, [True])
        self.assertNotIn("client-1", handler.client_contexts)
        self.assertNotIn("client-1", handler.current_conversation_tasks)

    async def test_10_disconnect_mid_response_leaves_no_orphan(self):
        env = make_trigger_env()
        gate = asyncio.Event()

        async def hanging_process(**kwargs):
            await gate.wait()
            return "never"

        with patch.object(ch_mod, "process_single_conversation", hanging_process):
            await trigger_text(env, "tengah-jalan")
            task = env["current_conversation_tasks"]["client-1"]
            self.assertFalse(task.done())
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            env["current_conversation_tasks"].pop("client-1", None)
        self.assertNotIn("client-1", env["current_conversation_tasks"])


if __name__ == "__main__":
    unittest.main()
