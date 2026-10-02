"""Orphan-turn guard — deterministic tests (fake sockets, tmp dirs).

Proves a text-input arriving with history_uid == "" gets exactly one
auto-created history before the turn (never an unpersisted turn), and
that a preset history is never duplicated. No network, no LLM.
"""

import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.open_llm_vtuber import websocket_handler as wsh_mod
from src.open_llm_vtuber.chat_history_manager import get_history


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, payload):
        self.sent.append(json.loads(payload))


class FakeAgent:
    def __init__(self):
        self._user_timezone = None
        self.memory_loads = []
        self.chat_calls = 0

    def set_memory_from_history(self, conf_uid, history_uid, user_timezone=None):
        self.memory_loads.append((conf_uid, history_uid, user_timezone))

    async def chat(self, batch_input):
        self.chat_calls += 1
        if False:
            yield None


def make_context(history_uid=""):
    return SimpleNamespace(
        character_config=SimpleNamespace(
            conf_uid="guardchar",
            human_name="User",
            avatar=None,
            character_name="Mili",
        ),
        history_uid=history_uid,
        user_timezone=None,
        agent_engine=FakeAgent(),
        asr_engine=SimpleNamespace(),
    )


def make_handler():
    handler = wsh_mod.WebSocketHandler.__new__(wsh_mod.WebSocketHandler)
    handler.client_contexts = {}
    handler.client_connections = {}
    handler.received_data_buffers = {}
    handler.chat_group_manager = SimpleNamespace(get_client_group=lambda uid: None)

    async def no_broadcast(*args, **kwargs):
        return None

    handler.broadcast_to_group = no_broadcast
    handler.current_conversation_tasks = {}
    handler._proactive_timer_tasks = {}
    handler._proactive_states = {}
    handler._proactive_machines = {}
    handler._proactive_maintenance = set()

    async def no_proactive(*args, **kwargs):
        return None

    handler._activate_proactive_for_history = no_proactive
    return handler


def history_files(conf_uid="guardchar"):
    d = os.path.join("chat_history", conf_uid)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".json"))


class OrphanGuardTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    async def trigger(self, handler, socket, client_uid, text="halo"):
        await handler._handle_conversation_trigger(
            socket,
            client_uid,
            {"type": "text-input", "text": text},
        )

    async def test_a_existing_history_no_new_file(self):
        handler = make_handler()
        socket = FakeSocket()
        handler.client_contexts["c1"] = make_context(history_uid="preset-uid")
        with patch.object(
            wsh_mod, "handle_conversation_trigger", autospec=True
        ) as mock_trigger:

            async def fake_trigger(**kwargs):
                return None

            mock_trigger.side_effect = fake_trigger
            await self.trigger(handler, socket, "c1")
        self.assertEqual(history_files(), [])
        self.assertEqual(handler.client_contexts["c1"].history_uid, "preset-uid")
        self.assertTrue(mock_trigger.await_count >= 1)
        self.assertFalse(
            any(m.get("type") == "new-history-created" for m in socket.sent)
        )

    async def test_b_orphan_creates_exactly_one_history(self):
        handler = make_handler()
        socket = FakeSocket()
        handler.client_contexts["c1"] = make_context(history_uid="")
        with patch.object(
            wsh_mod, "handle_conversation_trigger", autospec=True
        ) as mock_trigger:

            async def fake_trigger(**kwargs):
                return None

            mock_trigger.side_effect = fake_trigger
            await self.trigger(handler, socket, "c1")
        files = history_files()
        self.assertEqual(len(files), 1)
        new_uid = files[0][:-5]
        self.assertEqual(handler.client_contexts["c1"].history_uid, new_uid)
        self.assertTrue(mock_trigger.await_count >= 1)
        created = [m for m in socket.sent if m.get("type") == "new-history-created"]
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["history_uid"], new_uid)
        # Memory init ran for the new history.
        agent = handler.client_contexts["c1"].agent_engine
        self.assertIn(("guardchar", new_uid, None), agent.memory_loads)

    async def test_c_rapid_messages_reuse_one_history(self):
        handler = make_handler()
        socket = FakeSocket()
        handler.client_contexts["c1"] = make_context(history_uid="")
        with patch.object(
            wsh_mod, "handle_conversation_trigger", autospec=True
        ) as mock_trigger:

            async def fake_trigger(**kwargs):
                return None

            mock_trigger.side_effect = fake_trigger
            await self.trigger(handler, socket, "c1", text="satu")
            first_uid = handler.client_contexts["c1"].history_uid
            await self.trigger(handler, socket, "c1", text="dua")
        self.assertEqual(len(history_files()), 1)
        self.assertEqual(handler.client_contexts["c1"].history_uid, first_uid)
        self.assertTrue(first_uid)

    async def test_d_creation_failure_drops_turn_with_error(self):
        handler = make_handler()
        socket = FakeSocket()
        handler.client_contexts["c1"] = make_context(history_uid="")
        with (
            patch.object(
                wsh_mod, "handle_conversation_trigger", autospec=True
            ) as mock_trigger,
            patch.object(wsh_mod, "create_new_history", return_value=""),
        ):
            await self.trigger(handler, socket, "c1")
        self.assertFalse(mock_trigger.called)
        self.assertEqual(handler.client_contexts["c1"].history_uid, "")
        errors = [m for m in socket.sent if m.get("type") == "error"]
        self.assertEqual(len(errors), 1)

    async def test_e_new_socket_does_not_touch_old(self):
        handler = make_handler()
        handler.client_contexts["old"] = make_context(history_uid="old-uid")
        handler.client_contexts["new"] = make_context(history_uid="")
        with patch.object(
            wsh_mod, "handle_conversation_trigger", autospec=True
        ) as mock_trigger:

            async def fake_trigger(**kwargs):
                return None

            mock_trigger.side_effect = fake_trigger
            await self.trigger(
                handler,
                FakeSocket(),
                "new",
            )
        self.assertEqual(handler.client_contexts["old"].history_uid, "old-uid")
        self.assertTrue(handler.client_contexts["new"].history_uid)
        self.assertNotEqual(handler.client_contexts["new"].history_uid, "old-uid")
        self.assertEqual(len(history_files()), 1)

    async def test_f_preset_y_keeps_y(self):
        handler = make_handler()
        socket = FakeSocket()
        handler.client_contexts["c1"] = make_context(history_uid="history-Y")
        with patch.object(
            wsh_mod, "handle_conversation_trigger", autospec=True
        ) as mock_trigger:

            async def fake_trigger(**kwargs):
                return None

            mock_trigger.side_effect = fake_trigger
            await self.trigger(handler, socket, "c1", text="masuk Y")
        self.assertEqual(handler.client_contexts["c1"].history_uid, "history-Y")
        self.assertEqual(history_files(), [])


class OrphanPersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_g_orphan_turn_persists(self):
        tmp = tempfile.TemporaryDirectory()
        old = os.getcwd()
        os.chdir(tmp.name)
        try:
            handler = make_handler()
            socket = FakeSocket()
            handler.client_contexts["c1"] = make_context(history_uid="")
            await handler._handle_conversation_trigger(
                socket, "c1", {"type": "text-input", "text": "orphan halo"}
            )
            # The conversation runs as a fire-and-forget task; await it
            # so persistence assertions are deterministic.
            task = handler.current_conversation_tasks.get("c1")
            if task is not None and not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=60)
                except (asyncio.CancelledError, Exception):
                    pass
            # Real pipeline ran (FakeAgent.chat yields nothing): the human
            # message must be stored in the auto-created history.
            files = history_files()
            self.assertEqual(len(files), 1)
            stored = get_history("guardchar", files[0][:-5])
            texts = [m.get("content", "") for m in stored]
            self.assertIn("orphan halo", texts)
        finally:
            os.chdir(old)
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
