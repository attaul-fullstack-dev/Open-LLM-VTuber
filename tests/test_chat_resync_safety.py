"""Chat resync safety — deterministic tests (fake sockets, tmp dirs).

Proves the send -> resync race cannot silently drop an accepted user
message at the protocol level, and that the turn lifecycle stays coherent:

- history-data payloads carry history_uid (fetch + detached delivery).
- Fetching a history with a detached in-flight turn re-emits chain-start
  (thinking) to the new viewer; without a detached turn nothing extra fires.
- Detached completion delivers history-data + chain-end when the viewer runs
  no live turn, and skips chain-end when a newer live turn owns thinking.
- A trigger queued behind a detached orphan emits turn-queued immediately
  (no silent idle), then runs normally.
- A normal accepted turn emits chain-start tagged with its history_uid.

No network, no LLM.
"""

import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from src.open_llm_vtuber import websocket_handler as wsh_mod
from src.open_llm_vtuber.chat_history_manager import store_message


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, payload):
        self.sent.append(json.loads(payload))

    def types(self):
        return [m.get("type") for m in self.sent]


class FakeAgent:
    def __init__(self):
        self._user_timezone = None

    def set_memory_from_history(self, conf_uid, history_uid, user_timezone=None):
        return None

    async def chat(self, batch_input):
        if False:
            yield None


def make_context(history_uid="h1"):
    return SimpleNamespace(
        character_config=SimpleNamespace(
            conf_uid="resyncchar",
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
    handler.broadcast_to_group = None
    handler.current_conversation_tasks = {}
    handler._proactive_timer_tasks = {}
    handler._proactive_states = {}
    handler._proactive_machines = {}
    handler._proactive_maintenance = set()

    async def no_proactive(*args, **kwargs):
        return None

    handler._activate_proactive_for_history = no_proactive
    handler._cancel_proactive_timer = no_proactive
    return handler


def seed_history(uid="h1", texts=("halo",)):
    os.makedirs(os.path.join("chat_history", "resyncchar"), exist_ok=True)
    for text in texts:
        store_message(
            conf_uid="resyncchar",
            history_uid=uid,
            role="human",
            content=text,
            name="User",
        )


def make_detached(coro=None):
    async def _idle():
        if coro is not None:
            await coro()
        await asyncio.sleep(0)

    return asyncio.ensure_future(_idle())


class ResyncSafetyTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    async def test_a_fetch_history_data_carries_uid(self):
        handler = make_handler()
        socket = FakeSocket()
        handler.client_contexts["c1"] = make_context("h1")
        seed_history("h1")
        await handler._handle_fetch_history(socket, "c1", {"history_uid": "h1"})
        data = [m for m in socket.sent if m.get("type") == "history-data"]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0].get("history_uid"), "h1")
        self.assertTrue(any(m.get("content") == "halo" for m in data[0]["messages"]))
        # No detached turn: no extra lifecycle signals.
        controls = [m.get("text") for m in socket.sent if m.get("type") == "control"]
        self.assertEqual(controls, [])

    async def test_b_fetch_with_detached_turn_reemits_thinking(self):
        handler = make_handler()
        socket = FakeSocket()
        handler.client_contexts["c1"] = make_context("h1")
        seed_history("h1")
        gate = asyncio.Event()
        detached = make_detached(gate.wait)
        handler._detached_registry()["h1"] = detached
        try:
            await handler._handle_fetch_history(socket, "c1", {"history_uid": "h1"})
        finally:
            gate.set()
            await asyncio.wait_for(asyncio.shield(detached), timeout=5)
        starts = [
            m
            for m in socket.sent
            if m.get("type") == "control"
            and m.get("text") == "conversation-chain-start"
        ]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0].get("history_uid"), "h1")

    async def test_c_detached_delivery_closes_lifecycle_when_idle(self):
        handler = make_handler()
        socket = FakeSocket()
        handler.client_contexts["c1"] = make_context("h1")
        handler.client_connections["c1"] = socket
        handler._subscriber_registry()["c1"] = "h1"
        seed_history("h1", texts=("pesan", "jawaban"))
        await handler._deliver_history_to_subscriber("h1")
        self.assertEqual(
            [m.get("type") for m in socket.sent],
            ["history-data", "control"],
        )
        self.assertEqual(socket.sent[0].get("history_uid"), "h1")
        self.assertEqual(socket.sent[1].get("text"), "conversation-chain-end")
        self.assertEqual(socket.sent[1].get("history_uid"), "h1")

    async def test_d_delivery_skips_chain_end_behind_live_turn(self):
        handler = make_handler()
        socket = FakeSocket()
        handler.client_contexts["c1"] = make_context("h1")
        handler.client_connections["c1"] = socket
        handler._subscriber_registry()["c1"] = "h1"
        seed_history("h1")

        async def _live():
            await asyncio.sleep(5)

        live = asyncio.ensure_future(_live())
        handler.current_conversation_tasks["c1"] = live
        try:
            await handler._deliver_history_to_subscriber("h1")
        finally:
            live.cancel()
            try:
                await live
            except (asyncio.CancelledError, Exception):
                pass
        self.assertEqual(
            [m.get("type") for m in socket.sent],
            ["history-data"],
        )

    async def test_e_queued_trigger_signals_thinking_immediately(self):
        handler = make_handler()
        socket = FakeSocket()
        handler.client_contexts["c1"] = make_context("h1")
        seed_history("h1", texts=())
        gate = asyncio.Event()
        detached = make_detached(gate.wait)
        handler._detached_registry()["h1"] = detached
        # Real pipeline (FakeAgent yields nothing): the new trigger must
        # emit turn-queued right away, then wait for the orphan.
        await handler._handle_conversation_trigger(
            socket, "c1", {"type": "text-input", "text": "kedua"}
        )
        # The turn runs fire-and-forget: yield so it emits turn-queued
        # while still blocked on the orphan gate.
        for _ in range(10):
            await asyncio.sleep(0)
        queued = [
            m
            for m in socket.sent
            if m.get("type") == "control"
            and m.get("text") == "conversation-turn-queued"
        ]
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0].get("history_uid"), "h1")
        # Release the orphan; the queued turn then runs to completion and
        # persists its own human message (no concurrent-write corruption).
        gate.set()
        task = handler.current_conversation_tasks.get("c1")
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=60)
            except (asyncio.CancelledError, Exception):
                pass
        try:
            from src.open_llm_vtuber.chat_history_manager import get_history

            texts = [m.get("content", "") for m in get_history("resyncchar", "h1")]
        finally:
            handler._detached_registry().pop("h1", None)
        self.assertIn("kedua", texts)

    async def test_f_normal_turn_chain_start_carries_uid(self):
        handler = make_handler()
        socket = FakeSocket()
        handler.client_contexts["c1"] = make_context("h1")
        await handler._handle_conversation_trigger(
            socket, "c1", {"type": "text-input", "text": "halo normal"}
        )
        task = handler.current_conversation_tasks.get("c1")
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=60)
            except (asyncio.CancelledError, Exception):
                pass
        starts = [
            m
            for m in socket.sent
            if m.get("type") == "control"
            and m.get("text") == "conversation-chain-start"
        ]
        self.assertTrue(starts)
        self.assertEqual(starts[0].get("history_uid"), "h1")


if __name__ == "__main__":
    unittest.main()
