"""Session resilience across reconnect — deterministic handler-level tests.

Proves RECONNECT != NEW CONVERSATION using the real WebSocketHandler
methods, the real conversation trigger, real temp-dir history files, and
fake sockets/agents. No network, no provider, no production history.

Covered:
- reconnect trigger carrying history_uid adopts the session (no Session B)
- repeated reconnects keep exactly one history file
- explicit new-conversation flow still creates a session
- unknown/foreign history_uid falls back to the orphan guard (BUG A intact)
- disconnect detaches (never cancels) an in-flight single turn
- disconnect still cancels untagged/group-style tasks
- a new trigger on the same history waits for the detached orphan
  (no parallel persists, no duplicates)
- fetch-history never creates sessions
- TTS sender survives a dead socket so the turn still persists
- session isolation between two histories
"""

import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.open_llm_vtuber import websocket_handler as wh_mod
from src.open_llm_vtuber.conversations import conversation_handler as ch_mod
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    store_message,
)

CONF = "resiliencechar"
JKT = "Asia/Jakarta"


class FakeSocket:
    def __init__(self, fail_send=False):
        self.attempts = []
        self.fail_send = fail_send

    async def send_text(self, payload):
        self.attempts.append(payload)
        if self.fail_send:
            raise RuntimeError("socket is closed")


def message_types(socket):
    out = []
    for payload in socket.attempts:
        try:
            out.append(json.loads(payload).get("type"))
        except Exception:
            out.append("?")
    return out


def history_files():
    folder = os.path.join("chat_history", CONF)
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith(".json"))


def make_agent(history_uid=None):
    async def _close():
        return None

    agent = SimpleNamespace(
        _character_conf_uid=CONF,
        loaded=[],
        set_memory_from_history=lambda conf_uid, history_uid, user_timezone=None: agent.loaded.append(
            history_uid
        ),
        close=_close,
    )
    return agent


def make_context(history_uid=""):
    agent = make_agent()
    return SimpleNamespace(
        history_uid=history_uid,
        user_timezone=JKT,
        agent_engine=agent,
        character_config=SimpleNamespace(conf_uid=CONF),
        close=agent.close,
    )


def make_handler():
    handler = wh_mod.WebSocketHandler.__new__(wh_mod.WebSocketHandler)
    handler.client_connections = {}
    handler.client_contexts = {}
    handler.received_data_buffers = {}
    handler.current_conversation_tasks = {}
    handler._detached_turns = {}
    handler._history_subscribers = {}
    handler._proactive_timer_tasks = {}
    handler._proactive_states = {}
    handler._proactive_machines = {}
    handler._proactive_maintenance = set()
    handler._proactive_budget_state = {}
    handler.chat_group_manager = SimpleNamespace(
        get_client_group=lambda uid: None,
        client_group_map={},
    )
    return handler


class HandlerBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        for folder in ("chat_history", "character_state", "episodic"):
            os.makedirs(folder, exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()


class AdoptSessionTest(HandlerBase):
    def _trigger(self, handler, context, socket, client_uid, data):
        with (
            patch.object(
                wh_mod.WebSocketHandler,
                "_activate_proactive_for_history",
                new=AsyncMock(),
            ),
            patch.object(
                wh_mod, "handle_conversation_trigger", new=AsyncMock()
            ) as turn,
        ):
            asyncio.new_event_loop().run_until_complete(
                self._run(handler, context, socket, client_uid, data)
            )
        return turn

    async def _run(self, handler, context, socket, client_uid, data):
        handler.client_contexts[client_uid] = context
        await handler._handle_conversation_trigger(socket, client_uid, data)

    def test_reconnect_trigger_adopts_existing_session(self):
        """Session A -> reconnect (empty socket session) + text carrying
        history_uid=A -> turn runs on A, no Session B is minted."""
        history_a = create_new_history(CONF)
        store_message(CONF, history_a, "human", "good night my love")
        handler = make_handler()
        context = make_context("")  # fresh socket, unrestored session
        socket = FakeSocket()
        turn = self._trigger(
            handler,
            context,
            socket,
            "conn-2",
            {"type": "text-input", "text": "good night", "history_uid": history_a},
        )
        self.assertEqual(context.history_uid, history_a)
        self.assertEqual(context.agent_engine.loaded, [history_a])
        self.assertEqual(len(history_files()), 1)
        self.assertNotIn("new-history-created", message_types(socket))
        self.assertEqual(turn.await_count, 1)

    def test_repeated_reconnects_keep_single_session(self):
        history_a = create_new_history(CONF)
        handler = make_handler()
        for index in range(3):
            context = make_context("")
            socket = FakeSocket()
            self._trigger(
                handler,
                context,
                socket,
                f"conn-{index}",
                {"type": "text-input", "text": "hai", "history_uid": history_a},
            )
            self.assertEqual(context.history_uid, history_a)
        self.assertEqual(len(history_files()), 1)

    def test_unknown_uid_falls_back_to_orphan_guard(self):
        """BUG A preserved: no usable uid -> exactly one history is created."""
        handler = make_handler()
        context = make_context("")
        socket = FakeSocket()
        self._trigger(
            handler,
            context,
            socket,
            "conn-9",
            {"type": "text-input", "text": "hai", "history_uid": "no-such-session"},
        )
        self.assertTrue(context.history_uid)
        self.assertEqual(len(history_files()), 1)
        self.assertIn("new-history-created", message_types(socket))

    def test_missing_uid_falls_back_to_orphan_guard(self):
        handler = make_handler()
        context = make_context("")
        socket = FakeSocket()
        self._trigger(
            handler,
            context,
            socket,
            "conn-9",
            {"type": "text-input", "text": "hai"},
        )
        self.assertTrue(context.history_uid)
        self.assertEqual(len(history_files()), 1)

    def test_fetch_history_never_creates_sessions(self):
        handler = make_handler()
        context = make_context("some-uid")
        handler.client_contexts["conn-1"] = context
        socket = FakeSocket()

        async def _run():
            with patch.object(
                wh_mod.WebSocketHandler,
                "_activate_proactive_for_history",
                new=AsyncMock(),
            ):
                await handler._handle_fetch_history(
                    socket, "conn-1", {"history_uid": "ghost-uid"}
                )

        asyncio.new_event_loop().run_until_complete(_run())
        self.assertEqual(history_files(), [])
        self.assertEqual(message_types(socket).count("new-history-created"), 0)

    def test_sessions_stay_isolated(self):
        history_a = create_new_history(CONF)
        history_b = create_new_history(CONF)
        store_message(CONF, history_a, "human", "pesan A")
        store_message(CONF, history_b, "human", "pesan B")
        handler = make_handler()
        context = make_context("")
        socket = FakeSocket()
        self._trigger(
            handler,
            context,
            socket,
            "conn-2",
            {"type": "text-input", "text": "lanjut", "history_uid": history_b},
        )
        self.assertEqual(context.history_uid, history_b)
        texts_a = [m["content"] for m in get_history(CONF, history_a)]
        self.assertEqual(texts_a, ["pesan A"])


class DisconnectDetachTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        for folder in ("chat_history", "character_state", "episodic"):
            os.makedirs(folder, exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _handler_with_task(self, task, history_uid="hist-A"):
        handler = make_handler()
        handler.client_connections["conn-1"] = FakeSocket()
        context = make_context(history_uid)
        handler.client_contexts["conn-1"] = context
        handler.current_conversation_tasks["conn-1"] = task
        return handler

    async def test_disconnect_detaches_single_turn(self):
        """OPTION 1: the in-flight turn is NOT cancelled; it must be able
        to run to completion after its owner disconnects."""
        release = asyncio.Event()
        finished = []

        async def fake_turn():
            await release.wait()
            finished.append(True)

        task = asyncio.create_task(fake_turn())
        task._olv_single_turn = True
        task._olv_history_uid = "hist-A"
        task._olv_detached = False
        handler = self._handler_with_task(task)
        with patch.object(wh_mod, "handle_client_disconnect", new=AsyncMock()):
            await handler.handle_disconnect("conn-1")
        self.assertFalse(task.done(), "single turn must survive disconnect")
        self.assertTrue(task._olv_detached)
        self.assertNotIn("conn-1", handler.current_conversation_tasks)
        # Still owned and discoverable after the dead slot was dropped:
        # otherwise the next turn on this session would race the orphan.
        self.assertIs(handler._detached_turns["hist-A"], task)
        release.set()
        await task
        self.assertEqual(finished, [True])
        # Released once the orphan finishes.
        await asyncio.sleep(0)
        self.assertNotIn("hist-A", handler._detached_turns)

    async def test_finished_orphan_resyncs_reconnected_socket(self):
        """The persisted answer must reach the socket that replaced the dead
        one, otherwise the UI keeps showing a truncated conversation."""
        history_uid = create_new_history(CONF)
        store_message(CONF, history_uid, "human", "good night my love")
        store_message(CONF, history_uid, "ai", "good night my love too")

        dead = FakeSocket()
        live = FakeSocket()
        handler = make_handler()
        dead_uid, live_uid = "conn-dead", "conn-live"
        handler.client_connections[dead_uid] = dead
        handler.client_connections[live_uid] = live
        handler.client_contexts[live_uid] = make_context(history_uid)
        handler.current_conversation_tasks[dead_uid] = asyncio.create_task(
            asyncio.sleep(0)
        )
        handler.current_conversation_tasks[dead_uid]._olv_single_turn = True
        handler.current_conversation_tasks[dead_uid]._olv_history_uid = history_uid
        with patch.object(wh_mod, "handle_client_disconnect", new=AsyncMock()):
            await handler.handle_disconnect(dead_uid)
        # Reconnect: new socket resumes the same history.
        handler._history_subscribers[live_uid] = history_uid
        await handler._deliver_history_to_subscriber(history_uid)
        payloads = [json.loads(p) for p in live.attempts]
        data = [p for p in payloads if p.get("type") == "history-data"]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["history_uid"], history_uid)
        self.assertEqual(
            [(m["role"], m["content"]) for m in data[0]["messages"]],
            [("human", "good night my love"), ("ai", "good night my love too")],
        )

    async def test_resync_skipped_when_viewer_left(self):
        """A dead or unbound viewer must never raise inside the callback."""
        handler = make_handler()
        handler.client_connections["conn-live"] = FakeSocket(fail_send=True)
        handler.client_contexts["conn-live"] = make_context("hist-A")
        handler._history_subscribers["conn-live"] = "hist-A"
        await handler._deliver_history_to_subscriber("hist-missing")
        # No subscriber for the unknown history: no send, no exception.
        handler._history_subscribers.pop("conn-live")
        await handler._deliver_history_to_subscriber("hist-A")

    async def test_disconnect_still_cancels_untagged_tasks(self):
        async def fake_turn():
            await asyncio.sleep(30)

        task = asyncio.create_task(fake_turn())
        handler = self._handler_with_task(task)
        with patch.object(wh_mod, "handle_client_disconnect", new=AsyncMock()):
            await handler.handle_disconnect("conn-1")
        for _ in range(10):
            await asyncio.sleep(0)
        self.assertTrue(task.done())
        self.assertTrue(task.cancelled())


class OrphanSerializationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        for folder in ("chat_history", "character_state", "episodic"):
            os.makedirs(folder, exist_ok=True)
    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _env(self):
        return {
            "client_uid": "conn-new",
            "context": SimpleNamespace(history_uid="hist-A"),
            "websocket": FakeSocket(),
            "client_contexts": {},
            "client_connections": {},
            "received_data_buffers": {},
            "current_conversation_tasks": {},
            "detached_turns": {},
            "broadcast_to_group": lambda *a, **k: asyncio.sleep(0),
        }

    async def _trigger(self, env, text, group, process):
        # The turn runs inside its own task, so the patch must outlive this
        # call: it is stopped once that task finishes.
        patcher = patch.object(
            ch_mod, "process_single_conversation", side_effect=process
        )
        patcher.start()
        # Cleanups run LIFO, so nested triggers restore the real function in
        # the right order. A done-callback is not enough: the loop may close
        # first and the mock would leak into the next test module.
        self.addCleanup(patcher.stop)
        try:
            await ch_mod.handle_conversation_trigger(
                msg_type="text-input",
                data={"type": "text-input", "text": text},
                client_uid=env["client_uid"],
                context=env["context"],
                websocket=env["websocket"],
                client_contexts=env["client_contexts"],
                client_connections=env["client_connections"],
                chat_group_manager=group,
                received_data_buffers=env["received_data_buffers"],
                current_conversation_tasks=env["current_conversation_tasks"],
                broadcast_to_group=env["broadcast_to_group"],
                detached_turns=env["detached_turns"],
            )
        finally:
            task = env["current_conversation_tasks"].get(env["client_uid"])
            if task is None:
                patcher.stop()

    async def test_trigger_queued_behind_orphan_survives_its_own_disconnect(self):
        """Regression: a message sent while an orphan turn is still finishing
        must not be dropped when that socket dies too.

        The orphan wait used to run BEFORE the turn task existed, so the
        pending trigger lived only in the dying connection's receive loop and
        the user's message vanished (observed live as a missing "reconnect-1").
        """
        ran = []

        async def orphan():
            await asyncio.sleep(0.05)

        async def turn(**kwargs):
            ran.append(kwargs["user_input"])

        orphan_task = asyncio.create_task(orphan())
        orphan_task._olv_single_turn = True
        orphan_task._olv_history_uid = "hist-A"
        orphan_task._olv_detached = True
        env = self._env()
        env["detached_turns"]["hist-A"] = orphan_task
        group = SimpleNamespace(get_client_group=lambda uid: None)

        # Trigger returns immediately (the task owns the orphan wait).
        await self._trigger(env, "reconnect-1", group, turn)
        queued = env["current_conversation_tasks"]["conn-new"]
        self.assertFalse(queued.done(), "turn must be queued behind the orphan")

        # The socket dies while the trigger is still queued behind the orphan.
        queued._olv_detached = True
        for _ in range(50):
            await asyncio.sleep(0.005)
        self.assertEqual(ran, ["reconnect-1"])
        await asyncio.wait_for(queued, timeout=5)

    async def test_turn_sends_are_fail_soft_on_a_dead_socket(self):
        """A turn that outlives its socket must never die on a failed send.

        Observed live: a turn queued behind an orphan resumed on a socket whose
        ASGI response was already closed, and the very first lifecycle send
        raised "Unexpected ASGI message 'websocket.send'", killing the turn
        before the assistant response was persisted.
        """
        seen = []

        class DeadSocket:
            async def send_text(self, payload):
                seen.append(payload)
                raise RuntimeError(
                    "Unexpected ASGI message 'websocket.send', after sending"
                    " 'websocket.close' or response already completed."
                )

        env = self._env()
        env["websocket"] = DeadSocket()

        async def turn(**kwargs):
            await kwargs["websocket_send"]('{"type": "latency-event"}')
            await kwargs["websocket_send"]('{"type": "conversation-end"}')
            return "persisted"

        await self._trigger(env, "halo", SimpleNamespace(
            get_client_group=lambda uid: None), turn)
        task = env["current_conversation_tasks"]["conn-new"]
        # No exception escapes: the turn completes despite the dead socket.
        self.assertEqual(await asyncio.wait_for(task, timeout=5), "persisted")
        self.assertEqual(len(seen), 2)

    async def test_new_trigger_waits_for_detached_orphan(self):
        """Same history, dead owner: the new turn starts only after the
        orphan finished persisting -> no parallel persists, no duplicates."""
        events = []

        async def orphan():
            await asyncio.sleep(0.05)
            events.append("orphan-done")

        async def new_turn(**kwargs):
            events.append("new-turn-started")

        orphan_task = asyncio.create_task(orphan())
        orphan_task._olv_single_turn = True
        orphan_task._olv_history_uid = "hist-A"
        orphan_task._olv_detached = True
        env = self._env()
        # The dead owner's slot is already gone: only the history-keyed
        # registry still owns the orphan (real disconnect behaviour).
        env["detached_turns"]["hist-A"] = orphan_task
        group = SimpleNamespace(get_client_group=lambda uid: None)
        await self._trigger(env, "lanjut", group, new_turn)
        new_task = env["current_conversation_tasks"]["conn-new"]
        self.assertTrue(getattr(new_task, "_olv_single_turn", False))
        self.assertEqual(getattr(new_task, "_olv_history_uid", ""), "hist-A")
        # Let the new turn run: it must start only after the orphan done.
        await asyncio.wait_for(new_task, timeout=5)
        self.assertEqual(events, ["orphan-done", "new-turn-started"])

    async def test_same_client_replacement_still_cancels(self):
        """The existing one-active-turn-per-client rule is untouched."""
        started = asyncio.Event()
        cancelled = []

        async def slow_turn(**kwargs):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async def fast_turn(**kwargs):
            return None

        env = self._env()
        group = SimpleNamespace(get_client_group=lambda uid: None)
        await self._trigger(env, "satu", group, slow_turn)
        await asyncio.wait_for(started.wait(), timeout=5)
        await self._trigger(env, "dua", group, fast_turn)
        # Cancellation is delivered asynchronously; pump until observed.
        for _ in range(100):
            if cancelled:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(cancelled, [True])
        await asyncio.wait_for(
            env["current_conversation_tasks"]["conn-new"], timeout=5
        )


class DeadSocketTtsTest(unittest.IsolatedAsyncioTestCase):
    async def test_sender_task_survives_dead_socket(self):
        """TTS delivery failure must not kill the turn: text persists."""
        manager = TTSTaskManager()
        manager._payload_queue.put_nowait(({"type": "audio"}, 0))
        await manager._process_payload_queue(FakeSocket(fail_send=True).send_text)
        # Returned (did not raise): the caller keeps its accumulated text.


if __name__ == "__main__":
    unittest.main()
