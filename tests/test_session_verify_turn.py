"""Turn survivability across socket death — verification only.

Runs the REAL process_single_conversation with a fake agent/LLM (no
provider, no network) against temp-dir history files. Proves the
in-flight turn persists its assistant response even when the socket
dies mid-turn, and that no duplicate assistant turn is produced.

Temp CWD per test: production history is never touched.
"""

import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.open_llm_vtuber.agent.output_types import Actions, DisplayText, SentenceOutput
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
)
from src.open_llm_vtuber.conversations import conversation_utils as cu_mod
from src.open_llm_vtuber.conversations.single_conversation import (
    process_single_conversation,
)
from src.open_llm_vtuber import websocket_handler as wh_mod

CONF = "verifyturn"


class FakeSocket:
    """Live socket that can die mid-turn, like a real disconnect."""

    def __init__(self):
        self.attempts = []
        self.dead = False

    async def send_text(self, payload):
        if self.dead:
            raise RuntimeError("socket is closed")
        self.attempts.append(payload)


def sentence(text):
    return SentenceOutput(
        display_text=DisplayText(text=text, name="Mili", avatar=""),
        tts_text=text,
        actions=Actions(),
    )


class FakeAgent:
    """Yields a fixed multi-sentence reply with pacing, like streaming."""

    def __init__(self, sentences, pace=0.02):
        self.sentences = list(sentences)
        self._memory = []

    async def chat(self, input_data):
        for text in self.sentences:
            await asyncio.sleep(0.02)
            yield sentence(text)


def make_context(history_uid, agent):
    return SimpleNamespace(
        history_uid=history_uid,
        user_timezone="Asia/Jakarta",
        voice_output_enabled=False,
        asr_engine=None,
        live2d_model=None,
        tts_engine=None,
        translate_engine=None,
        character_config=SimpleNamespace(
            conf_uid=CONF,
            character_name="Mili",
            avatar="",
            human_name="Human",
        ),
        agent_engine=agent,
    )


def transcript(history_uid):
    return [
        (m["role"], m["content"])
        for m in get_history(CONF, history_uid)
        if m["role"] in ("human", "ai")
    ]


class TurnSurvivalTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        for folder in ("chat_history", "character_state", "episodic"):
            os.makedirs(folder, exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    async def _run_turn(self, context, socket, text):
        async def _websocket_send(payload):
            await socket.send_text(payload)

        with patch.object(cu_mod, "PLAYBACK_COMPLETE_TIMEOUT_S", 0.05):
            return await process_single_conversation(
                context=context,
                websocket_send=_websocket_send,
                client_uid="verify-client",
                user_input=text,
                images=None,
                session_emoji="😊",
                metadata={},
            )

    async def test_2_disconnect_mid_turn_still_persists_answer(self):
        """The exact reported bug: 'good night my love' sent, socket dies
        mid-generation, no cancel (detach semantics) -> the answer must
        still land in Session A exactly once."""
        history_uid = create_new_history(CONF)
        socket = FakeSocket()
        agent = FakeAgent(
            ["Good night, sayang.", "Mimpi indah ya.", "Besok cerita lagi."]
        )
        context = make_context(history_uid, agent)

        async def kill_midway():
            await asyncio.sleep(0.05)
            socket.dead = True  # disconnect lands mid-stream; nobody cancels

        killer = asyncio.create_task(kill_midway())
        result = await self._run_turn(context, socket, "good night my love")
        await killer

        self.assertIn("Good night", result)
        rows = transcript(history_uid)
        self.assertEqual(rows[0], ("human", "good night my love"))
        ai_rows = [c for r, c in rows if r == "ai"]
        self.assertEqual(len(ai_rows), 1, f"exactly one answer: {rows}")
        self.assertIn("Good night", ai_rows[0])
        self.assertIn("Besok cerita lagi.", ai_rows[0])

    async def test_5_dead_socket_from_start_still_persists(self):
        """TTS/socket delivery failure must not cancel persistence.

        The socket dies after the text is generated (TTS/finalize phase):
        delivery fails, but the full answer must still land in history.
        """
        history_uid = create_new_history(CONF)
        socket = FakeSocket()
        agent = FakeAgent(["Aku di sini kok.", "Tenang ya."])
        context = make_context(history_uid, agent)

        async def kill_late():
            await asyncio.sleep(0.12)
            socket.dead = True

        killer = asyncio.create_task(kill_late())
        result = await self._run_turn(context, socket, "halo mili, kamu di sana?")
        await killer
        rows = transcript(history_uid)
        ai_rows = [c for r, c in rows if r == "ai"]
        self.assertEqual(len(ai_rows), 1)
        self.assertIn("Aku di sini", ai_rows[0])
        self.assertIn("Aku di sini", result)

    async def test_1_normal_turn_persists_pair_and_no_duplicates(self):
        history_uid = create_new_history(CONF)
        socket = FakeSocket()
        agent = FakeAgent(["Hai juga!"])
        context = make_context(history_uid, agent)
        await self._run_turn(context, socket, "halo")
        await self._run_turn(context, socket, "lagi apa?")
        rows = transcript(history_uid)
        self.assertEqual(
            [r for r, _ in rows], ["human", "ai", "human", "ai"]
        )


class FullLifecycleTest(unittest.IsolatedAsyncioTestCase):
    """TEST 4 / 6 / 7 / 8 / 9 on the real handler + real trigger path.

    Every test runs in a temp CWD, so production history, character_state
    and backups are never read or written.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        for folder in ("chat_history", "character_state", "episodic"):
            os.makedirs(folder, exist_ok=True)
        self._handler = None
        self._orig_timeout = cu_mod.PLAYBACK_COMPLETE_TIMEOUT_S
        cu_mod.PLAYBACK_COMPLETE_TIMEOUT_S = 0.02
        self._patches = [
            patch.object(
                wh_mod.WebSocketHandler,
                "_activate_proactive_for_history",
                new=AsyncMock(),
            ).start()
        ]

    def tearDown(self):
        cu_mod.PLAYBACK_COMPLETE_TIMEOUT_S = self._orig_timeout
        for p in self._patches:
            p.stop()
        os.chdir(self._old)
        self._tmp.cleanup()

    def _build_handler(self):
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
            get_client_group=lambda uid: None, client_group_map={}
        )
        self._handler = handler
        return handler

    def _connect(self, client_uid, context):
        socket = FakeSocket()
        self._handler.client_connections[client_uid] = socket
        self._handler.client_contexts[client_uid] = context
        return socket

    async def _send(self, client_uid, context, socket, text, history_uid):
        # The real entry point: orphan guard / session adoption / turn spawn.
        await self._handler._handle_conversation_trigger(
            socket,
            client_uid,
            {
                "type": "text-input",
                "text": text,
                "history_uid": history_uid,
            },
        )
        task = self._handler.current_conversation_tasks.get(client_uid)
        if task is not None:
            await asyncio.wait_for(task, timeout=15)

    async def _broadcast(self, *args, **kwargs):
        return None

    async def _detach(self, client_uid):
        with patch.object(wh_mod, "handle_client_disconnect", new=AsyncMock()):
            await self._handler.handle_disconnect(client_uid)

    def _history_files(self):
        folder = os.path.join("chat_history", CONF)
        if not os.path.isdir(folder):
            return []
        return sorted(f for f in os.listdir(folder) if f.endswith(".json"))

    async def test_4_6_8_9_reconnect_lifecycle(self):
        """One session across reconnects: immediate send, mid-turn
        disconnect, repeated reconnects, and a post-turn refresh."""
        self._build_handler()
        sentences = iter(
            [
                ["Selamat malam, sayang."],
                ["Tidur nyenyak ya."],
                ["Besok cerita lagi, ok?"],
            ]
        )

        def agent_for():
            return SimpleNamespace(
                _character_conf_uid=CONF,
                set_memory_from_history=(
                    lambda conf_uid, history_uid, user_timezone=None: None
                ),
                close=_noop,
                chat=_streaming_agent(sentences),
            )

        # --- initial session A -------------------------------------------
        conn_a = "conn-a"
        context_a = make_context("", agent_for())
        socket_a = self._connect(conn_a, context_a)
        history_a = create_new_history(CONF)
        context_a.history_uid = history_a

        # TEST 6: send carries the active uid on a socket whose context was
        # never restored -> must adopt, never mint a second session.
        await self._send(conn_a, context_a, socket_a, "good night my love", history_a)
        self.assertEqual(context_a.history_uid, history_a)
        self.assertEqual(len(self._history_files()), 1)

        # TEST 4 + TEST 8: start a turn, drop the socket, then reconnect and
        # send again while the orphan is still running.
        conn_b = "conn-b"
        context_b = make_context("", agent_for())
        socket_b = self._connect(conn_b, context_b)
        turn_text = "aku mau tidur"
        slow = asyncio.create_task(
            self._send_later(context_b, socket_b, turn_text, history_a)
        )
        # let the turn register itself as in-flight
        for _ in range(200):
            await asyncio.sleep(0.005)
            task = self._handler.current_conversation_tasks.get(conn_b)
            if task is not None and not task.done():
                break
        await self._detach(conn_b)
        self.assertTrue(task._olv_detached)

        # reconnect twice, sending in between, still no second session
        conn_c = "conn-c"
        context_c = make_context("", agent_for())
        socket_c = self._connect(conn_c, context_c)
        await self._send(conn_c, context_c, socket_c, "still there?", history_a)
        await self._detach(conn_c)
        conn_d = "conn-d"
        context_d = make_context("", agent_for())
        socket_d = self._connect(conn_d, context_d)
        await self._send(conn_d, context_d, socket_d, "good morning", history_a)
        await slow

        self.assertEqual(len(self._history_files()), 1, "no Session B was created")
        rows = transcript(history_a)
        self.assertEqual(
            [r for r, _ in rows],
            ["human", "ai", "human", "ai", "human", "ai", "human", "ai"],
            f"one ordered pair per send, no duplicates: {rows}",
        )
        self.assertEqual(
            [c for r, c in rows if r == "human"],
            ["good night my love", "aku mau tidur", "still there?", "good morning"],
        )

        # TEST 9: refresh after the turn finished restores the same session
        # through the real fetch-and-set-history path.
        conn_e = "conn-e"
        context_e = make_context("", agent_for())
        socket_e = self._connect(conn_e, context_e)
        await self._handler._handle_fetch_history(
            socket_e, conn_e, {"type": "fetch-and-set-history", "history_uid": history_a}
        )
        self.assertEqual(context_e.history_uid, history_a)
        self.assertEqual(self._handler._history_subscribers[conn_e], history_a)
        restored = [
            json.loads(p)
            for p in socket_e.attempts
            if json.loads(p).get("type") == "history-data"
        ]
        self.assertEqual(len(restored), 1)
        self.assertEqual(
            [(m["role"], m["content"]) for m in restored[0]["messages"]],
            rows,
        )
        self.assertEqual(len(self._history_files()), 1)

    async def _send_later(self, context, socket, text, history_uid):
        await self._send("conn-b", context, socket, text, history_uid)

    async def test_7_explicit_new_conversation_still_creates_session_b(self):
        """TEST 7: reconnect logic must not break explicit new sessions."""
        self._build_handler()
        sentences = iter([["Hai."], ["Halo lagi."]])

        def agent_for():
            return SimpleNamespace(
                _character_conf_uid=CONF,
                set_memory_from_history=(
                    lambda conf_uid, history_uid, user_timezone=None: None
                ),
                close=_noop,
                chat=_streaming_agent(sentences),
            )

        conn_a = "conn-a"
        context_a = make_context("", agent_for())
        socket_a = self._connect(conn_a, context_a)
        history_a = create_new_history(CONF)
        context_a.history_uid = history_a
        await self._send(conn_a, context_a, socket_a, "di session A", history_a)

        conn_b = "conn-b"
        context_b = make_context("", agent_for())
        socket_b = self._connect(conn_b, context_b)
        history_b = create_new_history(CONF)
        context_b.history_uid = history_b
        await self._send(conn_b, context_b, socket_b, "di session B", history_b)

        self.assertNotEqual(history_a, history_b)
        self.assertEqual(len(self._history_files()), 2)
        self.assertEqual(
            [c for _, c in transcript(history_a)], ["di session A", "Hai."]
        )
        self.assertEqual(
            [c for _, c in transcript(history_b)], ["di session B", "Halo lagi."]
        )


async def _noop():
    return None


def _streaming_agent(sentences):
    """Agent that streams the next canned reply slowly enough to interrupt."""

    async def chat(input_data):
        batch = next(sentences, ["..."])
        for text in batch:
            await asyncio.sleep(0.15)
            yield sentence(text)

    return chat


if __name__ == "__main__":
    unittest.main()
