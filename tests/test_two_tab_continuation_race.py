"""Two tabs on ONE session: continuation, turn ownership, ACK, completion.

The earlier hand-off suite pinned the pre-fix behaviour on purpose: two tabs
that reached the row threshold at the same time each created their OWN
continuation, splitting one conversation into two siblings. That is the race
this file turns into a requirement:

* one source session has ONE continuation; every live tab on the source is
  moved to it, not to a private sibling;
* a second tab's active turn or unsettled delivery blocks the cut until that
  work is finished -- a session may not be continued under another tab;
* reconnecting while the cut is in progress neither creates a second session
  nor writes to the source;
* two turns on the same history never persist concurrently; the second waits
  for the first instead of being silently dropped or interleaved;
* a duplicate ACK is idempotent, and a receipt from the other tab never
  settles the owner's message;
* a playback completion from another tab never settles this tab's turn, and a
  consumed duplicate never settles the next turn;
* a finished detached turn reaches every tab currently viewing that history,
  not just the first subscriber.

Everything drives the real handler, the real transcripts and the real delivery
store. Only the provider transport is faked.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import src.open_llm_vtuber.websocket_handler as ws_module
from src.open_llm_vtuber import conversation_handoff as handoff
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.chat_group import ChatGroupManager
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    get_metadata,
    resumable_continuations,
    store_message,
    update_summary_metadata,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.conversations import conversation_handler as ch_module
from src.open_llm_vtuber.message_delivery import (
    ACK_RECEIVED,
    REASON_DUPLICATE,
    REASON_OWNER_MISMATCH,
    apply_ack,
    begin_attempt,
    delivery_record,
)
from src.open_llm_vtuber.message_handler import MessageHandler

CONF = "twotabraces"
SUMMARY = "Mili dan user membahas rencana akhir pekan."


class _FakeLLM:
    """Production agent, fake transport: no provider call is ever made."""

    model = "two-tab-race"
    max_tokens = 100

    async def chat_completion(self, messages, system=None, tools=None):
        yield "oke."


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class _Socket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def make_agent() -> BasicMemoryAgent:
    return BasicMemoryAgent(
        llm=_FakeLLM(),
        system="persona Mili",
        live2d_model=_FakeLive2D(),
        tts_preprocessor_config=TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
        ),
    )


def session_path(history_uid: str) -> str:
    return os.path.join("chat_history", CONF, f"{history_uid}.json")


def file_state(history_uid: str) -> tuple[bytes, int]:
    with open(session_path(history_uid), "rb") as handle:
        payload = handle.read()
    return payload, os.stat(session_path(history_uid)).st_mtime_ns


def transcript_files() -> list[str]:
    return sorted(
        name
        for name in os.listdir(os.path.join("chat_history", CONF))
        if name.endswith(".json")
    )


def make_context(history_uid: str) -> SimpleNamespace:
    agent = make_agent()
    context = SimpleNamespace(
        history_uid=history_uid,
        user_timezone="Asia/Jakarta",
        character_config=SimpleNamespace(conf_uid=CONF),
        agent_engine=agent,
    )
    agent.set_memory_from_history(
        conf_uid=CONF, history_uid=history_uid, user_timezone="Asia/Jakarta"
    )
    return context


def make_handler(contexts: dict, sockets: dict) -> object:
    handler = ws_module.WebSocketHandler.__new__(ws_module.WebSocketHandler)
    handler.client_contexts = contexts
    handler.client_connections = sockets
    handler.chat_group_manager = ChatGroupManager()
    handler.current_conversation_tasks = {}
    handler._detached_turns = {}
    handler._history_subscribers = {}
    handler.received_data_buffers = {}
    handler.client_protocols = {}
    handler._proactive_timer_tasks = {}
    handler.broadcast_to_group = AsyncMock()
    handler._cancel_proactive_timer = AsyncMock()
    handler._activate_proactive_for_history = AsyncMock()
    return handler


class TwoTabContinuationRaceTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mili-two-tab-race-")
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        self._tasks: list[asyncio.Task] = []

        self.source_uid = create_new_history(CONF)
        for index, (role, text) in enumerate(
            [
                ("human", "halo Mili"),
                ("ai", "Halo! Ada yang bisa dibantu?"),
                ("human", "lanjut cerita kemarin yuk"),
                ("ai", "Boleh. Kita sampai mana tadi?"),
            ]
        ):
            self.assertTrue(
                store_message(
                    CONF, self.source_uid, role, text, msg_id=f"src-{index}"
                )
            )
        self.assertTrue(
            update_summary_metadata(
                CONF,
                self.source_uid,
                expected_summarized_through=0,
                conversation_summary=SUMMARY,
                summarized_through=4,
                summary_updated_at="2026-10-28T00:00:00+00:00",
            )
        )
        self.source_bytes, self.source_mtime = file_state(self.source_uid)

        self.context_a = make_context(self.source_uid)
        self.context_b = make_context(self.source_uid)
        self.socket_a = _Socket()
        self.socket_b = _Socket()
        self.handler = make_handler(
            {"tab-a": self.context_a, "tab-b": self.context_b},
            {"tab-a": self.socket_a, "tab-b": self.socket_b},
        )

    async def asyncTearDown(self) -> None:
        for task in self._tasks:
            if not task.done():
                task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    # -- helpers -----------------------------------------------------------

    def _enabled(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.dict(os.environ, {handoff.AUTO_HANDOFF_ENV: "1"}))
        stack.enter_context(patch.object(handoff, "HANDOFF_ROWS", 2))
        return stack

    def _assert_source_untouched(self) -> None:
        self.assertEqual(
            file_state(self.source_uid),
            (self.source_bytes, self.source_mtime),
            "the source transcript must stay byte-identical",
        )

    async def _sleeping_turn(self, history_uid: str) -> asyncio.Task:
        task = asyncio.create_task(asyncio.sleep(30))
        task._olv_history_uid = history_uid
        self._tasks.append(task)
        await asyncio.sleep(0)
        return task

    def _frames(self, socket: _Socket, frame_type: str) -> list[dict]:
        return [frame for frame in socket.sent if frame.get("type") == frame_type]

    # -- 1. two tabs, one continuation -------------------------------------

    async def test_concurrent_handoff_from_two_tabs_creates_one_continuation(
        self,
    ):
        with self._enabled():
            await asyncio.gather(
                self.handler._consider_handoff("tab-a", self.source_uid),
                self.handler._consider_handoff("tab-b", self.source_uid),
            )

        uid_a = self.context_a.history_uid
        uid_b = self.context_b.history_uid
        self.assertEqual(uid_a, uid_b, "both tabs must continue the SAME session")
        self.assertNotEqual(uid_a, self.source_uid)
        self.assertEqual(
            len(transcript_files()),
            2,
            "one source + exactly one continuation; no sibling transcript",
        )
        self.assertEqual(
            [frame["history_uid"] for frame in self._frames(self.socket_a, "new-history-created")],
            [uid_a],
        )
        self.assertEqual(
            [frame["history_uid"] for frame in self._frames(self.socket_b, "new-history-created")],
            [uid_a],
        )
        self.assertEqual(
            [frame["continued_from"] for frame in self._frames(self.socket_a, "new-history-created")],
            [self.source_uid],
        )
        self.assertEqual(get_metadata(CONF, uid_a)["continued_from"], self.source_uid)
        self.assertEqual(
            [item["uid"] for item in resumable_continuations(CONF)],
            [uid_a],
        )
        self._assert_source_untouched()

    async def test_second_tab_reuses_the_same_continuation_after_the_first_moved(
        self,
    ):
        with self._enabled():
            await self.handler._consider_handoff("tab-a", self.source_uid)
        uid = self.context_a.history_uid
        with self._enabled():
            await self.handler._consider_handoff("tab-b", self.source_uid)

        self.assertEqual(self.context_b.history_uid, uid)
        self.assertEqual(len(transcript_files()), 2)
        self.assertEqual(
            [frame["history_uid"] for frame in self._frames(self.socket_b, "new-history-created")],
            [uid],
        )
        self._assert_source_untouched()

    async def test_a_later_tab_opening_the_source_joins_the_existing_continuation(
        self,
    ):
        """The old transcript stays readable; opening it must not fork it.

        A tab can open the frozen source after the conversation already moved
        (it is still listed and restorable). When it reaches the threshold it
        must join the continuation that exists, not mint a sibling target.
        """
        with self._enabled():
            await self.handler._consider_handoff("tab-a", self.source_uid)
        uid = self.context_a.history_uid
        self.assertNotEqual(uid, self.source_uid)

        late_context = make_context(self.source_uid)
        late_socket = _Socket()
        self.handler.client_contexts["tab-c"] = late_context
        self.handler.client_connections["tab-c"] = late_socket
        with self._enabled():
            await self.handler._consider_handoff("tab-c", self.source_uid)

        self.assertEqual(late_context.history_uid, uid)
        self.assertEqual(len(transcript_files()), 2, "no sibling continuation")
        self.assertEqual(
            [frame["history_uid"] for frame in self._frames(late_socket, "new-history-created")],
            [uid],
        )
        self._assert_source_untouched()

    # -- 2. another tab's work blocks the cut ------------------------------

    async def test_another_tabs_active_turn_blocks_handoff_until_it_finishes(
        self,
    ):
        active = await self._sleeping_turn(self.source_uid)
        self.handler.current_conversation_tasks["tab-a"] = active

        with self._enabled():
            await self.handler._consider_handoff("tab-b", self.source_uid)
        self.assertEqual(
            self.context_b.history_uid,
            self.source_uid,
            "a cut under tab-a's active turn would split that turn",
        )
        self.assertEqual(len(transcript_files()), 1)

        active.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await active
        with self._enabled():
            await self.handler._consider_handoff("tab-b", self.source_uid)

        self.assertEqual(self.context_a.history_uid, self.context_b.history_uid)
        self.assertNotEqual(self.context_a.history_uid, self.source_uid)
        self.assertEqual(len(transcript_files()), 2)
        self._assert_source_untouched()

    async def test_another_tabs_unsettled_delivery_blocks_the_cut(self):
        self.assertTrue(
            begin_attempt(
                CONF,
                "msg-pending",
                turn_id="turn-pending",
                owner="tab-a",
                history_uid=self.source_uid,
            )
        )
        with self._enabled():
            await self.handler._consider_handoff("tab-b", self.source_uid)
        self.assertEqual(self.context_b.history_uid, self.source_uid)
        self.assertEqual(len(transcript_files()), 1)

        ack = apply_ack(
            CONF,
            "msg-pending",
            turn_id="turn-pending",
            attempt=1,
            status=ACK_RECEIVED,
            seq=0,
            final=True,
            owner="tab-a",
            active_owners=frozenset({"tab-a", "tab-b"}),
        )
        self.assertTrue(ack.applied)

        with self._enabled():
            await self.handler._consider_handoff("tab-b", self.source_uid)
        self.assertEqual(self.context_a.history_uid, self.context_b.history_uid)
        self.assertNotEqual(self.context_b.history_uid, self.source_uid)
        self.assertEqual(len(transcript_files()), 2)
        self._assert_source_untouched()

    # -- 3. reconnect while the cut is in progress -------------------------

    async def test_reconnect_during_the_cut_joins_the_single_continuation(
        self,
    ):
        active = await self._sleeping_turn(self.source_uid)
        self.handler.current_conversation_tasks["tab-a"] = active

        # tab-b still triggers the evaluation, but it must defer.
        with self._enabled():
            await self.handler._consider_handoff("tab-b", self.source_uid)
        self.assertEqual(self.context_b.history_uid, self.source_uid)

        # A reconnecting tab fetches the source while tab-a's turn is alive.
        context_c = make_context(self.source_uid)
        self.context_c = context_c
        socket_c = _Socket()
        self.handler.client_contexts["tab-c"] = context_c
        self.handler.client_connections["tab-c"] = socket_c
        await self.handler._handle_fetch_history(
            socket_c,
            "tab-c",
            {"history_uid": self.source_uid, "timezone": "Asia/Jakarta"},
        )
        self.assertEqual(self.context_c.history_uid, self.source_uid)
        self.assertEqual(len(transcript_files()), 1, "fetch creates no session")

        active.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await active
        with self._enabled():
            await self.handler._consider_handoff("tab-b", self.source_uid)

        uid = self.context_b.history_uid
        self.assertNotEqual(uid, self.source_uid)
        self.assertEqual(self.context_a.history_uid, uid)
        self.assertEqual(self.context_c.history_uid, uid)
        self.assertEqual(len(transcript_files()), 2)
        for socket in (self.socket_a, self.socket_b, socket_c):
            self.assertEqual(
                [frame["history_uid"] for frame in self._frames(socket, "new-history-created")],
                [uid],
            )
        self._assert_source_untouched()

    # -- 4. two simultaneous requests on one history -----------------------

    async def test_two_tabs_sending_at_once_serialize_on_the_same_history(self):
        started: list[str] = []
        release = asyncio.Event()

        async def stub(**kwargs) -> str:
            started.append(kwargs["client_uid"])
            if kwargs["client_uid"] == "tab-a":
                await release.wait()
            return "ok"

        async def trigger(client_uid: str) -> None:
            await ch_module.handle_conversation_trigger(
                msg_type="text-input",
                data={"type": "text-input", "text": f"halo dari {client_uid}"},
                client_uid=client_uid,
                context=self.handler.client_contexts[client_uid],
                websocket=self.handler.client_connections[client_uid],
                client_contexts=self.handler.client_contexts,
                client_connections=self.handler.client_connections,
                chat_group_manager=self.handler.chat_group_manager,
                received_data_buffers={},
                current_conversation_tasks=self.handler.current_conversation_tasks,
                detached_turns=self.handler._detached_registry(),
                broadcast_to_group=AsyncMock(),
            )

        with patch.object(ch_module, "process_single_conversation", new=stub):
            await trigger("tab-a")
            first = self.handler.current_conversation_tasks["tab-a"]
            await trigger("tab-b")
            second = self.handler.current_conversation_tasks["tab-b"]
            self._tasks.extend([first, second])

            await asyncio.sleep(0.05)
            self.assertEqual(
                started,
                ["tab-a"],
                "the second tab must wait, not persist into the same history",
            )
            self.assertFalse(second.done())

            release.set()
            await asyncio.wait_for(asyncio.gather(first, second), timeout=2)

        self.assertEqual(started, ["tab-a", "tab-b"])
        self.assertEqual(
            [frame["text"] for frame in self._frames(self.socket_b, "control")],
            ["conversation-turn-queued"],
        )

    # -- 5. duplicate ACK / completion and old-vs-new turns ----------------

    async def test_duplicate_ack_from_two_tabs_settles_the_owner_once(self):
        self.assertTrue(
            begin_attempt(
                CONF,
                "msg-duplicate",
                turn_id="turn-duplicate",
                owner="tab-a",
                history_uid=self.source_uid,
            )
        )
        ack = {
            "type": "message-ack",
            "msg_id": "msg-duplicate",
            "turn_id": "turn-duplicate",
            "attempt": 1,
            "status": ACK_RECEIVED,
            "seq": 0,
            "final": True,
        }
        for _ in range(2):
            await self.handler._handle_message_ack(self.socket_a, "tab-a", dict(ack))
        first = delivery_record(CONF, "msg-duplicate")
        assert first is not None
        self.assertEqual(first["status"], "delivered")

        # The other tab is not the owner and may not settle it either.
        foreign = dict(ack)
        await self.handler._handle_message_ack(self.socket_b, "tab-b", foreign)
        second = delivery_record(CONF, "msg-duplicate")
        self.assertEqual(second, first, "a foreign receipt must change nothing")
        self.assertEqual(
            [frame for frame in self.socket_a.sent if frame.get("type") == "message-ack-result"],
            [],
            "a duplicate/foreign receipt is a normal race, not a store failure",
        )

        # The low-level outcome names the real reasons (the handler only
        # reports store failures, so check the decision itself).
        self.assertEqual(
            apply_ack(
                CONF,
                "msg-duplicate",
                turn_id="turn-duplicate",
                attempt=1,
                status=ACK_RECEIVED,
                seq=0,
                final=True,
                owner="tab-a",
                active_owners=frozenset({"tab-a", "tab-b"}),
            ).reason,
            REASON_DUPLICATE,
        )
        self.assertEqual(
            apply_ack(
                CONF,
                "msg-duplicate",
                turn_id="turn-duplicate",
                attempt=1,
                status=ACK_RECEIVED,
                seq=0,
                final=True,
                owner="tab-b",
                active_owners=frozenset({"tab-a", "tab-b"}),
            ).reason,
            REASON_OWNER_MISMATCH,
        )

    async def test_playback_completion_is_owned_by_one_tab_and_consumed_once(
        self,
    ):
        handler = MessageHandler()
        waiter = asyncio.create_task(
            handler.wait_for_response(
                "tab-a",
                "frontend-playback-complete",
                request_id="turn-new",
                timeout=1,
                accept_untagged_legacy=True,
            )
        )
        await asyncio.sleep(0)

        # Another tab's completion, and this tab's OLD completion, must not
        # release the new turn.
        handler.handle_message(
            "tab-b",
            {"type": "frontend-playback-complete", "request_id": "turn-new"},
        )
        handler.handle_message(
            "tab-a",
            {"type": "frontend-playback-complete", "request_id": "turn-old"},
        )
        await asyncio.sleep(0.01)
        self.assertFalse(waiter.done())

        handler.handle_message(
            "tab-a",
            {"type": "frontend-playback-complete", "request_id": "turn-new"},
        )
        response = await waiter
        self.assertEqual(response["request_id"], "turn-new")

        # A duplicate left behind is consumed; the NEXT turn stays untouched.
        handler.handle_message(
            "tab-a",
            {"type": "frontend-playback-complete", "request_id": "turn-new"},
        )
        later = asyncio.create_task(
            handler.wait_for_response(
                "tab-a",
                "frontend-playback-complete",
                request_id="turn-next",
                timeout=0.1,
                accept_untagged_legacy=True,
            )
        )
        self.assertIsNone(
            await later, "the duplicate must never settle turn-next"
        )

    # -- 6. resync reaches every tab on the history ------------------------

    async def test_resync_reaches_every_tab_viewing_the_same_history(self):
        self.handler._subscriber_registry()["tab-a"] = self.source_uid
        self.handler._subscriber_registry()["tab-b"] = self.source_uid
        await self.handler._deliver_history_to_subscriber(self.source_uid)

        for socket in (self.socket_a, self.socket_b):
            data = self._frames(socket, "history-data")
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["history_uid"], self.source_uid)
            self.assertEqual(
                [row["msg_id"] for row in data[0]["messages"]],
                [f"src-{index}" for index in range(4)],
            )


if __name__ == "__main__":
    unittest.main()
