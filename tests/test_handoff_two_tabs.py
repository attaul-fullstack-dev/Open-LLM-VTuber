"""Two-tab behaviour of the cross-transcript hand-off (Stage 4.9).

Stage 4.7 pinned the single-tab contract and one weak two-tab case ("repeated
history requests must not create sessions"). Stage 4.8 pinned the read side.
The remaining risk was operational: what happens when TWO tabs are open on the
SAME source session and the hand-off fires?

This file drives the real handler, the real transcripts and the real delivery
store with two live connections, and pins what happens:

* near-simultaneous hand-off coordinates: ONE continuation is created for the
  source and both tabs are moved into it. No message is lost or duplicated,
  and the conversation does not split into sibling sessions;
* every live tab on the source moves together, so neither keeps writing to a
  transcript that has already been continued;
* a disconnected tab can never move (``no-live-connection``), while the live
  tab still can;
* reconnecting onto the labelled-but-empty target resumes it (`resumable_sessions`)
  without creating sessions or touching the source;
* late/duplicate receipts after the move do not schedule a second hand-off;
* a cancelled turn in one tab is not a boundary, but another tab's finalized
  turn moves BOTH idle tabs into the same continuation.

Deliberately not covered here: delivery ownership (see
``tests/test_delivery_retry_reconnect.py``), the read-side payloads (see
``tests/test_handoff_ui_contract.py``) and rendering (frontend tests).
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
    get_history_list,
    get_metadata,
    resumable_continuations,
    store_message,
    update_summary_metadata,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.message_delivery import (
    ACK_RECEIVED,
    apply_ack,
    begin_attempt,
    unsettled_messages,
)

CONF = "twotabchar"
SUMMARY = "Mili dan user membahas rencana akhir pekan."


class _FakeLLM:
    """Production agent, fake transport: no provider call is ever made."""

    model = "two-tab-test"
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
    handler.broadcast_to_group = AsyncMock()
    handler._cancel_proactive_timer = AsyncMock()
    handler._activate_proactive_for_history = AsyncMock()
    return handler


class TwoTabHandoffTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mili-two-tabs-")
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
        """Opt into automatic hand-off for one test, with a low threshold."""

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

    async def _finished_task(self) -> asyncio.Task:
        task = asyncio.create_task(asyncio.sleep(0))
        self._tasks.append(task)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertTrue(task.done())
        return task

    async def _cancelled_task(self) -> asyncio.Task:
        task = asyncio.create_task(asyncio.sleep(30))
        self._tasks.append(task)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self.assertTrue(task.cancelled())
        return task

    async def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.01)
        return predicate()

    # -- 1. near-simultaneous hand-off from two tabs ----------------------

    async def test_concurrent_handoff_from_two_tabs_creates_one_shared_continuation(
        self,
    ):
        """One source, one continuation: both tabs converge on the same target.

        A cut is a property of the HISTORY, not of the connection that noticed
        the threshold. Two tabs evaluating it together must not split the
        conversation into siblings.
        """
        with self._enabled():
            await asyncio.gather(
                self.handler._consider_handoff("tab-a", self.source_uid),
                self.handler._consider_handoff("tab-b", self.source_uid),
            )

        uid_a = self.context_a.history_uid
        uid_b = self.context_b.history_uid
        self.assertEqual(uid_a, uid_b, "both tabs must share one continuation")
        self.assertNotEqual(uid_a, self.source_uid)

        metadata = get_metadata(CONF, uid_a)
        self.assertEqual(metadata["continued_from"], self.source_uid)
        self.assertEqual(get_history(CONF, uid_a), [])
        self.assertEqual(len(transcript_files()), 2)

        # Each tab was told about the SAME continuation, never a sibling.
        self.assertEqual(
            [frame["history_uid"] for frame in self.socket_a.sent],
            [uid_a],
        )
        self.assertEqual(
            [frame["history_uid"] for frame in self.socket_b.sent],
            [uid_a],
        )
        self.assertEqual(
            [frame["continued_from"] for frame in self.socket_a.sent],
            [self.source_uid],
        )
        self.assertEqual(
            [frame["continued_from"] for frame in self.socket_b.sent],
            [self.source_uid],
        )

        self._assert_source_untouched()
        self.assertEqual(len(get_history(CONF, self.source_uid)), 4)
        # The single empty target waits for its tabs; it is not offered as a
        # conversation, and the source stays the only listed one.
        self.assertEqual(
            [item["uid"] for item in resumable_continuations(CONF)],
            [uid_a],
        )
        self.assertEqual(
            [item["uid"] for item in get_history_list(CONF, protect_uids=())],
            [self.source_uid],
        )

    # -- 2. both tabs move; the source is frozen after the cut ------------

    async def test_both_tabs_move_to_the_continuation_and_stop_writing_the_source(
        self,
    ):
        with self._enabled():
            await self.handler._consider_handoff("tab-a", self.source_uid)
        continuation_uid = self.context_a.history_uid
        self.assertNotEqual(continuation_uid, self.source_uid)
        # A cut belongs to the history: the second live tab moves with it.
        self.assertEqual(self.context_b.history_uid, continuation_uid)

        # The next message from either tab lands in the continuation; the
        # source stays byte-identical for later reading/restoring.
        self.assertTrue(
            store_message(
                CONF,
                continuation_uid,
                "human",
                "pesan dari tab b",
                msg_id="tab-b-1",
            )
        )
        self.assertEqual(len(get_history(CONF, self.source_uid)), 4)
        self.assertEqual(
            [row["content"] for row in get_history(CONF, continuation_uid)],
            ["pesan dari tab b"],
        )
        # The continuation still carries a read-only view of the source it
        # continues; reading it mutates nothing.
        view = handoff.continuation_view(CONF, continuation_uid)
        assert view is not None
        self.assertEqual(view["continued_from"], self.source_uid)
        self.assertEqual(view["context_total"], 4)
        self.assertEqual(
            [row["msg_id"] for row in get_history(CONF, self.source_uid)],
            [f"src-{index}" for index in range(4)],
        )
        self._assert_source_untouched()

    # -- 3. a disconnected tab cannot move; the live one still can --------

    async def test_offline_tab_cannot_hand_off_while_the_live_tab_still_does(
        self,
    ):
        self.handler.client_connections.pop("tab-a")
        with self._enabled():
            await self.handler._consider_handoff("tab-a", self.source_uid)
        self.assertEqual(self.context_a.history_uid, self.source_uid)
        self.assertEqual(transcript_files(), [os.path.basename(session_path(self.source_uid))])
        self.assertEqual(self.socket_a.sent, [])

        with self._enabled():
            await self.handler._consider_handoff("tab-b", self.source_uid)
        self.assertNotEqual(self.context_b.history_uid, self.source_uid)
        self.assertEqual(len(transcript_files()), 2)
        self._assert_source_untouched()

    # -- 4. reconnect onto the labelled-but-empty target ------------------

    async def test_reconnect_resumes_the_empty_target_without_new_sessions(
        self,
    ):
        with self._enabled():
            await self.handler._consider_handoff("tab-a", self.source_uid)
        continuation_uid = self.context_a.history_uid

        # tab-b (still on the source) asks for the list: the empty target is
        # reported apart from the conversations, and only its own tab may
        # adopt it (it is the one that remembers the uid).
        list_socket = _Socket()
        await self.handler._handle_history_list_request(list_socket, "tab-b", {})
        self.assertEqual(len(list_socket.sent), 1)
        listing = list_socket.sent[0]
        self.assertEqual(listing["type"], "history-list")
        self.assertEqual(
            [item["uid"] for item in listing["histories"]], [self.source_uid]
        )
        self.assertEqual(
            [item["uid"] for item in listing["resumable_sessions"]],
            [continuation_uid],
        )

        reconnected_context = make_context(self.source_uid)
        self.handler.client_contexts["tab-c"] = reconnected_context
        self.handler.client_connections["tab-c"] = _Socket()
        socket = _Socket()
        await self.handler._handle_fetch_history(
            socket,
            "tab-c",
            {"history_uid": continuation_uid, "timezone": "Asia/Jakarta"},
        )

        frames = [frame for frame in socket.sent if frame.get("type") == "history-data"]
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(frame["history_uid"], continuation_uid)
        self.assertEqual(frame["continued_from"], self.source_uid)
        self.assertEqual(frame["messages"], [])
        self.assertEqual(
            [row["msg_id"] for row in frame["context_messages"]],
            [f"src-{index}" for index in range(4)],
        )
        self.assertEqual(len(transcript_files()), 2)
        self._assert_source_untouched()

    # -- 5. late / duplicate receipts after the move ----------------------

    async def test_late_and_duplicate_receipts_do_not_create_a_second_continuation(
        self,
    ):
        begin_attempt(
            CONF,
            "msg-settled",
            turn_id="turn-1",
            owner="tab-a",
            history_uid=self.source_uid,
        )
        for _ in range(2):
            apply_ack(
                CONF,
                "msg-settled",
                turn_id="turn-1",
                attempt=1,
                status=ACK_RECEIVED,
                seq=0,
                final=True,
                owner="tab-a",
            )

        with self._enabled():
            await self.handler._consider_handoff("tab-a", self.source_uid)
        continuation_uid = self.context_a.history_uid
        self.assertNotEqual(continuation_uid, self.source_uid)
        files_after_move = transcript_files()

        # The receipt for the OLD history arrives after the move (twice): it
        # must not schedule another hand-off for a history this tab left.
        with self._enabled():
            self.handler._on_delivery_settled("tab-a", self.source_uid)
            self.handler._on_delivery_settled("tab-a", self.source_uid)
            await asyncio.sleep(0.05)

        self.assertEqual(transcript_files(), files_after_move)
        self.assertEqual(self.context_a.history_uid, continuation_uid)
        self.assertEqual(
            unsettled_messages(CONF, owner="tab-a", limit=-1).messages, []
        )
        self._assert_source_untouched()

    # -- 6. a failed turn is not a boundary; a finalized one moves both ----

    async def test_cancelled_turn_in_one_tab_defers_until_a_finalized_cut(
        self,
    ):
        cancelled = await self._cancelled_task()
        with self._enabled():
            self.handler._on_turn_completed("tab-a", self.source_uid, cancelled)
            await asyncio.sleep(0.05)
        # A cancelled turn is not a turn boundary: neither tab may move.
        self.assertEqual(self.context_a.history_uid, self.source_uid)
        self.assertEqual(self.context_b.history_uid, self.source_uid)
        self.assertEqual(len(transcript_files()), 1)

        finished = await self._finished_task()
        with self._enabled():
            self.handler._on_turn_completed("tab-b", self.source_uid, finished)
            switched = await self._wait_until(
                lambda: self.context_a.history_uid != self.source_uid
            )
        self.assertTrue(switched, "the finalized turn should have handed off")
        # One continuation for the history: tab-a moves with tab-b even though
        # its own last turn was cancelled.
        self.assertEqual(
            self.context_a.history_uid, self.context_b.history_uid
        )
        self.assertNotEqual(self.context_a.history_uid, self.source_uid)
        self.assertEqual(len(transcript_files()), 2)
        self._assert_source_untouched()


if __name__ == "__main__":
    unittest.main()
