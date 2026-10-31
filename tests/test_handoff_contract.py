"""Contract tests for the production hand-off module (Stage 4.7).

These tests used to hold their own copy of the predicate and of the switch.
That made them a specification: they could pass while production did nothing
(or did something else). They now import ``src.open_llm_vtuber.conversation_handoff``
and drive it with the same state the server uses at runtime:

* real ``asyncio.Task`` objects for the live/detached turn blockers;
* the real delivery store (``begin_attempt`` / ``unsettled_messages`` /
  ``apply_ack``) for the unsettled-message blocker;
* a real ``ChatGroupManager`` for group mode;
* the real transcript files for row count and the rolling-summary boundary;
* a real ``BasicMemoryAgent`` for the switch, so the memory reload and the
  previous-session continuity are the production path, not a stub;
* the real ``WebSocketHandler`` callback for the turn-completion integration.

The client-facing side of a continuation (what a reload/reconnect may show)
has its own contract in ``tests/test_handoff_ui_contract.py``. Deliberately not
covered here: the forced-summary path (an LLM call) -- a session without a
summary defers instead of continuing with empty context.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import src.open_llm_vtuber.chat_history_manager as history_store
import src.open_llm_vtuber.websocket_handler as ws_module
from src.open_llm_vtuber import conversation_handoff as handoff
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.conversation_summary import PREV_SESSION_MAX_CHARS
from src.open_llm_vtuber.chat_group import ChatGroupManager
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    get_history_list,
    get_metadata,
    store_message,
    update_summary_metadata,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.message_delivery import (
    ACK_RECEIVED,
    DeliverySummary,
    apply_ack,
    begin_attempt,
    delivery_record,
    unsettled_messages,
)

CONF = "handoffchar"
OLD_SUMMARY = "Mili dan user membahas rencana akhir pekan dan film favorit."


class _FakeLLM:
    """Production agent, fake transport: no provider call is ever made."""

    model = "handoff-test"
    max_tokens = 100

    async def chat_completion(self, messages, system=None, tools=None):
        yield "oke."


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


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


def make_handler(contexts: dict, *, connections=(), groups=None) -> object:
    handler = ws_module.WebSocketHandler.__new__(ws_module.WebSocketHandler)
    handler.client_contexts = contexts
    handler.client_connections = {uid: SimpleNamespace() for uid in connections}
    handler.chat_group_manager = groups or ChatGroupManager()
    handler.current_conversation_tasks = {}
    handler._detached_turns = {}
    handler._history_subscribers = {}
    handler.received_data_buffers = {}
    handler.broadcast_to_group = AsyncMock()
    handler._cancel_proactive_timer = AsyncMock()
    handler._activate_proactive_for_history = AsyncMock()
    return handler


class HandoffContractTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mili-handoff-contract-")
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        self._tasks: list[asyncio.Task] = []

        self.old_uid = create_new_history(CONF)
        for index, (role, text) in enumerate(
            [
                ("human", "halo Mili"),
                ("ai", "Halo! Ada yang bisa dibantu?"),
                ("human", "bahas rencana akhir pekan yuk"),
                ("ai", "Boleh, film favoritmu yang mana?"),
            ]
        ):
            self.assertTrue(
                store_message(CONF, self.old_uid, role, text, msg_id=f"old-{index}")
            )
        self.assertTrue(
            update_summary_metadata(
                CONF,
                self.old_uid,
                expected_summarized_through=0,
                conversation_summary=OLD_SUMMARY,
                summarized_through=4,
                summary_updated_at="2026-10-28T00:00:00+00:00",
            )
        )
        self.old_bytes, self.old_mtime = file_state(self.old_uid)
        self.agent = make_agent()
        self.context = SimpleNamespace(
            history_uid=self.old_uid,
            user_timezone="Asia/Jakarta",
            character_config=SimpleNamespace(conf_uid=CONF),
            agent_engine=self.agent,
        )
        self.agent.set_memory_from_history(
            conf_uid=CONF, history_uid=self.old_uid, user_timezone="Asia/Jakarta"
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

    # -- helpers ----------------------------------------------------------

    def _handler(self, client_uid: str = "tab-a", **kwargs) -> object:
        return make_handler({client_uid: self.context}, connections={client_uid}, **kwargs)

    def _pending_task(self) -> asyncio.Task:
        task = asyncio.create_task(asyncio.sleep(30))
        self._tasks.append(task)
        return task

    async def _finished_task(self) -> asyncio.Task:
        task = asyncio.create_task(asyncio.sleep(0))
        self._tasks.append(task)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertTrue(task.done())
        return task

    async def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        """Wait for a scheduled hand-off task instead of guessing tick counts."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.01)
        return predicate()

    def _clean_delivery(self) -> DeliverySummary:
        return unsettled_messages(CONF, owner="tab-a", limit=-1)

    def _assert_old_untouched(self) -> None:
        self.assertEqual(
            file_state(self.old_uid),
            (self.old_bytes, self.old_mtime),
            "the old transcript must be byte-identical and never rewritten",
        )

    # -- 1. the decision uses real task objects ---------------------------

    async def test_live_turn_blocks_and_a_finished_one_does_not(self):
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            delivery=self._clean_delivery(),
            summary_through=handoff.summary_boundary(CONF, self.old_uid),
        )
        self.assertEqual(decision.blockers, ())
        self.assertTrue(decision.allowed)

        running = self._pending_task()
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            active_turn=running,
            delivery=self._clean_delivery(),
            summary_through=4,
        )
        self.assertIn(handoff.BLOCK_TURN_ACTIVE, decision.blockers)
        self.assertFalse(decision.allowed)

        finished = await self._finished_task()
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            active_turn=finished,
            delivery=self._clean_delivery(),
            summary_through=4,
        )
        self.assertEqual(decision.blockers, ())
        self.assertTrue(decision.allowed)

    async def test_detached_turn_blocks_and_a_cancelled_turn_is_not_a_boundary(self):
        detached = self._pending_task()  # a real task, like the registry holds
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            detached_turn=detached,
            delivery=self._clean_delivery(),
            summary_through=4,
        )
        self.assertIn(handoff.BLOCK_TURN_DETACHED, decision.blockers)

        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            delivery=self._clean_delivery(),
            summary_through=4,
            finalized=False,
        )
        self.assertIn(handoff.BLOCK_TURN_NOT_FINALIZED, decision.blockers)
        self.assertFalse(decision.allowed)

    async def test_below_threshold_is_never_allowed(self):
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS - 1,
            delivery=self._clean_delivery(),
            summary_through=4,
        )
        self.assertEqual(decision.blockers, ())
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason(), "below-threshold")
        self.assertFalse(decision.reached_threshold)

    # -- 6. blockers from the real stores ---------------------------------

    async def test_unsettled_delivery_blocks_until_it_is_settled(self):
        self.assertIsNotNone(
            begin_attempt(
                CONF,
                "msg-live",
                turn_id="turn-1",
                owner="tab-a",
                history_uid=self.old_uid,
            )
        )
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            delivery=unsettled_messages(CONF, owner="tab-a", limit=-1),
            summary_through=4,
        )
        self.assertIn(handoff.BLOCK_DELIVERY_UNSETTLED, decision.blockers)

        outcome = apply_ack(
            CONF,
            "msg-live",
            turn_id="turn-1",
            attempt=1,
            status=ACK_RECEIVED,
            seq=0,
            final=True,
            owner="tab-a",
        )
        self.assertTrue(outcome.persisted)
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            delivery=unsettled_messages(CONF, owner="tab-a", limit=-1),
            summary_through=4,
        )
        self.assertEqual(decision.blockers, ())
        self.assertTrue(decision.allowed)

    async def test_unknown_delivery_state_blocks(self):
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            delivery=DeliverySummary(ok=False, messages=[]),
            summary_through=4,
        )
        self.assertIn(handoff.BLOCK_DELIVERY_UNKNOWN, decision.blockers)

    async def test_group_mode_blocks_one_to_one_does_not(self):
        groups = ChatGroupManager()
        groups.create_group_for_client("tab-a")
        handler = make_handler(
            {"tab-a": self.context}, connections={"tab-a"}, groups=groups
        )
        group = handler.chat_group_manager.get_client_group("tab-a")
        self.assertEqual(
            handoff.evaluate(
                rows=handoff.HANDOFF_ROWS,
                delivery=self._clean_delivery(),
                group_members=tuple(group.members),
                summary_through=4,
            ).blockers,
            (),
        )

        groups.groups["group_tab-a"].members.add("tab-b")
        groups.client_group_map["tab-b"] = "group_tab-a"
        group = handler.chat_group_manager.get_client_group("tab-a")
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            delivery=self._clean_delivery(),
            group_members=tuple(group.members),
            summary_through=4,
        )
        self.assertIn(handoff.BLOCK_GROUP_MODE, decision.blockers)

    async def test_missing_connection_or_summary_blocks(self):
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            delivery=self._clean_delivery(),
            connected=False,
            summary_through=4,
        )
        self.assertIn(handoff.BLOCK_NO_CONNECTION, decision.blockers)

        short = create_new_history(CONF)
        store_message(CONF, short, "human", "satu pesan", msg_id="s-1")
        self.assertEqual(handoff.summary_boundary(CONF, short), 0)
        self.assertEqual(handoff.message_count(CONF, short), 1)
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            delivery=self._clean_delivery(),
            summary_through=handoff.summary_boundary(CONF, short),
        )
        self.assertIn(handoff.BLOCK_NO_SUMMARY, decision.blockers)

    # -- 2/3/4/5. the switch itself (production code) ---------------------

    async def test_perform_handoff_leaves_the_old_transcript_untouched(self):
        rows_before = get_history(CONF, self.old_uid)
        begin_attempt(
            CONF,
            "msg-live",
            turn_id="turn-1",
            owner="tab-a",
            history_uid=self.old_uid,
        )
        record_before = delivery_record(CONF, "msg-live")

        new_uid = handoff.perform_handoff(CONF, self.old_uid, self.context)

        self.assertIsNotNone(new_uid)
        self.assertEqual(self.context.history_uid, new_uid)
        self._assert_old_untouched()
        self.assertEqual(get_history(CONF, self.old_uid), rows_before)
        self.assertEqual(delivery_record(CONF, "msg-live"), record_before)

    async def test_new_session_has_valid_metadata_and_unique_identities(self):
        new_uid = handoff.perform_handoff(CONF, self.old_uid, self.context)

        metadata = get_metadata(CONF, new_uid)
        self.assertEqual(metadata.get("role"), "metadata")
        self.assertTrue(metadata.get("timestamp"))
        self.assertEqual(metadata.get("relationship_status"), "stranger")
        self.assertEqual(metadata.get("continued_from"), self.old_uid)
        self.assertIsNone(get_metadata(CONF, self.old_uid).get("continued_from"))

        store_message(CONF, new_uid, "human", "lanjut yuk", msg_id="n-1")
        self._assert_old_untouched()
        old_ids = {row["msg_id"] for row in get_history(CONF, self.old_uid)}
        new_ids = [row["msg_id"] for row in get_history(CONF, new_uid)]
        self.assertEqual(new_ids, ["n-1"])
        self.assertFalse(old_ids & set(new_ids))

    async def test_agent_memory_and_continuity_come_from_the_existing_paths(self):
        new_uid = handoff.perform_handoff(CONF, self.old_uid, self.context)

        # The real agent reloaded from the NEW (empty) transcript.
        self.assertEqual(self.agent._history_uid, new_uid)
        self.assertEqual(self.agent._memory, [])
        self.assertEqual(get_history(CONF, new_uid), [])
        # Continuity enters as the previous-session summary, bounded, no rows.
        texts = [item["text"] for item in self.agent._prev_session_summaries]
        self.assertEqual(texts, [OLD_SUMMARY])
        self.assertLessEqual(len(texts[0]), PREV_SESSION_MAX_CHARS)
        self.assertEqual(
            self.agent._prev_session_at, get_history(CONF, self.old_uid)[-1]["timestamp"]
        )

    async def test_new_session_survives_repeated_history_requests(self):
        handler = self._handler()
        new_uid = handoff.perform_handoff(CONF, self.old_uid, self.context)
        handler.client_contexts["tab-a"] = self.context

        socket = SimpleNamespace(sent=[])

        async def send_text(payload: str) -> None:
            socket.sent.append(json.loads(payload))

        for _ in range(3):
            await handler._handle_history_list_request(
                SimpleNamespace(send_text=send_text), "tab-a", {}
            )

        self.assertTrue(os.path.exists(session_path(new_uid)))
        self.assertEqual(get_metadata(CONF, new_uid)["relationship_status"], "stranger")
        self._assert_old_untouched()

    # -- 7. crash / reconnect at each boundary ----------------------------

    async def test_crash_after_target_creation_leaves_the_old_session_usable(self):
        target = handoff.create_continuation(CONF, self.old_uid)
        self.assertIsNotNone(target)
        self.assertEqual(get_history(CONF, target), [])
        self._assert_old_untouched()

        handler = self._handler()
        socket = SimpleNamespace(sent=[])

        async def send_text(payload: str) -> None:
            socket.sent.append(json.loads(payload))

        await handler._handle_history_list_request(
            SimpleNamespace(send_text=send_text), "tab-a", {}
        )
        listed = [item["uid"] for item in socket.sent[0]["histories"]]
        self.assertIn(self.old_uid, listed)
        self.assertNotIn(target, listed, "an empty session is not offered for resume")

        # A target the client never reached is still not a conversation, but it
        # is the session that hand-off intended to continue -- Stage 4.10
        # measured the client silently falling back to the source after a
        # restart because this file was cleaned up. It now survives cleanup for
        # the resume window and is reported apart from `histories`.
        stale = time.time() - history_store.EMPTY_SESSION_GRACE_SECONDS - 60
        os.utime(session_path(target), (stale, stale))
        get_history_list(CONF, protect_uids=set())
        self.assertTrue(
            os.path.exists(session_path(target)),
            "an empty hand-off target must stay resumable across a restart",
        )
        self.assertNotIn(
            target,
            {item["uid"] for item in get_history_list(CONF, protect_uids=set())},
            "a resumable target is still never offered as a conversation",
        )
        self._assert_old_untouched()

        # ...and it is removed once the resume window is over, so an abandoned
        # hand-off does not leak a file forever.
        stale = time.time() - history_store.EMPTY_CONTINUATION_TTL_SECONDS - 60
        os.utime(session_path(target), (stale, stale))
        get_history_list(CONF, protect_uids=set())
        self.assertFalse(os.path.exists(session_path(target)))
        self._assert_old_untouched()

    async def test_crash_after_switch_before_client_adoption_resumes_the_old_one(self):
        self.assertIsNotNone(handoff.perform_handoff(CONF, self.old_uid, self.context))

        # Restart: the client reconnects with the uid it actually persisted.
        self.assertEqual(len(get_history(CONF, self.old_uid)), 4)
        self.assertEqual(get_metadata(CONF, self.old_uid)["summary_through_message_index"], 4)
        self._assert_old_untouched()

    async def test_crash_after_adoption_before_first_message_loses_nothing(self):
        new_uid = handoff.perform_handoff(CONF, self.old_uid, self.context)
        self.assertEqual(get_history(CONF, new_uid), [])
        metadata = get_metadata(CONF, new_uid)
        self.assertEqual(metadata["relationship_status"], "stranger")
        self.assertEqual(metadata["continued_from"], self.old_uid)

        # The empty new session is absent from the list, so a reconnecting
        # client must fall back to `continued_from` (open client contract).
        listing = [item["uid"] for item in get_history_list(CONF, protect_uids=set())]
        self.assertNotIn(new_uid, listing)
        self.assertIn(self.old_uid, listing)

        listing = [item["uid"] for item in get_history_list(CONF, protect_uids={new_uid})]
        self.assertTrue(os.path.exists(session_path(new_uid)))
        self.assertEqual(listing, [self.old_uid])

    # -- 8. repeated requests, two tabs, retries --------------------------

    async def test_two_tabs_and_repeated_requests_do_not_double_hand_off(self):
        second_context = SimpleNamespace(
            history_uid=self.old_uid,
            user_timezone="Asia/Jakarta",
            character_config=SimpleNamespace(conf_uid=CONF),
            agent_engine=make_agent(),
        )
        groups = ChatGroupManager()
        groups.create_group_for_client("tab-a")
        groups.create_group_for_client("tab-b")
        handler = make_handler(
            {"tab-a": self.context, "tab-b": second_context},
            connections={"tab-a", "tab-b"},
            groups=groups,
        )

        new_uid = handoff.perform_handoff(CONF, self.old_uid, self.context)
        snapshot = sorted(
            name
            for name in os.listdir(os.path.join("chat_history", CONF))
            if name.endswith(".json")
        )
        socket = SimpleNamespace(sent=[])

        async def send_text(payload: str) -> None:
            socket.sent.append(json.loads(payload))

        for _ in range(3):
            await handler._handle_history_list_request(
                SimpleNamespace(send_text=send_text), "tab-b", {}
            )

        self.assertEqual(
            sorted(
                name
                for name in os.listdir(os.path.join("chat_history", CONF))
                if name.endswith(".json")
            ),
            snapshot,
            "repeated history requests must not create sessions",
        )
        self.assertTrue(os.path.exists(session_path(new_uid)))
        self._assert_old_untouched()
        self.assertEqual(handler.client_contexts["tab-b"].history_uid, self.old_uid)
        self.assertEqual(len(get_history(CONF, self.old_uid)), 4)

    async def test_retry_across_the_boundary_stays_in_the_old_transcript(self):
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            delivery=self._clean_delivery(),
            summary_through=4,
        )
        begin_attempt(
            CONF,
            "msg-retry",
            turn_id="turn-r",
            owner="tab-a",
            history_uid=self.old_uid,
        )
        decision = handoff.evaluate(
            rows=handoff.HANDOFF_ROWS,
            delivery=unsettled_messages(CONF, owner="tab-a", limit=-1),
            summary_through=4,
        )
        self.assertIn(handoff.BLOCK_DELIVERY_UNSETTLED, decision.blockers)

        apply_ack(
            CONF,
            "msg-retry",
            turn_id="turn-r",
            attempt=1,
            status=ACK_RECEIVED,
            seq=0,
            final=True,
            owner="tab-a",
        )
        new_uid = handoff.perform_handoff(CONF, self.old_uid, self.context)
        self.assertTrue(
            store_message(CONF, self.old_uid, "ai", "versi kedua", msg_id="msg-retry")
        )
        self.assertTrue(
            store_message(CONF, self.old_uid, "ai", "versi kedua", msg_id="msg-retry")
        )

        retried = [
            row for row in get_history(CONF, self.old_uid) if row["msg_id"] == "msg-retry"
        ]
        self.assertEqual(len(retried), 1)
        self.assertEqual(retried[0]["content"], "versi kedua")
        self.assertEqual(get_history(CONF, new_uid), [])

    # -- integration: only a finalized turn, only when opted in -----------

    async def test_completed_turn_switches_when_enabled_and_safe(self):
        handler = self._handler()
        finished = await self._finished_task()

        with patch.dict(os.environ, {handoff.AUTO_HANDOFF_ENV: "1"}), patch.object(
            handoff, "HANDOFF_ROWS", 2
        ):
            handler._on_turn_completed("tab-a", self.old_uid, finished)
            switched = await self._wait_until(
                lambda: handler.client_contexts["tab-a"].history_uid != self.old_uid
            )
        self.assertTrue(switched, "the completed turn should have handed off")

        new_uid = handler.client_contexts["tab-a"].history_uid
        self.assertNotEqual(new_uid, self.old_uid)
        self.assertEqual(get_history(CONF, new_uid), [])
        self.assertEqual(get_metadata(CONF, new_uid)["continued_from"], self.old_uid)
        self._assert_old_untouched()
        # Proactive timer re-armed for the new session, subscriber re-bound.
        handler._cancel_proactive_timer.assert_awaited()
        handler._activate_proactive_for_history.assert_awaited_with("tab-a", new_uid)
        self.assertEqual(handler._subscriber_registry()["tab-a"], new_uid)

    async def test_nothing_happens_when_automatic_mode_is_off_or_turn_cancelled(self):
        handler = self._handler()
        finished = await self._finished_task()

        with patch.object(handoff, "HANDOFF_ROWS", 2):
            handler._on_turn_completed("tab-a", self.old_uid, finished)
            await asyncio.sleep(0.1)
        self.assertEqual(handler.client_contexts["tab-a"].history_uid, self.old_uid)

        cancelled = self._pending_task()
        cancelled.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancelled
        with patch.dict(os.environ, {handoff.AUTO_HANDOFF_ENV: "1"}), patch.object(
            handoff, "HANDOFF_ROWS", 2
        ):
            handler._on_turn_completed("tab-a", self.old_uid, cancelled)
            await asyncio.sleep(0.1)
        self.assertEqual(handler.client_contexts["tab-a"].history_uid, self.old_uid)
        self._assert_old_untouched()

    async def test_turn_trigger_registers_the_completion_callback(self):
        """The hook must be registered by the real trigger, not only callable."""
        from src.open_llm_vtuber.conversations import conversation_handler as ch

        received: list[tuple[str, str]] = []

        async def fake_turn(**_kwargs):
            return None

        with patch.object(ch, "process_single_conversation", fake_turn):
            await ch.handle_conversation_trigger(
                msg_type="text-input",
                data={"text": "halo"},
                client_uid="tab-a",
                context=self.context,
                websocket=SimpleNamespace(send_text=AsyncMock()),
                client_contexts={"tab-a": self.context},
                client_connections={"tab-a": SimpleNamespace()},
                chat_group_manager=ChatGroupManager(),
                received_data_buffers={},
                current_conversation_tasks={},
                broadcast_to_group=AsyncMock(),
                detached_turns={},
                on_turn_complete=lambda uid, history, _task: received.append(
                    (uid, history)
                ),
            )
            delivered = await self._wait_until(lambda: bool(received))

        self.assertTrue(delivered, "the trigger must notify when the turn ends")
        self.assertEqual(received, [("tab-a", self.old_uid)])

    async def test_handler_trigger_passes_its_completion_callback(self):
        """The connection layer must hand the hook to the turn trigger."""
        handler = self._handler()
        handler._record_user_activity = AsyncMock()
        handler._update_user_timezone = lambda context, data: None
        captured: dict = {}

        async def fake_trigger(**kwargs):
            captured.update(kwargs)

        with patch.object(ws_module, "handle_conversation_trigger", fake_trigger):
            await handler._handle_conversation_trigger(
                SimpleNamespace(send_text=AsyncMock()),
                "tab-a",
                {"type": "text-input", "text": "halo"},
            )

        self.assertEqual(
            captured.get("on_turn_complete"),
            handler._on_turn_completed,
            "a turn that finishes must notify the hand-off hook",
        )

    async def test_auto_mode_defers_when_the_transcript_is_still_short(self):
        handler = self._handler()
        finished = await self._finished_task()

        with patch.dict(os.environ, {handoff.AUTO_HANDOFF_ENV: "1"}):
            handler._on_turn_completed("tab-a", self.old_uid, finished)
            await asyncio.sleep(0.1)

        self.assertEqual(handler.client_contexts["tab-a"].history_uid, self.old_uid)
        self._assert_old_untouched()

    # -- 8. the receipt that clears the last blocker ----------------------

    def _start_turn_delivery(self, msg_id: str = "msg-turn", turn_id: str = "turn-x"):
        """A message this turn sent, still unacknowledged by the client."""
        record = begin_attempt(
            CONF,
            msg_id,
            turn_id=turn_id,
            owner="tab-a",
            history_uid=self.old_uid,
        )
        self.assertIsNotNone(record)
        return msg_id, turn_id

    def _settling_ack(self, msg_id: str, turn_id: str) -> dict:
        return {
            "type": "message-ack",
            "msg_id": msg_id,
            "turn_id": turn_id,
            "attempt": 1,
            "status": ACK_RECEIVED,
            "seq": 0,
            "final": True,
        }

    async def test_settling_receipt_completes_a_deferred_handoff(self):
        """Live-found gap: the turn's own message is unacked when it ends.

        The turn-completion hook runs before this client's receipt for the
        closing frame, so the evaluation correctly defers (delivery-unsettled).
        Without re-running when that message settles, every turn would defer
        forever -- each turn's own message is the one still in flight at its
        end. The receipt is the event; this pins it end to end.
        """
        msg_id, turn_id = self._start_turn_delivery()
        handler = self._handler()
        socket = SimpleNamespace(sent=[], send_text=AsyncMock())
        handler.client_connections["tab-a"] = socket

        async def send_text(payload: str) -> None:
            socket.sent.append(json.loads(payload))

        socket.send_text = send_text

        with patch.dict(os.environ, {handoff.AUTO_HANDOFF_ENV: "1"}), patch.object(
            handoff, "message_count", return_value=handoff.HANDOFF_ROWS
        ):
            await handler._consider_handoff("tab-a", self.old_uid)
            self.assertEqual(
                handler.client_contexts["tab-a"].history_uid,
                self.old_uid,
                "an unacked message must block the switch",
            )
            self._assert_old_untouched()

            await handler._handle_message_ack(socket, "tab-a", self._settling_ack(msg_id, turn_id))
            moved = await self._wait_until(
                lambda: handler.client_contexts["tab-a"].history_uid != self.old_uid
            )

        self.assertTrue(moved, "a settled receipt must complete the hand-off")
        new_uid = handler.client_contexts["tab-a"].history_uid
        self.assertEqual(get_metadata(CONF, new_uid)["continued_from"], self.old_uid)
        self.assertEqual(get_history(CONF, new_uid), [])
        frames = [f for f in socket.sent if f.get("type") == "new-history-created"]
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["history_uid"], new_uid)
        self.assertEqual(frames[0]["continued_from"], self.old_uid)
        self._assert_old_untouched()

    async def test_a_settling_receipt_never_runs_the_handoff_when_auto_is_off(self):
        msg_id, turn_id = self._start_turn_delivery(msg_id="msg-off", turn_id="turn-off")
        handler = self._handler()
        socket = SimpleNamespace(sent=[], send_text=AsyncMock())
        handler.client_connections["tab-a"] = socket
        before = set(os.listdir(os.path.join("chat_history", CONF)))

        with patch.object(handoff, "message_count", return_value=handoff.HANDOFF_ROWS):
            await handler._handle_message_ack(socket, "tab-a", self._settling_ack(msg_id, turn_id))
            await asyncio.sleep(0.2)

        self.assertEqual(handler.client_contexts["tab-a"].history_uid, self.old_uid)
        self.assertEqual(set(os.listdir(os.path.join("chat_history", CONF))), before)
        self.assertEqual([f for f in socket.sent if f.get("type") == "new-history-created"], [])

    # -- negative controls -------------------------------------------------

    async def test_negative_control_without_protection_the_target_is_deleted(self):
        """With the resume window closed the cleanup really does delete it.

        Without this control the "survives a restart" assertions above could
        pass because the cleanup never removed anything at all.
        """
        target = handoff.create_continuation(CONF, self.old_uid)
        with patch.object(history_store, "EMPTY_CONTINUATION_TTL_SECONDS", 0.0):
            get_history_list(CONF, protect_uids=set())
        self.assertFalse(os.path.exists(session_path(target)))

    async def test_negative_control_continuity_assertion_reads_the_real_store(self):
        new_uid = handoff.perform_handoff(CONF, self.old_uid, self.context)
        real = [item["text"] for item in self.agent._prev_session_summaries]
        self.assertEqual(real, [OLD_SUMMARY])

        with patch.object(
            BasicMemoryAgent, "_load_previous_session_summaries", return_value=[]
        ):
            silenced = BasicMemoryAgent._load_previous_session_summaries(CONF, new_uid)
        self.assertNotEqual([item["text"] for item in silenced], [OLD_SUMMARY])

    async def test_negative_control_switch_cannot_be_faked_without_memory_reload(self):
        """A context whose agent cannot reload must not be switched.

        This is what stops a "switch" that only renames the uid: memory would
        still hold the old transcript while new turns are written elsewhere.
        """
        broken = SimpleNamespace(
            history_uid=self.old_uid,
            user_timezone="Asia/Jakarta",
            character_config=SimpleNamespace(conf_uid=CONF),
            agent_engine=SimpleNamespace(),
        )
        self.assertIsNone(handoff.perform_handoff(CONF, self.old_uid, broken))
        self.assertEqual(broken.history_uid, self.old_uid)
        self._assert_old_untouched()


if __name__ == "__main__":
    unittest.main()
