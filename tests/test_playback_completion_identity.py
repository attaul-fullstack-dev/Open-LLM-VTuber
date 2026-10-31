"""Playback completion must name the turn it belongs to.

Regression coverage for the audio lifecycle bugs:

* an untagged wait accepted a completion left over from an earlier turn;
* the frontend's ``audio_outcome`` was ignored, so a failed playback looked
  exactly like a successful one in the backend report;
* a legacy client (no identity, no outcome) still has to settle the wait and
  release the turn.

Only deterministic doubles are used: no browser, no TTS provider.
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from loguru import logger

from src.open_llm_vtuber.conversations import conversation_utils
from src.open_llm_vtuber.conversations.conversation_utils import (
    finalize_conversation_turn,
)
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager
from src.open_llm_vtuber.message_handler import MessageHandler
from src.open_llm_vtuber.request_latency import (
    RequestLatencyTracker,
    reset_latency_tracker,
    set_latency_tracker,
)


class _Socket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def _tracker(socket: _Socket, request_id="turn-new", msg_id="msg-1", attempt=2):
    return RequestLatencyTracker(
        websocket_send=socket.send_text,
        provider="fake",
        model="fake",
        request_id=request_id,
        msg_id=msg_id,
        msg_attempt=attempt,
    )


class MessageHandlerIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_tagged_completion_cannot_settle_the_new_turn(self):
        handler = MessageHandler()
        waiter = asyncio.create_task(
            handler.wait_for_response(
                "client",
                "frontend-playback-complete",
                request_id="turn-new",
                timeout=1,
                accept_untagged_legacy=True,
            )
        )
        await asyncio.sleep(0)

        handler.handle_message(
            "client",
            {
                "type": "frontend-playback-complete",
                "request_id": "turn-old",
                "audio_outcome": "played",
            },
        )
        await asyncio.sleep(0.01)
        self.assertFalse(waiter.done(), "a stale tagged completion must be ignored")

        handler.handle_message(
            "client",
            {
                "type": "frontend-playback-complete",
                "request_id": "turn-new",
                "audio_outcome": "played",
            },
        )
        response = await waiter
        self.assertEqual(response["request_id"], "turn-new")

    async def test_legacy_untagged_completion_still_releases_a_tagged_wait(self):
        handler = MessageHandler()
        waiter = asyncio.create_task(
            handler.wait_for_response(
                "client",
                "frontend-playback-complete",
                request_id="turn-new",
                timeout=1,
                accept_untagged_legacy=True,
            )
        )
        await asyncio.sleep(0)

        handler.handle_message(
            "client",
            {"type": "frontend-playback-complete"},
        )
        response = await waiter
        self.assertEqual(response["type"], "frontend-playback-complete")
        self.assertNotIn("request_id", response)

    async def test_duplicate_completion_is_consumed_once(self):
        handler = MessageHandler()
        first = await asyncio.wait_for(
            self._settle(handler, "turn-a"), timeout=1
        )
        self.assertEqual(first["request_id"], "turn-a")

        # The duplicate (same turn) arrives while no wait is registered. A
        # later turn must not be settled by it.
        handler.handle_message(
            "client",
            {
                "type": "frontend-playback-complete",
                "request_id": "turn-a",
                "audio_outcome": "played",
            },
        )
        later = asyncio.create_task(
            handler.wait_for_response(
                "client",
                "frontend-playback-complete",
                request_id="turn-b",
                timeout=0.2,
                accept_untagged_legacy=True,
            )
        )
        await asyncio.sleep(0.01)
        self.assertFalse(later.done(), "the consumed duplicate must not settle turn-b")
        self.assertIsNone(await later)

    async def _settle(self, handler, request_id):
        async def send():
            await asyncio.sleep(0)
            handler.handle_message(
                "client",
                {
                    "type": "frontend-playback-complete",
                    "request_id": request_id,
                    "audio_outcome": "played",
                },
            )

        asyncio.create_task(send())
        return await handler.wait_for_response(
            "client",
            "frontend-playback-complete",
            request_id=request_id,
            timeout=1,
            accept_untagged_legacy=True,
        )

    async def test_missing_completion_times_out(self):
        handler = MessageHandler()
        started = time.monotonic()
        response = await handler.wait_for_response(
            "client",
            "frontend-playback-complete",
            request_id="turn-new",
            timeout=0.05,
            accept_untagged_legacy=True,
        )
        self.assertIsNone(response)
        self.assertLess(time.monotonic() - started, 1.0)


class FinalizePlaybackOutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def _finalize(self, response):
        socket = _Socket()
        manager = TTSTaskManager()
        manager.task_list.append(asyncio.sleep(0))
        tracker = _tracker(socket)
        token = set_latency_tracker(tracker)
        waiter = AsyncMock(return_value=response)
        try:
            with patch.object(
                conversation_utils.message_handler,
                "wait_for_response",
                new=waiter,
            ):
                await finalize_conversation_turn(manager, socket.send_text, "client")
        finally:
            reset_latency_tracker(token)
        synth = next(
            frame for frame in socket.sent
            if frame.get("type") == "backend-synth-complete"
        )
        return socket, tracker, waiter, synth

    async def test_synth_complete_carries_the_turn_the_completion_must_echo(self):
        _, tracker, waiter, synth = await self._finalize(
            {
                "type": "frontend-playback-complete",
                "request_id": "turn-new",
                "audio_outcome": "played",
            }
        )

        self.assertEqual(synth["request_id"], "turn-new")
        self.assertEqual(synth["turn"], "turn-new")
        self.assertEqual(synth["msg_id"], "msg-1")
        self.assertEqual(synth["id"], "msg-1")
        self.assertEqual(synth["attempt"], 2)

        waiter.assert_awaited_once()
        kwargs = waiter.await_args.kwargs
        self.assertEqual(kwargs["request_id"], "turn-new")
        self.assertTrue(kwargs["accept_untagged_legacy"])
        self.assertEqual(tracker.playback_outcome, "played")
        self.assertEqual(tracker.playback_failed_segments, 0)

    async def test_failed_audio_is_consumed_and_never_reported_as_played(self):
        log_messages: list[str] = []
        sink = logger.add(
            lambda message: log_messages.append(str(message)),
            level="WARNING",
        )
        try:
            _, tracker, _, _ = await self._finalize(
                {
                    "type": "frontend-playback-complete",
                    "request_id": "turn-new",
                    "audio_outcome": "failed",
                    "audio_failed_segments": 2,
                    "audio_failure_reason": "media-error-4",
                }
            )
        finally:
            logger.remove(sink)

        self.assertEqual(tracker.playback_outcome, "failed")
        self.assertEqual(tracker.playback_failed_segments, 2)
        self.assertTrue(
            any("AUDIO_PLAYBACK_FAILED" in message for message in log_messages),
            "a failed playback must be logged as a failure",
        )
        self.assertFalse(
            any("AUDIO_PLAYBACK_COMPLETE" in message for message in log_messages),
            "a failed playback must never be logged as a success",
        )

    async def test_contradictory_played_label_with_failed_segments_fails_closed(self):
        _, tracker, _, _ = await self._finalize(
            {
                "type": "frontend-playback-complete",
                "request_id": "turn-new",
                "audio_outcome": "played",
                "audio_failed_segments": 1,
            }
        )
        self.assertEqual(tracker.playback_outcome, "failed")
        self.assertEqual(tracker.playback_failed_segments, 1)

    async def test_no_completion_records_timeout_and_still_ends_the_turn(self):
        socket = _Socket()
        manager = TTSTaskManager()
        manager.task_list.append(asyncio.sleep(0))
        tracker = _tracker(socket)
        token = set_latency_tracker(tracker)
        try:
            with patch.object(
                conversation_utils,
                "PLAYBACK_COMPLETE_TIMEOUT_S",
                0.01,
            ), patch.object(
                conversation_utils.message_handler,
                "wait_for_response",
                new=AsyncMock(return_value=None),
            ):
                await finalize_conversation_turn(
                    manager, socket.send_text, "client"
                )
        finally:
            reset_latency_tracker(token)

        self.assertEqual(tracker.playback_outcome, "timeout")
        types = [frame.get("type") for frame in socket.sent]
        self.assertIn("force-new-message", types)
        self.assertTrue(
            any(
                frame.get("type") == "control"
                and frame.get("text") == "conversation-chain-end"
                for frame in socket.sent
            )
        )


if __name__ == "__main__":
    unittest.main()
