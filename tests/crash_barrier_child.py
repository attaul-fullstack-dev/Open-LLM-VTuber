"""Crash-boundary driver for ``tests/test_crash_barrier_history.py``.

Deliberately NOT named ``test_*``: the repo's ``unittest discover`` pattern
must not collect it. The parent test starts one process per scenario, with its
own cwd (both the history store and the delivery store are cwd-relative).

Why a subprocess: the window this pins is "the AI row exists on disk, the
process is gone, no ``finally`` ran, nothing was flushed". ``os._exit`` in the
transport is the closest thing to a SIGKILL that a test can schedule
deterministically -- it skips atexit handlers, the event loop's shutdown and
every ``finally`` in the turn. An in-process mock cannot reproduce that.

Modes
-----
``crash_before_first``  first CONTENT frame: die before it is written.
``crash_after_first``   first CONTENT frame: write it, then die (no final).
``complete``            run the turn to the end and report what it returned.
``interrupt``           die-adjacent cancel: after frame 1 is on the wire, the
                        agent stream raises CancelledError (the interrupt path).
``provider_error``      a typed provider error after frame 1.
``raise_error``         the agent stream raises after frame 1.
``persist_fail``        every AI-row write raises a real OSError at the
                        atomic-replace boundary.

Every frame the turn hands to the transport is appended (and closed/flushed)
to ``frames_path`` as JSON lines, so the parent can assert exactly what left
the process before it died.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.open_llm_vtuber.agent.output_types import (  # noqa: E402
    Actions,
    DisplayText,
    SentenceOutput,
)
from src.open_llm_vtuber.conversations import conversation_utils  # noqa: E402
from src.open_llm_vtuber.conversations import single_conversation  # noqa: E402
from src.open_llm_vtuber.conversations.single_conversation import (  # noqa: E402
    process_single_conversation,
)

CRASH_EXIT = 9
HARNESS_EXIT = 70
CONTENT_TYPES = frozenset({"audio"})


def _append_frame(path: str, frame: dict) -> None:
    """Append one delivered frame; closing the file makes it visible to the
    parent even when this process is killed right afterwards."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(frame, ensure_ascii=False) + "\n")


def _count_content_frames(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            frames = [json.loads(line) for line in handle if line.strip()]
    except (OSError, ValueError):
        return 0
    return sum(1 for frame in frames if frame.get("type") in CONTENT_TYPES)


class _Transport:
    """Real delivery boundary: writes what it accepts, or kills the process."""

    def __init__(self, mode: str, frames_path: str) -> None:
        self.mode = mode
        self.frames_path = frames_path
        self.content_frames = 0

    async def __call__(self, payload: str) -> None:
        frame = json.loads(payload)
        if frame.get("type") in CONTENT_TYPES:
            self.content_frames += 1
            if self.mode == "crash_before_first" and self.content_frames == 1:
                # The guard already wrote the row; the frame never leaves.
                os._exit(CRASH_EXIT)
            _append_frame(self.frames_path, frame)
            if self.mode == "crash_after_first" and self.content_frames == 1:
                # Frame 1 is durably on the wire/disk, the final frame is not.
                os._exit(CRASH_EXIT)
            return
        _append_frame(self.frames_path, frame)


class _Agent:
    def __init__(self, mode: str, sentences, frames_path: str) -> None:
        self.mode = mode
        self.sentences = list(sentences)
        self.frames_path = frames_path
        self._llm = None

    async def chat(self, _batch_input):
        for index, text in enumerate(self.sentences):
            yield SentenceOutput(
                display_text=DisplayText(text=text, name="Mili"),
                tts_text=text,
                actions=Actions(),
            )
            if index != 0:
                continue
            if self.mode == "interrupt":
                await self._wait_for_delivery(1)
                raise asyncio.CancelledError()
            if self.mode == "provider_error":
                yield {
                    "type": "provider_error",
                    "category": "server_error",
                    "message": "provider stream failed",
                }
                return
            if self.mode == "raise_error":
                raise RuntimeError("provider stream died")

    async def _wait_for_delivery(self, count: int, timeout: float = 20.0) -> None:
        """Deterministic barrier: the interrupt fires after frame 1 was sent."""
        waited = 0.0
        while _count_content_frames(self.frames_path) < count:
            if waited >= timeout:
                raise AssertionError("content frame never reached the transport")
            await asyncio.sleep(0.01)
            waited += 0.01
        # Let the sender finish its post-send confirmation for this frame so
        # the interrupt path sees the same state a real interrupt would.
        await asyncio.sleep(0.05)


class _Live2D:
    def extract_emotion(self, _text):
        return []


def _context(mode: str, sentences, frames_path: str, conf: str, history_uid: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        history_uid=history_uid,
        agent_engine=_Agent(mode, sentences, frames_path),
        user_timezone="Asia/Jakarta",
        asr_engine=None,
        live2d_model=_Live2D(),
        tts_engine=None,
        translate_engine=None,
        voice_output_enabled=False,
        character_config=SimpleNamespace(
            conf_uid=conf,
            human_name="User",
            character_name="Mili",
            avatar="mili.png",
        ),
    )


async def _playback_complete(*_args, **_kwargs):
    return {"type": "frontend-playback-complete"}


async def _finalize_after_flush(tts_manager=None, websocket_send=None, client_uid=None, **_):
    """Flush the ordered sender, then run the REAL lifecycle tail.

    No browser exists, so the playback wait is answered immediately; the queue
    join is the deterministic part (it returns only after every queued frame
    was handed to the transport).
    """
    queue = getattr(tts_manager, "_payload_queue", None)
    if queue is not None:
        await asyncio.wait_for(queue.join(), timeout=20)
    conversation_utils.message_handler.wait_for_response = _playback_complete
    await conversation_utils.finalize_conversation_turn(
        tts_manager=tts_manager,
        websocket_send=websocket_send,
        client_uid=client_uid,
    )


def _install_failing_replace() -> None:
    """A real OSError at the atomic-write boundary for every AI row."""
    original = os.replace

    def write_targets_ai_row(src: str) -> bool:
        try:
            with open(src, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return False
        if not isinstance(data, list):
            return False
        return any(
            isinstance(item, dict) and item.get("role") == "ai" for item in data
        )

    def replace(src, dst):
        if write_targets_ai_row(src):
            raise OSError(errno.EIO, "simulated disk failure")
        return original(src, dst)

    os.replace = replace


async def _main_async(mode, conf, history_uid, msg_id, frames_path, result_path, sentences):
    transport = _Transport(mode, frames_path)
    context = _context(mode, sentences, frames_path, conf, history_uid)
    single_conversation.finalize_conversation_turn = _finalize_after_flush
    if mode == "persist_fail":
        _install_failing_replace()

    result = {"returned": None, "error": ""}
    try:
        result["returned"] = await process_single_conversation(
            context=context,
            websocket_send=transport,
            client_uid="client-crash",
            user_input="Halo Mili.",
            metadata={"msg_id": msg_id},
        )
    except asyncio.CancelledError:
        result["error"] = "CancelledError"
    except Exception as error:  # noqa: BLE001 - reported to the parent test
        result["error"] = type(error).__name__

    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)


def main(argv) -> int:
    if len(argv) != 8:
        print(
            "usage: <mode> <conf> <history_uid> <msg_id> <frames> <result> <sentences-json>",
            file=sys.stderr,
        )
        return HARNESS_EXIT
    mode, conf, history_uid, msg_id, frames_path, result_path, sentences_json = argv[1:]
    try:
        sentences = json.loads(sentences_json)
    except ValueError:
        print("sentences must be JSON", file=sys.stderr)
        return HARNESS_EXIT
    try:
        asyncio.run(
            _main_async(
                mode, conf, history_uid, msg_id, frames_path, result_path, sentences
            )
        )
    except BaseException:  # noqa: BLE001 - a harness bug must fail the test
        import traceback

        traceback.print_exc()
        return HARNESS_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
