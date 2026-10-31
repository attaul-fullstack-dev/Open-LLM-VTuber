import asyncio
import json
import re
import time
import uuid
from datetime import datetime
from typing import Awaitable, Callable, List, Optional, Dict, Tuple
from loguru import logger

from ..agent.output_types import DisplayText, Actions
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload
from .types import WebSocketSend

# Serialize ElevenLabs synthesis to avoid API concurrency-limit errors.
# Per event loop (not one global): asyncio primitives bind to the loop that
# first acquires them, so a single global semaphore breaks with "bound to a
# different event loop" whenever the loop is recreated (tests, embedding
# reuse) and every synthesis degrades to a silent payload. One slot per live
# loop preserves the concurrency guarantee exactly where it matters.
_ELEVENLABS_TTS_SEMAPHORES: Dict[int, asyncio.Semaphore] = {}


def _stamp_turn_identity(payload: Dict, tracker) -> Dict:
    """Attach this turn's identity to a content frame.

    ``request_id`` names the ATTEMPT, ``msg_id`` names the LOGICAL message
    (stable across retries). The frontend opens/updates one bubble per
    ``msg_id``, so both must travel on every content frame.

    The delivery protocol (Stage 2) is carried alongside without renaming any
    existing key, so an old client keeps working: ``id`` mirrors ``msg_id``,
    ``turn`` mirrors ``request_id``, ``attempt`` is the message attempt and
    ``final`` is false on every content frame. ``seq`` is assigned later, in
    send order (``_stamp_frame_sequence``): TTS payloads are prepared
    concurrently, so stamping a sequence here would number them by synthesis
    completion, not by delivery.
    """
    if not tracker:
        return payload
    request_id = getattr(tracker, "request_id", "") or ""
    if request_id:
        payload["request_id"] = request_id
        payload["turn"] = request_id
    msg_id = getattr(tracker, "msg_id", "") or ""
    if msg_id:
        payload["msg_id"] = msg_id
        payload["id"] = msg_id
        payload["attempt"] = int(getattr(tracker, "msg_attempt", 1) or 1)
        payload["final"] = False
    return payload


def _stamp_frame_sequence(payload: Dict, tracker) -> Dict:
    """Number a content frame in DELIVERY order and clear its ``final`` flag.

    Called from the single ordered sender, immediately before the frame is
    handed to the transport, so ``seq`` is the order the client actually sees.
    A payload belonging to another message (or a tracker-less context) is left
    untouched: it is not part of this message's sequence.
    """
    if not tracker:
        return payload
    msg_id = getattr(tracker, "msg_id", "") or ""
    if not msg_id or payload.get("msg_id") != msg_id:
        return payload
    seq = int(getattr(tracker, "msg_seq", 0) or 0)
    payload["seq"] = seq
    payload["final"] = False
    tracker.msg_seq = seq + 1
    return payload


def final_frame_metadata(tracker) -> Dict:
    """Metadata for the CLOSING frame of one message (``ai-final``).

    ``seq`` continues the content sequence and ``segments`` states how many
    content frames this attempt actually handed to the transport, so the
    client can tell a complete message from a truncated one. Returned empty
    for a tracker-less turn: no identity, no protocol fields.
    """
    if not tracker:
        return {}
    msg_id = getattr(tracker, "msg_id", "") or ""
    if not msg_id:
        return {}
    segments = int(getattr(tracker, "msg_seq", 0) or 0)
    return {
        "id": msg_id,
        "msg_id": msg_id,
        "attempt": int(getattr(tracker, "msg_attempt", 1) or 1),
        "seq": segments,
        "segments": segments,
        "final": True,
    }


def _elevenlabs_semaphore() -> asyncio.Semaphore:
    """The ElevenLabs concurrency slot for the running loop.

    Entries are keyed by loop identity and never pruned: each entry is tiny,
    production creates exactly one, and a recycled id can only inherit an
    uncontended slot from a dead loop (always safe).
    """
    loop = asyncio.get_running_loop()
    key = id(loop)
    existing = _ELEVENLABS_TTS_SEMAPHORES.get(key)
    if existing is not None:
        return existing
    slot = asyncio.Semaphore(1)
    _ELEVENLABS_TTS_SEMAPHORES[key] = slot
    return slot


class TTSTaskManager:
    """Manages TTS tasks and ensures ordered delivery to frontend while allowing parallel TTS generation"""

    def __init__(self) -> None:
        self.task_list: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        # Queue to store ordered payloads
        self._payload_queue: asyncio.Queue[Dict] = asyncio.Queue()
        # Task to handle sending payloads in order
        self._sender_task: Optional[asyncio.Task] = None
        # Counter for maintaining order
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        # Set when a pre-send history guard refuses a frame. No later frame
        # of this turn may leave the process: the transcript would otherwise
        # contain a gap that no durable row can describe.
        self._content_blocked = False

    async def speak(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
        synthesize_audio: bool = True,
        emotion_tag: Optional[str] = None,
        history_text: str = "",
        before_content_send: Optional[Callable[[str], Awaitable[bool]]] = None,
        after_content_send: Optional[
            Callable[[str, bool], Awaitable[None]]
        ] = None,
    ) -> None:
        """
        Queue a TTS task while maintaining order of delivery.

        Args:
            tts_text: Text to synthesize
            display_text: Text to display in UI
            actions: Live2D model actions
            live2d_model: Live2D model instance
            tts_engine: TTS engine instance
            websocket_send: WebSocket send function
            synthesize_audio: When False, skip audio synthesis entirely and
                send a text-only display payload (no TTS API call). Reuses the
                existing silent/text-only payload path so the frontend and the
                turn lifecycle behave exactly like the muted case.
            emotion_tag: ElevenLabs v3 audio tag (e.g. "happy") for THIS
                sentence only. Prepended as ``[tag]`` to the synthesis text
                when the engine is ElevenLabs with tags enabled; display
                text, history and translation input are never touched.
                Stateless per call, so no emotion can leak across sentences.
            history_text: Canonical display text carried by this frame. The
                optional guard writes this delta to the AI history row BEFORE
                the ordered sender hands the frame to the transport.
            before_content_send: Async guard invoked immediately before the
                sender sends this frame. Returning False stops the frame and
                every later frame of the turn.
            after_content_send: Async confirmation invoked after the
                transport accepts/refuses this frame. A refusal rolls the
                provisional history prefix back to the last accepted frame.
        """
        if self._content_blocked:
            logger.warning(
                "TTS content blocked: history was not durable before an "
                "earlier frame; dropping this segment"
            )
            return
        if emotion_tag and synthesize_audio and self._emotion_tags_allowed(tts_engine):
            from ..voice_emotion import tag_tts_text

            tts_text = tag_tts_text(tts_text, emotion_tag)
        if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)) == 0:
            logger.debug("Empty TTS text, sending silent display payload")
            # Get current sequence number for silent payload
            current_sequence = self._sequence_counter
            self._sequence_counter += 1

            # Start sender task if not running
            if not self._sender_task or self._sender_task.done():
                self._sender_task = asyncio.create_task(
                    self._process_payload_queue(websocket_send)
                )

            await self._send_silent_payload(
                display_text,
                actions,
                current_sequence,
                history_text=history_text,
                before_content_send=before_content_send,
                after_content_send=after_content_send,
            )
            return

        logger.debug(
            "Queuing TTS task (characters={}, speaker_set={})",
            len(tts_text),
            bool(display_text.name),
        )

        # Get current sequence number
        current_sequence = self._sequence_counter
        self._sequence_counter += 1

        # Start sender task if not running
        if not self._sender_task or self._sender_task.done():
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )

        # Create and queue the TTS task
        task = asyncio.create_task(
            self._process_tts(
                tts_text=tts_text,
                display_text=display_text,
                actions=actions,
                live2d_model=live2d_model,
                tts_engine=tts_engine,
                sequence_number=current_sequence,
                synthesize_audio=synthesize_audio,
                history_text=history_text,
                before_content_send=before_content_send,
                after_content_send=after_content_send,
            )
        )
        self.task_list.append(task)

    @staticmethod
    def _emotion_tags_allowed(tts_engine: TTSInterface) -> bool:
        """True only for ElevenLabs engines with tags enabled (fail-soft)."""
        try:
            if not tts_engine.__class__.__module__.endswith(".elevenlabs_tts"):
                return False
            return bool(getattr(tts_engine, "emotion_tags_enabled", True))
        except Exception:
            return False

    async def _discard_pending_payloads(
        self,
        buffered_payloads: Dict[int, Tuple],
    ) -> None:
        """Release queue accounting after delivery is abandoned.

        ``asyncio.Queue.join()`` is used by the turn finalizer; every item
        taken with ``get()`` must get exactly one ``task_done()`` even when a
        guard or a dead socket stops the sender early. Otherwise the turn
        waits forever on a queue that will never drain.
        """
        for _ in buffered_payloads:
            self._payload_queue.task_done()
        buffered_payloads.clear()
        while True:
            try:
                self._payload_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._payload_queue.task_done()

    @staticmethod
    async def _notify_content_result(
        callback: Optional[Callable[[str, bool], Awaitable[None]]],
        history_text: str,
        accepted: bool,
    ) -> None:
        if callback is None:
            return
        try:
            await callback(history_text, accepted)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Content-send confirmation failed (accepted={})", accepted
            )

    async def _process_payload_queue(self, websocket_send: WebSocketSend) -> None:
        """
        Process and send payloads in correct order.
        Runs continuously until all payloads are processed.

        Each queued item may carry ``(payload, sequence, history_text,
        before_send, after_send)``. The two-tuple form remains supported for
        callers/tests that only exercise ordering.
        """
        buffered_payloads: Dict[int, Tuple] = {}

        while True:
            try:
                item = await self._payload_queue.get()
                if isinstance(item, tuple) and len(item) == 5:
                    (
                        payload,
                        sequence_number,
                        history_text,
                        before_content_send,
                        after_content_send,
                    ) = item
                else:
                    payload, sequence_number = item
                    history_text = ""
                    before_content_send = None
                    after_content_send = None
                buffered_payloads[sequence_number] = (
                    payload,
                    history_text or "",
                    before_content_send,
                    after_content_send,
                )

                # Send payloads in order
                while self._next_sequence_to_send in buffered_payloads:
                    (
                        next_payload,
                        frame_history_text,
                        before_send,
                        after_send,
                    ) = buffered_payloads.pop(self._next_sequence_to_send)

                    if self._content_blocked:
                        self._payload_queue.task_done()
                        await self._discard_pending_payloads(buffered_payloads)
                        return

                    # Stage 4.1: the canonical AI row must be durable before
                    # this content frame can leave the process. The guard
                    # writes the accumulated prefix (this delta included) and
                    # returns False if the row is not safely on disk.
                    if before_send is not None and frame_history_text:
                        try:
                            allowed = await before_send(frame_history_text)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception(
                                "Pre-send history guard failed; content blocked"
                            )
                            allowed = False
                        if not allowed:
                            self._content_blocked = True
                            self._payload_queue.task_done()
                            await self._discard_pending_payloads(
                                buffered_payloads
                            )
                            return

                    # Delivery-order sequence: this is the one place that
                    # knows the order frames actually leave the process.
                    from ..request_latency import get_latency_tracker

                    next_payload = _stamp_frame_sequence(
                        next_payload, get_latency_tracker()
                    )
                    try:
                        await websocket_send(json.dumps(next_payload))
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        # Dead socket mid-turn: the text is already
                        # accumulated by the caller, so abandon delivery and
                        # let the turn persist instead of dying here.
                        await self._notify_content_result(
                            after_send, frame_history_text, False
                        )
                        logger.debug(
                            "TTS payload send skipped (connection unavailable): "
                            "type={}",
                            type(error).__name__,
                        )
                        self._payload_queue.task_done()
                        await self._discard_pending_payloads(buffered_payloads)
                        return
                    await self._notify_content_result(
                        after_send, frame_history_text, True
                    )
                    self._next_sequence_to_send += 1

                self._payload_queue.task_done()

            except asyncio.CancelledError:
                break

    async def _send_silent_payload(
        self,
        display_text: DisplayText,
        actions: Optional[Actions],
        sequence_number: int,
        history_text: str = "",
        before_content_send: Optional[Callable[[str], Awaitable[bool]]] = None,
        after_content_send: Optional[
            Callable[[str, bool], Awaitable[None]]
        ] = None,
    ) -> None:
        """Queue a silent audio payload"""
        from ..request_latency import get_latency_tracker

        audio_payload = prepare_audio_payload(
            audio_path=None,
            display_text=display_text,
            actions=actions,
        )
        _stamp_turn_identity(audio_payload, get_latency_tracker())
        if self._content_blocked:
            return
        await self._payload_queue.put(
            (
                audio_payload,
                sequence_number,
                history_text,
                before_content_send,
                after_content_send,
            )
        )

    async def _process_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        sequence_number: int,
        synthesize_audio: bool = True,
        history_text: str = "",
        before_content_send: Optional[Callable[[str], Awaitable[bool]]] = None,
        after_content_send: Optional[
            Callable[[str, bool], Awaitable[None]]
        ] = None,
    ) -> None:
        """Process TTS generation and queue the result for ordered delivery"""
        from ..request_latency import get_latency_tracker

        tracker = get_latency_tracker()
        audio_file_path = None
        synthesis_started = time.perf_counter()
        try:
            if synthesize_audio:
                audio_file_path = await self._generate_audio(tts_engine, tts_text)
                if tracker:
                    tracker.add_tts_synthesis(
                        (time.perf_counter() - synthesis_started) * 1000
                    )
            # When synthesis is disabled we intentionally pass audio_path=None;
            # prepare_audio_payload emits a silent display payload carrying only
            # display_text + actions, preserving order and the turn lifecycle.
            payload = prepare_audio_payload(
                audio_path=audio_file_path,
                display_text=display_text,
                actions=actions,
            )
            _stamp_turn_identity(payload, tracker)
            # Queue the payload with its sequence number and the pre-send
            # history guard for THIS frame.
            if not self._content_blocked:
                await self._payload_queue.put(
                    (
                        payload,
                        sequence_number,
                        history_text,
                        before_content_send,
                        after_content_send,
                    )
                )

        except Exception as e:
            logger.error(f"Error preparing audio payload: {e}")
            # Queue silent payload for error case
            payload = prepare_audio_payload(
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            _stamp_turn_identity(payload, tracker)
            if not self._content_blocked:
                await self._payload_queue.put(
                    (
                        payload,
                        sequence_number,
                        history_text,
                        before_content_send,
                        after_content_send,
                    )
                )

        finally:
            if audio_file_path:
                tts_engine.remove_file(audio_file_path)
                logger.debug("Audio cache file cleaned.")

    async def _generate_audio(
        self,
        tts_engine: TTSInterface,
        text: str,
    ) -> str:
        """Generate audio file from text."""
        logger.debug("Generating TTS audio (characters={})", len(text))

        async def generate() -> str:
            return await tts_engine.async_generate_audio(
                text=text,
                file_name_no_ext=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
            )

        if tts_engine.__class__.__module__.endswith(".elevenlabs_tts"):
            async with _elevenlabs_semaphore():
                logger.debug("ElevenLabs TTS concurrency slot acquired")
                return await generate()

        return await generate()

    def clear(self) -> None:
        """Clear all pending tasks and reset state"""
        self.task_list.clear()
        if self._sender_task:
            self._sender_task.cancel()
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        self._content_blocked = False
        # Create a new queue to clear any pending items
        self._payload_queue = asyncio.Queue()
