import asyncio
import base64
import binascii
import re
from typing import Awaitable, Callable, Optional, Union, Any, List, Dict, Tuple
import numpy as np
import json
from loguru import logger

from ..message_handler import message_handler
from .types import WebSocketSend, BroadcastContext
from .tts_manager import (
    TTSTaskManager,
    _stamp_frame_sequence,
    _stamp_turn_identity,
)
from ..agent.output_types import SentenceOutput, AudioOutput
from ..agent.input_types import BatchInput, TextData, ImageData, TextSource, ImageSource
from ..asr.asr_interface import ASRInterface
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload


# Upper bound for waiting on the frontend playback-complete signal. The
# frontend emits it after full audio playback; if it never arrives (dead
# socket, missed audio), the turn must still finish its lifecycle instead
# of hanging forever and blocking the receive loop via proactive shield.
PLAYBACK_COMPLETE_TIMEOUT_S = 120.0


class ContentPersistError(RuntimeError):
    """A content frame was withheld because its history row was not durable."""


async def safe_send(websocket_send: WebSocketSend, payload: str) -> bool:
    """Best-effort lifecycle send; never raises, never retries.

    A dead socket must not kill turn finalization: log once and let the
    caller continue the lifecycle (chain-end etc.). No retry — redelivery
    to a dead socket only produces duplicates or cascaded failures.
    """
    try:
        await websocket_send(payload)
        return True
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.warning(
            "Lifecycle send skipped (connection unavailable): type={}",
            type(error).__name__,
        )
        return False


# Convert class methods to standalone functions
# Multi-attachment limits for text-input images. The frontend enforces the
# same caps pre-send; the server re-checks authoritatively because clients
# are untrusted. Wire budget: aggregate raw bytes * 4/3 (base64) must stay
# well under uvicorn's 16 MiB ws-max-size for the whole JSON message.
MAX_IMAGES_PER_MESSAGE = 10
MAX_IMAGE_FILE_BYTES = 5 * 1024 * 1024
MAX_IMAGES_TOTAL_BYTES = 10 * 1024 * 1024

_VALID_IMAGE_SOURCES = frozenset({"camera", "screen", "clipboard", "upload"})


def _decoded_data_url_bytes(data_url: str) -> Optional[int]:
    """Decoded byte length of a data-URL payload, or None if undecodable."""
    try:
        comma = data_url.index(",")
        payload = data_url[comma + 1 :]
        if not payload:
            return None
        return len(base64.b64decode(payload, validate=True))
    except (ValueError, binascii.Error):
        return None


def sanitize_images(
    images: Any, request_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split text-input attachments into (valid, errors).

    Every accepted entry keeps its original dict untouched (including any
    client `name`/`size` keys) and its position, so ordering is preserved
    and same-name/same-content files can never overwrite each other — there
    is no name- or content-keyed lookup anywhere on this path.

    Each error is {"index", "name", "reason"} with reason in:
    not-a-list, not-a-dict, bad-source, unsupported-type, bad-data,
    too-large, too-many, total-too-large. Callers report errors to the user
    (error event) and continue the turn with the valid subset; they must
    never silently claim skipped files were processed. Pure w.r.t. the
    turn: per-message state only, nothing stored, nothing written to disk.
    """
    if images is None:
        return [], []
    if not isinstance(images, list):
        logger.warning(
            "Attachments rejected (request_id={}): 'images' is {}, not a list",
            request_id,
            type(images).__name__,
        )
        return [], [{"index": -1, "name": "images", "reason": "not-a-list"}]
    valid: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    total_bytes = 0
    for index, entry in enumerate(images):
        fallback = f"image #{index + 1}"
        if not isinstance(entry, dict):
            errors.append({"index": index, "name": fallback, "reason": "not-a-dict"})
            continue
        name = entry.get("name")
        label = name if isinstance(name, str) and name else fallback
        if entry.get("source") not in _VALID_IMAGE_SOURCES:
            errors.append({"index": index, "name": label, "reason": "bad-source"})
            continue
        mime = entry.get("mime_type")
        if not isinstance(mime, str) or not mime.startswith("image/"):
            errors.append({"index": index, "name": label, "reason": "unsupported-type"})
            continue
        data = entry.get("data")
        if not isinstance(data, str) or not data.startswith("data:image/"):
            errors.append({"index": index, "name": label, "reason": "bad-data"})
            continue
        size = _decoded_data_url_bytes(data)
        if size is None:
            errors.append({"index": index, "name": label, "reason": "bad-data"})
            continue
        if size > MAX_IMAGE_FILE_BYTES:
            errors.append({"index": index, "name": label, "reason": "too-large"})
            continue
        if len(valid) >= MAX_IMAGES_PER_MESSAGE:
            errors.append({"index": index, "name": label, "reason": "too-many"})
            continue
        if total_bytes + size > MAX_IMAGES_TOTAL_BYTES:
            errors.append({"index": index, "name": label, "reason": "total-too-large"})
            continue
        total_bytes += size
        valid.append(entry)
    if errors:
        logger.warning(
            "Attachments skipped (request_id={}): {}",
            request_id,
            [(e["name"], e["reason"]) for e in errors],
        )
    return valid, errors


def create_batch_input(
    input_text: str,
    images: Optional[List[Dict[str, Any]]],
    from_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> BatchInput:
    """Create batch input for agent processing"""
    return BatchInput(
        texts=[
            TextData(source=TextSource.INPUT, content=input_text, from_name=from_name)
        ],
        images=[
            ImageData(
                source=ImageSource(img["source"]),
                data=img["data"],
                mime_type=img["mime_type"],
            )
            for img in (images or [])
        ]
        if images
        else None,
        metadata=metadata,
    )


async def process_agent_output(
    output: Union[AudioOutput, SentenceOutput],
    character_config: Any,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
    synthesize_audio: bool = True,
    before_content_send: Optional[Callable[[str], Awaitable[bool]]] = None,
    after_content_send: Optional[
        Callable[[str, bool], Awaitable[None]]
    ] = None,
) -> str:
    """Process agent output with character information and optional translation"""
    output.display_text.name = character_config.character_name
    output.display_text.avatar = character_config.avatar

    full_response = ""
    try:
        if isinstance(output, SentenceOutput):
            full_response = await handle_sentence_output(
                output,
                live2d_model,
                tts_engine,
                websocket_send,
                tts_manager,
                translate_engine,
                synthesize_audio=synthesize_audio,
                before_content_send=before_content_send,
                after_content_send=after_content_send,
            )
        elif isinstance(output, AudioOutput):
            full_response = await handle_audio_output(
                output,
                websocket_send,
                before_content_send=before_content_send,
                after_content_send=after_content_send,
            )
        else:
            logger.warning(f"Unknown output type: {type(output)}")
    except ContentPersistError as error:
        # The guard already recorded the failure. Do not send this frame (or
        # any later one); the turn will report one explicit persistence error
        # and suppress ai-final.
        logger.error("Content withheld: {}", error)
    except Exception as e:
        logger.error(f"Error processing agent output: {e}")
        await websocket_send(
            json.dumps(
                {"type": "error", "message": f"Error processing response: {str(e)}"}
            )
        )

    return full_response


async def handle_sentence_output(
    output: SentenceOutput,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
    synthesize_audio: bool = True,
    before_content_send: Optional[Callable[[str], Awaitable[bool]]] = None,
    after_content_send: Optional[
        Callable[[str, bool], Awaitable[None]]
    ] = None,
) -> str:
    """Handle sentence output type with optional translation support"""
    full_response = ""
    async for display_text, tts_text, actions in output:
        logger.debug("Processing TTS output (characters={})", len(tts_text))

        if translate_engine:
            if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)):
                tts_text = translate_engine.translate(tts_text)
            logger.info("TTS translation completed (characters={})", len(tts_text))
        else:
            logger.debug("🚫 No translation engine available. Skipping translation.")

        # Voice Emotion: per-sentence tag from THIS sentence's own detected
        # emotions (stateless — a previous turn/sentence can never leak in).
        # Only the synthesis text carries it; display/history/translation
        # above are untouched.
        emotion_tag = None
        try:
            from ..voice_emotion import emotion_tag_for

            emotion_tag = emotion_tag_for(getattr(actions, "emotions", None))
        except Exception:
            emotion_tag = None

        full_response += display_text.text
        await tts_manager.speak(
            tts_text=tts_text,
            display_text=display_text,
            actions=actions,
            live2d_model=live2d_model,
            tts_engine=tts_engine,
            websocket_send=websocket_send,
            synthesize_audio=synthesize_audio,
            emotion_tag=emotion_tag,
            history_text=display_text.text,
            before_content_send=before_content_send,
            after_content_send=after_content_send,
        )
    return full_response


async def handle_audio_output(
    output: AudioOutput,
    websocket_send: WebSocketSend,
    before_content_send: Optional[Callable[[str], Awaitable[bool]]] = None,
    after_content_send: Optional[
        Callable[[str, bool], Awaitable[None]]
    ] = None,
) -> str:
    """Process and send AudioOutput directly to the client"""
    from ..request_latency import get_latency_tracker

    full_response = ""
    async for audio_path, display_text, transcript, actions in output:
        frame_text = str(transcript or "")
        # Direct output has no TTS queue, so this block is its delivery
        # boundary. The canonical row must exist before the frame can go out.
        if before_content_send is not None and frame_text:
            try:
                allowed = await before_content_send(frame_text)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("Pre-send history guard failed: {}", error)
                allowed = False
            if not allowed:
                raise ContentPersistError(
                    "AI text was not persisted before direct audio send"
                )

        audio_payload = prepare_audio_payload(
            audio_path=audio_path,
            display_text=display_text,
            actions=actions.to_dict() if actions else None,
        )
        tracker = get_latency_tracker()
        _stamp_turn_identity(audio_payload, tracker)
        # Sent directly (no TTS queue), so this is already the delivery point.
        _stamp_frame_sequence(audio_payload, tracker)
        try:
            await websocket_send(json.dumps(audio_payload))
        except asyncio.CancelledError:
            raise
        except Exception:
            if after_content_send is not None:
                try:
                    await after_content_send(frame_text, False)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Direct-audio rollback callback failed")
            raise
        else:
            if after_content_send is not None:
                try:
                    await after_content_send(frame_text, True)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Direct-audio confirmation callback failed")
        full_response += transcript
    return full_response


async def send_conversation_start_signals(
    websocket_send: WebSocketSend,
    history_uid: str = "",
    request_id: str = "",
    msg_id: str = "",
    attempt: int = 1,
) -> None:
    """Send initial conversation signals (best-effort; see safe_send).

    request_id identifies THIS attempt; msg_id identifies the LOGICAL
    assistant message and is stable across retries. The frontend opens the
    bubble on msg_id (falling back to request_id for old servers), so both
    travel together on every frame of the turn.

    The delivery-protocol aliases (``id``/``turn``/``attempt``) are additive:
    a client that predates them reads exactly the keys it always did.
    """
    payload: Dict[str, Any] = {
        "type": "control",
        "text": "conversation-chain-start",
    }
    if history_uid:
        payload["history_uid"] = history_uid
    if request_id:
        payload["request_id"] = request_id
        payload["turn"] = request_id
    if msg_id:
        payload["msg_id"] = msg_id
        payload["id"] = msg_id
        payload["attempt"] = int(attempt or 1)
    await safe_send(
        websocket_send,
        json.dumps(payload),
    )


async def send_canonical_final(
    websocket_send: WebSocketSend,
    history_uid: str,
    request_id: str,
    text: str,
    msg_id: str = "",
    final_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Deliver the persisted canonical AI text for one completed turn.

    The persisted row is authoritative; the frontend reconciles the live
    bubble(s) carrying request_id to exactly this text (idempotent: a
    repeated event changes nothing). Best-effort like every other signal:
    a dead socket must never break the turn. Never called when nothing was
    persisted (cancelled/skipped turns send no canonical event).

    This frame CLOSES the message for the delivery protocol: it carries
    ``final: true`` plus the attempt and the content-frame count, which is
    what a client ACK must name to settle the message. Clients that do not
    speak the protocol ignore the extra keys and behave exactly as before.

    Returns whether the transport accepted the frame. A recovery caller must
    not count a refused send as a delivered redelivery.
    """
    if not history_uid or not request_id or not text:
        return False
    # NOTE: routed by `type`, not by `text`: the canonical response text
    # itself travels in `text`, so the control-text routing slot must not
    # be reused for it (a duplicate "text" key would destroy the route).
    frame: Dict[str, Any] = {
        "type": "ai-final",
        "history_uid": history_uid,
        "request_id": request_id,
        "turn": request_id,
        "text": text,
    }
    if msg_id:
        frame["msg_id"] = msg_id
        frame["id"] = msg_id
    if final_metadata:
        frame.update(final_metadata)
    return await safe_send(websocket_send, json.dumps(frame))


async def process_user_input(
    user_input: Union[str, np.ndarray],
    asr_engine: ASRInterface,
    websocket_send: WebSocketSend,
) -> str:
    """Process user input, converting audio to text if needed"""
    if isinstance(user_input, np.ndarray):
        logger.info("Transcribing audio input...")
        input_text = await asr_engine.async_transcribe_np(user_input)
        await websocket_send(
            json.dumps({"type": "user-input-transcription", "text": input_text})
        )
        return input_text
    return user_input


def _normalize_playback_outcome(value: Any) -> Optional[str]:
    """Return a known frontend outcome, or None for a legacy/unknown frame."""
    if not isinstance(value, str):
        return None
    outcome = value.strip().lower()
    if outcome in {"played", "failed", "skipped", "text_only", "interrupted"}:
        return outcome
    return None


def _playback_failed_segments(value: Any) -> int:
    """Read the frontend's failure count without trusting its type."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


async def finalize_conversation_turn(
    tts_manager: TTSTaskManager,
    websocket_send: WebSocketSend,
    client_uid: str,
    broadcast_ctx: Optional[BroadcastContext] = None,
) -> None:
    """Finalize a conversation turn"""
    from ..request_latency import get_latency_tracker

    tracker = get_latency_tracker()
    # Identity is optional: group turns and tests may run without a tracker.
    # When it exists, all three fields came from the same turn that produced
    # the audio frames, so the frontend can echo them on completion.
    request_id = str(getattr(tracker, "request_id", "") or "").strip() or None
    msg_id = str(getattr(tracker, "msg_id", "") or "").strip() or None
    try:
        attempt = int(getattr(tracker, "msg_attempt", 1) or 1)
    except (TypeError, ValueError):
        attempt = 1

    if tts_manager.task_list:
        await asyncio.gather(*tts_manager.task_list)
        # Keep the legacy `type` and add identity as optional fields. An old
        # frontend ignores them and still sends its untagged completion; a
        # new frontend echoes them so this backend can reject stale turns.
        synth_complete: Dict[str, Any] = {"type": "backend-synth-complete"}
        if request_id:
            synth_complete["request_id"] = request_id
            synth_complete["turn"] = request_id
        if msg_id:
            synth_complete["msg_id"] = msg_id
            synth_complete["id"] = msg_id
            synth_complete["attempt"] = attempt
        await safe_send(websocket_send, json.dumps(synth_complete))

        if tracker:
            tracker.mark("playback_start")
        response = await message_handler.wait_for_response(
            client_uid,
            "frontend-playback-complete",
            request_id=request_id,
            timeout=PLAYBACK_COMPLETE_TIMEOUT_S,
            accept_untagged_legacy=bool(request_id),
        )
        if tracker:
            tracker.mark("playback_end")
            tracker.add_playback_wait(
                tracker.phase_duration("playback_start", "playback_end") or 0.0
            )

        if not response:
            # Frontend never confirmed playback (dead socket or missed
            # audio). The turn still ends: emit the lifecycle tail so no
            # task hangs and the next trigger starts clean.
            if tracker:
                tracker.playback_outcome = "timeout"
            logger.warning(
                "No playback completion response from {} "
                "(timeout={}s); ending turn anyway",
                client_uid,
                PLAYBACK_COMPLETE_TIMEOUT_S,
            )
        else:
            outcome = _normalize_playback_outcome(
                response.get("audio_outcome")
            )
            failed_segments = _playback_failed_segments(
                response.get("audio_failed_segments")
            )
            # Fail closed: a frame that admits failed segments may never be
            # reported as a successful playback, even if its aggregate label
            # is contradictory.
            if failed_segments and outcome != "failed":
                outcome = "failed"
            if tracker:
                tracker.playback_outcome = outcome or "unknown"
                tracker.playback_failed_segments = failed_segments

            if outcome == "failed":
                reason = str(
                    response.get("audio_failure_reason") or "unknown"
                )[:160]
                logger.warning(
                    "AUDIO_PLAYBACK_FAILED client_uid={} request_id={} "
                    "msg_id={} failed_segments={} reason={}",
                    client_uid,
                    request_id or "-",
                    msg_id or "-",
                    failed_segments,
                    reason,
                )
            elif outcome == "played":
                logger.info(
                    "AUDIO_PLAYBACK_COMPLETE client_uid={} request_id={} "
                    "msg_id={} played_segments={}",
                    client_uid,
                    request_id or "-",
                    msg_id or "-",
                    response.get("audio_played_segments", "-"),
                )
            elif outcome is None:
                # Old servers send no identity, so old clients send no
                # outcome. Compatibility means accepting that frame, not
                # inventing an outcome that was never reported.
                logger.debug(
                    "Playback completion from {} has no audio_outcome "
                    "(legacy client); treating playback as unverified",
                    client_uid,
                )
            else:
                logger.info(
                    "AUDIO_PLAYBACK_NOT_PLAYED client_uid={} request_id={} "
                    "msg_id={} outcome={}",
                    client_uid,
                    request_id or "-",
                    msg_id or "-",
                    outcome,
                )

    await safe_send(websocket_send, json.dumps({"type": "force-new-message"}))

    if broadcast_ctx and broadcast_ctx.broadcast_func:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            {"type": "force-new-message"},
            broadcast_ctx.current_client_uid,
        )

    await send_conversation_end_signal(websocket_send, broadcast_ctx)


async def send_conversation_end_signal(
    websocket_send: WebSocketSend,
    broadcast_ctx: Optional[BroadcastContext],
    session_emoji: str = "😊",
) -> None:
    """Send conversation chain end signal"""
    chain_end_msg = {
        "type": "control",
        "text": "conversation-chain-end",
    }

    await safe_send(websocket_send, json.dumps(chain_end_msg))

    if broadcast_ctx and broadcast_ctx.broadcast_func and broadcast_ctx.group_members:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            chain_end_msg,
        )

    logger.info(f"😎👍✅ Conversation Chain {session_emoji} completed!")


def cleanup_conversation(tts_manager: TTSTaskManager, session_emoji: str) -> None:
    """Clean up conversation resources"""
    tts_manager.clear()
    logger.debug(f"🧹 Clearing up conversation {session_emoji}.")


EMOJI_LIST = [
    "🐶",
    "🐱",
    "🐭",
    "🐹",
    "🐰",
    "🦊",
    "🐻",
    "🐼",
    "🐨",
    "🐯",
    "🦁",
    "🐮",
    "🐷",
    "🐸",
    "🐵",
    "🐔",
    "🐧",
    "🐦",
    "🐤",
    "🐣",
    "🐥",
    "🦆",
    "🦅",
    "🦉",
    "🦇",
    "🐺",
    "🐗",
    "🐴",
    "🦄",
    "🐝",
    "🌵",
    "🎄",
    "🌲",
    "🌳",
    "🌴",
    "🌱",
    "🌿",
    "☘️",
    "🍀",
    "🍂",
    "🍁",
    "🍄",
    "🌾",
    "💐",
    "🌹",
    "🌸",
    "🌛",
    "🌍",
    "⭐️",
    "🔥",
    "🌈",
    "🌩",
    "⛄️",
    "🎃",
    "🎄",
    "🎉",
    "🎏",
    "🎗",
    "🀄️",
    "🎭",
    "🎨",
    "🧵",
    "🪡",
    "🧶",
    "🥽",
    "🥼",
    "🦺",
    "👔",
    "👕",
    "👜",
    "👑",
]
