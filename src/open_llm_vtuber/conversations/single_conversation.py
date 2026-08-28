from typing import Union, List, Dict, Any, Optional
import asyncio
import json
from loguru import logger
import numpy as np
import time
import uuid

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext
from ..request_latency import (
    RequestLatencyTracker,
    reset_latency_tracker,
    set_latency_tracker,
)

# Import necessary types from agent outputs
from ..agent.output_types import SentenceOutput, AudioOutput


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags

    Returns:
        str: Complete response text
    """
    # Create TTSTaskManager for this conversation
    tts_manager = TTSTaskManager()
    full_response = ""  # Initialize full_response here
    proactive = bool((metadata or {}).get("request_origin") == "proactive")
    llm = getattr(context.agent_engine, "_llm", None)
    base_url = str(getattr(llm, "base_url", ""))
    provider = "ollama_cloud" if "ollama.com" in base_url else type(llm).__name__
    latency = RequestLatencyTracker(
        websocket_send=websocket_send,
        provider=provider,
        model=str(getattr(llm, "model", "unknown")),
        request_origin="proactive" if proactive else "user",
        request_id=(metadata or {}).get("latency_request_id") or uuid.uuid4().hex,
        client_user_send_ms=(metadata or {}).get("client_user_send_ms"),
        client_websocket_send_ms=(metadata or {}).get("client_websocket_send_ms"),
    )
    latency_token = set_latency_tracker(latency)

    try:
        await latency.emit("backend-received")
        # Send initial signals
        await send_conversation_start_signals(websocket_send)
        latency.mark("websocket_first_output")
        logger.info(f"New Conversation Chain {session_emoji} started!")

        skip_history = bool(metadata and metadata.get("skip_history", False))
        input_text = ""
        batch_input = None
        if proactive:
            logger.info("Starting proactive assistant turn")
        else:
            # Process and persist a real user input exactly as before.
            input_text = await process_user_input(
                user_input, context.asr_engine, websocket_send
            )
            batch_input = create_batch_input(
                input_text=input_text,
                images=images,
                from_name=context.character_config.human_name,
                metadata=metadata,
            )

            if context.history_uid and not skip_history:
                save_started = time.perf_counter()
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="human",
                    content=input_text,
                    name=context.character_config.human_name,
                )
                latency.add_history_save((time.perf_counter() - save_started) * 1000)

            if skip_history:
                logger.debug("Skipping storing user input to history")

            logger.info("User input received (characters={})", len(input_text))
            if images:
                logger.info(f"With {len(images)} images")

        latency.mark("agent_start")
        try:
            if proactive:
                proactive_chat = getattr(context.agent_engine, "chat_proactively", None)
                if not callable(proactive_chat):
                    raise RuntimeError(
                        "Active conversation agent does not support proactive chat"
                    )
                agent_output_stream = proactive_chat(
                    followup_context=(metadata or {}).get("proactive_followup")
                )
            else:
                # agent.chat yields Union[SentenceOutput, Dict[str, Any]]
                agent_output_stream = context.agent_engine.chat(batch_input)

            async for output_item in agent_output_stream:
                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "tool_call_status"
                ):
                    # Handle tool status event: send WebSocket message
                    output_item["name"] = context.character_config.character_name
                    output_item["request_id"] = latency.request_id
                    logger.debug(f"Sending tool status update: {output_item}")

                    await websocket_send(json.dumps(output_item))

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    # Handle SentenceOutput or AudioOutput
                    tts_started = time.perf_counter()
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.tts_engine,
                        websocket_send=websocket_send,  # Pass websocket_send for audio/tts messages
                        tts_manager=tts_manager,
                        translate_engine=context.translate_engine,
                    )
                    latency.add_tts_enqueue(
                        (time.perf_counter() - tts_started) * 1000
                    )
                    # Ensure response_part is treated as a string before concatenation
                    response_part_str = (
                        str(response_part) if response_part is not None else ""
                    )
                    full_response += response_part_str  # Accumulate text response
                else:
                    logger.warning(
                        f"Received unexpected item type from agent chat stream: {type(output_item)}"
                    )
                    logger.debug(
                        "Unexpected item type from agent chat stream (content omitted)"
                    )

        except Exception as e:
            logger.exception(
                f"Error processing agent response stream: {e}"
            )  # Log with stack trace
            await websocket_send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                )
            )
            # full_response will contain partial response before error
        latency.mark("agent_end")
        # --- End processing agent response ---

        # Wait for any pending TTS tasks
        if tts_manager.task_list:
            latency.mark("tts_wait_start")
            await asyncio.gather(*tts_manager.task_list)
            latency.mark("tts_wait_end")
            latency.add_tts_wait(
                latency.phase_duration("tts_wait_start", "tts_wait_end") or 0.0
            )
            await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )

        if context.history_uid and full_response:  # Check full_response before storing
            save_started = time.perf_counter()
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            latency.add_history_save((time.perf_counter() - save_started) * 1000)
            logger.info("AI response completed (characters={})", len(full_response))
            if not skip_history and not proactive:
                observer = getattr(
                    context.agent_engine,
                    "observe_character_events",
                    None,
                )
                if observer is None:
                    observer = getattr(
                        context.agent_engine, "observe_relationship_event", None
                    )
                if observer is not None:
                    event_started = time.perf_counter()
                    observer(input_text, full_response)
                    latency.add_time(
                        "character_event_ms",
                        (time.perf_counter() - event_started) * 1000,
                    )

        latency.mark("websocket_final_output")
        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        latency.interrupted = True
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        latency.internal_error = type(e).__name__
        logger.error(f"Error in conversation chain: {e}")
        await websocket_send(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        try:
            await latency.complete()
        except Exception as latency_error:
            logger.warning(
                "Latency finalization failed: type={}",
                type(latency_error).__name__,
            )
        reset_latency_tracker(latency_token)
        cleanup_conversation(tts_manager, session_emoji)
