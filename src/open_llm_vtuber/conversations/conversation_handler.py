import asyncio
import json
from typing import Dict, Optional, Callable

import numpy as np
from fastapi import WebSocket
from loguru import logger

from ..chat_group import ChatGroupManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext
from .group_conversation import process_group_conversation
from .single_conversation import process_single_conversation
from .conversation_utils import EMOJI_LIST
from .types import GroupConversationState
from prompts import prompt_loader


async def handle_conversation_trigger(
    msg_type: str,
    data: dict,
    client_uid: str,
    context: ServiceContext,
    websocket: WebSocket,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    chat_group_manager: ChatGroupManager,
    received_data_buffers: Dict[str, np.ndarray],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    broadcast_to_group: Callable,
    detached_turns: Optional[Dict[str, asyncio.Task]] = None,
) -> None:
    """Handle triggers that start a conversation"""
    metadata = None

    if msg_type == "ai-speak-signal":
        try:
            # Get proactive speak prompt from config
            prompt_name = "proactive_speak_prompt"
            prompt_file = context.system_config.tool_prompts.get(prompt_name)
            if prompt_file:
                user_input = prompt_loader.load_util(prompt_file)
            else:
                logger.warning("Proactive speak prompt not configured, using default")
                user_input = "Please say something."
        except Exception as e:
            logger.error(f"Error loading proactive speak prompt: {e}")
            user_input = "Please say something."

        # Add metadata to indicate this is a proactive speak request
        # that should be skipped in both memory and history
        metadata = {
            "proactive_speak": True,
            "skip_memory": True,  # Skip storing in AI's internal memory
            "skip_history": True,  # Skip storing in local conversation history
        }

        await websocket.send_text(
            json.dumps(
                {
                    "type": "full-text",
                    "text": "AI wants to speak something...",
                }
            )
        )
    elif msg_type == "text-input":
        user_input = data.get("text", "")
        metadata = {
            "latency_request_id": data.get("request_id"),
            "client_user_send_ms": data.get("client_user_send_ms"),
            "client_websocket_send_ms": data.get("client_websocket_send_ms"),
        }
    else:  # mic-audio-end
        user_input = received_data_buffers[client_uid]
        received_data_buffers[client_uid] = np.array([])
        if isinstance(user_input, np.ndarray) and user_input.size == 0:
            # Empty voice turn (silence/misfire/double-end): never start
            # a ghost conversation on zero audio.
            logger.info("Ignoring empty mic-audio-end: no audio buffered")
            return

    images = data.get("images")
    session_emoji = np.random.choice(EMOJI_LIST)

    group = chat_group_manager.get_client_group(client_uid)
    if group and len(group.members) > 1:
        # Use group_id as task key for group conversations. Same
        # one-active-turn rule as the single path: cancel the in-flight
        # group turn so a new trigger is never silently dropped and two
        # turns never run in parallel.
        task_key = group.group_id
        previous = current_conversation_tasks.get(task_key)
        if previous is not None and not previous.done():
            logger.info(
                "Cancelling in-flight group turn for new trigger: group_id={}",
                task_key,
            )
            previous.cancel()
        else:
            logger.info(f"Starting new group conversation for {task_key}")

        current_conversation_tasks[task_key] = asyncio.create_task(
            process_group_conversation(
                client_contexts=client_contexts,
                client_connections=client_connections,
                broadcast_func=broadcast_to_group,
                group_members=group.members,
                initiator_client_uid=client_uid,
                user_input=user_input,
                images=images,
                session_emoji=session_emoji,
                metadata=metadata,
            )
        )
    else:
        # Use client_uid as task key for individual conversations.
        # ONE ACTIVE TURN PER CLIENT: cancel any in-flight turn first so a
        # new trigger never runs parallel to the old one (no duplicate
        # history, no interleaved TTS). The cancelled task cleans its own
        # TTS queue in `finally` and stores no AI response (CancelledError
        # skips the persist step); already-sent audio cannot be recalled.
        previous = current_conversation_tasks.get(client_uid)
        if previous is not None and not previous.done():
            logger.info("Cancelling in-flight turn for new trigger")
            previous.cancel()
        turn_history_uid = getattr(context, "history_uid", "") or ""
        # A turn detached by a mid-turn disconnect (see handle_disconnect)
        # keeps running so its response still persists. If THIS trigger
        # targets the same history, wait for the orphan first: two turns
        # must never persist into one transcript concurrently.
        # Same-client replacement above already cancelled, so anything
        # awaited here belongs to a dead connection.
        orphan_candidates = []
        if turn_history_uid and detached_turns is not None:
            orphan = detached_turns.get(turn_history_uid)
            if orphan is not None and not orphan.done():
                orphan_candidates.append(orphan)
        for other_key, other_task in list(current_conversation_tasks.items()):
            if (
                other_key != client_uid
                and other_task is not None
                and not other_task.done()
                and getattr(other_task, "_olv_detached", False)
                and getattr(other_task, "_olv_history_uid", "") == turn_history_uid
                and turn_history_uid
            ):
                orphan_candidates.append(other_task)
        for orphan_task in orphan_candidates:
            logger.info(
                "Waiting for detached in-flight turn on the same history "
                "before starting the new trigger"
            )
            try:
                await asyncio.shield(orphan_task)
            except asyncio.CancelledError:
                # Our own cancellation must propagate; the orphan itself may
                # legitimately have been cancelled, so shield and re-check.
                if orphan_task.cancelled():
                    continue
                raise
            except Exception as error:
                # A dead orphan never blocks the new turn.
                logger.debug(
                    "Detached turn finished with: type={}",
                    type(error).__name__,
                )
        new_task = asyncio.create_task(
            process_single_conversation(
                context=context,
                websocket_send=websocket.send_text,
                client_uid=client_uid,
                user_input=user_input,
                images=images,
                session_emoji=session_emoji,
                metadata=metadata,
            )
        )
        # Lifecycle tags (same task object, no new structures): what session
        # this turn belongs to, and whether its owner disconnected mid-turn.
        new_task._olv_single_turn = True
        new_task._olv_history_uid = turn_history_uid
        new_task._olv_detached = False
        current_conversation_tasks[client_uid] = new_task


async def handle_individual_interrupt(
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    context: ServiceContext,
    heard_response: str,
):
    if client_uid in current_conversation_tasks:
        task = current_conversation_tasks[client_uid]
        if task and not task.done():
            task.cancel()
            logger.info("🛑 Conversation task was successfully interrupted")

        try:
            context.agent_engine.handle_interrupt(heard_response)
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")

        if context.history_uid:
            # Only persist the AI turn when something was actually spoken. The
            # client sends `interrupt-signal` when a new message arrives
            # mid-turn, so an interrupt raised before the first sentence leaves
            # `heard_response` empty; storing that produced an empty assistant
            # turn in the transcript (fed back into the next LLM context and
            # rendered as a blank bubble). Same guard as the normal completion
            # path in single_conversation (`if ... and full_response`). The
            # "[Interrupted by user]" marker is always kept so the transcript
            # still records that the turn was cut short.
            if heard_response and heard_response.strip():
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="ai",
                    content=heard_response,
                    name=context.character_config.character_name,
                    avatar=context.character_config.avatar,
                )
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="system",
                content="[Interrupted by user]",
            )


async def handle_group_interrupt(
    group_id: str,
    heard_response: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    chat_group_manager: ChatGroupManager,
    client_contexts: Dict[str, ServiceContext],
    broadcast_to_group: Callable,
) -> None:
    """Handles interruption for a group conversation"""
    task = current_conversation_tasks.get(group_id)
    if not task or task.done():
        return

    # Get state and speaker info before cancellation
    state = GroupConversationState.get_state(group_id)
    current_speaker_uid = state.current_speaker_uid if state else None

    # Get context from current speaker
    context = None
    group = chat_group_manager.get_group_by_id(group_id)
    if current_speaker_uid:
        context = client_contexts.get(current_speaker_uid)
        logger.info(f"Found current speaker context for {current_speaker_uid}")
    if not context and group and group.members:
        logger.warning(f"No context found for group {group_id}, using first member")
        context = client_contexts.get(next(iter(group.members)))

    # Now cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"🛑 Group conversation {group_id} cancelled successfully.")

    current_conversation_tasks.pop(group_id, None)
    GroupConversationState.remove_state(group_id)  # Clean up state after we've used it

    # Store messages with speaker info
    if context and group:
        for member_uid in group.members:
            if member_uid in client_contexts:
                try:
                    member_ctx = client_contexts[member_uid]
                    member_ctx.agent_engine.handle_interrupt(heard_response)
                    # Skip empty assistant turns (see the individual path for
                    # the rationale); the interruption marker is always stored.
                    if heard_response and heard_response.strip():
                        store_message(
                            conf_uid=member_ctx.character_config.conf_uid,
                            history_uid=member_ctx.history_uid,
                            role="ai",
                            content=heard_response,
                            name=context.character_config.character_name,
                            avatar=context.character_config.avatar,
                        )
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="system",
                        content="[Interrupted by user]",
                    )
                except Exception as e:
                    logger.error(f"Error handling interrupt for {member_uid}: {e}")

    await broadcast_to_group(
        list(group.members),
        {
            "type": "interrupt-signal",
            "text": "conversation-interrupted",
        },
    )
