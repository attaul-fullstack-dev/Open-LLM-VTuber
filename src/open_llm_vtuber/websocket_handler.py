from typing import Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
from enum import Enum
import numpy as np
from loguru import logger

from .service_context import ServiceContext
from .chat_group import (
    ChatGroupManager,
    handle_group_operation,
    handle_client_disconnect,
    broadcast_to_group,
)
from .message_handler import message_handler
from .utils.stream_audio import prepare_audio_payload
from .chat_history_manager import (
    create_new_history,
    get_history,
    delete_history,
    get_history_list,
    update_metadate,
)
from .config_manager.utils import scan_config_alts_directory, scan_bg_directory
from .conversations.conversation_handler import (
    handle_conversation_trigger,
    handle_group_interrupt,
    handle_individual_interrupt,
)
from .conversations.single_conversation import process_single_conversation
from .conversations.conversation_utils import EMOJI_LIST
from .proactive_chat import (
    ProactiveChatConfig,
    ProactiveRuntimeState,
    ProactiveStateMachine,
)


class MessageType(Enum):
    """Enum for WebSocket message types"""

    GROUP = ["add-client-to-group", "remove-client-from-group"]
    HISTORY = [
        "fetch-history-list",
        "fetch-and-set-history",
        "create-new-history",
        "delete-history",
        "reset-relationship",
        "compact-conversation",
        "rename-history",
        "fetch-character-memory",
        "delete-character-memory",
        "reset-character-memory",
        "reset-character-state",
    ]
    CONVERSATION = ["mic-audio-end", "text-input", "ai-speak-signal"]
    CONFIG = ["fetch-configs", "switch-config"]
    CONTROL = ["interrupt-signal", "audio-play-start"]
    DATA = ["mic-audio-data"]


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    images: Optional[List[str]]
    history_uid: Optional[str]
    file: Optional[str]
    display_text: Optional[dict]


def create_locked_send_text(websocket: WebSocket):
    """Serialize all ``send_text`` calls on one connection.

    Multiple coroutines write to the same WebSocket: the background
    conversation task, the TTS payload sender task, and the receive loop
    (interrupts/errors). When the transport write buffer is paused (slow
    client, large audio payloads), two concurrent drains race inside
    websockets' legacy protocol and its bare ``assert waiter is None or
    waiter.cancelled()`` raises ``AssertionError`` with an empty message.
    Serializing sends removes that race without reordering messages beyond
    the lock itself.
    """
    send_lock = asyncio.Lock()
    original_send_text = websocket.send_text

    async def locked_send_text(message: str) -> None:
        async with send_lock:
            await original_send_text(message)

    return locked_send_text


class WebSocketHandler:
    """Handles WebSocket connections and message routing"""

    def __init__(self, default_context_cache: ServiceContext):
        """Initialize the WebSocket handler with default context"""
        self.client_connections: Dict[str, WebSocket] = {}
        self.client_contexts: Dict[str, ServiceContext] = {}
        self.chat_group_manager = ChatGroupManager()
        self.current_conversation_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}
        self._proactive_timer_tasks: Dict[str, asyncio.Task] = {}
        self._proactive_states: Dict[str, Dict[str, ProactiveRuntimeState]] = {}
        self._proactive_machines: Dict[str, ProactiveStateMachine] = {}
        self._proactive_maintenance: set[str] = set()

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    @staticmethod
    def _proactive_config(context: ServiceContext) -> ProactiveChatConfig:
        settings = (
            context.character_config.agent_config.agent_settings.basic_memory_agent
        )
        return ProactiveChatConfig(
            enabled=settings.proactive_enabled,
            initial_idle_min_seconds=settings.initial_idle_min_seconds,
            initial_idle_max_seconds=settings.initial_idle_max_seconds,
            followup_idle_min_seconds=settings.followup_idle_min_seconds,
            followup_idle_max_seconds=settings.followup_idle_max_seconds,
            ignored_before_backoff=settings.ignored_before_backoff,
            backoff_min_seconds=settings.backoff_min_seconds,
            backoff_max_seconds=settings.backoff_max_seconds,
        )

    async def _cancel_proactive_timer(self, client_uid: str) -> None:
        task = self._proactive_timer_tasks.pop(client_uid, None)
        if not task or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _pause_proactive_for_maintenance(self, client_uid: str) -> None:
        self._proactive_maintenance.add(client_uid)
        await self._cancel_proactive_timer(client_uid)

    async def _resume_proactive_after_maintenance(
        self, client_uid: str, context: ServiceContext
    ) -> None:
        self._proactive_maintenance.discard(client_uid)
        await self._activate_proactive_for_history(client_uid, context.history_uid)

    async def _activate_proactive_for_history(
        self,
        client_uid: str,
        history_uid: Optional[str],
        *,
        user_activity: bool = True,
    ) -> None:
        """Start one efficient randomized timer for the active chat."""
        await self._cancel_proactive_timer(client_uid)
        if not history_uid or client_uid not in self.client_connections:
            return
        context = self.client_contexts.get(client_uid)
        if not context:
            return
        try:
            machine = self._proactive_machines.get(client_uid)
            config = self._proactive_config(context)
            if machine is None or machine.config != config:
                machine = ProactiveStateMachine(config)
                self._proactive_machines[client_uid] = machine
        except (AttributeError, ValueError) as error:
            logger.warning(
                "Proactive chat disabled because configuration is invalid: type={}",
                type(error).__name__,
            )
            return
        if not machine.config.enabled:
            return

        states = self._proactive_states.setdefault(client_uid, {})
        state = states.get(history_uid)
        if state is None:
            state = machine.new_state(history_uid)
            states[history_uid] = state
        elif user_activity:
            machine.record_user_activity(state)

        self._proactive_timer_tasks[client_uid] = asyncio.create_task(
            self._run_proactive_timer(client_uid, history_uid, state, machine),
            name=f"proactive-chat-{client_uid}-{history_uid}",
        )

    async def _record_user_activity(self, client_uid: str) -> None:
        """Reset idle/backoff state and give user input priority over a timer."""
        context = self.client_contexts.get(client_uid)
        history_uid = context.history_uid if context else None
        machine = self._proactive_machines.get(client_uid)
        state = (
            self._proactive_states.get(client_uid, {}).get(history_uid)
            if history_uid
            else None
        )
        if machine and state:
            generation_was_in_progress = state.proactive_generation_in_progress
            machine.record_user_activity(state)
            state.proactive_generation_in_progress = generation_was_in_progress

        task = self._proactive_timer_tasks.pop(client_uid, None)
        if task and not task.done() and task is not asyncio.current_task():
            # Before generation begins, cancellation is immediate.  Once the
            # provider/TTS turn has started, wait for that single turn instead
            # of overlapping two LLM streams on the same session agent.
            if state and state.proactive_generation_in_progress:
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
            else:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def _proactive_conditions_allow(
        self,
        client_uid: str,
        history_uid: str,
    ) -> bool:
        context = self.client_contexts.get(client_uid)
        websocket = self.client_connections.get(client_uid)
        if (
            not context
            or not websocket
            or context.history_uid != history_uid
            or client_uid in self._proactive_maintenance
        ):
            return False
        group = self.chat_group_manager.get_client_group(client_uid)
        return not group or len(group.members) <= 1

    async def _run_proactive_timer(
        self,
        client_uid: str,
        history_uid: str,
        state: ProactiveRuntimeState,
        machine: ProactiveStateMachine,
    ) -> None:
        """Sleep until randomized eligibility, then generate at most one turn."""
        current_task = asyncio.current_task()
        try:
            while self._proactive_conditions_allow(client_uid, history_uid):
                await asyncio.sleep(machine.seconds_until_eligible(state))
                if not self._proactive_conditions_allow(client_uid, history_uid):
                    return

                active = self.current_conversation_tasks.get(client_uid)
                if active and not active.done() and active is not current_task:
                    try:
                        await asyncio.shield(active)
                    except Exception:
                        pass
                    if not self._proactive_conditions_allow(client_uid, history_uid):
                        return
                    # Do not speak immediately after a long response/TTS turn.
                    machine.record_user_activity(state)
                    continue

                if not machine.is_eligible(state):
                    continue

                context = self.client_contexts[client_uid]
                websocket = self.client_connections[client_uid]
                revision = state.activity_revision
                state.proactive_generation_in_progress = True
                self.current_conversation_tasks[client_uid] = current_task
                # Yield once before any conversation/provider work.  A user
                # input arriving on the same event-loop turn increments the
                # revision and wins without starting proactive generation.
                await asyncio.sleep(0)
                if revision != state.activity_revision:
                    state.proactive_generation_in_progress = False
                    return
                logger.info(
                    "Proactive chat generation started: request_origin=proactive, "
                    "ignored_count={}",
                    state.consecutive_ignored_proactive,
                )
                followup_context = machine.proactive_followup_context(state)
                response = await process_single_conversation(
                    context=context,
                    websocket_send=websocket.send_text,
                    client_uid=client_uid,
                    user_input="",
                    images=None,
                    session_emoji=str(np.random.choice(EMOJI_LIST)),
                    metadata={
                        "request_origin": "proactive",
                        "proactive_followup": followup_context.as_dict(),
                    },
                )
                state.proactive_generation_in_progress = False
                if response and revision == state.activity_revision:
                    machine.record_proactive_sent(state, response_text=response)
                elif revision != state.activity_revision:
                    # User activity arrived after generation had meaningfully
                    # started.  End this scheduler so the user handler can
                    # install a fresh timer after starting the reply turn.
                    return
                else:
                    # Empty/cancelled work or user activity gets a fresh idle
                    # period and never increments the ignored counter.
                    machine.record_user_activity(state)
        except asyncio.CancelledError:
            state.proactive_generation_in_progress = False
            raise
        except Exception as error:
            state.proactive_generation_in_progress = False
            logger.warning(
                "Proactive generation failed safely: type={}",
                type(error).__name__,
            )
        finally:
            if self.current_conversation_tasks.get(client_uid) is current_task:
                self.current_conversation_tasks.pop(client_uid, None)
            if self._proactive_timer_tasks.get(client_uid) is current_task:
                self._proactive_timer_tasks.pop(client_uid, None)

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "add-client-to-group": self._handle_group_operation,
            "remove-client-from-group": self._handle_group_operation,
            "request-group-info": self._handle_group_info,
            "fetch-history-list": self._handle_history_list_request,
            "fetch-and-set-history": self._handle_fetch_history,
            "create-new-history": self._handle_create_history,
            "delete-history": self._handle_delete_history,
            "reset-relationship": self._handle_reset_relationship,
            "compact-conversation": self._handle_compact_conversation,
            "rename-history": self._handle_rename_history,
            "fetch-character-memory": self._handle_fetch_character_memory,
            "delete-character-memory": self._handle_delete_character_memory,
            "reset-character-memory": self._handle_reset_character_memory,
            "reset-character-state": self._handle_reset_character_state,
            "interrupt-signal": self._handle_interrupt,
            "mic-audio-data": self._handle_audio_data,
            "mic-audio-end": self._handle_conversation_trigger,
            "raw-audio-data": self._handle_raw_audio_data,
            "text-input": self._handle_conversation_trigger,
            "ai-speak-signal": self._handle_conversation_trigger,
            "fetch-configs": self._handle_fetch_configs,
            "switch-config": self._handle_config_switch,
            "fetch-backgrounds": self._handle_fetch_backgrounds,
            "audio-play-start": self._handle_audio_play_start,
            "request-init-config": self._handle_init_config_request,
            "heartbeat": self._handle_heartbeat,
        }

    async def handle_new_connection(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle new WebSocket connection setup

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client

        Raises:
            Exception: If initialization fails
        """
        try:
            # Serialize all sends on this connection (see create_locked_send_text
            # for the race this prevents). Shadowing the instance method keeps
            # every later ``websocket.send_text(...)`` call -- including the one
            # passed into the service context -- behind the same lock.
            websocket.send_text = create_locked_send_text(websocket)

            session_service_context = await self._init_service_context(
                websocket.send_text, client_uid
            )

            await self._store_client_data(
                websocket, client_uid, session_service_context
            )

            await self._send_initial_messages(
                websocket, client_uid, session_service_context
            )

            logger.info(f"Connection established for client {client_uid}")

        except Exception as e:
            logger.error(
                f"Failed to initialize connection for client {client_uid}: {e}"
            )
            await self._cleanup_failed_connection(client_uid)
            raise

    async def _store_client_data(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Store client data and initialize group status"""
        self.client_connections[client_uid] = websocket
        self.client_contexts[client_uid] = session_service_context
        self.received_data_buffers[client_uid] = np.array([])

        self.chat_group_manager.client_group_map[client_uid] = ""
        await self.send_group_update(websocket, client_uid)

    async def _send_initial_messages(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Send initial connection messages to the client"""
        await websocket.send_text(
            json.dumps({"type": "full-text", "text": "Connection established"})
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": session_service_context.live2d_model.model_info,
                    "conf_name": session_service_context.character_config.conf_name,
                    "conf_uid": session_service_context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

        # Send initial group status
        await self.send_group_update(websocket, client_uid)

        # Start microphone
        await websocket.send_text(json.dumps({"type": "control", "text": "start-mic"}))

    async def _init_service_context(
        self, send_text: Callable, client_uid: str
    ) -> ServiceContext:
        """Initialize service context for a new session by cloning the default context"""
        session_service_context = ServiceContext()
        await session_service_context.load_cache(
            config=self.default_context_cache.config.model_copy(deep=True),
            system_config=self.default_context_cache.system_config.model_copy(
                deep=True
            ),
            character_config=self.default_context_cache.character_config.model_copy(
                deep=True
            ),
            live2d_model=self.default_context_cache.live2d_model,
            asr_engine=self.default_context_cache.asr_engine,
            tts_engine=self.default_context_cache.tts_engine,
            vad_engine=self.default_context_cache.vad_engine,
            translate_engine=self.default_context_cache.translate_engine,
            mcp_server_registery=self.default_context_cache.mcp_server_registery,
            tool_adapter=self.default_context_cache.tool_adapter,
            send_text=send_text,
            client_uid=client_uid,
        )
        return session_service_context

    async def handle_websocket_communication(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle ongoing WebSocket communication

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client
        """
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    message_handler.handle_message(client_uid, data)
                    await self._route_message(websocket, client_uid, data)
                except WebSocketDisconnect:
                    raise
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                    continue

        except WebSocketDisconnect:
            logger.info(f"Client {client_uid} disconnected")
            raise
        except Exception as e:
            logger.error(f"Fatal error in WebSocket communication: {e}")
            raise

    async def _route_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Route incoming message to appropriate handler

        Args:
            websocket: The WebSocket connection
            client_uid: Client identifier
            data: Message data
        """
        msg_type = data.get("type")
        if not msg_type:
            logger.warning("Message received without type")
            return

        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(websocket, client_uid, data)
        else:
            if msg_type != "frontend-playback-complete":
                logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_group_operation(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Handle group-related operations"""
        operation = data.get("type")
        target_uid = data.get(
            "invitee_uid" if operation == "add-client-to-group" else "target_uid"
        )

        await self._cancel_proactive_timer(client_uid)
        await handle_group_operation(
            operation=operation,
            client_uid=client_uid,
            target_uid=target_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )
        context = self.client_contexts.get(client_uid)
        group = self.chat_group_manager.get_client_group(client_uid)
        if context and (not group or len(group.members) <= 1):
            await self._activate_proactive_for_history(client_uid, context.history_uid)

    async def handle_disconnect(self, client_uid: str) -> None:
        """Handle client disconnection"""
        await self._cancel_proactive_timer(client_uid)
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response="",
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )

        await handle_client_disconnect(
            client_uid=client_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

        # Clean up other client data
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self._proactive_states.pop(client_uid, None)
        self._proactive_machines.pop(client_uid, None)
        self._proactive_maintenance.discard(client_uid)
        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        # Call context close to clean up resources (e.g., MCPClient)
        context = self.client_contexts.get(client_uid)
        if context:
            await context.close()

        logger.info(f"Client {client_uid} disconnected")
        message_handler.cleanup_client(client_uid)

    async def _cleanup_failed_connection(self, client_uid: str) -> None:
        """Clean up failed connection data"""
        await self._cancel_proactive_timer(client_uid)
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.chat_group_manager.client_group_map.pop(client_uid, None)
        self._proactive_states.pop(client_uid, None)
        self._proactive_machines.pop(client_uid, None)
        self._proactive_maintenance.discard(client_uid)

        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        message_handler.cleanup_client(client_uid)

    async def broadcast_to_group(
        self, group_members: list[str], message: dict, exclude_uid: str = None
    ) -> None:
        """Broadcasts a message to group members"""
        await broadcast_to_group(
            group_members=group_members,
            message=message,
            client_connections=self.client_connections,
            exclude_uid=exclude_uid,
        )

    async def send_group_update(self, websocket: WebSocket, client_uid: str):
        """Sends group information to a client"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            current_members = self.chat_group_manager.get_group_members(client_uid)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": current_members,
                        "is_owner": group.owner_uid == client_uid,
                    }
                )
            )
        else:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": [],
                        "is_owner": False,
                    }
                )
            )

    async def _handle_interrupt(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle conversation interruption"""
        heard_response = data.get("text", "")
        context = self.client_contexts[client_uid]
        group = self.chat_group_manager.get_client_group(client_uid)

        if group and len(group.members) > 1:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response=heard_response,
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )
        else:
            await handle_individual_interrupt(
                client_uid=client_uid,
                current_conversation_tasks=self.current_conversation_tasks,
                context=context,
                heard_response=heard_response,
            )

    async def _handle_history_list_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for chat history list"""
        context = self.client_contexts[client_uid]
        histories = get_history_list(context.character_config.conf_uid)
        await websocket.send_text(
            json.dumps({"type": "history-list", "histories": histories})
        )

    async def _handle_fetch_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle fetching and setting specific chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        await self._cancel_proactive_timer(client_uid)
        context = self.client_contexts[client_uid]
        # Update history_uid in service context
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
        )

        messages = [
            msg
            for msg in get_history(
                context.character_config.conf_uid,
                history_uid,
            )
            if msg["role"] != "system"
        ]
        await websocket.send_text(
            json.dumps({"type": "history-data", "messages": messages})
        )
        # Selecting history is activity.  Reconnect therefore starts a fresh
        # idle period and never replays timers/messages from the old socket.
        await self._activate_proactive_for_history(client_uid, history_uid)

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation of new chat history"""
        await self._cancel_proactive_timer(client_uid)
        context = self.client_contexts[client_uid]
        history_uid = create_new_history(context.character_config.conf_uid)
        if history_uid:
            context.history_uid = history_uid
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                    }
                )
            )
            await self._activate_proactive_for_history(client_uid, history_uid)

    async def _handle_delete_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle deletion of chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        await self._cancel_proactive_timer(client_uid)
        context = self.client_contexts[client_uid]
        success = delete_history(
            context.character_config.conf_uid,
            history_uid,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-deleted",
                    "success": success,
                    "history_uid": history_uid,
                }
            )
        )
        if history_uid == context.history_uid:
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
            )
            context.history_uid = None
        self._proactive_states.get(client_uid, {}).pop(history_uid, None)
        if context.history_uid:
            await self._activate_proactive_for_history(client_uid, context.history_uid)

    async def _handle_reset_relationship(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Reset Mili's relationship for ALL conversations (character-level)."""
        await self._pause_proactive_for_maintenance(client_uid)
        context = self.client_contexts[client_uid]
        try:
            reset = getattr(context.agent_engine, "reset_relationship", None)
            success = bool(context.history_uid and callable(reset) and reset())
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "relationship-reset",
                        "success": success,
                        "history_uid": context.history_uid,
                    }
                )
            )
        finally:
            await self._resume_proactive_after_maintenance(client_uid, context)

    async def _handle_compact_conversation(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Manually compact the active conversation now (rolling-summary path)."""
        await self._pause_proactive_for_maintenance(client_uid)
        context = self.client_contexts[client_uid]
        try:
            compact = getattr(context.agent_engine, "compact_conversation", None)
            success, error = False, None
            if context.history_uid and callable(compact):
                success, error = await compact()
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "compact-result",
                        "success": success,
                        "history_uid": context.history_uid,
                        "error": error,
                    }
                )
            )
        finally:
            await self._resume_proactive_after_maintenance(client_uid, context)

    async def _handle_rename_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Persist a manual conversation title in history metadata."""
        history_uid = data.get("history_uid")
        title = str(data.get("title", "") or "").strip()
        context = self.client_contexts[client_uid]
        success = bool(
            history_uid
            and title
            and update_metadate(
                context.character_config.conf_uid,
                history_uid,
                {"title": title},
            )
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-renamed",
                    "success": success,
                    "history_uid": history_uid,
                    "title": title if success else None,
                }
            )
        )

    async def _handle_fetch_character_memory(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Return Mili's stored long-term facts (text + timestamp only)."""
        context = self.client_contexts[client_uid]
        memories = getattr(
            context.agent_engine, "list_character_memories", lambda: []
        )()
        await websocket.send_text(
            json.dumps(
                {
                    "type": "character-memory",
                    "memories": memories,
                }
            )
        )

    async def _handle_delete_character_memory(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Forget one stored long-term fact (by text)."""
        await self._pause_proactive_for_maintenance(client_uid)
        context = self.client_contexts[client_uid]
        try:
            text = str(data.get("text", "") or "")
            remove = getattr(context.agent_engine, "remove_character_memory", None)
            success = bool(text and callable(remove) and remove(text))
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "character-memory-deleted",
                        "success": success,
                        "text": text,
                    }
                )
            )
        finally:
            await self._resume_proactive_after_maintenance(client_uid, context)

    async def _handle_reset_character_memory(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Clear all of Mili's long-term memory (relationship untouched)."""
        await self._pause_proactive_for_maintenance(client_uid)
        context = self.client_contexts[client_uid]
        try:
            reset = getattr(context.agent_engine, "reset_character_memory", None)
            success = bool(callable(reset) and reset())
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "character-memory-reset",
                        "success": success,
                    }
                )
            )
        finally:
            await self._resume_proactive_after_maintenance(client_uid, context)

    async def _handle_reset_character_state(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Reset relationship to stranger and clear memory; transcripts stay."""
        await self._pause_proactive_for_maintenance(client_uid)
        context = self.client_contexts[client_uid]
        try:
            reset = getattr(context.agent_engine, "reset_character_state", None)
            success = bool(callable(reset) and reset())
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "character-state-reset",
                        "success": success,
                    }
                )
            )
        finally:
            await self._resume_proactive_after_maintenance(client_uid, context)

    async def _handle_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming audio data"""
        audio_data = data.get("audio", [])
        if audio_data:
            await self._record_user_activity(client_uid)
            self.received_data_buffers[client_uid] = np.append(
                self.received_data_buffers[client_uid],
                np.array(audio_data, dtype=np.float32),
            )

    async def _handle_raw_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming raw audio data for VAD processing"""
        context = self.client_contexts[client_uid]
        chunk = data.get("audio", [])
        if chunk:
            for audio_bytes in context.vad_engine.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "interrupt"})
                    )
                elif audio_bytes == b"<|RESUME|>":
                    pass
                elif len(audio_bytes) > 1024:
                    # Detected audio activity (voice)
                    await self._record_user_activity(client_uid)
                    self.received_data_buffers[client_uid] = np.append(
                        self.received_data_buffers[client_uid],
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "mic-audio-end"})
                    )

    async def _handle_conversation_trigger(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle triggers that start a conversation"""
        msg_type = data.get("type", "")
        if msg_type in {"text-input", "mic-audio-end"}:
            await self._record_user_activity(client_uid)
        await handle_conversation_trigger(
            msg_type=msg_type,
            data=data,
            client_uid=client_uid,
            context=self.client_contexts[client_uid],
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            broadcast_to_group=self.broadcast_to_group,
        )
        if msg_type in {"text-input", "mic-audio-end"}:
            context = self.client_contexts.get(client_uid)
            if context:
                await self._activate_proactive_for_history(
                    client_uid,
                    context.history_uid,
                    user_activity=False,
                )

    async def _handle_fetch_configs(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available configurations"""
        context = self.client_contexts[client_uid]
        config_files = scan_config_alts_directory(context.system_config.config_alts_dir)
        await websocket.send_text(
            json.dumps({"type": "config-files", "configs": config_files})
        )

    async def _handle_config_switch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle switching to a different configuration"""
        config_file_name = data.get("file")
        if config_file_name:
            await self._pause_proactive_for_maintenance(client_uid)
            context = self.client_contexts[client_uid]
            try:
                await context.handle_config_switch(websocket, config_file_name)
                self._proactive_machines.pop(client_uid, None)
                self._proactive_states.pop(client_uid, None)
            finally:
                await self._resume_proactive_after_maintenance(client_uid, context)

    async def _handle_fetch_backgrounds(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available background images"""
        bg_files = scan_bg_directory()
        await websocket.send_text(
            json.dumps({"type": "background-files", "files": bg_files})
        )

    async def _handle_audio_play_start(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Handle audio playback start notification
        """
        group_members = self.chat_group_manager.get_group_members(client_uid)
        if len(group_members) > 1:
            display_text = data.get("display_text")
            if display_text:
                silent_payload = prepare_audio_payload(
                    audio_path=None,
                    display_text=display_text,
                    actions=None,
                    forwarded=True,
                )
                await self.broadcast_to_group(
                    group_members, silent_payload, exclude_uid=client_uid
                )

    async def _handle_group_info(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle group info request"""
        await self.send_group_update(websocket, client_uid)

    async def _handle_init_config_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for initialization configuration"""
        context = self.client_contexts.get(client_uid)
        if not context:
            context = self.default_context_cache

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": context.live2d_model.model_info,
                    "conf_name": context.character_config.conf_name,
                    "conf_uid": context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

    async def _handle_heartbeat(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle heartbeat messages from clients"""
        try:
            await websocket.send_json({"type": "heartbeat-ack"})
        except Exception as e:
            logger.error(f"Error sending heartbeat acknowledgment: {e}")
