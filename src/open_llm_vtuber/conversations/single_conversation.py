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
    send_canonical_final,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    safe_send,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager, final_frame_metadata
from ..chat_history_manager import (
    load_message_snapshot,
    remove_message,
    store_message,
)
from ..message_delivery import begin_attempt
from ..service_context import ServiceContext
from ..world_state import load_and_reconcile_world_state
from ..request_latency import (
    RequestLatencyTracker,
    reset_latency_tracker,
    set_latency_tracker,
)

# Import necessary types from agent outputs
from ..agent.output_types import SentenceOutput, AudioOutput

# Strong references to in-flight episodic capture tasks (see usage below).
_EPISODIC_CAPTURE_TASKS: "set[asyncio.Task]" = set()

# Strong references to in-flight attachment capture tasks (see usage below).
_ATTACHMENT_CAPTURE_TASKS: "set[asyncio.Task]" = set()


def _transport_accepted_text(
    websocket_send: WebSocketSend, generated: str, *, strict: bool
) -> str:
    """Words this turn may honestly claim in the transcript.

    A proactive dispatch passes a delivery probe as ``websocket_send``; a user
    turn passes the plain socket callable, which has no ``delivered_text``, so
    the generated text is returned unchanged there.

    ``strict`` marks a turn that broke off (provider error or raise): it may
    only claim content frames the transport ACCEPTED. A completed turn keeps
    its generated text, unless the transport REFUSED frames -- that is positive
    proof part of it never left this process, and then only the accepted prefix
    is honest. History is what a reload renders and what Mili remembers, so a
    sentence the transport dropped must never be recorded as something she
    said. This is still transport handoff, not proof of rendering: there is no
    client ACK.
    """
    accepted = getattr(websocket_send, "delivered_text", None)
    if accepted is None:
        return "" if strict else generated
    if getattr(websocket_send, "refused", False):
        return accepted
    if strict:
        return accepted if getattr(websocket_send, "delivered", False) else ""
    return generated


class _DurableAIContent:
    """Keep the AI history row ahead of every content frame.

    The ordered TTS sender calls :meth:`before_content` immediately before the
    transport sees a frame. The row is upserted by the stable ``msg_id`` with
    the accumulated text (including this frame), so a crash at any point after
    the write leaves a canonical reply that a reload can render.

    The transport is still not transactional: a crash between the write and
    the send can leave one frame's text ahead of what the user heard. That is
    the unavoidable side of persist-before-send. On a normal refusal the
    after-callback rolls the row back to the accepted prefix, so a state a
    crash can leave behind never records words the transport explicitly
    dropped.

    :meth:`finalize` then writes the canonical text of the COMPLETED turn
    (what the old single end-of-turn write used to store), which keeps two
    long-standing contracts: a dead socket must not lose an answer that was
    generated, and a turn whose text never reached a content frame (a
    transport-less caller, direct audio, a mocked producer) still ends up in
    history. A retry reuses the same row; a final/retry never appends a second
    one.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        conf_uid: str,
        history_uid: str,
        msg_id: str,
        name: str,
        avatar: Optional[str],
    ) -> None:
        self.enabled = bool(enabled and conf_uid and history_uid and msg_id)
        self.conf_uid = conf_uid
        self.history_uid = history_uid
        self.msg_id = msg_id
        self.name = name
        self.avatar = avatar
        self.accepted = ""
        self.persisted = ""
        self.failed = False
        self._snapshot_ok = True
        self._previous = None
        if not self.enabled:
            return
        readable, previous = load_message_snapshot(
            conf_uid, history_uid, "ai", msg_id
        )
        if not readable:
            # Never guess whether a same-id row already exists. Withholding
            # content is safer than deleting or overwriting data we could not
            # inspect.
            self._snapshot_ok = False
            self.failed = True
            return
        self._previous = previous
        if previous is not None:
            self.persisted = str(previous.get("content") or "")

    def _store(
        self,
        content: str,
        *,
        name: Optional[str] = None,
        avatar: Optional[str] = None,
    ) -> bool:
        try:
            return store_message(
                conf_uid=self.conf_uid,
                history_uid=self.history_uid,
                role="ai",
                content=content,
                name=self.name if name is None else name,
                avatar=self.avatar if avatar is None else avatar,
                msg_id=self.msg_id,
            )
        except Exception:
            logger.exception(
                "History guard raised while storing AI content (msg_id={})",
                self.msg_id,
            )
            return False

    def _restore(self, content: str) -> bool:
        """Put the row back to ``content`` (or the pre-turn snapshot)."""
        if not self.enabled:
            return True
        if content:
            if self._store(content):
                self.persisted = content
                return True
            self.failed = True
            return False
        if not self._snapshot_ok:
            return False
        previous = self._previous
        if previous is not None:
            if self._store(
                str(previous.get("content") or ""),
                name=previous.get("name"),
                avatar=previous.get("avatar"),
            ):
                self.persisted = str(previous.get("content") or "")
                return True
            self.failed = True
            return False
        if remove_message(self.conf_uid, self.history_uid, "ai", self.msg_id):
            self.persisted = ""
            return True
        self.failed = True
        return False

    async def before_content(self, text_delta: str) -> bool:
        if not self.enabled:
            return True
        if self.failed:
            return False
        delta = str(text_delta or "")
        if not delta:
            return True
        candidate = self.accepted + delta
        if candidate == self.persisted:
            return True
        if self._store(candidate):
            self.persisted = candidate
            return True
        self.failed = True
        logger.error(
            "HISTORY_GUARD_BLOCKED msg_id={} (the content frame was not sent)",
            self.msg_id,
        )
        return False

    async def after_content(self, text_delta: str, accepted: bool) -> None:
        if not self.enabled:
            return
        delta = str(text_delta or "")
        if accepted:
            self.accepted += delta
            return
        # The transport refused this frame. The row currently includes it
        # because persistence necessarily happened first; trim back to what
        # really left the process.
        self._restore(self.accepted)

    async def finalize(self, canonical_text: str = "") -> tuple[bool, str]:
        """Persist the turn's canonical text; return ``(stored, text)``.

        ``canonical_text`` is the answer this turn produced (already trimmed
        to what a proactive dispatch may honestly claim). It is written even
        when the transport refused the tail: a delivery failure must not lose
        generated text, which is the behaviour the transcript had before the
        pre-send guard existed.

        An empty canonical text (a turn that produced nothing, or a provider
        failure that discarded its partial reply) removes the provisional row
        instead of claiming a reply. A failed write never reports success.
        """
        if not self.enabled:
            return True, ""
        text = str(canonical_text or "")
        if self.failed:
            stored = self._restore(self.accepted)
            return stored, self.accepted if stored else ""
        if not text:
            # Nothing to claim. Remove the provisional row (or restore a
            # previous attempt's row) instead of inventing a reply.
            return self._restore(""), ""
        if self.persisted == text:
            return True, text
        stored = self._store(text)
        if not stored:
            self.failed = True
            return False, ""
        self.persisted = text
        return True, text

    async def abort(self) -> None:
        """Drop unaccepted provisional text on cancel/error, keep accepted."""
        if not self.enabled:
            return
        self._restore(self.accepted)


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
    # Set when a proactive turn ended without being a complete reply (provider
    # error or a raise). Whether any of its words already reached the client
    # cannot be judged here: TTS payloads are still queued, so the delivery
    # probe is read AFTER the turn has flushed its queue (see the settlement
    # just before the history write).
    proactive_partial_undecided = False
    proactive = bool((metadata or {}).get("request_origin") == "proactive")
    # Stable identity of the assistant message this turn produces. A caller
    # that REDELIVERS a message (e.g. a reminder retry) supplies the id it
    # used before via metadata["msg_id"]; otherwise a fresh one is minted.
    # request_id stays per-attempt, msg_id identifies the logical message:
    # the persisted row and every frame of the turn carry this same value.
    msg_id = str((metadata or {}).get("msg_id") or "").strip() or uuid.uuid4().hex
    request_id = (metadata or {}).get("latency_request_id") or uuid.uuid4().hex
    llm = getattr(context.agent_engine, "_llm", None)
    base_url = str(getattr(llm, "base_url", ""))
    provider = "ollama_cloud" if "ollama.com" in base_url else type(llm).__name__
    conf_uid = getattr(context.character_config, "conf_uid", "") or ""
    history_uid = getattr(context, "history_uid", "") or ""
    # The row is the source of truth a reload renders. Create its guard before
    # any content can be queued; the TTS sender invokes it at the actual send
    # boundary, including multi-segment and interrupted turns.
    durable_content = _DurableAIContent(
        enabled=bool(history_uid),
        conf_uid=conf_uid,
        history_uid=history_uid,
        msg_id=msg_id,
        name=context.character_config.character_name,
        avatar=context.character_config.avatar,
    )
    # Open the delivery attempt BEFORE the first frame goes out, so a receipt
    # naming this msg_id/turn_id/attempt can be matched. The attempt counter
    # comes from the store (never from the caller): a redelivery of the SAME
    # msg_id gets attempt N+1 and can never rewind or duplicate it.
    delivery_attempt = None
    if conf_uid and msg_id:
        delivery_attempt = begin_attempt(
            conf_uid,
            msg_id,
            turn_id=request_id,
            owner=client_uid,
            # Stable client instance (from `client-hello`). Persisted with the
            # record so a reconnect -- which arrives under a NEW client_uid --
            # can still be recognised as the client that owns this message.
            client_ref=getattr(context, "delivery_client_ref", "") or "",
            history_uid=getattr(context, "history_uid", "") or "",
        )
        if delivery_attempt is None:
            # The store refused the write. The turn still runs and still sends
            # its frames (identity is harmless and keeps old clients working),
            # but no receipt can settle it -- so nothing here may claim the
            # ACK protocol is active for this message.
            logger.warning(
                "MESSAGE_DELIVERY_STATE_UNAVAILABLE msg_id={} "
                "(client receipts will not be recorded)",
                msg_id,
            )
    attempt = int((delivery_attempt or {}).get("attempt") or 1)
    latency = RequestLatencyTracker(
        websocket_send=websocket_send,
        provider=provider,
        model=str(getattr(llm, "model", "unknown")),
        request_origin="proactive" if proactive else "user",
        request_id=request_id,
        msg_id=msg_id,
        msg_attempt=attempt,
        client_user_send_ms=(metadata or {}).get("client_user_send_ms"),
        client_websocket_send_ms=(metadata or {}).get("client_websocket_send_ms"),
    )
    latency_token = set_latency_tracker(latency)

    try:
        await latency.emit("backend-received")
        # Stage 7: lazy World/Life reconcile on the conversation trigger
        # (covers normal + proactive turns, reconnect, restart offline gap).
        # Idempotent and fail-soft: a world failure never breaks the turn.
        try:
            conf_uid = getattr(context.character_config, "conf_uid", None)
            if conf_uid:
                load_and_reconcile_world_state(
                    conf_uid, tz=getattr(context, "user_timezone", None)
                )
        except Exception as error:
            logger.warning(
                "World state reconcile skipped: type={}", type(error).__name__
            )
        # Send initial signals
        await send_conversation_start_signals(
            websocket_send,
            history_uid=getattr(context, "history_uid", "") or "",
            request_id=latency.request_id,
            msg_id=msg_id,
            attempt=attempt,
        )
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
            # Phase 3 deterministic search routing: runs once per user
            # turn, before the agent. On SEARCH_REQUIRED the existing
            # ToolExecutor (with Phase 1 reliability) runs the search and
            # its labeled block joins the LLM input text. History keeps
            # the clean user text; nothing router-related is persisted.
            llm_text = input_text
            # Episodic retrieval must see the clean user turn: the search
            # block below is LLM-input only and would otherwise pollute the
            # query. Same rule as history persistence.
            turn_metadata = dict(metadata or {})
            if input_text.strip():
                turn_metadata["episodic_query"] = input_text.strip()
            if input_text.strip():
                from ..mcpp.search_router import (
                    SEARCH_EXECUTED_NO_RESULTS,
                    SEARCH_EXECUTED_SUCCESS,
                    SEARCH_EXECUTION_ERROR,
                    SEARCH_REQUIRED_NOT_EXECUTED,
                    execute_router_search,
                    is_search_required,
                )

                if is_search_required(input_text):
                    router_executor = getattr(
                        context.agent_engine, "_tool_executor", None
                    )
                    if router_executor is None:
                        logger.warning(
                            "Search router required search but no tool "
                            "executor is available; continuing without "
                            "search results"
                        )
                        latency.set_search_state(SEARCH_REQUIRED_NOT_EXECUTED)
                    else:
                        router_result = await execute_router_search(
                            router_executor, input_text
                        )
                        latency.set_search_state(
                            router_result.state, router_result.result_count
                        )
                        # Register the executed query so a duplicate
                        # same-turn model search is served from cache
                        # (application-level guard; cleared in finally).
                        try:
                            router_executor.note_router_search(
                                router_result.query,
                                router_result.block_text,
                            )
                        except Exception:
                            pass
                        if router_result.state in (
                            SEARCH_EXECUTED_SUCCESS,
                            SEARCH_EXECUTED_NO_RESULTS,
                            SEARCH_EXECUTION_ERROR,
                        ):
                            llm_text = f"{input_text}\n\n{router_result.block_text}"
            batch_input = create_batch_input(
                input_text=llm_text,
                images=images,
                from_name=context.character_config.human_name,
                metadata=turn_metadata,
            )

            if context.history_uid and not skip_history:
                save_started = time.perf_counter()
                human_saved = store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="human",
                    content=input_text,
                    name=context.character_config.human_name,
                )
                latency.add_history_save((time.perf_counter() - save_started) * 1000)
                if not human_saved:
                    # The turn continues (the answer can still be recorded and
                    # displayed), but nothing may claim the user's line was
                    # stored. Reported, never silently dropped.
                    logger.error(
                        "HUMAN_MESSAGE_NOT_PERSISTED history_uid={} "
                        "(characters={})",
                        context.history_uid,
                        len(input_text),
                    )
                    await safe_send(
                        websocket_send,
                        json.dumps(
                            {
                                "type": "error",
                                "code": "history-persist-failed",
                                "message": (
                                    "Your message could not be saved to the "
                                    "chat history."
                                ),
                            }
                        ),
                    )

            if skip_history:
                logger.debug("Skipping storing user input to history")

            logger.info(
                "TURN_STARTED history_uid={} (characters={})",
                context.history_uid,
                len(input_text),
            )
            if images:
                logger.info(f"With {len(images)} images")

        latency.mark("agent_start")
        # Backend semantic-emotion labels observed this turn. Collected for
        # the reactive life-state step after the stream (group chats skip
        # reactive observation in v1).
        turn_emotions: List[str] = []
        provider_error_occurred = False
        try:
            if proactive:
                proactive_chat = getattr(context.agent_engine, "chat_proactively", None)
                if not callable(proactive_chat):
                    raise RuntimeError(
                        "Active conversation agent does not support proactive chat"
                    )
                proactive_kwargs = {
                    "followup_context": (metadata or {}).get("proactive_followup"),
                    "intent_context": (metadata or {}).get("proactive_intent"),
                }
                due_reminders = (metadata or {}).get("proactive_due_reminders")
                if due_reminders:
                    proactive_kwargs["due_reminders"] = due_reminders
                agent_output_stream = proactive_chat(**proactive_kwargs)
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

                elif (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "provider_error"
                ):
                    provider_error_occurred = True
                    category = output_item.get("category", "provider_error")
                    message = output_item.get("message") or "LLM provider error"
                    logger.warning(
                        "Provider error surfaced to client: category={}", category
                    )
                    await websocket_send(
                        json.dumps({"type": "error", "message": message})
                    )

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    # Handle SentenceOutput or AudioOutput
                    tts_started = time.perf_counter()
                    actions = getattr(output_item, "actions", None)
                    if actions is not None:
                        for label in getattr(actions, "emotions", None) or []:
                            if isinstance(label, str) and label.strip():
                                turn_emotions.append(label.strip().lower())
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.tts_engine,
                        websocket_send=websocket_send,  # Pass websocket_send for audio/tts messages
                        tts_manager=tts_manager,
                        translate_engine=context.translate_engine,
                        synthesize_audio=getattr(
                            context, "voice_output_enabled", True
                        ),
                        before_content_send=durable_content.before_content,
                        after_content_send=durable_content.after_content,
                    )
                    latency.add_tts_enqueue(
                        (time.perf_counter() - tts_started) * 1000
                    )
                    # Ensure response_part is treated as a string before concatenation
                    response_part_str = (
                        str(response_part) if response_part is not None else ""
                    )
                    full_response += response_part_str  # Accumulate text response
                    if durable_content.failed:
                        # The guard refused a frame. Stop producing more
                        # content: later frames would only deepen the gap,
                        # and no canonical final may close this turn.
                        break
                else:
                    logger.warning(
                        f"Received unexpected item type from agent chat stream: {type(output_item)}"
                    )
                    logger.debug(
                        "Unexpected item type from agent chat stream (content omitted)"
                    )

            if provider_error_occurred:
                # A provider failure is not a complete assistant turn, so its
                # partial text must not be persisted as a whole reply. A user
                # turn has no delivery probe, so nothing changed there. A
                # proactive turn is decided later: the sentence divider FLUSHES
                # its unterminated tail when the provider stream ends, and the
                # queued payloads have not been flushed yet at this point, so
                # only the post-finalize check can tell whether the user
                # actually saw those words.
                if not proactive:
                    full_response = ""
                else:
                    proactive_partial_undecided = True

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
            if proactive:
                # Same rule as the typed provider error above: a raise is not a
                # finished reply, but frames that were queued before it may
                # still reach the transport while the turn flushes. Decide from
                # the probe after finalization instead of from the exception.
                proactive_partial_undecided = True
            # full_response will contain partial response before error
        latency.mark("agent_end")
        # --- End processing agent response ---

        # Reactive life state BEFORE finalize: the widget auto-fetches on
        # conversation-chain-end, so the reactive persist must already be
        # done when chain-end is emitted. Same guard as the character
        # observer below (visible normal turns with a real response only).
        # Pure + persist, no LLM calls.
        if context.history_uid and full_response and not skip_history and not proactive:
            reactive = getattr(context.agent_engine, "observe_reactive_state", None)
            if callable(reactive):
                try:
                    reactive(turn_emotions)
                except Exception as reactive_error:
                    logger.warning(
                        "Reactive observation skipped: type={}",
                        type(reactive_error).__name__,
                    )

        # Wait for any pending TTS tasks
        if tts_manager.task_list:
            latency.mark("tts_wait_start")
            await asyncio.gather(*tts_manager.task_list)
            latency.mark("tts_wait_end")
            latency.add_tts_wait(
                latency.phase_duration("tts_wait_start", "tts_wait_end") or 0.0
            )
            # backend-synth-complete is emitted ONCE by finalize_conversation_turn
            # below (same `task_list` guard); sending it here too made the
            # frontend release twice per turn (duplicate playback-complete).

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )

        # The ordered TTS queue has been flushed by now, so the delivery probe
        # is final. Align the transcript with it: a broken turn may only claim
        # accepted frames, and a completed proactive turn must not record
        # sentences the transport refused (a dead socket used to write text
        # nobody received, and the reminder retry then duplicated it).
        if proactive_partial_undecided:
            full_response = _transport_accepted_text(
                websocket_send, full_response, strict=True
            )
        elif proactive:
            full_response = _transport_accepted_text(
                websocket_send, full_response, strict=False
            )

        # Write the completed turn's canonical text over the provisional row
        # the pre-send guard has been updating frame by frame. ``history_text``
        # is empty when the turn produced nothing to claim (or refused every
        # frame, for a proactive dispatch): the provisional row is then
        # removed/restored, never presented as a reply. A persistence failure
        # keeps the row at the accepted prefix but suppresses ai-final.
        save_started = time.perf_counter()
        history_persisted, history_text = await durable_content.finalize(
            full_response
        )
        latency.add_history_save((time.perf_counter() - save_started) * 1000)

        if context.history_uid and (history_text or durable_content.failed):
            if not history_persisted or durable_content.failed:
                # Nothing may claim this turn was fully stored: not the
                # success log, and above all not the canonical `ai-final`
                # (the frontend treats it as the authoritative persisted row).
                # The turn still closes normally: the lifecycle tail must not
                # hang on a storage failure.
                logger.error(
                    "HISTORY_PERSIST_FAILED history_uid={} msg_id={} "
                    "(characters={}): ai-final suppressed",
                    context.history_uid,
                    msg_id,
                    len(history_text or full_response),
                )
                await safe_send(
                    websocket_send,
                    json.dumps(
                        {
                            "type": "error",
                            "code": "history-persist-failed",
                            "message": (
                                "Mili's reply could not be saved to the chat "
                                "history."
                            ),
                        }
                    ),
                )
            elif history_text:
                logger.info(
                    "TURN_PERSISTED history_uid={} msg_id={} (characters={})",
                    context.history_uid,
                    msg_id,
                    len(history_text),
                )
                # Canonical sync: the row just persisted is authoritative.
                # A provider-error turn may keep its accepted prefix in
                # history, but it is not a completed reply and must not carry
                # a false `final`: no canonical closing frame is emitted.
                if not (provider_error_occurred and not proactive):
                    try:
                        await send_canonical_final(
                            websocket_send,
                            history_uid=context.history_uid,
                            request_id=latency.request_id,
                            text=history_text,
                            msg_id=msg_id,
                            # Closing frame of the delivery protocol: `final`,
                            # the attempt, and the content-frame count, so the
                            # client's ACK can settle exactly this attempt.
                            final_metadata=final_frame_metadata(latency),
                        )
                    except Exception:
                        pass

        if (
            history_text
            and context.history_uid
            and not skip_history
            and not proactive
            and not provider_error_occurred
        ):
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
                observer(input_text, history_text)
                latency.add_time(
                    "character_event_ms",
                    (time.perf_counter() - event_started) * 1000,
                )

            # Auxiliary auto-title: one LLM call per conversation lifetime,
            # after the visible response is stored. Never breaks the chat;
            # a failure just keeps "Percakapan Baru".
            title_fn = getattr(
                context.agent_engine, "ensure_conversation_title", None
            )
            if callable(title_fn):
                try:
                    auto_title = await title_fn()
                except Exception:
                    auto_title = ""
                if auto_title:
                    try:
                        await websocket_send(
                            json.dumps(
                                {
                                    "type": "history-auto-titled",
                                    "success": True,
                                    "history_uid": context.history_uid,
                                    "title": auto_title,
                                }
                            )
                        )
                    except Exception:
                        pass

        latency.mark("websocket_final_output")
        # Episodic capture runs after the user-visible response, never
        # blocking it: gate + one LLM call + validated store, fail-soft.
        if input_text and not proactive and context.history_uid:
            capture = getattr(context.agent_engine, "capture_episodic_event", None)
            if callable(capture):
                try:
                    task = asyncio.create_task(capture(input_text, context.history_uid))
                    # Strong reference until done: an unreferenced task may be
                    # garbage-collected mid-flight, losing the event and its
                    # latency measurement. Never awaited here.
                    _EPISODIC_CAPTURE_TASKS.add(task)
                    task.add_done_callback(_EPISODIC_CAPTURE_TASKS.discard)
                except Exception as error:
                    logger.debug(
                        "Episodic capture scheduling skipped: type={}",
                        type(error).__name__,
                    )
        # Attachment capture runs after the user-visible response, never
        # blocking it: one batched describe call + validated store,
        # fail-soft. Only turns that carried sanitized images qualify.
        if images and not proactive and context.history_uid:
            capture = getattr(context.agent_engine, "capture_attachment_memory", None)
            if callable(capture):
                try:
                    task = asyncio.create_task(
                        capture(
                            images,
                            input_text,
                            context.history_uid,
                            latency.request_id,
                        )
                    )
                    # Strong reference until done: an unreferenced task may be
                    # garbage-collected mid-flight, losing the record and its
                    # latency measurement. Never awaited here.
                    _ATTACHMENT_CAPTURE_TASKS.add(task)
                    task.add_done_callback(_ATTACHMENT_CAPTURE_TASKS.discard)
                except Exception as error:
                    logger.debug(
                        "Attachment capture scheduling skipped: type={}",
                        type(error).__name__,
                    )
        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        latency.interrupted = True
        try:
            # Drop a provisional frame that was written but not confirmed by
            # the transport; keep the rows that were actually accepted. No
            # ai-final is emitted on cancellation, so a partial turn cannot
            # masquerade as complete.
            await durable_content.abort()
        except Exception:
            logger.exception("History guard abort failed during cancellation")
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        latency.internal_error = type(e).__name__
        try:
            await durable_content.abort()
        except Exception:
            logger.exception("History guard abort failed during error path")
        logger.error(f"Error in conversation chain: {e}")
        await safe_send(
            websocket_send,
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"}),
        )
        raise
    finally:
        # Turn-scoped router-search guard state is always dropped here:
        # normal completion, cancellation, exception, and early return.
        try:
            router_executor_for_cleanup = getattr(
                context.agent_engine, "_tool_executor", None
            )
            if router_executor_for_cleanup is not None:
                router_executor_for_cleanup.clear_router_search()
        except Exception:
            pass
        try:
            await latency.complete()
        except Exception as latency_error:
            logger.warning(
                "Latency finalization failed: type={}",
                type(latency_error).__name__,
            )
        reset_latency_tracker(latency_token)
        cleanup_conversation(tts_manager, session_emoji)
