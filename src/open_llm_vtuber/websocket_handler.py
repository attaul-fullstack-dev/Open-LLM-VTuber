from typing import Dict, List, Optional, Callable, TypedDict, Any
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
    get_metadata,
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
from .world_state import load_and_reconcile_world_state, utcnow
from .character_state import set_character_timezone
from .conversations.single_conversation import process_single_conversation
from .conversations.conversation_utils import EMOJI_LIST
from .autonomous_decision import OUTCOME_GOAL_BEHAVIOR
from .proactive_gate import (
    ProactiveBudgetState,
    ProactiveGateConfig,
    classify_trigger,
    evaluate_gate,
    load_proactive_state,
    record_proactive_answered,
    record_proactive_dispatch,
    record_suppressed,
    save_proactive_state,
)
from .proactive_chat import (
    ProactiveChatConfig,
    ProactiveIntent,
    ProactiveIntentContext,
    ProactiveIntentStrategy,
    ProactiveIntentSignals,
    ProactiveRuntimeState,
    ProactiveStateMachine,
    ProactiveTurnStrategy,
    band_for,
    build_semantic_proactive_context,
    compute_intent_signals,
    resolve_proactive_intent_decision,
)


def _log_typed_proactive_decision(machine, state, decision) -> None:
    """Log the typed Autonomous Decision Layer record for one proactive turn.

    Observation only. The existing ``ProactiveStateMachine`` remains the sole
    authority on eligibility, cooldown and intent; this never gates, never
    reschedules and never touches the socket. Fail-soft by construction.
    """
    try:
        from .autonomous_decision import classify_proactive_decision
        from .world_state import utcnow

        try:
            eligible = bool(machine.is_eligible(state))
        except Exception:
            # A decision was produced, so treat it as eligible for labelling.
            eligible = True
        typed = classify_proactive_decision(
            eligible=eligible,
            reason=str(getattr(decision, "reason", "") or ""),
            moment=utcnow(),
            strategy=getattr(decision, "strategy", None),
            intent=getattr(decision, "intent", None),
        )
        logger.debug(
            "Autonomous decision typed: outcome={} reason={} acts={} "
            "cooldown_until={} strategy={} intent={}",
            typed.outcome,
            typed.reason,
            typed.acts,
            typed.cooldown_until,
            typed.metadata.get("strategy"),
            typed.metadata.get("intent"),
        )
    except Exception as error:
        logger.debug(
            "Typed proactive decision unavailable: type={}", type(error).__name__
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
    CONTROL = [
        "interrupt-signal",
        "audio-play-start",
        "voice-output-toggle",
    ]
    DATA = ["mic-audio-data"]


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    # text-input attachments: list of {source, data, mime_type} dicts with
    # optional client metadata (name, size). Validated per file by
    # sanitize_images in conversation_utils; never a list of bare strings.
    images: Optional[List[Dict[str, Any]]]
    history_uid: Optional[str]
    file: Optional[str]
    display_text: Optional[dict]
    timezone: Optional[str]
    enabled: Optional[bool]


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
        # Turns detached by a mid-turn disconnect, keyed by history_uid.
        # Ownership outlives the dead socket so the turn can finish, stay
        # serialised against the next trigger, and resync to the new socket.
        self._detached_turns: Dict[str, asyncio.Task] = {}
        # Which live client is currently viewing which history. Lets a
        # finished detached turn deliver its result to the reconnected UI.
        self._history_subscribers: Dict[str, str] = {}
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}
        self._proactive_timer_tasks: Dict[str, asyncio.Task] = {}
        self._proactive_states: Dict[str, Dict[str, ProactiveRuntimeState]] = {}
        # Proactive V2: persisted accounting (survives reconnect + restart) and
        # the deterministic gate config. The gate is evaluated BEFORE any
        # provider call, so a suppressed turn never reaches the LLM.
        self._proactive_budget_state: Dict[str, ProactiveBudgetState] = {}
        self._proactive_gate_config = ProactiveGateConfig()
        self._proactive_machines: Dict[str, ProactiveStateMachine] = {}
        self._proactive_maintenance: set[str] = set()

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    def _detached_registry(self) -> Dict[str, asyncio.Task]:
        """Detached-turn registry, created on demand.

        Lazily initialised so a handler built without __init__ (tests,
        partial construction) still behaves correctly instead of raising.
        """
        registry = self.__dict__.get("_detached_turns")
        if registry is None:
            registry = {}
            self.__dict__["_detached_turns"] = registry
        return registry

    def _subscriber_registry(self) -> Dict[str, str]:
        """client_uid -> history_uid view map, created on demand."""
        registry = self.__dict__.get("_history_subscribers")
        if registry is None:
            registry = {}
            self.__dict__["_history_subscribers"] = registry
        return registry

    @staticmethod
    def _update_user_timezone(context: ServiceContext, data: dict) -> None:
        """Remember the session IANA timezone reported by the frontend.

        Used for user-local World State time rules. Missing/invalid values
        keep the previous session value (or UTC fallback when never set).
        """
        try:
            tz = data.get("timezone") if isinstance(data, dict) else None
        except Exception:
            tz = None
        if isinstance(tz, str) and tz.strip():
            context.user_timezone = tz.strip()[:64]
            agent = getattr(context, "agent_engine", None)
            if agent is not None and hasattr(agent, "_user_timezone"):
                agent._user_timezone = context.user_timezone
            # Persist last-known zone so restart/proactive turns keep
            # user-local interpretation before the next frontend message.
            try:
                conf_uid = getattr(
                    getattr(context, "character_config", None), "conf_uid", None
                )
                if conf_uid:
                    set_character_timezone(conf_uid, context.user_timezone)
            except Exception:
                pass

    def _proactive_config(self, context: ServiceContext) -> ProactiveChatConfig:
        settings = (
            context.character_config.agent_config.agent_settings.basic_memory_agent
        )
        # Proactive V2: build the deterministic gate config from the same
        # settings object, so config and runtime can never drift apart.
        self._proactive_gate_config = ProactiveGateConfig(
            minimum_proactive_gap_seconds=settings.minimum_proactive_gap_seconds,
            proactive_daily_hard_limit=settings.proactive_daily_hard_limit,
            maximum_unanswered_consecutive=settings.maximum_unanswered_consecutive,
            ignored_threshold_before_backoff=settings.ignored_before_backoff,
            backoff_multiplier=settings.backoff_multiplier,
            max_backoff_seconds=settings.max_backoff_seconds,
            quiet_hours_start_hour=settings.quiet_hours_start_hour,
            quiet_hours_end_hour=settings.quiet_hours_end_hour,
            meaningful_trigger_budget_per_hour=settings.meaningful_trigger_budget_per_hour,
            idle_trigger_budget_per_hour=settings.idle_trigger_budget_per_hour,
            idle_trigger_budget_per_day=settings.idle_trigger_budget_per_day,
            behavior_on_budget_exhausted=settings.behavior_on_budget_exhausted,
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
            intent_strategy=settings.proactive_intent_strategy,
            intent_weights=settings.proactive_intent_weights,
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

    # ------------------------------------------------------------------
    # Proactive V2 — deterministic gate, persisted budget, trigger priority
    # ------------------------------------------------------------------

    def _budget_state(
        self, client_uid: str, conf_uid: Optional[str]
    ) -> ProactiveBudgetState:
        """Return (loading once) the persisted proactive accounting.

        Survives reconnect and backend restart; a reconnect must never hand out
        a fresh budget.
        """
        store = getattr(self, "_proactive_budget_state", None)
        if store is None:
            store = {}
            self._proactive_budget_state = store
        state = store.get(client_uid)
        if state is None:
            state = (
                load_proactive_state(conf_uid) if conf_uid else ProactiveBudgetState()
            )
            self._proactive_budget_state[client_uid] = state
        return state

    def _persist_budget(self, client_uid: str, conf_uid: Optional[str]) -> None:
        state = getattr(self, "_proactive_budget_state", {}).get(client_uid)
        if state is None or not conf_uid:
            return
        save_proactive_state(conf_uid, state)

    @staticmethod
    def _meaningful_life_event(
        context: ServiceContext, since_iso: Optional[str]
    ) -> bool:
        """True when the life clock completed an activity since ``since_iso``.

        Read-only from the existing world state; no model call. This is the
        bridge that lets Mili comment on her OWN life without any LLM spent on
        deciding that something happened.
        """
        try:
            from .world_state import load_world_state

            agent = getattr(context, "agent_engine", None)
            conf_uid = getattr(agent, "_character_conf_uid", None)
            if not conf_uid:
                return False
            snapshot = load_world_state(conf_uid)
            history = list(getattr(snapshot, "recent_activity_history", None) or [])
            if not history:
                return False
            latest = history[-1]
            stamp = str(latest.get("at", "") or "")
            if not stamp:
                return False
            if since_iso and stamp <= str(since_iso):
                return False
            # A completed, non-trivial activity (not simply waking up).
            return latest.get("from") not in (None, "", "idle", "sleeping")
        except Exception as error:
            logger.debug(
                "Life-event trigger probe unavailable: type={}", type(error).__name__
            )
            return False

    def _goal_evidence_trigger(self, context: ServiceContext, now=None) -> bool:
        """True when an *active* stored goal has fresh deterministic evidence.

        This is the Autonomous Decision Layer's one contribution to proactive
        speech. It reads state that is already on disk (goals + episodic
        events), runs the pure ``classify_goal_evidence`` classifier, and pins
        the evidence it acted on so the same event can never fire the same goal
        twice. It performs **no** LLM call, creates no timer and no queue: the
        existing Proactive V2 gate below remains the only authority on whether
        anything is actually dispatched.

        ``now`` is an injectable clock (UTC aware). ``None`` means "use the
        real clock" — the production path. Tests pass a fake instant so the
        72h evidence horizon stays deterministic.

        Fail-soft: any problem means "no goal evidence", i.e. exactly the
        pre-existing behaviour.
        """
        try:
            agent = getattr(context, "agent_engine", None)
            classifier = getattr(agent, "classify_goal_evidence", None)
            if not callable(classifier):
                return False
            try:
                decision = classifier(now) if now is not None else classifier()
            except TypeError:
                # Test double / legacy facade without a clock parameter.
                decision = classifier()
            if decision is None or not getattr(decision, "acts", False):
                return False
            if str(getattr(decision, "outcome", "")) != OUTCOME_GOAL_BEHAVIOR:
                return False
            goal_id = (getattr(decision, "metadata", {}) or {}).get("goal_id")
            evidence_id = (getattr(decision, "metadata", {}) or {}).get("evidence_id")
            anchor = getattr(agent, "record_goal_evidence", None)
            if callable(anchor) and goal_id and evidence_id:
                anchor(goal_id, evidence_id)
            logger.info(
                "Autonomous decision: outcome={} reason={} goal_id={} "
                "evidence_id={} priority=high",
                getattr(decision, "outcome", ""),
                getattr(decision, "reason", ""),
                goal_id,
                evidence_id,
            )
            return True
        except Exception as error:
            logger.debug(
                "Goal evidence trigger unavailable: type={}", type(error).__name__
            )
            return False

    def _future_intention_trigger(self, context: ServiceContext, now=None) -> bool:
        """True when a pending reminder request is due right now.

        Reads the existing ``future_intentions`` rows (same character-state
        file as memories/goals) through the agent facade's pure ``due``
        check. No LLM, no timer, no queue. Consumption happens at dispatch
        time (not here) so a gate-suppressed turn does not silently drop a
        reminder. Fail-soft: any problem means "no due reminder".
        """
        try:
            agent = getattr(context, "agent_engine", None)
            probe = getattr(agent, "has_due_future_intention", None)
            if not callable(probe):
                return False
            try:
                return bool(probe(now) if now is not None else probe())
            except TypeError:
                return bool(probe())
        except Exception as error:
            logger.debug(
                "Future intention trigger unavailable: type={}",
                type(error).__name__,
            )
            return False

    def _proactive_trigger(
        self,
        context: ServiceContext,
        signals: ProactiveIntentSignals,
        budget: ProactiveBudgetState,
        now=None,
    ):
        """Deterministic trigger priority. Never calls a model."""
        return classify_trigger(
            user_question_pending=bool(
                getattr(signals, "user_question_pending", False)
            ),
            unfinished_topic=bool(getattr(signals, "unfinished_topic", False)),
            has_useful_memory=bool(getattr(signals, "has_useful_memory", False)),
            memory_relevance_score=float(
                getattr(signals, "memory_relevance_score", 0.0) or 0.0
            ),
            meaningful_life_event=self._meaningful_life_event(
                context, budget.last_proactive_at
            ),
            goal_evidence=self._goal_evidence_trigger(context, now=now),
            explicit_reminder=self._future_intention_trigger(context, now=now),
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
        # Proactive V2: a user reply always clears the unanswered/dormant
        # pressure, and the reset is persisted so a reconnect cannot restore
        # a stale "unanswered" verdict.
        try:
            agent = getattr(context, "agent_engine", None) if context else None
            conf_uid = getattr(agent, "_character_conf_uid", None)
            if conf_uid:
                record_proactive_answered(self._budget_state(client_uid, conf_uid))
                self._persist_budget(client_uid, conf_uid)
        except Exception as error:
            logger.debug(
                "Proactive budget reset on user activity skipped: type={}",
                type(error).__name__,
            )

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

    @staticmethod
    def _proactive_intent_signals(
        context: ServiceContext,
        state: ProactiveRuntimeState,
    ) -> ProactiveIntentSignals:
        """Derive rich heuristic signals from existing state (no LLM calls)."""
        agent = getattr(context, "agent_engine", None)
        history = list(getattr(agent, "_memory", None) or [])

        memory_texts = []
        list_memories = getattr(agent, "list_character_memories", None)
        if callable(list_memories):
            try:
                memory_texts = [
                    str(item.get("text", "")).strip()
                    for item in (list_memories() or [])
                    if isinstance(item, dict) and str(item.get("text", "")).strip()
                ]
            except Exception:
                memory_texts = []

        # Existing relationship state only; weak modifier (rank/3).
        relationship_familiarity = 0.0
        rank = {"stranger": 0, "familiar": 1, "close": 2, "dating": 3}.get(
            str(getattr(agent, "relationship_status", "") or "")
        )
        if rank is not None:
            relationship_familiarity = rank / 3.0

        return compute_intent_signals(
            history,
            memory_texts,
            consecutive_ignored_proactive=state.consecutive_ignored_proactive,
            recent_proactive_intents=tuple(state.recent_proactive_intents),
            recent_proactive_topic_signatures=tuple(
                tuple(signature)
                for signature in state.recent_proactive_topic_signatures
            ),
            relationship_familiarity=relationship_familiarity,
        )

    @staticmethod
    def _proactive_context_from_heuristic_decision(
        decision,
        state: ProactiveRuntimeState,
        signals: ProactiveIntentSignals,
    ) -> ProactiveIntentContext:
        """Build the preserved Heuristics v2 prompt hints for one turn."""
        return ProactiveIntentContext(
            intent=decision.intent or ProactiveIntent.CASUAL_OBSERVATION,
            strategy=decision.strategy,
            user_has_replied_since_last_proactive=(
                state.consecutive_ignored_proactive == 0
            ),
            consecutive_ignored=max(0, state.consecutive_ignored_proactive),
            recent_silence_acknowledgment=(
                ProactiveIntent.REACT_TO_SILENCE in state.recent_proactive_intents
            ),
            topic_continuity_band=band_for(signals.topic_continuity_score, 0.35, 0.6),
            topic_staleness_band=band_for(signals.topic_staleness_score, 0.4, 0.7),
            user_engagement_band=band_for(signals.recent_user_engagement, 0.4, 0.7),
            dominant_topic_keywords=signals.dominant_recent_topic,
            avoid_recent_topics=tuple(
                tuple(signature)
                for signature in state.recent_proactive_topic_signatures[-3:]
            ),
        )

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

                # ---- Proactive V2 deterministic gate -------------------
                # Runs BEFORE the in-flight lock and BEFORE any provider
                # call: a suppressed turn must never reach the LLM. Signals
                # and trigger priority are local computations; there is no
                # model call anywhere in this block.
                conf_uid = getattr(
                    getattr(context, "agent_engine", None), "_character_conf_uid", None
                )
                budget = self._budget_state(client_uid, conf_uid)
                gate_signals = self._proactive_intent_signals(context, state)
                trigger = self._proactive_trigger(context, gate_signals, budget)
                gate_config = getattr(
                    self, "_proactive_gate_config", ProactiveGateConfig()
                )
                decision_gate = evaluate_gate(
                    budget,
                    gate_config,
                    trigger,
                    now=utcnow(),
                    tz=getattr(self, "_user_timezone_for_client", None)
                    or getattr(context, "user_timezone", None),
                    connection_valid=True,
                    generation_in_progress=False,
                )
                if not decision_gate.allowed:
                    logger.debug(
                        "Proactive suppressed: reason={} priority={} daily_remaining={}",
                        decision_gate.reason,
                        decision_gate.priority,
                        decision_gate.daily_remaining,
                    )
                    self._persist_budget(client_uid, conf_uid)
                    # Wait out the current window instead of spinning.
                    await asyncio.sleep(
                        min(60.0, max(5.0, machine.seconds_until_eligible(state)))
                    )
                    continue
                # ------------------------------------------------------

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
                record_proactive_dispatch(
                    budget,
                    gate_config,
                    trigger,
                    now=utcnow(),
                    tz=getattr(context, "user_timezone", None),
                )
                self._persist_budget(client_uid, conf_uid)
                # A dispatched reminder is consumed exactly once, so the same
                # due row can never nag twice. Suppressed turns never reach
                # here, so a gate-suppressed reminder stays pending.
                if str(getattr(trigger, "reason", "")) == "explicit_reminder":
                    try:
                        agent = getattr(context, "agent_engine", None)
                        consume = getattr(
                            agent, "consume_due_future_intentions", None
                        )
                        if callable(consume):
                            try:
                                consumed = consume(utcnow())
                            except TypeError:
                                consumed = consume()
                            logger.info(
                                "Future intention consumed on dispatch: count={}",
                                int(consumed or 0),
                            )
                    except Exception as error:
                        logger.debug(
                            "Future intention consume skipped: type={}",
                            type(error).__name__,
                        )
                logger.info(
                    "Proactive chat generation started: request_origin=proactive, "
                    "ignored_count={} priority={} trigger={} "
                    "daily_used={}/{}",
                    state.consecutive_ignored_proactive,
                    trigger.priority,
                    trigger.reason,
                    budget.daily_request_count,
                    gate_config.proactive_daily_hard_limit,
                )
                followup_context = machine.proactive_followup_context(state)
                # Reuse the signals the gate already computed: identical input,
                # one computation, no extra cost.
                signals = gate_signals
                forced_ignored_question = (
                    followup_context.previous_proactive_ignored
                    and followup_context.previous_proactive_expected_response
                )
                if (
                    machine.config.intent_strategy == ProactiveIntentStrategy.HEURISTIC
                    and not forced_ignored_question
                ):
                    signals = self._proactive_intent_signals(context, state)
                decision = resolve_proactive_intent_decision(
                    followup_context, state, machine, signals
                )
                # Typed Autonomous Decision Layer record (observation only:
                # the existing machine decision above stays authoritative).
                _log_typed_proactive_decision(machine, state, decision)
                if decision.strategy == ProactiveTurnStrategy.SEMANTIC_AUTO:
                    try:
                        intent_context = build_semantic_proactive_context(state)
                    except Exception as error:
                        logger.warning(
                            "Semantic proactive context unavailable; using "
                            "Heuristics v2 fallback: error_type={}",
                            type(error).__name__,
                        )
                        signals = self._proactive_intent_signals(context, state)
                        decision = resolve_proactive_intent_decision(
                            followup_context,
                            state,
                            machine,
                            signals,
                            strategy=ProactiveIntentStrategy.HEURISTIC,
                            fallback_reason="semantic_context_construction_failed",
                        )
                        intent_context = (
                            self._proactive_context_from_heuristic_decision(
                                decision, state, signals
                            )
                        )
                elif decision.strategy == ProactiveTurnStrategy.HEURISTIC:
                    intent_context = self._proactive_context_from_heuristic_decision(
                        decision, state, signals
                    )
                else:
                    # Deterministic ignored-question priority.  The follow-up
                    # block carries the substantive instruction.
                    intent_context = self._proactive_context_from_heuristic_decision(
                        decision, state, signals
                    )

                if decision.strategy == ProactiveTurnStrategy.SEMANTIC_AUTO:
                    logger.info(
                        "[PROACTIVE INTENT] strategy=semantic_auto forced=false"
                    )
                elif decision.strategy == ProactiveTurnStrategy.FORCED_IGNORED_QUESTION:
                    logger.info(
                        "[PROACTIVE INTENT] "
                        "strategy=forced_ignored_question forced=true"
                    )
                else:
                    logger.info(
                        "[PROACTIVE INTENT] strategy={} intent={} reason={}",
                        decision.strategy,
                        decision.intent,
                        decision.reason,
                    )
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
                        "proactive_intent": intent_context.as_dict(),
                    },
                )
                state.proactive_generation_in_progress = False
                if response and revision == state.activity_revision:
                    machine.record_proactive_sent(
                        state, response_text=response, intent=decision.intent
                    )
                elif revision != state.activity_revision:
                    # User activity arrived after generation had meaningfully
                    # started.  End this scheduler so the user handler can
                    # install a fresh timer after starting the reply turn.
                    return
                else:
                    # Empty/cancelled work or user activity gets a fresh idle
                    # period and never increments the ignored counter.
                    machine.record_user_activity(state)
                    # Proactive V2: a suppressed/undelivered proactive turn
                    # still costs nothing from the budget, but it must widen
                    # the gap so a silent backend cannot keep firing.
                    try:
                        record_suppressed(
                            budget,
                            gate_config,
                            now=utcnow(),
                            tz=getattr(context, "user_timezone", None),
                        )
                        self._persist_budget(client_uid, conf_uid)
                    except Exception as error:
                        logger.debug(
                            "Suppressed proactive accounting skipped: type={}",
                            type(error).__name__,
                        )
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
            "voice-output-toggle": self._handle_voice_output_toggle,
            "request-init-config": self._handle_init_config_request,
            "heartbeat": self._handle_heartbeat,
            "fetch-world-state": self._handle_fetch_world_state,
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
        """Handle client disconnection (idempotent; safe to run twice)."""
        # Capture the context BEFORE popping: group notify and close()
        # need it, and a second cleanup call must not crash on lookups.
        context = self.client_contexts.get(client_uid)
        await self._cancel_proactive_timer(client_uid)
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            try:
                await handle_group_interrupt(
                    group_id=group.group_id,
                    heard_response="",
                    current_conversation_tasks=self.current_conversation_tasks,
                    chat_group_manager=self.chat_group_manager,
                    client_contexts=self.client_contexts,
                    broadcast_to_group=self.broadcast_to_group,
                )
            except Exception as error:
                # A broken peer socket must never abort our own cleanup.
                logger.warning(
                    "Group interrupt notify skipped: type={}",
                    type(error).__name__,
                )

        try:
            await handle_client_disconnect(
                client_uid=client_uid,
                chat_group_manager=self.chat_group_manager,
                client_connections=self.client_connections,
                send_group_update=self.send_group_update,
            )
        except Exception as error:
            logger.warning(
                "Group disconnect notify skipped: type={}",
                type(error).__name__,
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
            if task is not None and not task.done() and getattr(
                task, "_olv_single_turn", False
            ):
                # RECONNECT ≠ NEW CONVERSATION: an in-flight single turn is
                # detached, never cancelled, so its response still persists
                # into its session and a post-reconnect resync picks it up.
                # Socket sends on the dead connection fail safely
                # (safe_send / fail-soft TTS queue); nothing here can raise.
                task._olv_detached = True
                try:
                    history_uid = getattr(task, "_olv_history_uid", "")
                except Exception:
                    history_uid = ""
                logger.info(
                    "TURN_DETACHED history_uid={} "
                    "(in-flight turn continues after disconnect)",
                    history_uid,
                )
                # Keep a strong reference AND stay discoverable by history:
                # a new trigger on this session must wait for the orphan
                # instead of persisting a second turn concurrently. Keyed by
                # history_uid because the dead client_uid is already gone.
                if history_uid:
                    self._detached_registry()[history_uid] = task
                    task.add_done_callback(
                        lambda finished, uid=history_uid: self._on_detached_turn_done(
                            finished, uid
                        )
                    )
            elif task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        # Drop this (now stale) connection's history binding; a reconnect
        # re-registers itself, and an explicit new session rebinds below.
        bound = self._subscriber_registry().get(client_uid)
        if bound:
            self._subscriber_registry().pop(client_uid, None)

        # Call context close to clean up resources (e.g., MCPClient)
        if context:
            try:
                await context.close()
            except Exception as error:
                logger.warning(
                    "Service context close skipped: type={}",
                    type(error).__name__,
                )

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
        # Safe lookup: the context may already be gone if a disconnect
        # cleanup raced this message. Never crash the receive loop here.
        context = self.client_contexts.get(client_uid)
        if context is None:
            logger.warning("Interrupt ignored: unknown client (post-cleanup?)")
            return
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
        logger.info("SESSION_RESTORE history_uid={}", history_uid)
        self._update_user_timezone(context, data)
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
            user_timezone=context.user_timezone,
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
            json.dumps(
                {
                    "type": "history-data",
                    "history_uid": history_uid,
                    "messages": messages,
                }
            )
        )
        self._subscriber_registry()[client_uid] = history_uid
        # A turn detached by a mid-turn disconnect may still be running on
        # this history: its chain-start went to a dead socket, so tell the
        # new viewer explicitly instead of leaving the UI idle. Only
        # DETACHED turns qualify — a live turn already owns its lifecycle
        # (re-emitting chain-start would clear its audio queue mid-stream).
        # The matching chain-end is sent by _deliver_history_to_subscriber
        # when that turn completes.
        try:
            detached = self._detached_registry().get(history_uid)
            if detached is not None and not detached.done():
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "control",
                            "text": "conversation-chain-start",
                            "history_uid": history_uid,
                        }
                    )
                )
        except Exception:
            pass
        # Selecting history is activity.  Reconnect therefore starts a fresh
        # idle period and never replays timers/messages from the old socket.
        await self._activate_proactive_for_history(client_uid, history_uid)

    def _on_detached_turn_done(self, task: asyncio.Task, history_uid: str) -> None:
        """Detached turn finished: release it and resync the live viewer.

        Runs as a done-callback, so it must never raise.
        """
        self._detached_registry().pop(history_uid, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.warning(
                "Detached turn ended with error: history_uid={} type={}",
                history_uid,
                type(error).__name__,
            )
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._deliver_history_to_subscriber(history_uid))

    async def _deliver_history_to_subscriber(self, history_uid: str) -> None:
        """Push an authoritative transcript to whoever is viewing it now.

        Used when a turn detached by a disconnect finishes: the response it
        persisted must reach the reconnected socket, otherwise the UI keeps
        showing the half-finished conversation until a manual reload.
        """
        # _history_subscribers maps client_uid -> history_uid; find the live
        # socket currently viewing this history.
        client_uid = next(
            (
                uid
                for uid, bound in self._subscriber_registry().items()
                if bound == history_uid
            ),
            None,
        )
        websocket = self.client_connections.get(client_uid) if client_uid else None
        if websocket is None:
            return
        context = self.client_contexts.get(client_uid)
        if context is None:
            return
        try:
            messages = [
                msg
                for msg in get_history(
                    context.character_config.conf_uid, history_uid
                )
                if msg["role"] != "system"
            ]
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "history-data",
                        "history_uid": history_uid,
                        "messages": messages,
                    }
                )
            )
            logger.info(
                "TURN_RESYNCED history_uid={} client_uid={}",
                history_uid,
                client_uid,
            )
            # Close the lifecycle a fetch-time chain-start may have opened:
            # the detached turn is done, so the viewer must return to idle.
            # Skipped when the viewer already runs a newer live turn — that
            # turn owns thinking now and will close it with its own chain-end.
            live = self.current_conversation_tasks.get(client_uid or "")
            if live is None or live.done():
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "control",
                            "text": "conversation-chain-end",
                            "history_uid": history_uid,
                        }
                    )
                )
        except Exception as error:
            logger.warning(
                "Detached turn resync skipped: history_uid={} type={}",
                history_uid,
                type(error).__name__,
            )

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation of new chat history"""
        await self._cancel_proactive_timer(client_uid)
        context = self.client_contexts[client_uid]
        history_uid = create_new_history(context.character_config.conf_uid)
        if history_uid:
            logger.info(
                "SESSION_CREATE history_uid={} (explicit new conversation)",
                history_uid,
            )
            context.history_uid = history_uid
            self._update_user_timezone(context, data)
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
                user_timezone=context.user_timezone,
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
            self._update_user_timezone(context, data)
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
                user_timezone=context.user_timezone,
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
        if context.vad_engine is None:
            # Backend VAD disabled: raw-audio endpointing unavailable.
            # Fail safe without crashing the voice session; the client
            # uses explicit mic-audio-end instead.
            logger.debug("Ignoring raw-audio-data: backend VAD is disabled")
            return
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

    async def _ensure_history_for_trigger(
        self,
        websocket: WebSocket,
        client_uid: str,
        context: ServiceContext,
        data: WSMessage,
    ) -> bool:
        """Create exactly one history for a history-less socket (orphan guard).

        Mirrors the _handle_create_history setup (uid + memory init +
        new-history-created notify) so the turn about to start persists
        normally. Returns True when context.history_uid is valid
        afterwards; False means the caller must drop the trigger with an
        explicit error reply (fail closed, never a silent orphan turn).
        """
        history_uid = create_new_history(context.character_config.conf_uid)
        if not history_uid:
            logger.error("Orphan guard: history creation failed; dropping trigger")
            try:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Could not start a conversation: history unavailable. Please resend.",
                        }
                    )
                )
            except Exception:
                pass
            return False
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
            user_timezone=context.user_timezone,
        )
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                    }
                )
            )
        except Exception:
            pass
        logger.info(
            "SESSION_CREATE history_uid={} (orphan guard: history-less socket)",
            history_uid,
        )
        return True

    async def _handle_conversation_trigger(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle triggers that start a conversation"""
        msg_type = data.get("type", "")
        if msg_type in {"text-input", "mic-audio-end"}:
            await self._record_user_activity(client_uid)
            context = self.client_contexts.get(client_uid)
            if context is not None:
                self._update_user_timezone(context, data)
            # Orphan-turn guard: a fresh socket has history_uid == "" until
            # history selection completes. A turn started here would run
            # but persist nothing (and a later history-data would wipe the
            # frontend bubble). Atomically create exactly one history per
            # socket instead; the sequential receive loop makes the
            # check-and-create race-free. Fail closed on creation failure.
            #
            # RECONNECT ≠ NEW CONVERSATION: the frontend sends the active
            # history_uid it already holds with every text-input. When the
            # socket's session is still unrestored but the payload names an
            # existing history, adopt it instead of minting a new session.
            # This closes the reconnect race (send lands before
            # fetch-and-set-history completes) without touching BUG A:
            # a trigger with no usable uid still gets exactly one history.
            if context is not None and not context.history_uid:
                claimed_uid = str((data or {}).get("history_uid", "") or "").strip()
                adopted = False
                if claimed_uid:
                    try:
                        if get_metadata(
                            context.character_config.conf_uid, claimed_uid
                        ):
                            context.history_uid = claimed_uid
                            context.agent_engine.set_memory_from_history(
                                conf_uid=context.character_config.conf_uid,
                                history_uid=claimed_uid,
                                user_timezone=context.user_timezone,
                            )
                            adopted = True
                            logger.info(
                                "SESSION_RESTORE history_uid={} "
                                "(adopted from trigger payload)",
                                claimed_uid,
                            )
                    except Exception as error:
                        logger.debug(
                            "Session adopt skipped: type={}",
                            type(error).__name__,
                        )
                if not adopted:
                    ensured = await self._ensure_history_for_trigger(
                        websocket, client_uid, context, data
                    )
                    if not ensured:
                        return
        trigger_context = self.client_contexts.get(client_uid)
        if trigger_context is None:
            # Disconnect cleanup raced this trigger: answer with an error
            # event instead of crashing the receive loop with KeyError.
            logger.warning(
                "Conversation trigger ignored: unknown client (post-cleanup?)"
            )
            try:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Session ended during send; please resend.",
                        }
                    )
                )
            except Exception:
                pass
            return
        await handle_conversation_trigger(
            msg_type=msg_type,
            data=data,
            client_uid=client_uid,
            context=trigger_context,
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            detached_turns=self._detached_registry(),
            broadcast_to_group=self.broadcast_to_group,
        )
        if msg_type in {"text-input", "mic-audio-end"}:
            context = self.client_contexts.get(client_uid)
            if context:
                if context.history_uid:
                    self._subscriber_registry()[client_uid] = context.history_uid
                await self._activate_proactive_for_history(
                    client_uid,
                    context.history_uid,
                    user_activity=False,
                )

    async def _handle_fetch_world_state(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Return the authoritative World/Life snapshot (read-only).

        Used by the optional Life State observability widget. Reconciles
        lazily with the session timezone, persists only when changed, and
        never triggers LLM/provider calls, proactive turns, or schedulers.
        """
        context = self.client_contexts.get(client_uid)
        if context is None:
            return
        self._update_user_timezone(context, data)
        try:
            # Read-only widget path: reconcile time but never write an
            # autonomous decision (decide=False).
            snapshot = load_and_reconcile_world_state(
                context.character_config.conf_uid,
                tz=context.user_timezone,
                decide=False,
            )
            payload = {
                "type": "world-state",
                "location": snapshot.location,
                "activity": snapshot.activity,
                "energy": snapshot.energy,
                "mood": snapshot.mood,
                "time_context": snapshot.time_context,
                "activity_started_at": snapshot.activity_started_at,
                "last_update_at": snapshot.last_update_at,
            }
        except Exception as error:
            logger.warning("World state fetch skipped: type={}", type(error).__name__)
            payload = {"type": "world-state", "error": "unavailable"}
        await websocket.send_text(json.dumps(payload))

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

    async def _handle_voice_output_toggle(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Set the session Voice Output (TTS synthesis) enabled state.

        When OFF the backend skips ElevenLabs/audio synthesis entirely and
        sends text-only display payloads, so no credits are consumed.
        """
        context = self.client_contexts.get(client_uid)
        enabled = bool(data.get("enabled", True))
        if context is None:
            return
        context.voice_output_enabled = enabled
        logger.info(
            "Voice output toggle for client {} -> enabled={}", client_uid, enabled
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
