from typing import (
    AsyncIterator,
    List,
    Dict,
    Any,
    Callable,
    Literal,
    Union,
    Optional,
)
import asyncio
from datetime import datetime, timezone
from loguru import logger
from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ..stateless_llm.claude_llm import AsyncLLM as ClaudeAsyncLLM
from ..stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleAsyncLLM
from ...chat_history_manager import (
    get_history,
    get_history_list,
    get_metadata,
    update_metadate,
    update_summary_metadata,
)
from ...proactive_chat import (
    ProactiveFollowupContext,
    ProactiveIntentContext,
    ProactiveTurnStrategy,
    format_followup_instruction,
    format_intent_instruction,
)
from ...character_state import (
    CharacterState,
    activate_goal,
    add_character_memory as persist_character_memory,
    build_character_memory_context,
    complete_goal,
    ensure_seed_goals,
    goal_status_counts,
    load_character_state,
    migrate_relationship_if_needed,
    record_goal_evidence,
    record_interaction_preference,
    remove_character_memory as remove_persisted_character_memory,
    reset_character_memory as reset_persisted_character_memory,
    reset_character_state as reset_persisted_character_state,
    save_character_state,
    set_character_relationship,
)
from ...interaction_preferences import (
    active_preferences,
    build_interaction_preference_context,
)
from ...autonomous_decision import goal_evidence_summary
from ...character_memory_commands import parse_memory_command
from ...self_model import build_self_context, derive_activity_preferences
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource
from prompts import prompt_loader
from ...mcpp.tool_manager import ToolManager
from ...mcpp.json_detector import StreamJSONDetector
from ...mcpp.types import ToolCallObject
from ...mcpp.tool_executor import ToolExecutor
from ..context_window import (
    ContextBudgetExceeded,
    ContextSelection,
    estimate_tokens,
    select_messages_for_context,
)
from ..conversation_summary import (
    PREV_SESSION_MAX_CHARS,
    PREV_SESSION_MAX_COUNT,
    IncrementalSummarizer,
    SummaryState,
    build_previous_session_context,
    build_summary_message,
)
from ..relationship_context import (
    RelationshipState,
    RelationshipStatus,
    build_relationship_context,
    detect_relationship_update,
    normalize_relationship_status,
)
from ...world_state import (
    DECISION_GOAL_ACTIVITY_HINTS,
    DecisionContextSignals,
    DecisionInputs,
    apply_reactive,
    build_world_state_context,
    format_session_recency,
    format_temporal_anchor,
    load_and_reconcile_world_state,
    load_world_state,
    memory_age_label,
    reconcile,
    save_world_state,
    utcnow,
)
from ...episodic_memory import (
    extract_and_store_episodic,
    is_episodic_candidate,
    load_episodic_events,
    render_episodic_context,
    retrieve_episodic_events,
)
import time
from ...request_latency import (
    get_latency_tracker,
    reset_latency_phase,
    set_latency_phase,
)

# Request-only cue for every proactive turn. Proactive generation is a
# system-initiated turn, so the provider request never contains a *current*
# user turn from the real conversation; the transcript is always history.
# Some OpenAI-compatible providers (observed: Ollama Cloud) can complete a
# stream with zero assistant tokens when the request ends on an assistant
# message (e.g. after Mili's first proactive message, or after any normal
# user/assistant exchange). This short, model-neutral, request-only user turn
# gives the provider a current generation cue. It is appended after context
# selection, before the provider call, and is never persisted: it never enters
# _memory, history, summary, memory parsing, relationship logic, or the UI.
PROACTIVE_TURN_CUE = "Continue the conversation naturally on your own."


# Cost safety guard (independent of the proactive budget): hard cap on tool
# round-trips per turn. Both tool interaction loops are `while True` by
# design, so without this a model that keeps calling tools would generate an
# unbounded number of provider requests inside ONE user turn. On reaching the
# cap the loop stops cleanly and the existing graceful/fail-soft response
# path finishes the turn.
TOOL_LOOP_MAX_ITERATIONS = 4


class BasicMemoryAgent(AgentInterface):
    """Agent with basic chat memory and tool calling support."""

    _system: str = "You are a helpful assistant."

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        use_mcpp: bool = False,
        interrupt_method: Literal["system", "user"] = "user",
        tool_prompts: Dict[str, str] = None,
        tool_manager: Optional[ToolManager] = None,
        tool_executor: Optional[ToolExecutor] = None,
        mcp_prompt_string: str = "",
        context_management_enabled: bool = True,
        context_window_override: Optional[int] = None,
        context_safety_margin: int = 1024,
        rolling_summary_enabled: bool = True,
        summary_target_tokens: int = 320,
        summary_max_tokens: int = 384,
        summary_min_new_messages: int = 4,
        character_name: Optional[str] = None,
        character_avatar: Optional[str] = None,
    ):
        """Initialize agent with LLM and configuration."""
        super().__init__()
        self._character_name = (character_name or "Mili").strip() or "Mili"
        self._character_avatar = character_avatar or ""
        self._memory = []
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._use_mcpp = use_mcpp
        self.interrupt_method = interrupt_method
        self._tool_prompts = tool_prompts or {}
        self._interrupt_handled = False
        self.prompt_mode_flag = False

        self._tool_manager = tool_manager
        self._tool_executor = tool_executor
        self._mcp_prompt_string = mcp_prompt_string
        self._json_detector = StreamJSONDetector()
        self._context_management_enabled = context_management_enabled
        self._context_window_override = context_window_override
        self._context_safety_margin = context_safety_margin
        self._rolling_summary_enabled = rolling_summary_enabled
        self._summary_min_new_messages = summary_min_new_messages
        self._summary_state = SummaryState()
        self._summary_conf_uid: Optional[str] = None
        self._summary_history_uid: Optional[str] = None
        self._summary_lock = asyncio.Lock()
        self._summarizer = IncrementalSummarizer(
            llm=llm,
            target_tokens=summary_target_tokens,
            maximum_tokens=summary_max_tokens,
        )
        self._relationship_state = RelationshipState()
        self._character_state = CharacterState()
        self._character_conf_uid: Optional[str] = None
        # IANA timezone for user-local World State time rules (None = UTC).
        self._user_timezone: Optional[str] = None
        # Absolute timestamp (UTC ISO) of the latest message in a *previous*
        # session. Cached once per history load; the recency line itself is
        # rendered per turn so midnight crossings stay correct with zero
        # per-iteration I/O.
        self._prev_session_at: Optional[str] = None
        # Cached previous-session summaries (newest first), each
        # {"at": <absolute ISO stamp>, "text": ...}. Resolved once per
        # history load from persisted rolling summaries; age labels render
        # per turn below so midnight crossings stay correct.
        self._prev_session_summaries: List[Dict[str, str]] = []
        # Active session file id; used to keep its metadata truthful.
        self._history_uid: str = ""

        # Clean user text for episodic retrieval (set per turn from
        # BatchInput metadata). Kept separate from _memory because the LLM
        # input text may carry an appended search block that must not pollute
        # the retrieval query. Empty means "fall back to the last user turn".
        self._episodic_query: str = ""
        # Per-turn episodic retrieval selection (query, events), shared by the
        # prompt block and the Autonomous Decision Layer.
        self._episodic_selection_cache: Any = None

        self._formatted_tools_openai = []
        self._formatted_tools_claude = []
        if self._tool_manager:
            self._formatted_tools_openai = self._tool_manager.get_formatted_tools(
                "OpenAI"
            )
            self._formatted_tools_claude = self._tool_manager.get_formatted_tools(
                "Claude"
            )
            logger.debug(
                f"Agent received pre-formatted tools - OpenAI: {len(self._formatted_tools_openai)}, Claude: {len(self._formatted_tools_claude)}"
            )
        else:
            logger.debug(
                "ToolManager not provided, agent will not have pre-formatted tools."
            )

        self._set_llm(llm)
        self.set_system(system if system else self._system)

        if self._use_mcpp and not all(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is True, but some MCP components are missing in the agent. Tool calling might not work as expected."
            )
        elif not self._use_mcpp and any(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is False, but some MCP components were passed to the agent."
            )

        logger.info("BasicMemoryAgent initialized.")

    def _set_llm(self, llm: StatelessLLMInterface):
        """Set the LLM for chat completion."""
        self._llm = llm
        self.chat = self._chat_function_factory()

    def set_system(self, system: str):
        """Set the system prompt."""
        logger.debug(
            "Memory Agent: setting system prompt (chars={})", len(system or "")
        )

        if self.interrupt_method == "user":
            system = f"{system}\n\nIf you received `[interrupted by user]` signal, you were interrupted."

        self._system = system

    def _select_context(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        protected_start: Optional[int] = None,
    ) -> ContextSelection:
        """Budget one API request without changing ``_memory`` or disk history."""
        return select_messages_for_context(
            messages=messages,
            system_prompt=system_prompt,
            model=getattr(self._llm, "model", None),
            reserved_output_tokens=getattr(self._llm, "max_tokens", None),
            safety_margin=self._context_safety_margin,
            context_window_override=self._context_window_override,
            tools=tools,
            protected_start=protected_start,
        )

    def _log_context_stats(self, selection: ContextSelection) -> None:
        stats = selection.stats
        logger.info(
            "Context stats: model={}, context_limit={}, reserved_output={}, "
            "safety_margin={}, maximum_input_budget={}, system_tokens={}, tool_tokens={}, "
            "history_tokens_before={}, history_tokens_after={}, "
            "messages_before={}, messages_after={}, trimmed={}, "
            "estimated_input_tokens={}, fallback_limit={}",
            stats.model,
            stats.context_limit,
            stats.reserved_output,
            stats.safety_margin,
            stats.maximum_input_budget,
            stats.system_tokens,
            stats.tool_tokens,
            stats.history_tokens_before,
            stats.history_tokens_after,
            stats.messages_before,
            stats.messages_after,
            stats.trimmed,
            stats.estimated_input_tokens,
            stats.used_fallback_limit,
        )
        tracker = get_latency_tracker()
        if tracker:
            tracker.message_count = stats.messages_after
            tracker.estimated_input_tokens = stats.estimated_input_tokens

    def _prepare_context(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        protected_start: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self._context_management_enabled:
            logger.debug(
                "Context management disabled: model={}, messages={}",
                getattr(self._llm, "model", "unknown"),
                len(messages),
            )
            return list(messages)
        selection = self._select_context(
            messages,
            system_prompt,
            tools=tools,
            protected_start=protected_start,
        )
        self._log_context_stats(selection)
        return selection.messages

    def _load_summary_state(self, conf_uid: str, history_uid: str) -> None:
        metadata = get_metadata(conf_uid, history_uid)
        text = metadata.get("conversation_summary", "")
        through = metadata.get("summary_through_message_index", 0)
        try:
            through = max(0, min(int(through or 0), len(self._memory)))
        except (TypeError, ValueError):
            through = 0
        self._summary_conf_uid = conf_uid
        self._summary_history_uid = history_uid
        self._summary_state = SummaryState(
            text=text if isinstance(text, str) else "",
            summarized_through=through,
            updated_at=metadata.get("summary_updated_at"),
        )

    def _load_character_state(self, conf_uid: str) -> None:
        """Load (and lazily migrate) the character-level state for this conf."""
        state = load_character_state(conf_uid)
        state = migrate_relationship_if_needed(conf_uid, state)
        self._character_state = state
        self._character_conf_uid = conf_uid
        # Seed the finite self-model goals once, right where the state is
        # loaded. Idempotent: a character that already has goals keeps exactly
        # those (no re-seed, no status rewrite, no reactivation), and the seed
        # is persisted immediately so a restart never re-derives it. Fail-soft:
        # a goal problem must never break history load.
        self._ensure_goals_seeded()
        # Restart/offline fallback: a turn may carry no timezone (proactive,
        # reconnect). The persisted last-known zone keeps local
        # interpretation stable; a fresh session value always wins.
        if not self._user_timezone:
            self._user_timezone = state.user_timezone
        self._relationship_state = RelationshipState(
            status=state.relationship_status,
            updated_at=state.relationship_updated_at,
            reason=state.relationship_reason,
        )
        # Stage 7: initialize (or lazily reconcile, e.g. after an offline
        # gap / server restart) the character-scoped World/Life State.
        # Fail-soft by design: world problems must never break history load.
        try:
            load_and_reconcile_world_state(conf_uid, tz=self._user_timezone)
        except Exception as error:
            logger.warning("World state init skipped: type={}", type(error).__name__)
        logger.info(
            "Character state stats: relationship_status={}, "
            "character_memory_count={}, relationship_update_trigger=load_history",
            state.relationship_status,
            len(state.memories),
        )

    @staticmethod
    def _latest_other_session_at(
        conf_uid: str, current_history_uid: str
    ) -> Optional[str]:
        """Latest message timestamp of any session except the current one.

        Fail-soft: unreadable store or no previous session yields None and
        the recency line is simply omitted from the prompt.
        """
        try:
            # Read-only scan: must never delete empty histories (legacy
            # migration reads them later in this same load path).
            histories = get_history_list(conf_uid, cleanup=False)
        except Exception:
            return None
        best = ""
        for item in histories or []:
            try:
                if str(item.get("uid", "")) == str(current_history_uid):
                    continue
                stamp = str(item.get("timestamp") or "")
            except Exception:
                continue
            if stamp and stamp > best:
                best = stamp
        return best or None

    @staticmethod
    def _load_previous_session_summaries(
        conf_uid: str, current_history_uid: str
    ) -> List[Dict[str, str]]:
        """Persisted rolling summaries of previous sessions (newest first).

        Read-only and bounded: at most PREV_SESSION_MAX_COUNT sessions,
        each truncated to PREV_SESSION_MAX_CHARS. Skips the current
        session, empty/corrupt stores, and sessions without a usable
        summary. Each item keeps the absolute ``at`` stamp
        (``summary_updated_at``, falling back to the session timestamp);
        relative age labels render per turn, so nothing relative is ever
        stored.
        """
        try:
            histories = get_history_list(conf_uid, cleanup=False)
        except Exception:
            return []
        ranked = []
        for item in histories or []:
            try:
                uid = str((item or {}).get("uid", ""))
                if not uid or uid == str(current_history_uid):
                    continue
                stamp = str((item or {}).get("timestamp") or "")
            except Exception:
                continue
            if stamp:
                ranked.append((stamp, uid))
        ranked.sort(reverse=True)
        out: List[Dict[str, str]] = []
        for stamp, uid in ranked:
            if len(out) >= PREV_SESSION_MAX_COUNT:
                break
            try:
                metadata = get_metadata(conf_uid, uid)
            except Exception:
                continue
            if not isinstance(metadata, dict):
                continue
            text = metadata.get("conversation_summary", "")
            if not isinstance(text, str) or not text.strip():
                continue
            at = metadata.get("summary_updated_at") or stamp
            if not isinstance(at, str) or not at.strip():
                continue
            out.append(
                {"at": at.strip(), "text": text.strip()[:PREV_SESSION_MAX_CHARS]}
            )
        # A partially recovered older conversation (forensic pass) is appended
        # last so it can never displace a real stored summary. It reuses this
        # loader's exact item shape, so the existing renderer, age tags and
        # bounds apply unchanged and nothing new enters the prompt.
        try:
            from ...recovered_context import load_recovered_previous_session

            recovered = load_recovered_previous_session()
            if recovered and recovered.get("text"):
                out.append(recovered)
        except Exception as error:
            logger.debug(
                "Recovered context skipped: type={}", type(error).__name__
            )
        return out

    @property
    def relationship_status(self) -> RelationshipStatus:
        """Expose the character-level relationship state for backend/tests."""
        return self._relationship_state.status

    def _relationship_system_prompt(self, base_prompt: str) -> str:
        parts = [
            base_prompt,
            # Temporal anchor: reliable "today" (date + weekday + user tz).
            # Real clock by design (a date, not a ticking clock); the pure
            # formatter stays deterministic under test via fixed moments.
            format_temporal_anchor(tz=self._user_timezone),
        ]
        # Previous-session recency: absolute timestamp rendered per turn
        # (midnight-safe), cached timestamp resolved once per history load.
        recency_line = format_session_recency(
            getattr(self, "_prev_session_at", None), tz=self._user_timezone
        )
        if recency_line:
            parts.append(recency_line)
        # Cross-session continuity: cached previous-session summaries,
        # rendered per turn with dynamic age tags. Read-only cache from
        # history load; never the current session; bounded count/chars.
        prev_items = []
        for cached in getattr(self, "_prev_session_summaries", None) or []:
            try:
                text = str((cached or {}).get("text", ""))
                at = str((cached or {}).get("at", ""))
            except Exception:
                continue
            if text.strip() and at.strip():
                prev_items.append(
                    {
                        "age_tag": memory_age_label(at, tz=self._user_timezone),
                        "text": text,
                    }
                )
        prev_block = build_previous_session_context(prev_items)
        if prev_block:
            parts.append(prev_block)
        # Stage 7: VERY COMPACT read-only World/Life snapshot. Reconciled
        # lazily on every turn (conversation + proactive share this path),
        # persisted only when something actually changed. Fail-soft: the
        # prompt simply omits the line when no character is loaded or the
        # store is unavailable. Never touches Emotion/transformers output.
        snapshot = None
        world_line = ""
        if self._character_conf_uid:
            try:
                snapshot = load_and_reconcile_world_state(
                    self._character_conf_uid,
                    tz=self._user_timezone,
                    inputs=self._decision_inputs(),
                )
                world_line = build_world_state_context(snapshot)
            except Exception as error:
                logger.warning(
                    "World state context skipped: type={}", type(error).__name__
                )
        # Self Model v1: static identity + read-only live references. Pure
        # composer, no store, no LLM call. See self_model.py.
        live2d_name = getattr(self._live2d_model, "live2d_model_name", None)
        parts.append(
            build_self_context(
                character_name=self._character_name,
                avatar_present=bool(self._live2d_model) or bool(self._character_avatar),
                live2d_model_name=live2d_name,
                activity=getattr(snapshot, "activity", None),
                location=getattr(snapshot, "location", None),
                relationship_status=self._relationship_state.status,
                memory_count=len(self._character_state.memories),
            )
        )
        parts.append(
            build_relationship_context(
                self._relationship_state.status,
                updated_at=self._relationship_state.updated_at,
                tz=self._user_timezone,
            ),
        )
        # Persistent interaction preferences: the user's standing
        # constraints on HOW Mili talks to them. Placed after the identity
        # blocks (persona, relationship) and before facts/episodic context,
        # so they shape interaction without ever replacing the persona.
        preference_context = ""
        try:
            preference_context = build_interaction_preference_context(
                self._character_state.interaction_preferences,
                tz=self._user_timezone,
            )
        except Exception as error:
            logger.warning(
                "Interaction preference context skipped: type={}",
                type(error).__name__,
            )
        if preference_context:
            parts.append(preference_context)
        memory_context = build_character_memory_context(
            self._character_state, tz=self._user_timezone
        )
        if memory_context:
            parts.append(memory_context)
        # Episodic experiences: selected past events from any session,
        # retrieved against the latest user turn. Separate block from
        # long-term facts; empty when nothing relevant scores.
        try:
            episodic_block = self._episodic_context_for_prompt()
        except Exception as error:
            logger.warning("Episodic retrieval skipped: type={}", type(error).__name__)
            episodic_block = ""
        if episodic_block:
            parts.append(episodic_block)
        if world_line:
            parts.append(world_line)
        return "\n\n".join(parts)

    def _sync_relationship_metadata(self, status: str, reason: str) -> None:
        """Mirror the character-level relationship into the active session file.

        New history files are created with a hardcoded ``stranger`` default
        and were never refreshed, so the per-session metadata permanently
        disagreed with the real character-level state (and fed the legacy
        migration a wrong value). The character-level state stays the single
        source of truth; this only keeps the session record truthful.
        Best-effort and fail-soft: never affects the conversation.
        """
        if not self._character_conf_uid or not self._history_uid:
            return
        try:
            update_metadate(
                self._character_conf_uid,
                self._history_uid,
                {
                    "relationship_status": status,
                    "relationship_reason": reason,
                    "relationship_updated_at": self._relationship_state.updated_at,
                },
            )
        except Exception as error:
            logger.debug(
                "Session relationship metadata not synced: type={}",
                type(error).__name__,
            )

    def set_relationship_status(
        self,
        status: RelationshipStatus,
        *,
        trigger: str = "manual_backend_update",
    ) -> bool:
        """Persist an explicit relationship update at character level."""
        normalized = normalize_relationship_status(status)
        if normalized != status:
            raise ValueError(f"Unsupported relationship status: {status}")
        if not self._character_conf_uid:
            logger.warning("Relationship update skipped: no active character context")
            return False
        if normalized == self._relationship_state.status:
            return True

        updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_started = time.perf_counter()
        state = set_character_relationship(
            self._character_conf_uid,
            normalized,
            trigger,
            updated_at=updated_at,
        )
        tracker = get_latency_tracker()
        if tracker:
            tracker.add_character_state_save(
                (time.perf_counter() - save_started) * 1000
            )
        if state is None:
            logger.warning(
                "Relationship update failed: trigger={}",
                trigger,
            )
            return False

        self._character_state = state
        self._relationship_state = RelationshipState(
            status=normalized,
            updated_at=updated_at,
            reason=trigger,
        )
        self._sync_relationship_metadata(normalized, trigger)
        logger.info(
            "Relationship stats: relationship_status={}, "
            "relationship_updated=True, relationship_update_trigger={}",
            normalized,
            trigger,
        )
        return True

    def reset_relationship(self) -> bool:
        """Reset Mili's relationship for every conversation (character-level)."""
        return self.set_relationship_status("stranger", trigger="manual_reset")

    # ------------------------------------------------------------------
    # Self-model goals — explicit lifecycle, never inferred.
    # ------------------------------------------------------------------

    def _ensure_goals_seeded(self) -> bool:
        """Seed the finite goals once and persist. Idempotent, fail-soft.

        Returns True when a seed was actually written. A character that already
        carries goals is left untouched: no re-seeding, no status rewrite, no
        reactivation, no evidence-anchor reset.
        """
        try:
            state = getattr(self, "_character_state", None)
            if state is None:
                return False
            existing = list(getattr(state, "goals", None) or [])
            seeded = ensure_seed_goals(existing)
            if len(seeded) == len(existing) and all(
                str(item.get("id", "")) == str(other.get("id", ""))
                and str(item.get("status", "")) == str(other.get("status", ""))
                for item, other in zip(seeded, existing)
            ):
                # Already present; still normalise into the live object so the
                # in-memory view matches what is on disk.
                state.goals = seeded
                return False
            state.goals = seeded
            saved = save_character_state(self._character_conf_uid, state)
            logger.info(
                "Goal seeds: goals={} seeded=True persisted={}",
                goal_status_counts(seeded),
                bool(saved),
            )
            return bool(saved)
        except Exception as error:
            logger.debug("Goal seeding skipped: type={}", type(error).__name__)
            return False

    def goal_snapshot(self) -> Dict[str, Any]:
        """Compact goal state for the decision layer (pure, fail-soft)."""
        try:
            state = getattr(self, "_character_state", None)
            summary = goal_evidence_summary(getattr(state, "goals", None))
            summary["counts"] = goal_status_counts(getattr(state, "goals", None))
            return summary
        except Exception as error:
            logger.debug("Goal snapshot unavailable: type={}", type(error).__name__)
            return {"seed": 0, "active": 0, "done": 0, "active_ids": (), "counts": {}}

    def set_goal_status(self, goal_id: str, status: str) -> bool:
        """Explicit goal transition: ``seed``->``active``->``done``.

        The only supported moves are the two forward transitions; anything else
        (unknown id, wrong current status, ``done``->anything) is rejected, so
        there is no auto-complete and no reactivation. Nothing here is driven
        by a model: the caller states the transition, and the decision layer
        only ever reads the result.
        """
        try:
            state = getattr(self, "_character_state", None)
            if state is None:
                return False
            goals = list(getattr(state, "goals", None) or [])
            target = str(status or "").strip().lower()
            if target == "active":
                updated, changed = activate_goal(goals, goal_id)
            elif target == "done":
                updated, changed = complete_goal(goals, goal_id)
            else:
                logger.debug(
                    "Goal transition rejected: goal_id={} status={}",
                    goal_id,
                    target,
                )
                return False
            if not changed:
                return False
            state.goals = updated
            saved = save_character_state(self._character_conf_uid, state)
            logger.info(
                "Goal transition: goal_id={} status={} persisted={} counts={}",
                goal_id,
                target,
                bool(saved),
                goal_status_counts(updated),
            )
            return bool(saved)
        except Exception as error:
            logger.debug("Goal transition failed: type={}", type(error).__name__)
            return False

    def record_goal_evidence(self, goal_id: str, evidence_id: str) -> bool:
        """Pin the evidence a goal already acted on (dedup anchor)."""
        try:
            state = getattr(self, "_character_state", None)
            if state is None:
                return False
            updated, changed = record_goal_evidence(
                list(getattr(state, "goals", None) or []), goal_id, evidence_id
            )
            if not changed:
                return False
            state.goals = updated
            return bool(save_character_state(self._character_conf_uid, state))
        except Exception as error:
            logger.debug(
                "Goal evidence anchor failed: type={}", type(error).__name__
            )
            return False

    def classify_goal_evidence(self, moment: Optional[datetime] = None):
        """Typed goal-evidence decision for the current state (pure read)."""
        try:
            from ...autonomous_decision import classify_goal_evidence as classify
            from ...episodic_memory import load_episodic_events
            from ... import world_state as _world_state

            state = getattr(self, "_character_state", None)
            if state is None:
                return None
            return classify(
                goals=getattr(state, "goals", None) or [],
                episodic_events=load_episodic_events(self._character_conf_uid) or [],
                moment=moment if moment is not None else _world_state.utcnow(),
            )
        except Exception as error:
            logger.debug(
                "Goal evidence classification skipped: type={}",
                type(error).__name__,
            )
            return None

    def add_character_memory(
        self, text: str, *, explicit: bool = True, kind: str = ""
    ) -> bool:
        """Persist one long-term fact shared across all chats."""
        if not self._character_conf_uid:
            logger.warning("Character memory update skipped: no active character")
            return False
        save_started = time.perf_counter()
        state = persist_character_memory(
            self._character_conf_uid, text, explicit=explicit, kind=kind
        )
        tracker = get_latency_tracker()
        if tracker:
            tracker.add_character_state_save(
                (time.perf_counter() - save_started) * 1000
            )
        if state is None:
            logger.warning(
                "Character memory update failed: character_memory_updated=False"
            )
            return False
        self._character_state = state
        logger.info(
            "Character memory stats: character_memory_updated=True, "
            "character_memory_count={}",
            len(state.memories),
        )
        return True

    def remove_character_memory(self, text: str) -> bool:
        """Forget stored facts overlapping the given text (character-level)."""
        if not self._character_conf_uid:
            return False
        state = remove_persisted_character_memory(self._character_conf_uid, text)
        if state is None:
            logger.warning(
                "Character memory removal failed: character_memory_updated=False"
            )
            return False
        self._character_state = state
        logger.info(
            "Character memory stats: character_memory_updated=True, "
            "character_memory_count={}",
            len(state.memories),
        )
        return True

    def list_interaction_preferences(self) -> List[Dict[str, Any]]:
        """Active interaction preferences (newest first, one per category)."""
        return active_preferences(
            getattr(self._character_state, "interaction_preferences", None)
        )

    def list_character_memories(self) -> List[Dict[str, Any]]:
        """Return stored long-term facts (for backend controls / future UI)."""
        return list(self._character_state.memories)

    async def ensure_conversation_title(self) -> str:
        """Generate and persist ONE title for the active conversation, or "".

        Auxiliary and fail-soft: runs after the user-visible response is
        already stored, uses the conversation's own LLM, and never raises.
        An existing title (manual or automatic) is never overwritten, and a
        failure simply keeps "Percakapan Baru".
        """
        try:
            conf_uid = self._character_conf_uid
            history_uid = self._history_uid
            if not conf_uid or not history_uid:
                return ""
            metadata = get_metadata(conf_uid, history_uid)
            messages = get_history(conf_uid, history_uid)
            from ...conversation_title import (
                generate_title,
                has_stored_title,
                should_generate_title,
            )

            if has_stored_title(metadata):
                return str(metadata.get("title", "")).strip()
            if not should_generate_title(metadata, messages):
                return ""
            chat_fn = getattr(getattr(self, "_llm", None), "chat_completion", None)
            title = await generate_title(chat_fn, messages)
            if not title:
                return ""
            # Re-check: a manual rename racing this turn wins, never overwritten.
            current = get_metadata(conf_uid, history_uid)
            if has_stored_title(current):
                return str(current.get("title", "")).strip()
            if not update_metadate(conf_uid, history_uid, {"title": title}):
                return ""
            logger.info("Conversation title generated (chars={})", len(title))
            return title
        except Exception as error:
            logger.debug("Conversation title skipped: type={}", type(error).__name__)
            return ""

    def reset_character_memory(self) -> bool:
        """Clear Mili's long-term memory; relationship is untouched."""
        if not self._character_conf_uid:
            return False
        state = reset_persisted_character_memory(self._character_conf_uid)
        if state is None:
            logger.warning(
                "Character memory reset failed: character_memory_updated=False"
            )
            return False
        self._character_state = state
        logger.info(
            "Character memory stats: character_memory_updated=True, "
            "character_memory_count=0, character_memory_reset=True"
        )
        return True

    def reset_character_state(self) -> bool:
        """Reset relationship to stranger and clear memory (no transcript touch)."""
        if not self._character_conf_uid:
            return False
        state = reset_persisted_character_state(self._character_conf_uid)
        if state is None:
            logger.warning(
                "Character state reset failed: character_memory_updated=False"
            )
            return False
        self._character_state = state
        self._relationship_state = RelationshipState(
            status=state.relationship_status,
            updated_at=state.relationship_updated_at,
            reason=state.relationship_reason,
        )
        logger.info(
            "Character state stats: relationship_status=stranger, "
            "character_memory_count=0, character_state_reset=True"
        )
        return True

    def _observe_character_memory_request(self, user_text: str) -> bool:
        """Honor explicit remember/forget requests with cheap local rules."""
        if not self._character_conf_uid:
            return False
        result = parse_memory_command(user_text)
        if result.action == "none" or not result.payload:
            return False
        if result.action == "forget":
            success = self.remove_character_memory(result.payload)
        else:
            success = self.add_character_memory(result.payload, explicit=True)
        logger.info(
            "Character memory command: memory_command={}, matched_trigger={}, "
            "success={}",
            result.action,
            result.matched_trigger,
            success,
        )
        return success

    def _decision_context(self) -> Optional[DecisionContextSignals]:
        """Compact contextual signals (ADL v2) from already-stored data.

        Interaction preferences are read through their own active-row helper
        and episodic memory through the existing deterministic retriever, both
        bounded. No LLM call, no new store, no new lifecycle: this runs inside
        the same per-turn decision-context build. Fail-soft — any problem
        returns None, which keeps the pre-v2 decision behaviour exactly.

        Semantic continuity (ADL v2): returned only when a VERIFIED Mili or
        shared-activity candidate exists. The episodic schema has no
        actor/participant field and extraction only ever sees the user turn,
        so nothing here may claim a recent user memory is Mili world state:
        ``continuity_event_ids`` stays empty and the signal is suppressed. A
        stored interaction preference likewise never makes the context look
        relevant, because it carries no semantic consumer in this layer yet.
        """
        try:
            from ...autonomous_decision import build_context_signals

            # Reuse this turn's single retrieval selection. The gate still
            # runs its own bounded recency pass internally; only the redundant
            # full relevance retrieval is skipped.
            events = load_episodic_events(self._character_conf_uid) or []
            prefetched = self._episodic_selection()
            signals = build_context_signals(
                interaction_preferences=(
                    getattr(self._character_state, "interaction_preferences", None)
                    or []
                ),
                episodic_events=events,
                query=self._episodic_query_text(),
                moment=utcnow(),
                continuity_event_ids=(),
                prefetched_relevant=prefetched,
            )
            if not signals.continuity_candidate:
                return None
            return signals
        except Exception as error:
            logger.debug(
                "Decision context signals unavailable: type={}",
                type(error).__name__,
            )
            return None

    def _decision_inputs(self) -> Optional[DecisionInputs]:
        """Read-only influence context for the autonomous decision stage.

        Built only from state already loaded for this character (relationship,
        non-completed goals, established activity preferences derived from
        long-term memories). Pure, no I/O beyond what is already in memory, no
        LLM, and fail-soft: any problem returns None, which keeps the previous
        decision behaviour unchanged. Goals are read only — never completed,
        reactivated, or rewritten here.
        """
        try:
            state = getattr(self, "_character_state", None)
            if state is None:
                return None
            goal_activities = tuple(
                DECISION_GOAL_ACTIVITY_HINTS[str(goal.get("id", ""))]
                for goal in (state.goals or [])
                if isinstance(goal, dict)
                and str(goal.get("status", "seed")) != "done"
                and str(goal.get("id", "")) in DECISION_GOAL_ACTIVITY_HINTS
            )
            preferences = derive_activity_preferences(
                memories=state.memories, tz=getattr(self, "_user_timezone", None)
            )
            preferred_activities = tuple(
                candidate.activity
                for candidate in preferences
                if getattr(candidate, "established", False)
            )
            context = self._decision_context()
            # Pre-v2 guard, deliberately NOT widened by context: goals and
            # established preferences alone decide this branch, so having a
            # stored interaction preference can never change the decision.
            if not goal_activities and not preferred_activities:
                relationship = str(state.relationship_status or "").strip().lower()
                if relationship in ("", "stranger"):
                    return DecisionInputs(mood_bias=True, context=context)
            return DecisionInputs(
                relationship_status=str(state.relationship_status or "stranger"),
                goal_activities=goal_activities,
                preferred_activities=preferred_activities,
                mood_bias=True,
                context=context,
            )
        except Exception as error:
            logger.debug(
                "Decision inputs unavailable (defaults kept): type={}",
                type(error).__name__,
            )
            return None

    def _episodic_query_text(self) -> str:
        """This turn's clean retrieval query (G1).

        Prefers ``metadata["episodic_query"]`` set by ``single_conversation``
        from the raw user turn, so the LLM-input-only search block can never
        pollute retrieval. Falls back to the LAST user message of the
        transcript when metadata is unavailable — never the whole ``_memory``.
        """
        query = (getattr(self, "_episodic_query", "") or "").strip()
        if query:
            return query
        for message in reversed(self._memory or []):
            if (
                message.get("role") == "user"
                and str(message.get("content", "")).strip()
            ):
                return str(message.get("content", ""))
        return ""

    def _episodic_selection(self) -> List[Dict[str, Any]]:
        """ONE deterministic retrieval per turn, shared by every consumer.

        The prompt block and the Autonomous Decision Layer both need episodic
        recall; running retrieval twice per turn was duplicated I/O and
        duplicated scoring over the same store. The selection is cached per
        query for the turn, so both consumers see identical events.

        Deterministic, local and LLM-free. Emits G3 telemetry: counts and
        timing only — never event text, user text, or any payload.
        """
        query = self._episodic_query_text()
        if not self._character_conf_uid or not query:
            return []
        cached = getattr(self, "_episodic_selection_cache", None)
        if isinstance(cached, tuple) and cached[0] == query:
            return list(cached[1])
        started = time.perf_counter()
        events = load_episodic_events(self._character_conf_uid) or []
        if not events:
            logger.debug("Episodic retrieval: no stored events.")
            self._episodic_selection_cache = (query, [])
            return []
        selected = retrieve_episodic_events(events, query) or []
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if selected:
            # G3 telemetry: considered / selected / elapsed only.
            logger.info(
                "Episodic retrieval: events={} selected={} elapsed_ms={}",
                len(events),
                len(selected),
                elapsed_ms,
            )
        else:
            logger.debug(
                "Episodic retrieval: no relevant event for query_chars={} events={}.",
                len(query),
                len(events),
            )
        self._episodic_selection_cache = (query, list(selected))
        return list(selected)

    def _episodic_context_for_prompt(self) -> str:
        """Render relevant episodic experiences for the current turn.

        Fail-soft: any problem yields "" and the turn is unaffected.
        Never writes, never calls a model.
        """
        if not self._character_conf_uid:
            return ""
        selected = self._episodic_selection()
        if not selected:
            return ""
        return render_episodic_context(
            selected, tz=getattr(self, "_user_timezone", None)
        )

    async def capture_episodic_event(self, user_text: str, history_uid: str) -> None:
        """Fire-and-forget episodic capture for one completed turn.

        Runs after the user-visible response; gate + single LLM call +
        validated store. Never raises; never blocks the conversation.

        Latency is measured inside this task (``perf_counter`` from task
        start to task end) rather than on the request tracker, because the
        turn tracker is already reset and serialized by the time this task
        runs. The task-owned number is logged on its own line and is written
        to the tracker on a best-effort basis.
        """
        started = time.perf_counter()
        outcome = "skipped"
        try:
            if not self._character_conf_uid or not history_uid:
                return
            if not is_episodic_candidate(user_text):
                return
            outcome = "rejected"
            llm = getattr(self, "_llm", None)
            chat_fn = getattr(llm, "chat_completion", None)
            if not callable(chat_fn):
                return
            stored = await extract_and_store_episodic(
                chat_fn,
                self._character_conf_uid,
                user_text,
                history_uid,
                utcnow(),
                getattr(self, "_user_timezone", None),
            )
            outcome = "stored" if stored is not None else "rejected"
        except Exception as error:
            outcome = "error"
            logger.warning(
                "Episodic capture failed (turn unaffected): type={}",
                type(error).__name__,
            )
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            try:
                tracker = get_latency_tracker()
                if tracker:
                    tracker.add_time("episodic_ms", elapsed_ms)
            except Exception as error:
                logger.debug(
                    "Episodic latency tracker write skipped: type={}",
                    type(error).__name__,
                )
            logger.info(
                "Episodic capture latency: outcome={} elapsed_ms={}",
                outcome,
                round(elapsed_ms, 2),
            )

    def observe_character_events(
        self,
        user_text: str,
        assistant_text: str,
    ) -> bool:
        """Observe one completed visible turn: relationship, memory, preference."""
        relationship_updated = self.observe_relationship_event(
            user_text, assistant_text
        )
        memory_updated = self._observe_character_memory_request(user_text)
        auto_updated = self._observe_automatic_memory(user_text)
        preference_updated = self._observe_interaction_preference(user_text)
        return (
            relationship_updated
            or memory_updated
            or auto_updated
            or preference_updated
        )

    def _observe_automatic_memory(self, user_text: str) -> bool:
        """Persist stable facts stated in ordinary chat, with no command word.

        The user should never have to say "ingat this" for something that is
        plainly a lasting fact about them ("aku suka kopi susu"). This runs on
        the existing post-turn observer, uses the deterministic extractor in
        ``character_memory_commands`` (no model call, no new store, no new
        lifecycle) and writes through the same ``add_character_memory`` path as
        the explicit command, so deduplication, bounds and the 600-token prompt
        budget are shared.

        Transient statements, questions and reactions are rejected by the
        extractor -- those belong to episodic memory, which already captures
        events on its own. Stored as ``explicit=False`` so an inferred trait is
        never presented as something the user asked Mili to remember, and a
        later explicit "ingat ..." still wins.

        Fail-soft: a failure here can never break the conversation.
        """
        if not self._character_conf_uid:
            return False
        try:
            from ...character_memory_commands import extract_stable_facts

            stored_any = False
            for fact in extract_stable_facts(user_text):
                if self.add_character_memory(fact, explicit=False):
                    stored_any = True
            if not stored_any:
                return False
            # Refresh the in-memory copy so the fact also applies for the rest
            # of THIS session, not only from the next one.
            try:
                refreshed = load_character_state(self._character_conf_uid)
                if refreshed is not None:
                    self._character_state = refreshed
            except Exception as error:
                logger.debug(
                    "Character state refresh after auto-memory skipped: type={}",
                    type(error).__name__,
                )
            return True
        except Exception as error:
            logger.debug(
                "Automatic memory capture skipped: type={}", type(error).__name__
            )
            return False

    def _observe_interaction_preference(self, user_text: str) -> bool:
        """Capture a durable interaction preference stated in ordinary chat.

        Runs on the existing post-turn observer, so it adds no LLM call and
        no new lifecycle. Deterministic local detection only; fail-soft.
        """
        if not self._character_conf_uid:
            return False
        stored = record_interaction_preference(self._character_conf_uid, user_text)
        if stored is None:
            return False
        # Refresh the in-memory copy so the preference also applies for the
        # rest of THIS session, not only from the next one.
        try:
            refreshed = load_character_state(self._character_conf_uid)
            if refreshed is not None:
                self._character_state = refreshed
        except Exception as error:
            logger.debug(
                "Character state refresh after preference skipped: type={}",
                type(error).__name__,
            )
        tracker = get_latency_tracker()
        if tracker:
            tracker.add_context(0.0, None)
        return True

    def observe_reactive_state(self, emotion_keys: List[str]) -> bool:
        """Apply one deterministic reactive transition (no LLM calls).

        ``emotion_keys`` are backend semantic-emotion labels observed on the
        just-completed visible turn. The state is first lazily reconciled
        (time), then the reactive step runs on top; a single persist covers
        both. Fail-soft: never breaks the conversation path.
        """
        if not self._character_conf_uid:
            return False
        try:
            moment = utcnow()
            raw = load_world_state(self._character_conf_uid)
            reconciled, _ = reconcile(
                raw, moment, getattr(self, "_user_timezone", None)
            )
            updated, changed = apply_reactive(
                reconciled,
                emotion_keys or [],
                moment,
                tz=getattr(self, "_user_timezone", None),
            )
            if not changed:
                return False
            if not save_world_state(self._character_conf_uid, updated):
                return False
            logger.info(
                "Reactive state stats: mood={}, activity={}, energy={}, "
                "reactive_updated=True",
                updated.mood,
                updated.activity,
                updated.energy,
            )
            return True
        except Exception as error:
            logger.warning("Reactive state skipped: type={}", type(error).__name__)
            return False

    async def compact_conversation(self) -> tuple[bool, Optional[str]]:
        """Manually compress the active conversation using the rolling-summary pipeline.

        The transcript is never modified: only the rolling summary and its
        boundary advance, so later automatic summaries continue incrementally.
        On failure the previous summary and boundary are preserved.
        """
        compact_started = time.perf_counter()
        if not self._summary_conf_uid or not self._summary_history_uid:
            return False, "No active conversation to compact."
        async with self._summary_lock:
            start = self._summary_state.summarized_through
            candidates = self._memory[start:]
            if not candidates:
                return True, None
            if len(candidates) < self._summary_min_new_messages:
                return False, "Not enough new messages to compact yet."
            try:
                tracker = get_latency_tracker()
                if tracker:
                    tracker.mark("summary_start")
                phase_token = set_latency_phase("summary")
                try:
                    updated_summary = await self._summarizer.summarize(
                        self._summary_state.text,
                        candidates,
                    )
                finally:
                    reset_latency_phase(phase_token)
                if tracker:
                    tracker.mark("summary_end")
            except Exception as error:
                logger.warning(
                    "Manual compact failed; keeping prior summary: type={}",
                    type(error).__name__,
                )
                return False, (
                    "Summary generation failed; transcript and previous "
                    "summary are untouched."
                )
            updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            tracker = get_latency_tracker()
            save_started = time.perf_counter()
            persisted = update_summary_metadata(
                self._summary_conf_uid,
                self._summary_history_uid,
                expected_summarized_through=start,
                conversation_summary=updated_summary,
                summarized_through=len(self._memory),
                summary_updated_at=updated_at,
            )
            if tracker:
                tracker.add_metadata_save((time.perf_counter() - save_started) * 1000)
            if not persisted:
                self._load_summary_state(
                    self._summary_conf_uid,
                    self._summary_history_uid,
                )
                return False, "Could not persist the compacted summary."
            self._summary_state = SummaryState(
                text=updated_summary,
                summarized_through=len(self._memory),
                updated_at=updated_at,
            )
            logger.info(
                "Manual compact stats: summary_updated=True, "
                "summarized_through={}, messages_compacted={}",
                self._summary_state.summarized_through,
                len(candidates),
            )
            logger.info(
                "[COMPACT LATENCY] total_ms={} messages_compacted={}",
                round((time.perf_counter() - compact_started) * 1000, 2),
                len(candidates),
            )
            return True, None

    def observe_relationship_event(
        self,
        user_text: str,
        assistant_text: str,
    ) -> bool:
        """Evaluate one completed visible turn without another LLM request."""
        update = detect_relationship_update(
            self._relationship_state.status,
            user_text,
            assistant_text,
        )
        if update is None:
            logger.info(
                "Relationship stats: relationship_status={}, "
                "relationship_updated=False, relationship_update_trigger=skipped",
                self._relationship_state.status,
            )
            return False
        return self.set_relationship_status(
            update.new_status,
            trigger=update.trigger,
        )

    async def _maybe_update_summary(
        self,
        *,
        messages: List[Dict[str, Any]],
        protected_start: int,
        initial_selection: ContextSelection,
    ) -> tuple[bool, bool, int, int]:
        """Return update/failure/candidate count/current eviction boundary."""
        if (
            not self._rolling_summary_enabled
            or not self._context_management_enabled
            or not initial_selection.stats.trimmed
            or not self._summary_conf_uid
            or not self._summary_history_uid
        ):
            return False, False, 0, 0

        protected_count = len(messages) - protected_start
        selected_history_count = len(initial_selection.messages) - protected_count
        evicted_through = max(0, protected_start - selected_history_count)

        async with self._summary_lock:
            start = self._summary_state.summarized_through
            if evicted_through <= start:
                return False, False, 0, evicted_through
            candidates = messages[start:evicted_through]
            if len(candidates) < self._summary_min_new_messages:
                return False, False, 0, evicted_through

            try:
                tracker = get_latency_tracker()
                if tracker:
                    tracker.mark("summary_start")
                phase_token = set_latency_phase("summary")
                try:
                    updated_summary = await self._summarizer.summarize(
                        self._summary_state.text,
                        candidates,
                    )
                finally:
                    reset_latency_phase(phase_token)
                if tracker:
                    tracker.mark("summary_end")
                updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                save_started = time.perf_counter()
                persisted = update_summary_metadata(
                    self._summary_conf_uid,
                    self._summary_history_uid,
                    expected_summarized_through=start,
                    conversation_summary=updated_summary,
                    summarized_through=evicted_through,
                    summary_updated_at=updated_at,
                )
                if tracker:
                    tracker.add_metadata_save(
                        (time.perf_counter() - save_started) * 1000
                    )
                if not persisted:
                    self._load_summary_state(
                        self._summary_conf_uid,
                        self._summary_history_uid,
                    )
                    return False, False, 0, evicted_through
                self._summary_state = SummaryState(
                    text=updated_summary,
                    summarized_through=evicted_through,
                    updated_at=updated_at,
                )
                return True, False, len(candidates), evicted_through
            except Exception as error:
                logger.warning(
                    "Rolling summary update failed; keeping prior summary: type={}",
                    type(error).__name__,
                )
                return False, True, len(candidates), evicted_through

    async def _prepare_context_with_summary(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        protected_start: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Apply Stage 3 budgeting, update summary if needed, then inject it."""
        context_started = time.perf_counter()
        tracker = get_latency_tracker()
        if tracker:
            tracker.mark("context_build_start")
        summary_before_ms = tracker.summary_ms if tracker else 0.0
        if not self._context_management_enabled:
            result = self._prepare_context(
                messages,
                system_prompt,
                tools=tools,
                protected_start=protected_start,
            )
            if tracker:
                tracker.add_context((time.perf_counter() - context_started) * 1000)
            return result
        if protected_start is None:
            protected_start = max(0, len(messages) - 1)
        initial_selection = self._select_context(
            messages,
            system_prompt,
            tools=tools,
            protected_start=protected_start,
        )
        updated, failed, new_count, evicted_through = await self._maybe_update_summary(
            messages=messages,
            protected_start=protected_start,
            initial_selection=initial_selection,
        )

        summary_present = bool(self._summary_state.text.strip())
        if summary_present:
            # Stage 3 already chose not to send messages before evicted_through.
            # Keep that same boundary even when a summary refresh fails, while
            # retaining the unsummarized transcript for a later retry.
            through = min(
                max(self._summary_state.summarized_through, evicted_through),
                protected_start,
            )
            summary_role = (
                "assistant" if isinstance(self._llm, ClaudeAsyncLLM) else "system"
            )
            request_source = [
                build_summary_message(
                    self._summary_state.text,
                    role=summary_role,
                    age_tag=memory_age_label(
                        self._summary_state.updated_at, tz=self._user_timezone
                    ),
                ),
                *messages[through:],
            ]
            final_protected_start = 1 + protected_start - through
            final_selection = self._select_context(
                request_source,
                system_prompt,
                tools=tools,
                protected_start=final_protected_start,
            )
        else:
            final_selection = initial_selection

        self._log_context_stats(final_selection)
        summary_included = (
            summary_present
            and bool(final_selection.messages)
            and (
                final_selection.messages[0]
                .get("content", "")
                .startswith("Conversation context from earlier messages")
            )
        )
        logger.info(
            "Summary stats: summary_present={}, summary_included={}, "
            "summary_tokens={}, summarized_through={}, new_messages_summarized={}, "
            "summary_updated={}, summary_generation_failed={}",
            summary_present,
            summary_included,
            estimate_tokens(self._summary_state.text) if summary_present else 0,
            self._summary_state.summarized_through,
            new_count,
            updated,
            failed,
        )
        if tracker:
            tracker.mark("context_build_end")
            elapsed_ms = (time.perf_counter() - context_started) * 1000
            summary_delta = max(0.0, tracker.summary_ms - summary_before_ms)
            tracker.add_context(max(0.0, elapsed_ms - summary_delta), final_selection)
        return final_selection.messages

    @staticmethod
    def _context_error_message(error: ContextBudgetExceeded) -> str:
        logger.warning("Context request rejected before provider call: {}", error)
        return (
            "Pesan ini terlalu panjang untuk context window model. "
            "Pendekkan pesannya atau sesuaikan context_window_override."
        )

    def _add_message(
        self,
        message: Union[str, List[Dict[str, Any]]],
        role: str,
        display_text: DisplayText | None = None,
        skip_memory: bool = False,
    ):
        """Add message to memory."""
        if skip_memory:
            return

        text_content = ""
        if isinstance(message, list):
            for item in message:
                if item.get("type") == "text":
                    text_content += item["text"] + " "
            text_content = text_content.strip()
        elif isinstance(message, str):
            text_content = message
        else:
            logger.warning(
                f"_add_message received unexpected message type: {type(message)}"
            )
            text_content = str(message)

        if not text_content and role == "assistant":
            return

        message_data = {
            "role": role,
            "content": text_content,
        }

        if display_text:
            if display_text.name:
                message_data["name"] = display_text.name
            if display_text.avatar:
                message_data["avatar"] = display_text.avatar

        if (
            self._memory
            and self._memory[-1]["role"] == role
            and self._memory[-1]["content"] == text_content
        ):
            return

        self._memory.append(message_data)

    def set_memory_from_history(
        self, conf_uid: str, history_uid: str, user_timezone: str | None = None
    ) -> None:
        """Load memory from chat history."""
        self._user_timezone = user_timezone
        self._history_uid = history_uid
        messages = get_history(conf_uid, history_uid)
        self._prev_session_at = self._latest_other_session_at(conf_uid, history_uid)
        self._prev_session_summaries = self._load_previous_session_summaries(
            conf_uid, history_uid
        )

        self._memory = []
        for msg in messages:
            role = "user" if msg["role"] == "human" else "assistant"
            content = msg["content"]
            if isinstance(content, str) and content:
                self._memory.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )
            else:
                logger.warning(
                    "Skipping invalid message from history (content omitted)"
                )
        self._load_summary_state(conf_uid, history_uid)
        self._load_character_state(conf_uid)
        logger.info(f"Loaded {len(self._memory)} messages from history.")

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle user interruption."""
        if self._interrupt_handled:
            return

        self._interrupt_handled = True

        if self._memory and self._memory[-1]["role"] == "assistant":
            if not self._memory[-1]["content"].endswith("..."):
                self._memory[-1]["content"] = heard_response + "..."
            else:
                self._memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self._memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )

        interrupt_role = "system" if self.interrupt_method == "system" else "user"
        self._memory.append(
            {
                "role": interrupt_role,
                "content": "[Interrupted by user]",
            }
        )
        logger.info(f"Handled interrupt with role '{interrupt_role}'.")

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """Format input data to text prompt."""
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(
                    f"[User shared content from clipboard: {text_data.content}]"
                )

        if input_data.images:
            message_parts.append("\n[User has also provided images]")

        return "\n".join(message_parts).strip()

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """Prepare messages for LLM API call."""
        # Episodic retrieval query: the clean user turn from metadata, so an
        # appended search block (LLM input only, never persisted) cannot
        # pollute retrieval. Reset every turn to avoid stale reuse.
        self._episodic_query = str(
            (input_data.metadata or {}).get("episodic_query", "") or ""
        ).strip()
        # New turn: drop the previous selection so a stale recall can never
        # leak into this turn's prompt or decision context.
        self._episodic_selection_cache = None
        messages = self._memory.copy()
        user_content = []
        text_prompt = self._to_text_prompt(input_data)
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})

        if input_data.images:
            image_added = False
            for img_data in input_data.images:
                if isinstance(img_data.data, str) and img_data.data.startswith(
                    "data:image"
                ):
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img_data.data, "detail": "auto"},
                        }
                    )
                    image_added = True
                else:
                    logger.error(
                        f"Invalid image data format: {type(img_data.data)}. Skipping image."
                    )

            if not image_added and not text_prompt:
                logger.warning(
                    "User input contains images but none could be processed."
                )

        if user_content:
            user_message = {"role": "user", "content": user_content}
            messages.append(user_message)

            skip_memory = False
            if input_data.metadata and input_data.metadata.get("skip_memory", False):
                skip_memory = True

            if not skip_memory:
                self._add_message(
                    text_prompt if text_prompt else "[User provided image(s)]", "user"
                )
        else:
            logger.warning("No content generated for user message.")

        return messages

    async def _claude_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle Claude interaction loop with tool support."""
        messages = initial_messages.copy()
        protected_start = max(0, len(initial_messages) - 1)
        current_turn_text = ""
        pending_tool_calls = []
        current_assistant_message_content = []
        # Cost guard: counts provider round-trips caused by tool use.
        tool_iterations = 0

        while True:
            if tool_iterations >= TOOL_LOOP_MAX_ITERATIONS:
                logger.warning(
                    "Tool loop cap reached: stopping further provider "
                    "round-trips at max_tool_iterations={}",
                    TOOL_LOOP_MAX_ITERATIONS,
                )
                break
            current_system_prompt = self._relationship_system_prompt(self._system)
            try:
                request_messages = await self._prepare_context_with_summary(
                    messages,
                    current_system_prompt,
                    tools=tools,
                    protected_start=protected_start,
                )
            except ContextBudgetExceeded as error:
                yield self._context_error_message(error)
                return
            stream = self._llm.chat_completion(
                request_messages, current_system_prompt, tools=tools
            )
            tool_iterations += 1
            pending_tool_calls.clear()
            current_assistant_message_content.clear()

            async for event in stream:
                if event["type"] == "text_delta":
                    text = event["text"]
                    current_turn_text += text
                    yield text
                    if (
                        not current_assistant_message_content
                        or current_assistant_message_content[-1]["type"] != "text"
                    ):
                        current_assistant_message_content.append(
                            {"type": "text", "text": text}
                        )
                    else:
                        current_assistant_message_content[-1]["text"] += text
                elif event["type"] == "tool_use_complete":
                    tool_call_data = event["data"]
                    logger.info(
                        f"Tool request: {tool_call_data['name']} (ID: {tool_call_data['id']})"
                    )
                    pending_tool_calls.append(tool_call_data)
                    current_assistant_message_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call_data["id"],
                            "name": tool_call_data["name"],
                            "input": tool_call_data["input"],
                        }
                    )
                # elif event["type"] == "message_delta":
                #     if event["data"]["delta"].get("stop_reason"):
                #         stop_reason = event["data"]["delta"].get("stop_reason")
                elif event["type"] == "message_stop":
                    break
                elif event["type"] == "error":
                    logger.error(
                        "LLM API error (message_chars={})",
                        len(str(event.get("message", ""))),
                    )
                    yield f"[Error from LLM: {event['message']}]"
                    return

            if pending_tool_calls:
                filtered_assistant_content = [
                    block
                    for block in current_assistant_message_content
                    if not (
                        block.get("type") == "text"
                        and not block.get("text", "").strip()
                    )
                ]

                if filtered_assistant_content:
                    messages.append(
                        {"role": "assistant", "content": filtered_assistant_content}
                    )
                    assistant_text_for_memory = "".join(
                        [
                            c["text"]
                            for c in filtered_assistant_content
                            if c["type"] == "text"
                        ]
                    ).strip()
                    if assistant_text_for_memory:
                        self._add_message(assistant_text_for_memory, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "Claude Tool interaction requested but ToolExecutor is not available."
                    )
                    yield "[Error: ToolExecutor not configured]"
                    return

                tracker = get_latency_tracker()
                if tracker:
                    tracker.mark("tool_start")
                tool_started = time.perf_counter()
                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="Claude",
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "Tool executor finished without final results marker."
                    )
                tracker = get_latency_tracker()
                if tracker:
                    tracker.mark("tool_end")
                    tracker.add_tool((time.perf_counter() - tool_started) * 1000)

                if tool_results_for_llm:
                    messages.append({"role": "user", "content": tool_results_for_llm})

                # stop_reason = None
                continue
            else:
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                return

    async def _openai_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle OpenAI interaction with tool support."""
        messages = initial_messages.copy()
        protected_start = max(0, len(initial_messages) - 1)
        current_turn_text = ""
        pending_tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]] = []
        current_system_prompt = self._system
        # Cost guard: counts provider round-trips caused by tool use.
        tool_iterations = 0

        while True:
            if self.prompt_mode_flag:
                if self._mcp_prompt_string:
                    base_system_prompt = f"{self._system}\n\n{self._mcp_prompt_string}"
                else:
                    logger.warning("Prompt mode active but mcp_prompt_string is empty!")
                    base_system_prompt = self._system
                tools_for_api = None
            else:
                base_system_prompt = self._system
                tools_for_api = tools
            current_system_prompt = self._relationship_system_prompt(base_system_prompt)

            try:
                request_messages = await self._prepare_context_with_summary(
                    messages,
                    current_system_prompt,
                    tools=tools_for_api,
                    protected_start=protected_start,
                )
            except ContextBudgetExceeded as error:
                yield self._context_error_message(error)
                return
            if tool_iterations >= TOOL_LOOP_MAX_ITERATIONS:
                logger.warning(
                    "Tool loop cap reached: stopping further provider "
                    "round-trips at max_tool_iterations={}",
                    TOOL_LOOP_MAX_ITERATIONS,
                )
                break
            stream = self._llm.chat_completion(
                request_messages, current_system_prompt, tools=tools_for_api
            )
            tool_iterations += 1
            pending_tool_calls.clear()
            current_turn_text = ""
            assistant_message_for_api = None
            detected_prompt_json = None
            goto_next_while_iteration = False

            async for event in stream:
                if self.prompt_mode_flag:
                    if isinstance(event, str):
                        current_turn_text += event
                        if self._json_detector:
                            potential_json = self._json_detector.process_chunk(event)
                            if potential_json:
                                try:
                                    if isinstance(potential_json, list):
                                        detected_prompt_json = potential_json
                                    elif isinstance(potential_json, dict):
                                        detected_prompt_json = [potential_json]

                                    if detected_prompt_json:
                                        break
                                except Exception as e:
                                    logger.error(f"Error parsing detected JSON: {e}")
                                    if self._json_detector:
                                        self._json_detector.reset()
                                    yield f"[Error parsing tool JSON: {e}]"
                                    goto_next_while_iteration = True
                                    break
                        yield event
                else:
                    if isinstance(event, str):
                        current_turn_text += event
                        yield event
                    elif isinstance(event, list) and all(
                        isinstance(tc, ToolCallObject) for tc in event
                    ):
                        pending_tool_calls = event
                        assistant_message_for_api = {
                            "role": "assistant",
                            "content": current_turn_text if current_turn_text else None,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in pending_tool_calls
                            ],
                        }
                        break
                    elif event == "__API_NOT_SUPPORT_TOOLS__":
                        logger.warning(
                            f"LLM {getattr(self._llm, 'model', '')} has no native tool support. Switching to prompt mode."
                        )
                        self.prompt_mode_flag = True
                        if self._tool_manager:
                            self._tool_manager.disable()
                        if self._json_detector:
                            self._json_detector.reset()
                        goto_next_while_iteration = True
                        break
            if goto_next_while_iteration:
                continue

            if detected_prompt_json:
                logger.info("Processing tools detected via prompt mode JSON.")
                self._add_message(current_turn_text, "assistant")

                parsed_tools = self._tool_executor.process_tool_from_prompt_json(
                    detected_prompt_json
                )
                if parsed_tools:
                    tool_results_for_llm = []
                    if not self._tool_executor:
                        logger.error(
                            "Prompt Tool interaction requested but ToolExecutor/MCPClient is not available."
                        )
                        yield "[Error: ToolExecutor/MCPClient not configured for prompt mode]"
                        continue

                    tool_started = time.perf_counter()
                    tool_executor_iterator = self._tool_executor.execute_tools(
                        tool_calls=parsed_tools,
                        caller_mode="Prompt",
                    )
                    try:
                        while True:
                            update = await anext(tool_executor_iterator)
                            if update.get("type") == "final_tool_results":
                                tool_results_for_llm = update.get("results", [])
                                break
                            else:
                                yield update
                    except StopAsyncIteration:
                        logger.warning(
                            "Prompt mode tool executor finished without final results marker."
                        )
                    tracker = get_latency_tracker()
                    if tracker:
                        tracker.add_tool((time.perf_counter() - tool_started) * 1000)

                    if tool_results_for_llm:
                        result_strings = [
                            res.get("content", "Error: Malformed result")
                            for res in tool_results_for_llm
                        ]
                        combined_results_str = "\n".join(result_strings)
                        messages.append(
                            {"role": "user", "content": combined_results_str}
                        )
                continue

            elif pending_tool_calls and assistant_message_for_api:
                messages.append(assistant_message_for_api)
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "OpenAI Tool interaction requested but ToolExecutor/MCPClient is not available."
                    )
                    yield "[Error: ToolExecutor/MCPClient not configured for OpenAI mode]"
                    continue

                tracker = get_latency_tracker()
                if tracker:
                    tracker.mark("tool_start")
                tool_started = time.perf_counter()
                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="OpenAI",
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "OpenAI tool executor finished without final results marker."
                    )
                tracker = get_latency_tracker()
                if tracker:
                    tracker.mark("tool_end")
                    tracker.add_tool((time.perf_counter() - tool_started) * 1000)

                if tool_results_for_llm:
                    messages.extend(tool_results_for_llm)
                continue

            else:
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                return

    def _chat_function_factory(
        self,
    ) -> Callable[[BatchInput], AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]]:
        """Create the chat pipeline function."""

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_memory(
            input_data: BatchInput,
        ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
            """Process chat with memory and tools."""
            self.reset_interrupt()
            self.prompt_mode_flag = False

            messages = self._to_messages(input_data)
            protected_start = max(0, len(messages) - 1)
            tools = None
            tool_mode = None
            llm_supports_native_tools = False

            if self._use_mcpp and self._tool_manager:
                tools = None
                if isinstance(self._llm, ClaudeAsyncLLM):
                    tool_mode = "Claude"
                    tools = self._formatted_tools_claude
                    llm_supports_native_tools = True
                elif isinstance(self._llm, OpenAICompatibleAsyncLLM):
                    tool_mode = "OpenAI"
                    tools = self._formatted_tools_openai
                    llm_supports_native_tools = True
                else:
                    logger.warning(
                        f"LLM type {type(self._llm)} not explicitly handled for tool mode determination."
                    )

                if llm_supports_native_tools and not tools:
                    logger.warning(
                        f"No tools available/formatted for '{tool_mode}' mode, despite MCP being enabled."
                    )

            if self._use_mcpp and tool_mode == "Claude":
                logger.debug(
                    f"Starting Claude tool interaction loop with {len(tools)} tools."
                )
                async for output in self._claude_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            elif self._use_mcpp and tool_mode == "OpenAI":
                logger.debug(
                    f"Starting OpenAI tool interaction loop with {len(tools)} tools."
                )
                async for output in self._openai_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            else:
                logger.info("Starting simple chat completion.")
                current_system_prompt = self._relationship_system_prompt(self._system)
                try:
                    request_messages = await self._prepare_context_with_summary(
                        messages,
                        current_system_prompt,
                        protected_start=protected_start,
                    )
                except ContextBudgetExceeded as error:
                    yield self._context_error_message(error)
                    return
                token_stream = self._llm.chat_completion(
                    request_messages,
                    current_system_prompt,
                )
                complete_response = ""
                async for event in token_stream:
                    text_chunk = ""
                    if isinstance(event, dict) and event.get("type") == "text_delta":
                        text_chunk = event.get("text", "")
                    elif isinstance(event, str):
                        text_chunk = event
                    else:
                        continue
                    if text_chunk:
                        yield text_chunk
                        complete_response += text_chunk
                if complete_response:
                    self._add_message(complete_response, "assistant")

        return chat_with_memory

    def _proactive_chat_function_factory(
        self,
        followup_context: Optional[
            Union[Dict[str, Any], ProactiveFollowupContext]
        ] = None,
        intent_context: Optional[Union[Dict[str, Any], ProactiveIntentContext]] = None,
    ) -> Callable[[], AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]]:
        """Create an assistant-only turn using the normal character context.

        The generation cue is appended to the effective system prompt only for
        this request.  It never enters ``_memory`` or the persisted transcript.
        The generated assistant message does enter ``_memory`` normally.
        ``followup_context`` carries the deterministic ignored-proactive state
        (see ``proactive_chat.ProactiveFollowupContext``); it only shapes the
        prompt and never triggers an extra model call.
        """

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def proactive_with_memory() -> AsyncIterator[Union[str, Dict[str, Any]]]:
            self.reset_interrupt()
            self.prompt_mode_flag = False

            messages = self._memory.copy()
            try:
                prompt_name = self._tool_prompts.get(
                    "proactive_speak_prompt", "proactive_speak_prompt"
                )
                proactive_instruction = prompt_loader.load_util(prompt_name).strip()
            except Exception as error:
                logger.warning(
                    "Proactive prompt unavailable; using safe fallback: type={}",
                    type(error).__name__,
                )
                proactive_instruction = (
                    "Initiate one natural, context-aware message as the character. "
                    "Do not mention timers or system behavior."
                )

            current_system_prompt = "\n\n".join(
                [
                    self._relationship_system_prompt(self._system),
                    "Internal instruction for this turn only:\n"
                    + proactive_instruction,
                ]
            )
            if isinstance(intent_context, dict):
                parsed_intent = ProactiveIntentContext.from_dict(intent_context)
            else:
                parsed_intent = intent_context
            if isinstance(followup_context, dict):
                parsed_followup = ProactiveFollowupContext.from_dict(followup_context)
            else:
                parsed_followup = followup_context
            # In semantic-auto mode, an ignored statement does not make
            # silence the topic.  A genuinely unanswered proactive question
            # remains deterministic and keeps the existing escalation block.
            semantic_ignored_statement = (
                parsed_intent is not None
                and parsed_intent.strategy == ProactiveTurnStrategy.SEMANTIC_AUTO
                and parsed_followup is not None
                and not parsed_followup.previous_proactive_expected_response
            )
            followup_block = (
                None
                if semantic_ignored_statement
                else format_followup_instruction(parsed_followup)
            )
            if followup_block:
                current_system_prompt = "\n\n".join(
                    [current_system_prompt, followup_block]
                )
            # When the ignored-question follow-up block is present it already
            # carries the turn's instructions; emit only compact intent lines.
            intent_block = format_intent_instruction(
                parsed_intent, include_guidance=followup_block is None
            )
            if intent_block:
                current_system_prompt = "\n\n".join(
                    [current_system_prompt, intent_block]
                )
            try:
                request_messages = await self._prepare_context_with_summary(
                    messages,
                    current_system_prompt,
                    protected_start=len(messages),
                )
            except ContextBudgetExceeded as error:
                logger.warning(
                    "Proactive generation skipped because context does not fit: {}",
                    error,
                )
                return

            # Ephemeral proactive turn cue: proactive generation is a
            # system-initiated turn, so there is never a *current* user turn in
            # the transcript (all of it is history). Some OpenAI-compatible
            # providers complete with an empty stream when the request ends on
            # an assistant message and no current user cue exists. Inject ONE
            # request-only internal user turn so the provider produces the new
            # assistant message. It lives only in this local request list: it
            # never enters _memory, history, summary, memory parsing,
            # relationship logic, or the UI.
            request_messages = [
                *request_messages,
                {"role": "user", "content": PROACTIVE_TURN_CUE},
            ]
            logger.info(
                "Proactive generation: proactive_turn_cue=True, "
                "request_message_count_after={}",
                len(request_messages),
            )
            tracker = get_latency_tracker()
            if tracker:
                tracker.message_count = len(request_messages)

            token_stream = self._llm.chat_completion(
                request_messages,
                current_system_prompt,
            )
            complete_response = ""
            async for event in token_stream:
                text_chunk = ""
                if isinstance(event, dict) and event.get("type") == "text_delta":
                    text_chunk = event.get("text", "")
                elif isinstance(event, str):
                    text_chunk = event
                if text_chunk:
                    yield text_chunk
                    complete_response += text_chunk
            if complete_response:
                self._add_message(complete_response, "assistant")

        return proactive_with_memory

    async def chat_proactively(
        self,
        followup_context: Optional[
            Union[Dict[str, Any], ProactiveFollowupContext]
        ] = None,
        intent_context: Optional[Union[Dict[str, Any], ProactiveIntentContext]] = None,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Generate one proactive assistant message without a fake user turn."""
        proactive_chat = self._proactive_chat_function_factory(
            followup_context, intent_context
        )
        async for output in proactive_chat():
            yield output

    async def chat(
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Run chat pipeline."""
        chat_func_decorated = self._chat_function_factory()
        async for output in chat_func_decorated(input_data):
            yield output

    def reset_interrupt(self) -> None:
        """Reset interrupt flag."""
        self._interrupt_handled = False

    def start_group_conversation(
        self, human_name: str, ai_participants: List[str]
    ) -> None:
        """Start a group conversation."""
        if not self._tool_prompts:
            logger.warning("Tool prompts dictionary is not set.")
            return

        other_ais = ", ".join(name for name in ai_participants)
        prompt_name = self._tool_prompts.get("group_conversation_prompt", "")

        if not prompt_name:
            logger.warning("No group conversation prompt name found.")
            return

        try:
            group_context = prompt_loader.load_util(prompt_name).format(
                human_name=human_name, other_ais=other_ais
            )
            self._memory.append({"role": "user", "content": group_context})
        except FileNotFoundError:
            logger.error(f"Group conversation prompt file not found: {prompt_name}")
        except KeyError as e:
            logger.error(f"Missing formatting key in group conversation prompt: {e}")
        except Exception as e:
            logger.error(f"Failed to load group conversation prompt: {e}")
