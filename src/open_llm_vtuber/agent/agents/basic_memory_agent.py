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
    get_metadata,
    update_summary_metadata,
)
from ...character_state import (
    CharacterState,
    add_character_memory as persist_character_memory,
    build_character_memory_context,
    load_character_state,
    migrate_relationship_if_needed,
    remove_character_memory as remove_persisted_character_memory,
    reset_character_memory as reset_persisted_character_memory,
    reset_character_state as reset_persisted_character_state,
    set_character_relationship,
)
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
    IncrementalSummarizer,
    SummaryState,
    build_summary_message,
)
from ..relationship_context import (
    RelationshipState,
    RelationshipStatus,
    build_relationship_context,
    detect_relationship_update,
    normalize_relationship_status,
)
import re
import time
from ...request_latency import (
    get_latency_tracker,
    reset_latency_phase,
    set_latency_phase,
)

# Conservative, local-only character memory triggers. No LLM classifier and no
# extra API call per message. Only explicit user requests are honored:
#   - Remember:  "Ingat ya, makanan favoritku ramen."
#                "Jangan lupa kalau aku suka kopi."
#   - Forget:    "Lupakan kalau makanan favoritku ramen."
_MEMORY_REMEMBER = re.compile(
    r"(?:tolong\s+)?(?:ingat|catat)\s+(?:ya|yah|dong|deh|dulu|kalau)?\s*[,:]?\s*|"
    r"jangan\s+lupa\s+(?:ya|yah|dong|deh)?\s*[,:]?\s*",
    re.IGNORECASE,
)
_MEMORY_FORGET = re.compile(
    r"(?:tolong\s+)?lupakan\s+(?:ya|yah|dong|deh)?\s*"
    r"(?:kalau|soal|tentang|yang\s+kamu\s+ingat)?\s+|"
    r"hapus\s+dari\s+ingatan\s+",
    re.IGNORECASE,
)


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
    ):
        """Initialize agent with LLM and configuration."""
        super().__init__()
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
        self._relationship_state = RelationshipState(
            status=state.relationship_status,
            updated_at=state.relationship_updated_at,
            reason=state.relationship_reason,
        )
        logger.info(
            "Character state stats: relationship_status={}, "
            "character_memory_count={}, relationship_update_trigger=load_history",
            state.relationship_status,
            len(state.memories),
        )

    @property
    def relationship_status(self) -> RelationshipStatus:
        """Expose the character-level relationship state for backend/tests."""
        return self._relationship_state.status

    def _relationship_system_prompt(self, base_prompt: str) -> str:
        parts = [
            base_prompt,
            build_relationship_context(self._relationship_state.status),
        ]
        memory_context = build_character_memory_context(self._character_state)
        if memory_context:
            parts.append(memory_context)
        return "\n\n".join(parts)

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
            logger.warning(
                "Relationship update skipped: no active character context"
            )
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

    def add_character_memory(self, text: str, *, explicit: bool = True) -> bool:
        """Persist one long-term fact shared across all chats."""
        if not self._character_conf_uid:
            logger.warning("Character memory update skipped: no active character")
            return False
        save_started = time.perf_counter()
        state = persist_character_memory(
            self._character_conf_uid, text, explicit=explicit
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

    def list_character_memories(self) -> List[Dict[str, Any]]:
        """Return stored long-term facts (for backend controls / future UI)."""
        return list(self._character_state.memories)

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
        text = (user_text or "").strip()
        if not text:
            return False
        forget_match = _MEMORY_FORGET.match(text)
        if forget_match:
            target = text[forget_match.end():].strip(" .,!?;:，。！？；：")
            if len(target) >= 3:
                return self.remove_character_memory(target)
        remember_match = _MEMORY_REMEMBER.match(text)
        if remember_match:
            raw = text[remember_match.end():].strip()
            content = raw.strip(" .,!?;:，。！？；：")
            if (
                len(content) >= 4
                and not raw.endswith("?")
                and not raw.endswith("？")
            ):
                return self.add_character_memory(content, explicit=True)
        return False

    def observe_character_events(
        self,
        user_text: str,
        assistant_text: str,
    ) -> bool:
        """Observe one completed visible turn: relationship + explicit memory."""
        relationship_updated = self.observe_relationship_event(
            user_text, assistant_text
        )
        memory_updated = self._observe_character_memory_request(user_text)
        return relationship_updated or memory_updated

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
                tracker.add_metadata_save(
                    (time.perf_counter() - save_started) * 1000
                )
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
                build_summary_message(self._summary_state.text, role=summary_role),
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
        summary_included = summary_present and bool(final_selection.messages) and (
            final_selection.messages[0].get("content", "").startswith(
                "Conversation context from earlier messages"
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

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load memory from chat history."""
        messages = get_history(conf_uid, history_uid)

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

        while True:
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

        while True:
            if self.prompt_mode_flag:
                if self._mcp_prompt_string:
                    base_system_prompt = (
                        f"{self._system}\n\n{self._mcp_prompt_string}"
                    )
                else:
                    logger.warning("Prompt mode active but mcp_prompt_string is empty!")
                    base_system_prompt = self._system
                tools_for_api = None
            else:
                base_system_prompt = self._system
                tools_for_api = tools
            current_system_prompt = self._relationship_system_prompt(
                base_system_prompt
            )

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
            stream = self._llm.chat_completion(
                request_messages, current_system_prompt, tools=tools_for_api
            )
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
    ) -> Callable[[], AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]]:
        """Create an assistant-only turn using the normal character context.

        The generation cue is appended to the effective system prompt only for
        this request.  It never enters ``_memory`` or the persisted transcript.
        The generated assistant message does enter ``_memory`` normally.
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
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Generate one proactive assistant message without a fake user turn."""
        proactive_chat = self._proactive_chat_function_factory()
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
