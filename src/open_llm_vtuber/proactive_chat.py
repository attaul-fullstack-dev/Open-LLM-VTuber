"""Lightweight runtime scheduling primitives for proactive character chat.

The state in this module is deliberately ephemeral.  Conversation history,
rolling summaries, relationship state, and long-term character memory keep
their existing persistence rules; idle timestamps never enter those stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
import time
from typing import Callable, Dict, List, Mapping, Optional


class ProactiveIntent:
    """Lightweight proactive behavior intents (internal only, never shown)."""

    REACT_TO_IGNORED_QUESTION = "react_to_ignored_question"
    REACT_TO_SILENCE = "react_to_silence"
    CONTINUE_PREVIOUS_TOPIC = "continue_previous_topic"
    START_NEW_TOPIC = "start_new_topic"
    ASK_USER_SOMETHING = "ask_user_something"
    BRING_UP_MEMORY = "bring_up_memory"
    CASUAL_OBSERVATION = "casual_observation"


# Fixed iteration order for weighted selection (deterministic wheel layout).
INTENT_SELECTION_ORDER = (
    ProactiveIntent.REACT_TO_SILENCE,
    ProactiveIntent.CONTINUE_PREVIOUS_TOPIC,
    ProactiveIntent.START_NEW_TOPIC,
    ProactiveIntent.ASK_USER_SOMETHING,
    ProactiveIntent.BRING_UP_MEMORY,
    ProactiveIntent.CASUAL_OBSERVATION,
)

# react_to_ignored_question is priority-driven and intentionally excluded
# from weighted selection.
DEFAULT_INTENT_WEIGHTS: Dict[str, float] = {
    ProactiveIntent.REACT_TO_SILENCE: 5,
    ProactiveIntent.CONTINUE_PREVIOUS_TOPIC: 20,
    ProactiveIntent.START_NEW_TOPIC: 30,
    ProactiveIntent.ASK_USER_SOMETHING: 20,
    ProactiveIntent.BRING_UP_MEMORY: 15,
    ProactiveIntent.CASUAL_OBSERVATION: 10,
}

# Per recent occurrence, a repeated intent's weight is multiplied down so
# selections stay varied.  Silence acknowledgment decays extra fast.
_INTENT_REPEAT_PENALTY = 0.25
_SILENCE_REPEAT_PENALTY = 0.1


@dataclass(frozen=True)
class ProactiveChatConfig:
    """Centralized proactive timing configuration."""

    enabled: bool = True
    initial_idle_min_seconds: int = 45
    initial_idle_max_seconds: int = 90
    followup_idle_min_seconds: int = 90
    followup_idle_max_seconds: int = 240
    ignored_before_backoff: int = 3
    backoff_min_seconds: int = 180
    backoff_max_seconds: int = 360
    intent_weights: Optional[Mapping[str, float]] = None

    def __post_init__(self) -> None:
        ranges = (
            (
                "initial idle",
                self.initial_idle_min_seconds,
                self.initial_idle_max_seconds,
            ),
            (
                "follow-up idle",
                self.followup_idle_min_seconds,
                self.followup_idle_max_seconds,
            ),
            ("backoff", self.backoff_min_seconds, self.backoff_max_seconds),
        )
        for label, minimum, maximum in ranges:
            if minimum < 0 or maximum < minimum:
                raise ValueError(
                    f"Invalid proactive {label} range: {minimum}-{maximum}"
                )
        if self.ignored_before_backoff < 1:
            raise ValueError("ignored_before_backoff must be at least 1")
        if self.intent_weights is not None:
            for key, value in self.intent_weights.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0
                ):
                    raise ValueError(
                        f"Invalid proactive intent weight for {key!r}: {value!r}"
                    )


# Sentence-initial interrogatives (Indonesian + English) used as a fallback
# signal when a proactive message expects a reply without a question mark.
_QUESTION_WORD_PATTERN = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(?:apakah|apa|gimana|bagaimana|kenapa|kok|kapan|dimana|di mana|siapa|"
    r"what|why|how|when|where|who)\b",
    re.IGNORECASE,
)


def message_expects_response(text: Optional[str]) -> bool:
    """Deterministic check for whether a message asks for a user reply.

    A question mark anywhere is the primary signal.  As a fallback, a
    sentence-initial interrogative word (Indonesian or English) also counts.
    This is a lightweight heuristic over already-generated text -- it never
    triggers any model call.
    """
    if not text:
        return False
    if "?" in text:
        return True
    return bool(_QUESTION_WORD_PATTERN.search(text))


@dataclass(frozen=True)
class ProactiveFollowupContext:
    """Internal-only signal about a possibly ignored proactive message.

    Consumed by proactive generation to react naturally to being ignored;
    it must never be printed verbatim to the user.
    """

    previous_proactive_ignored: bool
    consecutive_ignored: int
    previous_proactive_expected_response: bool

    def as_dict(self) -> dict:
        return {
            "previous_proactive_ignored": self.previous_proactive_ignored,
            "consecutive_ignored": self.consecutive_ignored,
            "previous_proactive_expected_response": self.previous_proactive_expected_response,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["ProactiveFollowupContext"]:
        if not isinstance(data, dict):
            return None
        return cls(
            previous_proactive_ignored=bool(data.get("previous_proactive_ignored")),
            consecutive_ignored=int(data.get("consecutive_ignored") or 0),
            previous_proactive_expected_response=bool(
                data.get("previous_proactive_expected_response")
            ),
        )


@dataclass(frozen=True)
class ProactiveIntentContext:
    """Internal-only proactive intent signal for one generation turn."""

    intent: str
    user_has_replied_since_last_proactive: bool
    consecutive_ignored: int
    recent_silence_acknowledgment: bool

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "user_has_replied_since_last_proactive": (
                self.user_has_replied_since_last_proactive
            ),
            "consecutive_ignored": self.consecutive_ignored,
            "recent_silence_acknowledgment": self.recent_silence_acknowledgment,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["ProactiveIntentContext"]:
        if not isinstance(data, dict):
            return None
        intent = str(data.get("intent") or "")
        known = set(INTENT_SELECTION_ORDER) | {
            ProactiveIntent.REACT_TO_IGNORED_QUESTION
        }
        if intent not in known:
            intent = ProactiveIntent.CASUAL_OBSERVATION
        return cls(
            intent=intent,
            user_has_replied_since_last_proactive=bool(
                data.get("user_has_replied_since_last_proactive")
            ),
            consecutive_ignored=int(data.get("consecutive_ignored") or 0),
            recent_silence_acknowledgment=bool(
                data.get("recent_silence_acknowledgment")
            ),
        )


@dataclass(frozen=True)
class ProactiveIntentSignals:
    """Cheap local context signals that adjust intent selection weights."""

    has_useful_memory: bool = False
    has_recent_context: bool = False
    unfinished_topic: bool = False


@dataclass
class ProactiveRuntimeState:
    """Per-connection, per-chat state; never persisted."""

    history_uid: str
    last_user_activity_monotonic: float
    next_proactive_eligible_at: float
    last_proactive_monotonic: Optional[float] = None
    consecutive_ignored_proactive: int = 0
    proactive_generation_in_progress: bool = False
    activity_revision: int = 0
    # Ephemeral anti-repetition memory of recent intents (not persisted).
    recent_proactive_intents: List[str] = field(default_factory=list)
    # Text and expected-reply flag of the most recent successfully sent
    # proactive message.  Cleared whenever the user responds.
    last_proactive_text: Optional[str] = None
    last_proactive_expected_response: bool = False


def format_followup_instruction(
    context: Optional[ProactiveFollowupContext],
) -> Optional[str]:
    """Internal system-prompt block describing a possibly ignored proactive turn.

    Returns ``None`` when there is nothing to react to.  The wording stays
    persona-neutral and never reveals counters, timers, or system mechanics
    to the user; it only steers the next proactive generation.
    """
    if context is None or not context.previous_proactive_ignored:
        return None

    count = max(1, context.consecutive_ignored)
    if context.previous_proactive_expected_response:
        expectation = "yes -- it asked the user a direct question"
        priority = (
            "your previous message asked the user something and they never "
            "answered; strongly prefer reacting to the unanswered question "
            "first -- notice the silence with mild irritation, confusion, "
            "teasing, embarrassment, or a short complaint, escalating "
            "naturally with each ignored follow-up (first: mild confusion or "
            "teasing; second: more impatient or annoyed; third or later: "
            "resigned, sulking, briefly giving up, or changing topic); never "
            "repeat the exact same question"
        )
    else:
        expectation = "no -- it was a statement, not a question"
        priority = (
            "your previous proactive message was a statement, not a question; "
            "do NOT claim the user failed to answer a question -- you may "
            "notice the silence more generally, tease lightly, or naturally "
            "move to something else"
        )

    return (
        "Internal follow-up context for this turn only. Never shown to the "
        "user; never mention counters, timers, idle detection, or proactive "
        "behavior:\n"
        "- Your previous proactive message was ignored by the user (no reply "
        "so far).\n"
        f"- Consecutive proactive messages ignored: {count}.\n"
        f"- That message expected a reply: {expectation}.\n"
        f"Priority for this turn: {priority}."
    )


_INTENT_GUIDANCE: Dict[str, str] = {
    ProactiveIntent.REACT_TO_IGNORED_QUESTION: (
        "Follow the unanswered-question priority: react naturally to being "
        "ignored after asking something."
    ),
    ProactiveIntent.REACT_TO_SILENCE: (
        "The user went quiet with no question pending; notice it lightly at "
        "most and never demand a reply or reuse stock silence phrases."
    ),
    ProactiveIntent.CONTINUE_PREVIOUS_TOPIC: (
        "Continue something from the recent conversation naturally, as if "
        "the thought just came back."
    ),
    ProactiveIntent.START_NEW_TOPIC: (
        "Bring up a subject of your own choosing (curiosity, hypotheticals, "
        "daily life, games, stories, movies, tech, food, places, or the "
        "user's known interests); just start talking, never announce a "
        "topic or offer options."
    ),
    ProactiveIntent.ASK_USER_SOMETHING: (
        "Ask something you are genuinely curious about, spontaneously."
    ),
    ProactiveIntent.BRING_UP_MEMORY: (
        "Naturally recall something you already know about the user; never "
        "expose memory metadata, IDs, or internal terminology."
    ),
    ProactiveIntent.CASUAL_OBSERVATION: (
        "A short thought or remark that does not require a reply."
    ),
}

_INTENT_STYLE_GUIDANCE = (
    "You are initiating by your own choice: 1-3 short sentences of "
    "conversational Indonesian; a question is optional. Never claim "
    "specific past personal events (finishing a book or movie, going "
    "somewhere, what a friend said) unless the conversation supports them; "
    "saying you were just thinking about something or remember discussing "
    "something is fine."
)


def format_intent_instruction(
    context: Optional[ProactiveIntentContext],
    *,
    include_guidance: bool = True,
) -> Optional[str]:
    """Internal system-prompt block steering this proactive turn's intent.

    Returns ``None`` when there is no intent context.  The block is internal
    only: never shown verbatim to the user and never persisted to history.
    When ``include_guidance`` is False (e.g. the ignored-question follow-up
    block already carries the turn's instructions), only the compact context
    lines are emitted to avoid duplicating guidance tokens.
    """
    if context is None:
        return None

    intent = context.intent
    if intent not in _INTENT_GUIDANCE:
        intent = ProactiveIntent.CASUAL_OBSERVATION
    lines = [
        "Internal proactive context for this turn only. Never shown to the "
        "user; never mention intents, counters, timers, or system "
        "mechanics.",
        f"intent: {intent}",
        "user_has_replied_since_last_proactive: "
        f"{str(context.user_has_replied_since_last_proactive).lower()}",
        f"consecutive_ignored: {context.consecutive_ignored}",
        "recent_silence_acknowledgment: "
        f"{str(context.recent_silence_acknowledgment).lower()}",
    ]
    if include_guidance:
        lines.extend(
            [
                f"Intent for this turn: {_INTENT_GUIDANCE[intent]}",
                _INTENT_STYLE_GUIDANCE,
            ]
        )
    return "\n".join(lines)


def resolve_proactive_intent(
    followup_context: Optional[ProactiveFollowupContext],
    state: ProactiveRuntimeState,
    machine: "ProactiveStateMachine",
    signals: ProactiveIntentSignals,
    *,
    random: Optional[Callable[[], float]] = None,
) -> str:
    """Pick this turn's intent; an unanswered proactive question wins."""
    if (
        followup_context is not None
        and followup_context.previous_proactive_ignored
        and followup_context.previous_proactive_expected_response
    ):
        return ProactiveIntent.REACT_TO_IGNORED_QUESTION
    return machine.select_proactive_intent(state, signals, random=random)


def _pick_intent(weights: Mapping[str, float], spin: float) -> str:
    """Deterministic weighted wheel over a fixed intent order."""
    ordered = [
        (key, float(weights[key]))
        for key in INTENT_SELECTION_ORDER
        if float(weights.get(key, 0.0)) > 0.0
    ]
    if not ordered:
        return ProactiveIntent.CASUAL_OBSERVATION
    total = sum(weight for _, weight in ordered)
    cursor = spin * total
    for key, weight in ordered:
        cursor -= weight
        if cursor < 0:
            return key
    return ordered[-1][0]


class ProactiveStateMachine:
    """Pure timing/state transitions with injectable clock and randomness."""

    def __init__(
        self,
        config: ProactiveChatConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        randint: Callable[[int, int], int] = random.randint,
        random: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self._monotonic = monotonic
        self._randint = randint
        self._random = random

    def _delay(self, minimum: int, maximum: int) -> float:
        return float(self._randint(minimum, maximum))

    def new_state(self, history_uid: str) -> ProactiveRuntimeState:
        now = self._monotonic()
        return ProactiveRuntimeState(
            history_uid=history_uid,
            last_user_activity_monotonic=now,
            next_proactive_eligible_at=now
            + self._delay(
                self.config.initial_idle_min_seconds,
                self.config.initial_idle_max_seconds,
            ),
        )

    def record_user_activity(self, state: ProactiveRuntimeState) -> None:
        now = self._monotonic()
        state.activity_revision += 1
        state.last_user_activity_monotonic = now
        state.consecutive_ignored_proactive = 0
        state.proactive_generation_in_progress = False
        state.last_proactive_text = None
        state.last_proactive_expected_response = False
        state.next_proactive_eligible_at = now + self._delay(
            self.config.initial_idle_min_seconds,
            self.config.initial_idle_max_seconds,
        )

    def record_proactive_sent(
        self,
        state: ProactiveRuntimeState,
        *,
        response_text: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> None:
        now = self._monotonic()
        state.last_proactive_monotonic = now
        state.consecutive_ignored_proactive += 1
        state.proactive_generation_in_progress = False
        state.last_proactive_text = response_text or None
        state.last_proactive_expected_response = message_expects_response(response_text)
        if intent:
            state.recent_proactive_intents.append(intent)
            del state.recent_proactive_intents[:-3]
        if state.consecutive_ignored_proactive >= self.config.ignored_before_backoff:
            minimum = self.config.backoff_min_seconds
            maximum = self.config.backoff_max_seconds
        else:
            minimum = self.config.followup_idle_min_seconds
            maximum = self.config.followup_idle_max_seconds
        state.next_proactive_eligible_at = now + self._delay(minimum, maximum)

    def proactive_followup_context(
        self, state: ProactiveRuntimeState
    ) -> ProactiveFollowupContext:
        """Snapshot whether the last proactive message went unanswered.

        ``consecutive_ignored_proactive`` counts successful proactive sends
        since the last user activity, so a value >= 1 means the previous
        assistant message was proactive and the user has not replied since.
        """
        ignored = state.consecutive_ignored_proactive >= 1
        return ProactiveFollowupContext(
            previous_proactive_ignored=ignored,
            consecutive_ignored=max(0, state.consecutive_ignored_proactive),
            previous_proactive_expected_response=(
                ignored and state.last_proactive_expected_response
            ),
        )

    def select_proactive_intent(
        self,
        state: ProactiveRuntimeState,
        signals: ProactiveIntentSignals,
        *,
        random: Optional[Callable[[], float]] = None,
    ) -> str:
        """Weighted, anti-repetitive local intent pick (no LLM involved)."""
        return _pick_intent(
            self.effective_intent_weights(state, signals),
            (random or self._random)(),
        )

    def effective_intent_weights(
        self,
        state: ProactiveRuntimeState,
        signals: ProactiveIntentSignals,
    ) -> Dict[str, float]:
        """Resolve context-aware, anti-repetition-adjusted intent weights."""
        merged = dict(DEFAULT_INTENT_WEIGHTS)
        if self.config.intent_weights:
            for key, value in self.config.intent_weights.items():
                if key in merged:
                    merged[key] = float(value)
        if not signals.has_useful_memory:
            merged[ProactiveIntent.BRING_UP_MEMORY] = 0.0
        if not signals.has_recent_context:
            merged[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC] = 0.0
            for key in (
                ProactiveIntent.START_NEW_TOPIC,
                ProactiveIntent.ASK_USER_SOMETHING,
                ProactiveIntent.CASUAL_OBSERVATION,
            ):
                merged[key] *= 1.5
        if signals.unfinished_topic:
            merged[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC] *= 2.0
        recent = state.recent_proactive_intents[-3:]
        for key in merged:
            occurrences = recent.count(key)
            if occurrences:
                penalty = (
                    _SILENCE_REPEAT_PENALTY
                    if key == ProactiveIntent.REACT_TO_SILENCE
                    else _INTENT_REPEAT_PENALTY
                )
                merged[key] *= penalty**occurrences
        return merged

    def seconds_until_eligible(self, state: ProactiveRuntimeState) -> float:
        return max(0.0, state.next_proactive_eligible_at - self._monotonic())

    def is_eligible(self, state: ProactiveRuntimeState) -> bool:
        return (
            self.config.enabled
            and not state.proactive_generation_in_progress
            and self.seconds_until_eligible(state) <= 0
        )
