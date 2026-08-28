"""Lightweight runtime scheduling primitives for proactive character chat.

The state in this module is deliberately ephemeral.  Conversation history,
rolling summaries, relationship state, and long-term character memory keep
their existing persistence rules; idle timestamps never enter those stores.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
import time
from typing import Callable, Optional


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


class ProactiveStateMachine:
    """Pure timing/state transitions with injectable clock and randomness."""

    def __init__(
        self,
        config: ProactiveChatConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        randint: Callable[[int, int], int] = random.randint,
    ) -> None:
        self.config = config
        self._monotonic = monotonic
        self._randint = randint

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
    ) -> None:
        now = self._monotonic()
        state.last_proactive_monotonic = now
        state.consecutive_ignored_proactive += 1
        state.proactive_generation_in_progress = False
        state.last_proactive_text = response_text or None
        state.last_proactive_expected_response = message_expects_response(response_text)
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

    def seconds_until_eligible(self, state: ProactiveRuntimeState) -> float:
        return max(0.0, state.next_proactive_eligible_at - self._monotonic())

    def is_eligible(self, state: ProactiveRuntimeState) -> bool:
        return (
            self.config.enabled
            and not state.proactive_generation_in_progress
            and self.seconds_until_eligible(state) <= 0
        )
