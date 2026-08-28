"""Lightweight runtime scheduling primitives for proactive character chat.

The state in this module is deliberately ephemeral.  Conversation history,
rolling summaries, relationship state, and long-term character memory keep
their existing persistence rules; idle timestamps never enter those stores.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
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
        state.next_proactive_eligible_at = now + self._delay(
            self.config.initial_idle_min_seconds,
            self.config.initial_idle_max_seconds,
        )

    def record_proactive_sent(self, state: ProactiveRuntimeState) -> None:
        now = self._monotonic()
        state.last_proactive_monotonic = now
        state.consecutive_ignored_proactive += 1
        state.proactive_generation_in_progress = False
        if state.consecutive_ignored_proactive >= self.config.ignored_before_backoff:
            minimum = self.config.backoff_min_seconds
            maximum = self.config.backoff_max_seconds
        else:
            minimum = self.config.followup_idle_min_seconds
            maximum = self.config.followup_idle_max_seconds
        state.next_proactive_eligible_at = now + self._delay(minimum, maximum)

    def seconds_until_eligible(self, state: ProactiveRuntimeState) -> float:
        return max(0.0, state.next_proactive_eligible_at - self._monotonic())

    def is_eligible(self, state: ProactiveRuntimeState) -> bool:
        return (
            self.config.enabled
            and not state.proactive_generation_in_progress
            and self.seconds_until_eligible(state) <= 0
        )
