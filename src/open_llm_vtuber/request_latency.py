"""Per-request latency instrumentation without logging conversation content."""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from loguru import logger


WebSocketEmitter = Callable[[str], Awaitable[None]]


_current_tracker: ContextVar[Optional["RequestLatencyTracker"]] = ContextVar(
    "request_latency_tracker", default=None
)
_latency_phase: ContextVar[str] = ContextVar("request_latency_phase", default="chat")
_recent_stats: deque[dict[str, float]] = deque(maxlen=20)


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


def get_latency_tracker() -> Optional["RequestLatencyTracker"]:
    return _current_tracker.get()


def set_latency_tracker(tracker: "RequestLatencyTracker"):
    return _current_tracker.set(tracker)


def reset_latency_tracker(token) -> None:
    _current_tracker.reset(token)


def get_latency_phase() -> str:
    return _latency_phase.get()


def set_latency_phase(phase: str):
    return _latency_phase.set(phase)


def reset_latency_phase(token) -> None:
    _latency_phase.reset(token)


def classify_bottleneck(metrics: dict[str, float]) -> str:
    """Classify synthetic measurements; live conclusions still require live data."""
    context_ms = metrics.get("context_build_ms", 0.0) + metrics.get("summary_ms", 0.0)
    provider_ms = metrics.get("ollama_request_to_first_token_ms", 0.0)
    tool_ms = metrics.get("tool_ms", 0.0)
    if provider_ms >= max(1000.0, context_ms * 2, tool_ms * 2):
        return "provider_or_model_ttft"
    if context_ms >= max(1000.0, provider_ms * 2, tool_ms * 2):
        return "backend_context"
    if tool_ms >= max(1000.0, provider_ms * 2, context_ms * 2):
        return "tool_processing"
    return "mixed_or_inconclusive"


@dataclass
class RequestLatencyTracker:
    websocket_send: WebSocketEmitter
    provider: str
    model: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    client_user_send_ms: Optional[float] = None
    client_websocket_send_ms: Optional[float] = None
    started_ms: float = field(default_factory=monotonic_ms)
    received_epoch_ms: float = field(default_factory=lambda: time.time() * 1000.0)
    context_build_ms: float = 0.0
    summary_ms: float = 0.0
    tool_ms: float = 0.0
    character_event_ms: float = 0.0
    tts_ms: float = 0.0
    provider_request_ms: Optional[float] = None
    provider_headers_ms: Optional[float] = None
    provider_first_token_ms: Optional[float] = None
    provider_final_token_ms: Optional[float] = None
    output_chars: int = 0
    message_count: int = 0
    estimated_input_tokens: int = 0
    summary_triggered: bool = False
    tool_used: bool = False
    _first_token_emitted: bool = False

    async def emit(self, event: str, **values: Any) -> None:
        payload = {
            "type": "latency-event",
            "event": event,
            "request_id": self.request_id,
            **values,
        }
        await self.websocket_send(json.dumps(payload))

    async def provider_first_token(self) -> None:
        now = monotonic_ms()
        if self.provider_first_token_ms is None:
            self.provider_first_token_ms = now
        if not self._first_token_emitted:
            self._first_token_emitted = True
            await self.emit("first-token")

    def provider_started(self) -> None:
        if self.provider_request_ms is None:
            self.provider_request_ms = monotonic_ms()

    def provider_headers_received(self) -> None:
        if self.provider_headers_ms is None:
            self.provider_headers_ms = monotonic_ms()

    def provider_finished(self) -> None:
        self.provider_final_token_ms = monotonic_ms()

    def add_context(self, duration_ms: float, selection: Any = None) -> None:
        self.context_build_ms += max(0.0, duration_ms)
        if selection is not None:
            stats = selection.stats
            self.message_count = stats.messages_after
            self.estimated_input_tokens = stats.estimated_input_tokens

    def add_summary(self, duration_ms: float, triggered: bool) -> None:
        self.summary_ms += max(0.0, duration_ms)
        self.summary_triggered = self.summary_triggered or triggered

    def add_tool(self, duration_ms: float) -> None:
        self.tool_ms += max(0.0, duration_ms)
        self.tool_used = True

    def metrics(self, completed_ms: Optional[float] = None) -> dict[str, Any]:
        completed = completed_ms or monotonic_ms()
        request_to_first = 0.0
        generation = 0.0
        request_to_headers = 0.0
        if self.provider_request_ms is not None and self.provider_headers_ms is not None:
            request_to_headers = self.provider_headers_ms - self.provider_request_ms
        if self.provider_request_ms is not None and self.provider_first_token_ms is not None:
            request_to_first = self.provider_first_token_ms - self.provider_request_ms
        if self.provider_first_token_ms is not None and self.provider_final_token_ms is not None:
            generation = self.provider_final_token_ms - self.provider_first_token_ms
        estimated_output_tokens = max(0, (self.output_chars + 3) // 4)
        tokens_per_second = (
            estimated_output_tokens / (generation / 1000.0)
            if generation > 0 and estimated_output_tokens
            else 0.0
        )
        frontend_to_backend = None
        user_send_to_backend = None
        if self.client_user_send_ms is not None:
            candidate = self.received_epoch_ms - self.client_user_send_ms
            if 0 <= candidate <= 300_000:
                user_send_to_backend = candidate
        if self.client_websocket_send_ms is not None:
            candidate = self.received_epoch_ms - self.client_websocket_send_ms
            if 0 <= candidate <= 300_000:
                frontend_to_backend = candidate
        values = {
            "provider": self.provider,
            "model": self.model,
            "frontend_user_send_to_backend_ms": user_send_to_backend,
            "frontend_to_backend_ms": frontend_to_backend,
            "context_build_ms": round(self.context_build_ms, 2),
            "summary_ms": round(self.summary_ms, 2),
            "summary_triggered": self.summary_triggered,
            "tool_ms": round(self.tool_ms, 2),
            "tool_used": self.tool_used,
            "character_event_ms": round(self.character_event_ms, 2),
            "tts_ms": round(self.tts_ms, 2),
            "ollama_request_to_first_token_ms": round(request_to_first, 2),
            "ollama_request_to_headers_ms": round(request_to_headers, 2),
            "ollama_generation_ms": round(generation, 2),
            "output_tokens_per_second_estimate": round(tokens_per_second, 2),
            "message_count": self.message_count,
            "estimated_input_tokens": self.estimated_input_tokens,
            "total_response_ms": round(completed - self.started_ms, 2),
        }
        values["bottleneck_hint"] = classify_bottleneck(values)
        return values

    async def complete(self) -> dict[str, Any]:
        values = self.metrics()
        safe_values = {key: value for key, value in values.items() if value is not None}
        logger.info(
            "[LLM LATENCY] " + " ".join(f"{key}={value}" for key, value in safe_values.items())
        )
        numeric = {
            "ttft": float(values["ollama_request_to_first_token_ms"]),
            "total": float(values["total_response_ms"]),
        }
        _recent_stats.append(numeric)
        ttfts = [item["ttft"] for item in _recent_stats if item["ttft"] > 0]
        totals = [item["total"] for item in _recent_stats]
        if totals:
            logger.info(
                "[LLM LATENCY STATS] requests={} average_ttft_ms={} min_ttft_ms={} "
                "max_ttft_ms={} average_total_ms={}",
                len(_recent_stats),
                round(sum(ttfts) / len(ttfts), 2) if ttfts else 0.0,
                round(min(ttfts), 2) if ttfts else 0.0,
                round(max(ttfts), 2) if ttfts else 0.0,
                round(sum(totals) / len(totals), 2),
            )
        await self.emit("response-complete", metrics=safe_values)
        return values
