"""Per-request latency instrumentation without logging conversation content.

Phase 2 design goals:
- Every request carries a stable ``request_id`` from frontend to completion.
- Durations always use ``time.perf_counter()`` (monotonic); wall-clock is only
  kept for human-readable debugging.
- The whole lifecycle is partitioned into phases; anything left over is
  reported as ``unattributed_ms`` with the single largest time gap named.
- Provider values that never occurred are reported as ``None``, never 0.0, so
  "provider never started" is distinguishable from "provider answered in 0ms".
- No chat text, response text, persona, summary, memory, API key, or headers
  are ever logged.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

import httpx


WebSocketEmitter = Callable[[str], Awaitable[None]]


_current_tracker: ContextVar[Optional["RequestLatencyTracker"]] = ContextVar(
    "request_latency_tracker", default=None
)
_latency_phase: ContextVar[str] = ContextVar("request_latency_phase", default="chat")
_recent_stats: deque[dict[str, float]] = deque(maxlen=20)

# Open-LLM-VTuber process start, used only for human-readable epoch offsets.
_PROCESS_STARTED_EPOCH_MS = time.time() * 1000.0


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


def _friendly_name(mark_name: str) -> str:
    """Convert an internal mark name into a stable phase label."""
    if mark_name.endswith("_start"):
        return mark_name[: -len("_start")]
    if mark_name.endswith("_end"):
        return mark_name[: -len("_end")]
    return mark_name


# ---------------------------------------------------------------------------
# Bottleneck classification (v2)
# ---------------------------------------------------------------------------

_BOTTLENECK_CATEGORIES = (
    "provider_ttft",
    "provider_generation",
    "provider_retry",
    "provider_timeout",
    "context",
    "summary",
    "tool",
    "tts",
    "disk_persistence",
    "response_processing",
    "websocket",
    "client_disconnect",
    "unattributed",
    "mixed",
    "unknown",
)

#: ``unattributed`` is only reported when this much wall-clock time cannot be
#: mapped to any measured phase. Normal requests show sub-millisecond
#: unattributed time; a corridor between marks that sits inside a *known*
#: phase (tts_wait, playback_wait, provider TTFT, ...) is measured latency,
#: not unattributed time.
_UNATTRIBUTED_THRESHOLD_MS = 500.0


def classify_bottleneck(metrics: dict[str, Any]) -> str:
    """Classify a request's dominant latency source.

    Accepts the metrics dict produced by ``RequestLatencyTracker.metrics``.
    Missing/None values are treated as 0.0.
    """
    value = lambda key: metrics.get(key) or 0.0  # noqa: E731

    if metrics.get("interrupted") or metrics.get("client_disconnected"):
        return "client_disconnect"

    provider_ttft = value("ollama_request_to_first_token_ms")
    provider_gen = value("ollama_generation_ms")
    context_ms = value("context_build_ms")
    summary_ms = value("summary_ms")
    tool_ms = value("tool_ms")
    # TTS wall-clock blocking plus browser playback wait: serial audio phases.
    tts_ms = value("tts_total_ms") + value("playback_wait_ms")
    disk_ms = (
        value("history_save_ms") + value("metadata_save_ms")
        + value("character_state_save_ms")
    )
    resp_proc_ms = value("response_processing_ms")
    total_ms = value("total_response_ms")
    agent_stream_ms = value("agent_stream_ms")
    # Agent-side time inside agent_stream that is NOT the provider call itself
    # (sentence splitting, expression handling, audio prep, ...).
    agent_non_provider = max(
        0.0, agent_stream_ms - value("provider_wallclock_ms")
    )
    attempts = int(metrics.get("provider_attempt_count") or 0)
    provider_started = bool(metrics.get("provider_started"))
    headers_received = bool(metrics.get("provider_headers_received"))

    if provider_started and not headers_received:
        # Headers never arrived: connection hang / SDK retries exhausted.
        if attempts > 1:
            return "provider_retry"
        if total_ms >= 1000:
            return "provider_timeout"
        return "unknown"

    if provider_started and attempts > 1:
        return "provider_retry"

    # ``unattributed`` is reserved for wall-clock time that genuinely cannot
    # be mapped to a measured phase. It is driven by ``unattributed_ms`` (the
    # leftover after all exclusive segments), never by ``largest_gap_ms``: a
    # large gap between marks that sits inside a known phase such as tts_wait
    # is measured latency, and mislabeling it "unattributed" hides the real
    # bottleneck.
    if value("unattributed_ms") >= _UNATTRIBUTED_THRESHOLD_MS:
        return "unattributed"

    if not provider_started:
        # The provider was never called; blame the largest pre-provider phase.
        candidates = [
            (value("agent_stream_ms"), "response_processing"),
            (context_ms, "context"),
            (summary_ms, "summary"),
            (tool_ms, "tool"),
            (tts_ms, "tts"),
            (disk_ms, "disk_persistence"),
            (resp_proc_ms, "response_processing"),
        ]
        top_ms, top_name = max(candidates, key=lambda item: item[0])
        if top_ms >= 1000:
            return top_name
        return "unknown"

    contributions = [
        (provider_ttft, "provider_ttft"),
        (provider_gen, "provider_generation"),
        (context_ms, "context"),
        (summary_ms, "summary"),
        (tool_ms, "tool"),
        (tts_ms, "tts"),
        (agent_non_provider, "response_processing"),
        (disk_ms, "disk_persistence"),
        (resp_proc_ms, "response_processing"),
    ]
    top_ms, top_label = max(contributions, key=lambda item: item[0])

    if top_ms >= 1000 and top_ms >= 0.5 * max(total_ms, 1.0):
        return top_label
    if top_ms >= 500:
        return top_label
    if total_ms >= 1000:
        return "mixed"
    return "unknown"


# ---------------------------------------------------------------------------
# Provider attempt tracking transport
# ---------------------------------------------------------------------------


class AttemptTrackingTransport(httpx.AsyncBaseTransport):
    """Wraps the underlying httpx transport and records per-attempt timing.

    Every call to ``handle_async_request`` is one provider attempt (the OpenAI
    SDK retries by re-issuing the request, which flows through this transport
    again). Stream end is observed by wrapping the response's ``aiter_raw`` so
    an attempt that fails mid-stream is classified as such.

    This wrapper does not change retry/timeout behavior in any way.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport):
        self._inner = inner

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        tracker = get_latency_tracker()
        if tracker is None:
            return await self._inner.handle_async_request(request)
        tracker.provider_attempt_started()
        try:
            response = await self._inner.handle_async_request(request)
        except Exception as error:
            tracker.provider_attempt_error(type(error).__name__)
            raise
        tracker.provider_attempt_headers()

        original_aiter = response.aiter_raw

        async def tracked_aiter(chunk_size=None):
            try:
                async for chunk in original_aiter(chunk_size):
                    yield chunk
                tracker.provider_attempt_stream_end()
            except Exception as error:
                tracker.provider_attempt_error(type(error).__name__)
                raise

        # httpx.Response is a regular class; the instance attribute shadows the
        # class method, so consumers calling response.aiter_raw() go through us.
        response.aiter_raw = tracked_aiter
        return response


def build_attempt_tracking_http_client() -> httpx.AsyncClient:
    """Build an httpx client with the same timeout the OpenAI SDK uses.

    The SDK's retry loop lives in openai's client layer (it re-issues the
    request, which flows through this transport again), so httpx itself must
    not add retries. We replicate the SDK default timeout exactly (from
    ``openai._constants``) so behavior is unchanged; the only difference is
    the attempt-tracking transport wrapper.
    """
    from openai._constants import DEFAULT_TIMEOUT

    return httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT,
        transport=AttemptTrackingTransport(httpx.AsyncHTTPTransport()),
    )


def describe_provider_config(client: Any) -> dict[str, Any]:
    """Read (never change) the provider client timeout/retry configuration."""
    timeout = getattr(client, "timeout", None)
    timeout_value = None
    if timeout is not None:
        try:
            timeout_value = timeout.total_timeout
        except AttributeError:
            timeout_value = None
    return {
        "provider_client_timeout_s": timeout_value,
        "provider_max_retries": getattr(client, "max_retries", None),
    }


# ---------------------------------------------------------------------------
# The tracker
# ---------------------------------------------------------------------------


@dataclass
class RequestLatencyTracker:
    websocket_send: WebSocketEmitter
    provider: str
    model: str
    request_id: str = field(
        default_factory=lambda: "chat-"
        + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:6]
    )
    client_user_send_ms: Optional[float] = None
    client_websocket_send_ms: Optional[float] = None
    request_origin: str = "user"
    started_ms: float = field(default_factory=monotonic_ms)
    received_epoch_ms: float = field(default_factory=lambda: time.time() * 1000.0)

    # --- accumulators for repeated phases (sums, mutually exclusive) ---
    context_build_ms: float = 0.0
    summary_ms: float = 0.0
    tool_ms: float = 0.0
    character_event_ms: float = 0.0
    tts_enqueue_ms: float = 0.0
    tts_synthesis_ms: float = 0.0
    tts_wait_ms: float = 0.0
    playback_wait_ms: float = 0.0
    history_save_ms: float = 0.0
    metadata_save_ms: float = 0.0
    character_state_save_ms: float = 0.0

    # --- phase boundaries (monotonic absolute ms) ---
    _marks: dict[str, float] = field(default_factory=dict)

    # --- provider state ---
    provider_attempts: list[dict[str, Any]] = field(default_factory=list)
    _provider_error: Optional[str] = None

    # --- request-level flags ---
    output_chars: int = 0
    message_count: int = 0
    estimated_input_tokens: int = 0
    summary_triggered: bool = False
    tool_used: bool = False
    interrupted: bool = False
    client_disconnected: bool = False
    internal_error: Optional[str] = None
    provider_call_expected: bool = True

    _provider_stream_completed: bool = False
    _first_token_emitted: bool = False

    @property
    def provider_request_ms(self) -> Optional[float]:
        """Backward-compatible absolute monotonic ms of the provider request."""
        return self._marks.get("provider_request_start")

    @property
    def provider_headers_ms(self) -> Optional[float]:
        return self._marks.get("provider_headers_received")

    @property
    def provider_first_token_ms(self) -> Optional[float]:
        return self._marks.get("provider_first_content_token")

    @property
    def provider_final_token_ms(self) -> Optional[float]:
        return self._marks.get("provider_stream_end")

    def __post_init__(self) -> None:
        self.mark("request_received")

    # -- phase marks -------------------------------------------------------

    def mark(self, name: str) -> None:
        """Record a named point on the monotonic timeline (first wins)."""
        self._marks.setdefault(name, monotonic_ms())

    def mark_at(self, name: str, absolute_ms: float) -> None:
        """Record a named point at an explicit absolute monotonic ms (tests)."""
        self._marks.setdefault(name, absolute_ms)

    def phase_duration(self, start_name: str, end_name: str) -> Optional[float]:
        start = self._marks.get(start_name)
        end = self._marks.get(end_name)
        if start is None or end is None:
            return None
        return max(0.0, end - start)

    # -- accumulation helpers ---------------------------------------------

    def add_time(self, attr: str, duration_ms: float) -> None:
        current = getattr(self, attr, 0.0) or 0.0
        setattr(self, attr, current + max(0.0, duration_ms))

    def add_context(self, duration_ms: float, selection: Any = None) -> None:
        self.add_time("context_build_ms", duration_ms)
        if selection is not None:
            stats = selection.stats
            self.message_count = stats.messages_after
            self.estimated_input_tokens = stats.estimated_input_tokens

    def add_summary(self, duration_ms: float, triggered: bool) -> None:
        self.add_time("summary_ms", duration_ms)
        self.summary_triggered = self.summary_triggered or triggered

    def add_tool(self, duration_ms: float) -> None:
        self.add_time("tool_ms", duration_ms)
        self.tool_used = True

    def add_tts_enqueue(self, duration_ms: float) -> None:
        self.add_time("tts_enqueue_ms", duration_ms)

    def add_tts_synthesis(self, duration_ms: float) -> None:
        self.add_time("tts_synthesis_ms", duration_ms)

    def add_tts_wait(self, duration_ms: float) -> None:
        self.add_time("tts_wait_ms", duration_ms)

    def add_playback_wait(self, duration_ms: float) -> None:
        self.add_time("playback_wait_ms", duration_ms)

    def add_history_save(self, duration_ms: float) -> None:
        self.add_time("history_save_ms", duration_ms)

    def add_metadata_save(self, duration_ms: float) -> None:
        self.add_time("metadata_save_ms", duration_ms)

    def add_character_state_save(self, duration_ms: float) -> None:
        self.add_time("character_state_save_ms", duration_ms)

    # -- events ------------------------------------------------------------

    async def emit(self, event: str, **values: Any) -> None:
        payload = {
            "type": "latency-event",
            "event": event,
            "request_id": self.request_id,
            **values,
        }
        await self.websocket_send(json.dumps(payload))

    async def provider_first_token(self) -> None:
        self.mark("provider_first_content_token")
        if not self._first_token_emitted:
            self._first_token_emitted = True
            await self.emit("first-token")

    def provider_first_chunk(self) -> None:
        self.mark("provider_first_chunk")

    # -- provider lifecycle (compat methods used by the LLM layer) ---------

    def provider_started(self) -> None:
        self.mark("provider_request_start")

    def provider_prepare_started(self) -> None:
        self.mark("provider_prepare_start")

    def provider_headers_received(self) -> None:
        self.mark("provider_headers_received")

    def provider_finished(self) -> None:
        """Provider call ended (success or error): mark the request end."""
        self.mark("provider_request_end")

    def provider_prepare_done(self) -> None:
        self.mark("provider_prepare_end")

    def provider_stream_completed(self, ms: Optional[float] = None) -> None:
        """Stream consumed to completion on the natural (non-error) path."""
        self._provider_stream_completed = True
        if ms is not None:
            self.mark_at("provider_stream_end", ms)
        else:
            self.mark("provider_stream_end")

    # -- attempt tracking --------------------------------------------------

    def provider_attempt_started(self) -> None:
        self.provider_attempts.append(
            {
                "attempt": len(self.provider_attempts) + 1,
                "phase": get_latency_phase(),
                "started_ms": monotonic_ms(),
                "headers_ms": None,
                "stream_end_ms": None,
                "error": None,
            }
        )

    def provider_attempt_headers(self) -> None:
        if not self.provider_attempts:
            return
        self.provider_attempts[-1]["headers_ms"] = monotonic_ms()

    def provider_attempt_stream_end(self) -> None:
        if not self.provider_attempts:
            return
        self.provider_attempts[-1]["stream_end_ms"] = monotonic_ms()

    def provider_attempt_error(self, error_type: str) -> None:
        if not self.provider_attempts:
            return
        attempt = self.provider_attempts[-1]
        attempt["error"] = error_type
        attempt["stream_end_ms"] = monotonic_ms()
        self._provider_error = error_type

    def record_provider_error(self, error_type: str) -> None:
        self._provider_error = error_type

    # -- request outcome ---------------------------------------------------

    def _compute_outcome(self) -> str:
        if self.interrupted:
            return "interrupted"
        if self.client_disconnected:
            return "client_disconnect"
        if self.internal_error:
            return "internal_error"
        started = "provider_request_start" in self._marks
        headers = "provider_headers_received" in self._marks
        attempts = len(self.provider_attempts)
        if started and not headers:
            if attempts > 1:
                return "provider_error"
            if self._provider_error:
                return "provider_error"
            return "provider_cancelled"
        if started and attempts > 1:
            return "retry"
        if started and not self._provider_stream_completed:
            if self._provider_error:
                return "provider_error"
            return "provider_cancelled"
        if not started:
            if self.tool_used:
                return "tool_only"
            if self.output_chars:
                return "success"
            return "empty_response"
        if not self.output_chars and not self.tool_used:
            return "empty_response"
        return "success"

    # -- gap / unattributed analysis --------------------------------------

    def _timeline_gaps(
        self, completed_ms: float
    ) -> tuple[float, Optional[str], Optional[str]]:
        """Largest gap between consecutive recorded phase boundaries."""
        points = [
            (ms, name) for name, ms in self._marks.items()
        ]
        points.append((completed_ms, "request_complete"))
        points.sort(key=lambda item: item[0])
        largest = 0.0
        from_name: Optional[str] = None
        to_name: Optional[str] = None
        for (t1, n1), (t2, n2) in zip(points, points[1:]):
            gap = t2 - t1
            if gap > largest:
                largest = gap
                from_name = _friendly_name(n1)
                to_name = _friendly_name(n2)
        return largest, from_name, to_name

    def _exclusive_segments(self, completed_ms: float) -> list[tuple[str, float]]:
        """Partition [request_received, complete] into non-overlapping spans."""
        received = self._marks.get("request_received", self.started_ms)
        agent_start = self._marks.get("agent_start")
        agent_end = self._marks.get("agent_end")
        tts_start = self._marks.get("tts_wait_start")
        tts_end = self._marks.get("tts_wait_end")
        playback_start = self._marks.get("playback_start")
        playback_end = self._marks.get("playback_end")

        segments: list[tuple[str, float]] = []

        def span(start: float, end: float) -> float:
            return max(0.0, end - start)

        if agent_start is not None:
            segments.append(("conversation_prep_ms", span(received, agent_start)))
        if agent_start is not None and agent_end is not None:
            segments.append(("agent_stream_ms", span(agent_start, agent_end)))
        # response_processing: the agent-side window between the agent stream
        # ending and the TTS/playback phases starting (sentence splitting,
        # expression handling, audio prep). It is a known phase, so it must be
        # part of known_pipeline rather than counting as unattributed.
        resp_tail = tts_start or playback_start
        if agent_end is not None and resp_tail is not None:
            segments.append(("response_processing_ms", span(agent_end, resp_tail)))
        if tts_start is not None and tts_end is not None:
            segments.append(("tts_wait_ms", span(tts_start, tts_end)))
        if playback_start is not None and playback_end is not None:
            segments.append(("playback_wait_ms", span(playback_start, playback_end)))
        tail_start = playback_end or tts_end or agent_end or received
        segments.append(("post_processing_ms", span(tail_start, completed_ms)))
        return segments

    # -- metrics -----------------------------------------------------------

    def metrics(self, completed_ms: Optional[float] = None) -> dict[str, Any]:
        completed = completed_ms if completed_ms is not None else monotonic_ms()
        total = max(0.0, completed - self.started_ms)

        provider_started = "provider_request_start" in self._marks
        headers_received = "provider_headers_received" in self._marks
        first_chunk = "provider_first_chunk" in self._marks
        first_content = "provider_first_content_token" in self._marks
        stream_end = self._provider_stream_completed

        request_to_headers = self.phase_duration(
            "provider_request_start", "provider_headers_received"
        )
        request_to_first_token = self.phase_duration(
            "provider_request_start", "provider_first_content_token"
        )
        generation = self.phase_duration(
            "provider_first_content_token", "provider_stream_end"
        )
        prepare_ms = self.phase_duration("provider_prepare_start", "provider_prepare_end")
        first_chunk_to_content = self.phase_duration(
            "provider_first_chunk", "provider_first_content_token"
        )

        provider_wallclock = self.phase_duration(
            "provider_request_start", "provider_stream_end"
        )

        estimated_output_tokens = max(0, (self.output_chars + 3) // 4)
        tokens_per_second = (
            estimated_output_tokens / (generation / 1000.0)
            if generation is not None and generation > 0 and estimated_output_tokens
            else None
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

        segments = self._exclusive_segments(completed)
        known_pipeline = sum(segment[1] for segment in segments)
        unattributed = max(0.0, total - known_pipeline)
        largest_gap, gap_from, gap_to = self._timeline_gaps(completed)

        provider_attempt_count = len(self.provider_attempts)
        chat_attempts = [
            attempt for attempt in self.provider_attempts
            if attempt.get("phase") == "chat"
        ]
        provider_retry_overhead = None
        if len(chat_attempts) > 1:
            first_start = chat_attempts[0]["started_ms"]
            last_end = (
                chat_attempts[-1].get("stream_end_ms")
                or chat_attempts[-1].get("headers_ms")
                or chat_attempts[-1]["started_ms"]
            )
            provider_retry_overhead = max(0.0, last_end - first_start)

        # TTS timing semantics (Phase 2.1): synthesis tasks run as concurrent
        # asyncio tasks, so the summed per-chunk durations are *cumulative
        # work* (tts_synthesis_ms, may exceed wall-clock) while tts_wait_ms is
        # the wall-clock blocking span of the gather that the request actually
        # waited on. Only wall-clock figures are labelled as totals.
        tts_blocking = self.tts_wait_ms
        tts_total = tts_blocking  # wall-clock blocking; never > total_response_ms
        audio_blocking = tts_blocking + self.playback_wait_ms  # serial phases
        response_processing = 0.0
        agent_end = self._marks.get("agent_end")
        if agent_end is not None:
            tts_start = self._marks.get("tts_wait_start", agent_end)
            response_processing = max(0.0, tts_start - agent_end)

        values: dict[str, Any] = {
            "request_id": self.request_id,
            "request_origin": self.request_origin,
            "provider": self.provider,
            "model": self.model,
            "request_outcome": self._compute_outcome(),
            "interrupted": self.interrupted,
            "client_disconnected": self.client_disconnected,
            "error_category": self.internal_error or self._provider_error,
            "frontend_user_send_to_backend_ms": _round(user_send_to_backend),
            "frontend_to_backend_ms": _round(frontend_to_backend),
            "conversation_prep_ms": _round(
                self.phase_duration("request_received", "agent_start")
            ),
            "agent_stream_ms": _round(
                self.phase_duration("agent_start", "agent_end")
            ),
            "provider_wallclock_ms": _round(provider_wallclock),
            "context_build_ms": round(self.context_build_ms, 2),
            "summary_ms": round(self.summary_ms, 2),
            "summary_triggered": self.summary_triggered,
            "tool_ms": round(self.tool_ms, 2),
            "tool_used": self.tool_used,
            "character_event_ms": round(self.character_event_ms, 2),
            "tts_enqueue_ms": round(self.tts_enqueue_ms, 2),
            "tts_synthesis_ms": round(self.tts_synthesis_ms, 2),
            "tts_wait_ms": round(self.tts_wait_ms, 2),
            "tts_blocking_ms": round(tts_blocking, 2),
            "playback_wait_ms": round(self.playback_wait_ms, 2),
            "tts_total_ms": round(tts_total, 2),
            "audio_blocking_ms": round(audio_blocking, 2),
            "history_save_ms": round(self.history_save_ms, 2),
            "metadata_save_ms": round(self.metadata_save_ms, 2),
            "character_state_save_ms": round(self.character_state_save_ms, 2),
            "response_processing_ms": round(response_processing, 2),
            "provider_call_expected": self.provider_call_expected,
            "provider_started": provider_started,
            "provider_headers_received": headers_received,
            "provider_first_chunk_received": first_chunk,
            "provider_first_token_received": first_content,
            "provider_stream_completed": stream_end,
            "provider_attempt_count": provider_attempt_count,
            "provider_prepare_ms": _round(prepare_ms),
            "ollama_request_to_headers_ms": _round(request_to_headers),
            "ollama_request_to_first_token_ms": _round(request_to_first_token),
            "provider_first_chunk_to_content_ms": _round(first_chunk_to_content),
            "ollama_generation_ms": _round(generation),
            "provider_retry_overhead_ms": _round(provider_retry_overhead),
            "output_tokens_per_second_estimate": _round(tokens_per_second),
            "message_count": self.message_count,
            "estimated_input_tokens": self.estimated_input_tokens,
            "known_pipeline_ms": round(known_pipeline, 2),
            "unattributed_ms": round(unattributed, 2),
            "largest_gap_ms": round(largest_gap, 2),
            "largest_gap_from": gap_from,
            "largest_gap_to": gap_to,
            "total_response_ms": round(total, 2),
        }
        values["bottleneck_hint"] = classify_bottleneck(values)
        return values

    # -- completion --------------------------------------------------------

    async def complete(self) -> dict[str, Any]:
        self.mark("request_complete")
        values = self.metrics()
        safe_values = {
            key: value for key, value in values.items() if value is not None
        }
        line = " ".join(
            f"{key}={value}" for key, value in safe_values.items()
        )
        logger.info("[LLM TRACE] " + line)

        for attempt in self.provider_attempts:
            started = attempt.get("started_ms")
            headers = attempt.get("headers_ms")
            stream_end = attempt.get("stream_end_ms")
            logger.info(
                "[LLM ATTEMPT] request_id={} attempt={} phase={} "
                "started_rel_ms={} headers_rel_ms={} stream_end_rel_ms={} "
                "error={}",
                self.request_id,
                attempt.get("attempt"),
                attempt.get("phase"),
                _round(started - self.started_ms) if started else None,
                _round(headers - self.started_ms) if headers else None,
                _round(stream_end - self.started_ms) if stream_end else None,
                attempt.get("error"),
            )

        numeric = {
            "ttft": float(values["ollama_request_to_first_token_ms"] or 0.0),
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
        try:
            await self.emit("response-complete", metrics=safe_values)
        except Exception as emit_error:
            # Instrumentation must never turn a completed request into an
            # error: the trace/outcome were already logged above. This is a
            # post-processing failure of the latency event itself, not of the
            # request (e.g. the client disconnected right at completion).
            logger.warning(
                "[LLM TRACE] response-complete event send failed: type={} "
                "(outcome already logged; this is a post-processing failure)",
                type(emit_error).__name__,
            )
        return values


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)
