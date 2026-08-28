import asyncio
import unittest

from src.open_llm_vtuber.request_latency import (
    RequestLatencyTracker,
    classify_bottleneck,
)


def _tracker(t0: float = 10_000.0, send=None):
    async def _noop(_payload):
        return None

    tracker = RequestLatencyTracker(
        websocket_send=send or _noop,
        provider="ollama_cloud",
        model="test-model",
    )
    # Pin a deterministic start so every mark is expressible as t0 + offset.
    tracker.started_ms = t0
    tracker._marks.clear()
    tracker.mark_at("request_received", t0)
    return tracker, t0


class RequestLatencyPhase2Tests(unittest.IsolatedAsyncioTestCase):
    # ---- TEST A: normal request, all phases fast -------------------------

    def test_a_normal_request_has_small_unattributed(self):
        tracker, t0 = _tracker()
        tracker.mark_at("agent_start", t0 + 5)
        tracker.mark_at("context_build_start", t0 + 6)
        tracker.mark_at("context_build_end", t0 + 11)
        tracker.mark_at("provider_request_start", t0 + 11)
        tracker.mark_at("provider_headers_received", t0 + 12)
        tracker.mark_at("provider_first_chunk", t0 + 12)
        tracker.mark_at("provider_first_content_token", t0 + 511)
        tracker.provider_stream_completed(ms=t0 + 811)
        tracker.mark_at("agent_end", t0 + 812)
        tracker.mark_at("tts_wait_start", t0 + 813)
        tracker.mark_at("tts_wait_end", t0 + 816)
        tracker.mark_at("playback_start", t0 + 816)
        tracker.mark_at("playback_end", t0 + 818)
        tracker.output_chars = 200
        tracker.add_context(5)
        tracker.add_tts_enqueue(1)
        tracker.add_tts_synthesis(3)
        tracker.add_tts_wait(3)
        tracker.add_playback_wait(2)

        values = tracker.metrics(completed_ms=t0 + 820)

        self.assertEqual(values["total_response_ms"], 820)
        self.assertEqual(values["ollama_request_to_first_token_ms"], 500)
        self.assertEqual(values["ollama_generation_ms"], 300)
        self.assertLess(values["unattributed_ms"], 50)
        self.assertEqual(values["request_outcome"], "success")
        self.assertNotEqual(values["bottleneck_hint"], "unattributed")

    # ---- TEST B: provider TTFT spike -------------------------------------

    def test_b_provider_ttft_spike_is_flagged(self):
        tracker, t0 = _tracker()
        tracker.mark_at("agent_start", t0 + 2)
        tracker.mark_at("provider_request_start", t0 + 3)
        tracker.mark_at("provider_headers_received", t0 + 10)
        tracker.mark_at("provider_first_chunk", t0 + 10)
        tracker.mark_at("provider_first_content_token", t0 + 8003)
        tracker.provider_stream_completed(ms=t0 + 8503)
        tracker.mark_at("agent_end", t0 + 8504)
        tracker.output_chars = 150

        values = tracker.metrics(completed_ms=t0 + 8510)
        self.assertEqual(values["ollama_request_to_first_token_ms"], 8000)
        self.assertEqual(values["bottleneck_hint"], "provider_ttft")

    # ---- TEST C: hidden 2.5s gap after provider --------------------------
    #
    # Phase 2.1: a corridor between provider_stream_end and agent_end sits
    # inside the agent_stream phase (agent-side response processing), so it is
    # NOT "unattributed" -- it is mapped to response_processing. The gap is
    # still reported in largest_gap_* for humans, but the hint follows the
    # phase that owns the time.

    def test_c_gap_after_provider_is_response_processing_not_unattributed(self):
        tracker, t0 = _tracker()
        tracker.mark_at("agent_start", t0 + 1)
        tracker.mark_at("provider_request_start", t0 + 2)
        tracker.mark_at("provider_headers_received", t0 + 3)
        tracker.mark_at("provider_first_content_token", t0 + 4)
        tracker.provider_stream_completed(ms=t0 + 1000)
        # 2.5 seconds of agent-side processing between provider end and agent end
        tracker.mark_at("agent_end", t0 + 3500)
        tracker.mark_at("request_complete", t0 + 3505)
        tracker.output_chars = 100

        values = tracker.metrics(completed_ms=t0 + 3505)
        self.assertGreaterEqual(values["largest_gap_ms"], 2500)
        self.assertEqual(values["largest_gap_from"], "provider_stream")
        self.assertEqual(values["largest_gap_to"], "agent")
        # The gap is inside agent_stream minus the provider span.
        self.assertEqual(values["bottleneck_hint"], "response_processing")

    # ---- TEST D: retry ----------------------------------------------------

    def test_d_retry_is_visible_and_attributed(self):
        tracker, t0 = _tracker()
        tracker.provider_attempts = [
            {
                "attempt": 1,
                "phase": "chat",
                "started_ms": t0 + 1000,
                "headers_ms": None,
                "stream_end_ms": None,
                "error": "APIConnectionError",
            },
            {
                "attempt": 2,
                "phase": "chat",
                "started_ms": t0 + 11000,
                "headers_ms": t0 + 11001,
                "stream_end_ms": t0 + 11501,
                "error": None,
            },
        ]
        tracker.mark_at("agent_start", t0 + 1)
        tracker.mark_at("provider_request_start", t0 + 1000)
        tracker.mark_at("provider_headers_received", t0 + 11001)
        tracker.mark_at("provider_first_content_token", t0 + 11002)
        tracker.provider_stream_completed(ms=t0 + 11501)
        tracker.mark_at("agent_end", t0 + 11502)
        tracker.output_chars = 60

        values = tracker.metrics(completed_ms=t0 + 11505)
        self.assertEqual(values["provider_attempt_count"], 2)
        self.assertEqual(values["request_outcome"], "retry")
        self.assertEqual(values["bottleneck_hint"], "provider_retry")
        # The retry overhead must be a positive, visible number.
        self.assertGreater(values["provider_retry_overhead_ms"], 10000)

    # ---- TEST E: provider never started ----------------------------------

    def test_e_provider_never_started_uses_none_not_zero(self):
        tracker, t0 = _tracker()
        tracker.mark_at("agent_start", t0 + 1000)
        tracker.mark_at("agent_end", t0 + 20000)
        tracker.internal_error = "ContextBudgetExceeded"

        values = tracker.metrics(completed_ms=t0 + 20000)
        # Never-occurred provider values must be None, never 0.0.
        self.assertIsNone(values["ollama_request_to_first_token_ms"])
        self.assertIsNone(values["ollama_request_to_headers_ms"])
        self.assertIsNone(values["ollama_generation_ms"])
        self.assertFalse(values["provider_started"])
        self.assertEqual(values["request_outcome"], "internal_error")
        # The delay happened before the provider: agent stream is the largest
        # phase, and it is called out.
        self.assertEqual(values["bottleneck_hint"], "response_processing")

    # ---- TEST F: client disconnect ----------------------------------------

    def test_f_client_disconnect_outcome(self):
        tracker, t0 = _tracker()
        tracker.client_disconnected = True
        tracker.mark_at("agent_start", t0 + 1)
        tracker.mark_at("provider_request_start", t0 + 2)
        tracker.mark_at("agent_end", t0 + 3000)

        values = tracker.metrics(completed_ms=t0 + 3000)
        self.assertEqual(values["request_outcome"], "client_disconnect")
        self.assertEqual(values["bottleneck_hint"], "client_disconnect")

    # ---- TEST G: interrupt ------------------------------------------------

    def test_g_interrupt_outcome(self):
        tracker, t0 = _tracker()
        tracker.interrupted = True
        tracker.mark_at("agent_start", t0 + 1)
        tracker.mark_at("provider_request_start", t0 + 2)
        tracker.mark_at("provider_headers_received", t0 + 3)
        tracker.mark_at("provider_first_content_token", t0 + 400)
        tracker.mark_at("agent_end", t0 + 450)

        values = tracker.metrics(completed_ms=t0 + 450)
        self.assertEqual(values["request_outcome"], "interrupted")
        self.assertTrue(values["interrupted"])
        # Provider never completed its stream.
        self.assertFalse(values["provider_stream_completed"])

    # ---- TEST H: TTS async: enqueue vs synthesis stay separate -----------

    def test_h_tts_enqueue_and_synthesis_are_not_mixed(self):
        tracker, t0 = _tracker()
        tracker.add_tts_enqueue(1.2)
        tracker.add_tts_synthesis(850.0)
        tracker.add_tts_wait(40.0)
        tracker.add_playback_wait(10.0)

        values = tracker.metrics(completed_ms=t0 + 1000)
        self.assertEqual(values["tts_enqueue_ms"], 1.2)
        self.assertEqual(values["tts_synthesis_ms"], 850.0)
        self.assertEqual(values["tts_wait_ms"], 40.0)
        # Phase 2.1: tts_total_ms is the wall-clock blocking time (the gather
        # span), NOT the sum of overlapping async synthesis work.
        self.assertEqual(values["tts_blocking_ms"], 40.0)
        self.assertEqual(values["tts_total_ms"], 40.0)
        self.assertEqual(values["audio_blocking_ms"], 50.0)
        # Cumulative work may exceed wall-clock; the wall-clock total never
        # exceeds the total request duration.
        self.assertGreater(values["tts_synthesis_ms"], values["tts_total_ms"])
        self.assertLessEqual(values["tts_total_ms"], values["total_response_ms"])

    # ---- Phase 2.1: TTS wall-clock invariant ------------------------------

    def test_tts_total_never_exceeds_total_response(self):
        # Simulate two overlapping synthesis tasks: 3s + 3s of cumulative
        # work inside a ~3.5s wall-clock request.
        tracker, t0 = _tracker()
        tracker.add_tts_synthesis(3000.0)
        tracker.add_tts_synthesis(3000.0)
        tracker.add_tts_wait(3500.0)
        tracker.add_playback_wait(400.0)

        values = tracker.metrics(completed_ms=t0 + 4000)
        self.assertEqual(values["tts_synthesis_ms"], 6000.0)  # cumulative work
        self.assertEqual(values["tts_total_ms"], 3500.0)      # wall-clock block
        self.assertEqual(values["audio_blocking_ms"], 3900.0)
        self.assertLessEqual(values["tts_total_ms"], values["total_response_ms"])
        self.assertLessEqual(values["audio_blocking_ms"], values["total_response_ms"])

    # ---- Phase 2.1 classifier tests --------------------------------------

    def test_classifier_tts_dominant_with_tiny_unattributed_is_tts(self):
        tracker, t0 = _tracker()
        tracker.mark_at("agent_start", t0 + 5)
        tracker.mark_at("provider_request_start", t0 + 6)
        tracker.mark_at("provider_headers_received", t0 + 7)
        tracker.mark_at("provider_first_content_token", t0 + 506)
        tracker.provider_stream_completed(ms=t0 + 906)
        tracker.mark_at("agent_end", t0 + 907)
        tracker.mark_at("tts_wait_start", t0 + 908)
        tracker.mark_at("tts_wait_end", t0 + 3408)
        tracker.mark_at("playback_start", t0 + 3408)
        tracker.mark_at("playback_end", t0 + 4008)
        tracker.output_chars = 100
        tracker.add_tts_wait(2500.0)
        tracker.add_playback_wait(600.0)

        values = tracker.metrics(completed_ms=t0 + 4008)
        # The live trace shape: unattributed ~0.2ms but a 2.5s tts_wait gap.
        self.assertLess(values["unattributed_ms"], 50)
        self.assertGreaterEqual(values["largest_gap_ms"], 2500)
        self.assertEqual(values["largest_gap_from"], "tts_wait")
        self.assertEqual(values["largest_gap_to"], "tts_wait")
        self.assertEqual(values["bottleneck_hint"], "tts")

    def test_classifier_real_unattributed_wins(self):
        # Spec TEST B: 1500ms mapped to known phases, 2500ms genuinely
        # unexplained -> the hint must be unattributed, not a known phase.
        self.assertEqual(
            classify_bottleneck(
                {
                    "context_build_ms": 500,
                    "agent_stream_ms": 1000,
                    "provider_wallclock_ms": 900,
                    "ollama_request_to_first_token_ms": 500,
                    "ollama_generation_ms": 400,
                    "tts_total_ms": 0,
                    "playback_wait_ms": 0,
                    "unattributed_ms": 2500,
                    "total_response_ms": 4000,
                    "provider_started": True,
                    "provider_headers_received": True,
                    "provider_attempt_count": 1,
                }
            ),
            "unattributed",
        )

    def test_response_processing_window_is_known_not_unattributed(self):
        # The agent_end -> tts_wait_start window is the measured
        # response_processing phase; it must be part of known_pipeline and
        # never surface as unattributed time.
        tracker, t0 = _tracker()
        tracker.mark_at("agent_start", t0 + 1)
        tracker.mark_at("agent_end", t0 + 1000)
        tracker.mark_at("tts_wait_start", t0 + 3500)
        tracker.mark_at("tts_wait_end", t0 + 3501)
        tracker.add_tts_wait(1.0)
        tracker.output_chars = 100

        values = tracker.metrics(completed_ms=t0 + 3501)
        self.assertEqual(values["response_processing_ms"], 2500.0)
        self.assertLess(values["unattributed_ms"], 50)
        self.assertEqual(values["bottleneck_hint"], "response_processing")

    def test_classifier_provider_ttft_beats_tts(self):
        tracker, t0 = _tracker()
        tracker.mark_at("agent_start", t0 + 2)
        tracker.mark_at("provider_request_start", t0 + 3)
        tracker.mark_at("provider_headers_received", t0 + 4)
        tracker.mark_at("provider_first_content_token", t0 + 8003)
        tracker.provider_stream_completed(ms=t0 + 8403)
        tracker.mark_at("agent_end", t0 + 8404)
        tracker.mark_at("tts_wait_start", t0 + 8405)
        tracker.mark_at("tts_wait_end", t0 + 9405)
        tracker.output_chars = 100
        tracker.add_tts_wait(1000.0)

        values = tracker.metrics(completed_ms=t0 + 9410)
        self.assertEqual(values["bottleneck_hint"], "provider_ttft")

    def test_classifier_playback_dominant_maps_to_tts(self):
        # playback_wait is the browser-side tail of the same audio pipeline;
        # per the existing taxonomy it folds into the "tts" category.
        tracker, t0 = _tracker()
        tracker.mark_at("agent_start", t0 + 1)
        tracker.mark_at("provider_request_start", t0 + 2)
        tracker.mark_at("provider_headers_received", t0 + 3)
        tracker.mark_at("provider_first_content_token", t0 + 203)
        tracker.provider_stream_completed(ms=t0 + 403)
        tracker.mark_at("agent_end", t0 + 404)
        tracker.mark_at("tts_wait_start", t0 + 405)
        tracker.mark_at("tts_wait_end", t0 + 505)
        tracker.mark_at("playback_start", t0 + 505)
        tracker.mark_at("playback_end", t0 + 5505)
        tracker.output_chars = 100
        tracker.add_tts_wait(100.0)
        tracker.add_playback_wait(5000.0)

        values = tracker.metrics(completed_ms=t0 + 5510)
        self.assertEqual(values["bottleneck_hint"], "tts")

    # ---- TEST I: history persistence latency ------------------------------

    def test_i_history_save_latency_is_measured(self):
        tracker, t0 = _tracker()
        tracker.add_history_save(1200.0)
        tracker.add_metadata_save(30.0)
        tracker.add_character_state_save(5.0)

        values = tracker.metrics(completed_ms=t0 + 2000)
        self.assertEqual(values["history_save_ms"], 1200.0)
        self.assertEqual(values["metadata_save_ms"], 30.0)
        self.assertEqual(values["character_state_save_ms"], 5.0)

    # ---- TEST J: streaming regression: first token emitted immediately ----

    async def test_j_first_token_event_is_emitted_without_buffering(self):
        sent = []

        async def send(payload):
            sent.append(payload)

        tracker, t0 = _tracker(t0=10_000.0, send=send)
        tracker.mark_at("provider_request_start", t0 + 10)
        tracker.mark_at("provider_headers_received", t0 + 11)
        tracker.mark_at("provider_first_content_token", t0 + 900)

        await tracker.provider_first_token()
        # The first-token event must be emitted at once, not at completion.
        self.assertTrue(any('"event": "first-token"' in item for item in sent))
        # A second call must not duplicate the event.
        await tracker.provider_first_token()
        first_token_events = [
            item for item in sent if '"event": "first-token"' in item
        ]
        self.assertEqual(len(first_token_events), 1)

    # ---- backward-compatible behavior ------------------------------------

    def test_summary_and_tool_latency_are_reported_separately(self):
        async def send(_payload):
            return None

        tracker = RequestLatencyTracker(send, "ollama_cloud", "test-model")
        tracker.add_context(50)
        tracker.add_summary(3000, triggered=True)
        tracker.add_tool(700)
        values = tracker.metrics(completed_ms=tracker.started_ms + 4000)

        self.assertEqual(values["context_build_ms"], 50)
        self.assertEqual(values["summary_ms"], 3000)
        self.assertTrue(values["summary_triggered"])
        self.assertEqual(values["tool_ms"], 700)
        self.assertTrue(values["tool_used"])

    def test_classifier_distinguishes_context_from_provider(self):
        self.assertEqual(
            classify_bottleneck(
                {
                    "context_build_ms": 4000,
                    "summary_ms": 0,
                    "tool_ms": 0,
                    "ollama_request_to_first_token_ms": 300,
                    "provider_started": True,
                    "provider_headers_received": True,
                    "provider_attempt_count": 1,
                    "total_response_ms": 4500,
                    "largest_gap_ms": 0,
                }
            ),
            "context",
        )


class WebSocketSendLockTests(unittest.IsolatedAsyncioTestCase):
    """Regression for the AssertionError root cause (Phase 2.1).

    Live traces showed ``Error in conversation chain:  `` (bare assert, empty
    message) when the TTS payload sender task and the main flow wrote to the
    same WebSocket at the same time. The websockets legacy protocol raises
    ``AssertionError`` when two coroutines drain a paused transport
    concurrently (``_drain_helper``); serializing sends removes the race.
    """

    async def test_library_drain_helper_asserts_on_concurrent_drains(self):
        # Direct, deterministic proof of the library-level race: two
        # concurrent _drain_helper calls while the transport is paused.
        from websockets.legacy.server import WebSocketServerProtocol

        proto = WebSocketServerProtocol(
            ws_handler=lambda ws: None, ws_server=None
        )
        proto._paused = True
        proto._drain_waiter = None

        async def drainer():
            await proto._drain_helper()

        first = asyncio.create_task(drainer())
        await asyncio.sleep(0.02)  # first task now owns _drain_waiter
        second = asyncio.create_task(drainer())
        with self.assertRaises(AssertionError):
            await second
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first

    async def test_locked_send_text_serializes_concurrent_senders(self):
        from src.open_llm_vtuber.websocket_handler import (
            create_locked_send_text,
        )

        class RacingWebSocket:
            """Fake that raises when two sends overlap (like _drain_helper)."""

            def __init__(self):
                self._in_flight = False
                self.sent = []

            async def send_text(self, message: str) -> None:
                if self._in_flight:
                    raise AssertionError("")
                self._in_flight = True
                try:
                    await asyncio.sleep(0.02)  # hold the "transport write"
                    self.sent.append(message)
                finally:
                    self._in_flight = False

        # Unlocked concurrent sends reproduce the live AssertionError.
        racing = RacingWebSocket()
        results = await asyncio.gather(
            racing.send_text("a"),
            racing.send_text("b"),
            return_exceptions=True,
        )
        self.assertTrue(
            any(isinstance(result, AssertionError) for result in results)
        )

        # The locked wrapper serializes the same workload without errors.
        locked_ws = RacingWebSocket()
        locked_send = create_locked_send_text(locked_ws)
        results = await asyncio.gather(
            locked_send("a"),
            locked_send("b"),
            locked_send("c"),
            return_exceptions=True,
        )
        self.assertTrue(
            all(not isinstance(result, BaseException) for result in results)
        )
        self.assertEqual(sorted(locked_ws.sent), ["a", "b", "c"])

    async def test_complete_emit_failure_does_not_raise(self):
        # If the response-complete latency event cannot be sent (e.g. client
        # disconnected at completion), complete() must log and return, not
        # propagate the error into the request's finalization.
        async def failing_send(_payload):
            raise AssertionError("")

        tracker, t0 = _tracker(send=failing_send)
        tracker.mark_at("agent_start", t0 + 1)
        tracker.mark_at("agent_end", t0 + 100)
        tracker.output_chars = 50

        values = await tracker.complete()
        self.assertEqual(values["request_outcome"], "success")


if __name__ == "__main__":
    unittest.main()
