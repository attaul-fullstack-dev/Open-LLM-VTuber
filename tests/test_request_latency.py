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

    def test_c_hidden_gap_is_reported_as_unattributed(self):
        tracker, t0 = _tracker()
        tracker.mark_at("agent_start", t0 + 1)
        tracker.mark_at("provider_request_start", t0 + 2)
        tracker.mark_at("provider_headers_received", t0 + 3)
        tracker.mark_at("provider_first_content_token", t0 + 4)
        tracker.provider_stream_completed(ms=t0 + 1000)
        # 2.5 seconds of unexplained time between provider end and agent end
        tracker.mark_at("agent_end", t0 + 3500)
        tracker.mark_at("request_complete", t0 + 3505)
        tracker.output_chars = 100

        values = tracker.metrics(completed_ms=t0 + 3505)
        self.assertGreaterEqual(values["largest_gap_ms"], 2500)
        self.assertEqual(values["largest_gap_from"], "provider_stream")
        self.assertEqual(values["largest_gap_to"], "agent")
        self.assertEqual(values["bottleneck_hint"], "unattributed")

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

        values = tracker.metrics(completed_ms=t0 + 1000)
        self.assertEqual(values["tts_enqueue_ms"], 1.2)
        self.assertEqual(values["tts_synthesis_ms"], 850.0)
        self.assertEqual(values["tts_wait_ms"], 40.0)
        self.assertAlmostEqual(values["tts_total_ms"], 891.2)

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


if __name__ == "__main__":
    unittest.main()
