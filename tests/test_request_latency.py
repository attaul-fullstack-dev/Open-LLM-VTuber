import unittest

from src.open_llm_vtuber.request_latency import (
    RequestLatencyTracker,
    classify_bottleneck,
)


class RequestLatencyTests(unittest.IsolatedAsyncioTestCase):
    def test_provider_ttft_bottleneck_is_distinct_from_backend(self):
        self.assertEqual(
            classify_bottleneck(
                {
                    "context_build_ms": 50,
                    "summary_ms": 0,
                    "tool_ms": 0,
                    "ollama_request_to_first_token_ms": 5000,
                }
            ),
            "provider_or_model_ttft",
        )

    def test_backend_context_bottleneck_is_detected(self):
        self.assertEqual(
            classify_bottleneck(
                {
                    "context_build_ms": 4000,
                    "summary_ms": 0,
                    "tool_ms": 0,
                    "ollama_request_to_first_token_ms": 300,
                }
            ),
            "backend_context",
        )

    async def test_first_token_event_and_generation_are_separate(self):
        sent = []

        async def send(payload):
            sent.append(payload)

        tracker = RequestLatencyTracker(
            websocket_send=send,
            provider="ollama_cloud",
            model="test-model",
        )
        tracker.provider_request_ms = 1000
        tracker.provider_headers_ms = 1100
        tracker.provider_first_token_ms = None
        await tracker.provider_first_token()
        tracker.provider_final_token_ms = tracker.provider_first_token_ms + 1200
        tracker.output_chars = 400
        values = tracker.metrics(completed_ms=tracker.started_ms + 2000)

        self.assertTrue(any('"event": "first-token"' in item for item in sent))
        self.assertGreater(values["ollama_request_to_first_token_ms"], 0)
        self.assertEqual(values["ollama_generation_ms"], 1200)
        self.assertGreater(values["output_tokens_per_second_estimate"], 0)

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


if __name__ == "__main__":
    unittest.main()
