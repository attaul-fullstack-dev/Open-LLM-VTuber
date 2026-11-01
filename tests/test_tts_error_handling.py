"""Unit tests for TTS failure detection and structured error handling.

Tests:
1. ElevenLabs ApiError quota_exceeded detection and classification.
2. Distinct classification for authentication, permission, not found, and rate limits.
3. Verification that TTSManager._process_tts creates explicit failure payloads
   carrying tts_status='failed' and structured tts_error metadata, while
   preserving the user/assistant display text.
4. Verification that no automatic retry is attempted on quota_exceeded.
5. Verification that successful audio generation path remains completely intact.
6. Verification that finalize_conversation_turn fails closed and never claims
   successful playback when audio is null / failed.
"""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.open_llm_vtuber.conversations import conversation_utils
from src.open_llm_vtuber.conversations.conversation_utils import (
    finalize_conversation_turn,
)
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager
from src.open_llm_vtuber.message_handler import MessageHandler
from src.open_llm_vtuber.request_latency import (
    RequestLatencyTracker,
    reset_latency_tracker,
    set_latency_tracker,
)
from src.open_llm_vtuber.agent.output_types import DisplayText
from src.open_llm_vtuber.tts.elevenlabs_tts import TTSEngine as ElevenLabsTTSEngine
from src.open_llm_vtuber.tts.tts_interface import (
    TTSInterface,
    TTSError,
    TTSQuotaExceededError,
)
from src.open_llm_vtuber.utils.stream_audio import prepare_audio_payload


class _FakeApiError(Exception):
    """Simulates elevenlabs.core.api_error.ApiError."""

    def __init__(self, status_code: int, body: dict):
        super().__init__(f"status_code: {status_code}, body: {body}")
        self.status_code = status_code
        self.body = body


class TTSErrorClassificationTest(unittest.TestCase):
    """Tests classification of ElevenLabs exceptions in generate_audio."""

    def setUp(self):
        # Initialize TTSEngine with mocked client
        with patch("src.open_llm_vtuber.tts.elevenlabs_tts.ElevenLabs") as mock_eleven:
            self.mock_client = MagicMock()
            mock_eleven.return_value = self.mock_client
            self.engine = ElevenLabsTTSEngine(
                api_key="fake-test-key",
                voice_id="obflpiaPKugbCJ9NMBBv",
                model_id="eleven_v3_conversational",
            )

    def test_quota_exceeded_error_classification(self):
        """ElevenLabs HTTP 401 with code quota_exceeded must raise TTSQuotaExceededError."""
        quota_err = _FakeApiError(
            status_code=401,
            body={
                "detail": {
                    "type": "invalid_request",
                    "code": "quota_exceeded",
                    "message": "This request exceeds your quota of 10000. You have 0 credits remaining.",
                    "status": "quota_exceeded",
                    "request_id": "req-quota-123",
                }
            },
        )
        self.mock_client.text_to_speech.convert.side_effect = quota_err

        with self.assertRaises(TTSQuotaExceededError) as ctx:
            self.engine.generate_audio("Halo dunia")

        err = ctx.exception
        self.assertEqual(err.error_code, "quota_exceeded")
        self.assertEqual(err.error_category, "quota_exceeded")
        self.assertEqual(err.http_status, 401)
        self.assertFalse(err.retryable)
        self.assertIn("exceeds your quota", str(err))

        err_dict = err.to_dict(provider="elevenlabs_tts")
        self.assertEqual(err_dict["provider"], "elevenlabs_tts")
        self.assertEqual(err_dict["code"], "quota_exceeded")
        self.assertEqual(err_dict["category"], "quota_exceeded")
        self.assertEqual(err_dict["http_status"], 401)
        self.assertFalse(err_dict["retryable"])

    def test_authentication_error_classification(self):
        """ElevenLabs HTTP 401 with invalid_api_key must be classified as authentication."""
        auth_err = _FakeApiError(
            status_code=401,
            body={
                "detail": {
                    "status": "invalid_api_key",
                    "message": "Invalid API key provided.",
                }
            },
        )
        self.mock_client.text_to_speech.convert.side_effect = auth_err

        with self.assertRaises(TTSError) as ctx:
            self.engine.generate_audio("Halo")

        err = ctx.exception
        self.assertEqual(err.error_code, "invalid_api_key")
        self.assertEqual(err.error_category, "authentication")
        self.assertEqual(err.http_status, 401)
        self.assertFalse(err.retryable)

    def test_permission_denied_classification(self):
        """ElevenLabs HTTP 403 must be classified as permission error."""
        perm_err = _FakeApiError(
            status_code=403,
            body={"detail": {"message": "Voice access forbidden for this tier."}},
        )
        self.mock_client.text_to_speech.convert.side_effect = perm_err

        with self.assertRaises(TTSError) as ctx:
            self.engine.generate_audio("Halo")

        err = ctx.exception
        self.assertEqual(err.error_category, "permission")
        self.assertEqual(err.http_status, 403)
        self.assertFalse(err.retryable)

    def test_voice_not_found_classification(self):
        """ElevenLabs HTTP 404 must be classified as not_found error."""
        not_found_err = _FakeApiError(
            status_code=404,
            body={"detail": {"message": "Voice ID not found."}},
        )
        self.mock_client.text_to_speech.convert.side_effect = not_found_err

        with self.assertRaises(TTSError) as ctx:
            self.engine.generate_audio("Halo")

        err = ctx.exception
        self.assertEqual(err.error_category, "not_found")
        self.assertEqual(err.http_status, 404)
        self.assertFalse(err.retryable)

    def test_rate_limited_classification(self):
        """ElevenLabs HTTP 429 must be classified as rate_limited and non-auto-retryable."""
        rate_err = _FakeApiError(
            status_code=429,
            body={"detail": {"message": "Too many requests."}},
        )
        self.mock_client.text_to_speech.convert.side_effect = rate_err

        with self.assertRaises(TTSError) as ctx:
            self.engine.generate_audio("Halo")

        err = ctx.exception
        self.assertEqual(err.error_category, "rate_limited")
        self.assertEqual(err.http_status, 429)
        self.assertFalse(err.retryable)


class TTSManagerFailureHandlingTest(unittest.IsolatedAsyncioTestCase):
    """Tests TTSManager behavior when TTS fails due to quota or other errors."""

    def setUp(self):
        self._token = set_latency_tracker(None)

    def tearDown(self):
        if hasattr(self, "_token"):
            reset_latency_tracker(self._token)

    async def test_process_tts_quota_failure_emits_structured_failure_payload(self):
        """When TTS raises TTSQuotaExceededError, a silent payload with tts_status='failed'
        and structured tts_error metadata must be enqueued, and text must be preserved.
        """
        tts_manager = TTSTaskManager()
        display = DisplayText(text="Halo, apa kabar?", name="Mili")

        mock_engine = MagicMock(spec=TTSInterface)
        mock_engine.__class__.__module__ = "src.open_llm_vtuber.tts.elevenlabs_tts"

        call_count = 0

        async def fake_async_generate(text, file_name_no_ext=None):
            nonlocal call_count
            call_count += 1
            raise TTSQuotaExceededError(
                "This request exceeds your quota of 10000. You have 0 credits remaining.",
                http_status=401,
            )

        mock_engine.async_generate_audio = AsyncMock(side_effect=fake_async_generate)

        # Call _process_tts directly
        await tts_manager._process_tts(
            tts_text="Halo, apa kabar?",
            display_text=display,
            actions=None,
            live2d_model=None,
            tts_engine=mock_engine,
            sequence_number=0,
            synthesize_audio=True,
            history_text="Halo, apa kabar?",
        )

        # Verify no automatic retry occurred (called exactly once)
        self.assertEqual(call_count, 1, "TTS must not be retried automatically on quota_exceeded")

        # Verify payload queued in _payload_queue
        self.assertFalse(tts_manager._payload_queue.empty())
        item = await tts_manager._payload_queue.get()
        payload, seq, hist_text, before_send, after_send = item

        self.assertEqual(seq, 0)
        self.assertEqual(hist_text, "Halo, apa kabar?")
        self.assertEqual(payload.get("type"), "audio")
        self.assertIsNone(payload.get("audio"), "Audio must be null when TTS fails")
        self.assertEqual(payload.get("display_text", {}).get("text"), "Halo, apa kabar?")
        self.assertEqual(payload.get("tts_status"), "failed")

        tts_error = payload.get("tts_error")
        self.assertIsInstance(tts_error, dict)
        self.assertEqual(tts_error.get("code"), "quota_exceeded")
        self.assertEqual(tts_error.get("category"), "quota_exceeded")
        self.assertEqual(tts_error.get("http_status"), 401)
        self.assertFalse(tts_error.get("retryable"))
        self.assertEqual(tts_error.get("provider"), "elevenlabs_tts")

    async def test_process_tts_success_preserves_audio_bytes(self):
        """When TTS succeeds, audio must be populated and tts_status must not be failed."""
        tts_manager = TTSTaskManager()
        display = DisplayText(text="Sukses", name="Mili")

        # Create a tiny valid non-silent wav file for testing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # 44-byte minimal PCM wav header + non-zero data
            import struct
            num_samples = 800
            samples = [1000] * num_samples
            data_bytes = struct.pack(f"<{num_samples}h", *samples)
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                36 + len(data_bytes),
                b"WAVE",
                b"fmt ",
                16,
                1,  # PCM
                1,  # mono
                16000,
                32000,
                2,
                16,
                b"data",
                len(data_bytes),
            )
            f.write(header + data_bytes)
            temp_path = f.name

        try:
            mock_engine = MagicMock(spec=TTSInterface)
            mock_engine.__class__.__module__ = "src.open_llm_vtuber.tts.elevenlabs_tts"
            mock_engine.async_generate_audio = AsyncMock(return_value=temp_path)

            await tts_manager._process_tts(
                tts_text="Sukses",
                display_text=display,
                actions=None,
                live2d_model=None,
                tts_engine=mock_engine,
                sequence_number=1,
                synthesize_audio=True,
                history_text="Sukses",
            )

            self.assertFalse(tts_manager._payload_queue.empty())
            item = await tts_manager._payload_queue.get()
            payload, seq, hist_text, _, _ = item

            self.assertEqual(seq, 1)
            self.assertEqual(payload.get("type"), "audio")
            self.assertIsNotNone(payload.get("audio"))
            self.assertNotEqual(payload.get("audio"), "")
            self.assertIsNone(payload.get("tts_status"))
            self.assertIsNone(payload.get("tts_error"))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class PlaybackFinalizeFailureTest(unittest.IsolatedAsyncioTestCase):
    """Verifies turn finalization fails closed when audio outcome is failed."""

    async def test_turn_finalize_fails_closed_on_quota_failure(self):
        class _Socket:
            def __init__(self):
                self.sent = []

            async def send_text(self, payload: str):
                self.sent.append(json.loads(payload))

        socket = _Socket()
        tracker = RequestLatencyTracker(
            websocket_send=socket.send_text,
            provider="fake",
            model="fake",
            request_id="turn-quota-test",
        )
        token = set_latency_tracker(tracker)

        tts_mgr = TTSTaskManager()
        tts_mgr.task_list.append(asyncio.sleep(0))

        mock_waiter = AsyncMock(
            return_value={
                "type": "frontend-playback-complete",
                "request_id": "turn-quota-test",
                "audio_outcome": "failed",
                "audio_segments": 1,
                "audio_played_segments": 0,
                "audio_failed_segments": 1,
                "audio_failure_reason": "quota_exceeded",
            }
        )

        try:
            with patch.object(
                conversation_utils.message_handler,
                "wait_for_response",
                new=mock_waiter,
            ):
                await finalize_conversation_turn(
                    tts_manager=tts_mgr,
                    websocket_send=socket.send_text,
                    client_uid="test-client",
                )
        finally:
            reset_latency_tracker(token)

        self.assertEqual(tracker.playback_outcome, "failed")
        self.assertEqual(tracker.playback_failed_segments, 1)


if __name__ == "__main__":
    unittest.main()
