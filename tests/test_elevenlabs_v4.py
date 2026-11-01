"""Unit tests for ElevenLabs V4 Turbo migration and parameter filtering.

Tests:
1. V4 parameter sanitization: verifies that when model_id="eleven_v4_turbo",
   only stability and similarity_boost are sent; style, speed, and use_speaker_boost
   are strictly filtered out to prevent HTTP 400 Bad Request.
2. Backward compatibility: verifies that legacy models (e.g. eleven_v3_conversational)
   still receive style, speed, and use_speaker_boost.
3. Voice ID integrity: verifies that the exact configured voice_id is passed.
4. Voice emotion audio tags: verifies [tag] is prepended to synthesis text without
   modifying display text or history text.
5. Error handling: verifies quota_exceeded and structured TTSError mapping.
6. Voice OFF safety: verifies that when synthesize_audio=False, client.text_to_speech.convert
   is called exactly 0 times and payload has audio=None.
7. Interruption cleanup: verifies tts_manager.clear() clears task_list and payload queue.
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.open_llm_vtuber.agent.output_types import DisplayText
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager
from src.open_llm_vtuber.tts.elevenlabs_tts import TTSEngine as ElevenLabsTTSEngine
from src.open_llm_vtuber.tts.tts_interface import (
    TTSError,
    TTSQuotaExceededError,
)
from src.open_llm_vtuber.voice_emotion import tag_tts_text, emotion_tag_for


class _FakeApiError(Exception):
    def __init__(self, status_code: int, body: dict):
        super().__init__(f"status_code: {status_code}, body: {body}")
        self.status_code = status_code
        self.body = body


class ElevenLabsV4UnitTests(unittest.TestCase):
    def setUp(self):
        self.mock_client_patcher = patch("src.open_llm_vtuber.tts.elevenlabs_tts.ElevenLabs")
        self.mock_eleven_class = self.mock_client_patcher.start()
        self.mock_client = MagicMock()
        self.mock_eleven_class.return_value = self.mock_client

        # Default convert mock returns iterable of bytes
        self.mock_client.text_to_speech.convert.return_value = [b"ID3fake_mp3_audio_chunk"]

    def tearDown(self):
        self.mock_client_patcher.stop()

    def test_1_v4_turbo_parameter_filtering(self):
        """Verifies that V4 Turbo only receives stability & similarity_boost."""
        engine = ElevenLabsTTSEngine(
            api_key="sk_test_fake_key",
            voice_id="cgSgspJ2msm6clMCkdW9",
            model_id="eleven_v4_turbo",
            output_format="mp3_44100_128",
            stability=0.5,
            similarity_boost=0.75,
            style=0.4,
            use_speaker_boost=True,
            speed=0.85,
        )

        audio_path = engine.generate_audio("Halo Mili, ini tes V4 Turbo.")
        self.assertIsNotNone(audio_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        # Check call arguments to client.text_to_speech.convert
        self.mock_client.text_to_speech.convert.assert_called_once()
        kwargs = self.mock_client.text_to_speech.convert.call_args.kwargs

        self.assertEqual(kwargs.get("model_id"), "eleven_v4_turbo")
        self.assertEqual(kwargs.get("voice_id"), "cgSgspJ2msm6clMCkdW9")
        self.assertEqual(kwargs.get("output_format"), "mp3_44100_128")

        voice_settings = kwargs.get("voice_settings")
        self.assertIsInstance(voice_settings, dict)
        self.assertEqual(voice_settings.get("stability"), 0.5)
        self.assertEqual(voice_settings.get("similarity_boost"), 0.75)

        # CRITICAL: Unsupported V4 parameters MUST NOT be present
        self.assertNotIn("style", voice_settings)
        self.assertNotIn("speed", voice_settings)
        self.assertNotIn("use_speaker_boost", voice_settings)

    def test_2_v3_backward_compatibility_parameters(self):
        """Verifies that legacy V3 models still receive style, speed, and use_speaker_boost."""
        engine = ElevenLabsTTSEngine(
            api_key="sk_test_fake_key",
            voice_id="cgSgspJ2msm6clMCkdW9",
            model_id="eleven_v3_conversational",
            output_format="mp3_44100_128",
            stability=0.5,
            similarity_boost=0.5,
            style=0.2,
            use_speaker_boost=True,
            speed=0.80,
        )

        audio_path = engine.generate_audio("Halo Mili V3.")
        self.assertIsNotNone(audio_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        kwargs = self.mock_client.text_to_speech.convert.call_args.kwargs
        voice_settings = kwargs.get("voice_settings")

        self.assertEqual(kwargs.get("model_id"), "eleven_v3_conversational")
        self.assertIn("style", voice_settings)
        self.assertEqual(voice_settings.get("style"), 0.2)
        self.assertIn("speed", voice_settings)
        self.assertEqual(voice_settings.get("speed"), 0.80)
        self.assertIn("use_speaker_boost", voice_settings)
        self.assertEqual(voice_settings.get("use_speaker_boost"), True)

    def test_3_voice_id_integrity(self):
        """Verifies that the Jessica premade voice ID is passed exactly."""
        engine = ElevenLabsTTSEngine(
            api_key="sk_test_fake_key",
            voice_id="cgSgspJ2msm6clMCkdW9",
            model_id="eleven_v4_turbo",
        )
        audio_path = engine.generate_audio("Test Voice ID.")
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        kwargs = self.mock_client.text_to_speech.convert.call_args.kwargs
        self.assertEqual(kwargs.get("voice_id"), "cgSgspJ2msm6clMCkdW9")

    def test_4_voice_emotion_tags(self):
        """Verifies that emotion tags are prepended to synthesis text without touching display text."""
        tag = emotion_tag_for(["joy"])
        self.assertEqual(tag, "happy")

        display_text = "Aku senang sekali hari ini!"
        tts_text = tag_tts_text(display_text, tag)
        self.assertEqual(tts_text, "[happy] Aku senang sekali hari ini!")
        self.assertEqual(display_text, "Aku senang sekali hari ini!")

    def test_5_error_handling_quota_exceeded(self):
        """Verifies that quota_exceeded ApiError is raised as TTSQuotaExceededError."""
        self.mock_client.text_to_speech.convert.side_effect = _FakeApiError(
            status_code=401,
            body={"detail": {"status": "quota_exceeded", "message": "You have exceeded your quota."}},
        )

        engine = ElevenLabsTTSEngine(
            api_key="sk_test_fake_key",
            voice_id="cgSgspJ2msm6clMCkdW9",
            model_id="eleven_v4_turbo",
        )

        with self.assertRaises(TTSQuotaExceededError):
            engine.generate_audio("Test quota error.")

    def test_6_voice_off_prevents_sdk_call(self):
        """Verifies that Voice OFF results in 0 SDK calls and silent payload."""
        async def run_async():
            manager = TTSTaskManager()
            mock_ws_send = AsyncMock()
            mock_engine = MagicMock(spec=ElevenLabsTTSEngine)
            mock_engine.__class__.__module__ = "src.open_llm_vtuber.tts.elevenlabs_tts"

            # Speak under Voice OFF
            await manager.speak(
                tts_text="Pesan diam tanpa suara.",
                display_text=DisplayText(text="Pesan diam tanpa suara.", name="Mili"),
                actions=None,
                live2d_model=MagicMock(),
                tts_engine=mock_engine,
                websocket_send=mock_ws_send,
                synthesize_audio=False,  # Voice OFF!
            )

            # Wait for tasks
            if manager.task_list:
                await asyncio.gather(*manager.task_list)
            await asyncio.sleep(0.1)

            # Engine generate_audio or async_generate_audio must NOT be called
            mock_engine.generate_audio.assert_not_called()
            mock_engine.async_generate_audio.assert_not_called()

            # Sent payload must have audio=None
            mock_ws_send.assert_called()
            first_call_arg = json.loads(mock_ws_send.call_args[0][0])
            self.assertEqual(first_call_arg.get("type"), "audio")
            self.assertIsNone(first_call_arg.get("audio"))
            self.assertEqual(first_call_arg.get("volumes"), [])
            self.assertEqual(first_call_arg.get("display_text", {}).get("text"), "Pesan diam tanpa suara.")

            manager.clear()

        import json
        asyncio.run(run_async())

    def test_7_interruption_clears_queue(self):
        """Verifies that tts_manager.clear() purges task list and queues."""
        manager = TTSTaskManager()
        manager._sequence_counter = 5
        manager._payload_queue.put_nowait(("payload", 0, "txt", None, None))
        self.assertEqual(manager._payload_queue.qsize(), 1)

        manager.clear()
        self.assertEqual(manager._sequence_counter, 0)
        self.assertEqual(manager._payload_queue.qsize(), 0)
        self.assertEqual(len(manager.task_list), 0)


if __name__ == "__main__":
    unittest.main()
