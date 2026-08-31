import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.open_llm_vtuber.agent.output_types import Actions, DisplayText
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


class _FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, payload):
        self.messages.append(json.loads(payload))


class VoiceOutputToggleBackendTests(unittest.IsolatedAsyncioTestCase):
    async def _drain(self, manager):
        """Let queued TTS tasks run and wait for payload delivery."""
        if manager.task_list:
            await asyncio.gather(*manager.task_list)
        # The sender task is an endless drain loop; give it an event-loop turn
        # after the tasks enqueue their payloads so the websocket receives them.
        await asyncio.sleep(0.1)

    async def test_synthesize_audio_false_skips_engine_and_sends_silent_display_payload(self):
        """Voice Output OFF must not call the TTS engine and must still deliver text/actions."""
        engine = MagicMock()
        engine.async_generate_audio = AsyncMock(
            side_effect=AssertionError("synthesis must not run when disabled")
        )
        engine.__class__.__module__ = "some_module.edge_tts"
        engine.remove_file = MagicMock()

        live2d_model = MagicMock()
        websocket = _FakeWebSocket()
        manager = TTSTaskManager()

        display = DisplayText(text="Ya udah, hati-hati ya.", name="Mili", avatar="a")
        actions = Actions(emotions=["neutral"])

        await manager.speak(
            tts_text="Ya udah, hati-hati ya.",
            display_text=display,
            actions=actions,
            live2d_model=live2d_model,
            tts_engine=engine,
            websocket_send=websocket.send_text,
            synthesize_audio=False,
        )
        await self._drain(manager)

        engine.async_generate_audio.assert_not_awaited()
        self.assertEqual(len(websocket.messages), 1)
        payload = websocket.messages[0]
        self.assertEqual(payload["type"], "audio")
        # Silent payload: no synthesized audio bytes.
        self.assertFalse(payload.get("audio"))
        # Display text and actions still flow so text + contextual face work.
        self.assertEqual(payload["display_text"]["text"], "Ya udah, hati-hati ya.")
        self.assertEqual(payload["actions"]["emotions"], ["neutral"])

    async def test_synthesize_audio_true_still_calls_engine(self):
        """Voice Output ON keeps the existing synthesis path intact."""
        engine = MagicMock()
        engine.async_generate_audio = AsyncMock(return_value="/tmp/fake.wav")
        engine.__class__.__module__ = "some_module.edge_tts"
        engine.remove_file = MagicMock()

        live2d_model = MagicMock()
        websocket = _FakeWebSocket()
        manager = TTSTaskManager()

        display = DisplayText(text="Oke, ceritain aja.", name="Mili", avatar="a")
        actions = Actions(emotions=["neutral"])

        await manager.speak(
            tts_text="Oke, ceritain aja.",
            display_text=display,
            actions=actions,
            live2d_model=live2d_model,
            tts_engine=engine,
            websocket_send=websocket.send_text,
            synthesize_audio=True,
        )
        await self._drain(manager)

        # The regression guard: Voice Output ON must still call the TTS engine.
        engine.async_generate_audio.assert_awaited_once()

    async def test_websocket_voice_output_toggle_sets_context_flag(self):
        """The voice-output-toggle WS message flips the per-client runtime flag."""
        context = SimpleNamespace(voice_output_enabled=True)
        handler = WebSocketHandler(default_context_cache=None)
        handler.client_contexts["client"] = context
        websocket = _FakeWebSocket()

        await handler._handle_voice_output_toggle(websocket, "client", {"enabled": False})
        self.assertFalse(context.voice_output_enabled)

        await handler._handle_voice_output_toggle(websocket, "client", {"enabled": True})
        self.assertTrue(context.voice_output_enabled)

    async def test_websocket_voice_output_toggle_missing_context_is_safe(self):
        handler = WebSocketHandler(default_context_cache=None)
        websocket = _FakeWebSocket()
        # No context registered -> must not raise.
        await handler._handle_voice_output_toggle(websocket, "ghost", {"enabled": False})


if __name__ == "__main__":
    unittest.main()