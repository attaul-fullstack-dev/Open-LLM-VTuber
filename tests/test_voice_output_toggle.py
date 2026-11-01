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
        """The voice-output-toggle WS message flips the per-client runtime flag and sends confirmation."""
        context = SimpleNamespace(voice_output_enabled=True)
        handler = WebSocketHandler(default_context_cache=None)
        handler.client_contexts["client"] = context
        websocket = _FakeWebSocket()

        await handler._handle_voice_output_toggle(websocket, "client", {"enabled": False})
        self.assertFalse(context.voice_output_enabled)
        self.assertEqual(len(websocket.messages), 1)
        self.assertEqual(websocket.messages[0], {"type": "voice-output-state", "enabled": False})

        await handler._handle_voice_output_toggle(websocket, "client", {"enabled": True})
        self.assertTrue(context.voice_output_enabled)
        self.assertEqual(len(websocket.messages), 2)
        self.assertEqual(websocket.messages[1], {"type": "voice-output-state", "enabled": True})

    async def test_callable_synthesize_audio_skips_when_context_is_false(self):
        """Callable synthesize_audio dynamically evaluated as False must skip engine call."""
        context = SimpleNamespace(voice_output_enabled=False)
        engine = MagicMock()
        engine.async_generate_audio = AsyncMock(
            side_effect=AssertionError("Must not synthesize when voice is OFF")
        )
        engine.__class__.__module__ = "some_module.edge_tts"
        engine.remove_file = MagicMock()

        live2d_model = MagicMock()
        websocket = _FakeWebSocket()
        manager = TTSTaskManager()

        display = DisplayText(text="Teks saja ya.", name="Mili", avatar="a")
        actions = Actions(emotions=["smile"])

        await manager.speak(
            tts_text="Teks saja ya.",
            display_text=display,
            actions=actions,
            live2d_model=live2d_model,
            tts_engine=engine,
            websocket_send=websocket.send_text,
            synthesize_audio=lambda: context.voice_output_enabled,
        )
        await self._drain(manager)

        engine.async_generate_audio.assert_not_awaited()
        self.assertEqual(len(websocket.messages), 1)
        payload = websocket.messages[0]
        self.assertEqual(payload["type"], "audio")
        self.assertFalse(payload.get("audio"))
        self.assertEqual(payload["display_text"]["text"], "Teks saja ya.")
        self.assertEqual(payload["actions"]["emotions"], ["smile"])

    async def test_in_flight_callable_evaluates_at_synthesis_time(self):
        """If context flips to False before synthesis starts, the callable cancels generation."""
        context = SimpleNamespace(voice_output_enabled=True)
        engine = MagicMock()
        engine.async_generate_audio = AsyncMock(return_value="/tmp/fake.wav")
        engine.__class__.__module__ = "some_module.edge_tts"
        engine.remove_file = MagicMock()

        live2d_model = MagicMock()
        websocket = _FakeWebSocket()
        manager = TTSTaskManager()

        display = DisplayText(text="Menunggu proses.", name="Mili", avatar="a")
        actions = Actions(emotions=["neutral"])

        # Flip the flag immediately after queueing, before tasks run
        await manager.speak(
            tts_text="Menunggu proses.",
            display_text=display,
            actions=actions,
            live2d_model=live2d_model,
            tts_engine=engine,
            websocket_send=websocket.send_text,
            synthesize_audio=lambda: context.voice_output_enabled,
        )
        context.voice_output_enabled = False

        await self._drain(manager)

        # Dynamic callable in _process_tts checked context and skipped synthesis
        engine.async_generate_audio.assert_not_awaited()
        self.assertEqual(len(websocket.messages), 1)
        self.assertFalse(websocket.messages[0].get("audio"))

    async def test_rapid_toggle_sequence_consistency(self):
        """Rapid toggles OFF -> ON -> OFF settle consistently to final state."""
        context = SimpleNamespace(voice_output_enabled=True)
        handler = WebSocketHandler(default_context_cache=None)
        handler.client_contexts["client"] = context
        websocket = _FakeWebSocket()

        for state in [False, True, False, True, False]:
            await handler._handle_voice_output_toggle(websocket, "client", {"enabled": state})

        self.assertFalse(context.voice_output_enabled)
        self.assertEqual(len(websocket.messages), 5)
        self.assertEqual([m["enabled"] for m in websocket.messages], [False, True, False, True, False])

    async def test_websocket_voice_output_toggle_missing_context_is_safe(self):
        handler = WebSocketHandler(default_context_cache=None)
        websocket = _FakeWebSocket()
        # No context registered -> must not raise.
        await handler._handle_voice_output_toggle(websocket, "ghost", {"enabled": False})
        self.assertEqual(len(websocket.messages), 0)

    async def test_concurrent_websocket_connections_synchronized(self):
        """When client 1 toggles Voice OFF, client 2 sharing same conf_uid receives broadcast and flips."""
        char_cfg = SimpleNamespace(conf_uid="test_char_concurrent")
        ctx1 = SimpleNamespace(voice_output_enabled=True, character_config=char_cfg)
        ctx2 = SimpleNamespace(voice_output_enabled=True, character_config=char_cfg)
        handler = WebSocketHandler(default_context_cache=None)
        handler.client_contexts["client1"] = ctx1
        handler.client_contexts["client2"] = ctx2
        ws1 = _FakeWebSocket()
        ws2 = _FakeWebSocket()
        handler.client_connections["client1"] = ws1
        handler.client_connections["client2"] = ws2

        await handler._handle_voice_output_toggle(ws1, "client1", {"enabled": False})

        # Both contexts are updated to False
        self.assertFalse(ctx1.voice_output_enabled)
        self.assertFalse(ctx2.voice_output_enabled)
        # Client 1 received direct confirmation
        self.assertEqual(ws1.messages, [{"type": "voice-output-state", "enabled": False}])
        # Client 2 received broadcast notification
        self.assertEqual(ws2.messages, [{"type": "voice-output-state", "enabled": False}])

    def test_character_state_voice_output_persistence(self):
        """Verify durable persistence of voice output preference in character state."""
        import tempfile
        import uuid
        import os
        from src.open_llm_vtuber.character_state import (
            get_character_voice_output_enabled,
            set_character_voice_output_enabled,
            load_character_state,
        )

        test_conf = f"test_persistence_{uuid.uuid4().hex}"
        # Uninitialized returns default True
        self.assertTrue(get_character_voice_output_enabled(test_conf, default=True))

        # Explicitly set False
        saved = set_character_voice_output_enabled(test_conf, False)
        self.assertIsNotNone(saved)
        self.assertFalse(saved.voice_output_enabled)

        # Fresh load verifies False persisted
        loaded = load_character_state(test_conf)
        self.assertFalse(loaded.voice_output_enabled)
        self.assertFalse(get_character_voice_output_enabled(test_conf))

        # Clean up temporary test file
        from src.open_llm_vtuber.character_state import get_character_state_path
        path = get_character_state_path(test_conf)
        if os.path.exists(path):
            os.remove(path)
        lock_path = path + ".lock"
        if os.path.exists(lock_path):
            os.remove(lock_path)


if __name__ == "__main__":
    unittest.main()