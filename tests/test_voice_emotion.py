"""Voice Emotion V2 — deterministic per-sentence tags (no network, no mic).

Covers: mapping allowlist + priority + exact-match + fail-soft, tag helper,
speak() wiring (elevenlabs-only, display/history untouched, OFF = zero
calls, per-sentence independence), semaphore serialization, config default.
"""

import asyncio
import os
import tempfile
import unittest

from src.open_llm_vtuber.voice_emotion import (
    emotion_tag_for,
    tag_tts_text,
)


class MappingTest(unittest.TestCase):
    def test_happy(self):
        self.assertEqual(emotion_tag_for(["joy"]), "happy")

    def test_sad(self):
        self.assertEqual(emotion_tag_for(["sadness"]), "sad")

    def test_angry_and_strong(self):
        self.assertEqual(emotion_tag_for(["anger"]), "angry")
        self.assertEqual(emotion_tag_for(["anger_strong"]), "angry")

    def test_fearful_surprised_disgusted_shy_playful(self):
        self.assertEqual(emotion_tag_for(["fear"]), "fearful")
        self.assertEqual(emotion_tag_for(["surprise"]), "surprised")
        self.assertEqual(emotion_tag_for(["disgust"]), "disgusted")
        self.assertEqual(emotion_tag_for(["smirk"]), "playful")
        self.assertEqual(emotion_tag_for(["embarrassed"]), "shy")

    def test_neutral_calm_none(self):
        self.assertIsNone(emotion_tag_for(["neutral"]))
        self.assertIsNone(emotion_tag_for(["calm"]))
        self.assertIsNone(emotion_tag_for(["content"]))

    def test_unmapped_spec_emotions_none(self):
        # annoyed/assertive/caring have no extractor keys: plain synthesis,
        # never an invented tag.
        self.assertIsNone(emotion_tag_for(["annoyed"]))
        self.assertIsNone(emotion_tag_for(["assertive"]))
        self.assertIsNone(emotion_tag_for(["caring"]))

    def test_priority_fixed(self):
        self.assertEqual(emotion_tag_for(["joy", "anger"]), "angry")

    def test_exact_match_no_substring(self):
        self.assertIsNone(emotion_tag_for(["enjoy"]))
        self.assertIsNone(emotion_tag_for(["adjust"]))

    def test_case_insensitive(self):
        self.assertEqual(emotion_tag_for(["JOY"]), "happy")

    def test_fail_soft_inputs(self):
        self.assertIsNone(emotion_tag_for(None))
        self.assertIsNone(emotion_tag_for("joy"))
        self.assertIsNone(emotion_tag_for([]))
        self.assertIsNone(emotion_tag_for([None, 123, ""]))

    def test_tag_helper(self):
        self.assertEqual(tag_tts_text("halo", "happy"), "[happy] halo")
        self.assertEqual(tag_tts_text("halo", None), "halo")
        self.assertEqual(tag_tts_text("", "happy"), "")


class _FakeElevenEngine:
    """Fake with elevenlabs module identity; records synthesis texts."""

    def __init__(self, enabled=True):
        self.texts = []
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.emotion_tags_enabled = enabled
        self.removed = []

    async def async_generate_audio(self, text, file_name_no_ext=None):
        import wave

        self.calls += 1
        self.texts.append(text)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        with wave.open(tmp.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 1600)
        return tmp.name

    def remove_file(self, path):
        self.removed.append(path)
        try:
            os.remove(path)
        except OSError:
            pass


_FakeElevenEngine.__module__ = "src.open_llm_vtuber.tts.elevenlabs_tts"


class _FakeOtherEngine(_FakeElevenEngine):
    pass


_FakeOtherEngine.__module__ = "src.open_llm_vtuber.tts.edge_tts"


class _FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, payload):
        import json as _json

        self.messages.append(_json.loads(payload))


class SpeakWiringTest(unittest.IsolatedAsyncioTestCase):
    async def _speak_all(self, mgr, **kwargs):
        from src.open_llm_vtuber.agent.output_types import Actions

        websocket = _FakeWebSocket()
        kwargs.setdefault("actions", Actions(emotions=[]))
        await mgr.speak(websocket_send=websocket.send_text, **kwargs)
        if mgr.task_list:
            await asyncio.gather(*mgr.task_list)
        await asyncio.sleep(0.1)
        return websocket

    def _display(self, text):
        from src.open_llm_vtuber.agent.output_types import DisplayText

        return DisplayText(text=text, name="Mili", avatar="a")

    async def test_tag_prepended_for_elevenlabs_only(self):
        from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager

        mgr = TTSTaskManager()
        engine = _FakeElevenEngine()
        websocket = await self._speak_all(
            mgr,
            tts_text="aku senang",
            display_text=self._display("aku senang"),
            actions=None,
            live2d_model=None,
            tts_engine=engine,
            emotion_tag="happy",
        )
        self.assertEqual(engine.texts, ["[happy] aku senang"])
        self.assertEqual(len(websocket.messages), 1)
        # Display text untouched by the synthesis tag.
        self.assertEqual(websocket.messages[0]["display_text"]["text"], "aku senang")

    async def test_no_tag_for_other_engines(self):
        from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager

        mgr = TTSTaskManager()
        engine = _FakeOtherEngine()
        await self._speak_all(
            mgr,
            tts_text="aku senang",
            display_text=self._display("aku senang"),
            actions=None,
            live2d_model=None,
            tts_engine=engine,
            emotion_tag="happy",
        )
        self.assertEqual(engine.texts, ["aku senang"])

    async def test_kill_switch(self):
        from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager

        mgr = TTSTaskManager()
        engine = _FakeElevenEngine(enabled=False)
        await self._speak_all(
            mgr,
            tts_text="aku senang",
            display_text=self._display("aku senang"),
            actions=None,
            live2d_model=None,
            tts_engine=engine,
            emotion_tag="happy",
        )
        self.assertEqual(engine.texts, ["aku senang"])

    async def test_per_sentence_independence_no_leak(self):
        from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager

        mgr = TTSTaskManager()
        engine = _FakeElevenEngine()
        websocket = _FakeWebSocket()
        await mgr.speak(
            tts_text="aku senang", display_text=self._display("a"),
            actions=None, live2d_model=None, tts_engine=engine,
            websocket_send=websocket.send_text, emotion_tag="happy",
        )
        await mgr.speak(
            tts_text="biasa saja", display_text=self._display("b"),
            actions=None, live2d_model=None, tts_engine=engine,
            websocket_send=websocket.send_text, emotion_tag=None,
        )
        if mgr.task_list:
            await asyncio.gather(*mgr.task_list)
        await asyncio.sleep(0.1)
        self.assertEqual(engine.texts, ["[happy] aku senang", "biasa saja"])

    async def test_voice_off_zero_calls(self):
        from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager

        mgr = TTSTaskManager()
        engine = _FakeElevenEngine()
        websocket = _FakeWebSocket()
        await mgr.speak(
            tts_text="aku senang", display_text=self._display("aku senang"),
            actions=None, live2d_model=None, tts_engine=engine,
            websocket_send=websocket.send_text, synthesize_audio=False,
            emotion_tag="happy",
        )
        if mgr.task_list:
            await asyncio.gather(*mgr.task_list)
        await asyncio.sleep(0.1)
        self.assertEqual(engine.calls, 0)
        self.assertEqual(len(websocket.messages), 1)

    async def test_serialization_single_flight(self):
        from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager

        mgr = TTSTaskManager()
        engine = _FakeElevenEngine()
        websocket = _FakeWebSocket()
        await asyncio.gather(*[
            mgr.speak(
                tts_text=f"kalimat {i}", display_text=self._display("x"),
                actions=None, live2d_model=None, tts_engine=engine,
                websocket_send=websocket.send_text, emotion_tag="happy",
            )
            for i in range(4)
        ])
        if mgr.task_list:
            await asyncio.gather(*mgr.task_list)
        await asyncio.sleep(0.1)
        self.assertEqual(engine.calls, 4)
        self.assertEqual(engine.max_in_flight, 1)

    def test_semaphore_survives_loop_recreation(self):
        # Regression: the old module-global semaphore bound to the first loop
        # and every later loop degraded to silent payloads ("bound to a
        # different event loop"). Two brand-new loops must both synthesize.
        from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager

        async def one_cycle():
            mgr = TTSTaskManager()
            engine = _FakeElevenEngine()
            websocket = _FakeWebSocket()
            await mgr.speak(
                tts_text="halo", display_text=self._display("halo"),
                actions=None, live2d_model=None, tts_engine=engine,
                websocket_send=websocket.send_text, emotion_tag="happy",
            )
            if mgr.task_list:
                await asyncio.gather(*mgr.task_list)
            await asyncio.sleep(0.1)
            return engine, websocket

        for _ in range(2):
            engine, websocket = asyncio.run(one_cycle())
            self.assertEqual(engine.calls, 1)
            self.assertEqual(len(websocket.messages), 1)
            self.assertEqual(engine.texts, ["[happy] halo"])


class ConfigTest(unittest.TestCase):
    def test_defaults_and_factory_plumb(self):
        from src.open_llm_vtuber.config_manager.tts import ElevenLabsTTSConfig

        cfg = ElevenLabsTTSConfig(api_key="k", voice_id="v")
        self.assertTrue(cfg.emotion_tags_enabled)
        dumped = cfg.model_dump()
        self.assertIn("emotion_tags_enabled", dumped)


if __name__ == "__main__":
    unittest.main()
