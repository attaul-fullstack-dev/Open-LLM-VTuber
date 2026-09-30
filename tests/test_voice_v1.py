"""Voice V1 minimal activation — deterministic tests (no mic, no network).

Covers A-G: buffer flush, VAD-None guard, overlap cancel, no fake AI on
cancel, single-turn integrity, fail-soft cleanup, format boundary.
"""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.conversations import conversation_handler as ch_mod
from src.open_llm_vtuber.conversations.conversation_utils import (
    cleanup_conversation,
    message_handler,
)
from src.open_llm_vtuber.conversations.single_conversation import (
    process_single_conversation,
)
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


async def _instant_playback_complete(*_args, **_kwargs):
    return {"type": "frontend-playback-complete"}


class _FakeLLM:
    model = "voice-v1-test"
    max_tokens = 100

    def __init__(self, chunks=("hai.",), delay=0.0):
        self.chunks = list(chunks)
        self.delay = delay
        self.calls = 0

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls += 1
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []

    def extract_emotion_keys(self, _text):
        return []


class _FakeASR:
    def __init__(self, text="halo mili"):
        self.text = text
        self.seen = []

    async def async_transcribe_np(self, audio):
        self.seen.append(np.asarray(audio))
        return self.text


class _FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, message):
        self.messages.append(message)


def _make_agent(history_uid, conf_uid="mili-voice", llm=None):
    tts_config = TTSPreprocessorConfig(
        remove_special_char=True,
        translator_config={
            "translate_audio": False,
            "translate_provider": "deeplx",
        },
    )
    agent = BasicMemoryAgent(
        llm=llm or _FakeLLM(),
        system="persona Mili voice test",
        live2d_model=_FakeLive2D(),
        tts_preprocessor_config=tts_config,
        context_window_override=1200,
        context_safety_margin=100,
    )
    agent.set_memory_from_history(conf_uid, history_uid)
    return agent


def _make_context(agent, history_uid, conf_uid="mili-voice", asr=None):
    return SimpleNamespace(
        agent_engine=agent,
        asr_engine=asr or _FakeASR(),
        character_config=SimpleNamespace(
            conf_uid=conf_uid,
            human_name="Human",
            character_name="Mili",
            avatar="",
        ),
        history_uid=history_uid,
        user_timezone="Asia/Jakarta",
        voice_output_enabled=False,  # silent payloads: no TTS engine needed
        live2d_model=_FakeLive2D(),
        tts_engine=None,
        translate_engine=None,
    )


class VoiceTriggerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self._patches = []

    def tearDown(self):
        for p in self._patches:
            p.stop()
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def _patch_process(self, stub):
        p = patch.object(ch_mod, "process_single_conversation", stub)
        p.start()
        self._patches.append(p)

    def _trigger_args(self, tasks, buffers, ws=None):
        return {
            "msg_type": "mic-audio-end",
            "data": {},
            "client_uid": "c1",
            "context": SimpleNamespace(),
            "websocket": ws or _FakeWebSocket(),
            "client_contexts": {},
            "client_connections": {},
            "chat_group_manager": SimpleNamespace(get_client_group=lambda _uid: None),
            "received_data_buffers": buffers,
            "current_conversation_tasks": tasks,
            "broadcast_to_group": lambda *a, **k: None,
        }

    async def test_a_buffer_flushed_exactly_once(self):
        seen = []

        async def stub(**kwargs):
            seen.append(np.asarray(kwargs["user_input"]))
            return "ok"

        self._patch_process(stub)
        tasks = {}
        buffers = {"c1": np.array([0.1, -0.2, 0.3], dtype=np.float32)}
        await ch_mod.handle_conversation_trigger(**self._trigger_args(tasks, buffers))
        await tasks["c1"]
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(
            seen[0], np.array([0.1, -0.2, 0.3], dtype=np.float32)
        )
        self.assertEqual(buffers["c1"].size, 0)

    async def test_b_vad_none_is_fail_safe(self):
        handler = WebSocketHandler.__new__(WebSocketHandler)
        handler.client_contexts = {"c1": SimpleNamespace(vad_engine=None)}
        ws = _FakeWebSocket()
        await handler._handle_raw_audio_data(
            ws, "c1", {"audio": [1, 2, 3]}
        )  # must not raise
        self.assertEqual(ws.messages, [])

    async def test_c_two_rapid_triggers_leave_one_active_turn(self):
        completions = []

        async def slow_stub(**kwargs):
            await asyncio.sleep(5)
            completions.append(1)
            return "slow-ok"

        self._patch_process(slow_stub)
        tasks = {}
        buffers = {"c1": np.array([0.5], dtype=np.float32)}
        await ch_mod.handle_conversation_trigger(**self._trigger_args(tasks, buffers))
        first = tasks["c1"]
        buffers["c1"] = np.array([0.6], dtype=np.float32)
        await ch_mod.handle_conversation_trigger(**self._trigger_args(tasks, buffers))
        second = tasks["c1"]
        self.assertIsNot(first, second)
        await asyncio.sleep(0.2)
        self.assertTrue(first.cancelled() or first.done())
        self.assertFalse(second.done())
        second.cancel()
        try:
            await second
        except asyncio.CancelledError:
            pass
        self.assertEqual(completions, [])

    async def test_empty_buffer_starts_no_turn(self):
        called = []

        async def stub(**kwargs):
            called.append(1)
            return "ok"

        self._patch_process(stub)
        tasks = {}
        buffers = {"c1": np.array([], dtype=np.float32)}
        await ch_mod.handle_conversation_trigger(**self._trigger_args(tasks, buffers))
        self.assertEqual(called, [])
        self.assertNotIn("c1", tasks)

    async def test_g_format_boundary_verbatim_float32(self):
        handler = WebSocketHandler.__new__(WebSocketHandler)
        buffers = {"c1": np.array([], dtype=np.float32)}
        handler.received_data_buffers = buffers

        async def fake_record(_uid):
            return None

        handler._record_user_activity = fake_record
        ws = _FakeWebSocket()
        # Client VAD emits 16 kHz float32 in [-1, 1]; backend stores verbatim.
        payload = [0.25, -0.5, 0.0, 1.0]
        await handler._handle_audio_data(ws, "c1", {"audio": payload})
        self.assertEqual(buffers["c1"].dtype, np.float32)
        np.testing.assert_array_equal(
            buffers["c1"], np.array(payload, dtype=np.float32)
        )


class VoiceTurnIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self._playback_patch = patch.object(
            message_handler, "wait_for_response", _instant_playback_complete
        )
        self._playback_patch.start()

    def tearDown(self):
        self._playback_patch.stop()
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    async def test_e_single_voice_turn_one_user_one_ai(self):
        history_uid = create_new_history("mili-voice")
        llm = _FakeLLM(chunks=("hai juga.",))
        agent = _make_agent(history_uid, llm=llm)
        context = _make_context(agent, history_uid)
        ws = _FakeWebSocket()
        result = await process_single_conversation(
            context=context,
            websocket_send=ws.send_text,
            client_uid="c1",
            user_input=np.array([0.1] * 1600, dtype=np.float32),
            images=None,
            session_emoji="🎤",
            metadata={},
        )
        self.assertIn("hai juga", result)
        history = get_history("mili-voice", history_uid)
        roles = [m["role"] for m in history]
        self.assertEqual(roles.count("human"), 1)
        self.assertEqual(roles.count("ai"), 1)
        human = next(m for m in history if m["role"] == "human")
        self.assertEqual(human["content"], "halo mili")

    async def test_d_cancelled_turn_stores_no_ai_response(self):
        history_uid = create_new_history("mili-voice")
        llm = _FakeLLM(chunks=("satu.", "dua.", "tiga."), delay=0.3)
        agent = _make_agent(history_uid, llm=llm)
        context = _make_context(agent, history_uid)
        ws = _FakeWebSocket()
        task = asyncio.create_task(
            process_single_conversation(
                context=context,
                websocket_send=ws.send_text,
                client_uid="c1",
                user_input=np.array([0.1] * 1600, dtype=np.float32),
                images=None,
                session_emoji="🎤",
                metadata={},
            )
        )
        await asyncio.sleep(0.2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        history = get_history("mili-voice", history_uid)
        roles = [m["role"] for m in history]
        self.assertIn("human", roles)
        self.assertNotIn("ai", roles)

    async def test_f_cleanup_after_cancel_is_fail_safe(self):
        manager = TTSTaskManager()

        async def pending():
            await asyncio.sleep(30)

        manager.task_list.append(asyncio.create_task(pending()))
        for task in manager.task_list:
            task.cancel()
        cleanup_conversation(manager, "🎤")  # must not raise
        self.assertEqual(manager.task_list, [])


if __name__ == "__main__":
    unittest.main()
