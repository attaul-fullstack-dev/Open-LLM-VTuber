"""Group-chat Attachment Memory capture — deterministic tests.

Runs the real ``handle_group_member_turn`` with fake agents/sockets (no
provider, no network) against temp-dir history files. Covers the capture
wiring added for group chats: scheduled after the member response is
persisted, only for image turns with a valid history uid, fire-and-forget,
fail-soft, and once per responding character.

Temp CWD per test: production history is never touched.
"""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.open_llm_vtuber.agent.output_types import (
    Actions,
    DisplayText,
    SentenceOutput,
)
from src.open_llm_vtuber.chat_history_manager import create_new_history, get_history
from src.open_llm_vtuber.conversations import conversation_utils as cu_mod
from src.open_llm_vtuber.conversations import group_conversation as gc_mod
from src.open_llm_vtuber.conversations.group_conversation import (
    handle_group_member_turn,
)
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager
from src.open_llm_vtuber.conversations.types import GroupConversationState

CONF = "groupattach"
DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
DATA_URL_2 = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//"
    "//////////////////////////////////2wBDAP//////////////////////////"
    "//////////////////////////////////////////////////wAARCAABAAEDASIAAhEB"
    "AxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAF"
    "QEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/AP"
    "/EABQRAQAAAAAAAAAAAAAAAAAAAMD/2gAIAQMBAT8AH//Z"
)


def sentence(text):
    return SentenceOutput(
        display_text=DisplayText(text=text, name="Mili", avatar=""),
        tts_text=text,
        actions=Actions(),
    )


class FakeAgent:
    """Yields a fixed reply; records attachment capture calls."""

    def __init__(self, name="Mili"):
        self.name = name
        self.capture_calls = []

    async def chat(self, input_data):
        yield sentence(f"Balasan dari {self.name}.")

    async def capture_attachment_memory(
        self, images, user_text, history_uid, request_id
    ):
        self.capture_calls.append(
            {
                "images": images,
                "user_text": user_text,
                "history_uid": history_uid,
                "request_id": request_id,
            }
        )


class RaisingAgent(FakeAgent):
    async def capture_attachment_memory(
        self, images, user_text, history_uid, request_id
    ):
        self.capture_calls.append({"ok": True})
        raise RuntimeError("vision provider down")


class MissingCaptureAgent:
    async def chat(self, input_data):
        yield sentence("Balasan tanpa capture.")


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, payload):
        self.sent.append(payload)


def make_context(conf_uid, history_uid, agent):
    return SimpleNamespace(
        history_uid=history_uid,
        user_timezone="Asia/Jakarta",
        voice_output_enabled=False,
        asr_engine=None,
        live2d_model=None,
        tts_engine=None,
        translate_engine=None,
        character_config=SimpleNamespace(
            conf_uid=conf_uid,
            character_name="Mili",
            avatar="",
            human_name="Human",
        ),
        agent_engine=agent,
    )


def sanitized_image(name="kucing.png", data=DATA_URL, source="upload"):
    """Shape produced by conversations.conversation_utils.sanitize_images."""
    return {
        "source": source,
        "data": data,
        "mime_type": "image/png",
        "name": name,
        "size": 70,
    }


def make_state():
    state = GroupConversationState(group_id="g1")
    state.conversation_history = []
    state.memory_index = {}
    state.group_queue = []
    return state


def run_member_turn(
    member_uid,
    context,
    state,
    images,
    metadata=None,
    group_members=None,
    members=None,
):
    async def _run():
        sent = []

        async def _broadcast(*args):
            sent.append(args[-1])

        sockets = members or {member_uid: FakeSocket()}
        with patch.object(cu_mod, "PLAYBACK_COMPLETE_TIMEOUT_S", 0.05):
            await handle_group_member_turn(
                current_member_uid=member_uid,
                state=state,
                client_contexts={member_uid: context},
                client_connections=sockets,
                broadcast_func=_broadcast,
                group_members=group_members or [member_uid],
                images=images,
                tts_manager=TTSTaskManager(),
                metadata=metadata,
            )
        return sent

    return asyncio.run(_run())


class GroupAttachmentCaptureTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _history(self, conf_uid=CONF):
        history_uid = create_new_history(conf_uid)
        state = make_state()
        state.conversation_history = ["Human: lihat kucingku"]
        state.memory_index = {"m1": 0}
        return history_uid, state

    def test_image_turn_schedules_capture_once_after_persist(self):
        history_uid, state = self._history()
        agent = FakeAgent()
        context = make_context(CONF, history_uid, agent)
        images = [sanitized_image()]

        run_member_turn("m1", context, state, images)

        # Response persisted before the capture runs.
        rows = get_history(CONF, history_uid)
        self.assertTrue(any("Balasan dari Mili." in m["content"] for m in rows))
        # Fire-and-forget: allow the task to land.
        asyncio.run(asyncio.sleep(0.05))
        self.assertEqual(len(agent.capture_calls), 1)
        call = agent.capture_calls[0]
        self.assertEqual(call["images"], images)
        self.assertEqual(call["history_uid"], history_uid)
        self.assertIn("lihat kucingku", call["user_text"])
        self.assertEqual(call["request_id"], "")

    def test_capture_receives_sanitized_metadata_for_dedup(self):
        history_uid, state = self._history()
        agent = FakeAgent()
        context = make_context(CONF, history_uid, agent)
        images = [sanitized_image(), sanitized_image("motor.jpg", DATA_URL_2)]

        run_member_turn("m1", context, state, images)
        asyncio.run(asyncio.sleep(0.05))

        call = agent.capture_calls[0]
        self.assertEqual(len(call["images"]), 2)
        # name/size survive so the store can label and dedupe by content hash.
        self.assertEqual(call["images"][0]["name"], "kucing.png")
        self.assertEqual(call["images"][1]["name"], "motor.jpg")
        for image in call["images"]:
            self.assertTrue(image["data"].startswith("data:image/"))

    def test_text_only_turn_never_schedules_capture(self):
        history_uid, state = self._history()
        agent = FakeAgent()
        context = make_context(CONF, history_uid, agent)

        run_member_turn("m1", context, state, None)
        asyncio.run(asyncio.sleep(0.05))

        self.assertEqual(agent.capture_calls, [])
        rows = get_history(CONF, history_uid)
        self.assertTrue(any("Balasan dari Mili." in m["content"] for m in rows))

    def test_empty_image_list_never_schedules_capture(self):
        history_uid, state = self._history()
        agent = FakeAgent()
        context = make_context(CONF, history_uid, agent)

        run_member_turn("m1", context, state, [])
        asyncio.run(asyncio.sleep(0.05))

        self.assertEqual(agent.capture_calls, [])

    def test_missing_history_uid_skips_capture(self):
        state = make_state()
        state.conversation_history = ["Human: lihat"]
        state.memory_index = {"m1": 0}
        agent = FakeAgent()
        context = make_context(CONF, None, agent)

        run_member_turn("m1", context, state, [sanitized_image()])
        asyncio.run(asyncio.sleep(0.05))

        self.assertEqual(agent.capture_calls, [])

    def test_proactive_turn_skips_capture(self):
        history_uid, state = self._history()
        agent = FakeAgent()
        context = make_context(CONF, history_uid, agent)

        run_member_turn(
            "m1",
            context,
            state,
            [sanitized_image()],
            metadata={"request_origin": "proactive"},
        )
        asyncio.run(asyncio.sleep(0.05))

        self.assertEqual(agent.capture_calls, [])

    def test_capture_failure_does_not_fail_the_chat_response(self):
        history_uid, state = self._history()
        agent = RaisingAgent()
        context = make_context(CONF, history_uid, agent)

        sent = run_member_turn("m1", context, state, [sanitized_image()])
        asyncio.run(asyncio.sleep(0.05))

        # The member response still streamed and persisted.
        self.assertTrue(sent)
        rows = get_history(CONF, history_uid)
        self.assertTrue(any("Balasan dari Mili." in m["content"] for m in rows))
        self.assertEqual(len(agent.capture_calls), 1)

    def test_agent_without_capture_hook_does_not_break_turn(self):
        history_uid, state = self._history()
        agent = MissingCaptureAgent()
        context = make_context(CONF, history_uid, agent)

        sent = run_member_turn("m1", context, state, [sanitized_image()])
        rows = get_history(CONF, history_uid)
        self.assertTrue(any("Balasan tanpa capture." in m["content"] for m in rows))
        self.assertTrue(sent)

    def test_each_character_captures_its_own_copy(self):
        """Group chat: every responding character stores under its own conf."""
        history_a = create_new_history("char-a")
        history_b = create_new_history("char-b")
        state = make_state()
        state.conversation_history = ["Human: dua gambar ini"]
        state.memory_index = {"m1": 0, "m2": 0}
        agent_a = FakeAgent("Mili")
        agent_b = FakeAgent("Lilith")
        contexts = {
            "m1": make_context("char-a", history_a, agent_a),
            "m2": make_context("char-b", history_b, agent_b),
        }
        sockets = {"m1": FakeSocket(), "m2": FakeSocket()}
        images = [sanitized_image()]

        async def _broadcast(*args):
            return None

        async def _run():
            for uid in ("m1", "m2"):
                with patch.object(cu_mod, "PLAYBACK_COMPLETE_TIMEOUT_S", 0.05):
                    await handle_group_member_turn(
                        current_member_uid=uid,
                        state=state,
                        client_contexts=contexts,
                        client_connections=sockets,
                        broadcast_func=_broadcast,
                        group_members=["m1", "m2"],
                        images=images,
                        tts_manager=TTSTaskManager(),
                        metadata=None,
                    )
            # Let both fire-and-forget tasks land.
            await asyncio.sleep(0.05)

        asyncio.run(_run())
        self.assertEqual(len(agent_a.capture_calls), 1)
        self.assertEqual(len(agent_b.capture_calls), 1)
        self.assertEqual(agent_a.capture_calls[0]["history_uid"], history_a)
        self.assertEqual(agent_b.capture_calls[0]["history_uid"], history_b)

    def test_capture_task_is_strongly_referenced(self):
        """A fire-and-forget task must not be garbage-collected mid-flight."""
        self.assertTrue(hasattr(gc_mod, "_GROUP_ATTACHMENT_CAPTURE_TASKS"))
        history_uid, state = self._history()
        agent = FakeAgent()
        context = make_context(CONF, history_uid, agent)
        run_member_turn("m1", context, state, [sanitized_image()])
        asyncio.run(asyncio.sleep(0.05))
        # Completed tasks are discarded; in-flight ones stay referenced.
        self.assertEqual(
            [
                task
                for task in gc_mod._GROUP_ATTACHMENT_CAPTURE_TASKS
                if not task.done()
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
