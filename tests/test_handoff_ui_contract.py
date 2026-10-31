"""Continuation-view contract: what the client may show, read-only.

Stage 4.7 made the hand-off itself production code (old transcript untouched,
memory reloaded, client told). That left one hole: a continuation session has
no rows of its own yet, so every client-side path that renders "the active
history" would present a long conversation as a brand-new empty chat.

This file pins the READ side of the contract:

* ``conversation_handoff.continuation_view`` returns the source uid, its stamp
  and a BOUNDED tail of the source transcript's own rows -- same objects, same
  ``msg_id``/``timestamp``/text, nothing rewritten;
* ``history-data`` (both the fetch path and the post-detached-turn resync)
  carries that view, so a reload/reconnect can render the conversation as
  continued;
* ``history-list`` never lists a continuation before its first row (it has no
  rows, so it would render as an empty chat); an empty target with verified
  lineage is reported apart, in ``resumable_sessions``, and the cleanup keeps
  it for the resume window instead of deleting it as an orphan;
* nothing here writes to either transcript.

Deliberately not covered: rendering (frontend), and the live-server hand-off
(documented separately; ``MILI_HANDOFF_AUTO`` must not be enabled for it to
run by accident).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import src.open_llm_vtuber.websocket_handler as ws_module
from src.open_llm_vtuber import conversation_handoff as handoff
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.chat_history_manager import (
    EMPTY_CONTINUATION_TTL_SECONDS,
    create_new_history,
    get_history,
    get_history_list,
    get_metadata,
    resumable_continuations,
    store_message,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig

CONF = "continuationchar"


class _FakeLLM:
    model = "continuation-test"
    max_tokens = 100

    async def chat_completion(self, messages, system=None, tools=None):
        yield "oke."


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def make_agent() -> BasicMemoryAgent:
    return BasicMemoryAgent(
        llm=_FakeLLM(),
        system="persona Mili",
        live2d_model=_FakeLive2D(),
        tts_preprocessor_config=TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
        ),
    )


def transcript_path(history_uid: str) -> str:
    return os.path.join("chat_history", CONF, f"{history_uid}.json")


def file_state(history_uid: str) -> tuple[bytes, int]:
    path = transcript_path(history_uid)
    with open(path, "rb") as handle:
        payload = handle.read()
    return payload, os.stat(path).st_mtime_ns


def make_handler(context) -> object:
    handler = ws_module.WebSocketHandler.__new__(ws_module.WebSocketHandler)
    handler.client_contexts = {"tab-a": context}
    handler.client_connections = {"tab-a": SimpleNamespace()}
    handler.current_conversation_tasks = {}
    handler._detached_turns = {}
    handler._history_subscribers = {}
    handler.received_data_buffers = {}
    handler.broadcast_to_group = AsyncMock()
    handler._cancel_proactive_timer = AsyncMock()
    handler._activate_proactive_for_history = AsyncMock()
    return handler


class ContinuationViewTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mili-continuation-")
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        self.source_uid = create_new_history(CONF)
        for index, (role, text) in enumerate(
            [
                ("human", "halo Mili"),
                ("ai", "Halo! Ada yang bisa dibantu?"),
                ("human", "lanjut cerita kemarin yuk"),
                ("ai", "Boleh. Kita sampai mana tadi?"),
            ]
        ):
            self.assertTrue(
                store_message(
                    CONF,
                    self.source_uid,
                    role,
                    text,
                    name="Mili" if role == "ai" else None,
                    msg_id=f"src-{index}",
                )
            )
        self.source_rows = get_history(CONF, self.source_uid)
        self.source_bytes, self.source_mtime = file_state(self.source_uid)
        self.new_uid = handoff.create_continuation(CONF, self.source_uid)
        self.assertTrue(self.new_uid)
        self.agent = make_agent()
        self.context = SimpleNamespace(
            history_uid=self.new_uid,
            user_timezone="Asia/Jakarta",
            character_config=SimpleNamespace(conf_uid=CONF),
            agent_engine=self.agent,
        )
        self.agent.set_memory_from_history(
            conf_uid=CONF,
            history_uid=self.new_uid,
            user_timezone="Asia/Jakarta",
        )

    async def asyncTearDown(self) -> None:
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    # -- read-only view ---------------------------------------------------

    def test_view_carries_source_tail_unchanged(self):
        view = handoff.continuation_view(CONF, self.new_uid)
        self.assertIsNotNone(view)
        assert view is not None
        self.assertEqual(view["continued_from"], self.source_uid)
        self.assertTrue(view["continued_at"])
        self.assertEqual(view["context_total"], len(self.source_rows))
        self.assertEqual(view["context_messages"], self.source_rows)
        # Identity fields survive verbatim: the client keys bubbles on them.
        self.assertEqual(
            [row["msg_id"] for row in view["context_messages"]],
            [row["msg_id"] for row in self.source_rows],
        )
        self.assertEqual(
            [row["timestamp"] for row in view["context_messages"]],
            [row["timestamp"] for row in self.source_rows],
        )
        # Reading never writes.
        self.assertEqual(file_state(self.source_uid), (self.source_bytes, self.source_mtime))

    def test_view_is_bounded_but_reports_the_real_total(self):
        for index in range(40):
            self.assertTrue(
                store_message(
                    CONF,
                    self.source_uid,
                    "human" if index % 2 == 0 else "ai",
                    f"baris tambahan {index}",
                    msg_id=f"add-{index}",
                )
            )
        all_rows = get_history(CONF, self.source_uid)
        view = handoff.continuation_view(CONF, self.new_uid)
        assert view is not None
        self.assertEqual(view["context_total"], len(all_rows))
        self.assertEqual(len(view["context_messages"]), handoff.CONTINUATION_CONTEXT_ROWS)
        self.assertEqual(view["context_messages"], all_rows[-handoff.CONTINUATION_CONTEXT_ROWS:])

    def test_ordinary_session_has_no_continuation_view(self):
        self.assertIsNone(handoff.continuation_view(CONF, self.source_uid))
        self.assertIsNone(handoff.continuation_view("", self.source_uid))
        self.assertIsNone(handoff.continuation_view(CONF, ""))

    def test_missing_source_degrades_to_empty_preview(self):
        metadata_uid = handoff.create_continuation(CONF, "no-such-session")
        self.assertTrue(metadata_uid)
        view = handoff.continuation_view(CONF, metadata_uid)
        assert view is not None
        self.assertEqual(view["continued_from"], "no-such-session")
        self.assertEqual(view["context_messages"], [])
        self.assertEqual(view["context_total"], 0)

    def test_chained_continuation_uses_its_own_source(self):
        second = handoff.create_continuation(CONF, self.new_uid)
        self.assertTrue(second)
        store_message(CONF, self.new_uid, "human", "satu pesan di sesi tengah")
        view = handoff.continuation_view(CONF, second)
        assert view is not None
        self.assertEqual(view["continued_from"], self.new_uid)
        self.assertEqual(view["context_total"], 1)
        self.assertEqual(view["context_messages"][0]["content"], "satu pesan di sesi tengah")

    # -- frames -----------------------------------------------------------

    def test_history_data_payload_carries_the_continuation(self):
        payload = ws_module.WebSocketHandler._history_data_payload(CONF, self.new_uid)
        self.assertEqual(payload["type"], "history-data")
        self.assertEqual(payload["history_uid"], self.new_uid)
        self.assertEqual(payload["messages"], [])
        self.assertEqual(payload["continued_from"], self.source_uid)
        self.assertEqual(payload["context_messages"], self.source_rows)
        self.assertEqual(payload["context_total"], len(self.source_rows))

        plain = ws_module.WebSocketHandler._history_data_payload(CONF, self.source_uid)
        self.assertNotIn("continued_from", plain)
        self.assertNotIn("context_messages", plain)

    async def test_fetch_history_sends_the_continuation_and_writes_nothing(self):
        handler = make_handler(self.context)
        socket = _FakeSocket()
        await handler._handle_fetch_history(
            socket, "tab-a", {"history_uid": self.new_uid, "timezone": "Asia/Jakarta"}
        )
        self.assertEqual(len(socket.sent), 1)
        frame = socket.sent[0]
        self.assertEqual(frame["type"], "history-data")
        self.assertEqual(frame["history_uid"], self.new_uid)
        self.assertEqual(frame["continued_from"], self.source_uid)
        self.assertEqual(frame["context_messages"], self.source_rows)
        # The new transcript still holds only its own metadata row.
        self.assertEqual(get_history(CONF, self.new_uid), [])
        self.assertEqual(
            file_state(self.source_uid), (self.source_bytes, self.source_mtime)
        )

    # -- listing / resume -------------------------------------------------

    def test_history_list_marks_the_continuation_once_it_has_rows(self):
        # A resume happens on real conversations only: an empty session (a
        # half-finished hand-off) is not offered, exactly as before.
        listing = get_history_list(CONF, protect_uids=())
        self.assertNotIn(self.new_uid, {item["uid"] for item in listing})
        # Not a conversation -- but still the session a reconnect must adopt.
        self.assertEqual(
            [item["uid"] for item in resumable_continuations(CONF)],
            [self.new_uid],
        )

        self.assertTrue(store_message(CONF, self.new_uid, "human", "lanjut ya"))
        by_uid = {item["uid"]: item for item in get_history_list(CONF, protect_uids=())}
        self.assertIn(self.new_uid, by_uid)
        self.assertEqual(by_uid[self.new_uid]["continued_from"], self.source_uid)
        self.assertIsNone(by_uid[self.source_uid].get("continued_from"))

    def test_an_empty_continuation_target_is_disposable_after_the_resume_window(
        self,
    ):
        # Crash-safety is unchanged in kind, only in duration: a target created
        # but never reached is not a conversation, it is not offered anywhere,
        # and it IS removed -- once the resume window a restart needs is over.
        self.assertTrue(
            resumable_continuations(CONF),
            "an empty target with verified lineage must stay resumable",
        )
        stale = time.time() - EMPTY_CONTINUATION_TTL_SECONDS - 60
        os.utime(transcript_path(self.new_uid), (stale, stale))
        get_history_list(CONF, protect_uids=())
        self.assertFalse(os.path.exists(transcript_path(self.new_uid)))
        self.assertEqual(resumable_continuations(CONF), [])
        self.assertTrue(os.path.exists(transcript_path(self.source_uid)))

    def test_metadata_labels_survive_a_listing(self):
        metadata = get_metadata(CONF, self.new_uid)
        self.assertEqual(metadata["continued_from"], self.source_uid)
        get_history_list(CONF, protect_uids=())
        self.assertEqual(get_metadata(CONF, self.new_uid)["continued_from"], self.source_uid)

    # -- resync path ------------------------------------------------------

    async def test_detached_turn_resync_also_carries_the_continuation(self):
        handler = make_handler(self.context)
        socket = _FakeSocket()
        handler.client_connections = {"tab-a": socket}
        handler._history_subscribers = {"tab-a": self.new_uid}
        await handler._deliver_history_to_subscriber(self.new_uid)
        history_frames = [f for f in socket.sent if f.get("type") == "history-data"]
        self.assertEqual(len(history_frames), 1)
        self.assertEqual(history_frames[0]["continued_from"], self.source_uid)
        self.assertEqual(history_frames[0]["context_messages"], self.source_rows)


if __name__ == "__main__":
    unittest.main()
