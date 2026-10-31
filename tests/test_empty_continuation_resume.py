"""A hand-off target that is still empty must survive a restart, resumably.

Stage 4.10 measured the hole this file closes: the client was handed off to a
new transcript, the server restarted before the first message, and the next
``history-list`` did not contain that transcript (it has no rows, so it is not a
conversation). The client fell back to the SOURCE session without telling
anyone -- while an explicit fetch by uid still read the continuation correctly.

The fix under test has three parts, and each half is pinned here separately:

* an empty session with a VERIFIED lineage (``continued_from`` points at a
  sibling transcript that exists and has rows) is not an orphan: the cleanup
  keeps it for ``EMPTY_CONTINUATION_TTL_SECONDS`` and removes it afterwards;
* it is still NOT in ``histories`` -- the sidebar keeps listing conversations,
  so an empty session can never render as an active empty chat;
* ``history-list`` reports it apart, in ``resumable_sessions``, so only a tab
  that already remembers that exact uid may adopt it.

Everything unverifiable keeps the old behaviour exactly: a metadata-only file
without lineage, with a dangling lineage, or pointing at a source that has no
rows is removed after the ordinary grace window and is never offered.

Nothing here writes to a SOURCE transcript; every test re-reads its bytes.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace

import src.open_llm_vtuber.websocket_handler as ws_module
from src.open_llm_vtuber import conversation_handoff as handoff
from src.open_llm_vtuber.chat_history_manager import (
    EMPTY_CONTINUATION_TTL_SECONDS,
    EMPTY_SESSION_GRACE_SECONDS,
    create_new_history,
    get_history,
    get_history_list,
    get_metadata,
    resumable_continuations,
    store_message,
    update_metadate,
)

CONF = "resumechar"


def session_path(history_uid: str) -> str:
    return os.path.join("chat_history", CONF, f"{history_uid}.json")


def age_file(history_uid: str, seconds: float) -> None:
    stamp = time.time() - seconds
    os.utime(session_path(history_uid), (stamp, stamp))


def file_bytes(history_uid: str) -> bytes:
    with open(session_path(history_uid), "rb") as handle:
        return handle.read()


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def make_handler(active: dict) -> object:
    """A handler wired with live client contexts only (no full __init__)."""
    handler = ws_module.WebSocketHandler.__new__(ws_module.WebSocketHandler)
    handler.client_contexts = {
        client_uid: SimpleNamespace(
            history_uid=history_uid,
            character_config=SimpleNamespace(conf_uid=CONF),
        )
        for client_uid, history_uid in active.items()
    }
    handler._detached_turns = {}
    return handler


def history_list_frame(active: dict) -> dict:
    handler = make_handler(active)
    socket = FakeSocket()
    asyncio.run(
        handler._handle_history_list_request(
            socket, next(iter(handler.client_contexts)), {}
        )
    )
    assert len(socket.sent) == 1
    return socket.sent[0]


class EmptyContinuationResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mili-resume-")
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs(os.path.join("chat_history", CONF), exist_ok=True)
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
        self.source_bytes = file_bytes(self.source_uid)
        self.continuation_uid = handoff.create_continuation(CONF, self.source_uid)
        self.assertTrue(self.continuation_uid)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    # -- 1. a valid empty continuation survives the restart window ---------

    def test_empty_continuation_survives_a_restart_and_stays_resumable(self):
        # The server is restarted minutes/hours later: no client holds the
        # target any more, and the ordinary 300 s grace window has long passed.
        age_file(self.continuation_uid, EMPTY_SESSION_GRACE_SECONDS + 600)
        # A different tab is on the source transcript.
        frame = history_list_frame({"tab-a": self.source_uid})

        self.assertTrue(
            os.path.exists(session_path(self.continuation_uid)),
            "a hand-off target must survive cleanup while it is still resumable",
        )
        listed = [item["uid"] for item in frame["histories"]]
        self.assertIn(self.source_uid, listed)
        self.assertNotIn(
            self.continuation_uid,
            listed,
            "an empty session must never be offered as a conversation",
        )
        resumable = {item["uid"]: item for item in frame["resumable_sessions"]}
        self.assertEqual(set(resumable), {self.continuation_uid})
        self.assertEqual(
            resumable[self.continuation_uid]["continued_from"], self.source_uid
        )
        self.assertTrue(resumable[self.continuation_uid]["continued_at"])
        self.assertEqual(file_bytes(self.source_uid), self.source_bytes)

    def test_explicit_fetch_returns_the_continuation_with_its_context(self):
        payload = ws_module.WebSocketHandler._history_data_payload(
            CONF, self.continuation_uid
        )
        self.assertEqual(payload["history_uid"], self.continuation_uid)
        self.assertEqual(payload["messages"], [])
        self.assertEqual(payload["continued_from"], self.source_uid)
        self.assertEqual(payload["context_messages"], self.source_rows)
        self.assertEqual(payload["context_total"], len(self.source_rows))
        self.assertEqual(get_history(CONF, self.continuation_uid), [])
        self.assertEqual(file_bytes(self.source_uid), self.source_bytes)

    # -- 2. the resumed continuation is the one that gets the first message -

    def test_first_message_lands_in_the_continuation_not_the_source(self):
        # What the client does after adopting a `resumable_sessions` uid.
        age_file(self.continuation_uid, EMPTY_SESSION_GRACE_SECONDS + 60)
        frame = history_list_frame({"tab-new": self.continuation_uid})
        self.assertIn(
            self.continuation_uid,
            {item["uid"] for item in frame["resumable_sessions"]},
        )

        self.assertTrue(
            store_message(
                CONF,
                self.continuation_uid,
                "human",
                "aku kembali",
                msg_id="resume-1",
            )
        )

        self.assertEqual(
            [row["content"] for row in get_history(CONF, self.continuation_uid)],
            ["aku kembali"],
        )
        self.assertEqual(len(get_history(CONF, self.source_uid)), 4)
        self.assertEqual(file_bytes(self.source_uid), self.source_bytes)
        # Metadata written at hand-off survives the first message.
        metadata = get_metadata(CONF, self.continuation_uid)
        self.assertEqual(metadata["continued_from"], self.source_uid)
        self.assertEqual(metadata["role"], "metadata")

    # -- 3. unverifiable empty sessions keep the old protection ------------

    def test_orphans_and_invalid_lineages_are_never_offered_or_resumable(self):
        plain = create_new_history(CONF)
        dangling = create_new_history(CONF)
        update_metadate(CONF, dangling, {"continued_from": "no-such-session"})
        empty_source = create_new_history(CONF)
        source_without_rows = create_new_history(CONF)
        update_metadate(
            CONF, empty_source, {"continued_from": source_without_rows}
        )
        self_referencing = create_new_history(CONF)
        update_metadate(
            CONF, self_referencing, {"continued_from": self_referencing}
        )

        frame = history_list_frame({"tab-a": self.source_uid})
        resumable = {item["uid"] for item in frame["resumable_sessions"]}
        self.assertEqual(resumable, {self.continuation_uid})

        for uid in (
            plain,
            dangling,
            empty_source,
            source_without_rows,
            self_referencing,
        ):
            self.assertNotIn(uid, resumable)
            self.assertNotIn(uid, {item["uid"] for item in frame["histories"]})
            age_file(uid, EMPTY_SESSION_GRACE_SECONDS + 60)

        get_history_list(CONF, protect_uids=set())

        for uid in (plain, dangling, empty_source, self_referencing):
            self.assertFalse(
                os.path.exists(session_path(uid)),
                f"an empty session without verified lineage must still be cleaned: {uid}",
            )
        self.assertFalse(
            os.path.exists(session_path(source_without_rows)),
            "the empty 'source' had nothing to continue and is an orphan itself",
        )
        self.assertTrue(os.path.exists(session_path(self.continuation_uid)))

    def test_a_continuation_with_rows_is_an_ordinary_conversation(self):
        self.assertTrue(
            store_message(
                CONF, self.continuation_uid, "human", "lanjut ya", msg_id="own-1"
            )
        )
        frame = history_list_frame({"tab-a": self.continuation_uid})
        by_uid = {item["uid"]: item for item in frame["histories"]}
        self.assertIn(self.continuation_uid, by_uid)
        self.assertEqual(by_uid[self.continuation_uid]["continued_from"], self.source_uid)
        self.assertNotIn(
            self.continuation_uid,
            {item["uid"] for item in frame["resumable_sessions"]},
            "a session with rows is listed as a conversation, not as resumable",
        )

    # -- 4. repeated requests and two tabs change nothing -------------------

    def test_repeated_requests_and_two_tabs_create_no_duplicates(self):
        handler = make_handler({"tab-a": self.continuation_uid, "tab-b": self.source_uid})
        before = set(os.listdir(os.path.join("chat_history", CONF)))

        for _ in range(3):
            socket = FakeSocket()
            asyncio.run(
                handler._handle_history_list_request(socket, "tab-a", {})
            )
            asyncio.run(
                handler._handle_history_list_request(socket, "tab-b", {})
            )
            self.assertEqual(
                [item["uid"] for item in socket.sent[0]["resumable_sessions"]],
                [self.continuation_uid],
            )

        self.assertEqual(set(os.listdir(os.path.join("chat_history", CONF))), before)
        frame = history_list_frame({"tab-a": self.continuation_uid})
        self.assertEqual(
            [item["uid"] for item in frame["resumable_sessions"]],
            [self.continuation_uid],
        )
        self.assertEqual(file_bytes(self.source_uid), self.source_bytes)

        # A retried first message is still one row.
        self.assertTrue(
            store_message(CONF, self.continuation_uid, "human", "halo", msg_id="dup")
        )
        self.assertTrue(
            store_message(CONF, self.continuation_uid, "human", "halo", msg_id="dup")
        )
        self.assertEqual(len(get_history(CONF, self.continuation_uid)), 1)
        self.assertEqual(get_metadata(CONF, self.continuation_uid)["role"], "metadata")

    # -- 5. the resume window is bounded ----------------------------------

    def test_the_resume_window_is_bounded(self):
        age_file(self.continuation_uid, EMPTY_CONTINUATION_TTL_SECONDS + 60)

        get_history_list(CONF, protect_uids=set())

        self.assertFalse(
            os.path.exists(session_path(self.continuation_uid)),
            "an empty target that nobody came back for is still removed eventually",
        )
        self.assertEqual(resumable_continuations(CONF), [])
        self.assertEqual(file_bytes(self.source_uid), self.source_bytes)


if __name__ == "__main__":
    unittest.main()
