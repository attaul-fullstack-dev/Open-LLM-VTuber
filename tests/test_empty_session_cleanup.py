"""The empty-session cleanup must never delete a session that is in use.

``create_new_history`` writes the metadata skeleton *before* the client has
said anything, so a brand-new conversation is metadata-only for a moment. The
history-list cleanup treated every metadata-only file as an orphan and removed
it. If that happened between creation and the first message, the next
``store_message`` re-created the file from an empty list -- without a metadata
row -- so that conversation silently lost its sidebar title and its
relationship defaults. A second tab, a reconnect, or any client asking for the
history list was enough to trigger it.

These tests pin both halves of the fix:

* a session that a live client is on, or that was just created, is kept;
* an old metadata-only file that nobody is on is still removed, so the
  cleanup keeps doing its job.

The last test is a negative control: with the protection and the grace window
bypassed (the old behaviour), the same sequence deletes the file, which is what
makes the other assertions meaningful.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import src.open_llm_vtuber.chat_history_manager as history_store
from src.open_llm_vtuber.chat_history_manager import (
    EMPTY_SESSION_GRACE_SECONDS,
    create_new_history,
    get_history,
    get_history_list,
    get_metadata,
    store_message,
    update_metadate,
)
import src.open_llm_vtuber.websocket_handler as ws_module

CONF = "cleanupchar"


def session_path(history_uid: str) -> str:
    return os.path.join("chat_history", CONF, f"{history_uid}.json")


def age_file(history_uid: str, seconds: float) -> None:
    """Make a file look older than the grace window."""
    stamp = time.time() - seconds
    os.utime(session_path(history_uid), (stamp, stamp))


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def make_handler(active: dict) -> object:
    """Handler wired with live client contexts only (no full __init__)."""
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


class EmptySessionCleanupTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mili-empty-cleanup-")
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs(os.path.join("chat_history", CONF), exist_ok=True)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def _list_via_handler(self, handler) -> list:
        socket = FakeSocket()
        import asyncio

        asyncio.run(
            handler._handle_history_list_request(
                socket, next(iter(handler.client_contexts)), {}
            )
        )
        self.assertEqual(len(socket.sent), 1)
        self.assertEqual(socket.sent[0]["type"], "history-list")
        return socket.sent[0]["histories"]

    # -- the session being used is protected ------------------------------

    def test_new_session_survives_another_clients_history_list(self):
        older = create_new_history(CONF)
        store_message(CONF, older, "human", "pesan lama", msg_id="old-1")

        # Same moment: this client creates a new conversation and stays on it,
        # while a second tab asks for the history list.
        fresh = create_new_history(CONF)
        handler = make_handler({"tab-new": fresh, "tab-old": older})

        listed = self._list_via_handler(handler)

        self.assertTrue(
            os.path.exists(session_path(fresh)),
            "a session a live client is on must not be cleaned up",
        )
        self.assertEqual([item["uid"] for item in listed], [older])

    def test_first_message_keeps_the_metadata_written_at_creation(self):
        fresh = create_new_history(CONF)
        update_metadate(CONF, fresh, {"title": "Rencana Akhir Pekan"})
        handler = make_handler({"tab-new": fresh})

        self._list_via_handler(handler)
        self.assertTrue(store_message(CONF, fresh, "human", "halo", msg_id="n-1"))

        metadata = get_metadata(CONF, fresh)
        self.assertEqual(metadata["role"], "metadata")
        self.assertEqual(metadata["title"], "Rencana Akhir Pekan")
        self.assertEqual(metadata["relationship_status"], "stranger")
        self.assertTrue(metadata["timestamp"])
        with open(session_path(fresh), encoding="utf-8") as handle:
            raw = json.load(handle)
        self.assertEqual(raw[0]["role"], "metadata")
        self.assertEqual([row["content"] for row in get_history(CONF, fresh)], ["halo"])

    def test_protected_session_is_kept_even_when_it_is_old(self):
        fresh = create_new_history(CONF)
        other = create_new_history(CONF)
        store_message(CONF, other, "human", "lain", msg_id="o-1")
        age_file(fresh, EMPTY_SESSION_GRACE_SECONDS + 60)

        get_history_list(CONF, protect_uids={fresh})

        self.assertTrue(
            os.path.exists(session_path(fresh)),
            "an active history is protected regardless of its age",
        )

    def test_just_created_session_is_kept_even_without_protection(self):
        fresh = create_new_history(CONF)
        other = create_new_history(CONF)
        store_message(CONF, other, "human", "lain", msg_id="o-1")

        # No client claims it (e.g. another server process just created it):
        # the grace window still covers the create -> first-message gap.
        get_history_list(CONF, protect_uids=set())

        self.assertTrue(os.path.exists(session_path(fresh)))

    # -- genuinely abandoned sessions are still cleaned up ----------------

    def test_orphan_empty_session_is_removed_once_it_is_old(self):
        orphan = create_new_history(CONF)
        used = create_new_history(CONF)
        store_message(CONF, used, "human", "ada isi", msg_id="u-1")
        age_file(orphan, EMPTY_SESSION_GRACE_SECONDS + 60)

        listed = get_history_list(CONF, protect_uids=set())

        self.assertFalse(os.path.exists(session_path(orphan)))
        self.assertEqual([item["uid"] for item in listed], [used])

    def test_sessions_with_messages_are_never_modified(self):
        used = create_new_history(CONF)
        store_message(CONF, used, "human", "halo", msg_id="a")
        store_message(CONF, used, "ai", "hai", msg_id="b")
        with open(session_path(used), "rb") as handle:
            before = handle.read()
        age_file(used, EMPTY_SESSION_GRACE_SECONDS * 10)

        get_history_list(CONF, protect_uids=set())

        with open(session_path(used), "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_repeated_requests_and_a_retry_create_no_duplicates(self):
        fresh = create_new_history(CONF)
        handler = make_handler({"tab-new": fresh})

        for _ in range(3):
            self._list_via_handler(handler)

        self.assertTrue(store_message(CONF, fresh, "human", "halo", msg_id="dup"))
        self.assertTrue(store_message(CONF, fresh, "human", "halo", msg_id="dup"))

        sessions = [
            name
            for name in os.listdir(os.path.join("chat_history", CONF))
            if name.endswith(".json")
        ]
        self.assertEqual(sessions, [f"{fresh}.json"])
        self.assertEqual(len(get_history(CONF, fresh)), 1)
        self.assertEqual(get_metadata(CONF, fresh)["relationship_status"], "stranger")

    # -- negative control --------------------------------------------------

    def test_negative_control_old_behaviour_deletes_the_new_session(self):
        """Without the protection and the grace window the bug comes back.

        This is the old rule: every metadata-only file is an orphan. It must
        delete the session here, otherwise the "survives" assertions above
        would pass for a reason that has nothing to do with the fix.
        """
        fresh = create_new_history(CONF)
        other = create_new_history(CONF)
        store_message(CONF, other, "human", "lain", msg_id="o-1")

        with patch.object(history_store, "EMPTY_SESSION_GRACE_SECONDS", 0.0):
            get_history_list(CONF, protect_uids=set())

        self.assertFalse(os.path.exists(session_path(fresh)))

        # And the consequence the fix prevents: the first message would write
        # a file with no metadata row at all.
        store_message(CONF, fresh, "human", "halo", msg_id="n-1")
        self.assertEqual(get_metadata(CONF, fresh), {})


if __name__ == "__main__":
    unittest.main()
