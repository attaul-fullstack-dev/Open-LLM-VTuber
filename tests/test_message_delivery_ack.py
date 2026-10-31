"""Stage 2: client ACK state machine + delivery-protocol frame metadata.

The backend may only keep a message "delivering" once it has proof the client
took it, and that proof is the client's own receipt. These tests pin the whole
contract with the real store, the real WebSocket handler and a real turn:

    only an ACK matching msg_id + turn_id + the ACTIVE attempt, from the
    client that OWNS the delivery, may move the state;
    duplicate/stale/foreign/unknown ACKs change nothing;
    state never moves backwards;
    a retry reuses the msg_id with a higher attempt and leaves ONE row;
    a store that cannot be written is reported as failure, never as success.

Explicitly out of scope (and asserted as such where it matters): exactly-once
delivery. This is at-least-once with client-side dedupe by msg_id, and
``received`` says nothing about rendering or playback.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from src.open_llm_vtuber.agent.output_types import Actions, DisplayText, SentenceOutput
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    store_message,
)
from src.open_llm_vtuber.conversations.single_conversation import (
    process_single_conversation,
)
from src.open_llm_vtuber.message_delivery import (
    ACK_DROPPED,
    ACK_RECEIVED,
    STATUS_DELIVERED,
    STATUS_DELIVERING,
    STATUS_RETRY_PENDING,
    apply_ack,
    begin_attempt,
    claim_retry,
    delivery_record,
    get_delivery_state_path,
    load_delivery_state,
)
from src.open_llm_vtuber.websocket_handler import WebSocketHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = "mili-ack"
OWNER = "client-a"


class _Transport:
    """Collects frames sent on one connection; can refuse like a dead socket."""

    def __init__(self, refuse=False):
        self.sent = []
        self.refuse = refuse

    async def __call__(self, payload):
        if self.refuse:
            raise ConnectionError("socket is gone")
        self.sent.append(json.loads(payload))

    async def send_text(self, payload):
        self.sent.append(json.loads(payload))

    def websocket(self, _client_uid=None):
        return SimpleNamespace(send_text=self.send_text)

    def audioish(self):
        return [frame for frame in self.sent if frame.get("type") == "audio"]

    def canonical(self):
        return [frame for frame in self.sent if frame.get("type") == "ai-final"]


def _make_handler(*client_uids):
    """A real handler instance with prepared client contexts (no socket)."""
    handler = WebSocketHandler.__new__(WebSocketHandler)
    handler.client_contexts = {
        uid: SimpleNamespace(
            character_config=SimpleNamespace(conf_uid=CONF), history_uid="h-1"
        )
        for uid in client_uids
    }
    return handler


class _TempDirCase(unittest.TestCase):
    """Every test runs inside its own data directory (cwd-relative stores)."""

    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    # ---- helpers ---------------------------------------------------------

    def _begin(self, msg_id="m-1", turn="turn-1", owner=OWNER, **kwargs):
        record = begin_attempt(
            CONF, msg_id, turn_id=turn, owner=owner, history_uid="h-1", **kwargs
        )
        self.assertIsNotNone(record, "the state store must accept a fresh attempt")
        return record

    def _ack(
        self, msg_id="m-1", turn="turn-1", attempt=1, status=ACK_RECEIVED, **kwargs
    ):
        return apply_ack(
            CONF,
            msg_id,
            turn_id=turn,
            attempt=attempt,
            status=status,
            owner=kwargs.pop("owner", OWNER),
            **kwargs,
        )

    def _record(self, msg_id="m-1"):
        return delivery_record(CONF, msg_id)


# --------------------------------------------------------------------------- #
# 1. The happy path: received(final) settles exactly that attempt
# --------------------------------------------------------------------------- #


class ReceivedAckTest(_TempDirCase):
    def test_received_final_frame_completes_the_message(self):
        self._begin()
        outcome = self._ack(seq=2, final=True)

        self.assertTrue(outcome.applied)
        self.assertTrue(outcome.persisted)
        self.assertEqual(outcome.reason, "delivered")
        record = self._record()
        self.assertEqual(record["status"], STATUS_DELIVERED)
        self.assertEqual(record["attempt"], 1)
        self.assertEqual(record["acked_seq"], 2)

    def test_received_earlier_frame_only_records_progress(self):
        self._begin()
        outcome = self._ack(seq=0, final=False)

        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.reason, "partial")
        record = self._record()
        # A partially received message is NOT settled: the rest may never have
        # arrived, and the backend must not claim the message was delivered.
        self.assertEqual(record["status"], STATUS_DELIVERING)
        self.assertEqual(record["acked_seq"], 0)

    def test_acked_seq_never_moves_backwards(self):
        self._begin()
        self._ack(seq=3, final=False)
        self._ack(seq=1, final=False)
        self.assertEqual(self._record()["acked_seq"], 3)
        self.assertEqual(self._record()["status"], STATUS_DELIVERING)


# --------------------------------------------------------------------------- #
# 2. Dropped: the message returns to RETRY_PENDING
# --------------------------------------------------------------------------- #


class DroppedAckTest(_TempDirCase):
    def test_dropped_frame_marks_the_message_retry_pending(self):
        self._begin()
        outcome = self._ack(status=ACK_DROPPED, seq=0, reason="interrupted")

        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.reason, "retry-pending")
        record = self._record()
        self.assertEqual(record["status"], STATUS_RETRY_PENDING)
        self.assertEqual(record["reason"], "interrupted")

    def test_received_cannot_resurrect_an_attempt_already_dropped(self):
        self._begin()
        self._ack(status=ACK_DROPPED, seq=0, reason="interrupted")
        outcome = self._ack(seq=2, final=True)

        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "retry-pending")
        self.assertEqual(self._record()["status"], STATUS_RETRY_PENDING)

    def test_retry_after_drop_settles_the_new_attempt(self):
        self._begin()
        self._ack(status=ACK_DROPPED, seq=0, reason="interrupted")
        record = self._begin(turn="turn-2")
        self.assertEqual(record["attempt"], 2)
        self.assertEqual(record["status"], STATUS_DELIVERING)

        outcome = self._ack(turn="turn-2", attempt=2, seq=1, final=True)
        self.assertTrue(outcome.applied)
        self.assertEqual(self._record()["status"], STATUS_DELIVERED)


# --------------------------------------------------------------------------- #
# 3-5. Duplicate / stale attempt / wrong turn
# --------------------------------------------------------------------------- #


class AckMatchingTest(_TempDirCase):
    def test_duplicate_ack_is_idempotent_and_writes_nothing(self):
        self._begin()
        first = self._ack(seq=1, final=True)
        self.assertTrue(first.applied)
        path = get_delivery_state_path(CONF)
        before = open(path, "rb").read()

        second = self._ack(seq=1, final=True)
        self.assertFalse(second.applied)
        self.assertTrue(second.persisted, "a refused duplicate is not a store failure")
        self.assertEqual(second.reason, "duplicate")
        self.assertEqual(open(path, "rb").read(), before)
        self.assertEqual(self._record()["status"], STATUS_DELIVERED)

    def test_ack_for_a_stale_attempt_never_settles_the_active_one(self):
        self._begin()
        self._begin(turn="turn-2")  # retry opened attempt 2
        outcome = self._ack(turn="turn-2", attempt=1, seq=1, final=True)

        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "stale-attempt")
        record = self._record()
        self.assertEqual(record["attempt"], 2)
        self.assertEqual(record["status"], STATUS_DELIVERING)

    def test_ack_for_a_not_yet_opened_attempt_is_refused(self):
        self._begin()
        outcome = self._ack(attempt=7, seq=1, final=True)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "unknown-attempt")
        self.assertEqual(self._record()["status"], STATUS_DELIVERING)

    def test_ack_with_the_wrong_turn_changes_nothing(self):
        self._begin()
        outcome = self._ack(turn="some-other-turn", seq=1, final=True)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "turn-mismatch")
        self.assertEqual(self._record()["status"], STATUS_DELIVERING)

    def test_ack_for_an_unknown_message_changes_nothing(self):
        self._begin()
        outcome = self._ack(msg_id="m-never-seen", seq=1, final=True)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "unknown-message")
        self.assertEqual(self._record()["status"], STATUS_DELIVERING)

    def test_malformed_ack_is_refused_without_touching_the_store(self):
        self._begin()
        for outcome, reason in (
            (
                apply_ack(CONF, None, turn_id="t", attempt=1, status=ACK_RECEIVED),
                "invalid-msg-id",
            ),
            (
                apply_ack(
                    CONF, "m-1", turn_id="turn-1", attempt=0, status=ACK_RECEIVED
                ),
                "invalid-attempt",
            ),
            (
                apply_ack(CONF, "m-1", turn_id="turn-1", attempt=1, status="rendered"),
                "invalid-status",
            ),
            (
                # Truthy but not a JSON boolean: "false" must not settle.
                apply_ack(
                    CONF,
                    "m-1",
                    turn_id="turn-1",
                    attempt=1,
                    status=ACK_RECEIVED,
                    seq=1,
                    final="false",
                ),
                "invalid-final",
            ),
            (
                # The closing frame must identify its seq; a bare final=true
                # is not enough proof that the whole message arrived.
                apply_ack(
                    CONF,
                    "m-1",
                    turn_id="turn-1",
                    attempt=1,
                    status=ACK_RECEIVED,
                    final=True,
                ),
                "invalid-seq",
            ),
        ):
            self.assertFalse(outcome.applied)
            self.assertTrue(outcome.persisted)
            self.assertEqual(outcome.reason, reason)
        self.assertEqual(self._record()["status"], STATUS_DELIVERING)

    def test_handler_does_not_coerce_a_truthy_string_into_final(self):
        self._begin()
        handler = _make_handler(OWNER)
        transport = _Transport()

        asyncio.run(
            handler._handle_message_ack(
                transport.websocket(OWNER),
                OWNER,
                {
                    "type": "message-ack",
                    "msg_id": "m-1",
                    "turn_id": "turn-1",
                    "attempt": 1,
                    "status": "received",
                    "seq": 1,
                    # JSON from a buggy client: truthy in Python, not a bool.
                    "final": "false",
                },
            )
        )

        self.assertEqual(self._record()["status"], STATUS_DELIVERING)
        # A refused malformed receipt is not a storage failure, so the handler
        # must not report a persistence error either.
        self.assertEqual(transport.sent, [])


# --------------------------------------------------------------------------- #
# 8. Late ACKs never move state backwards
# --------------------------------------------------------------------------- #


class NoRegressionTest(_TempDirCase):
    def test_late_dropped_ack_cannot_undo_delivered(self):
        self._begin()
        self._ack(seq=2, final=True)
        outcome = self._ack(status=ACK_DROPPED, seq=1, reason="interrupted")

        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "already-delivered")
        self.assertEqual(self._record()["status"], STATUS_DELIVERED)

    def test_late_stale_ack_after_a_new_attempt_is_ignored(self):
        self._begin()
        self._ack(seq=2, final=True)
        self._begin(turn="turn-2")
        late = self._ack(turn="turn-1", attempt=1, status=ACK_DROPPED, seq=2)

        self.assertFalse(late.applied)
        self.assertEqual(late.reason, "turn-mismatch")
        self.assertEqual(self._record()["attempt"], 2)
        self.assertEqual(self._record()["status"], STATUS_DELIVERING)


# --------------------------------------------------------------------------- #
# 9. A retry reuses the msg_id, raises the attempt, and stores ONE row
# --------------------------------------------------------------------------- #


class RetryIdentityTest(_TempDirCase):
    def test_retry_reuses_msg_id_with_an_increasing_attempt_and_one_history_row(self):
        history_uid = create_new_history(CONF)
        self._begin(turn="turn-1")
        self.assertTrue(
            store_message(CONF, history_uid, "ai", "versi pertama", msg_id="m-1")
        )
        self._ack(status=ACK_DROPPED, seq=0, reason="interrupted")

        record = self._begin(turn="turn-2")
        self.assertEqual(record["attempt"], 2)
        self.assertTrue(
            store_message(CONF, history_uid, "ai", "versi kedua", msg_id="m-1")
        )

        ai_rows = [row for row in get_history(CONF, history_uid) if row["role"] == "ai"]
        self.assertEqual(len(ai_rows), 1, "a retry must not duplicate the row")
        self.assertEqual(ai_rows[0]["content"], "versi kedua")
        self.assertEqual(ai_rows[0]["msg_id"], "m-1")
        self.assertEqual(
            len(load_delivery_state(CONF)["messages"]),
            1,
            "one record per logical message",
        )


# --------------------------------------------------------------------------- #
# 13. A store that cannot be written is never reported as success
# --------------------------------------------------------------------------- #


class PersistFailureTest(_TempDirCase):
    def test_unreadable_store_is_reported_as_not_persisted(self):
        self._begin()
        path = get_delivery_state_path(CONF)
        # Real filesystem failure: the store's path is now a DIRECTORY, so the
        # next read-modify-write cannot open it (IsADirectoryError).
        os.remove(path)
        os.makedirs(path)

        outcome = self._ack(seq=1, final=True)

        self.assertFalse(outcome.persisted, "a failed write is NOT a success")
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "persist-failed")
        self.assertNotEqual(outcome.reason, "delivered")

    def test_crash_during_ack_write_leaves_the_old_state_and_allows_redelivery(self):
        history_uid = create_new_history(CONF)
        record = begin_attempt(
            CONF,
            "m-1",
            turn_id="turn-1",
            owner=OWNER,
            client_ref="ref-crash",
            history_uid=history_uid,
        )
        self.assertEqual(record["status"], STATUS_DELIVERING)
        self.assertTrue(
            store_message(
                CONF, history_uid, "ai", "Ingat minum obat ya.", msg_id="m-1"
            )
        )
        path = get_delivery_state_path(CONF)
        with open(path, "r", encoding="utf-8") as handle:
            before = json.load(handle)

        # The ACK reached apply_ack but the atomic rename failed (crash,
        # full disk, or permission loss). The receipt must be reported as
        # unpersisted, and the previous "delivering" state must survive
        # intact -- never a half-written document and never a false
        # "delivered".
        with patch(
            "src.open_llm_vtuber.message_delivery.os.replace",
            side_effect=OSError("simulated crash during rename"),
        ):
            outcome = apply_ack(
                CONF,
                "m-1",
                turn_id="turn-1",
                attempt=1,
                status=ACK_RECEIVED,
                seq=1,
                final=True,
                owner=OWNER,
                client_ref="ref-crash",
                active_owners=frozenset({OWNER}),
            )

        self.assertFalse(outcome.persisted)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "persist-failed")
        with open(path, "r", encoding="utf-8") as handle:
            after = json.load(handle)
        self.assertEqual(after, before, "the old document must be intact")
        self.assertEqual(delivery_record(CONF, "m-1")["status"], STATUS_DELIVERING)

        # A reconnect can now redeliver the same msg_id. If the old ACK did
        # reach the client, this duplicates the final frame; the client
        # dedupes by msg_id. Exactly-once is not claimed.
        redelivery = claim_retry(
            CONF,
            "m-1",
            owner="client-b",
            client_ref="ref-crash",
            active_owners=frozenset({"client-b"}),
            history_uid=history_uid,
        )
        self.assertIsNotNone(redelivery)
        self.assertEqual(redelivery["msg_id"], "m-1")
        self.assertEqual(redelivery["attempt"], 2)

    def test_handler_reports_the_failure_frame_and_claims_nothing(self):
        self._begin()
        path = get_delivery_state_path(CONF)
        os.remove(path)
        os.makedirs(path)

        handler = _make_handler(OWNER)
        transport = _Transport()
        asyncio.run(
            handler._handle_message_ack(
                transport.websocket(OWNER),
                OWNER,
                {
                    "type": "message-ack",
                    "msg_id": "m-1",
                    "turn_id": "turn-1",
                    "attempt": 1,
                    "status": "received",
                    "seq": 1,
                    "final": True,
                },
            )
        )

        reports = [
            frame
            for frame in transport.sent
            if frame.get("type") == "message-ack-result"
        ]
        self.assertEqual(len(reports), 1)
        self.assertFalse(reports[0]["success"])
        self.assertEqual(reports[0]["error"], "ack-not-recorded")
        self.assertEqual(reports[0]["reason"], "persist-failed")
        # The failed receipt is echoed in full so the client can requeue it
        # (especially after a reconnect flush already removed its copy).
        self.assertEqual(
            reports[0]["ack"],
            {
                "type": "message-ack",
                "msg_id": "m-1",
                "turn_id": "turn-1",
                "attempt": 1,
                "status": "received",
                "final": True,
                "seq": 1,
            },
        )


# --------------------------------------------------------------------------- #
# 7. One client can never settle another client's message
# --------------------------------------------------------------------------- #


class OwnershipTest(_TempDirCase):
    def test_ack_from_another_client_uid_does_not_settle_the_message(self):
        self._begin(owner=OWNER)
        handler = _make_handler(OWNER, "client-b")
        transport = _Transport()

        asyncio.run(
            handler._handle_message_ack(
                transport.websocket("client-b"),
                "client-b",
                {
                    "type": "message-ack",
                    "msg_id": "m-1",
                    "turn_id": "turn-1",
                    "attempt": 1,
                    "status": "received",
                    "seq": 1,
                    "final": True,
                },
            )
        )

        record = self._record()
        self.assertEqual(record["status"], STATUS_DELIVERING)
        self.assertEqual(record["owner"], OWNER)
        # Not a storage problem, so nothing is reported as a failure either.
        self.assertEqual(transport.sent, [])


class HandlerIdentityTest(_TempDirCase):
    """The handler must not bypass the msg_id + turn + attempt + owner checks."""

    def _handler(self):
        handler = _make_handler(OWNER, "client-b")
        handler.client_protocols = {
            OWNER: {"client_ref": "ref-tab-1"},
            "client-b": {"client_ref": "ref-tab-1"},
        }
        return handler

    def _send(self, handler, client_uid, **overrides):
        transport = _Transport()
        payload = {
            "type": "message-ack",
            "msg_id": "m-1",
            "turn_id": "turn-1",
            "attempt": 1,
            "status": "received",
            "seq": 1,
            "final": True,
        }
        payload.update(overrides)
        asyncio.run(
            handler._handle_message_ack(
                transport.websocket(client_uid), client_uid, payload
            )
        )
        return transport

    def test_cross_client_turn_and_attempt_receipts_never_settle_the_active_one(self):
        self._begin(owner=OWNER, client_ref="ref-tab-1")
        handler = self._handler()

        # A second live connection with the SAME client_ref is still not the
        # owner. client_ref is a correlation token, not authentication.
        foreign = self._send(handler, "client-b")
        self.assertEqual(foreign.sent, [])
        self.assertEqual(self._record()["status"], STATUS_DELIVERING)

        # Same owner, wrong turn.
        wrong_turn = self._send(handler, OWNER, turn_id="turn-2")
        self.assertEqual(wrong_turn.sent, [])
        self.assertEqual(self._record()["status"], STATUS_DELIVERING)

        # Same owner/turn, but an attempt that was never opened.
        future_attempt = self._send(handler, OWNER, attempt=2)
        self.assertEqual(future_attempt.sent, [])
        self.assertEqual(self._record()["status"], STATUS_DELIVERING)

        # The genuine receipt still works, so the refusals above were not a
        # broken handler.
        genuine = self._send(handler, OWNER)
        self.assertEqual(genuine.sent, [])
        self.assertEqual(self._record()["status"], STATUS_DELIVERED)


# --------------------------------------------------------------------------- #
# Routing: message-ack never reaches the chat/trigger path
# --------------------------------------------------------------------------- #


class RoutingTest(_TempDirCase):
    def test_message_ack_has_its_own_handler(self):
        handler = WebSocketHandler.__new__(WebSocketHandler)
        handlers = WebSocketHandler._init_message_handlers(handler)
        self.assertIn("message-ack", handlers)
        self.assertEqual(handlers["message-ack"].__name__, "_handle_message_ack")
        # No chat-side handler may consume it.
        for forbidden in ("text-input", "mic-audio-end", "ai-speak-signal"):
            self.assertIsNot(handlers["message-ack"], handlers[forbidden])

    def test_route_message_dispatches_an_ack_without_a_conversation_trigger(self):
        handler = WebSocketHandler.__new__(WebSocketHandler)
        seen = []

        async def fake_ack(websocket, client_uid, data):
            seen.append((client_uid, data.get("type")))

        handler._handle_message_ack = fake_ack
        # Built AFTER the swap, so the table binds the recording handler.
        handler._message_handlers = WebSocketHandler._init_message_handlers(handler)

        asyncio.run(
            handler._route_message(
                SimpleNamespace(),
                "client-a",
                {"type": "message-ack", "msg_id": "m-1"},
            )
        )
        self.assertEqual(seen, [("client-a", "message-ack")])


# --------------------------------------------------------------------------- #
# 14. Two processes cannot settle the same message twice
# --------------------------------------------------------------------------- #

CHILD_SCRIPT = r"""
import json
import os
import sys
import time

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

from src.open_llm_vtuber.message_delivery import apply_ack

ready_dir = payload["ready_dir"]
open(os.path.join(ready_dir, payload["slot"] + ".ready"), "w").close()
deadline = time.time() + 20
while not os.path.exists(os.path.join(ready_dir, "start")):
    if time.time() > deadline:
        break
    time.sleep(0.005)

outcome = apply_ack(
    payload["conf"],
    payload["msg_id"],
    turn_id=payload["turn_id"],
    attempt=payload["attempt"],
    status="received",
    seq=1,
    final=True,
    owner=payload["owner"],
    base_dir=payload["base_dir"],
)

with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump(
        {"applied": outcome.applied, "reason": outcome.reason, "persisted": outcome.persisted},
        handle,
    )
"""


class CrossProcessTest(_TempDirCase):
    def test_two_processes_cannot_both_settle_the_same_message(self):
        base_dir = os.path.join(self._temp.name, "state")
        os.makedirs(base_dir, exist_ok=True)
        record = begin_attempt(
            CONF,
            "m-race",
            turn_id="turn-race",
            owner=OWNER,
            base_dir=base_dir,
        )
        self.assertEqual(record["attempt"], 1)

        ready_dir = os.path.join(self._temp.name, "ready")
        os.makedirs(ready_dir, exist_ok=True)
        results = []
        processes = []
        for slot in ("a", "b"):
            result_path = os.path.join(self._temp.name, f"result-{slot}.json")
            results.append(result_path)
            payload = json.dumps(
                {
                    "conf": CONF,
                    "msg_id": "m-race",
                    "turn_id": "turn-race",
                    "attempt": 1,
                    "owner": OWNER,
                    "base_dir": base_dir,
                    "ready_dir": ready_dir,
                    "slot": slot,
                    "result_path": result_path,
                }
            )
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        CHILD_SCRIPT,
                        ROOT,
                        self._temp.name,
                        payload,
                    ],
                    cwd=self._temp.name,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )

        deadline = 20
        for _ in range(deadline * 200):
            if all(
                os.path.exists(os.path.join(ready_dir, f"{slot}.ready"))
                for slot in ("a", "b")
            ):
                break
            import time

            time.sleep(0.01)
        open(os.path.join(ready_dir, "start"), "w").close()

        for process in processes:
            _stdout, stderr = process.communicate(timeout=60)
            self.assertEqual(
                process.returncode,
                0,
                stderr.decode("utf-8", "replace")[:2000],
            )

        outcomes = []
        for path in results:
            with open(path, "r", encoding="utf-8") as handle:
                outcomes.append(json.load(handle))

        applied = [outcome for outcome in outcomes if outcome["applied"]]
        self.assertEqual(
            len(applied),
            1,
            f"exactly one process may settle the message: {outcomes}",
        )
        other = [outcome for outcome in outcomes if not outcome["applied"]][0]
        self.assertEqual(other["reason"], "duplicate")

        final = delivery_record(CONF, "m-race", base_dir=base_dir)
        self.assertEqual(final["status"], STATUS_DELIVERED)
        with open(
            get_delivery_state_path(CONF, base_dir=base_dir), "r", encoding="utf-8"
        ) as handle:
            self.assertIsInstance(json.load(handle), dict)


# --------------------------------------------------------------------------- #
# 12 + 15. A real turn: protocol frames, legacy client, no regression
# --------------------------------------------------------------------------- #


class _FakeLive2D:
    model_info = {}


class _TwoSentenceAgent:
    """Minimal agent: two display segments, no audio, no LLM."""

    _llm = SimpleNamespace(model="ack-test", base_url="http://localhost:11434")

    def __init__(self):
        self.calls = 0

    def chat(self, _batch_input):
        self.calls += 1

        async def stream():
            yield SentenceOutput(
                display_text=DisplayText(text="Aku di sini. ", name="Mili"),
                tts_text="Aku di sini. ",
                actions=Actions(),
            )
            yield SentenceOutput(
                display_text=DisplayText(text="Tenang ya.", name="Mili"),
                tts_text="Tenang ya.",
                actions=Actions(),
            )

        return stream()


async def _drain_finalize(tts_manager=None, **_kwargs):
    for task in list(getattr(tts_manager, "task_list", None) or []):
        await asyncio.gather(task, return_exceptions=True)
    for _ in range(200):
        queue = getattr(tts_manager, "_payload_queue", None)
        if queue is None or queue.empty():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)


class TurnProtocolTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.history_uid = create_new_history(CONF)
        self.agent = _TwoSentenceAgent()

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    def _context(self):
        return SimpleNamespace(
            history_uid=self.history_uid,
            agent_engine=self.agent,
            asr_engine=None,
            live2d_model=_FakeLive2D(),
            tts_engine=None,
            translate_engine=None,
            voice_output_enabled=False,
            user_timezone="Asia/Jakarta",
            character_config=SimpleNamespace(
                conf_uid=CONF,
                human_name="User",
                character_name="Mili",
                avatar="mili.png",
            ),
        )

    async def _turn(self, transport, metadata=None):
        with patch(
            "src.open_llm_vtuber.conversations.single_conversation."
            "finalize_conversation_turn",
            new=_drain_finalize,
        ):
            return await process_single_conversation(
                context=self._context(),
                websocket_send=transport,
                client_uid=OWNER,
                user_input="",
                images=None,
                session_emoji="🌸",
                metadata=metadata,
            )

    async def test_multi_segment_turn_carries_one_identity_and_legacy_client_still_works(
        self,
    ):
        transport = _Transport()
        text = await self._turn(transport, metadata={"skip_history": True})

        self.assertEqual(text, "Aku di sini. Tenang ya.")
        audio = transport.audioish()
        self.assertEqual(len(audio), 2, "two segments, two content frames")
        self.assertEqual([frame["seq"] for frame in audio], [0, 1])
        self.assertTrue(all(frame["final"] is False for frame in audio))

        start = [
            frame
            for frame in transport.sent
            if frame.get("text") == "conversation-chain-start"
        ][0]
        msg_id = start["msg_id"]
        self.assertEqual(start["id"], msg_id)
        self.assertEqual(start["turn"], start["request_id"])
        self.assertEqual(start["attempt"], 1)
        self.assertTrue(all(frame["msg_id"] == msg_id for frame in audio))

        final = transport.canonical()[0]
        self.assertEqual(final["msg_id"], msg_id)
        self.assertEqual(final["attempt"], 1)
        self.assertEqual(final["seq"], 2)
        self.assertEqual(final["segments"], 2)
        self.assertTrue(final["final"])

        # LEGACY CLIENT: it never sent an ACK. Nothing settles the message and
        # nothing pretends it did -- the record stays open for a later retry.
        record = delivery_record(CONF, msg_id)
        self.assertEqual(record["status"], STATUS_DELIVERING)
        self.assertEqual(record["attempt"], 1)

        # The frames above are exactly what a protocol client acknowledges.
        outcome = apply_ack(
            CONF,
            final["msg_id"],
            turn_id=final["turn"],
            attempt=final["attempt"],
            status=ACK_RECEIVED,
            seq=final["seq"],
            final=True,
            owner=OWNER,
        )
        self.assertTrue(outcome.applied)
        self.assertEqual(delivery_record(CONF, msg_id)["status"], STATUS_DELIVERED)

        # One AI row, carrying the same identity.
        ai_rows = [
            row for row in get_history(CONF, self.history_uid) if row["role"] == "ai"
        ]
        self.assertEqual(len(ai_rows), 1)
        self.assertEqual(ai_rows[0]["msg_id"], msg_id)

    async def test_retry_after_a_dropped_ack_reuses_the_msg_id_and_stores_one_row(self):
        first_transport = _Transport()
        await self._turn(first_transport, metadata={"skip_history": True})
        msg_id = first_transport.canonical()[0]["msg_id"]

        self.assertTrue(
            apply_ack(
                CONF,
                msg_id,
                turn_id=first_transport.canonical()[0]["turn"],
                attempt=1,
                status=ACK_DROPPED,
                seq=0,
                reason="interrupted",
                owner=OWNER,
            ).applied
        )
        self.assertEqual(delivery_record(CONF, msg_id)["status"], STATUS_RETRY_PENDING)

        retry_transport = _Transport()
        await self._turn(retry_transport, metadata={"msg_id": msg_id})

        final = retry_transport.canonical()[0]
        self.assertEqual(final["msg_id"], msg_id)
        self.assertEqual(final["attempt"], 2, "a retry increments the attempt")

        applied = apply_ack(
            CONF,
            msg_id,
            turn_id=final["turn"],
            attempt=2,
            status=ACK_RECEIVED,
            seq=final["seq"],
            final=True,
            owner=OWNER,
        )
        self.assertTrue(applied.applied)

        # The late ACK of attempt 1 can no longer change anything.
        stale = apply_ack(
            CONF,
            msg_id,
            turn_id=first_transport.canonical()[0]["turn"],
            attempt=1,
            status=ACK_DROPPED,
            owner=OWNER,
        )
        self.assertFalse(stale.applied)
        self.assertEqual(delivery_record(CONF, msg_id)["status"], STATUS_DELIVERED)

        ai_rows = [
            row for row in get_history(CONF, self.history_uid) if row["role"] == "ai"
        ]
        self.assertEqual(len(ai_rows), 1, "one logical message, one row")
        self.assertEqual(ai_rows[0]["msg_id"], msg_id)

    async def test_persist_failure_keeps_the_turn_honest_and_never_claims_an_ack(self):
        # Storage is broken for the AI row. Stage 4.1: a content frame may only
        # leave the process once its row is durable, so no frame is sent at
        # all (Stage 1 also forbids the canonical final), and the delivery
        # record must not quietly claim the message settled.
        msg_id = uuid.uuid4().hex
        transport = _Transport()
        with patch(
            "src.open_llm_vtuber.conversations.single_conversation.store_message",
            return_value=False,
        ):
            await self._turn(transport, metadata={"msg_id": msg_id})

        self.assertEqual(transport.canonical(), [], "no canonical claim without a row")
        self.assertEqual(
            transport.audioish(), [], "no unrecorded content frame may be sent"
        )
        self.assertEqual(
            {
                frame.get("msg_id")
                for frame in transport.sent
                if frame.get("type") == "audio"
            },
            set(),
        )
        # The attempt was still opened (identity is harmless) and stays
        # unsettled: no client ACK can ever name an undelivered message.
        self.assertEqual(
            delivery_record(CONF, msg_id)["status"], STATUS_DELIVERING
        )


if __name__ == "__main__":
    unittest.main()
