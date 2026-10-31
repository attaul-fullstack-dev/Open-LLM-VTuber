"""Stage 3: reconnect recovery -- ``client-hello`` summary + bounded redelivery.

A client that disappears before its receipt arrives leaves the backend in
``delivering``. When it comes back (with a NEW ``client_uid``), the backend must
be able to recover that message WITHOUT duplicating it, and without ever
replaying a message the client already took.

These tests pin, with the real store, the real handler and a real transcript:

    no receipt at all      -> resend the SAME msg_id under attempt + 1
    partially received     -> no resend (the client's resync carries the text)
    delivered              -> never in the summary, never resent
    dropped                -> resend is allowed again
    two workers            -> one atomic lease, one winner (cross-process)
    restart                -> ``delivered`` stays delivered
    legacy client          -> no hello, no summary, no resend, unchanged
    unreadable store       -> reported as UNKNOWN, nothing is resent

Explicitly out of scope, and asserted where it matters: exactly-once
delivery. This remains at-least-once with client-side dedupe by ``msg_id``;
``received`` never means rendered, played or read.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    store_message,
)
from src.open_llm_vtuber.conversations.conversation_utils import send_canonical_final
from src.open_llm_vtuber.message_delivery import (
    ACK_DROPPED,
    ACK_RECEIVED,
    MAX_RETRIES,
    RETRY_LEASE_SECONDS,
    STATUS_DELIVERED,
    STATUS_DELIVERING,
    STATUS_RETRY_PENDING,
    apply_ack,
    begin_attempt,
    claim_retry,
    delivery_record,
    get_delivery_state_path,
    release_retry_claim,
    unsettled_messages,
)
from src.open_llm_vtuber.websocket_handler import WebSocketHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = "mili-reconnect"
REF = "ref-tab-1"
OWNER = "client-a"


class _Socket:
    """Records every frame the handler sends on one connection."""

    def __init__(self, refuse: bool = False):
        self.frames = []
        self.refuse = refuse

    async def send_text(self, payload):
        if self.refuse:
            raise ConnectionError("socket is gone")
        self.frames.append(json.loads(payload))

    def of_type(self, kind):
        return [frame for frame in self.frames if frame.get("type") == kind]


class _FinalRefusingSocket(_Socket):
    """Accepts the summary, then the socket dies on the first resend."""

    async def send_text(self, payload):
        frame = json.loads(payload)
        if frame.get("type") == "ai-final":
            raise ConnectionError("socket died before the resend")
        self.frames.append(frame)


def _handler(*uids, protocols=None):
    """A real handler with prepared contexts (no socket, no turn machinery)."""
    handler = WebSocketHandler.__new__(WebSocketHandler)
    handler.client_contexts = {
        uid: SimpleNamespace(
            character_config=SimpleNamespace(conf_uid=CONF),
            history_uid="",
        )
        for uid in uids
    }
    handler.client_protocols = dict(protocols or {})
    # Built after the contexts exist so the routing table binds real handlers.
    handler._message_handlers = WebSocketHandler._init_message_handlers(handler)
    return handler


class _DataDirCase(unittest.TestCase):
    """Each test runs in its own cwd (both stores are cwd-relative)."""

    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.history = create_new_history(CONF)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    # ---- helpers ---------------------------------------------------------

    def _persist(self, msg_id, text="Ingat minum obat ya.", history=None):
        stored = store_message(
            CONF,
            history or self.history,
            "ai",
            text,
            name="Mili",
            avatar="mili.png",
            msg_id=msg_id,
        )
        self.assertTrue(stored, "the transcript must accept the row")
        return text

    def _begin(
        self, msg_id="m-1", turn="turn-1", owner=OWNER, client_ref=REF, history=None
    ):
        record = begin_attempt(
            CONF,
            msg_id,
            turn_id=turn,
            owner=owner,
            client_ref=client_ref,
            history_uid=history or self.history,
        )
        self.assertIsNotNone(record, "the store must accept a fresh attempt")
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

    def _hello(
        self,
        handler_uids,
        socket,
        *,
        client_uid,
        client_ref=REF,
        history=None,
        ack=True,
        protocols=None,
    ):
        handler = _handler(*handler_uids, protocols=protocols)
        payload = {
            "type": "client-hello",
            "protocol": 1,
            "ack": ack,
            "client_ref": client_ref,
            "history_uid": history if history is not None else self.history,
        }
        asyncio.run(handler._handle_client_hello(socket, client_uid, payload))
        return handler


# --------------------------------------------------------------------------- #
# 1. No receipt at all -> resend with the SAME msg_id, attempt + 1
# --------------------------------------------------------------------------- #


class ReconnectRedeliveryTest(_DataDirCase):
    def test_reconnect_without_any_ack_resends_the_same_msg_id(self):
        self._begin()
        text = self._persist("m-1")
        socket = _Socket()

        self._hello(["client-b"], socket, client_uid="client-b")

        summary = socket.of_type("delivery-summary")
        self.assertEqual(len(summary), 1)
        self.assertTrue(summary[0]["ok"])
        self.assertTrue(summary[0]["redelivery"])
        self.assertEqual([item["msg_id"] for item in summary[0]["unsettled"]], ["m-1"])
        self.assertTrue(summary[0]["unsettled"][0]["resendable"])

        finals = socket.of_type("ai-final")
        self.assertEqual(len(finals), 1, "one unsettled message -> one resend")
        frame = finals[0]
        # SAME logical message ...
        self.assertEqual(frame["msg_id"], "m-1")
        self.assertEqual(frame["id"], "m-1")
        # ... NEW attempt, and an honest closing frame.
        self.assertEqual(frame["attempt"], 2)
        self.assertTrue(frame["final"])
        self.assertTrue(frame["redelivered"])
        self.assertEqual(frame["seq"], 0)
        self.assertEqual(frame["segments"], 0)
        self.assertEqual(frame["history_uid"], self.history)
        self.assertEqual(frame["text"], text)
        # The turn id is the message's own attempt lineage, not a new turn.
        self.assertEqual(frame["request_id"], "turn-1")

        record = delivery_record(CONF, "m-1")
        self.assertEqual(record["attempt"], 2)
        self.assertEqual(record["status"], STATUS_DELIVERING)
        self.assertEqual(record["retry_count"], 1)
        # Ownership follows the delivery: only the connection that received the
        # frames may settle them.
        self.assertEqual(record["owner"], "client-b")

    def test_redelivered_frame_settles_under_the_new_connection(self):
        self._begin()
        self._persist("m-1")
        socket = _Socket()
        handler = self._hello(["client-b"], socket, client_uid="client-b")
        attempt = socket.of_type("ai-final")[0]["attempt"]

        asyncio.run(
            handler._handle_message_ack(
                socket,
                "client-b",
                {
                    "type": "message-ack",
                    "msg_id": "m-1",
                    "turn_id": "turn-1",
                    "attempt": attempt,
                    "status": "received",
                    "seq": 0,
                    "final": True,
                },
            )
        )

        record = delivery_record(CONF, "m-1")
        self.assertEqual(record["status"], STATUS_DELIVERED)
        self.assertEqual(record["attempt"], 2)
        # No failure frame: the receipt WAS recorded.
        self.assertEqual(socket.of_type("message-ack-result"), [])

    def test_redelivery_keeps_one_transcript_row(self):
        self._begin()
        self._persist("m-1", "Ingat minum obat ya.")
        socket = _Socket()
        self._hello(["client-b"], socket, client_uid="client-b")

        rows = [row for row in get_history(CONF, self.history) if row["role"] == "ai"]
        self.assertEqual(len(rows), 1, "a resend must not add a transcript row")
        self.assertEqual(rows[0]["msg_id"], "m-1")
        self.assertEqual(rows[0]["content"], "Ingat minum obat ya.")

    def test_a_message_without_persisted_text_is_not_claimed(self):
        # No history row: there is nothing honest to resend.
        self._begin()
        socket = _Socket()

        self._hello(["client-b"], socket, client_uid="client-b")

        self.assertEqual(socket.of_type("ai-final"), [])
        record = delivery_record(CONF, "m-1")
        # Not settled, and the retry budget was not spent on an impossible send.
        self.assertEqual(record["status"], STATUS_DELIVERING)
        self.assertEqual(record["retry_count"], 0)
        self.assertEqual(record["attempt"], 1)

    def test_a_refused_resend_releases_the_claim_and_the_budget(self):
        self._begin()
        self._persist("m-1")
        socket = _FinalRefusingSocket()

        self._hello(["client-b"], socket, client_uid="client-b")

        self.assertEqual(len(socket.of_type("delivery-summary")), 1)
        self.assertEqual(socket.of_type("ai-final"), [])
        record = delivery_record(CONF, "m-1")
        # The claim had to exist before the send, but the frame never left:
        # the attempt stays monotonic while the retry budget is returned and
        # the lease is released for the next reconnect.
        self.assertEqual(record["attempt"], 2)
        self.assertEqual(record["retry_count"], 0)
        self.assertEqual(record["lease_until"], 0.0)
        self.assertEqual(record["status"], STATUS_DELIVERING)

    def test_retry_budget_is_bounded(self):
        self._begin()
        self._persist("m-1")
        # Spend the whole budget the way successive (spaced out) reconnects
        # would: one claim per lease window. The clock starts in the past so
        # the LAST lease is already expired when the handler below runs -- the
        # only thing left to stop it is the retry budget, which is what this
        # test is about.
        clock = time.time() - 1000
        for index in range(MAX_RETRIES):
            claimed = claim_retry(
                CONF,
                "m-1",
                owner="client-b",
                client_ref=REF,
                active_owners=frozenset({"client-b"}),
                history_uid=self.history,
                now=clock + index * 10,
            )
            self.assertIsNotNone(claimed, f"retry {index + 1} must be allowed")

        record = delivery_record(CONF, "m-1")
        self.assertEqual(record["retry_count"], MAX_RETRIES)
        self.assertEqual(record["attempt"], MAX_RETRIES + 1)

        socket = _Socket()
        self._hello(["client-b"], socket, client_uid="client-b")
        self.assertEqual(
            socket.of_type("ai-final"),
            [],
            "an undeliverable message must stop, not loop forever",
        )
        self.assertEqual(delivery_record(CONF, "m-1")["attempt"], MAX_RETRIES + 1)

    def test_one_hello_resends_at_most_the_configured_batch(self):
        limit = WebSocketHandler.MAX_REDELIVERIES_PER_HELLO
        for index in range(limit + 3):
            msg_id = f"m-batch-{index:02d}"
            self._begin(msg_id=msg_id, turn=f"turn-{index}")
            self._persist(msg_id, f"pesan {index}")
        socket = _Socket()

        self._hello(["client-b"], socket, client_uid="client-b", history=self.history)

        finals = socket.of_type("ai-final")
        self.assertEqual(len(finals), limit)
        self.assertEqual(len({frame["msg_id"] for frame in finals}), limit)

    def test_non_resendable_messages_do_not_starve_the_resendable_batch(self):
        limit = WebSocketHandler.MAX_REDELIVERIES_PER_HELLO
        # Fill the summary's oldest slots with messages the client already took
        # frames of. They are unsettled but must not be resent; the one truly
        # missing message behind them must still get its single resend.
        for index in range(limit):
            msg_id = f"m-partial-{index:02d}"
            self._begin(msg_id=msg_id, turn=f"turn-partial-{index}")
            self._persist(msg_id, f"sebagian {index}")
            partial = self._ack(
                msg_id=msg_id,
                turn=f"turn-partial-{index}",
                seq=0,
                final=False,
            )
            self.assertEqual(partial.reason, "partial")
        self._begin(msg_id="m-missing", turn="turn-missing")
        self._persist("m-missing", "pesan yang belum sampai")

        socket = _Socket()
        self._hello(["client-b"], socket, client_uid="client-b")

        finals = socket.of_type("ai-final")
        self.assertEqual([frame["msg_id"] for frame in finals], ["m-missing"])


# --------------------------------------------------------------------------- #
# 2 + 3. Settled / partially received -> nothing is resent
# --------------------------------------------------------------------------- #


class CrashWindowTest(_DataDirCase):
    """Deterministic views of the windows between begin, send and ACK."""

    def test_crash_after_begin_before_send_redelivers_after_restart(self):
        self._begin()
        self._persist("m-1")

        socket = _Socket()
        self._hello(["client-b"], socket, client_uid="client-b")

        finals = socket.of_type("ai-final")
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["msg_id"], "m-1")
        self.assertEqual(finals[0]["attempt"], 2)
        self.assertEqual(finals[0]["request_id"], "turn-1")
        self.assertEqual(len(get_history(CONF, self.history)), 1)

    def test_crash_after_transport_accepted_final_before_ack_duplicates_the_frame(self):
        # The final frame really left the transport; the process then died
        # before the client's ACK could be persisted.
        self._begin()
        text = self._persist("m-1")
        accepted = _Socket()
        sent = asyncio.run(
            send_canonical_final(
                accepted.send_text,
                self.history,
                "turn-1",
                text,
                msg_id="m-1",
                final_metadata={"attempt": 1, "seq": 1, "segments": 1, "final": True},
            )
        )
        self.assertTrue(sent)
        self.assertEqual(len(accepted.of_type("ai-final")), 1)
        record = delivery_record(CONF, "m-1")
        self.assertEqual(record["status"], STATUS_DELIVERING)
        self.assertEqual(record["acked_seq"], -1)

        # Restart + reconnect: the same logical message is sent again under a
        # new attempt. That is a duplicate frame, not a second message; the
        # client dedupes by msg_id and the transcript still has one row.
        reconnect = _Socket()
        self._hello(["client-b"], reconnect, client_uid="client-b")
        finals = reconnect.of_type("ai-final")
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["msg_id"], "m-1")
        self.assertEqual(finals[0]["attempt"], 2)
        self.assertEqual(len(get_history(CONF, self.history)), 1)

    def test_crash_between_claim_and_send_is_recovered_by_a_dead_process_token(self):
        self._begin()
        self._persist("m-1")
        start = time.time()
        dead_owner = (
            "linux:00000000000000000000000000000000:4026531836:999999:1"
            "|client-old"
        )
        with patch(
            "src.open_llm_vtuber.message_delivery.claim_owner",
            return_value=dead_owner,
        ):
            claimed = claim_retry(
                CONF,
                "m-1",
                owner="client-old",
                client_ref=REF,
                active_owners=frozenset({"client-old"}),
                history_uid=self.history,
                now=start,
            )
        self.assertEqual(claimed["attempt"], 2)

        # The old process is gone before it could send. The reconnect must not
        # wait five seconds for its lease; the process token proves it dead.
        socket = _Socket()
        self._hello(["client-new"], socket, client_uid="client-new")

        finals = socket.of_type("ai-final")
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["msg_id"], "m-1")
        self.assertEqual(finals[0]["attempt"], 3)
        self.assertEqual(delivery_record(CONF, "m-1")["owner"], "client-new")

    def test_partial_audio_before_history_persist_has_no_text_to_redeliver(self):
        # Residual crash window, asserted so it cannot be mistaken for a
        # guarantee: the turn began and the client buffered a content frame,
        # but the process died before the AI transcript row was written. The
        # partial receipt makes the message non-resendable, and there is no
        # persisted text to resend even if it were. A reload loses the bubble.
        self._begin()
        partial = self._ack(seq=0, final=False)
        self.assertTrue(partial.applied)
        record = delivery_record(CONF, "m-1")
        self.assertEqual(record["status"], STATUS_DELIVERING)
        self.assertEqual(record["acked_seq"], 0)
        self.assertEqual(get_history(CONF, self.history), [])

        socket = _Socket()
        self._hello(["client-b"], socket, client_uid="client-b")

        self.assertEqual(socket.of_type("ai-final"), [])
        summary = socket.of_type("delivery-summary")[0]
        self.assertEqual(summary["unsettled"][0]["msg_id"], "m-1")
        self.assertFalse(
            summary["unsettled"][0]["resendable"],
            "a partially received frame is not replayed (history resync handles it)",
        )
        self.assertEqual(get_history(CONF, self.history), [])

    def test_two_live_connections_racing_a_hello_send_only_once(self):
        self._begin()
        self._persist("m-1")
        handler = _handler("client-b", "client-c")
        first = _Socket()
        second = _Socket()
        hello = {
            "type": "client-hello",
            "protocol": 1,
            "ack": True,
            "client_ref": REF,
            "history_uid": self.history,
        }
        asyncio.run(handler._handle_client_hello(first, "client-b", dict(hello)))
        asyncio.run(handler._handle_client_hello(second, "client-c", dict(hello)))

        self.assertEqual(len(first.of_type("ai-final")), 1)
        self.assertEqual(second.of_type("ai-final"), [])
        self.assertEqual(delivery_record(CONF, "m-1")["attempt"], 2)


class SettledReconnectTest(_DataDirCase):
    def test_reconnect_after_delivered_sends_nothing(self):
        self._begin()
        self._persist("m-1")
        settled = self._ack(seq=1, final=True)
        self.assertTrue(settled.applied)
        socket = _Socket()

        self._hello(["client-b"], socket, client_uid="client-b")

        summary = socket.of_type("delivery-summary")[0]
        self.assertEqual(summary["unsettled"], [], "a settled message is not pending")
        self.assertEqual(socket.of_type("ai-final"), [])
        self.assertEqual(delivery_record(CONF, "m-1")["status"], STATUS_DELIVERED)

    def test_reconnect_after_a_partial_receipt_does_not_resend(self):
        self._begin()
        self._persist("m-1")
        partial = self._ack(seq=0, final=False)
        self.assertEqual(partial.reason, "partial")
        socket = _Socket()

        self._hello(["client-b"], socket, client_uid="client-b")

        summary = socket.of_type("delivery-summary")[0]
        # Still unsettled (a partial message is never reported as delivered) ...
        self.assertEqual(len(summary["unsettled"]), 1)
        self.assertEqual(summary["unsettled"][0]["status"], STATUS_DELIVERING)
        # ... but NOT resent: the client already has part of it, and its own
        # history resync carries the authoritative text without replaying audio.
        self.assertFalse(summary["unsettled"][0]["resendable"])
        self.assertEqual(socket.of_type("ai-final"), [])

    def test_reconnect_after_a_drop_resends(self):
        self._begin()
        self._persist("m-1")
        dropped = self._ack(status=ACK_DROPPED, reason="interrupted")
        self.assertEqual(dropped.reason, "retry-pending")
        self.assertEqual(delivery_record(CONF, "m-1")["status"], STATUS_RETRY_PENDING)
        socket = _Socket()

        self._hello(["client-b"], socket, client_uid="client-b")

        finals = socket.of_type("ai-final")
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["msg_id"], "m-1")
        self.assertEqual(finals[0]["attempt"], 2)
        self.assertEqual(delivery_record(CONF, "m-1")["status"], STATUS_DELIVERING)

        # And the resend can be settled normally.
        settled = self._ack(attempt=2, owner="client-b", seq=0, final=True)
        self.assertEqual(settled.reason, "delivered")


# --------------------------------------------------------------------------- #
# 4 + 6 + 10. Ownership across a reconnect
# --------------------------------------------------------------------------- #


class ReconnectOwnershipTest(_DataDirCase):
    def test_pending_receipt_is_recorded_after_the_owner_reconnected(self):
        self._begin()
        self._persist("m-1")
        # The old connection is gone: only the new uid is in client_contexts.
        socket = _Socket()
        handler = self._hello(["client-ref-uid"], socket, client_uid="client-ref-uid")
        self.assertEqual(len(socket.of_type("ai-final")), 1)

        outcome = apply_ack(
            CONF,
            "m-1",
            turn_id="turn-1",
            attempt=2,
            status=ACK_RECEIVED,
            seq=0,
            final=True,
            owner="client-ref-uid",
            client_ref=REF,
            active_owners=frozenset(h for h in handler.client_contexts),
        )
        self.assertEqual(outcome.reason, "delivered")
        self.assertEqual(delivery_record(CONF, "m-1")["status"], STATUS_DELIVERED)

    def test_receipt_from_another_client_instance_is_refused(self):
        self._begin()
        self._ack(seq=0, final=False)  # keep it unsettled
        outcome = apply_ack(
            CONF,
            "m-1",
            turn_id="turn-1",
            attempt=1,
            status=ACK_RECEIVED,
            seq=1,
            final=True,
            owner="client-other",
            client_ref="ref-other-tab",
            active_owners=frozenset({"client-other"}),
        )
        self.assertEqual(outcome.reason, "owner-mismatch")
        self.assertEqual(delivery_record(CONF, "m-1")["status"], STATUS_DELIVERING)

    def test_same_client_instance_cannot_settle_while_the_owner_is_live(self):
        # Two live connections of the same browser (two tabs): the new tab must
        # not be able to settle the message the old tab is still delivering.
        self._begin()
        outcome = apply_ack(
            CONF,
            "m-1",
            turn_id="turn-1",
            attempt=1,
            status=ACK_RECEIVED,
            seq=1,
            final=True,
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({OWNER, "client-b"}),
        )
        self.assertEqual(outcome.reason, "owner-mismatch")
        self.assertEqual(delivery_record(CONF, "m-1")["status"], STATUS_DELIVERING)

    def test_live_owner_blocks_a_second_connection_from_claiming_the_message(self):
        self._begin()
        self._persist("m-1")
        claimed = claim_retry(
            CONF,
            "m-1",
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({OWNER, "client-b"}),
            history_uid=self.history,
        )
        self.assertIsNone(claimed, "a live owner's message must not be stolen")

        # Both connections are live in this handler; the owner one is still
        # connected, so the newcomer gets an empty summary and no resend.
        socket = _Socket()
        self._hello(
            [OWNER, "client-b"],
            socket,
            client_uid="client-b",
            protocols={
                OWNER: {"ack": True, "client_ref": REF, "history_uid": self.history}
            },
        )
        self.assertEqual(socket.of_type("delivery-summary")[0]["unsettled"], [])
        self.assertEqual(socket.of_type("ai-final"), [])
        self.assertEqual(delivery_record(CONF, "m-1")["attempt"], 1)

    def test_summary_never_includes_another_clients_messages(self):
        begin_attempt(
            CONF,
            "m-foreign",
            turn_id="turn-x",
            owner="client-zzz",
            client_ref="ref-zzz",
            history_uid=self.history,
        )
        self._persist("m-foreign", "pesan orang lain")

        summary = unsettled_messages(
            CONF,
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({"client-b"}),
            history_uid=self.history,
        )
        self.assertTrue(summary.ok)
        self.assertEqual(summary.messages, [])

        socket = _Socket()
        self._hello(["client-b"], socket, client_uid="client-b")
        self.assertEqual(socket.of_type("delivery-summary")[0]["unsettled"], [])
        self.assertEqual(socket.of_type("ai-final"), [])

    def test_summary_is_scoped_to_the_announced_history(self):
        other_history = create_new_history(CONF)
        self._begin(msg_id="m-other", turn="turn-other", history=other_history)
        self._persist("m-other", "sesi lain", history=other_history)
        socket = _Socket()

        self._hello(["client-b"], socket, client_uid="client-b", history=self.history)

        self.assertEqual(socket.of_type("delivery-summary")[0]["unsettled"], [])
        self.assertEqual(socket.of_type("ai-final"), [])


# --------------------------------------------------------------------------- #
# 5. Duplicate / late / stale receipts around a reconnect
# --------------------------------------------------------------------------- #


class ReceiptRaceTest(_DataDirCase):
    def test_duplicate_receipt_after_reconnect_is_idempotent(self):
        self._begin()
        self._persist("m-1")
        socket = _Socket()
        self._hello(["client-b"], socket, client_uid="client-b")

        first = self._ack(attempt=2, owner="client-b", seq=0, final=True)
        state_after_first = delivery_record(CONF, "m-1")
        second = self._ack(attempt=2, owner="client-b", seq=0, final=True)

        self.assertEqual(first.reason, "delivered")
        self.assertFalse(second.applied)
        # The stored state already says delivered, which is exactly what this
        # receipt says: nothing to write, and nothing to report as an error.
        self.assertIn(second.reason, {"duplicate", "already-delivered"})
        self.assertTrue(second.persisted)
        self.assertEqual(delivery_record(CONF, "m-1"), state_after_first)

    def test_stale_receipt_from_before_the_redelivery_cannot_settle_it(self):
        self._begin()
        self._persist("m-1")
        socket = _Socket()
        self._hello(["client-b"], socket, client_uid="client-b")  # attempt 2 now

        stale = self._ack(attempt=1, owner="client-b", seq=1, final=True)

        self.assertFalse(stale.applied)
        self.assertEqual(stale.reason, "stale-attempt")
        record = delivery_record(CONF, "m-1")
        self.assertEqual(record["attempt"], 2)
        self.assertEqual(record["status"], STATUS_DELIVERING)

    def test_late_drop_cannot_downgrade_a_delivered_message(self):
        self._begin()
        self._ack(seq=1, final=True)
        late = self._ack(status=ACK_DROPPED, reason="interrupted")
        self.assertEqual(late.reason, "already-delivered")
        self.assertEqual(delivery_record(CONF, "m-1")["status"], STATUS_DELIVERED)

    def test_receipt_after_reconnect_with_a_changed_turn_is_refused(self):
        self._begin()
        self._ack(seq=0, final=False)
        outcome = apply_ack(
            CONF,
            "m-1",
            turn_id="turn-other",
            attempt=1,
            status=ACK_RECEIVED,
            seq=1,
            final=True,
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({"client-b"}),
        )
        self.assertEqual(outcome.reason, "turn-mismatch")
        self.assertEqual(delivery_record(CONF, "m-1")["status"], STATUS_DELIVERING)

    def test_a_drop_releases_the_redelivery_lease_immediately(self):
        self._begin()
        self._persist("m-1")
        first = claim_retry(
            CONF,
            "m-1",
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({"client-b"}),
            history_uid=self.history,
        )
        self.assertEqual(first["attempt"], 2)
        dropped = apply_ack(
            CONF,
            "m-1",
            turn_id="turn-1",
            attempt=2,
            status=ACK_DROPPED,
            reason="interrupted",
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({"client-b"}),
        )
        self.assertEqual(dropped.reason, "retry-pending")

        # Same instant: the dropped receipt proves the lease's send window is
        # over, so a quick reconnect may claim attempt 3 without waiting for
        # the five-second lease to expire.
        second = claim_retry(
            CONF,
            "m-1",
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({"client-b"}),
            history_uid=self.history,
        )
        self.assertIsNotNone(second)
        self.assertEqual(second["attempt"], 3)


# --------------------------------------------------------------------------- #
# 6 + 5. Lease: one worker at a time, retry after the lease expires
# --------------------------------------------------------------------------- #


class LeaseTest(_DataDirCase):
    def test_active_lease_blocks_a_second_claim(self):
        self._begin()
        self._persist("m-1")
        first = claim_retry(
            CONF,
            "m-1",
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({"client-b"}),
            history_uid=self.history,
        )
        self.assertIsNotNone(first)
        second = claim_retry(
            CONF,
            "m-1",
            owner="client-c",
            client_ref=REF,
            active_owners=frozenset({"client-c"}),
            history_uid=self.history,
        )
        self.assertIsNone(second, "the lease must exclude a second sender")
        self.assertEqual(delivery_record(CONF, "m-1")["attempt"], 2)

    def test_a_lease_left_by_a_provably_dead_process_is_reclaimed_at_once(self):
        self._begin()
        self._persist("m-1")
        start = time.time()
        # This is the token shape a real crashed process leaves behind. The
        # boot id cannot match the running machine, so claim_identity proves
        # the owner dead without waiting for the five-second lease.
        dead_owner = (
            "linux:00000000000000000000000000000000:4026531836:999999:1"
            "|client-crashed"
        )
        with patch(
            "src.open_llm_vtuber.message_delivery.claim_owner",
            return_value=dead_owner,
        ):
            crashed = claim_retry(
                CONF,
                "m-1",
                owner="client-crashed",
                client_ref=REF,
                active_owners=frozenset({"client-crashed"}),
                history_uid=self.history,
                now=start,
            )
        self.assertEqual(crashed["attempt"], 2)
        self.assertEqual(crashed["lease_owner"], dead_owner)

        # The restarted process has a new client_uid, but the same client_ref.
        # It may take over inside the old lease because the owner is provably
        # dead. If the old process had already sent the frame, this is a
        # duplicate -- at-least-once, never exactly-once.
        restarted = claim_retry(
            CONF,
            "m-1",
            owner="client-restarted",
            client_ref=REF,
            active_owners=frozenset({"client-restarted"}),
            history_uid=self.history,
            now=start + 0.01,
        )
        self.assertIsNotNone(restarted)
        self.assertEqual(restarted["attempt"], 3)
        self.assertEqual(restarted["retry_count"], 2)
        self.assertEqual(restarted["owner"], "client-restarted")

    def test_a_legacy_lease_without_a_process_token_waits_for_expiry(self):
        self._begin()
        self._persist("m-1")
        start = time.time()
        # A record written by the previous Stage 3 build stored the bare
        # client_uid. Without a process token liveness cannot be proven, so
        # the safe fallback is the full lease.
        with patch(
            "src.open_llm_vtuber.message_delivery.claim_owner",
            return_value="client-b",
        ):
            first = claim_retry(
                CONF,
                "m-1",
                owner="client-b",
                client_ref=REF,
                active_owners=frozenset({"client-b"}),
                history_uid=self.history,
                now=start,
            )
        self.assertEqual(first["lease_owner"], "client-b")
        self.assertIsNone(
            claim_retry(
                CONF,
                "m-1",
                owner="client-b",
                client_ref=REF,
                active_owners=frozenset({"client-b"}),
                history_uid=self.history,
                now=start + 0.01,
            ),
            "an unprovable legacy lease must not be stolen",
        )
        later = claim_retry(
            CONF,
            "m-1",
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({"client-b"}),
            history_uid=self.history,
            now=start + RETRY_LEASE_SECONDS + 1,
        )
        self.assertIsNotNone(later)
        self.assertEqual(later["attempt"], 3)

    def test_expired_lease_allows_the_lost_receipt_to_be_retried(self):
        self._begin()
        self._persist("m-1")
        start = time.time()
        first = claim_retry(
            CONF,
            "m-1",
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({"client-b"}),
            history_uid=self.history,
            now=start,
        )
        self.assertEqual(first["attempt"], 2)

        # Inside the sending window: still leased.
        self.assertIsNone(
            claim_retry(
                CONF,
                "m-1",
                owner="client-b",
                client_ref=REF,
                active_owners=frozenset({"client-b"}),
                history_uid=self.history,
                now=start + RETRY_LEASE_SECONDS / 2,
            )
        )
        # After the window, the receipt that never arrived no longer blocks a
        # retry: the message is recoverable, with the same msg_id and attempt+1.
        second = claim_retry(
            CONF,
            "m-1",
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({"client-b"}),
            history_uid=self.history,
            now=start + RETRY_LEASE_SECONDS + 1,
        )
        self.assertIsNotNone(second)
        self.assertEqual(second["attempt"], 3)

    def test_a_refused_transport_send_returns_the_retry_budget(self):
        self._begin()
        self._persist("m-1")
        first = claim_retry(
            CONF,
            "m-1",
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({"client-b"}),
            history_uid=self.history,
        )
        self.assertEqual(first["attempt"], 2)
        self.assertEqual(first["retry_count"], 1)

        released = release_retry_claim(
            CONF,
            "m-1",
            owner="client-b",
            attempt=first["attempt"],
        )
        self.assertTrue(released)
        after_release = delivery_record(CONF, "m-1")
        self.assertEqual(after_release["retry_count"], 0)
        self.assertEqual(after_release["lease_until"], 0.0)
        # The attempt counter is monotonic even though the frame never left.
        self.assertEqual(after_release["attempt"], 2)

        second = claim_retry(
            CONF,
            "m-1",
            owner="client-b",
            client_ref=REF,
            active_owners=frozenset({"client-b"}),
            history_uid=self.history,
        )
        self.assertIsNotNone(second)
        self.assertEqual(second["attempt"], 3)
        self.assertEqual(second["retry_count"], 1)

    def test_claim_refuses_a_delivered_message(self):
        self._begin()
        self._ack(seq=1, final=True)
        self.assertIsNone(
            claim_retry(
                CONF,
                "m-1",
                owner=OWNER,
                client_ref=REF,
                active_owners=frozenset(),
                history_uid=self.history,
            )
        )


CLAIM_CHILD = r"""
import json
import os
import sys
import time

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

from src.open_llm_vtuber.message_delivery import claim_retry

ready_dir = payload["ready_dir"]
open(os.path.join(ready_dir, payload["slot"] + ".ready"), "w").close()
deadline = time.time() + 20
while not os.path.exists(os.path.join(ready_dir, "start")):
    if time.time() > deadline:
        break
    time.sleep(0.005)

claimed = claim_retry(
    payload["conf"],
    payload["msg_id"],
    owner=payload["owner"],
    client_ref=payload["client_ref"],
    active_owners=frozenset(),
    history_uid=payload["history_uid"],
    base_dir=payload["base_dir"],
)

with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "claimed": claimed is not None,
            "attempt": (claimed or {}).get("attempt"),
        },
        handle,
    )

# Stay alive until the peer has also attempted. A dead process's lease is now
# reclaimable immediately, so a process that exited right after claiming would
# no longer prove LIVE exclusion; the test must keep both claimants alive.
deadline = time.time() + 20
while not os.path.exists(payload["peer_result_path"]):
    if time.time() > deadline:
        break
    time.sleep(0.005)
"""


CLAIM_AND_EXIT_CHILD = r"""
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

from src.open_llm_vtuber.message_delivery import claim_retry

claimed = claim_retry(
    payload["conf"],
    payload["msg_id"],
    owner=payload["owner"],
    client_ref=payload["client_ref"],
    active_owners=frozenset(),
    history_uid=payload["history_uid"],
    base_dir=payload["base_dir"],
)

with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "claimed": claimed is not None,
            "attempt": (claimed or {}).get("attempt"),
            "lease_owner": (claimed or {}).get("lease_owner"),
        },
        handle,
    )
"""


class CrossProcessClaimTest(_DataDirCase):
    """15/6: two server processes must not both send the same message."""

    def test_a_claim_left_by_an_exited_process_is_reclaimed_immediately(self):
        base_dir = os.path.join(self._temp.name, "state")
        os.makedirs(base_dir, exist_ok=True)
        begin_attempt(
            CONF,
            "m-dead",
            turn_id="turn-dead",
            owner=OWNER,
            client_ref=REF,
            history_uid=self.history,
            base_dir=base_dir,
        )
        result_path = os.path.join(self._temp.name, "claim-dead.json")
        payload = json.dumps(
            {
                "conf": CONF,
                "msg_id": "m-dead",
                "owner": "client-dead",
                "client_ref": REF,
                "history_uid": self.history,
                "base_dir": base_dir,
                "result_path": result_path,
            }
        )
        process = subprocess.run(
            [sys.executable, "-c", CLAIM_AND_EXIT_CHILD, ROOT, self._temp.name, payload],
            cwd=self._temp.name,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            process.returncode, 0, process.stderr.decode("utf-8", "replace")[:2000]
        )
        with open(result_path, "r", encoding="utf-8") as handle:
            child = json.load(handle)
        self.assertTrue(child["claimed"])
        self.assertEqual(child["attempt"], 2)
        self.assertIn("|", child["lease_owner"], "the lease carries a process token")

        # The child is gone, but its five-second lease is still in the future.
        # A restarted process must not be stranded for that whole window.
        now = time.time()
        reclaimed = claim_retry(
            CONF,
            "m-dead",
            owner="client-parent",
            client_ref=REF,
            active_owners=frozenset(),
            history_uid=self.history,
            base_dir=base_dir,
            now=now + 0.01,
        )
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed["attempt"], 3)
        self.assertEqual(reclaimed["owner"], "client-parent")
        self.assertEqual(reclaimed["retry_count"], 2)
        self.assertGreater(reclaimed["lease_until"], now + 0.01)
        self.assertIn("|", reclaimed["lease_owner"])
        self.assertNotEqual(reclaimed["lease_owner"], child["lease_owner"])


    def test_two_processes_cannot_claim_the_same_message(self):
        base_dir = os.path.join(self._temp.name, "state")
        os.makedirs(base_dir, exist_ok=True)
        begin_attempt(
            CONF,
            "m-race",
            turn_id="turn-race",
            owner=OWNER,
            client_ref=REF,
            history_uid=self.history,
            base_dir=base_dir,
        )
        ready_dir = os.path.join(self._temp.name, "ready")
        os.makedirs(ready_dir, exist_ok=True)

        results = []
        processes = []
        for slot in ("a", "b"):
            result_path = os.path.join(self._temp.name, f"claim-{slot}.json")
            results.append(result_path)
            peer = "b" if slot == "a" else "a"
            payload = json.dumps(
                {
                    "conf": CONF,
                    "msg_id": "m-race",
                    "owner": f"client-{slot}",
                    "client_ref": REF,
                    "history_uid": self.history,
                    "base_dir": base_dir,
                    "ready_dir": ready_dir,
                    "slot": slot,
                    "result_path": result_path,
                    "peer_result_path": os.path.join(
                        self._temp.name, f"claim-{peer}.json"
                    ),
                }
            )
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", CLAIM_CHILD, ROOT, self._temp.name, payload],
                    cwd=self._temp.name,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )

        for _ in range(20 * 200):
            if all(
                os.path.exists(os.path.join(ready_dir, f"{slot}.ready"))
                for slot in ("a", "b")
            ):
                break
            time.sleep(0.01)
        open(os.path.join(ready_dir, "start"), "w").close()

        for process in processes:
            _out, stderr = process.communicate(timeout=60)
            self.assertEqual(
                process.returncode, 0, stderr.decode("utf-8", "replace")[:2000]
            )

        outcomes = []
        for path in results:
            with open(path, "r", encoding="utf-8") as handle:
                outcomes.append(json.load(handle))
        winners = [outcome for outcome in outcomes if outcome["claimed"]]
        self.assertEqual(len(winners), 1, f"exactly one worker may claim: {outcomes}")

        record = delivery_record(CONF, "m-race", base_dir=base_dir)
        self.assertEqual(record["attempt"], 2, "attempt advances exactly once")
        self.assertEqual(record["retry_count"], 1)
        with open(
            get_delivery_state_path(CONF, base_dir=base_dir), "r", encoding="utf-8"
        ) as handle:
            json.load(handle)  # still parseable


# --------------------------------------------------------------------------- #
# 7. Restart: a settled message never becomes pending again
# --------------------------------------------------------------------------- #

STATE_CHILD = r"""
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

from src.open_llm_vtuber.message_delivery import delivery_record, unsettled_messages

record = delivery_record(payload["conf"], payload["msg_id"], base_dir=payload["base_dir"])
summary = unsettled_messages(
    payload["conf"],
    owner=payload["owner"],
    client_ref=payload["client_ref"],
    active_owners=frozenset(),
    base_dir=payload["base_dir"],
)

with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "status": (record or {}).get("status"),
            "attempt": (record or {}).get("attempt"),
            "ok": summary.ok,
            "pending": [item["msg_id"] for item in summary.messages],
        },
        handle,
    )
"""


class RestartRecoveryTest(_DataDirCase):
    def _fresh_process_view(self, msg_id):
        base_dir = os.path.join(self._temp.name, "state")
        os.makedirs(base_dir, exist_ok=True)
        result_path = os.path.join(self._temp.name, f"view-{msg_id}.json")
        payload = json.dumps(
            {
                "conf": CONF,
                "msg_id": msg_id,
                "owner": "client-after-restart",
                "client_ref": REF,
                "base_dir": base_dir,
                "result_path": result_path,
            }
        )
        process = subprocess.run(
            [sys.executable, "-c", STATE_CHILD, ROOT, self._temp.name, payload],
            cwd=self._temp.name,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            process.returncode, 0, process.stderr.decode("utf-8", "replace")[:2000]
        )
        with open(result_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_restart_keeps_a_settled_message_settled(self):
        base_dir = os.path.join(self._temp.name, "state")
        os.makedirs(base_dir, exist_ok=True)
        begin_attempt(
            CONF,
            "m-done",
            turn_id="turn-done",
            owner=OWNER,
            client_ref=REF,
            history_uid=self.history,
            base_dir=base_dir,
        )
        outcome = apply_ack(
            CONF,
            "m-done",
            turn_id="turn-done",
            attempt=1,
            status=ACK_RECEIVED,
            seq=1,
            final=True,
            owner=OWNER,
            base_dir=base_dir,
        )
        self.assertEqual(outcome.reason, "delivered")

        view = self._fresh_process_view("m-done")

        self.assertEqual(view["status"], STATUS_DELIVERED)
        self.assertEqual(view["pending"], [], "a restart must not reopen it")
        self.assertTrue(view["ok"])

    def test_restart_keeps_an_unsettled_message_recoverable(self):
        base_dir = os.path.join(self._temp.name, "state")
        os.makedirs(base_dir, exist_ok=True)
        begin_attempt(
            CONF,
            "m-open",
            turn_id="turn-open",
            owner=OWNER,
            client_ref=REF,
            history_uid=self.history,
            base_dir=base_dir,
        )

        view = self._fresh_process_view("m-open")

        self.assertEqual(view["status"], STATUS_DELIVERING, "not falsely settled")
        self.assertEqual(view["pending"], ["m-open"])
        self.assertEqual(view["attempt"], 1)


# --------------------------------------------------------------------------- #
# 9. Legacy clients: no hello, no summary, no resend, unchanged behaviour
# --------------------------------------------------------------------------- #


class LegacyClientTest(_DataDirCase):
    def test_a_client_that_never_says_hello_gets_no_summary_and_no_resend(self):
        self._begin()
        self._persist("m-1")
        socket = _Socket()
        handler = _handler("client-legacy")
        # It only ever sends receipts (if any); nothing announces a protocol.
        asyncio.run(
            handler._route_message(
                socket, "client-legacy", {"type": "message-ack", "msg_id": "m-1"}
            )
        )
        self.assertEqual(socket.of_type("delivery-summary"), [])
        self.assertEqual(socket.of_type("ai-final"), [])
        # And its (malformed) receipt changed nothing.
        self.assertEqual(delivery_record(CONF, "m-1")["status"], STATUS_DELIVERING)

    def test_a_hello_without_ack_support_gets_the_summary_but_no_resend(self):
        self._begin()
        self._persist("m-1")
        socket = _Socket()

        self._hello(["client-b"], socket, client_uid="client-b", ack=False)

        summary = socket.of_type("delivery-summary")
        self.assertEqual(len(summary), 1)
        self.assertFalse(summary[0]["redelivery"])
        self.assertEqual(len(summary[0]["unsettled"]), 1)
        self.assertEqual(socket.of_type("ai-final"), [])
        self.assertEqual(delivery_record(CONF, "m-1")["attempt"], 1)

    def test_a_truthy_string_ack_does_not_enable_redelivery(self):
        self._begin()
        self._persist("m-1")
        socket = _Socket()

        # A buggy/legacy client may serialize booleans as strings. "false" is
        # truthy in Python; it must not turn on a protocol it cannot use.
        self._hello(["client-b"], socket, client_uid="client-b", ack="false")

        summary = socket.of_type("delivery-summary")
        self.assertEqual(len(summary), 1)
        self.assertFalse(summary[0]["redelivery"])
        self.assertEqual(socket.of_type("ai-final"), [])
        self.assertEqual(delivery_record(CONF, "m-1")["attempt"], 1)

    def test_client_hello_is_not_a_chat_trigger(self):
        handler = WebSocketHandler.__new__(WebSocketHandler)
        handlers = WebSocketHandler._init_message_handlers(handler)
        self.assertIn("client-hello", handlers)
        self.assertEqual(handlers["client-hello"].__name__, "_handle_client_hello")
        for forbidden in ("text-input", "mic-audio-end", "ai-speak-signal"):
            self.assertIsNot(handlers["client-hello"], handlers[forbidden])
        self.assertIsNot(handlers["client-hello"], handlers["message-ack"])


# --------------------------------------------------------------------------- #
# 8 + 13. Unreadable store: UNKNOWN, never "nothing pending", never a resend
# --------------------------------------------------------------------------- #


class UnreadableStoreTest(_DataDirCase):
    def _corrupt_store(self):
        path = get_delivery_state_path(CONF)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"messages": {not json')
        return path

    def test_corrupt_store_is_reported_as_unknown_and_nothing_is_resent(self):
        self._corrupt_store()
        self._persist("m-1")
        socket = _Socket()

        self._hello(["client-b"], socket, client_uid="client-b")

        summary = socket.of_type("delivery-summary")
        self.assertEqual(len(summary), 1)
        self.assertFalse(summary[0]["ok"], "unknown state must not look settled")
        self.assertEqual(summary[0]["unsettled"], [])
        self.assertEqual(socket.of_type("ai-final"), [])

    def test_corrupt_store_is_never_silently_overwritten(self):
        path = self._corrupt_store()
        record = begin_attempt(
            CONF,
            "m-1",
            turn_id="turn-1",
            owner=OWNER,
            client_ref=REF,
            history_uid=self.history,
        )
        # Refusing is the honest outcome: overwriting a damaged store would
        # erase every other in-flight attempt and let a later receipt "settle"
        # a message the store had already recorded.
        self.assertIsNone(record)
        with open(path, "r", encoding="utf-8") as handle:
            self.assertIn("not json", handle.read(), "the damaged data is preserved")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
