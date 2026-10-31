"""Durable delivery state per logical message (Stage 2: ACK received/dropped).

Stage 3 adds the recovery half: a client that reconnects announces itself with
``client-hello``, receives a bounded summary of the messages it still owns, and
the server schedules a REDELIVERY (same ``msg_id``, attempt + 1) for each one.
The claim that authorises a redelivery is an atomic lease, so two server
processes (or two live connections) can never send the same message at the
same moment.

What this store is for
----------------------
A logical assistant message has three identities on the wire:

* ``msg_id`` -- the LOGICAL message. Stable across retries, persisted on the
  history row, and the only key a retry may reuse.
* ``turn_id`` -- one delivery attempt's turn (the existing ``request_id``).
* ``attempt`` -- a monotonically increasing counter for that ``msg_id``.

The backend needs to know whether the client took the frames, and the only
honest signal it can get is the client's own receipt. This module keeps one
durable record per ``msg_id`` and applies those receipts:

    delivering --frames sent, nothing settled yet
    delivered  --the client acknowledged the FINAL frame of the ACTIVE attempt
    retry_pending --the client acknowledged a frame as DROPPED

Strictly what an ACK means
--------------------------
``received`` is proof of ONE thing: the client parsed a frame and put it in
its own message buffer. It is NOT proof that text was rendered, that audio
finished playing, or that the user looked at the screen. Nothing in this
module may be reported as read/heard proof.

No exactly-once claim
---------------------
This is at-least-once delivery with idempotent client-side rendering. A crash
between "frames sent" and "receipt recorded" leaves the record ``delivering``,
so a later attempt may resend; the client dedupes by ``msg_id``. Exactly-once
delivery is NOT provided and must not be claimed.

Rules enforced here
-------------------
* Only an ACK matching ``msg_id`` + ``turn_id`` + the ACTIVE ``attempt`` from
  the client that OWNS the delivery may change a record.
* Duplicate ACKs are idempotent (no second write, no error).
* ACKs from a stale attempt, another turn, another client or an unknown id are
  refused and leave the record untouched.
* State never moves backwards: a late ``dropped`` cannot undo ``delivered``,
  and ``received`` cannot resurrect an attempt already marked retry_pending.
* A retry can only advance the attempt counter (``begin_attempt``); attempts
  are never reused or rewound.
* Failures are explicit: :func:`apply_ack` reports ``persisted=False`` when the
  record could not be read or written, so a caller must never report that ACK
  as recorded.

Concurrency
-----------
The store is a single JSON document, rewritten in full per ACK. Every
read-modify-write runs under the same cross-process lock used by
``character_state`` (in-process RLock + advisory ``flock`` on a sidecar), so
two server processes sharing one data directory cannot both settle the same
message. Records are bounded; the oldest are pruned once the cap is reached.

Recovery (Stage 3) is bounded too: a redelivery needs an atomic lease
(:func:`claim_retry`), reuses the same ``msg_id`` with attempt + 1, and stops
after ``MAX_RETRIES``. A restart changes nothing about what is settled --
``delivered`` stays ``delivered`` and is never re-sent.

Stage 4 makes a lease recoverable after a process crash. ``lease_owner`` carries
the same process token used by reminder claims (:mod:`claim_identity`), so a
claim left by a process that is provably dead can be taken over before the
lease expires. A live process, a legacy lease without a token, or an
unreadable ``/proc`` still wait for the full lease. If the dead process had
already put the frame on the wire, the takeover may send the same ``msg_id``
again: this remains at-least-once, never exactly-once.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from .claim_identity import claim_owner, owner_is_dead

STATE_DIR = "character_state"
STATE_SUFFIX = ".message_delivery.json"

# Bounded store: an entry is only needed while a message can still be retried.
MAX_TRACKED_MESSAGES = 200
_MAX_REASON_CHARS = 200

# Recovery bounds (Stage 3). A message may be redelivered on reconnect, but
# only a finite number of times and never while another worker holds the
# lease. Both limits exist so a client that keeps disappearing cannot turn the
# reconnect path into an unbounded resend loop.
# The lease only has to cover the SEND itself (one text frame): it exists to
# stop a second worker from sending the same message at the same moment, not to
# pace later reconnects. A short window means an unstable connection that
# reconnects twice can still recover its message seconds later.
MAX_RETRIES = 5
RETRY_LEASE_SECONDS = 5.0
# Summary is a diagnostic/repair list, not a transcript: one reconnect must not
# ask a client to reconcile hundreds of messages.
MAX_SUMMARY_MESSAGES = 20
_MAX_CLIENT_REF_CHARS = 64

STATUS_DELIVERING = "delivering"
STATUS_DELIVERED = "delivered"
STATUS_RETRY_PENDING = "retry_pending"

ACK_RECEIVED = "received"
ACK_DROPPED = "dropped"
ACK_STATUSES = frozenset({ACK_RECEIVED, ACK_DROPPED})

# Outcome reasons. Not exhaustive: they are diagnostic strings, never control
# flow for the client.
REASON_DELIVERED = "delivered"
REASON_PARTIAL = "partial"
REASON_RETRY_PENDING = "retry-pending"
REASON_DUPLICATE = "duplicate"
REASON_ALREADY_DELIVERED = "already-delivered"
REASON_UNKNOWN_MESSAGE = "unknown-message"
REASON_STALE_ATTEMPT = "stale-attempt"
REASON_UNKNOWN_ATTEMPT = "unknown-attempt"
REASON_TURN_MISMATCH = "turn-mismatch"
REASON_OWNER_MISMATCH = "owner-mismatch"
REASON_INVALID_MSG_ID = "invalid-msg-id"
REASON_INVALID_ATTEMPT = "invalid-attempt"
REASON_INVALID_STATUS = "invalid-status"
REASON_INVALID_FINAL = "invalid-final"
REASON_INVALID_SEQ = "invalid-seq"
REASON_STATE_UNREADABLE = "state-unreadable"
REASON_PERSIST_FAILED = "persist-failed"
REASON_REDELIVERING = "redelivering"
# Claim refusals (claim_retry). Not an ACK outcome: nothing was sent, so
# nothing was settled either.
CLAIM_DELIVERED = "already-delivered"
CLAIM_LEASE_HELD = "lease-held"
CLAIM_RETRY_EXHAUSTED = "retry-exhausted"
CLAIM_UNKNOWN = "unknown-message"
CLAIM_NOT_OWNER = "owner-mismatch"
CLAIM_HISTORY_MISMATCH = "history-mismatch"


class DeliveryStateError(Exception):
    """The on-disk delivery store exists but cannot be trusted.

    Raised instead of silently starting from an empty document: overwriting a
    damaged store would erase every in-flight attempt's state and let a later
    ACK "complete" a message the store had already settled.
    """


@dataclass(frozen=True)
class AckOutcome:
    """Result of applying one client receipt.

    ``applied`` is True only when the stored record changed.
    ``persisted`` is False ONLY when the store could not be read or written;
    a refused (stale/duplicate/unknown) ACK leaves the store untouched and
    reports True, because the store still says exactly what it said before.
    Callers must never report an ACK as recorded when ``persisted`` is False.
    """

    applied: bool
    persisted: bool
    reason: str
    attempt: int
    state: Optional[Dict[str, Any]] = None


def get_delivery_state_path(conf_uid: str, base_dir: Optional[str] = None) -> str:
    """On-disk delivery state for one character (safe path component)."""
    if not conf_uid:
        raise ValueError("conf_uid cannot be empty")
    from .chat_history_manager import _sanitize_path_component

    directory = STATE_DIR if base_dir is None else base_dir
    return os.path.join(
        directory, f"{_sanitize_path_component(conf_uid)}{STATE_SUFFIX}"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalize_client_ref(value: Any) -> str:
    """Stable client-instance id supplied by ``client-hello`` (may be "")."""
    if not isinstance(value, str):
        return ""
    return value.strip()[:_MAX_CLIENT_REF_CHARS]


def _record_matches_principal(
    record: Dict[str, Any],
    *,
    owner: str,
    client_ref: str = "",
    active_owners: Optional[Iterable[str]] = None,
) -> bool:
    """May this connection act on this record?

    Two ways to qualify, and only two:

    1. it is the connection that OWNS the delivery; or
    2. it is the SAME CLIENT INSTANCE (persisted ``client_ref``) as the owner,
       and that owner is no longer connected. This is what lets a reconnected
       tab settle and retry its own message after the backend handed it a new
       ``client_uid``.

    Rule 2 deliberately requires the previous owner to be offline: with two
    live connections of the same browser (two tabs), neither can settle the
    other's in-flight message. A record with no ``client_ref`` (delivered to a
    connection that never sent ``client-hello``, e.g. a legacy client) can
    only ever be settled by its owner.

    ``client_ref`` is NOT authentication. It is an unauthenticated bearer
    correlation token supplied by the client (this project has no user login
    layer); anyone who learns it can act as that client instance after its
    owner disconnects. The live-owner check above bounds that window, but does
    not turn the token into an identity proof.
    """
    record_owner = str(record.get("owner") or "")
    if record_owner == str(owner or ""):
        return True
    reference = _normalize_client_ref(client_ref)
    record_reference = _normalize_client_ref(record.get("client_ref"))
    if reference and record_reference and reference == record_reference:
        live = {str(item) for item in (active_owners or ())}
        return record_owner not in live
    return False


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        try:
            return int(value.strip())
        except ValueError:  # pragma: no cover - defensive
            return None
    return None


def _get_lock(path: str):
    """The shared cross-process lock for one store file (lazy import)."""
    from .character_state import _get_state_lock

    return _get_state_lock(path)


def _is_resendable(record: Dict[str, Any]) -> bool:
    """May this message be sent again on reconnect?

    A message the client has already taken frames of -- but has not settled --
    is NOT resent: the client's own history resync carries the authoritative
    text, and re-sending would replay audio for something it already has. Only
    a message with NO receipt at all (the socket died before anything arrived)
    or one the client explicitly DROPPED needs a redelivery.
    """
    status = record.get("status")
    if status == STATUS_RETRY_PENDING:
        return True
    if status != STATUS_DELIVERING:
        return False
    acked = _as_int(record.get("acked_seq"))
    return (acked if acked is not None else -1) < 0


def _empty_state() -> Dict[str, Any]:
    return {"version": 1, "messages": {}}


def _read_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return _empty_state()
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise DeliveryStateError("delivery state is not an object")
    messages = data.get("messages")
    if messages is None:
        messages = {}
        data["messages"] = messages
    if not isinstance(messages, dict):
        raise DeliveryStateError("delivery state 'messages' is not an object")
    return data


def _prune(messages: Dict[str, Any]) -> None:
    """Drop the oldest records above the cap (insertion order = recency)."""
    overflow = len(messages) - MAX_TRACKED_MESSAGES
    if overflow <= 0:
        return
    for key in list(messages.keys())[:overflow]:
        messages.pop(key, None)


def _write_state(path: str, data: Dict[str, Any]) -> None:
    """Atomically and durably replace the store. Raises on any failure."""
    messages = data.get("messages")
    if isinstance(messages, dict):
        _prune(messages)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_delivery_state(
    conf_uid: str, base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Read the whole store; missing/corrupt yields an empty document."""
    try:
        path = get_delivery_state_path(conf_uid, base_dir)
    except ValueError:
        return _empty_state()
    try:
        with _get_lock(path):
            return _read_state(path)
    except Exception as error:
        logger.warning(
            "Message delivery state unreadable: type={}",
            type(error).__name__,
        )
        return _empty_state()


def delivery_record(
    conf_uid: str, msg_id: str, base_dir: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Current record for one logical message, or None."""
    text = _normalize_id(msg_id)
    if not text:
        return None
    record = load_delivery_state(conf_uid, base_dir).get("messages", {}).get(text)
    return dict(record) if isinstance(record, dict) else None


def begin_attempt(
    conf_uid: str,
    msg_id: str,
    *,
    turn_id: str,
    owner: str,
    client_ref: str = "",
    history_uid: str = "",
    base_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Open (or advance) the delivery attempt for one logical message.

    Returns the fresh record, or None when the state could not be persisted —
    in which case the caller must not claim ACK support for this turn: no
    receipt can match a record that was never written. The attempt counter is
    derived from the store, never from the caller, so a retry of the same
    ``msg_id`` always increases it and a caller cannot rewind it.

    Fail-soft by contract (same policy as every other state store on this
    path): a broken store never blocks the reply itself.
    """
    text = _normalize_id(msg_id)
    if not text:
        return None
    try:
        path = get_delivery_state_path(conf_uid, base_dir)
        with _get_lock(path):
            data = _read_state(path)
            messages = data["messages"]
            previous = messages.pop(text, None)
            attempt = 1
            if isinstance(previous, dict):
                previous_attempt = _as_int(previous.get("attempt")) or 0
                attempt = max(1, previous_attempt + 1)
            record = {
                "msg_id": text,
                "history_uid": str(history_uid or ""),
                "turn_id": str(turn_id or ""),
                "owner": str(owner or ""),
                # Persisted so a reconnect (which gets a NEW client_uid) can
                # still be recognised as the client that took this message.
                "client_ref": _normalize_client_ref(client_ref),
                "attempt": attempt,
                "status": STATUS_DELIVERING,
                "acked_seq": -1,
                "reason": "",
                # Recovery bookkeeping; a fresh turn starts unleased and
                # unretried, so the counts always describe redeliveries only.
                "retry_count": 0,
                "lease_owner": "",
                "lease_until": 0.0,
                "updated_at": _now_iso(),
            }
            messages[text] = record
            _write_state(path, data)
            return dict(record)
    except Exception as error:
        logger.warning(
            "Message delivery state not persisted (begin, msg_id={}): type={}",
            text,
            type(error).__name__,
        )
        return None


def apply_ack(
    conf_uid: str,
    msg_id: Any,
    *,
    turn_id: Any = "",
    attempt: Any = None,
    status: Any = None,
    seq: Any = None,
    final: Any = False,
    reason: Any = "",
    owner: str = "",
    client_ref: str = "",
    active_owners: Optional[Iterable[str]] = None,
    base_dir: Optional[str] = None,
) -> AckOutcome:
    """Apply one client receipt to the message it names.

    Refusal reasons (state untouched): ``invalid-msg-id``, ``invalid-attempt``,
    ``invalid-status``, ``invalid-final``, ``invalid-seq``, ``unknown-message``,
    ``turn-mismatch``,
    ``owner-mismatch``, ``stale-attempt``, ``unknown-attempt``,
    ``already-delivered``, ``retry-pending`` (a late ``received`` for an
    attempt already marked dropped) and ``duplicate``.

    Ownership (Stage 3): an ACK is accepted from the owning connection, or from
    the same client instance after a reconnect -- see
    :func:`_record_matches_principal`. A receipt from any other live client is
    refused, so one connection can never settle another's message.

    ``received`` on the frame that carries ``final`` moves the ACTIVE attempt
    to ``delivered``; ``received`` on any earlier frame only records progress
    (``acked_seq``) and keeps ``delivering``, so a partially received message
    is never reported as settled.
    """
    text = _normalize_id(msg_id)
    attempt_value = _as_int(attempt)
    if not text:
        return AckOutcome(False, True, REASON_INVALID_MSG_ID, 0)
    if attempt_value is None or attempt_value < 1:
        return AckOutcome(False, True, REASON_INVALID_ATTEMPT, 0)
    if status not in ACK_STATUSES:
        return AckOutcome(False, True, REASON_INVALID_STATUS, attempt_value)
    if not isinstance(final, bool):
        # A truthy string such as "false" must never settle a message. The
        # wire protocol sends a JSON boolean; anything else is malformed.
        return AckOutcome(False, True, REASON_INVALID_FINAL, attempt_value)
    seq_value = _as_int(seq)
    if status == ACK_RECEIVED and final and (seq_value is None or seq_value < 0):
        # The closing frame always carries its content-frame sequence number.
        # Without it the receipt cannot identify the closing frame, so it must
        # not be accepted as proof that the whole message arrived.
        return AckOutcome(False, True, REASON_INVALID_SEQ, attempt_value)
    turn_value = str(turn_id or "")
    reason_value = str(reason or "")[:_MAX_REASON_CHARS]

    try:
        path = get_delivery_state_path(conf_uid, base_dir)
        with _get_lock(path):
            data = _read_state(path)
            messages = data["messages"]
            record = messages.get(text)
            if not isinstance(record, dict):
                return AckOutcome(False, True, REASON_UNKNOWN_MESSAGE, attempt_value)
            record_attempt = _as_int(record.get("attempt")) or 0
            snapshot = dict(record)
            if str(record.get("turn_id") or "") != turn_value:
                return AckOutcome(
                    False, True, REASON_TURN_MISMATCH, attempt_value, snapshot
                )
            if not _record_matches_principal(
                record,
                owner=owner,
                client_ref=client_ref,
                active_owners=active_owners,
            ):
                return AckOutcome(
                    False, True, REASON_OWNER_MISMATCH, attempt_value, snapshot
                )
            if attempt_value < record_attempt:
                # A receipt for a superseded attempt can never settle the
                # active one, and can never move state backwards.
                return AckOutcome(
                    False, True, REASON_STALE_ATTEMPT, attempt_value, snapshot
                )
            if attempt_value > record_attempt:
                # No such attempt was opened: refuse instead of creating one.
                return AckOutcome(
                    False, True, REASON_UNKNOWN_ATTEMPT, attempt_value, snapshot
                )

            current_status = record.get("status")
            applied = False

            if status == ACK_DROPPED:
                if current_status == STATUS_DELIVERED:
                    return AckOutcome(
                        False, True, REASON_ALREADY_DELIVERED, attempt_value, snapshot
                    )
                if current_status == STATUS_RETRY_PENDING:
                    return AckOutcome(
                        False, True, REASON_DUPLICATE, attempt_value, snapshot
                    )
                record["status"] = STATUS_RETRY_PENDING
                record["reason"] = reason_value
                applied = True
            else:  # ACK_RECEIVED
                if current_status == STATUS_RETRY_PENDING:
                    # This attempt was already declared dropped; only a NEW
                    # attempt may settle the message again.
                    return AckOutcome(
                        False, True, REASON_RETRY_PENDING, attempt_value, snapshot
                    )
                acked = _as_int(record.get("acked_seq"))
                acked = -1 if acked is None else acked
                if seq_value is not None and seq_value > acked:
                    acked = seq_value
                    applied = True
                record["acked_seq"] = acked
                if final and current_status != STATUS_DELIVERED:
                    record["status"] = STATUS_DELIVERED
                    record["reason"] = ""
                    applied = True
                if not applied:
                    return AckOutcome(
                        False, True, REASON_DUPLICATE, attempt_value, snapshot
                    )

            # A receipt proves the send window this lease covered is over. In
            # particular, a `dropped` receipt must make the message retryable
            # immediately; leaving the old lease in place would make a quick
            # reconnect wait for it even though nothing is being sent.
            record["lease_owner"] = ""
            record["lease_until"] = 0.0
            record["updated_at"] = _now_iso()
            messages[text] = record
            _write_state(path, data)
            if status == ACK_DROPPED:
                outcome_reason = REASON_RETRY_PENDING
            elif record.get("status") == STATUS_DELIVERED:
                outcome_reason = REASON_DELIVERED
            else:
                outcome_reason = REASON_PARTIAL
            return AckOutcome(True, True, outcome_reason, attempt_value, dict(record))
    except DeliveryStateError as error:
        logger.error(
            "MESSAGE_ACK_STATE_UNREADABLE msg_id={} turn_id={} attempt={} type={}",
            text,
            turn_value,
            attempt_value,
            type(error).__name__,
        )
        return AckOutcome(False, False, REASON_STATE_UNREADABLE, attempt_value)
    except Exception as error:
        logger.error(
            "MESSAGE_ACK_NOT_PERSISTED msg_id={} turn_id={} attempt={} "
            "type={}: state NOT recorded",
            text,
            turn_value,
            attempt_value,
            type(error).__name__,
        )
        return AckOutcome(False, False, REASON_PERSIST_FAILED, attempt_value)


def release_retry_claim(
    conf_uid: str,
    msg_id: Any,
    *,
    owner: str,
    attempt: Any,
    base_dir: Optional[str] = None,
) -> bool:
    """Undo the retry budget spent by a claim whose transport send was refused.

    A claim has to be durable BEFORE the frame is sent, otherwise two workers
    could send it. But if the socket refuses that frame, no redelivery actually
    happened; charging the bounded retry budget for it would let a flaky
    connection exhaust the message without ever receiving it. This releases the
    lease and gives the retry back, while keeping the attempt counter monotonic
    (the next claim uses attempt + 1).

    Only the current claim holder may release, and only while the record is
    still ``delivering``. A concurrent ACK or a newer claim makes this a no-op.
    """
    text = _normalize_id(msg_id)
    attempt_value = _as_int(attempt)
    if not text or attempt_value is None or attempt_value < 1:
        return False
    try:
        path = get_delivery_state_path(conf_uid, base_dir)
        with _get_lock(path):
            data = _read_state(path)
            messages = data["messages"]
            record = messages.get(text)
            if not isinstance(record, dict):
                return False
            if (_as_int(record.get("attempt")) or 0) != attempt_value:
                return False
            if str(record.get("owner") or "") != str(owner or ""):
                return False
            if record.get("status") != STATUS_DELIVERING:
                return False
            retries = _as_int(record.get("retry_count")) or 0
            record["retry_count"] = max(0, retries - 1)
            record["lease_owner"] = ""
            record["lease_until"] = 0.0
            record["reason"] = "send-failed"
            record["updated_at"] = _now_iso()
            messages[text] = record
            _write_state(path, data)
            return True
    except Exception as error:
        logger.warning(
            "MESSAGE_DELIVERY_RELEASE_NOT_PERSISTED msg_id={} attempt={} type={}: "
            "retry budget remains spent",
            text,
            attempt_value,
            type(error).__name__,
        )
        return False


@dataclass(frozen=True)
class DeliverySummary:
    """Messages one reconnecting client still owes an answer for.

    ``ok`` is False when the store could not be read: the list is then empty
    because the state is UNKNOWN, which is not the same thing as "nothing is
    pending", and a caller must not present it as such.
    """

    ok: bool
    messages: List[Dict[str, Any]]


def unsettled_messages(
    conf_uid: str,
    *,
    owner: str,
    client_ref: str = "",
    active_owners: Optional[Iterable[str]] = None,
    history_uid: str = "",
    limit: int = MAX_SUMMARY_MESSAGES,
    base_dir: Optional[str] = None,
) -> DeliverySummary:
    """Bounded list of messages this client may still need to receive.

    Only messages this principal is entitled to (see
    :func:`_record_matches_principal`) and that are NOT settled are returned.
    A settled (``delivered``) message is never re-sent just because a client
    reconnected: that would replay audio the user already accepted.

    ``history_uid`` (from ``client-hello``) scopes the summary to the session
    the client is actually displaying, so a message from another history is
    neither leaked nor pointlessly retried.
    """
    try:
        path = get_delivery_state_path(conf_uid, base_dir)
        with _get_lock(path):
            data = _read_state(path)
    except Exception as error:
        logger.error(
            "MESSAGE_DELIVERY_SUMMARY_UNREADABLE type={}: recovery state NOT known",
            type(error).__name__,
        )
        return DeliverySummary(False, [])

    wanted_history = str(history_uid or "")
    selected: List[Dict[str, Any]] = []
    for record in data.get("messages", {}).values():
        if not isinstance(record, dict):
            continue
        if record.get("status") not in (STATUS_DELIVERING, STATUS_RETRY_PENDING):
            continue
        if wanted_history and str(record.get("history_uid") or "") != wanted_history:
            continue
        if not _record_matches_principal(
            record,
            owner=owner,
            client_ref=client_ref,
            active_owners=active_owners,
        ):
            continue
        entry = dict(record)
        entry["resendable"] = _is_resendable(record)
        selected.append(entry)

    # Resendable messages first, then oldest first within each group. A
    # summary filled with partially received messages must not hide the one
    # message that is actually missing and can be repaired. Within a group the
    # client's own order of arrival is preserved.
    selected.sort(
        key=lambda item: (
            not bool(item.get("resendable")),
            str(item.get("updated_at") or ""),
        )
    )
    if limit >= 0:
        selected = selected[:limit]
    return DeliverySummary(True, selected)


def claim_retry(
    conf_uid: str,
    msg_id: Any,
    *,
    owner: str,
    client_ref: str = "",
    active_owners: Optional[Iterable[str]] = None,
    history_uid: str = "",
    now: Optional[float] = None,
    lease_seconds: float = RETRY_LEASE_SECONDS,
    base_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Atomically take the right to REDELIVER one message, or return None.

    Claiming is the only thing that authorises a resend, and it happens inside
    the same cross-process lock as every other write to this store, so two
    server processes sharing a data directory cannot both win the same
    message; the lease then keeps a second win impossible for
    ``lease_seconds`` (the sending window), even if the first process is slow.

    One exception is deliberately narrow: when the persisted ``lease_owner``
    carries a process token that proves the claiming process is dead (the
    same rule used for reminder claims), the lease may be taken over before it
    expires. This recovers a crash between claim and send without waiting five
    seconds, while a live process, a legacy owner without a token, or
    unreadable process data keeps the full lease. If the dead process had
    already sent the frame, the takeover duplicates it; the client dedupes by
    ``msg_id`` and no exactly-once claim is made.

    A successful claim:

    * advances ``attempt`` (monotonic; the client dedupes on ``msg_id``),
    * transfers ownership to the claiming connection, because that connection
      is now the one whose ACK must settle it,
    * resets the per-attempt receipt cursor (``acked_seq``),
    * counts the retry, so an undeliverable message stops after
      ``MAX_RETRIES`` instead of looping forever.

    A ``delivered`` record can never be claimed again, and neither can one the
    client already took frames of (:func:`_is_resendable`): a reconnect must not
    replay a message the client already has.
    """
    text = _normalize_id(msg_id)
    if not text:
        return None
    clock = time.time() if now is None else float(now)
    try:
        path = get_delivery_state_path(conf_uid, base_dir)
        with _get_lock(path):
            data = _read_state(path)
            messages = data["messages"]
            record = messages.get(text)
            if not isinstance(record, dict):
                return None
            if record.get("status") == STATUS_DELIVERED:
                # Settled: a reconnect must never replay what the client took.
                return None
            if not _is_resendable(record):
                # Partially received, or any other non-resendable state: the
                # client's history resync covers the text, and resending would
                # replay audio it already has.
                return None
            if history_uid and str(record.get("history_uid") or "") != str(history_uid):
                return None
            if not _record_matches_principal(
                record,
                owner=owner,
                client_ref=client_ref,
                active_owners=active_owners,
            ):
                return None
            retries = _as_int(record.get("retry_count")) or 0
            if retries >= MAX_RETRIES:
                return None
            try:
                leased_until = float(record.get("lease_until") or 0.0)
            except (TypeError, ValueError):
                leased_until = 0.0
            if leased_until > clock:
                lease_owner = str(record.get("lease_owner") or "")
                try:
                    dead_owner = owner_is_dead(lease_owner)
                except Exception:
                    # Liveness probing must never make a live lease look
                    # reclaimable. Unreadable process identity waits.
                    dead_owner = False
                if not dead_owner:
                    return None
                logger.warning(
                    "MESSAGE_DELIVERY_DEAD_LEASE_RECLAIMED msg_id={} "
                    "lease_owner={}",
                    text,
                    lease_owner,
                )

            attempt = max(1, (_as_int(record.get("attempt")) or 0) + 1)
            record.update(
                {
                    "attempt": attempt,
                    "status": STATUS_DELIVERING,
                    "acked_seq": -1,
                    "reason": REASON_REDELIVERING,
                    "retry_count": retries + 1,
                    "owner": str(owner or ""),
                    "client_ref": _normalize_client_ref(client_ref)
                    or _normalize_client_ref(record.get("client_ref")),
                    "lease_owner": claim_owner(str(owner or "")),
                    "lease_until": clock + float(lease_seconds),
                    "updated_at": _now_iso(),
                }
            )
            messages[text] = record
            _write_state(path, data)
            return dict(record)
    except Exception as error:
        logger.error(
            "MESSAGE_DELIVERY_CLAIM_NOT_PERSISTED msg_id={} type={}: no "
            "redelivery authorised",
            text,
            type(error).__name__,
        )
        return None


__all__ = [
    "ACK_DROPPED",
    "ACK_RECEIVED",
    "ACK_STATUSES",
    "AckOutcome",
    "DeliveryStateError",
    "DeliverySummary",
    "MAX_RETRIES",
    "MAX_SUMMARY_MESSAGES",
    "MAX_TRACKED_MESSAGES",
    "RETRY_LEASE_SECONDS",
    "STATE_DIR",
    "STATE_SUFFIX",
    "STATUS_DELIVERED",
    "STATUS_DELIVERING",
    "STATUS_RETRY_PENDING",
    "apply_ack",
    "begin_attempt",
    "claim_retry",
    "delivery_record",
    "get_delivery_state_path",
    "load_delivery_state",
    "release_retry_claim",
    "unsettled_messages",
]
