"""Durable delivery receipts for proactive reminders (append-only journal).

Problem this closes: the dispatcher marks a reminder ``done`` only after a
delivered turn. If that completion write fails, the row stays ``pending``
with a live claim and becomes claimable again after the lease expires, so
the same reminder can be sent twice. A crash between the delivered turn and
the completion write has the same effect.

Fix: record the delivery fact in a small append-only journal *at the delivery
boundary*, before the completion write. A later claim (including after a
restart) sees the receipt, reconciles the row to ``done``, and refuses to
claim/deliver it again.

Guarantee, stated precisely (no exactly-once claim):

* At-most-once **after** a receipt is durable: once ``record_delivery``
  returns True (fsynced), a retry or restart will not deliver that id again.
* At-least-once **before** the receipt: the transport confirms no delivery
  ack, so a crash between the streamed message and the journal append can
  still resend. Exactly-once delivery is NOT provided by this module.
* Cross-process claim exclusivity comes from the character-state lease, not
  from this journal: the claim write is serialised by an advisory ``flock``
  on ``character_state/<conf>.json.lock`` and a row only becomes claimable
  again once ``claim_until`` passes. An owner whose lease lapses (stalled
  process, no heartbeat) can therefore be taken over mid-turn; the journal
  then blocks every FURTHER resend once its receipt is durable.

The journal is append-only and never rewritten or pruned in place. Rewriting
would race a concurrent append from another process and could drop a receipt,
reopening the duplicate window. One line per delivered reminder is small;
stale lines for rows already ``done``/``cancelled`` are harmless. A trailing
fragment left by a writer that died mid-append is never glued to the next
receipt: the append is separated from it, so a torn line can only cost the
receipts that never reached ``fsync`` -- never a later, complete one.

Appends take an advisory exclusive ``flock`` on the journal file and
``delivered_ids`` takes it shared, so a reader can no longer observe a
partially written line and discard it as corrupt -- which would have turned a
durable receipt into an invisible one and reopened the duplicate window.

Fail-soft by contract: any I/O problem returns False/empty and the caller
keeps the pre-existing behavior (completion retry, documented duplicate risk
only when the completion write ALSO fails).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional, Set

from loguru import logger

try:  # POSIX advisory locks; Windows keeps the previous unlocked behaviour.
    import fcntl
except ImportError:  # pragma: no cover - Windows-only fallback
    fcntl = None  # type: ignore[assignment]

STATE_DIR = "character_state"
JOURNAL_SUFFIX = ".deliveries.jsonl"

_MAX_ID_CHARS = 128


def _sanitize(conf_uid: str) -> str:
    from .chat_history_manager import _sanitize_path_component

    return _sanitize_path_component(conf_uid)


def get_delivery_journal_path(
    conf_uid: str, base_dir: Optional[str] = None
) -> str:
    """On-disk receipt journal for one character (safe path component)."""
    if not conf_uid:
        raise ValueError("conf_uid cannot be empty")
    directory = STATE_DIR if base_dir is None else base_dir
    return os.path.join(directory, f"{_sanitize(conf_uid)}{JOURNAL_SUFFIX}")


def _now_iso(moment: Optional[datetime] = None) -> str:
    stamp = moment if moment is not None else datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat(timespec="seconds")


def _fsync_directory(path: str) -> None:
    """Best-effort directory fsync so a newly created journal is durable."""
    try:
        descriptor = os.open(path or ".", os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _ends_with_newline(path: str) -> bool:
    """True when the journal's last byte terminates a line (or it is empty).

    Append-only journal + a writer killed mid-append = a trailing fragment
    without its newline. Appending the next receipt straight after it glues
    the two into one unparseable line, and ``delivered_ids`` silently skips
    corrupt lines -- so the FRESH receipt would be invisible and the reminder
    could be sent again after a failed completion. One leading newline costs
    a single ignored blank line in that rare case and nothing otherwise.

    Deliberately lock-free: when two writers both see a torn tail they both
    prepend a newline, which produces two blank lines instead of one. Blank
    lines are already skipped by the reader, so there is nothing to agree on
    and no lock to hold across the append.
    """
    try:
        with open(path, "rb") as handle:
            if handle.seek(0, os.SEEK_END) == 0:
                return True
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) == b"\n"
    except OSError:
        return True


def _lock_journal(descriptor: int, exclusive: bool) -> None:
    """Advisory lock on the journal file itself (best effort).

    The journal is only ever appended to and never replaced, so locking the
    file ties the lock to exactly the bytes being read or written -- no sidecar
    and no second lock registry. A reader that takes ``LOCK_SH`` cannot observe
    a half-written line: appends are serialised by ``LOCK_EX``, and the bytes of
    one ``record_delivery`` call become visible to readers only as a unit. That
    matters because ``delivered_ids`` silently skips corrupt lines, so a torn
    read would turn a durable receipt into an invisible one and let the reminder
    be sent again.

    Fail-soft: an unavailable lock (Windows, unusual filesystem) degrades to
    the previous lock-free behaviour instead of failing the receipt.
    """
    if fcntl is None:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    except OSError as error:  # pragma: no cover - unusual OS failure
        logger.warning(
            "Delivery journal lock not acquired: type={}", type(error).__name__
        )


def _write_all(descriptor: int, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        chunk = os.write(descriptor, payload[written:])
        if chunk <= 0:  # pragma: no cover - defensive
            raise OSError("short write on delivery journal")
        written += chunk


def record_delivery(
    conf_uid: str,
    intention_ids: Any = None,
    *,
    owner: str = "",
    now: Optional[datetime] = None,
    base_dir: Optional[str] = None,
) -> bool:
    """Durably append one receipt per delivered intention id.

    Returns True only when every line was written AND fsynced. False means
    "not durable": the caller must not rely on the receipt (and should still
    attempt the normal completion write).
    """
    ids = []
    for value in intention_ids or []:
        text = str(value or "").strip()[:_MAX_ID_CHARS]
        if text and text not in ids:
            ids.append(text)
    if not ids:
        return False
    try:
        path = get_delivery_journal_path(conf_uid, base_dir)
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        stamp = _now_iso(now)
        payload = "".join(
            json.dumps(
                {
                    "id": row_id,
                    "owner": str(owner or "")[:_MAX_ID_CHARS],
                    "delivered_at": stamp,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for row_id in ids
        ).encode("utf-8")
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            _lock_journal(descriptor, True)
            # Never concatenate onto a fragment left by a crashed writer. The
            # check is inside the lock, so two live appends cannot both see the
            # same fragment and both prepend a separator.
            if not _ends_with_newline(path):
                payload = b"\n" + payload
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)  # releases the lock
        _fsync_directory(directory)
        return True
    except Exception as error:
        logger.warning(
            "Reminder delivery receipt not persisted: type={}",
            type(error).__name__,
        )
        return False


def delivered_ids(
    conf_uid: str, base_dir: Optional[str] = None
) -> Set[str]:
    """Every id with a durable delivery receipt; corrupt lines are ignored."""
    try:
        path = get_delivery_journal_path(conf_uid, base_dir)
    except ValueError:
        return set()
    try:
        found: Set[str] = set()
        with open(path, "r", encoding="utf-8") as handle:
            # Shared lock: block while another process is mid-append instead of
            # reading a line that is not complete yet. Released on close.
            _lock_journal(handle.fileno(), False)
            for line in handle:
                entry = line.strip()
                if not entry:
                    continue
                try:
                    parsed = json.loads(entry)
                except (TypeError, ValueError):
                    continue
                if not isinstance(parsed, dict):
                    continue
                value = str(parsed.get("id") or "").strip()
                if value:
                    found.add(value)
        return found
    except FileNotFoundError:
        return set()
    except Exception as error:
        logger.warning(
            "Reminder delivery journal unreadable: type={}",
            type(error).__name__,
        )
        return set()


__all__ = [
    "JOURNAL_SUFFIX",
    "STATE_DIR",
    "delivered_ids",
    "get_delivery_journal_path",
    "record_delivery",
]
