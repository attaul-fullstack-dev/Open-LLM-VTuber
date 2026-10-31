import os
import re
import json
import time
import uuid
import threading
from datetime import datetime, timezone
from typing import Iterable, Literal, List, TypedDict, Optional
from loguru import logger


_history_locks: dict[str, object] = {}
_history_locks_guard = threading.Lock()


def _get_history_lock(filepath: str):
    """Serialise one history file's whole read-modify-write transaction.

    Cross-PROCESS, not just cross-thread. A history file is one JSON document
    rewritten in full on every append, so two server processes sharing the
    same ``chat_history/`` directory (the same thing the reminder claim store
    already assumes) would each read the same snapshot and the later
    ``os.replace`` would silently drop the other process' row -- a reminder
    that was delivered by process A disappearing from the transcript because
    process B stored a different message at the same moment.

    The lock is the filepath-keyed advisory ``flock`` already used by the
    character-state store (in-process RLock + ``<file>.lock`` sidecar), so a
    single mechanism covers every shared JSON store. It is re-entrant and the
    kernel releases it when a process dies, so a crash cannot wedge the store.
    Imported lazily: ``character_state`` imports this module at import time.
    """
    with _history_locks_guard:
        cached = _history_locks.get(filepath)
        if cached is None:
            from .character_state import _get_state_lock

            cached = _get_state_lock(filepath)
            _history_locks[filepath] = cached
        return cached


# Rows (metadata excluded) at which one session's transcript is reported as a
# performance risk to the operator. Every append rewrites the WHOLE file with
# fsync while the pre-send guard holds the event loop, so per-frame cost
# follows total rows. Measured on this codebase (tests/bench_loop_blocking.py,
# Stages 4.3/4.4): ~6 ms per content frame at the largest transcript this
# deployment really has (298 rows / 112 KB); ~15 ms at 1000 rows, where other
# loop tasks start slipping past their cadence.
#
# This is a SIGNAL, not a cap. The transcript is append-only by design (the
# rolling summary compacts the model's PROMPT, never the file -- see
# ``BasicMemoryAgent.compact_conversation``), so nothing here trims, moves or
# archives a single row: the operator decides when to start a new
# conversation.
TRANSCRIPT_SIZE_WARN_ROWS = 1000

_transcript_size_warned: set[str] = set()
_transcript_size_warned_guard = threading.Lock()


def _report_transcript_size(filepath: str, history_uid: str, rows: int) -> None:
    """One advisory log line per transcript per process; never raises.

    Called AFTER a successful append, so it describes a file that really is
    that large. Fail-soft: a logging problem must never turn a durable write
    into a reported failure.
    """
    if rows < TRANSCRIPT_SIZE_WARN_ROWS:
        return
    try:
        with _transcript_size_warned_guard:
            if filepath in _transcript_size_warned:
                return
            _transcript_size_warned.add(filepath)
        logger.warning(
            "Transcript size advisory: history_uid={} rows={} threshold={} -- "
            "every content frame rewrites this whole file with fsync, so "
            "per-frame event-loop blocking now grows past the measured "
            "comfortable range (~15 ms at 1000 rows). Nothing was trimmed; "
            "start a new conversation when convenient to restore frame "
            "cadence.",
            history_uid,
            rows,
            TRANSCRIPT_SIZE_WARN_ROWS,
        )
    except Exception:  # pragma: no cover - logging never fails a write
        pass


def _write_history_atomic(filepath: str, history_data: list) -> None:
    temporary_path = f"{filepath}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(history_data, file, ensure_ascii=False, indent=2)
            # os.replace() makes the NEW NAME visible atomically, but does not
            # force the bytes out of the page cache. Stage 4.1 hands a content
            # frame to the transport only after this write returns, so a
            # process/power loss straight afterwards must not expose an empty
            # or truncated row. Flush + fsync before the rename is the durable
            # boundary; a failure raises and store_message reports False.
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, filepath)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


class HistoryMessage(TypedDict):
    role: Literal["human", "ai"]
    timestamp: str
    content: str
    # Optional display information for the message
    name: Optional[str]
    avatar: Optional[str]
    # Stable identity of this LOGICAL message. Unlike the per-attempt
    # ``request_id`` broadcast on the wire, this value never changes when a
    # delivery is retried, so a reload, a resync and a resend all name the
    # same message. Rows written before this field existed simply lack it;
    # readers must treat a missing value as "no stable identity".
    msg_id: Optional[str]


def _is_safe_filename(filename: str) -> bool:
    """Validate filename for safety and allowed characters"""
    if not filename or len(filename) > 255:
        return False

    # Allow alphanumeric, hyphen, underscore, and common unicode characters
    # Block any filesystem special characters, control characters, and path separators
    pattern = re.compile(r"^[\w\-_\u0020-\u007E\u00A0-\uFFFF]+$")
    return bool(pattern.match(filename))


def _sanitize_path_component(component: str) -> str:
    """Sanitize and validate a path component"""
    # Remove any path components, get just the basename
    sanitized = os.path.basename(component.strip())

    if not _is_safe_filename(sanitized):
        raise ValueError(f"Invalid characters in path component: {component}")

    return sanitized


def _ensure_conf_dir(conf_uid: str) -> str:
    """Ensure the directory for a specific conf exists and return its path"""
    if not conf_uid:
        raise ValueError("conf_uid cannot be empty")

    safe_conf_uid = _sanitize_path_component(conf_uid)
    base_dir = os.path.join("chat_history", safe_conf_uid)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _get_safe_history_path(conf_uid: str, history_uid: str) -> str:
    """Get sanitized path for history file"""
    safe_conf_uid = _sanitize_path_component(conf_uid)
    safe_history_uid = _sanitize_path_component(history_uid)
    base_dir = os.path.join("chat_history", safe_conf_uid)
    full_path = os.path.normpath(os.path.join(base_dir, f"{safe_history_uid}.json"))
    if not full_path.startswith(base_dir):
        raise ValueError("Invalid path: Path traversal detected")
    return full_path


def create_new_history(conf_uid: str) -> str:
    """Create a new history file with a unique ID and return the history_uid"""
    if not conf_uid:
        logger.warning("No conf_uid provided")
        return ""

    # Use uuid.uuid4().hex to generate a UUID without hyphens
    # New format: UUID_YYYY-MM-DD_HH-MM-SS
    # Canonical persisted timestamps are tz-aware UTC. Existing naive
    # records keep sorting/parsing fine (same ISO prefix, no parser).
    history_uid = (
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex}"
    )
    conf_dir = _ensure_conf_dir(conf_uid)  # conf_uid is sanitized here

    # Create history file with empty metadata
    try:
        filepath = os.path.join(conf_dir, f"{history_uid}.json")
        initial_data = [
            {
                "role": "metadata",
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "relationship_status": "stranger",
                "relationship_reason": "default",
            }
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to create new history file: {e}")
        return ""

    logger.debug(f"Created new history file with empty metadata: {filepath}")
    return history_uid


def _normalize_msg_id(msg_id: str | None) -> str:
    """Canonical form of a message identity, or "" when unusable."""
    if not isinstance(msg_id, str):
        return ""
    return msg_id.strip()


def store_message(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai"],
    content: str,
    name: str | None = None,
    avatar: str | None = None,
    msg_id: str | None = None,
) -> bool:
    """Store a message in a specific history file

    Args:
        conf_uid: Configuration unique identifier
        history_uid: History unique identifier
        role: Message role ("human" or "ai")
        content: Message content
        name: Optional display name (default None)
        avatar: Optional avatar URL (default None)
        msg_id: Stable identity of the LOGICAL message. A retry of the same
            message MUST pass the same value: the row is then replaced in
            place instead of appended, so the transcript holds one row per
            logical message no matter how many delivery attempts happened.
            When omitted, a fresh id is minted (one logical message per
            call, exactly the previous behaviour).

    Returns:
        True only when the row is durably on disk (appended, or updated in
        place for an existing msg_id). False when NOTHING was written:
        missing identifiers, an unreadable transcript, or a failed atomic
        write (the traceback is logged). This is the one trustworthy success
        signal; a caller that reports the message as stored without checking
        it can claim something the transcript does not contain.
    """
    if not conf_uid or not history_uid:
        if not conf_uid:
            logger.warning("Missing conf_uid")
        if not history_uid:
            logger.warning("Missing history_uid")
        return False

    filepath = _get_safe_history_path(conf_uid, history_uid)
    logger.debug(f"Storing {role} message to {filepath}")

    try:
        # The lock itself can fail (unwritable sidecar lock file); that is a
        # storage failure like any other, not a caller bug.
        lock = _get_history_lock(filepath)
        with lock:
            history_data = []
            unreadable = False
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        history_data = json.load(f)
                except Exception:
                    # Refuse to write. This path used to fall through with an
                    # empty list and then atomically overwrite the whole file
                    # with the single message being appended, so one transient
                    # read failure at save time silently collapsed an entire
                    # conversation. Preserving the existing file is the only
                    # non-destructive option; the snapshot backups are the
                    # recovery path for genuinely corrupt files.
                    unreadable = True

            if unreadable:
                logger.critical(
                    "Refusing to append to an unreadable history file; "
                    "existing data preserved: {}",
                    filepath,
                )
                return False

            now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
            stable_id = _normalize_msg_id(msg_id) or uuid.uuid4().hex

            # Idempotent upsert: a redelivery of the SAME logical message keeps
            # one row. Content is replaced (a retry may have produced different
            # text) while position and timestamp stay those of the first write,
            # so the transcript never reorders itself around a retry.
            existing = next(
                (
                    item
                    for item in history_data
                    if item.get("msg_id") == stable_id
                    and item.get("role") == role
                ),
                None,
            )
            if existing is not None:
                existing["content"] = content
                if name is not None:
                    existing["name"] = name
                if avatar is not None:
                    existing["avatar"] = avatar
                _write_history_atomic(filepath, history_data)
                logger.debug(
                    "Updated {} message in place (msg_id={}, characters={})",
                    role,
                    stable_id,
                    len(content),
                )
                return True

            new_item = {
                "role": role,
                "timestamp": now_str,
                "content": content,
                "msg_id": stable_id,
            }

            # Add optional display information if provided
            if name is not None:
                new_item["name"] = name
            if avatar is not None:
                new_item["avatar"] = avatar

            history_data.append(new_item)
            _write_history_atomic(filepath, history_data)

            # Growth happens only on append (an upsert by msg_id replaces a
            # row in place), so the advisory is reported from here -- after
            # the write succeeded, outside the "nothing was written" paths.
            _report_transcript_size(
                filepath,
                history_uid,
                sum(1 for item in history_data if item.get("role") != "metadata"),
            )
    except Exception:
        # Disk full, permission denied, a vanished directory: the row is NOT
        # on disk. Report it instead of raising, so the caller can suppress
        # every "this was stored" claim while still closing the turn cleanly.
        logger.exception(
            "Failed to store {} message (history_uid={}, msg_id={}): row NOT "
            "persisted",
            role,
            history_uid,
            msg_id,
        )
        return False
    logger.debug(f"Successfully stored {role} message")
    return True


def load_message_snapshot(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai"],
    msg_id: str,
) -> tuple[bool, Optional[HistoryMessage]]:
    """Read one message by identity without guessing from its text.

    Returns ``(readable, row)``. ``readable`` is False when the file exists
    but could not be parsed; callers must not treat that as "row absent" and
    must not overwrite/delete data they could not inspect. A missing file is
    readable and has no row. Rows written before ``msg_id`` existed never
    match an identity (the same rule as ``store_message``).
    """
    stable_id = _normalize_msg_id(msg_id)
    if not conf_uid or not history_uid or not stable_id:
        return False, None

    filepath = _get_safe_history_path(conf_uid, history_uid)
    if not os.path.exists(filepath):
        return True, None

    try:
        lock = _get_history_lock(filepath)
        with lock:
            with open(filepath, "r", encoding="utf-8") as file:
                history_data = json.load(file)
            for item in history_data:
                if (
                    isinstance(item, dict)
                    and item.get("msg_id") == stable_id
                    and item.get("role") == role
                ):
                    return True, item  # type: ignore[return-value]
        return True, None
    except Exception:
        logger.exception(
            "Failed to inspect history message (history_uid={}, msg_id={})",
            history_uid,
            stable_id,
        )
        return False, None


def remove_message(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai"],
    msg_id: str,
) -> bool:
    """Atomically remove one message by identity.

    Used to roll back a provisional AI row that was written before a content
    frame, when that frame was refused by the transport and no text of this
    logical message was accepted. Returns True when the row is absent
    afterwards (including the already-absent case); False means the file
    could not be read or written, so the caller must not claim a rollback.
    Legacy rows without ``msg_id`` can never be selected.
    """
    stable_id = _normalize_msg_id(msg_id)
    if not conf_uid or not history_uid or not stable_id:
        return False

    filepath = _get_safe_history_path(conf_uid, history_uid)
    if not os.path.exists(filepath):
        return True

    try:
        lock = _get_history_lock(filepath)
        with lock:
            with open(filepath, "r", encoding="utf-8") as file:
                history_data = json.load(file)
            kept = [
                item
                for item in history_data
                if not (
                    isinstance(item, dict)
                    and item.get("msg_id") == stable_id
                    and item.get("role") == role
                )
            ]
            if len(kept) == len(history_data):
                return True
            _write_history_atomic(filepath, kept)
        logger.debug(
            "Removed {} message (msg_id={}) after refused delivery",
            role,
            stable_id,
        )
        return True
    except Exception:
        logger.exception(
            "Failed to remove history message (history_uid={}, msg_id={}): "
            "row may remain",
            history_uid,
            stable_id,
        )
        return False


def get_metadata(conf_uid: str, history_uid: str) -> dict:
    """Get metadata from history file"""
    if not conf_uid or not history_uid:
        return {}

    filepath = _get_safe_history_path(conf_uid, history_uid)
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)

        if history_data and history_data[0]["role"] == "metadata":
            return history_data[0]
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
    return {}


def update_metadate(conf_uid: str, history_uid: str, metadata: dict) -> bool:
    """Set metadata in history file

    Updates existing metadata with new fields, preserving existing ones.
    If no metadata exists, creates new metadata entry.
    """
    if not conf_uid or not history_uid:
        return False

    filepath = _get_safe_history_path(conf_uid, history_uid)
    if not os.path.exists(filepath):
        return False

    try:
        lock = _get_history_lock(filepath)
        with lock:
            with open(filepath, "r", encoding="utf-8") as f:
                history_data = json.load(f)

            if history_data and history_data[0]["role"] == "metadata":
                # Update existing metadata while preserving other fields
                history_data[0].update(metadata)
            else:
                # Create new metadata with timestamp if none exists
                new_metadata = {
                    "role": "metadata",
                    "timestamp": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                }
                new_metadata.update(metadata)  # Add new fields
                history_data.insert(0, new_metadata)

            _write_history_atomic(filepath, history_data)

        logger.debug(f"Updated metadata for history {history_uid}")
        return True
    except Exception as e:
        logger.error(f"Failed to set metadata: {e}")
    return False


def update_summary_metadata(
    conf_uid: str,
    history_uid: str,
    *,
    expected_summarized_through: int,
    conversation_summary: str,
    summarized_through: int,
    summary_updated_at: str,
) -> bool:
    """Atomically advance one conversation summary without stale overwrites."""
    if not conf_uid or not history_uid:
        return False

    filepath = _get_safe_history_path(conf_uid, history_uid)
    if not os.path.exists(filepath):
        return False

    lock = _get_history_lock(filepath)
    try:
        with lock:
            with open(filepath, "r", encoding="utf-8") as file:
                history_data = json.load(file)
            metadata = (
                history_data[0]
                if history_data and history_data[0].get("role") == "metadata"
                else None
            )
            current_through = int(
                (metadata or {}).get("summary_through_message_index", 0) or 0
            )
            if current_through != expected_summarized_through:
                logger.warning(
                    "Skipped stale summary update: expected_through={}, current_through={}",
                    expected_summarized_through,
                    current_through,
                )
                return False
            if metadata is None:
                metadata = {
                    "role": "metadata",
                    "timestamp": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                }
                history_data.insert(0, metadata)
            metadata.update(
                {
                    "conversation_summary": conversation_summary,
                    "summary_through_message_index": summarized_through,
                    "summary_updated_at": summary_updated_at,
                }
            )
            _write_history_atomic(filepath, history_data)
        return True
    except Exception as error:
        logger.error("Failed to persist conversation summary: {}", error)
        return False


def get_history(conf_uid: str, history_uid: str) -> List[HistoryMessage]:
    """Read chat history for the given conf_uid and history_uid"""
    if not conf_uid or not history_uid:
        if not conf_uid:
            logger.warning("Missing conf_uid")
        if not history_uid:
            logger.warning("Missing history_uid")
        return []

    filepath = _get_safe_history_path(conf_uid, history_uid)

    if not os.path.exists(filepath):
        logger.warning(f"History file not found: {filepath}")
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)
            # Filter out metadata
            return [msg for msg in history_data if msg["role"] != "metadata"]
    except Exception:
        return []


def delete_history(conf_uid: str, history_uid: str) -> bool:
    """Delete a specific history file"""
    if not conf_uid or not history_uid:
        logger.warning("Missing conf_uid or history_uid")
        return False

    filepath = _get_safe_history_path(conf_uid, history_uid)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"Successfully deleted history file: {filepath}")
            return True
    except Exception as e:
        logger.error(f"Failed to delete history file: {e}")
    return False


# How long a session that holds only its metadata row is assumed to be a
# session about to be USED rather than an abandoned artifact. ``create_new_history``
# writes that skeleton before the client has said anything, so deleting it
# immediately (the old behaviour) could destroy the metadata of a session that
# was already handed to a client: the first ``store_message`` then re-created
# the file WITHOUT a metadata row, silently losing the sidebar title and the
# relationship defaults for that conversation. A create -> first-message gap is
# seconds, and another server process' brand-new session is invisible to this
# process' live clients, so the age window covers both.
EMPTY_SESSION_GRACE_SECONDS = 300.0

# How long an EMPTY but VALID continuation session stays resumable. A hand-off
# creates its target and only then does the client write the first message into
# it; if the browser closes, the laptop sleeps or the server restarts inside
# that gap, the session is still the right one to continue -- but the ordinary
# grace window would delete it a few minutes later and the client would fall
# back to the SOURCE transcript without telling anyone (measured: Stage 4.10).
#
# So a continuation is protected from cleanup for a day, which covers a restart
# and coming back to the chat later, while a target left behind by a crashed
# hand-off is still eventually removed. It is never offered as a conversation:
# it has no rows of its own, so :func:`resumable_continuations` lists it apart
# and only a client that already remembers that exact uid may adopt it.
EMPTY_CONTINUATION_TTL_SECONDS = 24 * 60 * 60


def _continuation_source(conf_uid: str, metadata: dict) -> str:
    """The source uid of a VALID continuation, or ''.

    Valid means the ``continued_from`` label points at a sibling transcript that
    still exists and holds at least one human/AI row. That is what proves a
    hand-off really happened and that there is a conversation to carry on; a
    metadata-only file with a dangling, blank or self-referencing label is an
    abandoned artifact and must keep behaving exactly like any other empty
    session. Read-only.
    """
    if not isinstance(metadata, dict):
        return ""
    source = str(metadata.get("continued_from", "") or "").strip()
    if not source:
        return ""
    try:
        source_path = _get_safe_history_path(conf_uid, source)
    except Exception:
        return ""
    if not os.path.exists(source_path):
        return ""
    try:
        with open(source_path, "r", encoding="utf-8") as source_file:
            source_rows = json.load(source_file)
    except Exception:
        return ""
    if not isinstance(source_rows, list):
        return ""
    if not any(
        isinstance(source_row, dict)
        and source_row.get("role") in {"human", "ai"}
        for source_row in source_rows
    ):
        return ""
    return source


def _empty_session_is_disposable(
    filepath: str,
    history_uid: str,
    protected: Iterable[str],
    now: Optional[float] = None,
    *,
    continuation_source: str = "",
) -> bool:
    """May this metadata-only session file be removed?

    Only when it is provably unused: no live client has it as its active
    history (``protected``) and it is older than ``EMPTY_SESSION_GRACE_SECONDS``.
    Anything younger, or active anywhere, is kept -- it may be a session a
    client is already displaying and about to write its first message into.

    A session with a VALID continuation (``continuation_source``) gets the
    longer ``EMPTY_CONTINUATION_TTL_SECONDS``: it is not an orphan, it is a
    hand-off the client has not reached yet, and deleting it early is exactly
    what made a restart resume the wrong transcript.
    """
    if str(history_uid) in {str(uid) for uid in protected}:
        return False
    try:
        modified_at = os.path.getmtime(filepath)
    except OSError:
        # Vanished between listdir and here: nothing left to delete.
        return False
    clock = time.time() if now is None else float(now)
    age = clock - modified_at
    if continuation_source:
        return age >= EMPTY_CONTINUATION_TTL_SECONDS
    return age >= EMPTY_SESSION_GRACE_SECONDS


def get_history_list(
    conf_uid: str,
    cleanup: bool = True,
    *,
    protect_uids: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Get list of histories with their latest messages.

    ``cleanup=False`` makes the scan strictly read-only (no empty-history
    deletion) for callers like session-recency that must never mutate the
    store as a side effect.

    ``protect_uids`` names the histories live clients are currently on, so a
    session that exists but has no messages yet (a brand-new conversation, or
    one a client just switched to) is never deleted out from under them.
    """
    if not conf_uid:
        return []

    histories = []
    conf_dir = _ensure_conf_dir(conf_uid)
    empty_history_files = []

    try:
        for filename in os.listdir(conf_dir):
            if not filename.endswith(".json"):
                continue

            history_uid = filename[:-5]
            filepath = os.path.join(conf_dir, filename)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    messages = json.load(f)

                    # Filter out metadata for checking if history is empty
                    actual_messages = [
                        msg for msg in messages if msg["role"] != "metadata"
                    ]
                    metadata = (
                        messages[0]
                        if messages and messages[0].get("role") == "metadata"
                        else {}
                    )
                    continued_from = str(
                        metadata.get("continued_from", "") or ""
                    ).strip()
                    if not actual_messages:
                        # An empty session is still not offered for resume: a
                        # half-finished hand-off (crash between creating the
                        # target and adopting it) must never appear as a
                        # conversation, and the ordinary grace-window cleanup
                        # keeps meaning exactly what it did before. The client
                        # falls back to the newest REAL session instead.
                        #
                        # A valid continuation is the exception the client
                        # needs, but it is NOT added to ``histories``: it is
                        # reported separately by :func:`resumable_continuations`
                        # so only a tab that remembers this exact uid adopts it.
                        empty_history_files.append(
                            (
                                history_uid,
                                filepath,
                                _continuation_source(conf_uid, metadata),
                            )
                        )
                        continue

                    latest_message = actual_messages[-1]
                    history_info = {
                        "uid": history_uid,
                        "latest_message": latest_message,
                        "timestamp": (
                            latest_message["timestamp"] if latest_message else None
                        ),
                        "title": (
                            str(metadata.get("title", "")).strip() if metadata else ""
                        ),
                    }
                    if continued_from:
                        history_info["continued_from"] = continued_from
                    histories.append(history_info)
            except Exception as e:
                logger.error(f"Error reading history file {filename}: {e}")
                continue

        # Clean up empty histories if there are other non-empty ones. A
        # metadata-only session that is still young, or that a live client is
        # on, is NOT an orphan: deleting it would take its metadata with it
        # (see EMPTY_SESSION_GRACE_SECONDS).
        if cleanup and empty_history_files and len(os.listdir(conf_dir)) > 1:
            protected = tuple(protect_uids or ())
            for uid, filepath, continuation in empty_history_files:
                if not _empty_session_is_disposable(
                    filepath,
                    uid,
                    protected,
                    continuation_source=continuation,
                ):
                    logger.debug(
                        "Kept empty history file: uid={} (active or just created)",
                        uid,
                    )
                    continue
                try:
                    os.remove(filepath)
                    logger.info(f"Removed empty history file: {uid}")
                except FileNotFoundError:
                    continue
                except Exception as e:
                    logger.error(f"Failed to remove empty history file {uid}: {e}")

        histories.sort(
            key=lambda x: x["timestamp"] if x["timestamp"] else "", reverse=True
        )
        return histories

    except Exception as e:
        logger.error(f"Error listing histories: {e}")
        return []


def resumable_continuations(conf_uid: str) -> List[dict]:
    """Empty but VALID continuation sessions a client may still resume.

    Kept separate from :func:`get_history_list` on purpose. Such a session has
    no rows of its own, so listing it with the conversations would render it as
    an empty chat and offer it to every client; the sidebar stays a list of
    real conversations. What a reconnect needs is only the authority to adopt
    the uid it already remembered -- and only a uid returned here carries a
    verified lineage back to a transcript that still exists.

    Read-only: no cleanup, no metadata write, no source access beyond reading.
    """
    if not conf_uid:
        return []
    try:
        conf_dir = os.path.join("chat_history", _sanitize_path_component(conf_uid))
        filenames = os.listdir(conf_dir)
    except OSError:
        return []

    resumable: List[dict] = []
    for filename in filenames:
        if not filename.endswith(".json"):
            continue
        history_uid = filename[:-5]
        filepath = os.path.join(conf_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as history_file:
                rows = json.load(history_file)
        except Exception as error:
            logger.debug(
                "Resumable continuation skipped: uid={} error={}",
                history_uid,
                type(error).__name__,
            )
            continue
        if not isinstance(rows, list) or not rows:
            continue
        if any(
            isinstance(row, dict) and row.get("role") != "metadata" for row in rows
        ):
            # Holds messages: it is an ordinary conversation, listed elsewhere.
            continue
        metadata = rows[0] if isinstance(rows[0], dict) else {}
        source = _continuation_source(conf_uid, metadata)
        if not source:
            continue
        resumable.append(
            {
                "uid": history_uid,
                "continued_from": source,
                "continued_at": str(metadata.get("continued_at", "") or ""),
                "timestamp": str(metadata.get("timestamp", "") or "") or None,
            }
        )

    resumable.sort(key=lambda item: item["timestamp"] or "", reverse=True)
    return resumable


def modify_latest_message(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai", "system"],
    new_content: str,
) -> bool:
    """Modify the latest message in a specific history file if it matches the given role"""
    if not conf_uid or not history_uid:
        logger.warning("Missing conf_uid or history_uid")
        return False

    filepath = _get_safe_history_path(conf_uid, history_uid)
    if not os.path.exists(filepath):
        logger.warning(f"History file not found: {filepath}")
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)

        if not history_data:
            logger.warning("History is empty")
            return False

        latest_message = history_data[-1]
        if latest_message["role"] != role:
            logger.warning(
                f"Latest message role ({latest_message['role']}) doesn't match requested role ({role})"
            )
            return False

        latest_message["content"] = new_content
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Successfully modified latest {role} message")
        return True

    except Exception as e:
        logger.error(f"Failed to modify latest message: {e}")
        return False


def rename_history_file(
    conf_uid: str, old_history_uid: str, new_history_uid: str
) -> bool:
    """Rename a history file with a new history_uid"""
    if not conf_uid or not old_history_uid or not new_history_uid:
        logger.warning("Missing required parameters for rename")
        return False

    old_filepath = _get_safe_history_path(conf_uid, old_history_uid)
    new_filepath = _get_safe_history_path(conf_uid, new_history_uid)

    try:
        if os.path.exists(old_filepath):
            os.rename(old_filepath, new_filepath)
            logger.info(
                f"Renamed history file from {old_history_uid} to {new_history_uid}"
            )
            return True
    except Exception as e:
        logger.error(f"Failed to rename history file: {e}")
    return False
