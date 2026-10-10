"""Attachment memory: what Mili saw in past attachments, without the files.

Layers (kept distinct by design):
- transcript: full conversation record (chat_history/*.json, text only).
- conversation summary: compressed conversational context (metadata).
- episodic memory: selected past experiences (episodic/<conf>.json).
- attachment memory (this module): per-attachment metadata + a semantic
  summary of what was actually visible, keyed by character
  (attachment_memory/<conf>.json).
- long-term memory: stable facts/preferences (character_state.memories).

An attachment record stores metadata (filename, MIME type, source, receive
time, session/message refs, content hash, processing status) and, only when
the model genuinely described visible content, a short factual summary.
Failed attachments are recorded with status="failed" and an EMPTY summary so
the system can honestly say an attachment existed but was never read — never
a fabricated description. Original image bytes are never persisted here.

All persistence here is fail-soft: storage/parse failures return safe
defaults and never raise into the conversation pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .chat_history_manager import _sanitize_path_component
from .world_state import memory_age_label, utcnow


ATTACHMENT_DIR = "attachment_memory"
ATTACHMENT_MAX_RECORDS = 200
ATTACHMENT_TOP_N = 2
ATTACHMENT_MAX_TOKENS = 300
ATTACHMENT_SUMMARY_MAX_CHARS = 600

ATTACHMENT_STATUS_PROCESSED = "processed"
ATTACHMENT_STATUS_FAILED = "failed"
_ATTACHMENT_STATUSES = frozenset(
    {ATTACHMENT_STATUS_PROCESSED, ATTACHMENT_STATUS_FAILED}
)
_ATTACHMENT_SOURCES = frozenset({"camera", "screen", "clipboard", "upload"})

ATTACHMENT_SUMMARY_SYSTEM = """You describe attached images factually so they can be remembered later.
Return JSON only, exactly this shape:
{"items": [{"index": 0, "visible": true, "summary": "<one or two short factual sentences in the user's language>"}, ...]}
Rules: one entry per attached image, in order, with the matching index.
Describe only what is visibly present in each image. Never invent people,
text, objects, or details that are not clearly visible. If an image is
blank, unreadable, corrupted, or you cannot see it, use
{"index": N, "visible": false, "summary": null} for that entry.
Never describe personality, preferences, or permanent traits of anyone."""

_STOPWORDS = frozenset(
    "yang dan di ke dari untuk dengan pada adalah itu ini aku gw gue saya "
    "kamu kau anda dia mereka kita kami ko kok sih dong deh aja lah mah pun "
    "nggak tidak tak bukan sudah telah lagi masih juga atau tapi karena kalau "
    "kalo yang the a an and or of to in on for with is are was were it this "
    "that i you he she we they my your his her our their me him us them what "
    "yang gw saya foto gambar image picture photo".split()
)

_attachment_locks: Dict[str, threading.RLock] = {}
_attachment_locks_guard = threading.Lock()


def _get_attachment_lock(filepath: str) -> threading.RLock:
    with _attachment_locks_guard:
        return _attachment_locks.setdefault(filepath, threading.RLock())


def get_attachment_path(conf_uid: str) -> str:
    """Return the on-disk path for a character's attachment store."""
    if not conf_uid:
        raise ValueError("conf_uid cannot be empty")
    safe_conf_uid = _sanitize_path_component(conf_uid)
    return os.path.join(ATTACHMENT_DIR, f"{safe_conf_uid}.json")


def _write_attachment_atomic(filepath: str, records: list) -> None:
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary_path = f"{filepath}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, filepath)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def content_hash_for(data: Any) -> str:
    """Stable sha256 hex over the attachment transport string (pure)."""
    if not isinstance(data, str) or not data:
        return ""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _valid_record(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    status = str(item.get("status", "") or "")
    if status not in _ATTACHMENT_STATUSES:
        return None
    content_hash = str(item.get("content_hash", "") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
        return None
    summary = " ".join(str(item.get("summary", "") or "").split()).strip()
    if status == ATTACHMENT_STATUS_PROCESSED and not summary:
        # A processed record without a summary is a fabricated-memory risk;
        # never trust it from disk.
        return None
    if status == ATTACHMENT_STATUS_FAILED:
        summary = ""
    source = str(item.get("source", "") or "")
    if source not in _ATTACHMENT_SOURCES:
        source = "upload"
    mime_type = str(item.get("mime_type", "") or "")
    filename = item.get("filename")
    filename = str(filename).strip() if filename is not None else None
    if filename == "":
        filename = None
    return {
        "id": str(item.get("id", "") or uuid.uuid4().hex),
        "filename": filename,
        "mime_type": mime_type,
        "source": source,
        "content_hash": content_hash,
        "received_at": str(item.get("received_at", "") or ""),
        "session_uid": str(item.get("session_uid", "") or ""),
        "request_id": str(item.get("request_id", "") or ""),
        "status": status,
        "summary": summary[:ATTACHMENT_SUMMARY_MAX_CHARS],
        "summary_model": str(item.get("summary_model", "") or ""),
        "created_at": str(item.get("created_at", "") or ""),
        "tz": item.get("tz"),
    }


def load_attachment_memories(conf_uid: str) -> List[Dict[str, Any]]:
    """Load attachment records; missing/corrupt files yield an empty list."""
    try:
        filepath = get_attachment_path(conf_uid)
    except ValueError:
        return []
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as error:
        logger.error(
            "Failed to load attachment memories: error_type={}",
            type(error).__name__,
        )
        return []
    if not isinstance(data, list):
        return []
    records = []
    for item in data:
        parsed = _valid_record(item)
        if parsed is not None:
            records.append(parsed)
    return records


def save_attachment_memories(conf_uid: str, records: List[Dict[str, Any]]) -> bool:
    """Atomically persist attachment records; returns success."""
    try:
        filepath = get_attachment_path(conf_uid)
    except ValueError:
        return False
    try:
        lock = _get_attachment_lock(filepath)
        with lock:
            _write_attachment_atomic(filepath, list(records))
        return True
    except Exception as error:
        logger.error(
            "Failed to save attachment memories: error_type={}",
            type(error).__name__,
        )
        return False


def append_attachment_memory(
    conf_uid: str, record: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Validate, dedupe by content hash, append with created_at, cap size.

    A processed record whose bytes were already remembered returns None
    (duplicate). A new record whose bytes match an older FAILED record
    replaces it, so a retry that finally reads the image upgrades the
    honest "never read" entry instead of duplicating it.
    """
    if not isinstance(record, dict):
        return None
    status = str(record.get("status", "") or "")
    if status not in _ATTACHMENT_STATUSES:
        return None
    content_hash = str(record.get("content_hash", "") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
        return None
    summary = " ".join(str(record.get("summary", "") or "").split()).strip()
    if status == ATTACHMENT_STATUS_PROCESSED and not summary:
        return None
    if status == ATTACHMENT_STATUS_FAILED:
        summary = ""
    source = str(record.get("source", "") or "")
    if source not in _ATTACHMENT_SOURCES:
        return None
    mime_type = str(record.get("mime_type", "") or "")
    if not mime_type.startswith("image/"):
        return None
    records = load_attachment_memories(conf_uid)
    for existing in records:
        if existing.get("content_hash") != content_hash:
            continue
        if existing.get("status") == ATTACHMENT_STATUS_PROCESSED:
            return None
        # Upgrade path: drop the older failed entry for the same bytes.
        records = [item for item in records if item.get("content_hash") != content_hash]
        break
    now = utcnow().isoformat(timespec="seconds")
    filename = record.get("filename")
    filename = str(filename).strip() if filename is not None else None
    if filename == "":
        filename = None
    stored = {
        "id": str(record.get("id", "") or uuid.uuid4().hex),
        "filename": filename,
        "mime_type": mime_type,
        "source": source,
        "content_hash": content_hash,
        "received_at": str(record.get("received_at", "") or now),
        "session_uid": str(record.get("session_uid", "") or ""),
        "request_id": str(record.get("request_id", "") or ""),
        "status": status,
        "summary": summary[:ATTACHMENT_SUMMARY_MAX_CHARS],
        "summary_model": str(record.get("summary_model", "") or ""),
        "created_at": now,
        "tz": record.get("tz"),
    }
    records.append(stored)
    if len(records) > ATTACHMENT_MAX_RECORDS:
        records.sort(key=lambda item: str(item.get("created_at", "")))
        records = records[-ATTACHMENT_MAX_RECORDS:]
    if not save_attachment_memories(conf_uid, records):
        return None
    return stored


def delete_attachment_memory(conf_uid: str, record_id: str) -> bool:
    """Delete one attachment record by id; True when a record was removed."""
    if not record_id:
        return False
    records = load_attachment_memories(conf_uid)
    kept = [item for item in records if str(item.get("id", "")) != str(record_id)]
    if len(kept) == len(records):
        return False
    return save_attachment_memories(conf_uid, kept)


def clear_attachment_memories(conf_uid: str) -> int:
    """Delete all attachment records; returns the removed count."""
    records = load_attachment_memories(conf_uid)
    if not records:
        return 0
    if not save_attachment_memories(conf_uid, []):
        return 0
    return len(records)


def delete_attachment_memories_for_session(conf_uid: str, session_uid: str) -> int:
    """Delete records referencing one conversation; returns removed count.

    Used as a cascade when a chat history is deleted: an attachment memory
    points at its source conversation, so it must not outlive it.
    """
    if not session_uid:
        return 0
    records = load_attachment_memories(conf_uid)
    kept = [
        item for item in records if str(item.get("session_uid", "")) != str(session_uid)
    ]
    removed = len(records) - len(kept)
    if removed == 0:
        return 0
    if not save_attachment_memories(conf_uid, kept):
        return 0
    return removed


def _normalize_attachment_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split()).strip()


def _attachment_token_set(text: str) -> set:
    return {token for token in text.split() if token and token not in _STOPWORDS}


def _record_search_text(record: Dict[str, Any]) -> str:
    parts = [
        str(record.get("filename", "") or ""),
        str(record.get("summary", "") or ""),
    ]
    return _normalize_attachment_text(" ".join(parts))


def _score_record(query_tokens: set, record: Dict[str, Any], now: datetime) -> float:
    record_tokens = _attachment_token_set(_record_search_text(record))
    if not query_tokens or not record_tokens:
        return 0.0
    overlap = len(query_tokens & record_tokens)
    if overlap == 0:
        return 0.0
    score = float(overlap)

    def _parse_iso_or_none(value: Any) -> Optional[datetime]:
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    reference = _parse_iso_or_none(record.get("received_at", "")) or _parse_iso_or_none(
        record.get("created_at", "")
    )
    if reference is not None:
        try:
            age_days = (now - reference).total_seconds() / 86400.0
        except TypeError:
            age_days = None
        if age_days is not None and age_days >= 0:
            score += max(0.0, 0.5 - age_days / 14.0)
    return score


def retrieve_attachment_memories(
    records: List[Dict[str, Any]],
    query: Any,
    *,
    now: Optional[datetime] = None,
    top_n: int = ATTACHMENT_TOP_N,
) -> List[Dict[str, Any]]:
    """Selective keyword + recency retrieval over PROCESSED records only.

    Failed records (never read) are never returned: retrieval must not hand
    the model a memory it cannot honestly use. Pure, no LLM, no I/O.
    """
    cleaned = " ".join(str(query or "").split()).strip()
    if not cleaned or not records:
        return []
    try:
        moment = now if now is not None else utcnow()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        query_tokens = _attachment_token_set(_normalize_attachment_text(cleaned))
        scored = []
        for record in records:
            try:
                if record.get("status") != ATTACHMENT_STATUS_PROCESSED:
                    continue
                if not str(record.get("summary", "") or "").strip():
                    continue
                score = _score_record(query_tokens, record, moment)
            except Exception:
                continue
            if score > 0:
                scored.append(
                    (
                        score,
                        str(record.get("received_at", "") or ""),
                        str(record.get("id", "") or ""),
                        record,
                    )
                )
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [row[3] for row in scored[: max(top_n, 0)]]
    except Exception as error:
        logger.warning(
            "Attachment retrieval failed (no records returned): type={}",
            type(error).__name__,
        )
        return []


def _attachment_label(record: Dict[str, Any]) -> str:
    filename = str(record.get("filename", "") or "").strip()
    if filename:
        return filename
    source = str(record.get("source", "") or "upload")
    return f"{source} capture"


def render_attachment_context(
    records: List[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    tz: Optional[str] = None,
    max_tokens: int = ATTACHMENT_MAX_TOKENS,
) -> str:
    """Render retrieved records with render-time age tags (pure).

    The header states that originals may be gone, so the model answers only
    from the stored summary and never pretends to open the file again.
    """
    if not records:
        return ""
    from .agent.context_window import estimate_tokens

    lines: List[str] = []
    used_tokens = 0
    for record in records:
        summary = " ".join(str(record.get("summary", "") or "").split()).strip()
        if not summary:
            continue
        stamp = record.get("received_at") or record.get("created_at", "")
        age = memory_age_label(stamp, now, tz)
        line = (
            f"- {_attachment_label(record)} ({age}): {summary}"
            if age
            else (f"- {_attachment_label(record)}: {summary}")
        )
        line_tokens = estimate_tokens(line) + 4
        if used_tokens + line_tokens > max_tokens:
            break
        lines.append(line)
        used_tokens += line_tokens
    if not lines:
        return ""
    header = (
        "Relevant attachment memories (files shared earlier; the originals may "
        "no longer be available, answer only from these stored notes):"
    )
    return f"{header}\n" + "\n".join(lines)


def build_attachment_summary_prompt(
    items: List[Dict[str, Any]],
    user_text: str,
    request_dt: datetime,
    tz: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build (system, messages) for one batched describe call (pure).

    ``items`` are ``{"name", "mime_type", "source", "data"}`` dicts in turn
    order; the returned message carries every image as an ``image_url`` block
    so a single model call can describe each attachment by index.
    """
    moment = request_dt
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    names = ", ".join(
        f"[{index}] {str(item.get('name') or item.get('source', 'image'))}"
        for index, item in enumerate(items)
    )
    context = (
        f"Request time (UTC): {moment.isoformat(timespec='seconds')}. "
        f"User timezone: {tz or 'unknown (assume UTC)'}. "
        f"Attached images in order: {names}. "
        f"User message: {(user_text or '').strip() or '(no text)'}"
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": context}]
    for item in items:
        data = item.get("data")
        if isinstance(data, str) and data.startswith("data:image/"):
            content.append({"type": "image_url", "image_url": {"url": data}})
    return ATTACHMENT_SUMMARY_SYSTEM, [{"role": "user", "content": content}]


def parse_attachment_summary(
    raw: Any, count: int
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Parse the batched describe response defensively (pure).

    Returns ``(items, None)`` with one ``{"index", "visible", "summary"}``
    per expected image, or ``([], reason)`` where reason is one of
    ``empty_response``, ``invalid_json``, ``invalid_schema``. Individual
    entries that fail validation degrade to invisible rather than failing
    the whole batch: a malformed entry means "not read", never a guess.
    """
    if not isinstance(raw, str) or not raw.strip():
        return [], "empty_response"
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if not match:
        return [], "invalid_json"
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return [], "invalid_json"
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return [], "invalid_schema"
    parsed: List[Dict[str, Any]] = []
    for position in range(max(count, 0)):
        entry = data["items"][position] if position < len(data["items"]) else None
        if not isinstance(entry, dict):
            parsed.append({"index": position, "visible": False, "summary": None})
            continue
        try:
            index = int(entry.get("index", position))
        except (TypeError, ValueError):
            index = position
        visible = entry.get("visible") is True
        summary = " ".join(str(entry.get("summary", "") or "").split()).strip()
        if visible and summary:
            parsed.append({"index": index, "visible": True, "summary": summary})
        else:
            parsed.append({"index": index, "visible": False, "summary": None})
    return parsed, None


def _log_attachment_rejection(
    reason: str,
    *,
    session_uid: Optional[str] = None,
    error_type: Optional[str] = None,
) -> None:
    """Warn-level trace for a capture that produced no readable summary.

    Metadata only: never logs user text, image data, prompts, or secrets.
    """
    logger.warning(
        "Attachment capture rejected: reason={} session={} error_type={}",
        reason,
        session_uid or "none",
        error_type or "none",
    )


async def describe_and_store_attachments(
    llm_chat_fn: Any,
    conf_uid: str,
    items: List[Dict[str, Any]],
    user_text: str,
    history_uid: str,
    request_id: str,
    request_dt: datetime,
    tz: Optional[str] = None,
    summary_model: str = "",
) -> List[Dict[str, Any]]:
    """Describe one turn's attachments via a single LLM call and store.

    Every attachment gets a record: ``processed`` with the model's factual
    summary when its entry validated, otherwise ``failed`` with an EMPTY
    summary. Returns the stored records (possibly empty). Never raises.
    """
    try:
        if not conf_uid or not items:
            return []
        valid_items: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            data = item.get("data")
            if not isinstance(data, str) or not data.startswith("data:image/"):
                continue
            valid_items.append(item)
        if not valid_items:
            return []
        system, messages = build_attachment_summary_prompt(
            valid_items, user_text, request_dt, tz
        )
        chunks: List[str] = []
        try:
            stream = llm_chat_fn(messages, system)
            async for event in stream:
                if isinstance(event, str):
                    chunks.append(event)
                elif isinstance(event, dict) and event.get("type") == "text_delta":
                    chunks.append(str(event.get("text", "")))
        except Exception as error:
            _log_attachment_rejection(
                "llm_error", session_uid=history_uid, error_type=type(error).__name__
            )
            return _store_failed_records(
                conf_uid,
                valid_items,
                user_text,
                history_uid,
                request_id,
                request_dt,
                tz,
            )
        parsed, reason = parse_attachment_summary("".join(chunks), len(valid_items))
        if not parsed:
            _log_attachment_rejection(reason or "invalid_json", session_uid=history_uid)
            return _store_failed_records(
                conf_uid,
                valid_items,
                user_text,
                history_uid,
                request_id,
                request_dt,
                tz,
            )
        stored: List[Dict[str, Any]] = []
        for position, item in enumerate(valid_items):
            entry = parsed[position] if position < len(parsed) else None
            visible = bool(entry and entry.get("visible"))
            summary = str((entry or {}).get("summary", "") or "").strip()
            record = _new_record(
                item,
                user_text,
                history_uid,
                request_id,
                request_dt,
                tz,
                summary_model,
                status=(
                    ATTACHMENT_STATUS_PROCESSED
                    if (visible and summary)
                    else (ATTACHMENT_STATUS_FAILED)
                ),
                summary=summary if (visible and summary) else "",
            )
            saved = append_attachment_memory(conf_uid, record)
            if saved is not None:
                stored.append(saved)
        return stored
    except Exception as error:
        _log_attachment_rejection(
            "unexpected_error", session_uid=history_uid, error_type=type(error).__name__
        )
        return []


def _store_failed_records(
    conf_uid: str,
    items: List[Dict[str, Any]],
    user_text: str,
    history_uid: str,
    request_id: str,
    request_dt: datetime,
    tz: Optional[str],
) -> List[Dict[str, Any]]:
    """Record attachments the model never read (empty summaries)."""
    del user_text
    stored: List[Dict[str, Any]] = []
    for item in items:
        record = _new_record(
            item,
            "",
            history_uid,
            request_id,
            request_dt,
            tz,
            "",
            status=ATTACHMENT_STATUS_FAILED,
            summary="",
        )
        saved = append_attachment_memory(conf_uid, record)
        if saved is not None:
            stored.append(saved)
    return stored


def _new_record(
    item: Dict[str, Any],
    user_text: str,
    history_uid: str,
    request_id: str,
    request_dt: datetime,
    tz: Optional[str],
    summary_model: str,
    *,
    status: str,
    summary: str,
) -> Dict[str, Any]:
    del user_text
    moment = request_dt
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    name = item.get("name")
    name = str(name).strip() if name is not None else None
    source = str(item.get("source", "") or "")
    if source not in _ATTACHMENT_SOURCES:
        source = "upload"
    return {
        "filename": name,
        "mime_type": str(item.get("mime_type", "") or ""),
        "source": source,
        "content_hash": content_hash_for(item.get("data")),
        "received_at": moment.isoformat(timespec="seconds"),
        "session_uid": str(history_uid or ""),
        "request_id": str(request_id or ""),
        "status": status,
        "summary": summary,
        "summary_model": str(summary_model or ""),
        "tz": tz,
    }


def find_attachment_memory_copies(
    conf_uid: str,
    record_id: Optional[str] = None,
    filename: Optional[str] = None,
    summary_fragment: Optional[str] = None,
) -> Dict[str, Any]:
    """Locate every stored copy of one attachment's information.

    Checks the attachment store itself, episodic events, transcript files,
    rolling-summary metadata, long-term character facts, and the world-state
    file. Used to scope deletion honestly: a copy found here can still reach
    the model context after the attachment record itself is deleted.
    """
    from . import episodic_memory as _episodic
    from . import chat_history_manager as _history

    result: Dict[str, Any] = {
        "attachment_memory": None,
        "episodic": [],
        "transcripts": [],
        "summaries": [],
        "character_memories": [],
        "world_state_match": False,
    }
    needle_name = (filename or "").strip().lower()
    needle_summary = " ".join(str(summary_fragment or "").split()).strip().lower()
    if record_id:
        for record in load_attachment_memories(conf_uid):
            if str(record.get("id", "")) == str(record_id):
                result["attachment_memory"] = record
                if not needle_name:
                    needle_name = str(record.get("filename", "") or "").strip().lower()
                if not needle_summary:
                    needle_summary = (
                        " ".join(str(record.get("summary", "") or "").split())
                        .strip()
                        .lower()[:80]
                    )
                break
    if not needle_name and not needle_summary:
        return result
    for event in _episodic.load_episodic_events(conf_uid) or []:
        text = str(event.get("event_text", "") or "").lower()
        if (needle_name and needle_name in text) or (
            needle_summary and len(needle_summary) >= 20 and needle_summary in text
        ):
            result["episodic"].append(str(event.get("id", "")))
    try:
        histories = _history.get_history_list(conf_uid, cleanup=False) or []
    except Exception:
        histories = []
    for entry in histories:
        uid = entry.get("uid", "") if isinstance(entry, dict) else ""
        if not uid:
            continue
        try:
            messages = _history.get_history(conf_uid, uid) or []
        except Exception:
            continue
        for message in messages:
            content = str(message.get("content", "") or "").lower()
            if (needle_name and needle_name in content) or (
                needle_summary
                and len(needle_summary) >= 20
                and needle_summary in content
            ):
                if uid not in result["transcripts"]:
                    result["transcripts"].append(uid)
                break
        try:
            metadata = _history.get_metadata(conf_uid, uid) or {}
        except Exception:
            metadata = {}
        summary_text = str(metadata.get("conversation_summary", "") or "").lower()
        if (needle_name and needle_name in summary_text) or (
            needle_summary
            and len(needle_summary) >= 20
            and needle_summary in summary_text
        ):
            if uid not in result["summaries"]:
                result["summaries"].append(uid)
    try:
        from . import character_state as _characters

        state = _characters.load_character_state(conf_uid)
        for fact in getattr(state, "memories", None) or []:
            text = (
                str(fact.get("text", "") or "").lower()
                if isinstance(fact, dict)
                else ""
            )
            if (needle_name and needle_name in text) or (
                needle_summary and len(needle_summary) >= 20 and needle_summary in text
            ):
                result["character_memories"].append(
                    str(fact.get("text", "") or "") if isinstance(fact, dict) else ""
                )
    except Exception as error:
        logger.debug(
            "Attachment copy audit skipped character state: type={}",
            type(error).__name__,
        )
    try:
        from . import world_state as _world

        world_path = _world.get_world_state_path(conf_uid)
        with open(world_path, "r", encoding="utf-8") as handle:
            world_text = handle.read().lower()
        if (needle_name and needle_name in world_text) or (
            needle_summary
            and len(needle_summary) >= 20
            and needle_summary in world_text
        ):
            result["world_state_match"] = True
    except Exception as error:
        logger.debug(
            "Attachment copy audit skipped world state: type={}",
            type(error).__name__,
        )
    return result


def purge_attachment_memory(conf_uid: str, record_id: str) -> Dict[str, Any]:
    """Coordinated deletion for one attachment record (fail-safe).

    Removes the attachment record itself plus the episodic events that
    quote it (exact ids only — never the whole episodic store, never
    transcripts). Transcript rows, rolling summaries, character facts, and
    world state have no safe surgical delete, so matching copies there are
    REPORTED, not removed. Every source is attempted independently: a
    failure in one never blocks the others, and the result names exactly
    which sources still hold data. ``complete`` is True only when no known
    copy remains anywhere.
    """
    from . import episodic_memory as _episodic

    outcome: Dict[str, Any] = {
        "found": False,
        "record_id": str(record_id or ""),
        "filename": "",
        "attachment_removed": False,
        "episodic_removed": [],
        "episodic_failed": [],
        "character_memories_remaining": [],
        "transcripts_remaining": [],
        "summaries_remaining": [],
        "world_state_match": False,
        "complete": False,
    }
    if not conf_uid or not record_id:
        return outcome
    copies = find_attachment_memory_copies(conf_uid, record_id=record_id)
    record = copies.get("attachment_memory")
    if not isinstance(record, dict):
        return outcome
    outcome["found"] = True
    outcome["filename"] = str(record.get("filename", "") or "")
    matched_episodic = [str(item) for item in copies.get("episodic", [])]
    if matched_episodic:
        try:
            deleted = _episodic.delete_episodic_events_by_ids(
                conf_uid, matched_episodic
            )
        except Exception as error:
            logger.warning(
                "Attachment purge episodic delete failed: type={}",
                type(error).__name__,
            )
            deleted = {"removed": [], "missing": [], "saved": False}
        outcome["episodic_removed"] = [str(item) for item in deleted.get("removed", [])]
        if not deleted.get("saved", False):
            outcome["episodic_failed"] = list(matched_episodic)
    try:
        outcome["attachment_removed"] = delete_attachment_memory(conf_uid, record_id)
    except Exception as error:
        logger.warning(
            "Attachment purge record delete failed: type={}",
            type(error).__name__,
        )
        outcome["attachment_removed"] = False
    outcome["character_memories_remaining"] = list(copies.get("character_memories", []))
    outcome["transcripts_remaining"] = list(copies.get("transcripts", []))
    outcome["summaries_remaining"] = list(copies.get("summaries", []))
    outcome["world_state_match"] = bool(copies.get("world_state_match", False))
    outcome["complete"] = bool(
        outcome["attachment_removed"]
        and not outcome["episodic_failed"]
        and not outcome["character_memories_remaining"]
        and not outcome["transcripts_remaining"]
        and not outcome["summaries_remaining"]
        and not outcome["world_state_match"]
    )
    logger.info(
        "Attachment purge: found={} removed={} episodic_removed={} "
        "episodic_failed={} remaining_transcripts={} remaining_summaries={} "
        "remaining_character={} world_match={} complete={}",
        outcome["found"],
        outcome["attachment_removed"],
        len(outcome["episodic_removed"]),
        len(outcome["episodic_failed"]),
        len(outcome["transcripts_remaining"]),
        len(outcome["summaries_remaining"]),
        len(outcome["character_memories_remaining"]),
        outcome["world_state_match"],
        outcome["complete"],
    )
    return outcome
