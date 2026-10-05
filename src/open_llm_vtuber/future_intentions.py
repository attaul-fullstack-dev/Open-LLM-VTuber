"""Future intentions — explicit reminder requests with absolute UTC due times.

A future intention is a user-stated reminder request ("tolong ingetin aku
besok jam 7 ...", "jangan lupa nanti ..."). It lives in the existing
``character_state/<conf_uid>.json`` file (same atomic store as memories,
goals and preferences) — NOT a new memory system.

Layer separation:
- episodic memory: something that already happened (past only by design).
- character memories: durable facts ("ingat ...") — transient tasks rejected.
- interaction preferences: HOW Mili talks.
- future intentions (this module): WHAT Mili must remind the user about, WHEN.

Detection is deterministic and narrow: a turn must contain BOTH a reminder
verb AND a future anchor, and be short enough to be a request rather than
narration. No LLM call, no scheduler, no background work. All timestamps are
absolute UTC ISO-8601; relative labels ("besok", "nanti") are resolved ONCE
at capture into ``due_at`` and never persisted as truth.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from loguru import logger

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_CANCELLED = "cancelled"

_MAX_INTENTION_CHARS = 220
_MAX_PENDING = 20

# Reminder verbs: the user asks Mili to remind them.
_REMINDER_VERB = re.compile(
    r"\b(?:ingetin|ingetkan|ingatkan|ingatkah|jangan\s+lupa|jgn\s+lupa|"
    r"reminder|remind|tolong\s+ingat|catat\s+buat\s+besok)\b",
    re.IGNORECASE,
)

# Future anchors: when. Kept small and explicit; anything else stores with
# due_at=None (pending, visible in context, never proactively due).
_FUTURE_ANCHOR = re.compile(
    r"\b(?:besok|lusa|nanti|sore\s+ini|malam\s+ini|hari\s+ini|"
    r"minggu\s+depan|bulan\s+depan|tahun\s+depan|"
    r"tomorrow|tonight|next\s+week|next\s+month|"
    r"jam\s*\d{1,2}(?:[:.]\d{2})?|pukul\s*\d{1,2}(?:[:.]\d{2})?|"
    r"\b\d{1,2}[:.]\d{2}\b)\b",
    re.IGNORECASE,
)

# Questions that are pure recall checks, not reminder requests.
_RECALL_QUESTION = re.compile(
    r"^(?:kamu|lu|mili)?\s*(?:masih\s+)?(?:ingat|inget)\b.*\?\s*$",
    re.IGNORECASE,
)

_JAM = re.compile(
    r"(?:jam|pukul)?\s*(\d{1,2})(?:[:.](\d{2}))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FutureIntention:
    """One detected reminder request (pure data)."""

    text: str
    due_at: Optional[str]  # UTC ISO-8601 or None
    created_at: str  # UTC ISO-8601


def _now_aware(moment: Optional[datetime]) -> datetime:
    now = moment if moment is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _utc_iso(moment: datetime) -> str:
    return _now_aware(moment).isoformat(timespec="seconds")


def _resolve_tz(tz: Optional[str]) -> ZoneInfo:
    try:
        if tz:
            return ZoneInfo(str(tz))
    except Exception:
        pass
    return ZoneInfo("UTC")


def _normalize(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _parse_due_at(
    text: str, moment: datetime, tz: ZoneInfo
) -> Optional[str]:
    """Resolve the future anchor into an absolute UTC instant.

    Minimal and deterministic: besok/lusa/tomorrow → 07:00 local,
    explicit jam/HH:MM → that local time today (tomorrow if passed),
    nanti/sore/malam ini → +3h, minggu depan → +7d 07:00.
    Anything unrecognized → None (pending, never proactively due).
    """
    try:
        lowered = text.lower()
        local_now = moment.astimezone(tz)
        if "lusa" in lowered or "day after tomorrow" in lowered:
            due_local = (local_now + timedelta(days=2)).replace(
                hour=7, minute=0, second=0, microsecond=0
            )
            return due_local.astimezone(timezone.utc).isoformat(timespec="seconds")
        if "besok" in lowered or "tomorrow" in lowered:
            due_local = (local_now + timedelta(days=1)).replace(
                hour=7, minute=0, second=0, microsecond=0
            )
            return due_local.astimezone(timezone.utc).isoformat(timespec="seconds")
        if "minggu depan" in lowered or "next week" in lowered:
            due_local = (local_now + timedelta(days=7)).replace(
                hour=7, minute=0, second=0, microsecond=0
            )
            return due_local.astimezone(timezone.utc).isoformat(timespec="seconds")
        jam = _JAM.search(text)
        if jam:
            try:
                hour = max(0, min(23, int(jam.group(1))))
                minute = max(0, min(59, int(jam.group(2) or 0)))
            except (TypeError, ValueError):
                hour, minute = 7, 0
            due_local = local_now.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            if due_local <= local_now:
                due_local = due_local + timedelta(days=1)
            return due_local.astimezone(timezone.utc).isoformat(timespec="seconds")
        if (
            "nanti" in lowered
            or "sore ini" in lowered
            or "malam ini" in lowered
            or "tonight" in lowered
        ):
            return (moment + timedelta(hours=3)).isoformat(timespec="seconds")
        return None
    except Exception:
        return None


def detect_future_intention(
    user_text: Any,
    now: Optional[datetime] = None,
    tz: Optional[str] = None,
) -> Optional[FutureIntention]:
    """Detect ONE explicit reminder request in a single user turn.

    Returns None for ordinary conversation, narration about tomorrow,
    pure recall questions, overlong messages, and anything without BOTH
    a reminder verb and a future anchor. Pure and deterministic.
    """
    text = _normalize(user_text)
    if not text or len(text) > _MAX_INTENTION_CHARS:
        return None
    if _RECALL_QUESTION.match(text):
        return None
    if not _REMINDER_VERB.search(text):
        return None
    if not _FUTURE_ANCHOR.search(text):
        return None
    moment = _now_aware(now)
    zone = _resolve_tz(tz)
    return FutureIntention(
        text=text,
        due_at=_parse_due_at(text, moment, zone),
        created_at=_utc_iso(moment),
    )


def normalize_stored_intention(item: Any) -> Optional[Dict[str, Any]]:
    """Tolerant parse of one stored intention; None when unusable."""
    if not isinstance(item, dict):
        return None
    text = _normalize(item.get("text", ""))
    if not text:
        return None
    status = _normalize(item.get("status", "")) or STATUS_PENDING
    if status not in (STATUS_PENDING, STATUS_DONE, STATUS_CANCELLED):
        status = STATUS_PENDING
    due_at = _normalize(item.get("due_at", ""))
    created_at = _normalize(item.get("created_at", ""))
    return {
        "id": _normalize(item.get("id", "")) or uuid.uuid4().hex,
        "text": text,
        "due_at": due_at or None,
        "created_at": created_at or None,
        "status": status,
        "source": _normalize(item.get("source", "")) or "conversation",
    }


def pending_intentions(stored: Any) -> List[Dict[str, Any]]:
    """Pending intentions, oldest first (FIFO for reminders)."""
    out = []
    for item in stored or []:
        parsed = normalize_stored_intention(item)
        if parsed is not None and parsed["status"] == STATUS_PENDING:
            out.append(parsed)
    out.sort(key=lambda row: (row["created_at"] or "", row["id"]))
    return out


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def due_intentions(
    stored: Any,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Pending intentions whose absolute ``due_at`` has passed."""
    moment = _now_aware(now)
    out = []
    for item in pending_intentions(stored):
        stamp = _parse_iso(item.get("due_at"))
        if stamp is not None and stamp <= moment:
            out.append(item)
    return out


def add_future_intention(
    existing: Any,
    detected: FutureIntention,
    *,
    intention_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Append one intention with dedup (same normalized text stays one row).

    Pure: returns a new list, never mutates the input. Caps pending rows.
    """
    rows = []
    for item in existing or []:
        parsed = normalize_stored_intention(item)
        if parsed is not None:
            rows.append(parsed)
    normalized = _normalize(detected.text).lower()
    for row in rows:
        if row["status"] == STATUS_PENDING and row["text"].lower() == normalized:
            return rows
    rows.append(
        {
            "id": intention_id or uuid.uuid4().hex,
            "text": detected.text,
            "due_at": detected.due_at,
            "created_at": detected.created_at,
            "status": STATUS_PENDING,
            "source": "conversation",
        }
    )
    pending = [row for row in rows if row["status"] == STATUS_PENDING]
    if len(pending) > _MAX_PENDING:
        drop = len(pending) - _MAX_PENDING
        skipped = 0
        kept = []
        for row in rows:
            if row["status"] == STATUS_PENDING and skipped < drop:
                skipped += 1
                continue
            kept.append(row)
        rows = kept
    return rows


def complete_future_intention(
    existing: Any, intention_id: str
) -> tuple:
    """Mark one intention done. Pure ``(new_list, changed)``."""
    target = _normalize(intention_id)
    if not target:
        return (list(existing or []), False)
    changed = False
    out = []
    for item in existing or []:
        parsed = normalize_stored_intention(item)
        if parsed is None:
            continue
        if parsed["id"] == target and parsed["status"] == STATUS_PENDING:
            updated = dict(parsed)
            updated["status"] = STATUS_DONE
            out.append(updated)
            changed = True
        else:
            out.append(parsed)
    return (out, changed)


def cancel_future_intention(
    existing: Any, intention_id: str
) -> tuple:
    """Mark one intention cancelled. Pure ``(new_list, changed)``."""
    target = _normalize(intention_id)
    if not target:
        return (list(existing or []), False)
    changed = False
    out = []
    for item in existing or []:
        parsed = normalize_stored_intention(item)
        if parsed is None:
            continue
        if parsed["id"] == target and parsed["status"] == STATUS_PENDING:
            updated = dict(parsed)
            updated["status"] = STATUS_CANCELLED
            out.append(updated)
            changed = True
        else:
            out.append(parsed)
    return (out, changed)


def build_future_intention_context(
    stored: Any,
    *,
    tz: Optional[str] = None,
    max_tokens: int = 240,
) -> str:
    """Render pending intentions as a bounded prompt block (pure, no I/O).

    Absolute due times in the USER timezone; empty string when nothing is
    pending so the persona prompt stays untouched. No relative label is ever
    persisted — the rendered "in Xh" style suffix is derived per turn only
    when a due time exists, and the absolute instant is always shown.
    """
    pending = pending_intentions(stored)
    if not pending:
        return ""
    try:
        from .agent.context_window import estimate_tokens

        zone = _resolve_tz(tz)
        lines: List[str] = []
        used = 0
        for item in pending[:8]:
            due_raw = item.get("due_at")
            if due_raw:
                stamp = _parse_iso(due_raw)
                if stamp is not None:
                    local = stamp.astimezone(zone)
                    line = (
                        "- [remind due "
                        + local.strftime("%b %d %H:%M %Z")
                        + "] "
                        + item["text"]
                    )
                else:
                    line = "- [remind, no due time] " + item["text"]
            else:
                line = "- [remind, no due time] " + item["text"]
            cost = estimate_tokens(line) + 4
            if used + cost > max_tokens:
                break
            lines.append(line)
            used += cost
        if not lines:
            return ""
        header = (
            "Standing reminder requests from the user (explicit; surface them "
            "when due, never invent new ones):"
        )
        return f"{header}\n" + "\n".join(lines)
    except Exception as error:
        logger.warning(
            "Future intention context skipped: type={}",
            type(error).__name__,
        )
        return ""


__all__ = [
    "STATUS_CANCELLED",
    "STATUS_DONE",
    "STATUS_PENDING",
    "FutureIntention",
    "add_future_intention",
    "build_future_intention_context",
    "cancel_future_intention",
    "complete_future_intention",
    "detect_future_intention",
    "due_intentions",
    "normalize_stored_intention",
    "pending_intentions",
]
