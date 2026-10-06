"""Future intentions — reminder requests AND natural future plans.

Two capture paths share one bounded store (same ``character_state`` file —
NOT a new memory system):

- ``reminder``: explicit request ("tolong ingetin aku besok jam 7 ...").
  Narrow verb + anchor grammar (unchanged).
- ``plan`` (Phase 7): natural first-person future declaration ("besok aku
  ada ujian", "Selasa aku bakal nanya lagi soal bug websocket").
  Narrow subject + marker grammar; hypotheticals, questions, general
  statements and stage-direction roleplay are rejected. False positive is
  worse than false negative, so the grammar stays small on purpose.

Layer separation:
- episodic memory: something that already happened (past only by design).
- character memories: durable facts ("ingat ...") — transient tasks rejected.
- interaction preferences: HOW Mili talks.
- future intentions (this module): WHAT the user will do / must be reminded
  about, WHEN (absolute UTC ``due_at``, or None when undated).

No LLM call, no scheduler, no background work. Relative labels ("besok",
"Selasa", "nanti") are resolved ONCE at capture into ``due_at`` and never
persisted as truth. Undated plans (``due_at=None``) are recallable but never
proactively due.
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

KIND_REMINDER = "reminder"
KIND_PLAN = "plan"

_MAX_INTENTION_CHARS = 220
_MAX_PENDING = 20
_MIN_PLAN_CHARS = 12

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

# First-person subject: plans are statements about the USER's own future.
# Third-person/general narration ("orang biasanya kerja besok") has no
# subject here and is rejected. Fiction with a first-person narrator is a
# documented residual risk; verbatim recall (never invented detail) bounds it.
_PLAN_SUBJECT = re.compile(
    r"\b(?:aku|gw|gue|gua|saya|ane)\b",
    re.IGNORECASE,
)

# Future markers for natural plans: explicit anchors plus intent verbs.
# Kept narrow on purpose; habitual/uncertain wording is rejected elsewhere.
_PLAN_MARKER = re.compile(
    r"\b(?:besok|lusa|nanti|minggu\s+depan|bulan\s+depan|tahun\s+depan|"
    r"senin|selasa|rabu|kamis|jumat|sabtu|minggu|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"tomorrow|next\s+week|next\s+month|"
    r"bakal|akan|mau\b|rencana|berencana|planning|will\b|going\s+to|"
    r"jam\s*\d{1,2}(?:[:.]\d{2})?|pukul\s*\d{1,2}(?:[:.]\d{2})?)\b",
    re.IGNORECASE,
)

# Uncertainty / conditional: a plan that might not happen is not a plan.
_HYPOTHETICAL = re.compile(
    r"\b(?:kalau|kalo|jika|jikalau|bila|mungkin|misalnya|misal|seandainya|"
    r"andaikan|kayaknya|barangkali|siapa\s+tahu|maybe|if\b|perhaps)\b",
    re.IGNORECASE,
)

# Habitual statements describe routine, not a dated future intention.
_HABITUAL = re.compile(
    r"\b(?:biasanya|selalu|sering|setiap\s+(?:hari|pagi|minggu|bulan)|"
    r"tiap\s+(?:hari|pagi)|kebiasaan|usually|always|every\s+day)\b",
    re.IGNORECASE,
)

# Stage directions / quoted speech: roleplay framing, not a certain plan.
_ROLEPLAY_FRAME = re.compile(r"^\s*[\*\(>]|[\*]$")

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
    """One detected reminder request or future plan (pure data)."""

    text: str
    due_at: Optional[str]  # UTC ISO-8601 or None (undated plan)
    created_at: str  # UTC ISO-8601
    kind: str = KIND_REMINDER  # KIND_REMINDER | KIND_PLAN
    tz: Optional[str] = None  # user tz name at capture (audit only)


# Weekday (full names only — shorts are ambiguous) → Python weekday().
_WEEKDAYS = {
    "senin": 0, "monday": 0,
    "selasa": 1, "tuesday": 1,
    "rabu": 2, "wednesday": 2,
    "kamis": 3, "thursday": 3,
    "jumat": 4, "friday": 4,
    "sabtu": 5, "saturday": 5,
    "minggu": 6, "sunday": 6,
}


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

    Day words set the DATE (besok/lusa/weekday/minggu/bulan depan, all at
    07:00 local unless an explicit jam overrides the time); a bare jam sets
    today (tomorrow if passed); nanti/sore/malam ini mean +3h.
    Anything unrecognized → None (stored, recallable, never proactively due).
    """
    try:
        lowered = text.lower()
        local_now = moment.astimezone(tz)

        def _at(day: datetime, hour: int, minute: int = 0) -> str:
            due_local = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return due_local.astimezone(timezone.utc).isoformat(timespec="seconds")

        def _jam_or(h: int, m: int = 0) -> tuple:
            jam = _JAM.search(text)
            if jam:
                try:
                    h = max(0, min(23, int(jam.group(1))))
                    m = max(0, min(59, int(jam.group(2) or 0)))
                except (TypeError, ValueError):
                    pass
            return h, m

        for name, weekday in _WEEKDAYS.items():
            if re.search(r"\b" + name + r"\b", lowered):
                days_ahead = (weekday - local_now.weekday()) % 7
                day = local_now + timedelta(days=days_ahead)
                hour, minute = _jam_or(7)
                due_local = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if due_local <= local_now:
                    due_local = due_local + timedelta(days=7)
                return due_local.astimezone(timezone.utc).isoformat(timespec="seconds")
        if "lusa" in lowered or "day after tomorrow" in lowered:
            day = local_now + timedelta(days=2)
            hour, minute = _jam_or(7)
            return _at(day, hour, minute)
        if "besok" in lowered or "tomorrow" in lowered:
            day = local_now + timedelta(days=1)
            hour, minute = _jam_or(7)
            return _at(day, hour, minute)
        if "minggu depan" in lowered or "next week" in lowered:
            day = local_now + timedelta(days=7)
            hour, minute = _jam_or(7)
            return _at(day, hour, minute)
        if "bulan depan" in lowered or "next month" in lowered:
            day = local_now + timedelta(days=30)
            hour, minute = _jam_or(7)
            return _at(day, hour, minute)
        jam = _JAM.search(text)
        if jam:
            hour, minute = _jam_or(7)
            due_local = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
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
        kind=KIND_REMINDER,
        tz=str(tz or "") or None,
    )


def detect_future_plan(
    user_text: Any,
    now: Optional[datetime] = None,
    tz: Optional[str] = None,
) -> Optional[FutureIntention]:
    """Detect ONE natural future plan in a single user turn (category B).

    Requires first-person subject + future marker + minimal substance.
    Rejects hypotheticals (C), questions (D), habitual/general statements
    (E) and stage-direction roleplay frames (F). Undated plans return
    ``due_at=None``: stored and recallable, never proactively due.
    Pure and deterministic.
    """
    text = _normalize(user_text)
    if not text or len(text) < _MIN_PLAN_CHARS or len(text) > _MAX_INTENTION_CHARS:
        return None
    if text.rstrip().endswith(("?", "？")):
        return None
    if _RECALL_QUESTION.match(text):
        return None
    if _HYPOTHETICAL.search(text):
        return None
    if _HABITUAL.search(text):
        return None
    if _ROLEPLAY_FRAME.search(text):
        return None
    if not _PLAN_SUBJECT.search(text):
        return None
    if not _PLAN_MARKER.search(text):
        return None
    moment = _now_aware(now)
    zone = _resolve_tz(tz)
    return FutureIntention(
        text=text,
        due_at=_parse_due_at(text, moment, zone),
        created_at=_utc_iso(moment),
        kind=KIND_PLAN,
        tz=str(tz or "") or None,
    )


def detect_any_future_intention(
    user_text: Any,
    now: Optional[datetime] = None,
    tz: Optional[str] = None,
) -> Optional[FutureIntention]:
    """Reminder first (category A), then natural plan (category B)."""
    found = detect_future_intention(user_text, now=now, tz=tz)
    if found is not None:
        return found
    return detect_future_plan(user_text, now=now, tz=tz)


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
    kind = _normalize(item.get("kind", "")) or KIND_REMINDER
    if kind not in (KIND_REMINDER, KIND_PLAN):
        kind = KIND_REMINDER
    return {
        "id": _normalize(item.get("id", "")) or uuid.uuid4().hex,
        "text": text,
        "due_at": due_at or None,
        "created_at": created_at or None,
        "status": status,
        "source": _normalize(item.get("source", "")) or "conversation",
        "kind": kind,
        "tz": _normalize(item.get("tz", "")) or None,
    }


def pending_intentions(stored: Any) -> List[Dict[str, Any]]:
    """Pending intentions, temporally relevant first (deterministic).

    Due-soonest absolute ``due_at`` first so the bounded prompt block always
    carries the most time-critical rows; undated rows sort last (they are
    recallable but never urgent); ties break by oldest creation. ISO-8601
    strings sort chronologically, so no parsing is needed for ordering.
    """
    out = []
    for item in stored or []:
        parsed = normalize_stored_intention(item)
        if parsed is not None and parsed["status"] == STATUS_PENDING:
            out.append(parsed)
    out.sort(
        key=lambda row: (
            row["due_at"] or "9999",
            row["created_at"] or "",
            row["id"],
        )
    )
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
            "kind": getattr(detected, "kind", KIND_REMINDER) or KIND_REMINDER,
            "tz": getattr(detected, "tz", None),
        }
    )
    pending = [row for row in rows if row["status"] == STATUS_PENDING]
    if len(pending) > _MAX_PENDING:
        # Overflow evicts the least recall-critical rows first: undated rows
        # (no absolute WHEN, never proactively due) go before dated ones, so
        # a dated intention ("besok jam 07.00 ...") survives an accumulation
        # of undated immediate-desire rows. Oldest first within each group
        # (rows are appended chronologically); done/cancelled rows are never
        # evicted here. Deterministic and pure.
        drop = len(pending) - _MAX_PENDING
        pending_ids = [row["id"] for row in rows if row["status"] == STATUS_PENDING]
        undated_ids = [
            row["id"]
            for row in rows
            if row["status"] == STATUS_PENDING and not row.get("due_at")
        ]
        dated_ids = [
            row["id"]
            for row in rows
            if row["status"] == STATUS_PENDING and row.get("due_at")
        ]
        evict = set(undated_ids[:drop])
        remaining = drop - len(evict)
        if remaining > 0:
            evict.update(dated_ids[:remaining])
        # Sanity fallback: if id bookkeeping ever disagrees (duplicate ids),
        # evict the oldest pending rows in storage order as before.
        if len(evict) < drop:
            for row_id in pending_ids:
                if len(evict) >= drop:
                    break
                evict.add(row_id)
        rows = [row for row in rows if row["id"] not in evict]
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

    Each row also carries a ``stated <Mon DD>`` tag derived per turn from
    the persisted ``created_at`` (user timezone), so recall questions of the
    form "kemarin aku bilang ... apa?" ground to the right row. Rows without
    a parseable ``created_at`` render exactly as before (tag omitted).
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
            kind = item.get("kind") or KIND_REMINDER
            tag = "remind" if kind == KIND_REMINDER else "upcoming"
            stated = ""
            created_raw = item.get("created_at")
            if created_raw:
                created_stamp = _parse_iso(created_raw)
                if created_stamp is not None:
                    stated = ", stated " + created_stamp.astimezone(zone).strftime(
                        "%b %d"
                    )
            due_raw = item.get("due_at")
            if due_raw:
                stamp = _parse_iso(due_raw)
                if stamp is not None:
                    local = stamp.astimezone(zone)
                    line = (
                        "- ["
                        + tag
                        + " due "
                        + local.strftime("%b %d %H:%M %Z")
                        + stated
                        + "] "
                        + item["text"]
                    )
                else:
                    line = "- [" + tag + ", no due time" + stated + "] " + item["text"]
            else:
                line = "- [" + tag + ", no due time" + stated + "] " + item["text"]
            cost = estimate_tokens(line) + 4
            if used + cost > max_tokens:
                break
            lines.append(line)
            used += cost
        if not lines:
            return ""
        header = (
            "Standing reminder requests and upcoming user plans (explicit or "
            "directly stated; surface them when due, never invent new ones):"
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
    "KIND_PLAN",
    "KIND_REMINDER",
    "FutureIntention",
    "add_future_intention",
    "build_future_intention_context",
    "cancel_future_intention",
    "complete_future_intention",
    "detect_any_future_intention",
    "detect_future_intention",
    "detect_future_plan",
    "due_intentions",
    "normalize_stored_intention",
    "pending_intentions",
]
