"""Backend-verified reminder receipts (pre-turn persistence).

Problem: reminder capture used to run only in the post-turn observer, after
the model had already answered. The model could therefore promise "nanti aku
ingetin" even when the turn was never stored -- exactly the failure behind a
reminder that never appears in the Memory Manager.

Fix: the reminder is parsed and persisted BEFORE the response is generated,
and the verified outcome is rendered into the turn context. The model may only
confirm a scheduled reminder when the receipt says ``captured``. An ambiguous
reference renders a ``clarify`` receipt: nothing was stored, so the model must
ask one short clarifying question instead of inventing content or claiming a
reminder exists.

No LLM call, no new store, no new lifecycle: this reuses the existing
deterministic detector and the existing ``future_intentions`` store. Receipts
are ephemeral (per turn only); nothing new is persisted here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from loguru import logger

_MAX_RECEIPT_TEXT = 240


def _collapse(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _local_due(row: Dict[str, Any], tz: Optional[str]) -> str:
    raw = _collapse(row.get("due_at"))
    if not raw:
        return ""
    try:
        stamp = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return ""
    if stamp.tzinfo is None:
        return ""
    try:
        zone = ZoneInfo(str(tz)) if tz else ZoneInfo("UTC")
    except Exception:
        zone = ZoneInfo("UTC")
    return stamp.astimezone(zone).strftime("%Y-%m-%d %H:%M %Z")


def build_reminder_receipt_block(
    status: str,
    row: Optional[Dict[str, Any]] = None,
    *,
    tz: Optional[str] = None,
) -> str:
    """Render the verified outcome as a turn-context block (pure, no I/O).

    Empty string when the turn carried no reminder request, so ordinary turns
    stay byte-identical. The wording binds the model to the backend outcome.
    """
    if status == "captured" and isinstance(row, dict):
        due = _local_due(row, tz)
        payload = _collapse(row.get("delivery_text"))
        detail = f' Reminder text: "{payload[: _MAX_RECEIPT_TEXT]}".' if payload else ""
        if due:
            return (
                "Backend reminder receipt (verified, authoritative for this turn): "
                f"a reminder IS stored right now due {due}.{detail} You may confirm it "
                "is scheduled and that you will deliver it at that time. Never "
                "claim any other reminder as stored."
            )
        return (
            "Backend reminder receipt (verified, authoritative for this turn): "
            f"a reminder IS stored right now, but NO valid due time was resolved.{detail} "
            "It will NOT be delivered automatically. Do NOT claim it is scheduled "
            "or promise a delivery time; ask the user for a specific time if they "
            "want proactive delivery. Never claim any other reminder as stored."
        )
    if status == "clarify":
        return (
            "Backend reminder receipt (verified, authoritative for this turn): "
            "the user asked for a reminder that refers to earlier content, but "
            "the referenced text could NOT be resolved from the conversation. "
            "NOTHING was stored. Ask ONE short clarifying question about what "
            "exactly you should say, and do NOT claim a reminder is scheduled."
        )
    if status == "failed":
        return (
            "Backend reminder receipt (verified, authoritative for this turn): "
            "the user asked for a reminder but the write FAILED. It is NOT "
            "stored. Do NOT claim it is scheduled; say honestly that saving "
            "the reminder failed."
        )
    return ""


def process_reminder_turn(
    user_text: str,
    *,
    capture_fn,
    recent_messages: Optional[Any] = None,
    tz: Optional[str] = None,
) -> str:
    """Run verified capture and return the receipt block for this turn.

    ``capture_fn(status)`` must return ``(status, row)`` from
    ``character_state.capture_reminder``. Never raises: any failure renders a
    ``failed`` receipt so the model cannot claim a phantom save.
    """
    try:
        status, row = capture_fn()
    except Exception as error:
        logger.debug(
            "Reminder receipt capture failed: type={}", type(error).__name__
        )
        status, row = "failed", None
    block = build_reminder_receipt_block(status, row, tz=tz)
    if block:
        logger.info(
            "Reminder receipt: status={} due_present={} resolved={}",
            status,
            bool((row or {}).get("due_at")),
            bool((row or {}).get("delivery_text")),
        )
    return block


__all__ = [
    "build_reminder_receipt_block",
    "process_reminder_turn",
]
