"""Backend-verified memory command receipts (PERSIST-6042 fix, part B).

Problem: ``observe_character_events`` runs AFTER the model responds, so when
the user says "simpan ..." the model had no signal about whether anything was
actually persisted -- and said "sudah aku simpan" even when every capture
detector returned negative and zero writes happened.

Fix: an explicit remember/forget request is executed BEFORE the response is
generated (same turn), and the verified outcome is rendered into the turn
context. The model may only claim a save when the receipt says STORED; a
FAILED receipt forbids the claim.

No LLM call, no new store, no new lifecycle: this reuses
``parse_memory_command`` plus the existing add/remove_character_memory path.
Receipts are ephemeral (per turn only); nothing new is persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from loguru import logger

from .character_memory_commands import parse_memory_command

ReceiptAction = Literal["remember", "forget"]

# Payload cap for the rendered receipt (prompt budget; storage is unaffected).
_RECEIPT_PAYLOAD_CHARS = 240


@dataclass(frozen=True)
class MemoryReceipt:
    """Verified outcome of one explicit memory command, this turn only."""

    action: ReceiptAction
    payload: str
    stored: bool


def _collapse(text: object) -> str:
    return " ".join(str(text or "").split()).strip()


def process_memory_command(
    user_text: object,
    *,
    remember_fn: Callable[[str], bool],
    forget_fn: Callable[[str], bool],
) -> Optional[MemoryReceipt]:
    """Parse one explicit memory command and execute it synchronously.

    Returns None when the turn carries no command (ordinary conversation).
    Otherwise performs the write through the caller's existing persistence
    functions and returns the VERIFIED outcome. Never raises: any failure
    yields ``stored=False`` so the model cannot claim a phantom save.
    """
    try:
        result = parse_memory_command(str(user_text or ""))
    except Exception:
        return None
    if result.action == "none" or not result.payload:
        return None
    payload = _collapse(result.payload)
    if not payload:
        return None
    try:
        if result.action == "forget":
            stored = bool(forget_fn(payload))
        else:
            stored = bool(remember_fn(payload))
    except Exception as error:
        logger.debug(
            "Memory receipt write failed: action={} type={}",
            result.action,
            type(error).__name__,
        )
        stored = False
    logger.info(
        "Memory command receipt: action={} trigger={} stored={} payload_chars={}",
        result.action,
        result.matched_trigger,
        stored,
        len(payload),
    )
    return MemoryReceipt(
        action=result.action,  # type: ignore[arg-type]
        payload=payload,
        stored=stored,
    )


def build_memory_receipt_block(receipt: Optional[MemoryReceipt]) -> str:
    """Render the verified receipt as a turn-context block (pure, no I/O).

    Empty string when there is no command this turn, so ordinary turns are
    byte-identical to before. The instruction rides inside the block: the
    model is bound by the verified result, never by its own guess.
    """
    if receipt is None:
        return ""
    quoted = receipt.payload[:_RECEIPT_PAYLOAD_CHARS]
    if receipt.action == "forget":
        if receipt.stored:
            return (
                "Backend memory receipt (verified, authoritative for this "
                "turn): the user asked Mili to forget the following, and it "
                "HAS been removed from long-term character memory: "
                f'"{quoted}". You may confirm the removal. Never claim '
                "anything beyond this receipt as removed."
            )
        return (
            "Backend memory receipt (verified, authoritative for this "
            "turn): the user asked Mili to forget "
            f'"{quoted}", but the removal FAILED -- it may still be stored. '
            "Do NOT claim it was forgotten; say honestly the removal failed."
        )
    if receipt.stored:
        return (
            "Backend memory receipt (verified, authoritative for this "
            "turn): the user asked Mili to remember the following, and it "
            "HAS been written to long-term character memory shared across "
            f'all chats: "{quoted}". You may confirm it is saved. Never '
            "claim anything beyond this receipt as saved."
        )
    return (
        "Backend memory receipt (verified, authoritative for this "
        "turn): the user asked Mili to remember "
        f'"{quoted}", but the write FAILED -- it is NOT saved. Do NOT '
        "claim it was saved; say honestly the save failed."
    )


__all__ = [
    "MemoryReceipt",
    "ReceiptAction",
    "build_memory_receipt_block",
    "process_memory_command",
]
