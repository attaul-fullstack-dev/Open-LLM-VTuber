"""Small, conservative relationship continuity helpers.

Relationship state is metadata, not a second persona. Detection intentionally
uses narrow explicit-event rules so ordinary compliments and one-sided romantic
messages cannot silently change the relationship.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from datetime import datetime
from typing import Literal, Optional

from ..world_state import relative_day_parts


RelationshipStatus = Literal["stranger", "familiar", "close", "dating", "married"]
VALID_RELATIONSHIP_STATUSES = frozenset(
    {"stranger", "familiar", "close", "dating", "married"}
)


@dataclass(frozen=True)
class RelationshipState:
    status: RelationshipStatus = "stranger"
    updated_at: Optional[str] = None
    reason: str = "default"


@dataclass(frozen=True)
class RelationshipUpdate:
    new_status: RelationshipStatus
    trigger: str


_DATING_PROPOSAL = re.compile(
    r"\b(?:mau(?:kah|\s+nggak|\s+ngga|\s+gak|\s+ga)?\s+(?:jadi\s+)?pacar(?:ku|\s+aku)?|"
    r"jadi\s+pacar(?:ku|\s+aku)|pacaran\s+(?:sama|dengan)\s+aku|"
    r"kita\s+(?:jadian|pacaran)|(?:jadian|pacaran)\s+yuk|"
    r"mau\s+jadian\s+(?:sama|dengan)\s+aku|"
    r"jadi\s+pasangan(?:ku|\s+aku)?)\b",
    re.IGNORECASE,
)
_DATING_ACCEPTANCE = re.compile(
    r"(?:^|[,.!?;…]\s*)(?:\.\.\.)?\s*(?:iya|ya|mau)\b|"
    r"\b(?:iya|ya)\s*(?:[,.;…]+\s*)?(?:deh|dong|aku\s+mau|mau)\b|"
    r"\b(?:aku\s+(?:mau|terima)|kita\s+(?:jadian|pacaran)|jadi\s+pacarmu)\b",
    re.IGNORECASE,
)
_ROMANTIC_REJECTION = re.compile(
    r"(?:^|[.!?…]\s*)(?:nggak|ngga|gak|ga|tidak)(?:[.!?…]|$)|"
    r"\b(?:nggak|ngga|gak|ga|tidak|belum)\s+"
    r"(?:mau|bisa|setuju|pacaran|jadi\s+pacar)\b|"
    r"\b(?:jawaban(?:ku)?|jawabanku)\b.{0,24}\b"
    r"(?:nggak|ngga|gak|ga|tidak)\b|"
    r"\b(?:tapi|namun)\b.{0,80}\b"
    r"(?:nggak|ngga|gak|ga|tidak|belum|bukan)\b|"
    r"\b(?:bukan\s+(?:jadi\s+)?pacar|tetap\s+teman|teman\s+aja)\b|"
    r"\b(?:mungkin\b.{0,50}\b(?:nanti|suatu\s+hari)|sekarang\s+belum)\b|"
    r"\b(?:cuma|hanya)\s+teman|\bjangan\s+(?:ngarep|berharap)|\baku\s+tolak\b",
    re.IGNORECASE,
)

# Marriage is a one-way, terminal step above "dating": it is only ever reached
# from an explicitly established romance, and it is never inferred, undone or
# downgraded automatically. This layer records durable facts, it does not
# arbitrate the user's life, so there is deliberately no divorce rule.
_MARRIAGE_USER_EVENT = re.compile(
    r"\b(?:kita\s+(?:udah|sudah|udah\s+ya|sudah\s+ya)\s+"
    r"(?:nikah|menikah)|"
    r"(?:aku|kamu)\s+(?:udah\s+)?nikah(?:an)?|"
    r"kita\s+(?:sekarang\s+)?(?:suami\s+istri|istri\s+suami)|"
    r"aku\s+(?:suami|kekasih)\s+kamu|kamu\s+(?:istri|kekasih)\s+aku|"
    r"kita\s+(?:sudah|udah)\s+terikat|"
    r"pernikahan\s+kita|"
    r"kita\s+sudah\s+menikah)\b",
    re.IGNORECASE,
)
_MARRIAGE_ACCEPTANCE = re.compile(
    r"\b(?:kita\s+(?:udah|sudah|udah\s+ya|sudah\s+ya)\s+"
    r"(?:nikah|menikah)|"
    r"kita\s+(?:sekarang\s+)?(?:suami\s+istri|istri\s+suami)|"
    r"aku\s+(?:istri|kekasih)\s+kamu|kamu\s+(?:suami|kekasih)\s+aku|"
    r"(?:ya|iya|benar)\s*[,.\s]+kita\s+(?:udah|sudah)|"
    r"aku\s+(?:juga\s+)?setuju|aku\s+mau|sudah\s+kita\s+nikah)\b",
    re.IGNORECASE,
)
# Marriage-specific refusal. Kept separate from ``_ROMANTIC_REJECTION`` so the
# dating rule above keeps its exact previous behaviour.
_MARRIAGE_REJECTION = re.compile(
    r"\b(?:belum|nggak|ngga|gak|ga|tidak)\s+(?:nikah|menikah|terikat)\b|"
    r"\b(?:nggak|ngga|gak|ga|tidak|mau\s+nggak)\s+(?:mau|bisa|setuju)\s+"
    r"(?:nikah|menikah)\b|"
    r"\bbukan\s+(?:suami|istri|kekasih)\b|"
    r"\bmasih\s+(?:pacar|pacaran|belum\s+nikah)\b|"
    r"\bkita\s+(?:baru|masih)\s+(?:pacaran|kenal)\b|"
    r"\baku\s+tolak\b|\baku\s+nggak\s+mau\b",
    re.IGNORECASE,
)

# Acceptance must be near the start of the answer. This leaves room for a few
# natural tsundere hesitation sentences without treating a late, incidental
# "iya" as mutual agreement.
_DATING_ACCEPTANCE_PREFIX_CHARS = 240

_CLOSE_USER_EVENT = re.compile(
    r"\b(?:aku\s+(?:percaya|nyaman\s+cerita)\s+(?:sama|dengan)\s+kamu|"
    r"kamu\s+(?:berarti|penting)\s+(?:banget\s+)?buat\s+aku|"
    r"kita\s+(?:udah|sudah)\s+(?:dekat|akrab))\b",
    re.IGNORECASE,
)
_CLOSE_ACCEPTANCE = re.compile(
    r"\b(?:aku\s+juga|percaya\s+(?:sama|dengan)\s+(?:kamu|aku)|"
    r"kita\s+(?:memang\s+)?(?:udah|sudah)\s+(?:dekat|akrab)|"
    r"kamu\s+juga\s+(?:berarti|penting)|senang\s+kamu\s+percaya)\b",
    re.IGNORECASE,
)

_FAMILIAR_USER_EVENT = re.compile(
    r"\b(?:aku\s+(?:balik|datang)\s+lagi|masih\s+ingat\s+aku|"
    r"kita\s+(?:pernah|udah|sudah)\s+(?:ngobrol|ketemu|bahas))\b",
    re.IGNORECASE,
)
_FAMILIAR_ACCEPTANCE = re.compile(
    r"\b(?:ingat\s+(?:kok|lah|dong|kamu)|balik\s+lagi|datang\s+lagi|"
    r"tentu\s+(?:ingat|aja)|iya,?\s+(?:aku\s+)?ingat|"
    r"pernah\s+(?:ngobrol|bahas))\b",
    re.IGNORECASE,
)


def normalize_relationship_status(value: object) -> RelationshipStatus:
    normalized = str(value or "").strip().lower()
    if normalized in VALID_RELATIONSHIP_STATUSES:
        return normalized  # type: ignore[return-value]
    return "stranger"


def detect_relationship_update(
    current_status: RelationshipStatus,
    user_text: str,
    assistant_text: str,
) -> Optional[RelationshipUpdate]:
    """Detect only explicit, mutually acknowledged relationship events."""
    user = " ".join((user_text or "").split())
    assistant = " ".join((assistant_text or "").split())
    # Live2D expression markers are transport hints, not relationship language.
    assistant = re.sub(r"^\s*(?:\[[\w-]+\]\s*)+", "", assistant)
    if not user or not assistant:
        return None

    acceptance_prefix = assistant[:_DATING_ACCEPTANCE_PREFIX_CHARS]
    # Marriage sits ABOVE dating and is terminal: once married, nothing in this
    # layer can walk it back, and an ordinary conversation can never re-open the
    # question. Checked first so a married character never re-detects dating.
    if current_status == "married":
        return None
    if (
        current_status == "dating"
        and _MARRIAGE_USER_EVENT.search(user)
        and _MARRIAGE_ACCEPTANCE.search(acceptance_prefix)
        and not _MARRIAGE_REJECTION.search(assistant)
    ):
        return RelationshipUpdate("married", "explicit_marriage_event")
    if (
        current_status != "dating"
        and _DATING_PROPOSAL.search(user)
        and _DATING_ACCEPTANCE.search(acceptance_prefix)
        and not _ROMANTIC_REJECTION.search(assistant)
    ):
        return RelationshipUpdate("dating", "explicit_relationship_event")

    if current_status in {"stranger", "familiar"} and (
        _CLOSE_USER_EVENT.search(user)
        and _CLOSE_ACCEPTANCE.search(assistant)
        and not _ROMANTIC_REJECTION.search(assistant)
    ):
        return RelationshipUpdate("close", "mutual_trust_event")

    if current_status == "stranger" and (
        _FAMILIAR_USER_EVENT.search(user) and _FAMILIAR_ACCEPTANCE.search(assistant)
    ):
        return RelationshipUpdate("familiar", "returning_user_event")

    return None


_STATE_GUIDANCE = {
    "stranger": "Familiarity has not been established yet.",
    "familiar": "Mutual familiarity has been established in an earlier conversation.",
    "close": "Mutual trust has been established in an earlier conversation.",
    "dating": (
        "A romantic relationship has already been mutually established in this "
        "roleplay conversation. Do not behave as if that agreement never "
        "happened. How Mili expresses herself still follows her persona prompt "
        "and what the conversation has established, so read the persona rules "
        "and the stored memories before answering."
    ),
    "married": (
        "You and the user are already married in this roleplay conversation. "
        "Treat it as long-settled and durable: never re-open the proposal, "
        "never ask whether you are really together, and never fall back to "
        "'just dating', 'only a girlfriend/boyfriend' or any earlier stage. "
        "The marriage is a fact of this relationship, exactly like a stored "
        "memory, and it survives new conversations. How Mili expresses herself "
        "still follows her persona prompt and what the conversation has "
        "established."
    ),
}


def build_relationship_context(
    status: RelationshipStatus,
    updated_at: Optional[str] = None,
    *,
    tz: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    """Return compact internal guidance to append after the persona prompt.

    When a parseable ``updated_at`` is given, the status line carries a
    render-time age (e.g. ``close (status updated 3 days ago, Sep 26)``) so
    the LLM can weight recency. Stored state is never rewritten.
    """
    state_line = f"Current state: {status}."
    if updated_at:
        parts = relative_day_parts(updated_at, now, tz)
        if parts is not None:
            label, date_str = parts
            state_line = (
                f"Current state: {status} (status updated {label.lower()}, {date_str})."
            )
    return (
        "Internal relationship continuity (not user-visible metadata):\n"
        f"{state_line} {_STATE_GUIDANCE[status]}\n"
        "This state affects familiarity and openness only; Mili's core persona and "
        "all system rules remain unchanged. Never mention internal state names or "
        "this mechanism. If asked about the relationship, answer naturally instead."
    )
