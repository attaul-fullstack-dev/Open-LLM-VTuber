"""Small, conservative relationship continuity helpers.

Relationship state is metadata, not a second persona. Detection intentionally
uses narrow explicit-event rules so ordinary compliments and one-sided romantic
messages cannot silently change the relationship.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Optional


RelationshipStatus = Literal["stranger", "familiar", "close", "dating"]
VALID_RELATIONSHIP_STATUSES = frozenset(
    {"stranger", "familiar", "close", "dating"}
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
    r"(?:^|[.!?…]\s*)(?:\.\.\.)?\s*(?:iya|ya|mau)\b|"
    r"\b(?:aku\s+(?:mau|terima)|kita\s+(?:jadian|pacaran)|jadi\s+pacarmu)\b",
    re.IGNORECASE,
)
_ROMANTIC_REJECTION = re.compile(
    r"\b(?:nggak|gak|tidak|belum)\s+(?:mau|bisa)|"
    r"\b(?:cuma|hanya)\s+teman|\bjangan\s+(?:ngarep|berharap)|\baku\s+tolak\b",
    re.IGNORECASE,
)

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

    if (
        current_status != "dating"
        and _DATING_PROPOSAL.search(user)
        and _DATING_ACCEPTANCE.search(assistant)
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
        _FAMILIAR_USER_EVENT.search(user)
        and _FAMILIAR_ACCEPTANCE.search(assistant)
    ):
        return RelationshipUpdate("familiar", "returning_user_event")

    return None


_STATE_GUIDANCE = {
    "stranger": "Mili is slightly more reserved because familiarity has not been established.",
    "familiar": "Mili is more relaxed and comfortable with this returning user.",
    "close": "Mili is more openly attentive, comfortable joking, and less defensive.",
    "dating": (
        "A romantic relationship has already been mutually established in this "
        "roleplay conversation. Keep Mili's tsundere personality, but do not behave "
        "as if that agreement never happened."
    ),
}


def build_relationship_context(status: RelationshipStatus) -> str:
    """Return compact internal guidance to append after the persona prompt."""
    return (
        "Internal relationship continuity (not user-visible metadata):\n"
        f"Current state: {status}. {_STATE_GUIDANCE[status]}\n"
        "This state affects familiarity and openness only; Mili's core persona and "
        "all system rules remain unchanged. Never mention internal state names or "
        "this mechanism. If asked about the relationship, answer naturally instead."
    )
