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


# ---------------------------------------------------------------------------
# Explicit relationship assertions -- the ONLY way the state ever moves.
# ---------------------------------------------------------------------------
#
# A relationship change is a deliberate act by the user, so the detector needs a
# statement about the relationship *as it stands now*, plus Mili agreeing.
# Everything else is inert:
#
#   * narrative / past reference ("kita pernah bicara soal pernikahan",
#     "aku ingat waktu kita pacaran dulu", "kenapa kita duluplicity") -> no move
#   * questions ("kita pacaran duluan?")                                 -> no move
#   * a one-sided statement without acknowledgement                      -> no move
#
# This is what makes the state mutable without making it twitchy: it can go up
# AND down between any two tiers, but only on an explicit, mutually confirmed
# present-tense change.

# Present-tense assertion: "we are X now", "we became X", "let's be X",
# "from now on we are X", "let's go back to being X".
_ASSERT_NOW = (
    r"(?:kita|aku\s+dan\s+kamu|aku\s+kamu)\s+"
    r"(?:sekarang\s+|udah\s+|sudah\s+|udah\s+ya\s+|sudah\s+ya\s+|"
    r"justru\s+|kembali\s+|lagi\s+|mulai\s+|dari\s+sekarang\s+)*"
)
_ASSERT_WISH = r"(?:aku\s+(?:mau|ingin|berharap|pengen)|mulai\s+sekarang|kita\s+ubah)"
# Used only on the assertion path. A bare recall question ("kita pacaran
# duluan?") is not a change, but a proposal phrased as a question ("mau gak
# jadi pacarku?") absolutely is, so the legacy proposal detectors below are
# deliberately NOT gated on this.
_QUESTION_FORM = re.compile(
    r"\?\s*$|^\s*(?:kapan|kenapa|mengapa|apa|siapa|dimana|berapa|gimana)\b",
    re.IGNORECASE,
)
# Anything that points at the past, a hypothetical or a topic of conversation
# rather than a statement of the present state.
# Wording that points at the past, a memory or a topic of conversation instead
# of asserting what the relationship is NOW. Narrow on purpose: it must not
# swallow a genuine proposal such as "mau gak jadi pacarku?" nor a real
# re-connection such as "aku balik lagi, masih ingat aku?".
_NARRATIVE_FRAME = re.compile(
    r"\b(?:dulu(?:an)?|kelu|kemarin(?:nya)?|sebelumnya|pertama|"
    r"waktu\s+(?:kita|aku)|pas\s+(?:kita|aku)\s+(?:lagi\s+)?"
    r"(?:bicara|ngobrol|bahas|diskusi)|"
    r"kenapa\s+(?:kita|aku)|"
    r"pernah\s+(?:bahas|ngobrol|bicara|diskusi)|"
    r"topik\s+tentang|pembahasan\s+tentang)\b",
    re.IGNORECASE,
)

# target tier -> present-tense assertion patterns
_RELATIONSHIP_ASSERTIONS = (
    (
        "married",
        (
            _ASSERT_NOW
            + r"(?:nikah|menikah|bersuami|beristri|"
            r"suami\s+istri|istri\s+suami|terikat|kekasih)\b",
            _ASSERT_WISH
            + r"[^.?!]{0,24}\b(?:nikah|menikah|terikat)\b",
        ),
    ),
    (
        "dating",
        (
            # "pacaran lagi/kembali", "jadi pacaran lagi", "pacaran aja"
            _ASSERT_NOW
            + r"(?:pacaran|pacar|pacar\s+aja|pacar\s+lagi|jadians?)\b",
            _ASSERT_NOW + r"balik\s+(?:ke\s+)?(?:pacaran|pacar)\b",
            _ASSERT_NOW + r"ke\s+(?:pacaran|pacar)\b",
            _ASSERT_WISH
            + r"[^.?!]{0,24}\b(?:pacaran|pacar|kembali)\b",
        ),
    ),
    (
        "familiar",
        (
            _ASSERT_NOW + r"(?:berteman|teman)\s*(?:aja|saja)?\b",
            _ASSERT_NOW + r"hanya\s+(?:berteman|teman)\b",
            _ASSERT_WISH
            + r"[^.?!]{0,24}\b(?:berteman|teman\s+aja|hanya\s+teman)\b",
        ),
    ),
)


def _relationship_assertion_target(user_text: str) -> Optional[str]:
    """The tier an explicit present-tense assertion asks for, else None."""
    if _QUESTION_FORM.search(user_text):
        return None
    if _NARRATIVE_FRAME.search(user_text):
        return None
    for tier, patterns in _RELATIONSHIP_ASSERTIONS:
        for pattern in patterns:
            try:
                if re.search(pattern, user_text, re.IGNORECASE):
                    return tier
            except re.error:  # pragma: no cover - pattern is static
                continue
    return None


def _relationship_acknowledges(target: str, assistant_text: str) -> bool:
    """True when Mili agrees to the requested change (mutual, one-sided no)."""
    prefix = assistant_text[:_DATING_ACCEPTANCE_PREFIX_CHARS]
    if _ROMANTIC_REJECTION.search(assistant_text):
        return False
    if target == "married":
        return bool(_MARRIAGE_ACCEPTANCE.search(prefix)) and not _MARRIAGE_REJECTION.search(
            assistant_text
        )
    # A downshift is agreed with the same plain acceptance language.
    return bool(_DATING_ACCEPTANCE.search(prefix))


def detect_relationship_update(
    current_status: RelationshipStatus,
    user_text: str,
    assistant_text: str,
) -> Optional[RelationshipUpdate]:
    """Detect an explicit, mutually acknowledged relationship change.

    The relationship is PERSISTENT but MUTABLE: any tier may follow any other
    (stranger <-> familiar <-> close <-> dating <-> married), in either
    direction. What it is not is volatile -- the state never moves because a new
    session, a reconnect, a restart or an empty context happened, and it never
    moves on a passing mention of the past. Only an explicit present-tense
    request from the user, confirmed by Mili, changes anything.
    """
    user = " ".join((user_text or "").split())
    assistant = " ".join((assistant_text or "").split())
    # Live2D expression markers are transport hints, not relationship language.
    assistant = re.sub(r"^\s*(?:\[[\w-]+\]\s*)+", "", assistant)
    if not user or not assistant:
        return None

    acceptance_prefix = assistant[:_DATING_ACCEPTANCE_PREFIX_CHARS]

    # A mention of the past, or a question about it, is never a change. This
    # guards BOTH the explicit-assertion path below and the legacy detectors at
    # the bottom, which otherwise match a bare "pacaran" in a sentence like
    # "aku ingat waktu kita pacaran dulu".
    non_assertive = bool(_NARRATIVE_FRAME.search(user))

    # 1. An explicit present-tense assertion, in EITHER direction.
    target = None if non_assertive else _relationship_assertion_target(user)
    if target is not None and target != current_status:
        if _relationship_acknowledges(target, assistant):
            return RelationshipUpdate(
                target, f"explicit_relationship_change_to_{target}"
            )

    # 2. Legacy tiers keep their original, narrower detectors so no existing
    #    behaviour changes: a fresh "kita pacaran" from a stranger is a step up,
    #    not a downshift.
    if not non_assertive and current_status not in {"dating", "married"}:
        if (
            _DATING_PROPOSAL.search(user)
            and _DATING_ACCEPTANCE.search(acceptance_prefix)
            and not _ROMANTIC_REJECTION.search(assistant)
        ):
            return RelationshipUpdate("dating", "explicit_relationship_event")

    if not non_assertive and current_status in {"stranger", "familiar"} and (
        _CLOSE_USER_EVENT.search(user)
        and _CLOSE_ACCEPTANCE.search(assistant)
        and not _ROMANTIC_REJECTION.search(assistant)
    ):
        return RelationshipUpdate("close", "mutual_trust_event")

    if not non_assertive and current_status == "stranger" and (
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
