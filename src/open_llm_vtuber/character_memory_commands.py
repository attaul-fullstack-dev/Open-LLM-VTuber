"""Deterministic parsing for explicit character-memory commands.

This module intentionally recognizes only commands near the start of a message.
It never classifies ordinary conversation and never calls an external model.
"""

from dataclasses import dataclass
from typing import Literal, Optional
import re
import unicodedata


MemoryCommandAction = Literal["remember", "forget", "none"]


@dataclass(frozen=True)
class MemoryCommandResult:
    """A non-persistent parse result for one explicit memory command."""

    action: MemoryCommandAction
    payload: Optional[str] = None
    matched_trigger: Optional[str] = None


_NONE = MemoryCommandResult(action="none")
_PARTICLE = r"(?:yah|yak|yap|ya+|dong|deh)"
_SAFE_PREFIX = re.compile(
    r"^(?:(?:eh\s+iya|oh\s+iya|eh\s+btw|ngomong[-\s]ngomong|btw|eh|ok)"
    r"\s*[,.:;!?…\-–—]*\s+)",
    re.IGNORECASE,
)
_RECALL_QUESTION = re.compile(
    r"^(?:(?:kamu|lu|mili)\s+(?:masih\s+)?(?:ingat|inget)\b|"
    r"(?:masih\s+)?(?:ingat|inget)(?:\s+(?:gak|nggak|ga|ngga)\b|\s*[?？]))",
    re.IGNORECASE,
)
_CONNECTIVE = re.compile(
    r"^(?P<connector>kalo\s+misalnya|kalau|kalo|bahwa|that|about)\b",
    re.IGNORECASE,
)
_FORGET_CONNECTIVE = re.compile(
    r"^(?P<connector>yang\s+tadi\s+tentang|yang\s+tentang|"
    r"kalo\s+misalnya|kalau|kalo|bahwa|soal|tentang|that|about)\b",
    re.IGNORECASE,
)
_USELESS_PAYLOADS = {
    "baik baik",
    "deh",
    "dong",
    "ini",
    "itu",
    "satu hal",
    "ya",
    "yaa",
    "yaaa",
    "yah",
    "yak",
    "yap",
}
_TEMPORARY_REMINDER_START = re.compile(
    r"^(?:makan|tidur|minum|mandi|mat(?:iin|ikan)|balas|cek|periksa|"
    r"nyalakan|hidupkan|kerja|bangun|telepon|kirim)\b",
    re.IGNORECASE,
)
_FACT_PRONOUN = re.compile(r"\b(?:gw|gue|gua|aku|saya|ane|user)\b", re.IGNORECASE)
_FACT_MARKER = re.compile(
    r"\b(?:suka|nggak\s+suka|gak\s+suka|ga\s+suka|tidak\s+suka|"
    r"favorit|kesukaan|sedang|lagi\s+belajar|pengen|ingin|lebih\s+suka|"
    r"biasanya|nama|panggil)\b",
    re.IGNORECASE,
)


# Forget patterns are intentionally evaluated before remember patterns.
_FORGET_PATTERNS = (
    (
        "forget_jangan_ingat",
        re.compile(
            rf"^(?:udah\s+)?jangan\s+(?:ingat|inget)\b"
            rf"(?:\s+{_PARTICLE})?(?:\s+lagi)?",
            re.IGNORECASE,
        ),
    ),
    (
        "forget_hapus_ingatan",
        re.compile(
            r"^hapus\s+(?:dari\s+ingatan(?:\s+kamu|mu)?|memory\s+tentang)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "forget_lupakan",
        re.compile(r"^(?:lupakan|lupain)\b", re.IGNORECASE),
    ),
    (
        "forget_english",
        re.compile(r"^forget\b", re.IGNORECASE),
    ),
)

_REMEMBER_PATTERNS = (
    (
        "remember_future_prefix",
        re.compile(
            r"^(?:buat|untuk)\s+ke\s*depannya\s*[,.:;!?…\-–—]*\s*"
            r"(?:ingat|inget)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "remember_mulai_sekarang",
        re.compile(r"^mulai\s+sekarang\s+(?:ingat|inget)\b", re.IGNORECASE),
    ),
    (
        "remember_future",
        re.compile(
            r"^(?:ingat|inget)\s+(?:buat|untuk)\s+ke\s*depannya\b",
            re.IGNORECASE,
        ),
    ),
    (
        "remember_tolong_ingat",
        re.compile(
            rf"^tolong\s+(?:di)?(?:ingat|inget)\b"
            rf"(?:\s+(?:ini|baik-baik|satu\s+hal))?"
            rf"(?:\s+{_PARTICLE}){{0,2}}",
            re.IGNORECASE,
        ),
    ),
    (
        "remember_jangan_lupa",
        re.compile(
            rf"^(?:jangan|jgn)\s+(?:lupa|lupain)\b"
            rf"(?:\s+(?:satu\s+hal))?(?:\s+{_PARTICLE}){{0,2}}",
            re.IGNORECASE,
        ),
    ),
    (
        "remember_catat",
        re.compile(
            rf"^(?:catat|catet)\b"
            rf"(?:\s+(?:ini|di\s+ingatan(?:\s+kamu|mu)?))?"
            rf"(?:\s+{_PARTICLE}){{0,2}}",
            re.IGNORECASE,
        ),
    ),
    (
        "remember_simpan",
        re.compile(
            rf"^(?:simpan|simpen)\b\s+"
            rf"(?:ini|(?:di|ke)\s+ingatan(?:\s+kamu|mu)?|"
            rf"sebagai\s+ingatan|{_PARTICLE})(?:\s+{_PARTICLE})?",
            re.IGNORECASE,
        ),
    ),
    (
        "remember_masukkan",
        re.compile(
            r"^(?:masukkan|masukin)\b(?:\s+ini)?\s+ke\s+ingatan"
            r"(?:\s+kamu|mu)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "remember_ingat",
        re.compile(
            rf"^(?:ingat|inget)\b"
            rf"(?:\s+(?:ini|baik-baik|satu\s+hal))?"
            rf"(?:\s+{_PARTICLE}){{0,2}}",
            re.IGNORECASE,
        ),
    ),
    (
        "remember_english",
        re.compile(r"^(?:please\s+)?remember(?:\s+this)?\b", re.IGNORECASE),
    ),
)


def _collapse_spaces(text: str) -> str:
    return " ".join((text or "").strip().split())


def _strip_edge_noise(text: str) -> str:
    """Remove command separators/emoji at the edges, not inside the fact."""
    start = 0
    end = len(text)
    while start < end:
        char = text[start]
        if char.isspace() or unicodedata.category(char)[0] in {"P", "S"}:
            start += 1
            continue
        break
    while end > start:
        char = text[end - 1]
        if char.isspace() or unicodedata.category(char)[0] == "P":
            end -= 1
            continue
        break
    return text[start:end].strip()


def _extract_payload(
    text: str,
    command_end: int,
    *,
    forget: bool = False,
) -> tuple[str, bool]:
    payload = _strip_edge_noise(text[command_end:])
    connector = (_FORGET_CONNECTIVE if forget else _CONNECTIVE).match(payload)
    had_connector = connector is not None
    if connector:
        payload = _strip_edge_noise(payload[connector.end() :])
    return _collapse_spaces(payload), had_connector


def _meaningful_payload(payload: str) -> bool:
    if not payload or payload.endswith(("?", "？")):
        return False
    normalized = re.sub(r"[^\w]+", " ", payload.lower(), flags=re.UNICODE).strip()
    return len(normalized) >= 3 and normalized not in _USELESS_PAYLOADS


def _is_persistent_fact_reminder(payload: str, had_connector: bool) -> bool:
    """Keep ``jangan lupa`` conservative: facts yes, ordinary tasks no."""
    if _TEMPORARY_REMINDER_START.match(payload):
        return False
    if had_connector:
        return True
    return bool(_FACT_PRONOUN.search(payload) and _FACT_MARKER.search(payload))


def parse_memory_command(user_text: str) -> MemoryCommandResult:
    """Parse one explicit remember/forget instruction without side effects."""
    text = _collapse_spaces(user_text)
    if not text:
        return _NONE

    prefix = _SAFE_PREFIX.match(text)
    if prefix:
        text = text[prefix.end() :].lstrip()

    # Recall questions are conversation, not mutation commands.
    if _RECALL_QUESTION.match(text):
        return _NONE

    for trigger, pattern in _FORGET_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        payload, _ = _extract_payload(text, match.end(), forget=True)
        if trigger == "forget_jangan_ingat":
            payload = re.sub(r"\s+lagi$", "", payload, flags=re.IGNORECASE).strip()
        if _meaningful_payload(payload):
            return MemoryCommandResult("forget", payload, trigger)
        return _NONE

    for trigger, pattern in _REMEMBER_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        payload, had_connector = _extract_payload(text, match.end())
        if not _meaningful_payload(payload):
            return _NONE
        if trigger == "remember_jangan_lupa" and not _is_persistent_fact_reminder(
            payload, had_connector
        ):
            return _NONE
        return MemoryCommandResult("remember", payload, trigger)

    return _NONE
