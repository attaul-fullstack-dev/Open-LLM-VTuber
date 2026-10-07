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

# Persistence nouns: the "sebagai X" clause must frame X as stored
# information. Without this restriction a mid-sentence "simpan ... sebagai
# ..." (e.g. "aku simpan uang sebagai dana darurat") would be misread as a
# memory command. PERSIST-6042 fix: explicit save requests such as "simpan
# fakta unik berikut sebagai informasi permanen: kode ..." must persist.
_SEBAGAI_PERSIST_NOUN = (
    r"(?:informasi|ingatan|fakta|data|catatan|kenangan|pengingat|"
    r"arsip|memori|memory)\b"
)
_REMEMBER_SIMPAN_SEBAGAI = re.compile(
    r"(?:^|[\s,;:!?…\-–—]+)"
    r"(?:simpan|simpen|catat|catet)\b"
    r"(?P<mid>.{0,120}?)"
    r"\bsebagai\b\s+"
    rf"(?P<rest>{_SEBAGAI_PERSIST_NOUN}.+)",
    re.IGNORECASE,
)
# "Pastikan X tersimpan ..." asks Mili to guarantee persistence of X. When X
# is not stored yet, storing X is the honest fulfillment (dedup keeps a
# repeat harmless). A bare "pastikan ini tersimpan" carries no content and
# is rejected by _meaningful_payload below.
_REMEMBER_PASTIKAN_TERSIMPAN = re.compile(
    r"^\s*pastikan\b(?P<item>.{1,160}?)\btersimpan\b",
    re.IGNORECASE,
)
# Trailing adverbs that add no identity to a "pastikan" payload
# ("PERSIST-6042 benar-benar" -> "PERSIST-6042").
_PASTIKAN_TRAILING_FILLER = re.compile(
    r"\s+(?:benar-benar|betul-betul|sungguh-sungguh|sungguh|pasti|dengan\s+baik)\s*$",
    re.IGNORECASE,
)
# Triggers whose payload comes from a named regex group instead of the
# generic tail extraction (the command is not message-initial).
_GROUP_PAYLOAD_TRIGGERS = frozenset(
    {"remember_simpan_sebagai", "remember_pastikan_tersimpan"}
)
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
        "remember_simpan_sebagai",
        _REMEMBER_SIMPAN_SEBAGAI,
    ),
    (
        "remember_pastikan_tersimpan",
        _REMEMBER_PASTIKAN_TERSIMPAN,
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
        if trigger in _GROUP_PAYLOAD_TRIGGERS:
            # Mid-sentence triggers: .match() only tries position 0, so
            # .search() is required for non-initial commands.
            match = pattern.search(text)
        else:
            match = pattern.match(text)
        if not match:
            continue
        if trigger in _GROUP_PAYLOAD_TRIGGERS:
            payload = _group_payload(trigger, match)
        else:
            payload, had_connector = _extract_payload(text, match.end())
            if trigger == "remember_jangan_lupa" and not _is_persistent_fact_reminder(
                payload, had_connector
            ):
                return _NONE
            if not _meaningful_payload(payload):
                return _NONE
            return MemoryCommandResult("remember", payload, trigger)
        if not _meaningful_payload(payload):
            return _NONE
        return MemoryCommandResult("remember", payload, trigger)

    return _NONE


def _group_payload(trigger: str, match: "re.Match[str]") -> str:
    """Payload for mid-sentence triggers (command is not message-initial)."""
    if trigger == "remember_simpan_sebagai":
        mid = _collapse_spaces(match.group("mid") or "")
        rest = _collapse_spaces(match.group("rest") or "")
        combined = f"{mid} sebagai {rest}" if mid else f"sebagai {rest}"
        return _collapse_spaces(_strip_edge_noise(combined))
    if trigger == "remember_pastikan_tersimpan":
        item = _collapse_spaces(match.group("item") or "")
        item = _PASTIKAN_TRAILING_FILLER.sub("", item).strip()
        return _collapse_spaces(_strip_edge_noise(item))
    return ""


# ---------------------------------------------------------------------------
# Automatic stable-fact capture (no explicit command required)
# ---------------------------------------------------------------------------

# A STABLE fact about the user: something that keeps being true after the turn
# ends. Deliberately much narrower than ``_FACT_MARKER`` above, which also
# accepts temporary states ("sedang", "lagi") because an explicit "ingat this"
# is an unambiguous user instruction. Automatic capture has no such signal, so
# it must only fire on wording that is durable on its own.
_AUTO_FACT_MARKER = re.compile(
    r"\b(?:suka|nggak\s+suka|gak\s+suka|ga\s+suka|tidak\s+suka|"
    r"cinta|favorit|kesukaan|alergi|punya|pekerjaan|profesi|"
    r"tinggal\s+di|tinggal\s+sekarang|alamat|ulang\s+tahun|hobi|"
    r"nama\s+(?:gw|gue|gua|aku|ane|saya)|namaku|nama\s+aku|"
    r"biasa(?:nya)?|selalu|kebiasaan|selesai\s+belajar|"
    r"sedang\s+belajar|pindah\s+kerja)\b",
    re.IGNORECASE,
)

# Explicitly temporary. A fact that names one of these is an event, not a
# durable trait: it belongs to episodic memory (or to nothing at all).
_AUTO_FACT_TRANSIENT = re.compile(
    r"\b(?:hari\s+ini|malam\s+ini|sekarang|lagi\s+(?:sakit|bufuk|capek|"
    r"makan|main|kerja|belajar|tidur)|sedang\s+(?:sakit|bufuk|makan|"
    r"tidur|jalan|kerja)|tadi|padahal\s+kemarin|nanti|besok|"
    r"minggu\s+depan|tahun\s+depan|doang|sementara)\b",
    re.IGNORECASE,
)

# First-person subject, stripped from the stored fact so the memory reads as a
# statement about the user rather than a transcript quote.
_AUTO_FACT_SUBJECT = re.compile(
    r"^(?:aku|gw|gue|gua|ane|saya|user)\b[\s,]*",
    re.IGNORECASE,
)
# "nama gw ..." / "namaku ..." is a first-person statement too, but the pronoun
# sits inside it, so it needs its own subject form and a separate strip.
_AUTO_FACT_NAME_SUBJECT = re.compile(
    r"^(?:nama\s+(?:gw|gue|gua|aku|ane|saya)|namaku|nama\s+aku)\b[\s,]*",
    re.IGNORECASE,
)
# Discourse filler that carries no content once the subject is removed.
_AUTO_FACT_FILLER = re.compile(
    r"^(?:itu|ini|ya|sih|deh|dong|kok|ah|oh|hm|wah)\b[\s,]*",
    re.IGNORECASE,
)
# A pronoun repeated in a later clause ("nama gw Rizky, gw programmer di
# Jakarta") is dropped but the clause separator is kept.
_AUTO_FACT_CLAUSE_PRONOUN = re.compile(
    r"(^|[,;]\s+)(?:gw|gue|gua|aku|ane|saya)\b\s*",
    re.IGNORECASE,
)

# Questions, commands and reactions are conversation, never stored facts.
_AUTO_FACT_BLOCK = re.compile(
    r"(?:\?\s*$)|"
    r"^\s*(?:kamu|lu|anda|masi|memang|kenapa|mengapa|nggak\s+ya|tidak\s+ya|"
    r"kok|btw|oh|ah|hah|wkwk|hehe|lol)\b|"
    r"\b(?:apa|siapa|kapan|dimana|mengapa|kenapa|berapa|gimana|bisa\s+tolong|"
    r"tolong|bisa\s+bantu)\b",
    re.IGNORECASE,
)

_AUTO_FACT_MIN_LEN = 6
_AUTO_FACT_MAX_LEN = 160
_AUTO_FACT_MAX_PER_TURN = 2


def _split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?…])\s+|\n+", str(text or ""))
    return [p.strip() for p in parts if p and p.strip()]


def extract_stable_facts(user_text: str) -> tuple:
    """Pick durable facts out of one ordinary user turn. Pure, no I/O.

    This is what makes memory work without the user ever saying "ingat this".
    It is intentionally conservative and fully deterministic -- no model call,
    no scoring, no randomness:

    - a sentence must be first-person (``_AUTO_FACT_SUBJECT``) AND carry a
      durable-trait marker (``_AUTO_FACT_MARKER``);
    - anything naming a transient moment is rejected (``_AUTO_FACT_TRANSIENT``)
      because that is an event for episodic memory, not a lasting trait;
    - questions, commands and bare reactions are rejected outright;
    - the stored text is the sentence minus its first-person subject, so the
      prompt reads ``- [2 days ago | Oct 1] suka minum kopi susu``.

    Returns at most ``_AUTO_FACT_MAX_PER_TURN`` distinct fragments, in the
    order they were said. Duplicates against the existing store are handled by
    ``add_character_memory``, not here.
    """
    try:
        if not str(user_text or "").strip():
            return ()
        # An explicit command is the user's business, never auto-extracted.
        if parse_memory_command(user_text).action != "none":
            return ()
        out: list = []
        seen: set = set()
        for sentence in _split_sentences(user_text):
            if len(out) >= _AUTO_FACT_MAX_PER_TURN:
                break
            if _AUTO_FACT_BLOCK.search(sentence):
                continue
            if not (
                _AUTO_FACT_SUBJECT.match(sentence)
                or _AUTO_FACT_NAME_SUBJECT.match(sentence)
            ):
                continue
            if not _AUTO_FACT_MARKER.search(sentence):
                continue
            if _AUTO_FACT_TRANSIENT.search(sentence):
                continue
            fact = _collapse_spaces(_AUTO_FACT_SUBJECT.sub("", sentence))
            fact = _collapse_spaces(_AUTO_FACT_NAME_SUBJECT.sub("nama ", fact))
            fact = _collapse_spaces(_AUTO_FACT_CLAUSE_PRONOUN.sub(r"\1", fact))
            while True:
                stripped = _AUTO_FACT_FILLER.sub("", fact, count=1)
                if stripped == fact:
                    break
                fact = stripped
            fact = _collapse_spaces(fact).strip(" ,.;:!?\u2026-\u2013\u2014")
            if not (_AUTO_FACT_MIN_LEN <= len(fact) <= _AUTO_FACT_MAX_LEN):
                continue
            if not _meaningful_payload(fact):
                continue
            key = fact.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(fact)
        return tuple(out)
    except Exception:
        # Fail-soft: automatic capture must never break a conversation.
        return ()
