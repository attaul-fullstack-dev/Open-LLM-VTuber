"""Episodic memory: selected meaningful experiences, separate from facts.

Layers (kept distinct by design):
- transcript: full conversation record (chat_history/*.json).
- conversation summary: compressed conversational context (metadata).
- episodic memory (this module): selected past experiences with absolute
  timestamps and session provenance (episodic/<conf>.json).
- long-term memory: stable facts/preferences (character_state.memories).
- relationship: relational state. life/world: current simulated state.

An episodic event records that something HAPPENED, never an inference
about personality, preference, or permanent fact. All persistence here
is fail-soft: storage/parse failures return safe defaults and never
raise into the conversation pipeline.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .chat_history_manager import _sanitize_path_component
from .world_state import (
    memory_age_label,
    resolve_tz,
    user_local_datetime,
    utcnow,
)

EPISODIC_DIR = "episodic"
EPISODIC_MAX_EVENTS = 500
EPISODIC_TOP_N = 4
EPISODIC_MAX_TOKENS = 400
EPISODIC_CONFIDENCE_THRESHOLD = 0.6
EPISODIC_MIN_TEXT_CHARS = 30

EPISODIC_EXTRACTION_SYSTEM = """You extract at most ONE past experience from a single user chat turn.
Return JSON only, exactly one of:
{"event_text": "<one short factual sentence in the user's language>", "occurred_at": "<ISO-8601 UTC timestamp or null>", "confidence": 0.0}
{"event": null}
Rules: only real experiences that already happened (did, spent time on, finished, visited, fixed, learned from doing). Small talk, greetings, jokes, pure questions, and future plans/intentions (will do, want to buy, tomorrow I will) are NOT events: return {"event": null}. Never invent facts, people, or timestamps. If the time cannot be determined from the turn plus the given request time, use null for occurred_at. Never describe personality, preferences, or permanent traits. Confidence below 0.6 discards the event."""

# Past-experience markers (ID + EN). A turn must carry at least one to
# become an extraction candidate.
_PAST_MARKERS = (
    "tadi",
    "kemarin",
    "semalam",
    "hari ini",
    "minggu lalu",
    "bulan lalu",
    "selama",
    "sudah",
    "selesai",
    "habis",
    "baru saja",
    "pernah",
    "telah",
    "akhirnya",
    "kemaren",
    # Whole-word past-completion verb (ID morphology): "menghabiskan 3 jam
    # memperbaiki ..." states a completed expenditure of time. Listed as a
    # full word so the \b gate below matches only the word itself.
    "menghabiskan",
    # Colloquial "spent/used up" ("habisin 3 jam ..."): standalone word,
    # same completion semantics as "menghabiskan".
    "habisin",
    "yesterday",
    "today",
    "earlier",
    "just finished",
    "last week",
    "for hours",
    "for hour",
    "spent",
    "fixed",
    "finished",
    "completed",
    "days ago",
    "day ago",
)


# Whole-word compiled forms of the marker tuples below. The candidate gate
# must never match a marker as a mere substring of another word
# (PERSIST-6042 fix): "akan" inside "digunakan"/"makan", "telah" inside
# "setelah", "mau" inside "semau", etc. Trailing spaces in the tuples are a
# legacy word-boundary hack and are stripped here; \b does the job properly.
def _word_pattern(marker: str) -> "re.Pattern[str]":
    return re.compile(r"\b" + re.escape(marker.strip()) + r"\b")


def _has_any_word(patterns: tuple, lowered: str) -> bool:
    return any(rx.search(lowered) is not None for rx in patterns)


# Deterministic "N days ago" references (ID + EN). Resolved arithmetically
# from the request date; see resolve_occurred_at.
_N_DAYS_AGO = re.compile(r"(\d{1,3})\s*hari\s*(yang\s+)?lalu|(\d{1,3})\s*days?\s+ago")
# Relative-day expressions: day-precision only unless a clock time is stated.
_YESTERDAY_RE = re.compile(r"\b(kemarin|kemaren|semalam|yesterday)\b")
_TODAY_RE = re.compile(
    r"\b(hari ini|tadi|baru saja|barusan|today|earlier today|"
    r"this morning|this afternoon|this evening|just now)\b"
)
_CLOCK_MENTION_RE = re.compile(
    r"\b(?:jam|pukul|at)\s*\d{1,2}(?:[:.]\d{2})?\b"
    r"|\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b"
    r"|\b\d{1,2}\s*(?:am|pm|pagi|siang|sore|malam|night|morning|evening)\b"
)
_FUTURE_MARKERS = (
    "besok",
    "nanti",
    "akan ",
    " mau ",
    "mau beli",
    "rencana",
    "rencananya",
    "minggu depan",
    "bulan depan",
    "tomorrow",
    "will ",
    "plan to",
    "planning to",
    "going to",
    "want to buy",
)

# Whole-word marker matchers, compiled after the tuples above.
_PAST_WORD_RES = tuple(_word_pattern(marker) for marker in _PAST_MARKERS)
_FUTURE_WORD_RES = tuple(_word_pattern(marker) for marker in _FUTURE_MARKERS)
_SMALLTALK = (
    "wkwk",
    "haha",
    "hehe",
    "iya",
    "oke",
    "ok ",
    "sip",
    "hmm",
    "oh ",
    "wah",
    "lol",
    "lmao",
)

_ID_MONTHS = {
    "januari": 1,
    "jan": 1,
    "februari": 2,
    "feb": 2,
    "maret": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "mei": 5,
    "juni": 6,
    "jun": 6,
    "juli": 7,
    "jul": 7,
    "agustus": 8,
    "agu": 8,
    "ags": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "oktober": 10,
    "okt": 10,
    "november": 11,
    "nov": 11,
    "desember": 12,
    "des": 12,
}
_EN_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_STOPWORDS = frozenset(
    "yang dan di ke dari untuk dengan pada adalah itu ini aku gw gue saya "
    "kamu kau anda dia mereka kita kami ko kok sih dong deh aja lah mah pun "
    "nggak tidak tak bukan sudah telah lagi masih juga atau tapi karena kalau "
    "kalo yang the a an and or of to in on for with is are was were it this "
    "that i you he she we they my your his her our their me him us them what "
    "yang gw saya".split()
)

_episodic_locks: Dict[str, threading.RLock] = {}
_episodic_locks_guard = threading.Lock()


def _get_episodic_lock(filepath: str) -> threading.RLock:
    with _episodic_locks_guard:
        return _episodic_locks.setdefault(filepath, threading.RLock())


def get_episodic_path(conf_uid: str) -> str:
    """Return the on-disk path for a character's episodic store."""
    if not conf_uid:
        raise ValueError("conf_uid cannot be empty")
    safe_conf_uid = _sanitize_path_component(conf_uid)
    return os.path.join(EPISODIC_DIR, f"{safe_conf_uid}.json")


def _write_episodic_atomic(filepath: str, events: list) -> None:
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary_path = f"{filepath}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(events, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, filepath)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _valid_event(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    text = " ".join(str(item.get("event_text", "")).split()).strip()
    if not text:
        return None
    occurred = item.get("occurred_at")
    if occurred is not None:
        # Invalid stored timestamps degrade to null (render falls back to
        # created_at); never defaulted to current time.
        parsed = _parse_iso_or_none(occurred)
        occurred = parsed.isoformat(timespec="seconds") if parsed is not None else None
    return {
        "id": str(item.get("id", "") or uuid.uuid4().hex),
        "event_text": text,
        "occurred_at": occurred,
        "session_uid": str(item.get("session_uid", "") or ""),
        "source": str(item.get("source", "") or "conversation"),
        "created_at": str(item.get("created_at", "") or ""),
        "tz": item.get("tz"),
    }


def load_episodic_events(conf_uid: str) -> List[Dict[str, Any]]:
    """Load episodic events; missing/corrupt files yield an empty list."""
    try:
        filepath = get_episodic_path(conf_uid)
    except ValueError:
        return []
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as error:
        logger.error(
            "Failed to load episodic events: error_type={}",
            type(error).__name__,
        )
        return []
    if not isinstance(data, list):
        return []
    events = []
    for item in data:
        parsed = _valid_event(item)
        if parsed is not None:
            events.append(parsed)
    return events


def save_episodic_events(conf_uid: str, events: List[Dict[str, Any]]) -> bool:
    """Atomically persist episodic events; returns success."""
    try:
        filepath = get_episodic_path(conf_uid)
    except ValueError:
        return False
    try:
        lock = _get_episodic_lock(filepath)
        with lock:
            _write_episodic_atomic(filepath, list(events))
        return True
    except Exception as error:
        logger.error(
            "Failed to save episodic events: error_type={}",
            type(error).__name__,
        )
        return False


def _normalize_event_text(text: Any) -> str:
    return " ".join(str(text or "").lower().split()).strip(" .,!?;:，。！？；：")


def _token_set(text: str) -> set:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS and len(token) > 2
    }


def is_episodic_candidate(text: Any) -> bool:
    """Cheap local gate: is this turn worth one LLM extraction call (pure)?

    Conservative by design: small talk, short text, pure questions, and
    future intents never become candidates.
    """
    cleaned = " ".join(str(text or "").split()).strip()
    if len(cleaned) < EPISODIC_MIN_TEXT_CHARS:
        return False
    lowered = cleaned.lower()
    # Whole-word matching only (PERSIST-6042 fix): plain `in` would fire on
    # "akan" inside "digunakan" or "telah" inside "setelah".
    if not (_has_any_word(_PAST_WORD_RES, lowered) or _N_DAYS_AGO.search(lowered)):
        return False
    if _has_any_word(_FUTURE_WORD_RES, lowered):
        has_past = bool(
            _has_any_word(_PAST_WORD_RES, lowered) or _N_DAYS_AGO.search(lowered)
        )
        # Future intent dominates unless a completed past event is explicit.
        if (
            not re.search(
                r"\b(sudah|telah|selesai|habis|finished|completed|fixed)\b", lowered
            )
            or not has_past
        ):
            return False
        # Both past completion and future intent: still ambiguous, skip.
        if re.search(
            r"\b(besok|nanti|mau beli|rencana|tomorrow|will|plan to|going to)\b",
            lowered,
        ):
            return False
    if lowered.rstrip().endswith("?") and not (
        re.search(
            r"\b(tadi|kemarin|sudah|selesai|selama|yesterday|finished|spent)\b",
            lowered,
        )
        or _N_DAYS_AGO.search(lowered)
    ):
        return False
    alpha_ratio = sum(char.isalpha() for char in cleaned) / max(len(cleaned), 1)
    if alpha_ratio < 0.5:
        return False
    return True


def _local_day_start_utc(
    request_dt: datetime, tz: Optional[str], day_offset: int
) -> str:
    zone = resolve_tz(tz)
    if request_dt.tzinfo is None:
        aware = request_dt.replace(tzinfo=timezone.utc)
    else:
        aware = request_dt
    local = aware.astimezone(zone) if zone is not None else aware
    target = (local - timedelta(days=day_offset)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    base = target.astimezone(timezone.utc) if zone is not None else target
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return base.isoformat(timespec="seconds")


def _resolve_absolute_date(
    text: str, request_dt: datetime, tz: Optional[str]
) -> Optional[str]:
    """Resolve an explicit calendar date to UTC ISO (pure, else None)."""
    lowered = text.lower()
    # ISO-like: 2026-10-03 or 03/10(/2026) or 03-10.
    match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", lowered)
    day = month = year = None
    if match:
        year, month, day = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
    else:
        match = re.search(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", lowered)
        if match:
            day, month = int(match.group(1)), int(match.group(2))
            if match.group(3):
                year = int(match.group(3))
                if year < 100:
                    year += 2000
    if day is None:
        for name, number in {**_ID_MONTHS, **_EN_MONTHS}.items():
            hit = re.search(
                r"(\d{1,2})\s+" + re.escape(name) + r"(?:\s+(\d{4}))?",
                lowered,
            )
            if hit:
                day, month = int(hit.group(1)), number
                if hit.group(2):
                    year = int(hit.group(2))
                break
    if day is None or month is None:
        return None
    try:
        zone = resolve_tz(tz)
        if zone is None:
            if year is None:
                now_utc = request_dt
                if now_utc.tzinfo is None:
                    now_utc = now_utc.replace(tzinfo=timezone.utc)
                year = now_utc.year
            return datetime(year, month, day, tzinfo=timezone.utc).isoformat(
                timespec="seconds"
            )
        local_now = user_local_datetime(request_dt, tz)
        if year is None:
            year = local_now.year
            candidate = local_now.replace(
                year=year,
                month=month,
                day=day,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            if candidate.date() > local_now.date():
                candidate = candidate.replace(year=year - 1)
        else:
            candidate = local_now.replace(
                year=year,
                month=month,
                day=day,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        return candidate.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (ValueError, OverflowError):
        return None


def _resolve_n_days_ago(
    text: str, request_dt: datetime, tz: Optional[str]
) -> Optional[str]:
    """Resolve "N hari lalu / N days ago" arithmetically (pure, else None)."""
    match = _N_DAYS_AGO.search(text.lower())
    if not match:
        return None
    raw = match.group(1) or match.group(3)
    try:
        days = int(raw)
    except (TypeError, ValueError):
        return None
    if days < 1:
        return None
    return _local_day_start_utc(request_dt, tz, days)


def _has_explicit_clock_time(lowered: str) -> bool:
    """True when the user states a clock time for the event itself.

    A clock mentioned only as a range end ("sampai jam 2 pagi") is incidental
    and does not make a relative-day expression time-precise.
    """
    for match in _CLOCK_MENTION_RE.finditer(lowered):
        prefix = lowered[max(0, match.start() - 16) : match.start()]
        if re.search(r"\b(sampai|sampe|hingga|til|until)\s*$", prefix):
            continue
        return True
    return False


def resolve_relative_day_override(
    text: Any, request_dt: datetime, tz: Optional[str] = None
) -> Optional[str]:
    """Deterministic day-precision anchor for relative-day expressions.

    Returns local-day-start UTC ISO when the turn expresses a relative day
    ("N hari lalu" / "N days ago" / "kemarin" / "semalam" / "hari ini" /
    "tadi") and states no clock time for the event. Returns None otherwise,
    so an explicit absolute date or a user-stated clock time keeps priority
    and the model never becomes the source of truth for the day.
    """
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if _resolve_absolute_date(cleaned, request_dt, tz) is not None:
        return None
    if not (
        _N_DAYS_AGO.search(lowered)
        or _YESTERDAY_RE.search(lowered)
        or _TODAY_RE.search(lowered)
    ):
        return None
    if _has_explicit_clock_time(lowered):
        return None
    if _N_DAYS_AGO.search(lowered):
        return _resolve_n_days_ago(cleaned, request_dt, tz)
    if _YESTERDAY_RE.search(lowered):
        # "kemarin" and "semalam" both anchor to the previous local day.
        return _local_day_start_utc(request_dt, tz, 1)
    return _local_day_start_utc(request_dt, tz, 0)


def resolve_occurred_at(
    text: Any, request_dt: datetime, tz: Optional[str] = None
) -> Optional[str]:
    """Resolve when the event happened to absolute UTC ISO (pure).

    Precedence (most specific first): explicit calendar date, then
    "N days ago", then "yesterday/last night", then "today/just now". Vague
    references ("minggu lalu", weekday names without date) and anything
    unresolvable yield None — never an invented timestamp. Day precision
    is local-midnight converted to UTC.
    """
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return None
    absolute = _resolve_absolute_date(cleaned, request_dt, tz)
    if absolute is not None:
        return absolute
    lowered = cleaned.lower()
    n_days = _resolve_n_days_ago(cleaned, request_dt, tz)
    if n_days is not None:
        return n_days
    if _YESTERDAY_RE.search(lowered):
        return _local_day_start_utc(request_dt, tz, 1)
    if _TODAY_RE.search(lowered):
        return _local_day_start_utc(request_dt, tz, 0)
    return None


def build_extraction_prompt(
    user_text: str, request_dt: datetime, tz: Optional[str] = None
) -> Tuple[str, str]:
    """Build (system, user) extraction prompt (pure, no LLM call)."""
    moment = request_dt
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    context = (
        f"Request time (UTC): {moment.isoformat(timespec='seconds')}. "
        f"User timezone: {tz or 'unknown (assume UTC)'}."
    )
    return EPISODIC_EXTRACTION_SYSTEM, f"{context}\nUser turn: {user_text.strip()}"


def _parse_extraction_result_detail(
    raw: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[float]]:
    """Parse LLM extraction JSON defensively, with a stable failure reason.

    Returns ``(event, None, confidence)`` on success, otherwise
    ``(None, reason, confidence)`` where ``reason`` is one of
    ``empty_response``, ``invalid_json``, ``invalid_schema``, ``event_null``,
    ``empty_event_text``, or ``confidence_below_threshold``. Callers use the
    reason to log why a gate-passing candidate produced no event.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty_response", None
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if not match:
        return None, "invalid_json", None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, "invalid_json", None
    if not isinstance(data, dict):
        return None, "invalid_schema", None
    if data.get("event") is None and "event_text" not in data:
        return None, "event_null", None
    text = " ".join(str(data.get("event_text", "") or "").split()).strip()
    if not text:
        return None, "empty_event_text", None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None, "invalid_schema", None
    if not 0.0 <= confidence <= 1.0:
        return None, "invalid_schema", confidence
    if confidence < EPISODIC_CONFIDENCE_THRESHOLD:
        return None, "confidence_below_threshold", confidence
    occurred = data.get("occurred_at")
    if occurred is not None and not isinstance(occurred, str):
        return None, "invalid_schema", confidence
    return (
        {"event_text": text, "occurred_at": occurred, "confidence": confidence},
        None,
        confidence,
    )


def parse_extraction_result(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse LLM extraction JSON defensively (pure, None on any doubt)."""
    return _parse_extraction_result_detail(raw)[0]


def _parse_iso_or_none(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def is_duplicate_event(events: List[Dict[str, Any]], event_text: str) -> bool:
    """Normalized exact + near-duplicate detection (pure, no LLM)."""
    normalized = _normalize_event_text(event_text)
    if not normalized:
        return True
    new_tokens = _token_set(normalized)
    for item in events or []:
        existing = _normalize_event_text(item.get("event_text", ""))
        if not existing:
            continue
        if existing == normalized:
            return True
        old_tokens = _token_set(existing)
        if not old_tokens or not new_tokens:
            continue
        union = old_tokens | new_tokens
        if not union:
            continue
        if len(old_tokens & new_tokens) / len(union) >= 0.85:
            return True
    return False


def append_episodic_event(
    conf_uid: str, event: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Validate, dedupe, append with created_at, enforce soft cap."""
    text = " ".join(str(event.get("event_text", "") or "").split()).strip()
    if not text:
        return None
    occurred = event.get("occurred_at")
    if occurred is not None:
        # Invalid timestamps reject the event; never fall back to now.
        parsed_occurred = _parse_iso_or_none(occurred)
        if parsed_occurred is None:
            return None
        occurred = parsed_occurred.isoformat(timespec="seconds")
    events = load_episodic_events(conf_uid)
    if is_duplicate_event(events, text):
        return None
    now = utcnow().isoformat(timespec="seconds")
    stored = {
        "id": str(event.get("id", "") or uuid.uuid4().hex),
        "event_text": text,
        "occurred_at": event.get("occurred_at"),
        "session_uid": str(event.get("session_uid", "") or ""),
        "source": str(event.get("source", "") or "conversation"),
        "created_at": now,
        "tz": event.get("tz"),
    }
    events.append(stored)
    if len(events) > EPISODIC_MAX_EVENTS:
        events.sort(key=lambda item: str(item.get("created_at", "")))
        events = events[-EPISODIC_MAX_EVENTS:]
    if not save_episodic_events(conf_uid, events):
        return None
    return stored


def _score_event(query_tokens: set, event: Dict[str, Any], now: datetime) -> float:
    event_tokens = _token_set(_normalize_event_text(event.get("event_text", "")))
    if not query_tokens or not event_tokens:
        return 0.0
    overlap = len(query_tokens & event_tokens)
    if overlap == 0:
        return 0.0
    score = float(overlap)
    # Recency ranks by event time, not storage time: an old experience saved
    # yesterday is still old. created_at is only a fallback for rows without
    # an event timestamp. Recency boosts, never replaces, relevance.
    reference = _parse_iso_or_none(event.get("occurred_at", "")) or _parse_iso_or_none(
        event.get("created_at", "")
    )
    if reference is not None:
        try:
            age_days = (now - reference).total_seconds() / 86400.0
        except TypeError:
            age_days = None
        if age_days is not None and age_days >= 0:
            score += max(0.0, 0.5 - age_days / 14.0)
    return score


def retrieve_episodic_events(
    events: List[Dict[str, Any]],
    query: Any,
    *,
    now: Optional[datetime] = None,
    top_n: int = EPISODIC_TOP_N,
) -> List[Dict[str, Any]]:
    """Selective keyword + recency retrieval (pure, no LLM, no I/O)."""
    cleaned = " ".join(str(query or "").split()).strip()
    if not cleaned or not events:
        return []
    try:
        moment = now if now is not None else utcnow()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        query_tokens = _token_set(cleaned)
        scored = []
        for event in events:
            try:
                score = _score_event(query_tokens, event, moment)
            except Exception:
                continue
            if score > 0:
                # Tie-break on EVENT time, never storage time: two equally
                # relevant events must rank by when they happened.
                reference = _parse_iso_or_none(event.get("occurred_at", ""))
                if reference is None:
                    reference = _parse_iso_or_none(event.get("created_at", ""))
                scored.append(
                    (
                        score,
                        reference.isoformat(timespec="seconds") if reference else "",
                        # Final tie-break on a unique key so the result never
                        # depends on the order rows happened to be stored in.
                        str(event.get("id", "") or ""),
                        event,
                    )
                )
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [row[3] for row in scored[: max(top_n, 0)]]
    except Exception as error:
        logger.warning(
            "Episodic retrieval failed (no events returned): type={}",
            type(error).__name__,
        )
        return []


def render_episodic_context(
    events: List[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    tz: Optional[str] = None,
    max_tokens: int = EPISODIC_MAX_TOKENS,
) -> str:
    """Render retrieved events with render-time age tags (pure)."""
    if not events:
        return ""
    from .agent.context_window import estimate_tokens

    lines: List[str] = []
    used_tokens = 0
    for event in events:
        text = " ".join(str(event.get("event_text", "")).split()).strip()
        if not text:
            continue
        stamp = event.get("occurred_at") or event.get("created_at", "")
        age = memory_age_label(stamp, now, tz)
        line = f"- {age} {text}" if age else f"- {text}"
        line_tokens = estimate_tokens(line) + 4
        if used_tokens + line_tokens > max_tokens:
            break
        lines.append(line)
        used_tokens += line_tokens
    if not lines:
        return ""
    header = "Relevant episodic experiences (past events, not current facts):"
    return f"{header}\n" + "\n".join(lines)


def _log_extraction_rejection(
    reason: str,
    *,
    confidence: Optional[float] = None,
    event_chars: Optional[int] = None,
    session_uid: Optional[str] = None,
    error_type: Optional[str] = None,
) -> None:
    """Warn-level trace for a gate-passing candidate that produced no event.

    Metadata only: never logs user text, prompts, or credentials. Logging is
    side-effect free, so rejection stays fail-soft for the conversation.
    """
    logger.warning(
        "Episodic extraction rejected: reason={} confidence={} event_chars={} "
        "session={} error_type={}",
        reason,
        "none" if confidence is None else round(float(confidence), 2),
        "none" if event_chars is None else event_chars,
        session_uid or "none",
        error_type or "none",
    )


async def extract_and_store_episodic(
    llm_chat_fn: Any,
    conf_uid: str,
    user_text: str,
    history_uid: str,
    request_dt: datetime,
    tz: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Gate, extract via one LLM call, validate, and store (fail-soft).

    ``llm_chat_fn`` mirrors the summarizer call shape:
    ``await llm_chat_fn(messages, system)`` yielding str/dict chunks.
    Returns the stored event or None. Never raises. Every rejection after the
    heuristic gate passes is logged with a reason code so a missing event is
    always explainable.
    """
    try:
        if not conf_uid or not is_episodic_candidate(user_text):
            return None
        system, prompt = build_extraction_prompt(user_text, request_dt, tz)
        chunks: List[str] = []
        try:
            stream = llm_chat_fn([{"role": "user", "content": prompt}], system)
            async for event in stream:
                if isinstance(event, str):
                    chunks.append(event)
                elif isinstance(event, dict) and event.get("type") == "text_delta":
                    chunks.append(str(event.get("text", "")))
        except Exception as error:
            _log_extraction_rejection(
                "llm_error", session_uid=history_uid, error_type=type(error).__name__
            )
            return None
        parsed, reason, confidence = _parse_extraction_result_detail("".join(chunks))
        if parsed is None:
            _log_extraction_rejection(
                reason or "invalid_json",
                confidence=confidence,
                session_uid=history_uid,
            )
            return None
        occurred = parsed.get("occurred_at")
        event_chars = len(parsed["event_text"])
        # A relative-day expression with no stated clock time is day-precision:
        # the deterministic resolver owns occurred_at and the model's value is
        # ignored entirely, even when it is malformed.
        override = resolve_relative_day_override(user_text, request_dt, tz)
        if override is not None:
            occurred = override
        else:
            if occurred is not None:
                valid = _parse_iso_or_none(occurred)
                if valid is None:
                    _log_extraction_rejection(
                        "invalid_event",
                        confidence=confidence,
                        event_chars=event_chars,
                        session_uid=history_uid,
                    )
                    return None
                skew = (
                    valid - request_dt.replace(tzinfo=timezone.utc)
                    if request_dt.tzinfo is None
                    else valid - request_dt
                )
                if skew.total_seconds() > 3600:
                    occurred = None
            if occurred is None:
                occurred = resolve_occurred_at(user_text, request_dt, tz)
        stored = append_episodic_event(
            conf_uid,
            {
                "event_text": parsed["event_text"],
                "occurred_at": occurred,
                "session_uid": history_uid,
                "source": "conversation",
                "tz": tz,
            },
        )
        if stored is None:
            _log_extraction_rejection(
                "storage_rejected",
                confidence=confidence,
                event_chars=event_chars,
                session_uid=history_uid,
            )
        return stored
    except Exception as error:
        _log_extraction_rejection(
            "unexpected_error", session_uid=history_uid, error_type=type(error).__name__
        )
        return None
