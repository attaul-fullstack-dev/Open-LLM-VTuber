"""Canonical time provider for Mili.

Provides an authoritative, consistent source of time across all Mili modules
(AI agents, memory, future intentions, reminder scheduler, proactive behavior,
and Life State / World State):

- utc_now(): Aware UTC datetime (second precision).
- utc_iso(): UTC ISO-8601 string (e.g. "2026-10-31T10:00:00+00:00").
- user_now(moment, tz): Local datetime in the user's IANA timezone.
- server_now(): Server-local datetime (for system diagnostics only).
- resolve_tz(tz): Resolves an IANA timezone name safely with UTC fallback.
- ensure_aware(moment): Guarantees tzinfo is present (defaults to UTC). Accepts datetime, str, or None.
- parse_iso(value): Safely parses ISO-8601 strings into aware datetime.
- format_temporal_anchor(moment, tz): Standardized anchor for prompts.
- relative_day_parts(added_at, moment, tz): Computes (age_label, date_str).
- local_hour(moment, tz): Computes user-local hour (0-23).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger

_TZ_CACHE: Dict[str, Optional[ZoneInfo]] = {}
_TZ_WARNED: set[str] = set()


def utc_now() -> datetime:
    """Authoritative real wall-clock in UTC (second precision)."""
    return datetime.now(timezone.utc)


# Backward-compatible alias
utcnow = utc_now


def parse_iso(value: Any, fallback: Optional[datetime] = None) -> Optional[datetime]:
    """Parse an ISO-8601 string to an aware datetime. Returns fallback if unparseable."""
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def ensure_aware(value: Any = None) -> datetime:
    """Guarantee a datetime has timezone info, defaulting to UTC. Accepts datetime, ISO string, or None."""
    if value is None:
        return utc_now()
    if isinstance(value, str):
        parsed = parse_iso(value, fallback=None)
        if parsed is not None:
            return parsed
        return utc_now()
    if hasattr(value, "tzinfo"):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return utc_now()


_ensure_aware = ensure_aware


def utc_iso(moment: Optional[datetime] = None) -> str:
    """Return an aware UTC instant formatted as ISO-8601."""
    return ensure_aware(moment).astimezone(timezone.utc).isoformat(timespec="seconds")


_to_iso = utc_iso


def _iso_or_none(value: Any) -> Optional[str]:
    parsed = parse_iso(value, fallback=None)
    return utc_iso(parsed) if parsed is not None else None


def resolve_tz(tz: Optional[str]) -> Optional[ZoneInfo]:
    """Return a ZoneInfo for a user/session timezone name, or None (UTC fallback).

    ``None`` (or blank/invalid) means 'no user timezone known' and callers
    fall back to server-side UTC, preserving pre-timezone behavior.
    Results are cached so an unknown name warns at most once.
    """
    name = str(tz or "").strip()
    if not name:
        return None
    if name in _TZ_CACHE:
        return _TZ_CACHE[name]
    try:
        zone: Optional[ZoneInfo] = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        zone = None
        if name not in _TZ_WARNED:
            _TZ_WARNED.add(name)
            logger.warning(
                "Unknown user timezone, falling back to UTC: tz={}", name[:64]
            )
    _TZ_CACHE[name] = zone
    return zone


def user_now(
    moment: Optional[Any] = None, tz: Optional[str] = None
) -> datetime:
    """Wall-clock instant converted to the user's timezone (or UTC if unknown).

    Supports signature:
      user_now(moment, tz)
      user_now(tz='Asia/Jakarta')
    """
    if isinstance(moment, str) and tz is None and ("/" in moment or moment in ("UTC", "GMT")):
        tz = moment
        moment = None
    aware = ensure_aware(moment)
    zone = resolve_tz(tz)
    return aware.astimezone(zone) if zone is not None else aware.astimezone(timezone.utc)


# Backward-compatible alias
user_local_datetime = user_now


def server_now() -> datetime:
    """Wall-clock instant in the VPS/server local timezone (diagnostics only)."""
    return datetime.now().astimezone()


def local_hour(moment: Optional[datetime] = None, tz: Optional[str] = None) -> int:
    """Wall-clock hour (0-23) in the user timezone."""
    return user_now(moment, tz).hour


def format_temporal_anchor(
    moment: Optional[datetime] = None, tz: Optional[str] = None
) -> str:
    """Compact 'today + now' anchor for the system prompt.

    Gives the LLM a reliable current-date/weekday/clock-time/timezone
    reference, rebuilt at every request from the runtime system clock, so
    'what time is it' is answered from real time, never guessed or stale.
    """
    local = user_now(moment, tz)
    date_str = f"{local:%B} {local.day}, {local.year}"
    weekday = f"{local:%A}"
    clock_str = f"{local:%H:%M}"
    zone = resolve_tz(tz)
    tz_label = tz if zone is not None else "UTC"
    return (
        f"Current date: {date_str} ({weekday})\n"
        f"Current time: {clock_str}\n"
        f"Timezone: {tz_label}"
    )


def relative_day_parts(
    added_at: Any,
    moment: Optional[datetime] = None,
    tz: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """(label, date) age of a stored ISO timestamp in the user timezone.

    Day boundaries use user-local calendar dates. Returns None when
    ``added_at`` is unparseable so callers render the original text as-is.
    Future timestamps (clock skew) read as today, never negative.
    """
    added = parse_iso(added_at, fallback=None)
    if added is None:
        return None
    local_now = user_now(moment, tz)
    zone = resolve_tz(tz)
    local_added = added.astimezone(zone) if zone is not None else added
    delta_days = (local_now.date() - local_added.date()).days
    if delta_days < 0:
        delta_days = 0
    if delta_days == 0:
        label = "Today"
    elif delta_days == 1:
        label = "Yesterday"
    else:
        label = f"{delta_days} days ago"
    date_str = f"{local_added:%b} {local_added.day}"
    return label, date_str


def memory_age_label(
    added_at: Any,
    moment: Optional[datetime] = None,
    tz: Optional[str] = None,
) -> str:
    """Render a compact age tag like '[Today | Sep 29]' or '' when unparseable."""
    parts = relative_day_parts(added_at, moment, tz)
    if not parts:
        return ""
    label, date_str = parts
    return f"[{label} | {date_str}]"


__all__ = [
    "ensure_aware",
    "format_temporal_anchor",
    "local_hour",
    "memory_age_label",
    "parse_iso",
    "relative_day_parts",
    "resolve_tz",
    "server_now",
    "user_local_datetime",
    "user_now",
    "utc_iso",
    "utc_now",
    "utcnow",
]
