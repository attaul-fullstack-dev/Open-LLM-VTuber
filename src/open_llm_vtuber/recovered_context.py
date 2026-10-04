"""Recovered conversation context — read-only auxiliary source.

A deleted conversation was partially recovered by a forensic pass: 142 of its
1282 messages plus the complete 1128-character rolling ``conversation_summary``
(``summary_through_message_index = 1222``). That data lives outside
``chat_history/`` on purpose and is never written back as a normal session.

This module exposes it in exactly the shape the existing previous-session
loader already produces (``{"at": ..., "text": ...}``), so the current
renderer, age tags, truncation and prompt budget apply unchanged. It adds no
store, no lifecycle and no prompt section of its own.

Honesty rules baked in:

- it is labelled as RECOVERED and PARTIAL, so the model cannot believe it still
  holds the whole conversation;
- missing messages are never invented or interpolated;
- everything is bounded and fail-soft, and a missing/corrupt artifact simply
  contributes nothing.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

RECOVERED_DIR = "recovered_context"

_SUMMARY_FILE = "conversation_summary.STEP1.txt"
_RECORDS_FILE = "recovered_records.json"
_REPORT_FILE = "recovery_report.txt"

# Metadata of the forensic pass. Kept explicit rather than inferred so the
# prompt can state the real numbers instead of implying completeness.
RECOVERED_SESSION_UID = "2026-09-30_04-05-06_4602317181514907a3e4476aaf93e7d4"
RECOVERED_COUNT = 142
ORIGINAL_COUNT = 1282
RECOVERED_STATUS = "partial"
RECOVERED_SOURCE = "forensic_recovery"
RECOVERED_AT = "2026-10-04"

# Unlike live rolling summaries, this forensic remainder cannot be regenerated
# from the deleted transcript. Preserve the whole compact summary instead of
# reapplying the ordinary 600-character prefix cut. The later part of this
# record can contain the only surviving account of mutually established facts
# from that conversation; truncating it would silently change behavioral
# continuity without adding any safety. Quotes remain separately bounded.
MAX_SUMMARY_CHARS = 1200
MAX_RECOVERED_QUOTES = 4
MAX_QUOTE_CHARS = 220


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception:
        return ""


def is_available(base_dir: str = RECOVERED_DIR) -> bool:
    return bool(os.path.isfile(os.path.join(base_dir, _SUMMARY_FILE)))


def recovered_metadata(base_dir: str = RECOVERED_DIR) -> Dict[str, Any]:
    """Provenance of the recovery pass (pure, fail-soft)."""
    return {
        "original_session_uid": RECOVERED_SESSION_UID,
        "recovered_at": RECOVERED_AT,
        "recovery_status": RECOVERED_STATUS,
        "recovered_count": RECOVERED_COUNT,
        "original_count": ORIGINAL_COUNT,
        "source": RECOVERED_SOURCE,
    }


def _quotes(base_dir: str) -> List[str]:
    """A few verbatim recovered exchanges, for grounding only.

    Bounded and non-fabricated: every line comes straight from the recovered
    artifact. Quotation is what makes it safe to show -- it is never phrased as
    something Mili said in the current session.
    """
    raw = _read(os.path.join(base_dir, _RECORDS_FILE))
    if not raw.strip():
        return []
    try:
        records = json.loads(raw)
    except Exception:
        return []
    if not isinstance(records, list):
        return []
    out: List[str] = []
    pending_user = ""
    for item in records:
        try:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", ""))
            text = " ".join(str(item.get("content", "")).split())
            if not text:
                continue
            if role == "human":
                pending_user = text[:MAX_QUOTE_CHARS]
                continue
            if role == "ai" and pending_user:
                out.append(f'"{pending_user}" -> "{text[:MAX_QUOTE_CHARS]}"')
                pending_user = ""
                if len(out) >= MAX_RECOVERED_QUOTES:
                    break
        except Exception:
            continue
    return out


def load_recovered_previous_session(
    base_dir: str = RECOVERED_DIR,
) -> Optional[Dict[str, str]]:
    """One item shaped like a normal previous-session summary, or ``None``.

    The text is the recovered rolling summary plus an explicit provenance
    header. The header is load-bearing: it is what stops the model from
    treating this as a complete transcript it can quote in full.
    """
    summary = _read(os.path.join(base_dir, _SUMMARY_FILE)).strip()
    if not summary:
        return None
    meta = recovered_metadata(base_dir)
    quotes = _quotes(base_dir)
    header = (
        f"[Recovered partial transcript | session {meta['original_session_uid']} | "
        f"{meta['recovered_count']} of {meta['original_count']} messages recovered "
        f"from a deleted conversation on {meta['recovered_at']}]"
    )
    body = [header, summary[:MAX_SUMMARY_CHARS]]
    if quotes:
        body.append(
            "Verbatim fragments that DO survive from that conversation "
            "(older, partially recovered - do not claim the rest is available): "
            + " | ".join(quotes)
        )
    body.append(
        "This is a forensic recovery of an older conversation, not the full "
        "transcript and not the current session. Most of that conversation is "
        "permanently lost. If asked about anything beyond what is written here, "
        "say you do not remember it instead of inventing it."
    )
    return {"at": f"{meta['recovered_at']}T00:00:00+00:00", "text": "\n".join(body)}


__all__ = [
    "ORIGINAL_COUNT",
    "RECOVERED_AT",
    "RECOVERED_COUNT",
    "RECOVERED_SESSION_UID",
    "RECOVERED_SOURCE",
    "RECOVERED_STATUS",
    "is_available",
    "load_recovered_previous_session",
    "recovered_metadata",
]
