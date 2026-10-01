"""Deterministic web-search intent router (Phase 3).

Pure, synchronous, O(message length), no network, no LLM call.

The router owns ONLY web-search intent: given a user turn's text it
decides SEARCH_REQUIRED | NORMAL_CHAT | UNCERTAIN. Callers map
UNCERTAIN -> NORMAL_CHAT (no extra classification call).

Layered policy (first match wins):
  1. Opinion/personal guard  -> NORMAL_CHAT
  2. Physical-object / personal-activity guard -> NORMAL_CHAT
  3. Explicit web / info / imperative / freshness / verification rules
     -> SEARCH_REQUIRED
  4. Anything else -> UNCERTAIN

This module also owns the router-side orchestration helper
(`execute_router_search`) and the context-block format, so the
conversation pipeline stays thin. It reuses the existing ToolExecutor
(and therefore the Phase 1 reliability layer) without duplicating it.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Optional

SEARCH_REQUIRED = "SEARCH_REQUIRED"
NORMAL_CHAT = "NORMAL_CHAT"
UNCERTAIN = "UNCERTAIN"

# Per-turn application-level search states (kept outside _memory and
# mirrored onto the request latency tracker for log observability).
SEARCH_NOT_REQUESTED = "SEARCH_NOT_REQUESTED"
SEARCH_EXECUTED_SUCCESS = "SEARCH_EXECUTED_SUCCESS"
SEARCH_EXECUTED_NO_RESULTS = "SEARCH_EXECUTED_NO_RESULTS"
SEARCH_EXECUTION_ERROR = "SEARCH_EXECUTION_ERROR"
SEARCH_REQUIRED_NOT_EXECUTED = "SEARCH_REQUIRED_NOT_EXECUTED"

ROUTER_SEARCH_TOOL = "search"
MAX_ROUTER_BLOCK_CHARS = 4000

# --- pattern layers (all matched case-insensitively on lowered text) ---

_OPINION = re.compile(
    r"\b(menurutmu|menurut kamu|pendapatmu|sebaiknya aku|aku harus|mending|"
    r"enaknya|bagus nggak|bagus gak|do you think|in your opinion)\b"
)
_FRESH_COMMERCIAL = re.compile(r"\b(harga|price|versi|rilis|release|berita|news)\b")
_WEB_STRONG = re.compile(
    r"\b(internet|web\b|online|website|situs|link\b|sumber|url|browse|"
    r"googling|search engine|di web|di internet)\b"
)
_FRESH_STRONG = re.compile(
    r"\b(terbaru|terkini|saat ini|hari ini|latest|current|today|this week|"
    r"minggu ini|rilis|release|update)\b"
)
_COMMERCIAL = re.compile(
    r"\b(terbaik|review|rekomendasi|spesifikasi|spek|harga|price)\b"
)
_PRONOUN = re.compile(r"\b(aku|gw|gue|saya|aq)\b")
_PAST_MARK = re.compile(r"\b(tadi|kemarin|barusan|telah|sudah)\b")
_LOSS_MARK = re.compile(r"\b(nggak ketemu|ngga ketemu|tidak ketemu|hilang|ketemu)\b")
_PHYSICAL_NOUN = re.compile(
    r"\b(charger|cas\b|dompet|kunci|file|folder|dokumen|makan|makanan|"
    r"ide|inspirasi|lirik|jodoh|pacar|kerja|kerjaan|kost|kontrakan|"
    r"motor|hape|hp\b)\b"
)
_SEARCH_VERB = re.compile(
    r"\b(cari|carikan|nyari|mencari|search|find|cek|coba cek|coba cari|"
    r"tolong cari|tolong carikan|look up|cekkan)\b"
)
_FOOD_IDIOM = re.compile(r"\b(makan|ide|inspirasi|apa ya|enaknya|lirik|jodoh)\b")
_INTERROGATIVE = re.compile(
    r"\b(apa|apakah|siapa|berapa|bagaimana|gimana|kapan|dimana|di mana|"
    r"which|what|who|how much|how many|is there|are there|has there)\b|\?\s*$"
)
_EXPLICIT_INFO = re.compile(
    r"\b(cari|carikan|nyari|search|find|cek|lihat|kasih|berikan|bagi|"
    r"share|tunjukin|tunjukkan)\b.{0,25}\b(informasi|info|tentang|berita|"
    r"sumber|link|data|detail|artikel)\b"
    r"|\b(informasi|berita|sumber|artikel)\b.{0,25}\b(tentang|terkini|"
    r"terbaru)\b"
)
_IMPERATIVE_HEAD = re.compile(
    r"^(cari|carikan|search|find|cek|coba cek|coba cari|tolong cari|"
    r"tolong carikan)\b\s*(.+)$"
)
_VERIFY = re.compile(
    r"\b(cek|verifikasi|verify|lihat|konfirmasi|crosscheck|cross check)\b"
    r".{0,20}\b(apakah|apa|kebenaran|benar|betul|rilis|dirilis|fakta|hoax|hoaks)\b"
    r"|\bapakah\b.{0,40}\b(sudah|benar|betul|rilis|dirilis|rilisnya)\b"
)
_SOURCE_ASK = re.compile(
    r"\b(kasih|berikan|minta|bagi|share|tunjukin|tunjukkan|ada)\b.{0,15}\b"
    r"(sumber|link|referensi|source|tautan)\b"
    r"|\b(sumber|link|referensi)\b.{0,15}(dong|ya|nya|kah)\b"
)


def _normalize(text: Any) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return collapsed


def classify_search_intent(text: Any) -> str:
    """Classify a user turn (pure).

    Returns SEARCH_REQUIRED | NORMAL_CHAT | UNCERTAIN.
    """
    lowered = _normalize(text)
    if not lowered:
        return NORMAL_CHAT

    # Layer 1: opinion/personal questions stay chat (unless the turn is
    # really about web/current info, checked by later layers first...
    # no: opinion guard wins only without web/fresh-commercial markers).
    if _OPINION.search(lowered) and not (
        _WEB_STRONG.search(lowered) or _FRESH_COMMERCIAL.search(lowered)
    ):
        return NORMAL_CHAT

    has_web = bool(_WEB_STRONG.search(lowered))
    has_fresh = bool(_FRESH_STRONG.search(lowered))
    has_commercial = bool(_COMMERCIAL.search(lowered))

    # Layer 2: physical-object / personal-activity searching.
    if (
        _PRONOUN.search(lowered)
        or _LOSS_MARK.search(lowered)
        or _PAST_MARK.search(lowered)
    ) and _PHYSICAL_NOUN.search(lowered):
        if not (has_web or has_fresh or has_commercial):
            return NORMAL_CHAT
    # First-person past activity without web markers ("aku tadi main
    # game", "aku cari game tadi").
    if (
        _PRONOUN.search(lowered)
        and _PAST_MARK.search(lowered)
        and not (has_web or has_fresh)
        and not re.search(r"\b(informasi|tentang|berita|sumber)\b", lowered)
    ):
        return NORMAL_CHAT

    # Layer 3: explicit web / info / imperative / freshness / verify.
    if has_web:
        return SEARCH_REQUIRED
    if _EXPLICIT_INFO.search(lowered):
        return SEARCH_REQUIRED
    head = _IMPERATIVE_HEAD.match(lowered)
    if head and head.group(2).strip():
        if not _FOOD_IDIOM.search(lowered):
            return SEARCH_REQUIRED
    if _INTERROGATIVE.search(lowered) and (has_fresh or has_commercial):
        return SEARCH_REQUIRED
    # Commercial fragment without interrogative ("Harga emas sekarang"):
    # price noun + current-time marker. The time marker keeps idioms like
    # "harga dirinya jatuh" out.
    if re.search(r"\b(harga|price)\b", lowered) and re.search(
        r"\b(sekarang|saat ini|terkini|hari ini|current|today)\b", lowered
    ):
        return SEARCH_REQUIRED
    if _VERIFY.search(lowered):
        return SEARCH_REQUIRED
    if _SOURCE_ASK.search(lowered):
        return SEARCH_REQUIRED

    return UNCERTAIN


def is_search_required(text: Any) -> bool:
    """Final router verdict: UNCERTAIN maps to NORMAL_CHAT (pure)."""
    return classify_search_intent(text) == SEARCH_REQUIRED


def normalize_search_query(text: Any) -> str:
    """Minimal deterministic normalization (pure, no rewrite).

    The user's original request is preserved as the search query;
    only whitespace is collapsed. DuckDuckGo handles natural language.
    """
    return re.sub(r"\s+", " ", str(text or "")).strip()


# Phase 1 LLM-facing prefixes (mirrored here only to map an executed
# result back to router state; the formats themselves are owned by
# search_reliability.format_search_text — do not drift them apart).
_PREFIX_SUCCESS = "Search results found:"
_PREFIX_NO_RESULTS = "returned no results"
_PREFIX_ERROR = "Search execution failed:"


def state_from_search_text(is_error: bool, text: Any) -> str:
    """Map a run_single_tool search result to router state (pure)."""
    body = str(text or "")
    if is_error or _PREFIX_ERROR in body:
        return SEARCH_EXECUTION_ERROR
    if _PREFIX_SUCCESS in body:
        return SEARCH_EXECUTED_SUCCESS
    if _PREFIX_NO_RESULTS in body:
        return SEARCH_EXECUTED_NO_RESULTS
    # Unrecognized shape: fail closed, never claim success.
    return SEARCH_EXECUTION_ERROR


def build_router_search_block(
    query: str, state: str, result_count: Optional[int], result_text: str
) -> str:
    """Labeled context block for the turn's deterministic search (pure)."""
    shown_count = result_count if result_count is not None else 0
    body = str(result_text or "").strip()[:MAX_ROUTER_BLOCK_CHARS]
    lines = [
        "[WEB SEARCH EXECUTED BY APPLICATION]",
        f"query: {query}",
        f"status: {state}",
        f"result_count: {shown_count}",
        "",
        body,
        "",
        "A web search was already executed for this exact query in this "
        "turn. Do not call the search tool again for the same query; "
        "a further distinct search is allowed only if the user asks "
        "something new. Never claim a search returned results it did not "
        "return, and never infer that no results means the entity does "
        "not exist.",
    ]
    return "\n".join(lines).strip()


@dataclass
class RouterSearchResult:
    """Outcome of the router's deterministic search for one turn."""

    state: str
    result_count: Optional[int] = None
    block_text: str = ""
    query: str = ""


async def execute_router_search(executor: Any, query: str) -> RouterSearchResult:
    """Run one deterministic search through the existing ToolExecutor.

    Uses run_single_tool("search", ...) so the Phase 1 reliability layer
    (status + single retry) applies unchanged. Returns the state, count,
    and ready-to-inject context block. Never raises: executor failure
    maps to SEARCH_EXECUTION_ERROR.
    """
    clean_query = normalize_search_query(query)
    tool_id = f"router-search-{uuid.uuid4().hex[:8]}"
    try:
        is_error, text, _metadata, _items = await executor.run_single_tool(
            ROUTER_SEARCH_TOOL,
            tool_id,
            {"query": clean_query, "max_results": 10},
        )
    except Exception:
        return RouterSearchResult(
            state=SEARCH_EXECUTION_ERROR,
            block_text=build_router_search_block(
                clean_query,
                SEARCH_EXECUTION_ERROR,
                None,
                "Search execution failed: the search tool could not be "
                "executed. Do not treat this as 'no results'.",
            ),
            query=clean_query,
        )
    state = state_from_search_text(bool(is_error), text)
    count: Optional[int] = None
    match = re.search(r"Found (\d+) search results:", str(text or ""))
    if match:
        try:
            count = int(match.group(1))
        except (TypeError, ValueError):
            count = None
    block = ""
    if state in (SEARCH_EXECUTED_SUCCESS, SEARCH_EXECUTED_NO_RESULTS):
        block = build_router_search_block(clean_query, state, count, str(text or ""))
    elif state == SEARCH_EXECUTION_ERROR:
        block = build_router_search_block(clean_query, state, None, str(text or ""))
    return RouterSearchResult(
        state=state, result_count=count, block_text=block, query=clean_query
    )
