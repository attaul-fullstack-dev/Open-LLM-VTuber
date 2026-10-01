"""Structured reliability layer for the MCP web-search tool.

Phase 1 only: classify ``search`` results, retry once on empty results,
and keep tool-error semantics distinct from "no results". No new
provider, no router change, no LLM call, no schema change.

Internal model (never exposed as an LLM schema)::

    SUCCESS     request + parse ok, >=1 valid result
    NO_RESULTS  request + parse ok, 0 results
    SEARCH_ERROR  network/timeout/MCP exception/malformed response
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from loguru import logger

SEARCH_TOOL_NAME = "search"
MAX_SEARCH_ATTEMPTS = 2
MAX_LOGGED_QUERY_CHARS = 120
MAX_ERROR_CHARS = 200

SUCCESS = "SUCCESS"
NO_RESULTS = "NO_RESULTS"
SEARCH_ERROR = "SEARCH_ERROR"

# duckduckgo-mcp-server result markers (stable across 0.1.x and 0.7.x).
_FOUND_MARKER = re.compile(r"^Found (\d+) search results:", re.MULTILINE)
_EMPTY_MARKER = "No results were found"


@dataclass
class SearchOutcome:
    """Internal outcome of a (possibly retried) search execution."""

    status: str = SEARCH_ERROR
    result_count: Optional[int] = None
    attempts: int = 0
    queries: List[str] = field(default_factory=list)
    error: Optional[str] = None


def classify_search_text(text: Any) -> Tuple[str, Optional[int]]:
    """Classify raw search result text (pure).

    Returns (status, result_count). Unrecognized non-empty text is treated
    as SEARCH_ERROR: for this tool any response that is neither a result
    list nor the explicit empty marker is an unexpected format, and it
    must never be presented as a successful search.
    """
    if not isinstance(text, str) or not text.strip():
        return NO_RESULTS, 0
    match = _FOUND_MARKER.search(text)
    if match:
        try:
            return SUCCESS, int(match.group(1))
        except (TypeError, ValueError):
            return SEARCH_ERROR, None
    if _EMPTY_MARKER in text:
        return NO_RESULTS, 0
    return SEARCH_ERROR, None


def rephrase_search_query(query: Any) -> str:
    """Deterministic safe rephrase for the single retry (pure, no LLM).

    Strips quotation marks (which over-constrain DuckDuckGo phrase
    matching) and collapses whitespace. Returns "" when no distinct
    alternative can be derived.
    """
    base = str(query or "").strip()
    if not base:
        return ""
    alternative = re.sub(r"\s+", " ", base.replace('"', " ")).strip()
    if not alternative or alternative == base:
        return ""
    return alternative


def format_search_text(status: str, text: str, query: str = "", error: str = "") -> str:
    """Backward-compatible LLM-facing text for a search outcome (pure).

    Always a plain string; never includes tracebacks or internals.
    """
    body = str(text or "").strip()
    if status == SUCCESS:
        return f"Search results found:\n{body}" if body else "Search results found."
    if status == NO_RESULTS:
        shown = str(query or "").strip()[:MAX_LOGGED_QUERY_CHARS]
        return (
            "Search completed successfully but returned no results"
            + (f" for query: {shown}" if shown else "")
            + ". You may try a differently phrased query."
        )
    safe = re.sub(r"\s+", " ", str(error or "")).strip()[:MAX_ERROR_CHARS]
    detail = f" {safe}" if safe else ""
    return f"Search execution failed:{detail} Do not treat this as 'no results'."


def _sanitize_query(query: Any) -> str:
    return re.sub(r"\s+", " ", str(query or "")).strip()[:MAX_LOGGED_QUERY_CHARS]


def _extract_text(result_dict: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Pull (text, content_items) from an MCP result dict, defensively."""
    items: List[Dict[str, Any]] = []
    if isinstance(result_dict, dict):
        raw = result_dict.get("content_items", [])
        if isinstance(raw, list):
            items = [item for item in raw if isinstance(item, dict)]
    for item in items:
        if item.get("type") == "error":
            return str(item.get("text", "")), items
    for item in items:
        if item.get("type") == "text":
            return str(item.get("text", "")), items
    return "", items


async def execute_search_with_reliability(
    call_tool: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    tool_input: Any,
) -> Tuple[Dict[str, Any], SearchOutcome]:
    """Run ``search`` with structured status + at most one safe retry.

    ``call_tool`` performs a single MCP invocation and returns the raw
    result dict. Returns (final_result_dict, outcome); the final dict's
    first text item is already rewritten to the LLM-facing format, so
    downstream code needs no changes.
    """
    outcome = SearchOutcome()
    args = dict(tool_input) if isinstance(tool_input, dict) else {}
    first_query = _sanitize_query(args.get("query", ""))
    queries = [first_query]
    second_query = rephrase_search_query(first_query)
    # Exactly one retry on NO_RESULTS, using the rephrased query when one
    # exists; otherwise the original query is retried once (transient
    # bot-detection/rate-limit often clears on immediate retry).
    queries.append(second_query or first_query)

    final_dict: Dict[str, Any] = {"metadata": {}, "content_items": []}
    final_text = ""

    for attempt in range(1, MAX_SEARCH_ATTEMPTS + 1):
        outcome.attempts = attempt
        attempt_args = dict(args)
        attempt_args["query"] = queries[attempt - 1]
        try:
            result_dict = await call_tool(attempt_args)
        except Exception as error:
            outcome.status = SEARCH_ERROR
            outcome.error = f"{type(error).__name__}"
            detail = re.sub(r"\s+", " ", str(error)).strip()[:MAX_ERROR_CHARS]
            logger.warning(
                "search query={} attempt={} status={} error={}",
                _sanitize_query(queries[attempt - 1]),
                attempt,
                SEARCH_ERROR,
                outcome.error,
            )
            final_text = format_search_text(
                SEARCH_ERROR,
                "",
                queries[attempt - 1],
                f"{outcome.error}{(': ' + detail) if detail else ''}",
            )
            final_dict = {
                "metadata": {},
                "content_items": [{"type": "error", "text": final_text}],
            }
            break

        text, items = _extract_text(result_dict)
        if any(item.get("type") == "error" for item in items):
            outcome.status = SEARCH_ERROR
            outcome.error = "tool-reported error"
            logger.warning(
                "search query={} attempt={} status={} error={}",
                _sanitize_query(queries[attempt - 1]),
                attempt,
                SEARCH_ERROR,
                outcome.error,
            )
            final_text = format_search_text(
                SEARCH_ERROR, text, queries[attempt - 1], text
            )
            final_dict = {
                "metadata": result_dict.get("metadata", {})
                if isinstance(result_dict, dict)
                else {},
                "content_items": [{"type": "error", "text": final_text}],
            }
            break

        status, count = classify_search_text(text)
        outcome.status = status
        outcome.result_count = count
        logger.info(
            "search query={} attempt={} status={} result_count={}",
            _sanitize_query(queries[attempt - 1]),
            attempt,
            status,
            count,
        )
        if status == SUCCESS or attempt == MAX_SEARCH_ATTEMPTS:
            final_text = format_search_text(status, text, queries[attempt - 1])
            final_dict = {
                "metadata": result_dict.get("metadata", {})
                if isinstance(result_dict, dict)
                else {},
                "content_items": [{"type": "text", "text": final_text}],
            }
            break
        # NO_RESULTS with attempts left: loop retries once.

    outcome.queries = queries[: outcome.attempts]
    return final_dict, outcome
