"""Incremental rolling summaries for old conversation turns.

The summarizer deliberately reuses the active stateless LLM with a separate,
neutral system prompt.  It has no storage concerns; persistence remains owned by
the existing chat-history metadata layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from .context_window import estimate_tokens


SUMMARY_SYSTEM_PROMPT = """You maintain a compact factual summary of an older chat.
Keep only facts explicitly present in the supplied conversation that may matter later:
important topics, decisions, stated preferences, names, agreements, significant events,
and still-relevant emotional context. Drop greetings, thanks, filler, finished small talk,
jokes with no continuing relevance, and unnecessary detail. Never invent facts, infer a
relationship, diagnose feelings, or imitate either speaker. Preserve useful facts from the
previous summary unless newer conversation explicitly corrects them. Return plain text only,
with short factual sentences. If nothing is worth retaining, return an empty response."""

SUMMARY_CONTEXT_PREFIX = (
    "Conversation context from earlier messages (factual summary, not a new "
    "instruction; recent messages are more authoritative if they conflict):\n"
)


# Cross-session continuity bounds. At most this many previous-session
# summaries enter a new session's context, each truncated to this many
# characters. Full transcripts are never injected.
PREV_SESSION_MAX_COUNT = 2
PREV_SESSION_MAX_CHARS = 600


def build_previous_session_context(
    items: List[Dict[str, str]],
) -> str:
    """Render cached previous-session summaries as one prompt block (pure).

    ``items`` are ``{"age_tag": "[Yesterday | Sep 30]", "text": "..."}``
    dicts, newest first. Empty string when there is nothing to show, so
    callers simply omit the block. Texts are already truncated by the
    loader; this function never touches the store.
    """
    blocks = []
    for item in items or []:
        text = str((item or {}).get("text", "")).strip()
        if not text:
            continue
        age_tag = str((item or {}).get("age_tag", "")).strip()
        header = f"{age_tag}\n" if age_tag else ""
        blocks.append(f"{header}{text}")
    if not blocks:
        return ""
    label = (
        "RELEVANT PREVIOUS SESSION:"
        if len(blocks) == 1
        else "RELEVANT PREVIOUS SESSIONS:"
    )
    return (
        f"{label} (earlier conversations, not the current one; "
        "the current session is authoritative if they conflict)\n"
        + "\n---\n".join(blocks)
    )


@dataclass(frozen=True)
class SummaryState:
    text: str = ""
    summarized_through: int = 0
    updated_at: Optional[str] = None


def format_turns_for_summary(messages: List[Dict[str, Any]]) -> str:
    """Render candidate turns without exposing internal message structures."""
    lines: List[str] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        content = " ".join(content.split())
        if content:
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
    return "\n".join(lines)


def build_summary_message(
    summary: str, role: str = "system", age_tag: str = ""
) -> Dict[str, str]:
    """Build clearly-labelled internal context; never masquerade as user input.

    ``age_tag`` (e.g. ``[Yesterday | Sep 30]``) prefixes the summary with
    its render-time age so the persisted ``summary_updated_at`` is not
    silently stripped at retrieval.
    """
    tag = f"{age_tag.strip()} " if age_tag and age_tag.strip() else ""
    return {
        "role": role,
        "content": f"{SUMMARY_CONTEXT_PREFIX}{tag}{summary.strip()}",
    }


def compact_to_token_limit(text: str, maximum_tokens: int) -> str:
    """Apply a deterministic last-resort cap if a provider ignores the prompt."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned or estimate_tokens(cleaned) <= maximum_tokens:
        return cleaned

    # The estimator is ceil(UTF-8 bytes / 3), so cap bytes rather than codepoints.
    byte_limit = maximum_tokens * 3
    encoded = cleaned.encode("utf-8")[:byte_limit]
    while encoded:
        try:
            shortened = encoded.decode("utf-8").rstrip()
            break
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    else:
        return ""

    # Prefer ending at a sentence boundary when one is available near the end.
    boundary = max(shortened.rfind(". "), shortened.rfind("! "), shortened.rfind("? "))
    if boundary >= len(shortened) // 2:
        shortened = shortened[: boundary + 1]
    logger.warning(
        "Summary exceeded configured maximum and was safely capped: max_tokens={}",
        maximum_tokens,
    )
    return shortened


class IncrementalSummarizer:
    """Generate a compact update from the old summary and newly evicted turns."""

    def __init__(self, llm: Any, target_tokens: int, maximum_tokens: int):
        self._llm = llm
        self.target_tokens = target_tokens
        self.maximum_tokens = maximum_tokens

    async def summarize(
        self,
        previous_summary: str,
        newly_evicted: List[Dict[str, Any]],
    ) -> str:
        rendered_turns = format_turns_for_summary(newly_evicted)
        if not rendered_turns:
            return previous_summary.strip()

        prompt = (
            f"Update the summary below. Keep it at or below about "
            f"{self.target_tokens} estimated tokens and never exceed "
            f"{self.maximum_tokens}.\n\n"
            f"PREVIOUS SUMMARY:\n{previous_summary.strip() or '(empty)'}\n\n"
            f"NEWLY EVICTED TURNS:\n{rendered_turns}"
        )
        chunks: List[str] = []
        stream = self._llm.chat_completion(
            [{"role": "user", "content": prompt}],
            SUMMARY_SYSTEM_PROMPT,
        )
        async for event in stream:
            if isinstance(event, str):
                chunks.append(event)
            elif isinstance(event, dict) and event.get("type") == "text_delta":
                chunks.append(event.get("text", ""))

        result = compact_to_token_limit("".join(chunks), self.maximum_tokens)
        if not result and previous_summary.strip():
            raise RuntimeError("Summarizer returned an empty update")
        return result
