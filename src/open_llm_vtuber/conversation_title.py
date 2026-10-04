"""Automatic conversation titles — auxiliary, fail-soft, deterministic gates.

A new conversation starts untitled (the sidebar shows "Percakapan Baru").
Once the exchange carries enough meaningful context, ONE title is generated
with the conversation's own LLM, persisted into the existing history
metadata ``title`` field, and never regenerated.

Layer separation (deliberate):

- history metadata ``title``: owned by ``chat_history_manager``.
- sidebar rendering/numbering: frontend only, untouched here.
- relationship / memory / episodic / proactive: never consulted, never
  written by this module.

Everything here is fail-soft: a missing store, a failed model call, or a
malformed model reply simply yields ``""`` and the chat continues with
"Percakapan Baru". Title generation must never break a conversation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

TITLE_SYSTEM_PROMPT = """You write a very short title for a chat conversation.
Reply with ONLY the title, nothing else. Rules: 4-7 words, no quotes, no
markdown, no prefix like "Judul:" or "Title:", no trailing period. Use the
same language as the conversation. Name the main topic, not a greeting."""

# Pure greetings / test noise: never enough for a meaningful title.
_GREETING_ONLY = re.compile(
    r"^(?:halo+|hallo+|hai+|hello+|hi+|hei+|pagi+|siang+|sore+|malam+|"
    r"assalamu(?:alaikum)?(?:\s+wr\.?\s*wb\.?)?|"
    r"tes(?:ting)?|test(?:ing)?|ok(?:e|ey)?|oke+|sip+|wkwk+|haha+|hehe+|"
    r"yo+|bro+|kak+|mili+)[\s.!?…?]*$",
    re.IGNORECASE,
)

# Prefixes a model sometimes adds despite the system prompt.
_TITLE_PREFIX = re.compile(r"^(?:judul|title|topik|topic)\s*[:\-–—]\s*", re.IGNORECASE)

MIN_USER_TURNS = 2
MIN_ASSISTANT_TURNS = 1
MIN_MEANINGFUL_CHARS = 12
MIN_TOTAL_USER_CHARS = 20
TITLE_MAX_WORDS = 7
TITLE_MAX_CHARS = 60
TITLE_CONTEXT_TURNS = 8
TITLE_CONTEXT_CHARS_PER_TURN = 300


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _is_greeting_only(text: str) -> bool:
    return bool(text) and bool(_GREETING_ONLY.match(text))


def has_stored_title(metadata: Any) -> bool:
    """True when history metadata already carries a usable title."""
    if not isinstance(metadata, dict):
        return False
    return bool(_clean(metadata.get("title")))


def should_generate_title(metadata: Any, messages: Any) -> bool:
    """Deterministic gate: exactly one title per conversation, and only once
    the exchange is meaningful. Pure: no I/O, no model call."""
    if has_stored_title(metadata):
        return False
    if not isinstance(messages, list):
        return False
    user_texts = [
        _clean(item.get("content"))
        for item in messages
        if isinstance(item, dict)
        and item.get("role") == "human"
        and _clean(item.get("content"))
    ]
    assistant_turns = sum(
        1
        for item in messages
        if isinstance(item, dict)
        and item.get("role") == "ai"
        and _clean(item.get("content"))
    )
    if len(user_texts) < MIN_USER_TURNS or assistant_turns < MIN_ASSISTANT_TURNS:
        return False
    if sum(len(text) for text in user_texts) < MIN_TOTAL_USER_CHARS:
        return False
    meaningful = any(
        len(text) >= MIN_MEANINGFUL_CHARS and not _is_greeting_only(text)
        for text in user_texts
    )
    return meaningful


def sanitize_title(raw: Any) -> str:
    """Reduce a model reply to a short plain title, or ``""`` when unusable.

    Never invents a fallback: an empty result means "keep Percakapan Baru".
    """
    text = _clean(raw)
    # Strip markdown emphasis/bullets first so prefixed titles wrapped in
    # "**Judul: ...**" still expose the prefix to the rule below.
    text = text.strip("*_~#>-. ").strip()
    text = _clean(text)
    # Strip one layer of surrounding quotes/backticks.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    text = _TITLE_PREFIX.sub("", text).strip()
    text = text.strip("\"'`*_-–—:;.…!? ").strip()
    text = _clean(text)
    if not text:
        return ""
    words = text.split()
    if len(words) > TITLE_MAX_WORDS:
        text = " ".join(words[:TITLE_MAX_WORDS])
    if len(text) > TITLE_MAX_CHARS:
        cut = text[:TITLE_MAX_CHARS].rsplit(" ", 1)
        text = cut[0] if len(cut) > 1 and cut[0] else text[:TITLE_MAX_CHARS]
        text = text.strip()
    if len(text) < 2 or not re.search(r"[A-Za-z0-9\u00C0-\u024F]", text):
        return ""
    return text


def build_title_prompt(messages: List[Dict[str, Any]]) -> str:
    """Bounded title input from recent turns only. Never the full transcript."""
    lines: List[str] = []
    turns = [
        item
        for item in messages or []
        if isinstance(item, dict) and item.get("role") in ("human", "ai")
    ][-TITLE_CONTEXT_TURNS:]
    for item in turns:
        text = _clean(item.get("content"))[:TITLE_CONTEXT_CHARS_PER_TURN]
        if not text:
            continue
        speaker = "User" if item.get("role") == "human" else "Mili"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


async def generate_title(llm_chat_fn: Any, messages: Any) -> str:
    """One auxiliary model call; returns a sanitized title or ``""``.

    Never raises: every failure mode degrades to "no title".
    """
    try:
        if not callable(llm_chat_fn):
            return ""
        prompt = build_title_prompt(messages if isinstance(messages, list) else [])
        if not prompt.strip():
            return ""
        chunks: List[str] = []
        stream = llm_chat_fn([{"role": "user", "content": prompt}], TITLE_SYSTEM_PROMPT)
        # Support both async generators and plain awaitables returning text.
        if hasattr(stream, "__aiter__"):
            async for event in stream:
                if isinstance(event, str):
                    chunks.append(event)
                elif isinstance(event, dict) and event.get("type") == "text_delta":
                    chunks.append(str(event.get("text", "")))
        else:
            try:
                import inspect as _inspect

                result = await stream if _inspect.isawaitable(stream) else stream
            except Exception:
                return ""
            if isinstance(result, str):
                chunks.append(result)
        return sanitize_title("".join(chunks))
    except Exception:
        return ""


__all__ = [
    "TITLE_SYSTEM_PROMPT",
    "TITLE_MAX_WORDS",
    "TITLE_MAX_CHARS",
    "build_title_prompt",
    "generate_title",
    "has_stored_title",
    "sanitize_title",
    "should_generate_title",
]
