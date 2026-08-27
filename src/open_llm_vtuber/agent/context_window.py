"""Small, dependency-free context budgeting helpers for conversation agents.

Token counts in this module are deliberately conservative estimates.  They are
used to avoid provider context overflows when a model-specific tokenizer is not
already available; they are not intended for billing or usage reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, List, Optional, Sequence


# Keep model metadata in one place. Aliases can move over time, so callers may
# use context_window_override when a provider changes an alias target.
KNOWN_CONTEXT_LIMITS: Dict[str, int] = {
    "mistral-small-latest": 256_000,
    "mistral-small-2603": 256_000,
    "mistral-small-2506": 128_000,
    "openai/gpt-oss-20b": 131_072,
    "gemma4:31b-cloud": 262_144,
}

DEFAULT_CONTEXT_LIMIT = 32_768
DEFAULT_RESERVED_OUTPUT_TOKENS = 1_024
DEFAULT_SAFETY_MARGIN = 1_024
# Vision providers tokenize image pixels, not the base64 transport string.  A
# fixed conservative allowance keeps the context guard useful without treating
# a normal photo upload as hundreds of thousands of text tokens.
DEFAULT_IMAGE_INPUT_TOKENS = 4_096


class ContextBudgetExceeded(ValueError):
    """Raised when required system/current-turn content cannot fit safely."""


@dataclass(frozen=True)
class ContextStats:
    model: str
    context_limit: int
    reserved_output: int
    safety_margin: int
    maximum_input_budget: int
    system_tokens: int
    tool_tokens: int
    history_tokens_before: int
    history_tokens_after: int
    messages_before: int
    messages_after: int
    estimated_input_tokens: int
    trimmed: bool
    used_fallback_limit: bool


@dataclass(frozen=True)
class ContextSelection:
    messages: List[Dict[str, Any]]
    stats: ContextStats


def resolve_context_limit(
    model: str | None,
    override: Optional[int] = None,
) -> tuple[int, bool]:
    """Return the configured/model-specific limit and whether fallback was used."""
    if override is not None:
        if override <= 0:
            raise ValueError("context_window_override must be greater than zero")
        return override, False

    normalized_model = (model or "").strip().lower()
    if normalized_model in KNOWN_CONTEXT_LIMITS:
        return KNOWN_CONTEXT_LIMITS[normalized_model], False
    return DEFAULT_CONTEXT_LIMIT, True


def _stable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        return str(value)


def estimate_tokens(value: Any) -> int:
    """Conservatively estimate tokens as ceil(UTF-8 bytes / 3).

    English text is commonly closer to four bytes/characters per token, while
    non-ASCII text and structured payloads vary. Three UTF-8 bytes per token is
    intentionally cautious across Indonesian, CJK, JSON, and mixed content.
    """
    text = _stable_text(value)
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 3))


def _estimate_content_tokens(content: Any) -> int:
    """Estimate chat content while treating data-URL images as vision input.

    OpenAI-compatible multimodal messages carry uploads as base64 data URLs.
    Counting that transport encoding as ordinary text can reject an otherwise
    valid image request before it reaches a vision-capable provider.
    """
    if not isinstance(content, list):
        return estimate_tokens(content)

    total = 0
    for item in content:
        if not isinstance(item, dict):
            total += estimate_tokens(item)
        elif item.get("type") == "image_url":
            total += DEFAULT_IMAGE_INPUT_TOKENS
        elif item.get("type") == "text":
            total += estimate_tokens(item.get("text", ""))
        else:
            total += estimate_tokens(item)
    return total


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    # Account for role/name/other schema fields without serializing a base64
    # image payload, then add the text/vision content estimate separately.
    metadata = {key: value for key, value in message.items() if key != "content"}
    # Extra framing covers role/message delimiters used by chat templates.
    return estimate_tokens(metadata) + _estimate_content_tokens(message.get("content")) + 6


def estimate_messages_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def _history_turns(
    history: Sequence[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Group stored history into user-led turns for coherent suffix trimming."""
    turns: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for message in history:
        if message.get("role") == "system":
            if current:
                turns.append(current)
                current = []
            # Internal context such as a rolling summary is its own lowest-
            # priority unit, so recent user/assistant turns never depend on it.
            turns.append([message])
        elif message.get("role") == "user":
            if current:
                turns.append(current)
            current = [message]
        elif current:
            current.append(message)
        else:
            # Startup/orphan messages form the oldest removable unit. They are
            # preserved when no trimming is necessary.
            current.append(message)

    if current:
        turns.append(current)
    return turns


def select_messages_for_context(
    *,
    messages: Sequence[Dict[str, Any]],
    system_prompt: str,
    model: str | None,
    reserved_output_tokens: Optional[int],
    safety_margin: int = DEFAULT_SAFETY_MARGIN,
    context_window_override: Optional[int] = None,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    protected_start: Optional[int] = None,
) -> ContextSelection:
    """Select a coherent recent suffix without mutating the stored transcript.

    ``protected_start`` identifies the current user turn. Everything from that
    index onward (including tool-call/result messages in the same request) is
    mandatory and is never silently truncated.
    """
    original = list(messages)
    context_limit, used_fallback = resolve_context_limit(model, context_window_override)
    reserved_output = (
        reserved_output_tokens
        if reserved_output_tokens is not None
        else DEFAULT_RESERVED_OUTPUT_TOKENS
    )
    if reserved_output < 0:
        raise ValueError("reserved_output_tokens cannot be negative")
    if safety_margin < 0:
        raise ValueError("context_safety_margin cannot be negative")

    maximum_input_budget = context_limit - reserved_output - safety_margin
    if maximum_input_budget <= 0:
        raise ContextBudgetExceeded(
            "Context limit is not larger than reserved output plus safety margin."
        )

    if protected_start is None:
        protected_start = next(
            (
                index
                for index in range(len(original) - 1, -1, -1)
                if original[index].get("role") == "user"
            ),
            max(0, len(original) - 1),
        )
    if not 0 <= protected_start <= len(original):
        raise ValueError("protected_start is outside the message list")

    history = original[:protected_start]
    protected = original[protected_start:]
    system_tokens = estimate_message_tokens(
        {"role": "system", "content": system_prompt or ""}
    )
    tool_tokens = estimate_tokens(list(tools)) + (8 * len(tools)) if tools else 0
    history_tokens_before = estimate_messages_tokens(history)
    protected_tokens = estimate_messages_tokens(protected)
    fixed_tokens = system_tokens + tool_tokens + protected_tokens

    if fixed_tokens > maximum_input_budget:
        raise ContextBudgetExceeded(
            "Required system instructions, tools, and current user turn exceed "
            "the safe input budget. Shorten the current message/tools or raise "
            "context_window_override for the active model."
        )

    total_before = fixed_tokens + history_tokens_before
    if total_before <= maximum_input_budget:
        selected_history = history
        trimmed = False
    else:
        remaining = maximum_input_budget - fixed_tokens
        selected_turns: List[List[Dict[str, Any]]] = []
        for turn in reversed(_history_turns(history)):
            turn_tokens = estimate_messages_tokens(turn)
            if turn_tokens > remaining:
                break
            selected_turns.append(turn)
            remaining -= turn_tokens
        selected_history = [
            message for turn in reversed(selected_turns) for message in turn
        ]
        trimmed = len(selected_history) != len(history)

    selected = [*selected_history, *protected]
    history_tokens_after = estimate_messages_tokens(selected_history)
    estimated_input_tokens = (
        system_tokens + tool_tokens + history_tokens_after + protected_tokens
    )
    stats = ContextStats(
        model=model or "unknown",
        context_limit=context_limit,
        reserved_output=reserved_output,
        safety_margin=safety_margin,
        maximum_input_budget=maximum_input_budget,
        system_tokens=system_tokens,
        tool_tokens=tool_tokens,
        history_tokens_before=history_tokens_before,
        history_tokens_after=history_tokens_after,
        messages_before=len(original),
        messages_after=len(selected),
        estimated_input_tokens=estimated_input_tokens,
        trimmed=trimmed,
        used_fallback_limit=used_fallback,
    )
    return ContextSelection(messages=selected, stats=stats)
