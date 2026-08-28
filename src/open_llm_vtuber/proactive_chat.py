"""Lightweight runtime scheduling primitives for proactive character chat.

The state in this module is deliberately ephemeral.  Conversation history,
rolling summaries, relationship state, and long-term character memory keep
their existing persistence rules; idle timestamps never enter those stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
import time
from typing import Callable, Dict, List, Mapping, Optional


class ProactiveIntent:
    """Lightweight proactive behavior intents (internal only, never shown)."""

    REACT_TO_IGNORED_QUESTION = "react_to_ignored_question"
    REACT_TO_SILENCE = "react_to_silence"
    CONTINUE_PREVIOUS_TOPIC = "continue_previous_topic"
    START_NEW_TOPIC = "start_new_topic"
    ASK_USER_SOMETHING = "ask_user_something"
    BRING_UP_MEMORY = "bring_up_memory"
    CASUAL_OBSERVATION = "casual_observation"


# Fixed iteration order for weighted selection (deterministic wheel layout).
INTENT_SELECTION_ORDER = (
    ProactiveIntent.REACT_TO_SILENCE,
    ProactiveIntent.CONTINUE_PREVIOUS_TOPIC,
    ProactiveIntent.START_NEW_TOPIC,
    ProactiveIntent.ASK_USER_SOMETHING,
    ProactiveIntent.BRING_UP_MEMORY,
    ProactiveIntent.CASUAL_OBSERVATION,
)

# react_to_ignored_question is priority-driven and intentionally excluded
# from weighted selection.
DEFAULT_INTENT_WEIGHTS: Dict[str, float] = {
    ProactiveIntent.REACT_TO_SILENCE: 5,
    ProactiveIntent.CONTINUE_PREVIOUS_TOPIC: 20,
    ProactiveIntent.START_NEW_TOPIC: 30,
    ProactiveIntent.ASK_USER_SOMETHING: 20,
    ProactiveIntent.BRING_UP_MEMORY: 15,
    ProactiveIntent.CASUAL_OBSERVATION: 10,
}

# Per recent occurrence, a repeated intent's weight is multiplied down so
# selections stay varied.  Silence acknowledgment decays extra fast.
_INTENT_REPEAT_PENALTY = 0.25
_SILENCE_REPEAT_PENALTY = 0.1


@dataclass(frozen=True)
class ProactiveChatConfig:
    """Centralized proactive timing configuration."""

    enabled: bool = True
    initial_idle_min_seconds: int = 45
    initial_idle_max_seconds: int = 90
    followup_idle_min_seconds: int = 90
    followup_idle_max_seconds: int = 240
    ignored_before_backoff: int = 3
    backoff_min_seconds: int = 180
    backoff_max_seconds: int = 360
    intent_weights: Optional[Mapping[str, float]] = None

    def __post_init__(self) -> None:
        ranges = (
            (
                "initial idle",
                self.initial_idle_min_seconds,
                self.initial_idle_max_seconds,
            ),
            (
                "follow-up idle",
                self.followup_idle_min_seconds,
                self.followup_idle_max_seconds,
            ),
            ("backoff", self.backoff_min_seconds, self.backoff_max_seconds),
        )
        for label, minimum, maximum in ranges:
            if minimum < 0 or maximum < minimum:
                raise ValueError(
                    f"Invalid proactive {label} range: {minimum}-{maximum}"
                )
        if self.ignored_before_backoff < 1:
            raise ValueError("ignored_before_backoff must be at least 1")
        if self.intent_weights is not None:
            for key, value in self.intent_weights.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0
                ):
                    raise ValueError(
                        f"Invalid proactive intent weight for {key!r}: {value!r}"
                    )


# Sentence-initial interrogatives (Indonesian + English) used as a fallback
# signal when a proactive message expects a reply without a question mark.
_QUESTION_WORD_PATTERN = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(?:apakah|apa|gimana|bagaimana|kenapa|kok|kapan|dimana|di mana|siapa|"
    r"what|why|how|when|where|who)\b",
    re.IGNORECASE,
)


def message_expects_response(text: Optional[str]) -> bool:
    """Deterministic check for whether a message asks for a user reply.

    A question mark anywhere is the primary signal.  As a fallback, a
    sentence-initial interrogative word (Indonesian or English) also counts.
    This is a lightweight heuristic over already-generated text -- it never
    triggers any model call.
    """
    if not text:
        return False
    if "?" in text:
        return True
    return bool(_QUESTION_WORD_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Lightweight topic heuristics (pure Python; no embeddings, no model calls).
# ---------------------------------------------------------------------------

_TOPIC_STOPWORDS = frozenset(
    {
        # Indonesian pronouns / particles / filler / question words
        "aku",
        "saya",
        "kamu",
        "km",
        "anda",
        "dia",
        "ia",
        "kita",
        "kami",
        "mereka",
        "ini",
        "itu",
        "yang",
        "dan",
        "atau",
        "tapi",
        "tetapi",
        "kalau",
        "kalo",
        "jika",
        "sama",
        "ke",
        "dari",
        "di",
        "ke",
        "nggak",
        "gak",
        "ngga",
        "ga",
        "tidak",
        "tak",
        "bukan",
        "sudah",
        "udah",
        "belum",
        "juga",
        "aja",
        "saja",
        "banget",
        "kok",
        "dong",
        "sih",
        "deh",
        "nih",
        "tuh",
        "gitu",
        "begini",
        "begitu",
        "apa",
        "apakah",
        "gimana",
        "bagaimana",
        "kenapa",
        "kapan",
        "dimana",
        "mana",
        "siapa",
        "mau",
        "ingin",
        "bisa",
        "akan",
        "ada",
        "iya",
        "oke",
        "ok",
        "sip",
        "hmm",
        "hm",
        "hehe",
        "wkwk",
        "wah",
        "yaudah",
        "paham",
        "makasih",
        "terima",
        "kasih",
        "banyak",
        "semua",
        "lagi",
        "biar",
        "supaya",
        "untuk",
        "dengan",
        "pada",
        "oleh",
        "karena",
        "karna",
        "emang",
        "memang",
        "doang",
        "cuma",
        "cuman",
        "kayak",
        "gini",
        "terus",
        "trus",
        "abis",
        "habis",
        "nanti",
        "sekarang",
        "kemarin",
        "besok",
        "tadi",
        "nanya",
        "tanya",
        "bilang",
        "kata",
        "banget",
        "sangat",
        "masa",
        "paling",
        "user",
        "banget",
        # English filler / function words
        "the",
        "and",
        "but",
        "for",
        "with",
        "about",
        "from",
        "into",
        "you",
        "your",
        "yours",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "just",
        "really",
        "very",
        "too",
        "also",
        "not",
        "yes",
        "yeah",
        "okay",
        "hey",
        "hi",
        "lol",
        "um",
        "uh",
        "like",
        "get",
        "got",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "this",
        "that",
        "these",
        "those",
    }
)

_LOW_INFO_REPLIES = frozenset(
    {
        "iya",
        "gak",
        "nggak",
        "ga",
        "ok",
        "oke",
        "sip",
        "hmm",
        "hm",
        "hehe",
        "wkwk",
        "ya",
        "yoi",
        "gitu",
        "oh",
        "wah",
        "yo",
        "udah",
        "oke sip",
        "yaudah",
        "paham",
        "oke deh",
        "ohh",
        "he_em",
        "hem",
        "ah",
        "oh gitu",
        "oke deh",
        "sip deh",
    }
)

_CLOSURE_PHRASES = frozenset(
    {
        "oh gitu",
        "oke",
        "ok",
        "okeh",
        "yaudah",
        "ya udah",
        "paham",
        "sip",
        "makasih",
        "terima kasih",
        "udah",
        "gitu aja",
        "oke sip",
        "oke deh",
        "sip deh",
        "oh oke",
        "ohh",
        "udahan",
        "dah",
        "oke makasih",
        "sip makasih",
        "paham kok",
    }
)

_CLOSURE_VOCAB = frozenset(
    {
        "oh",
        "ohh",
        "oke",
        "ok",
        "okeh",
        "gitu",
        "yaudah",
        "ya",
        "udah",
        "dah",
        "paham",
        "sip",
        "makasih",
        "terima",
        "kasih",
        "aja",
        "deh",
        "sipp",
        "nya",
    }
)

_TRANSITION_MARKERS = (
    "ngomong-ngomong",
    "ngomong2",
    "btw",
    "beda topik",
    "ganti topik",
    "oh iya",
    "oh ya",
    "sidenote",
)


def clamp01(value: float) -> float:
    """Clamp a numeric score into the inclusive 0..1 range."""
    return max(0.0, min(1.0, float(value)))


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    cleaned = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return " ".join(cleaned.split())


def tokenize_for_topic(text: str) -> tuple:
    """Lowercase, strip punctuation, drop stopwords/filler/short tokens."""
    tokens = []
    seen = set()
    for word in _normalize_text(text).split():
        if len(word) < 3 or word in _TOPIC_STOPWORDS or word.isdigit():
            continue
        if word not in seen:
            seen.add(word)
            tokens.append(word)
    return tuple(tokens)


def topic_signature(texts, max_terms: int = 4) -> tuple:
    """Deterministic top-``max_terms`` keyword signature for a text group."""
    counter: Dict[str, int] = {}
    for text in texts:
        for token in tokenize_for_topic(text):
            counter[token] = counter.get(token, 0) + 1
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return tuple(token for token, _ in ordered[:max_terms])


def _term_set_overlap(set_a, set_b) -> float:
    """Overlap coefficient |A∩B| / min(|A|, |B|); 0.0 when degenerate."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    if not intersection:
        return 0.0
    return intersection / max(1, min(len(set_a), len(set_b)))


def topic_similarity(texts_a, texts_b) -> float:
    """Keyword-set similarity (overlap coefficient) between two text groups."""
    tokens_a = set()
    for text in texts_a:
        tokens_a.update(tokenize_for_topic(text))
    tokens_b = set()
    for text in texts_b:
        tokens_b.update(tokenize_for_topic(text))
    return clamp01(_term_set_overlap(tokens_a, tokens_b))


def signature_similarity(sig_a, sig_b) -> float:
    """Overlap between two precomputed keyword signatures (0..1)."""
    return clamp01(_term_set_overlap(set(sig_a or ()), set(sig_b or ())))


def _is_low_info_reply(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    return normalized in _LOW_INFO_REPLIES or len(normalized) <= 4


def _is_closure_like(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if normalized in _CLOSURE_PHRASES:
        return True
    if len(normalized) > 24:
        return False
    words = normalized.split()
    return bool(words) and all(word in _CLOSURE_VOCAB for word in words)


@dataclass(frozen=True)
class ProactiveFollowupContext:
    """Internal-only signal about a possibly ignored proactive message.

    Consumed by proactive generation to react naturally to being ignored;
    it must never be printed verbatim to the user.
    """

    previous_proactive_ignored: bool
    consecutive_ignored: int
    previous_proactive_expected_response: bool

    def as_dict(self) -> dict:
        return {
            "previous_proactive_ignored": self.previous_proactive_ignored,
            "consecutive_ignored": self.consecutive_ignored,
            "previous_proactive_expected_response": self.previous_proactive_expected_response,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["ProactiveFollowupContext"]:
        if not isinstance(data, dict):
            return None
        return cls(
            previous_proactive_ignored=bool(data.get("previous_proactive_ignored")),
            consecutive_ignored=int(data.get("consecutive_ignored") or 0),
            previous_proactive_expected_response=bool(
                data.get("previous_proactive_expected_response")
            ),
        )


@dataclass(frozen=True)
class ProactiveIntentContext:
    """Internal-only proactive intent signal for one generation turn."""

    intent: str
    user_has_replied_since_last_proactive: bool
    consecutive_ignored: int
    recent_silence_acknowledgment: bool
    # Compact semantic hints (bands + keyword tuples); never user-visible.
    topic_continuity_band: str = ""
    topic_staleness_band: str = ""
    user_engagement_band: str = ""
    dominant_topic_keywords: tuple = ()
    avoid_recent_topics: tuple = ()

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "user_has_replied_since_last_proactive": (
                self.user_has_replied_since_last_proactive
            ),
            "consecutive_ignored": self.consecutive_ignored,
            "recent_silence_acknowledgment": self.recent_silence_acknowledgment,
            "topic_continuity_band": self.topic_continuity_band,
            "topic_staleness_band": self.topic_staleness_band,
            "user_engagement_band": self.user_engagement_band,
            "dominant_topic_keywords": list(self.dominant_topic_keywords),
            "avoid_recent_topics": [list(sig) for sig in self.avoid_recent_topics],
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["ProactiveIntentContext"]:
        if not isinstance(data, dict):
            return None
        intent = str(data.get("intent") or "")
        known = set(INTENT_SELECTION_ORDER) | {
            ProactiveIntent.REACT_TO_IGNORED_QUESTION
        }
        if intent not in known:
            intent = ProactiveIntent.CASUAL_OBSERVATION
        valid_bands = {"low", "medium", "high"}
        keywords = tuple(
            str(word)
            for word in (data.get("dominant_topic_keywords") or ())[:4]
            if str(word).strip()
        )
        avoid = tuple(
            tuple(str(word) for word in signature[:4])
            for signature in (data.get("avoid_recent_topics") or ())[:3]
            if isinstance(signature, (list, tuple)) and signature
        )
        return cls(
            intent=intent,
            user_has_replied_since_last_proactive=bool(
                data.get("user_has_replied_since_last_proactive")
            ),
            consecutive_ignored=int(data.get("consecutive_ignored") or 0),
            recent_silence_acknowledgment=bool(
                data.get("recent_silence_acknowledgment")
            ),
            topic_continuity_band=(
                str(data.get("topic_continuity_band") or "")
                if str(data.get("topic_continuity_band") or "") in valid_bands
                else ""
            ),
            topic_staleness_band=(
                str(data.get("topic_staleness_band") or "")
                if str(data.get("topic_staleness_band") or "") in valid_bands
                else ""
            ),
            user_engagement_band=(
                str(data.get("user_engagement_band") or "")
                if str(data.get("user_engagement_band") or "") in valid_bands
                else ""
            ),
            dominant_topic_keywords=keywords,
            avoid_recent_topics=avoid,
        )


@dataclass(frozen=True)
class ProactiveIntentSignals:
    """Deterministic conversation signals for intent selection (no LLM calls).

    All scores are bounded to 0..1; booleans are plain flags.  Signals are
    computed on demand from current chat state and never persisted.
    """

    has_useful_memory: bool = False
    has_recent_context: bool = False
    unfinished_topic: bool = False
    # Rich heuristic signals (bounded 0..1 unless noted)
    recent_user_engagement: float = 0.0
    topic_continuity_score: float = 0.0
    topic_staleness_score: float = 0.0
    topic_repetition_score: float = 0.0
    user_question_pending: bool = False
    assistant_question_pending: bool = False
    recent_topic_closed: bool = False
    user_topic_change_detected: bool = False
    recent_user_message_length: float = 0.0
    recent_user_question_rate: float = 0.0
    recent_user_response_rate: float = 0.0
    recent_proactive_question_rate: float = 0.0
    recent_new_topic_rate: float = 0.0
    memory_relevance_score: float = 0.0
    conversation_energy: float = 0.0
    silence_reaction_recently_used: bool = False
    relationship_familiarity: float = 0.0
    # Keyword tuples (internal only; never logged raw or shown to the user)
    recent_topic_keywords: tuple = ()
    dominant_recent_topic: tuple = ()


# Tuning constants for the heuristic layer (kept internal on purpose so the
# public config stays small; see LATENCY/proactive docs).
_ENGAGEMENT_LENGTH_NORM = 100.0
_TOPIC_WINDOW = 8
_PROACTIVE_RATE_WINDOW = 5
_SIGNATURE_KEEP = 5
_CONTINUITY_HIGH = 0.6
_ENGAGEMENT_HIGH = 0.7
_STALENESS_HIGH = 0.7
_REPETITION_HIGH = 0.6
_MEMORY_RELEVANT = 0.6
_MEMORY_IRRELEVANT = 0.2
_FAMILIARITY_CLOSE = 0.67


def compute_intent_signals(
    history: list,
    memory_texts: list,
    *,
    consecutive_ignored_proactive: int = 0,
    recent_proactive_intents=(),
    recent_proactive_topic_signatures=(),
    relationship_familiarity: float = 0.0,
) -> ProactiveIntentSignals:
    """Derive rich deterministic signals from existing chat state.

    ``history`` is the current in-memory conversation (list of dicts with
    ``role``/``content``); ``memory_texts`` are long-term memory strings.
    Pure Python, bounded windows, no model/provider calls.
    """
    conversation = [
        (str(item.get("role") or ""), str(item.get("content") or ""))
        for item in history or ()
        if isinstance(item, dict)
        and item.get("role") in ("user", "assistant")
        and str(item.get("content") or "").strip()
    ]
    recent = conversation[-_TOPIC_WINDOW:]
    recent_user = [content for role, content in recent if role == "user"][-4:]

    has_recent_context = len(conversation) >= 2
    recent_user_message_length = (
        sum(len(text) for text in recent_user) / len(recent_user)
        if recent_user
        else 0.0
    )
    recent_user_question_rate = (
        sum(1 for text in recent_user if "?" in text) / len(recent_user)
        if recent_user
        else 0.0
    )

    if recent_user:
        length_score = clamp01(recent_user_message_length / _ENGAGEMENT_LENGTH_NORM)
        low_info_ratio = sum(
            1 for text in recent_user if _is_low_info_reply(text)
        ) / len(recent_user)
        recent_user_engagement = clamp01(
            0.45 * length_score
            + 0.35 * (1.0 - low_info_ratio)
            + 0.20 * recent_user_question_rate
        )
    else:
        recent_user_engagement = 0.0

    recent_contents = [content for _, content in recent]
    dominant_recent_topic = topic_signature(recent_contents, max_terms=4)
    recent_topic_keywords = topic_signature(recent_contents, max_terms=6)

    # Topic continuity: overlap between the earlier and latter half of the
    # recent window (same-subject conversations repeat meaningful keywords).
    topic_continuity_score = 0.0
    if len(recent) >= 4:
        mid = len(recent) // 2
        earlier_tokens = set()
        for _, content in recent[:mid]:
            earlier_tokens.update(tokenize_for_topic(content))
        latter_tokens = set()
        for _, content in recent[mid:]:
            latter_tokens.update(tokenize_for_topic(content))
        topic_continuity_score = clamp01(
            _term_set_overlap(earlier_tokens, latter_tokens)
        )

    last_role, last_content = conversation[-1] if conversation else ("", "")
    assistant_question_pending = bool(
        last_role == "assistant" and message_expects_response(last_content)
    )
    unfinished_topic = assistant_question_pending
    user_question_pending = bool(
        last_role == "user" and message_expects_response(last_content)
    )

    last_user_text = recent_user[-1] if recent_user else ""
    recent_topic_closed = bool(
        last_user_text
        and _is_closure_like(last_user_text)
        and not user_question_pending
        and not assistant_question_pending
    )

    # Explicit transition markers or a sharp lexical drop in the newest user
    # message indicate the user changed subject.
    user_topic_change_detected = False
    if last_user_text:
        normalized_last = _normalize_text(last_user_text)
        marker_hit = any(marker in normalized_last for marker in _TRANSITION_MARKERS)
        previous_texts = [content for _, content in conversation[:-1]][-4:]
        last_tokens = tokenize_for_topic(last_user_text)
        drop_hit = (
            len(last_tokens) >= 2
            and len(_normalize_text(last_user_text)) >= 12
            and previous_texts
            and topic_similarity([last_user_text], previous_texts) < 0.12
        )
        user_topic_change_detected = bool(marker_hit or drop_hit)

    intents_recent = list(recent_proactive_intents)[-_PROACTIVE_RATE_WINDOW:]
    question_intents = {
        ProactiveIntent.ASK_USER_SOMETHING,
        ProactiveIntent.REACT_TO_IGNORED_QUESTION,
    }
    recent_proactive_question_rate = (
        sum(1 for intent in intents_recent if intent in question_intents)
        / len(intents_recent)
        if intents_recent
        else 0.0
    )
    recent_new_topic_rate = (
        sum(1 for intent in intents_recent if intent == ProactiveIntent.START_NEW_TOPIC)
        / len(intents_recent)
        if intents_recent
        else 0.0
    )
    silence_reaction_recently_used = (
        ProactiveIntent.REACT_TO_SILENCE in list(recent_proactive_intents)[-3:]
    )

    dominant_set = set(dominant_recent_topic)
    topic_repetition_score = 0.0
    for signature in recent_proactive_topic_signatures or ():
        score = signature_similarity(dominant_recent_topic, tuple(signature))
        topic_repetition_score = max(topic_repetition_score, score)

    turns_on_topic = 0
    for _, content in reversed(recent):
        tokens = set(tokenize_for_topic(content))
        if tokens & dominant_set:
            turns_on_topic += 1
        else:
            break
    topic_staleness_score = clamp01(
        0.45 * clamp01(turns_on_topic / 6.0)
        + 0.35 * topic_repetition_score
        + 0.20 * (1.0 - recent_user_engagement)
    )

    user_count_recent = sum(1 for role, _ in recent if role == "user")
    followup_pairs = sum(
        1
        for index, (role, _) in enumerate(recent[:-1])
        if role == "user" and recent[index + 1][0] == "assistant"
    )
    recent_user_response_rate = (
        followup_pairs / user_count_recent if user_count_recent else 0.0
    )

    conversation_energy = clamp01(
        0.5 * recent_user_engagement
        + 0.3 * recent_user_question_rate
        + 0.2 * clamp01(len(recent) / _TOPIC_WINDOW)
    )

    context_tokens = set(dominant_recent_topic)
    for text in recent_user[-2:]:
        context_tokens.update(tokenize_for_topic(text))
    memory_relevance_score = 0.0
    for memory_text in memory_texts or ():
        memory_tokens = set(tokenize_for_topic(str(memory_text)))
        memory_relevance_score = max(
            memory_relevance_score,
            clamp01(_term_set_overlap(memory_tokens, context_tokens)),
        )

    return ProactiveIntentSignals(
        has_useful_memory=bool(memory_texts),
        has_recent_context=has_recent_context,
        unfinished_topic=unfinished_topic,
        recent_user_engagement=recent_user_engagement,
        topic_continuity_score=topic_continuity_score,
        topic_staleness_score=topic_staleness_score,
        topic_repetition_score=topic_repetition_score,
        user_question_pending=user_question_pending,
        assistant_question_pending=assistant_question_pending,
        recent_topic_closed=recent_topic_closed,
        user_topic_change_detected=user_topic_change_detected,
        recent_user_message_length=clamp01(
            recent_user_message_length / _ENGAGEMENT_LENGTH_NORM
        ),
        recent_user_question_rate=recent_user_question_rate,
        recent_user_response_rate=recent_user_response_rate,
        recent_proactive_question_rate=recent_proactive_question_rate,
        recent_new_topic_rate=recent_new_topic_rate,
        memory_relevance_score=memory_relevance_score,
        conversation_energy=conversation_energy,
        silence_reaction_recently_used=silence_reaction_recently_used,
        relationship_familiarity=clamp01(relationship_familiarity),
        recent_topic_keywords=recent_topic_keywords,
        dominant_recent_topic=dominant_recent_topic,
    )


@dataclass
class ProactiveRuntimeState:
    """Per-connection, per-chat state; never persisted."""

    history_uid: str
    last_user_activity_monotonic: float
    next_proactive_eligible_at: float
    last_proactive_monotonic: Optional[float] = None
    consecutive_ignored_proactive: int = 0
    proactive_generation_in_progress: bool = False
    activity_revision: int = 0
    # Ephemeral anti-repetition memory of recent intents (not persisted).
    recent_proactive_intents: List[str] = field(default_factory=list)
    # Ephemeral topic-level anti-repetition (keyword signatures, capped).
    recent_proactive_topic_signatures: List[tuple] = field(default_factory=list)
    # Text and expected-reply flag of the most recent successfully sent
    # proactive message.  Cleared whenever the user responds.
    last_proactive_text: Optional[str] = None
    last_proactive_expected_response: bool = False


def format_followup_instruction(
    context: Optional[ProactiveFollowupContext],
) -> Optional[str]:
    """Internal system-prompt block describing a possibly ignored proactive turn.

    Returns ``None`` when there is nothing to react to.  The wording stays
    persona-neutral and never reveals counters, timers, or system mechanics
    to the user; it only steers the next proactive generation.
    """
    if context is None or not context.previous_proactive_ignored:
        return None

    count = max(1, context.consecutive_ignored)
    if context.previous_proactive_expected_response:
        expectation = "yes -- it asked the user a direct question"
        priority = (
            "your previous message asked the user something and they never "
            "answered; strongly prefer reacting to the unanswered question "
            "first -- notice the silence with mild irritation, confusion, "
            "teasing, embarrassment, or a short complaint, escalating "
            "naturally with each ignored follow-up (first: mild confusion or "
            "teasing; second: more impatient or annoyed; third or later: "
            "resigned, sulking, briefly giving up, or changing topic); never "
            "repeat the exact same question"
        )
    else:
        expectation = "no -- it was a statement, not a question"
        priority = (
            "your previous proactive message was a statement, not a question; "
            "do NOT claim the user failed to answer a question -- you may "
            "notice the silence more generally, tease lightly, or naturally "
            "move to something else"
        )

    return (
        "Internal follow-up context for this turn only. Never shown to the "
        "user; never mention counters, timers, idle detection, or proactive "
        "behavior:\n"
        "- Your previous proactive message was ignored by the user (no reply "
        "so far).\n"
        f"- Consecutive proactive messages ignored: {count}.\n"
        f"- That message expected a reply: {expectation}.\n"
        f"Priority for this turn: {priority}."
    )


_INTENT_GUIDANCE: Dict[str, str] = {
    ProactiveIntent.REACT_TO_IGNORED_QUESTION: (
        "Follow the unanswered-question priority: react naturally to being "
        "ignored after asking something."
    ),
    ProactiveIntent.REACT_TO_SILENCE: (
        "The user went quiet with no question pending; notice it lightly at "
        "most and never demand a reply or reuse stock silence phrases."
    ),
    ProactiveIntent.CONTINUE_PREVIOUS_TOPIC: (
        "Continue something from the recent conversation naturally, as if "
        "the thought just came back."
    ),
    ProactiveIntent.START_NEW_TOPIC: (
        "Bring up a subject of your own choosing (curiosity, hypotheticals, "
        "daily life, games, stories, movies, tech, food, places, or the "
        "user's known interests); just start talking, never announce a "
        "topic or offer options."
    ),
    ProactiveIntent.ASK_USER_SOMETHING: (
        "Ask something you are genuinely curious about, spontaneously."
    ),
    ProactiveIntent.BRING_UP_MEMORY: (
        "Naturally recall something you already know about the user; never "
        "expose memory metadata, IDs, or internal terminology."
    ),
    ProactiveIntent.CASUAL_OBSERVATION: (
        "A short thought or remark that does not require a reply."
    ),
}

_INTENT_STYLE_GUIDANCE = (
    "You are initiating by your own choice: 1-3 short sentences of "
    "conversational Indonesian; a question is optional. Never claim "
    "specific past personal events (finishing a book or movie, going "
    "somewhere, what a friend said) unless the conversation supports them; "
    "saying you were just thinking about something or remember discussing "
    "something is fine."
)


def band_for(value: float, mid: float, high: float) -> str:
    """Map a 0..1 score to a compact low/medium/high band."""
    if value >= high:
        return "high"
    if value >= mid:
        return "medium"
    return "low"


def format_intent_instruction(
    context: Optional[ProactiveIntentContext],
    *,
    include_guidance: bool = True,
) -> Optional[str]:
    """Internal system-prompt block steering this proactive turn's intent.

    Returns ``None`` when there is no intent context.  The block is internal
    only: never shown verbatim to the user and never persisted to history.
    When ``include_guidance`` is False (e.g. the ignored-question follow-up
    block already carries the turn's instructions), only the compact context
    lines are emitted to avoid duplicating guidance tokens.
    """
    if context is None:
        return None

    intent = context.intent
    if intent not in _INTENT_GUIDANCE:
        intent = ProactiveIntent.CASUAL_OBSERVATION
    lines = [
        "Internal proactive context for this turn only. Never shown to the "
        "user; never mention intents, counters, timers, or system "
        "mechanics.",
        f"intent: {intent}",
        "user_has_replied_since_last_proactive: "
        f"{str(context.user_has_replied_since_last_proactive).lower()}",
        f"consecutive_ignored: {context.consecutive_ignored}",
        "recent_silence_acknowledgment: "
        f"{str(context.recent_silence_acknowledgment).lower()}",
    ]
    if include_guidance:
        # Compact mode (ignored-question turns) stays minimal: the follow-up
        # block already carries the turn's instructions.
        lines.extend(
            [
                f"topic_continuity: {context.topic_continuity_band or 'low'}",
                f"topic_staleness: {context.topic_staleness_band or 'low'}",
                f"user_engagement: {context.user_engagement_band or 'low'}",
            ]
        )
        if context.dominant_topic_keywords:
            lines.append(
                "dominant_topic_keywords: "
                + ", ".join(context.dominant_topic_keywords[:4])
            )
        if context.avoid_recent_topics:
            lines.append(
                "avoid_recent_topics: "
                + " | ".join(
                    " ".join(signature[:4])
                    for signature in context.avoid_recent_topics[:3]
                )
            )
        lines.extend(
            [
                f"Intent for this turn: {_INTENT_GUIDANCE[intent]}",
                _INTENT_STYLE_GUIDANCE,
            ]
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class ProactiveIntentDecision:
    """Compact, non-sensitive selector output for tests/debugging only.

    Never sent to the frontend and never persisted into history.  Contains
    only enums/numbers/keyword tuples -- no chat text, no memory contents.
    """

    intent: str
    reason: str
    effective_weights: Dict[str, float]
    signals: ProactiveIntentSignals


def _decision_reason(signals: ProactiveIntentSignals, intent: str) -> str:
    if signals.assistant_question_pending or signals.unfinished_topic:
        return "unfinished_topic"
    if signals.recent_topic_closed:
        return "topic_closed"
    if signals.topic_staleness_score > _STALENESS_HIGH:
        return "topic_stale"
    if signals.topic_continuity_score > _CONTINUITY_HIGH:
        return "high_topic_continuity"
    if (
        intent == ProactiveIntent.BRING_UP_MEMORY
        and signals.memory_relevance_score >= _MEMORY_RELEVANT
    ):
        return "memory_relevant"
    if signals.user_topic_change_detected:
        return "user_topic_change"
    return "weighted_default"


def resolve_proactive_intent_decision(
    followup_context: Optional[ProactiveFollowupContext],
    state: ProactiveRuntimeState,
    machine: "ProactiveStateMachine",
    signals: ProactiveIntentSignals,
    *,
    random: Optional[Callable[[], float]] = None,
) -> ProactiveIntentDecision:
    """Resolve this turn's intent with a compact explainable decision."""
    weights = machine.effective_intent_weights(state, signals)
    if (
        followup_context is not None
        and followup_context.previous_proactive_ignored
        and followup_context.previous_proactive_expected_response
    ):
        return ProactiveIntentDecision(
            intent=ProactiveIntent.REACT_TO_IGNORED_QUESTION,
            reason="ignored_question_priority",
            effective_weights=weights,
            signals=signals,
        )
    intent = _pick_intent(weights, (random or machine._random)())
    return ProactiveIntentDecision(
        intent=intent,
        reason=_decision_reason(signals, intent),
        effective_weights=weights,
        signals=signals,
    )


def resolve_proactive_intent(
    followup_context: Optional[ProactiveFollowupContext],
    state: ProactiveRuntimeState,
    machine: "ProactiveStateMachine",
    signals: ProactiveIntentSignals,
    *,
    random: Optional[Callable[[], float]] = None,
) -> str:
    """Pick this turn's intent; an unanswered proactive question wins."""
    return resolve_proactive_intent_decision(
        followup_context, state, machine, signals, random=random
    ).intent


def _pick_intent(weights: Mapping[str, float], spin: float) -> str:
    """Deterministic weighted wheel over a fixed intent order."""
    ordered = [
        (key, float(weights[key]))
        for key in INTENT_SELECTION_ORDER
        if float(weights.get(key, 0.0)) > 0.0
    ]
    if not ordered:
        return ProactiveIntent.CASUAL_OBSERVATION
    total = sum(weight for _, weight in ordered)
    cursor = spin * total
    for key, weight in ordered:
        cursor -= weight
        if cursor < 0:
            return key
    return ordered[-1][0]


class ProactiveStateMachine:
    """Pure timing/state transitions with injectable clock and randomness."""

    def __init__(
        self,
        config: ProactiveChatConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        randint: Callable[[int, int], int] = random.randint,
        random: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self._monotonic = monotonic
        self._randint = randint
        self._random = random

    def _delay(self, minimum: int, maximum: int) -> float:
        return float(self._randint(minimum, maximum))

    def new_state(self, history_uid: str) -> ProactiveRuntimeState:
        now = self._monotonic()
        return ProactiveRuntimeState(
            history_uid=history_uid,
            last_user_activity_monotonic=now,
            next_proactive_eligible_at=now
            + self._delay(
                self.config.initial_idle_min_seconds,
                self.config.initial_idle_max_seconds,
            ),
        )

    def record_user_activity(self, state: ProactiveRuntimeState) -> None:
        now = self._monotonic()
        state.activity_revision += 1
        state.last_user_activity_monotonic = now
        state.consecutive_ignored_proactive = 0
        state.proactive_generation_in_progress = False
        state.last_proactive_text = None
        state.last_proactive_expected_response = False
        state.next_proactive_eligible_at = now + self._delay(
            self.config.initial_idle_min_seconds,
            self.config.initial_idle_max_seconds,
        )

    def record_proactive_sent(
        self,
        state: ProactiveRuntimeState,
        *,
        response_text: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> None:
        now = self._monotonic()
        state.last_proactive_monotonic = now
        state.consecutive_ignored_proactive += 1
        state.proactive_generation_in_progress = False
        state.last_proactive_text = response_text or None
        state.last_proactive_expected_response = message_expects_response(response_text)
        if intent:
            state.recent_proactive_intents.append(intent)
            del state.recent_proactive_intents[:-3]
        if response_text:
            signature = topic_signature([response_text], max_terms=4)
            if signature:
                state.recent_proactive_topic_signatures.append(signature)
                del state.recent_proactive_topic_signatures[:-_SIGNATURE_KEEP]
        if state.consecutive_ignored_proactive >= self.config.ignored_before_backoff:
            minimum = self.config.backoff_min_seconds
            maximum = self.config.backoff_max_seconds
        else:
            minimum = self.config.followup_idle_min_seconds
            maximum = self.config.followup_idle_max_seconds
        state.next_proactive_eligible_at = now + self._delay(minimum, maximum)

    def proactive_followup_context(
        self, state: ProactiveRuntimeState
    ) -> ProactiveFollowupContext:
        """Snapshot whether the last proactive message went unanswered.

        ``consecutive_ignored_proactive`` counts successful proactive sends
        since the last user activity, so a value >= 1 means the previous
        assistant message was proactive and the user has not replied since.
        """
        ignored = state.consecutive_ignored_proactive >= 1
        return ProactiveFollowupContext(
            previous_proactive_ignored=ignored,
            consecutive_ignored=max(0, state.consecutive_ignored_proactive),
            previous_proactive_expected_response=(
                ignored and state.last_proactive_expected_response
            ),
        )

    def select_proactive_intent(
        self,
        state: ProactiveRuntimeState,
        signals: ProactiveIntentSignals,
        *,
        random: Optional[Callable[[], float]] = None,
    ) -> str:
        """Weighted, anti-repetitive local intent pick (no LLM involved)."""
        return _pick_intent(
            self.effective_intent_weights(state, signals),
            (random or self._random)(),
        )

    def effective_intent_weights(
        self,
        state: ProactiveRuntimeState,
        signals: ProactiveIntentSignals,
    ) -> Dict[str, float]:
        """Resolve context-aware, anti-repetition-adjusted intent weights.

        Base weights stay configurable; context modifiers are internal
        constants.  All returned weights are >= 0.
        """
        merged = dict(DEFAULT_INTENT_WEIGHTS)
        if self.config.intent_weights:
            for key, value in self.config.intent_weights.items():
                if key in merged:
                    merged[key] = float(value)

        if not signals.has_useful_memory:
            merged[ProactiveIntent.BRING_UP_MEMORY] = 0.0
        elif signals.memory_relevance_score >= _MEMORY_RELEVANT:
            merged[ProactiveIntent.BRING_UP_MEMORY] *= 1.5
        elif signals.memory_relevance_score < _MEMORY_IRRELEVANT:
            merged[ProactiveIntent.BRING_UP_MEMORY] *= 0.5

        if not signals.has_recent_context:
            merged[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC] = 0.0
            for key in (
                ProactiveIntent.START_NEW_TOPIC,
                ProactiveIntent.ASK_USER_SOMETHING,
                ProactiveIntent.CASUAL_OBSERVATION,
            ):
                merged[key] *= 1.5
        else:
            if signals.unfinished_topic:
                merged[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC] *= 2.0
            if (
                signals.recent_user_engagement > _ENGAGEMENT_HIGH
                and signals.topic_continuity_score > _CONTINUITY_HIGH
            ):
                merged[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC] *= 1.5
            if signals.topic_staleness_score > _STALENESS_HIGH:
                merged[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC] *= 0.25
                merged[ProactiveIntent.START_NEW_TOPIC] *= 1.5
                merged[ProactiveIntent.CASUAL_OBSERVATION] *= 1.25
            if signals.recent_topic_closed:
                merged[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC] *= 0.25
                merged[ProactiveIntent.START_NEW_TOPIC] *= 1.5
            if signals.user_topic_change_detected:
                merged[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC] *= 0.3
                merged[ProactiveIntent.START_NEW_TOPIC] *= 1.3
            if signals.user_question_pending:
                # The user asked something unresolved; do not start an
                # unrelated subject.
                merged[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC] *= 1.5
                merged[ProactiveIntent.START_NEW_TOPIC] *= 0.3
            if signals.topic_repetition_score > _REPETITION_HIGH:
                # Proactive turns keep landing on the same subject even when
                # the intent differs; make continuation much less likely.
                merged[ProactiveIntent.CONTINUE_PREVIOUS_TOPIC] *= 0.2

        if signals.silence_reaction_recently_used:
            merged[ProactiveIntent.REACT_TO_SILENCE] *= _SILENCE_REPEAT_PENALTY

        if signals.relationship_familiarity >= _FAMILIARITY_CLOSE:
            merged[ProactiveIntent.ASK_USER_SOMETHING] *= 1.15
            merged[ProactiveIntent.BRING_UP_MEMORY] *= 1.1
            merged[ProactiveIntent.CASUAL_OBSERVATION] *= 1.1

        # Intent-level anti-repetition (independent of topic repetition).
        recent = state.recent_proactive_intents[-3:]
        for key in merged:
            occurrences = recent.count(key)
            if occurrences:
                penalty = (
                    _SILENCE_REPEAT_PENALTY
                    if key == ProactiveIntent.REACT_TO_SILENCE
                    else _INTENT_REPEAT_PENALTY
                )
                merged[key] *= penalty**occurrences

        return {key: max(0.0, float(value)) for key, value in merged.items()}

    def seconds_until_eligible(self, state: ProactiveRuntimeState) -> float:
        return max(0.0, state.next_proactive_eligible_at - self._monotonic())

    def is_eligible(self, state: ProactiveRuntimeState) -> bool:
        return (
            self.config.enabled
            and not state.proactive_generation_in_progress
            and self.seconds_until_eligible(state) <= 0
        )
