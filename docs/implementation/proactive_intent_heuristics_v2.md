# Proactive Intent Heuristics v2 — Context-Aware Selector

**Status: PROACTIVE INTENT HEURISTICS V2 SELESAI**
**Commit:** see git log (`stage6-final-integration`)

> This is a **richer heuristic approximation**, not semantic understanding.
> The LLM still decides *how* Mili says things; heuristics only decide *what
> kind* of proactive behavior fits. Limitations are listed explicitly at the
> end.

## 1. Files Changed

| File | Change |
|---|---|
| `src/open_llm_vtuber/proactive_chat.py` | Topic text utilities (tokenizer, stopwords, signature, similarity, closure/transition vocab); `ProactiveIntentSignals` extended 3 → 20 fields; `compute_intent_signals()` pure calculator; `ProactiveIntentDecision` + `resolve_proactive_intent_decision()` + `_decision_reason()`; `band_for()`; dynamic modifiers in `effective_intent_weights()`; `ProactiveIntentContext` extended with hint bands/keywords; `format_intent_instruction()` emits compact hints; `ProactiveRuntimeState.recent_proactive_topic_signatures`; `record_proactive_sent()` records topic signature |
| `src/open_llm_vtuber/websocket_handler.py` | `_proactive_intent_signals(context, state)` now builds memory texts, relationship familiarity, and calls `compute_intent_signals`; timer logs one compact safe `[PROACTIVE INTENT]` line and passes hint fields to the intent context |
| `tests/test_proactive_chat.py` | +33 tests (heuristics + dynamic weights); old-config test updated to neutral-relevance expectation |
| `basic_memory_agent.py`, `single_conversation.py`, config, prompts | **untouched** (context flows through the existing metadata path) |

## 2. New Signals (all computed on demand, never persisted)

`recent_user_engagement`, `topic_continuity_score`, `topic_staleness_score`,
`topic_repetition_score`, `user_question_pending`, `assistant_question_pending`,
`recent_topic_closed`, `user_topic_change_detected`, `recent_user_message_length`,
`recent_user_question_rate`, `recent_user_response_rate`,
`recent_proactive_question_rate`, `recent_new_topic_rate`,
`memory_relevance_score`, `conversation_energy`, `silence_reaction_recently_used`,
`relationship_familiarity`, `recent_topic_keywords`, `dominant_recent_topic`
(+ existing `has_useful_memory`, `has_recent_context`, `unfinished_topic`).
All numeric signals bounded to 0..1 (`clamp01`), keyword tuples capped.

## 3. Topic Extraction Strategy

Pure-Python tokenizer: lowercase → strip punctuation → drop tokens <3 chars,
pure digits, and a conservative **Indonesian + English stopword/filler list**
(pronouns, particles, question words, intensifiers, `user` from memory phrasing).
`topic_signature(texts, max_terms)` = deterministic top-N tokens by
count (ties broken alphabetically). `dominant_recent_topic` = top-4 over the
recent window; `recent_topic_keywords` = top-6.

## 4. Topic Similarity Strategy

**Overlap coefficient** `|A∩B| / min(|A|,|B|)` on filtered token sets
(`topic_similarity` for text groups, `signature_similarity` for signatures).
No embeddings, no external NLP dependency. Continuity = 0.5 × overlap between
the earlier/latter halves of the recent 8-message window + 0.5 × dominant-
keyword presence in the newest messages.

## 5. Engagement Scoring

Over the last ≤4 user messages: `0.45×length_score (avg chars / 100)` +
`0.35×(1 − low-info ratio)` + `0.20×question rate`. Low-info = message in the
filler-reply set (`iya`, `gak`, `oke`, `sip`, `hmm`, …) or ≤4 chars. Short
messages are **not** automatically negative — a short but specific reply only
loses the length component.

## 6. Topic Staleness Scoring

`0.45×(consecutive on-topic turns from the end / 6)` + `0.35×topic_repetition`
+ `0.20×(1 − engagement)`, clamped. High staleness cuts continuation ×0.25 and
boosts `start_new_topic` ×1.5, `casual_observation` ×1.25.

## 7. Topic Closure Detection

A closure-like *final user message* (exact phrase such as `yaudah`, `oke
makasih`, `gitu aja`, **or** short message whose every word is closure vocab)
**and** no pending user question **and** no pending assistant question →
`recent_topic_closed`. Single phrases are never an absolute rule: any pending
question or a substantive follow-up cancels closure.

## 8. Topic-Change Detection

`user_topic_change_detected` fires on explicit transition markers
(`ngomong-ngomong`, `btw`, `beda topik`, `ganti topik`, `oh iya`, …) **or** a
lexical drop: newest user message with ≥2 meaningful tokens whose similarity to
the previous window is <0.12. Detected change cuts continuation ×0.3, boosts
`start_new_topic` ×1.3. Older topics cannot overpower an explicit marker.

## 9. Memory Relevance Strategy

Keyword overlap between each long-term memory entry and
(dominant topic ∪ newest user tokens), taking the max entry score. Bands:
no memory → weight 0; relevance ≥0.6 → ×1.5; <0.2 → ×0.5; middle → unchanged.
Memory text never logged, never persisted, storage untouched.

## 10. Dynamic Weighting Rules (all internal constants, final weights ≥0)

Base weights remain configurable via `proactive_intent_weights`. Applied in
order: memory bands → little-context rules (unchanged) → unfinished topic ×2 →
engagement+continuation boost ×1.5 → staleness rules → closure rules →
user-change rules → **user-question-pending (continuation ×1.5,
start-new ×0.3)** → topic repetition >0.6 (continuation ×0.2) →
silence-recently-used (×0.1) → relationship familiarity ≥0.67 (weak:
ask ×1.15, memory ×1.1, casual ×1.1) → intent anti-repetition penalties.

## 11. Intent vs Topic Anti-Repetition

Two independent mechanisms: `recent_proactive_intents` (existing, cap 3,
×0.25/occurrence; silence ×0.1) penalizes repeating the same *behavior*;
`recent_proactive_topic_signatures` (new, cap 5, keyword tuples of each sent
proactive message) penalizes re-landing on the same *subject* — so
`start_new_topic → ask_user_something → casual_observation` about the same
game-horror topic still gets corrected. Signatures also feed
`avoid_recent_topics` in the prompt hints. Both are ephemeral.

## 12. Performance Impact

Pure Python over ≤8 history messages, ≤N memory entries, tiny token sets —
runs **once** per proactive decision (microseconds), zero provider calls, no
daemons/threads, no new dependencies. Prompt cost: hint lines add ~40 bytes in
full mode; compact (ignored-question) mode unchanged.

## 13. Tests Added (33; full list in `tests/test_proactive_chat.py`)

Tokenizer/stopwords, same-topic vs unrelated similarity, signature
determinism, high-continuity conversation, similarity-drop change detection,
transition-marker detection, closure detection + question/followup
cancellation, pending question distinction, engagement bounds + direction,
energy bounds, memory relevance (none/irrelevant/relevant), repetition +
staleness from signatures, proactive rate signals, handler signal builder
(fake agent), ignored-question priority, all dynamic weight modifiers
(engagement/continuity, staleness, closure, change, pending, repetition,
memory bands, silence, familiarity), intent-vs-topic repetition independence,
weights ≥0 under adversarial signals, decision reasons
(`ignored_question_priority`/`topic_stale`/`topic_closed`/`memory_relevant`),
prompt hints full-vs-compact, hint dict round-trip + invalid-value tolerance,
topic signature recorded ephemerally and surviving user replies, band helper.

## 14. Full Test Results

| Suite | Result |
|---|---|
| `tests.test_proactive_chat` | 76/76 PASS |
| Full backend suite | **167/167 PASS** |
| ruff check + format | clean |
| compileall | OK |
| `git diff --check` | OK |

## 15. Git Diff Summary

3 files: `proactive_chat.py` (+~470), `websocket_handler.py` (+~90/−30),
`tests/test_proactive_chat.py` (+~450). No scheduler/timing changes, no
frontend changes, no config-schema changes (backward compatible).

## 16. Remaining Limitations (cases needing genuine semantic understanding)

- **Synonymy/paraphrase**: "film seram" vs "game horror" share no tokens →
  similarity misses related-but-differently-worded topics (and vice versa,
  token overlap can overrate surface-similar but unrelated messages).
- **Implicit topic shifts** without markers or keyword drops (user pivots via
  pronouns: "itu kok gitu sih?").
- **Resolution/answer detection**: whether an assistant reply actually
  *answered* a user question is not semantically verified (reply presence is
  treated as resolution).
- **Sarcasm/engagement nuance**: an enthusiastic "gila sih 😂" counts via
  length only; genuine disengagement dressed in long messages is not detected.
- **Memory relevance** is lexical only — a memory phrased with different words
  than the current topic scores low even if conceptually relevant.
- **Multi-topic messages** blend their keywords into one signature.
- Energy/engagement are conversational-activity measures only — they are not
  emotional signals and must not be read as mood.

## Live Verification

After restarting the server (it must load this commit), chat normally, let
Mili go idle a few times, then:

```bash
grep "\[PROACTIVE INTENT\]" logs/*.log* | tail -20
```

Each line is safe-by-construction (intent enum, reason enum, 0..1 scores —
no chat/memory text).
