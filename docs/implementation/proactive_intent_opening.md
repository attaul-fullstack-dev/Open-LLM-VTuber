# Proactive Self-Initiated Topic Opening

**Status: PROACTIVE INTENT PATCH SELESAI**

## Problem

Mili only reacted when the user went silent, so repeated proactive messages
tended to sound like "Kok diem?" instead of a person occasionally starting a
conversation herself.

## Files Changed

| File | Change |
|---|---|
| `src/open_llm_vtuber/proactive_chat.py` | `ProactiveIntent` constants; `INTENT_SELECTION_ORDER`; `DEFAULT_INTENT_WEIGHTS`; `ProactiveIntentContext` (+dict round-trip, unknown-intent fallback); `ProactiveIntentSignals`; `ProactiveChatConfig.intent_weights` (optional, validated); `ProactiveRuntimeState.recent_proactive_intents`; injectable `random` on the state machine; `effective_intent_weights()`; `select_proactive_intent()`; `resolve_proactive_intent()`; `format_intent_instruction()` (+compact mode) |
| `src/open_llm_vtuber/config_manager/agent.py` | `proactive_intent_weights` setting (default `None` → safe defaults) + i18n description |
| `src/open_llm_vtuber/websocket_handler.py` | `_proactive_intent_signals()` (local reads, no LLM/logging); intent resolved before generation; `proactive_intent` metadata; intent recorded on send |
| `src/open_llm_vtuber/conversations/single_conversation.py` | Forwards `metadata["proactive_intent"]` to `chat_proactively` |
| `src/open_llm_vtuber/agent/agents/basic_memory_agent.py` | `chat_proactively(followup_context, intent_context)`; injects the internal intent block into the effective system prompt for the turn only; auto-compacts when the ignored-question block is present |
| `prompts/utils/proactive_speak_prompt.txt` | Initiative variety, no topic-announcement, anti-fake-history, question-optional |
| `tests/test_proactive_chat.py` | +2 test classes, 18 new tests |

## Intent Architecture

Seven intents; selection is local and deterministic (injectable RNG) —
**zero extra LLM calls**:

```
resolve_proactive_intent(followup_context, state, machine, signals)
  ├─ ignored unanswered proactive question → react_to_ignored_question (priority)
  └─ otherwise → machine.select_proactive_intent(state, signals)
        = weighted wheel over INTENT_SELECTION_ORDER with
          effective_intent_weights(state, signals)
```

Signals (cheap local reads from the live agent, never logged):
`has_useful_memory` (character memories exist), `has_recent_context`
(≥2 history messages), `unfinished_topic` (last history message is an
assistant message expecting a reply, via the existing
`message_expects_response`).

## Priority Rules

1. `previous_proactive_ignored && previous_proactive_expected_response` →
   **always** `react_to_ignored_question`; random topic selection never
   overrides a pending unanswered question.
2. Ignored statement (no question) → normal weighted selection; the existing
   follow-up block still tells the model not to claim an unanswered question.
3. User reply resets ignored state (existing `record_user_activity`).

## Default Weights (configurable via `proactive_intent_weights`)

| Intent | Weight |
|---|---|
| start_new_topic | 30 |
| continue_previous_topic | 20 |
| ask_user_something | 20 |
| bring_up_memory | 15 |
| casual_observation | 10 |
| react_to_silence | 5 |

Self-initiated intents (start_new_topic + ask_user_something +
bring_up_memory) = 65% of the default wheel; `react_to_ignored_question` is
excluded from the wheel because it is priority-driven.

Context-aware adjustments: no useful memory → `bring_up_memory` = 0; little
recent context → `continue_previous_topic` = 0 and self-initiated intents
×1.5; unfinished topic → `continue_previous_topic` ×2.

## Anti-Repetition

`state.recent_proactive_intents` (ephemeral, capped at 3, never persisted).
Each recent occurrence multiplies the intent's weight by 0.25
(react_to_silence decays faster: ×0.1), so e.g. three consecutive
`start_new_topic` turns reduce its weight to 30×0.25³ ≈ 0.47 and other
intents take over. Empty wheel falls back to `casual_observation`.

## Long-Term Memory Usage

`bring_up_memory` only reuses the **existing** global character-memory
pipeline (memories already flow into the proactive system prompt). Storage,
format, and architecture untouched. The prompt instructs natural recall
("something you already know about the user") and forbids exposing memory
metadata/IDs/internal terminology. Memory intent is simply unavailable
(weight 0) when no memories exist — never forced.

## Fake Personal Experience Prevention

Both the intent block and the proactive speak prompt forbid claiming specific
past personal events (finishing a book/movie, going somewhere, what a friend
said) unless the conversation supports them. Allowed: "just thinking about…",
"feel like talking about…", "remember we talked about…" when context-supported.
No simulated-life/event system implemented (explicitly out of scope).

## Ignored-Question Behavior Preserved

`message_expects_response`, `ProactiveFollowupContext`,
`last_proactive_expected_response`, `last_proactive_text`,
`consecutive_ignored_proactive`, and the escalation wording are all unchanged.
The new layer sits *on top*: priority check first, weighted selection only
when nothing is pending. When both blocks are emitted, the intent block
auto-compacts to context lines only (no duplicated guidance tokens).

## Prompt Context (internal only)

```
Internal proactive context for this turn only. Never shown to the user;
never mention intents, counters, timers, or system mechanics.
intent: start_new_topic
user_has_replied_since_last_proactive: true
consecutive_ignored: 0
recent_silence_acknowledgment: false
Intent for this turn: Bring up a subject of your own choosing ...
You are initiating by your own choice: 1-3 short sentences ...
```

Turn-only, never persisted to history, no fake user messages, never printed
to the user.

## Tests Added (18)

Selection: unanswered-question priority (with adversarial RNG); ignored
statement still uses weighted selection; all six weighted intents selectable
(subTest per intent); repeated intent penalized (30×0.25³); silence
acknowledgment decays fast (×0.1); missing memory disables bring_up_memory;
little context zeroes continue and boosts self-initiated; unfinished topic
boosts continue; recent-intents trimming to 3; config override + unknown-key
ignore + negative-weight ValueError; old config without
`proactive_intent_weights` loads with defaults; intent context dict
round-trip + unknown-intent fallback.

Prompt/contract: intent context reaches the proactive system prompt;
react_to_ignored_question intent reaches the prompt alongside the follow-up
block; anti-fake-history contract present; intent metadata never enters
persisted history; no fake user message + exactly one LLM call;
`format_intent_instruction` contract (full vs compact vs None).

## Test Results

| Suite | Result |
|---|---|
| `tests.test_proactive_chat` | 43/43 PASS |
| Full backend suite | 135/135 PASS |
| ruff check + format (touched files) | clean |
| compileall | OK |
| `git diff --check` | OK |

## Git Diff Summary

8 files, ~420 insertions / ~30 deletions (backend + prompt + tests only;
frontend untouched, no rebuild needed).

## Remaining Limitations

- Intent selection uses lightweight heuristics, not semantic understanding of
  "interesting" topics; `unfinished_topic` only detects an assistant message
  awaiting a reply.
- `bring_up_memory` cannot point at a *specific* memory — the model picks
  from whatever the existing memory pipeline already includes.
- Anti-repetition state is in-memory and resets on server restart (by design).
- No simulated daily-life/event memory; Mili cannot truthfully reference
  invented offline activities (deliberately).
- Token cost: full intent block ≈ 0.3k conservative-token estimate per
  proactive turn (compact when combined with the ignored-question block).
