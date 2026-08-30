# Proactive Ignored-Question Patch

**Status: PROACTIVE IGNORED-QUESTION PATCH SELESAI**
**Commit:** `864b279` — `stage6-final-integration` (pushed)

## Problem

When Mili sent a proactive message containing a direct question and the user
stayed silent, the next proactive message tended to simply continue the topic
instead of noticing it was ignored.

Example (before):

> Mili: "Sekarang mau apa biar aktingnya berhenti?" (user silent)
> Next proactive: "Gak usah senyum-senyum gitu..."

Desired: a natural, persona-consistent reaction to being ignored, e.g.
"Lah, ditanya malah diem. Gimana sih?"

## Files Changed

| File | Change |
|---|---|
| `src/open_llm_vtuber/proactive_chat.py` | `message_expects_response()` deterministic detector; `ProactiveFollowupContext` dataclass (+`as_dict`/`from_dict`); new runtime state fields `last_proactive_text`, `last_proactive_expected_response`; `record_proactive_sent(response_text=...)`; `proactive_followup_context()`; `format_followup_instruction()` prompt helper |
| `src/open_llm_vtuber/websocket_handler.py` | Snapshots the followup context into conversation `metadata` before generation; passes the actual `response` text to `record_proactive_sent` |
| `src/open_llm_vtuber/conversations/single_conversation.py` | Forwards `metadata["proactive_followup"]` to `agent.chat_proactively(followup_context=...)` |
| `src/open_llm_vtuber/agent/agents/basic_memory_agent.py` | `chat_proactively(followup_context=...)` and `_proactive_chat_function_factory` append the internal followup block to the effective system prompt for that turn only |
| `tests/test_proactive_chat.py` | +2 test classes, 11 new tests (see Tests) |

## Ignored-Question Detection Strategy

Fully deterministic over already-generated text — **zero extra LLM/provider
calls** (verified by a test asserting `len(llm.calls) == 1` per proactive
turn).

`message_expects_response(text)`:

1. A `?` anywhere in the message → expects a reply.
2. Fallback: a sentence-initial interrogative word (Indonesian + English:
   `apakah, apa, gimana, bagaimana, kenapa, kok, kapan, dimana, di mana,
   siapa, what, why, how, when, where, who`) → expects a reply.
3. Otherwise (pure statement like "Gak usah senyum-senyum gitu...") → does
   not expect a reply.

The detector runs exactly once inside `record_proactive_sent`, on the text
of the proactive message that was actually sent.

## Runtime State Added/Changed

`ProactiveRuntimeState` (per-connection, per-chat, ephemeral — never
persisted to history/metadata/relationship):

| Field | Semantics |
|---|---|
| `consecutive_ignored_proactive` | (existing) successful proactive sends since last user activity |
| `last_proactive_text` | (new) text of the most recent sent proactive message |
| `last_proactive_expected_response` | (new) whether that message asked a question |

`record_user_activity()` (any user message) clears both new fields and the
ignored counter, so normal conversation resumes immediately. Nothing is
inserted into history; no fake user messages are created.

## How It Reaches the Proactive Prompt

```
_run_proactive_timer (websocket_handler)
  └─ machine.proactive_followup_context(state)      # snapshot BEFORE generate
       └─ metadata = {"proactive_followup": {...as_dict()}}
            └─ process_single_conversation
                 └─ single_conversation: forwards to agent.chat_proactively(followup_context=...)
                      └─ basic_memory_agent._proactive_chat_function_factory
                           └─ format_followup_instruction() appended to the
                              effective system prompt for THIS TURN ONLY
```

The block is internal-only; it never enters `_memory`, the persisted
transcript, or anything shown verbatim to the user. Contract:

- **Previous proactive = question, ignored:** strongly prefer reacting to the
  unanswered question — notice the silence with mild irritation / confusion /
  teasing / embarrassment, escalating naturally (1st: mild confusion or
  teasing; 2nd: more impatient; 3rd+: resigned, sulking, giving up, changing
  topic); never repeat the exact same question.
- **Previous proactive = statement, ignored:** do NOT claim the user failed
  to answer a question; may notice silence generally, tease lightly, or move
  on naturally.
- Never mention counters, timers, idle detection, or proactive behavior.

## Not Changed

Scheduler timings, asyncio architecture, history/memory/relationship
architecture, TTS, latency instrumentation, frontend, proactive interval
configuration, model/persona/sampling.

## Tests Added (all PASS)

`ProactiveFollowupStateTests` (pure, no LLM):

1. Detector variants (question mark, sentence-initial interrogative,
   statement, empty, None)
2. Proactive question ignored → context reports `ignored=true`,
   `expected_response=true`, `consecutive_ignored=1`
3. Proactive statement ignored → `expected_response=false` (never falsely
   told a question was ignored)
4. User reply clears ignored state (counter → 0, text cleared)
5. Consecutive ignored count increments 1 → 2 → 3 across sends
6. `as_dict()`/`from_dict()` round trip (+ `None`/invalid input → `None`)

`ProactiveFollowupPromptTests` (agent-level with fake LLM):

7. Ignored question reaches the proactive system prompt (unanswered-question
   priority wording present)
8. Ignored statement prompt contains "was a statement, not a question" +
   "do NOT claim the user failed to answer"; question wording absent
9. No followup context → no internal block in the prompt
10. No extra LLM call for classification (`len(llm.calls) == 1`)
11. `format_followup_instruction` contract (escalation wording, no
   meta-language, `None` when nothing ignored)

Existing scheduler/guard/pipeline tests unchanged and still pass.

## Test Results

| Suite | Result |
|---|---|
| `tests.test_proactive_chat` | 25/25 PASS |
| Full backend suite (`unittest discover`) | 116/116 PASS |
| ruff check (src + tests) | clean |
| compileall (all touched files) | OK |
| `git diff --check` | OK |
| Frontend | not touched (no rebuild needed) |

## Live Verification

Restart the server (it must load this commit), let Mili go idle until a
proactive message with a question fires, stay silent through 2–3 proactive
messages, then check that reactions acknowledge being ignored while staying
in persona.

Deliberately left uncommitted (unrelated to this patch): `model_dict.json`
(runtime Live2D addition) and the `frontend` submodule build artifacts.
