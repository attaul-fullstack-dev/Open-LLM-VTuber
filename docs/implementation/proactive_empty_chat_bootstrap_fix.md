# Proactive Empty-Chat Bootstrap Fix

**Status: PROACTIVE EMPTY-CHAT BOOTSTRAP FIX SELESAI**

## 1. Confirmed root cause

On a brand-new chat with zero real conversation messages, the proactive
scheduler fires correctly (`request_origin=proactive`, `strategy=semantic_auto`)
but the provider payload contains **only a system message and zero
conversational turns**. Ollama Cloud / `gemma4:31b-cloud` completes the stream
without producing any usable assistant token, so the backend reports
`request_outcome=empty_response`. The frontend "Thinking" flash is a symptom of
the backend producing no tokens.

## 2. Why message_count was 0

Trace: `_run_proactive_timer` → `process_single_conversation` (metadata
`request_origin=proactive`) → `BasicMemoryAgent.chat_proactively()` →
`_proactive_chat_function_factory`:

- `messages = self._memory.copy()` — empty chat ⇒ `[]`
- `_prepare_context_with_summary([], system, protected_start=0)` → context
  selection returns zero history messages ⇒ `tracker.message_count = 0`
- `chat_completion(request_messages, system)` builds
  `[{"role":"system",...}, *request_messages]` ⇒ provider sees system-only

## 3. Provider request shape before fix

```
messages = [ {role: system, content: persona + relationship + memory + semantic instruction + speaking rules} ]
```

Zero user/assistant turns. Provider returns an empty completion.

## 4. Provider request shape after fix (empty proactive chat only)

```
messages = [
  {role: system, content: <unchanged persona/relationship/memory/semantic context>},
  {role: user,   content: "Initiate a natural conversation with the user now."}   // request-only
]
```

Then Mili's natural assistant message streams back exactly as any other turn.

## 5. Bootstrap trigger wording/category

Category: short, model-neutral, request-only conversational cue. No internal
terminology, no topic instruction (semantic_auto already decides direction), no
persona name hard-coded (the persona prompt already establishes identity).

Wording: `"Initiate a natural conversation with the user now."`
Constant: `PROACTIVE_EMPTY_CHAT_BOOTSTRAP` in `basic_memory_agent.py`.

## 6. Exact bootstrap condition

Applied only when **all** hold:

- the request is a proactive turn (the code lives inside
  `proactive_with_memory`, which is only reachable via `chat_proactively()`), and
- the active chat transcript has **zero real dialogue messages** — no
  `user`/`assistant` entry with non-empty content (`_has_real_dialogue`).

Global context (persona, relationship, long-term character memory, summary) is
allowed and does NOT make the chat "non-empty" — it lives in the system prompt.

## 7. Where bootstrap is injected

In `basic_memory_agent.py`, `_proactive_chat_function_factory` →
`proactive_with_memory`, **after** `_prepare_context_with_summary(...)` returns
`request_messages` and **before** `self._llm.chat_completion(...)`. It is
appended to the local `request_messages` list only.

## 8. Proof it is request-only

- The append targets the local `request_messages` copy; `messages` (the
  `_memory` snapshot) is never mutated.
- It happens after context selection, summary update, and `tracker.add_context`
  — so it never enters `_select_context`, `_maybe_update_summary`, summary text,
  or the latency context stats.
- `_add_message` is only ever called with the generated assistant response.

## 9. Proof no fake user history is created

- `_memory` after an empty-chat proactive generation contains exactly one
  entry: `{role: assistant, content: <Mili's message>}`.
- Disk history (`get_history`) is untouched by generation (persistence happens
  later in `single_conversation` for the real assistant output only).
- Regression test `test_bootstrap_not_persisted` and
  `test_no_fake_user_anywhere` assert the trigger appears nowhere: not in
  `_memory`, history, summary state, or any provider message before the last.

## 10. semantic_auto compatibility

Unchanged. When the websocket handler supplies the semantic intent context, the
`<semantic_proactive_context>` block is still appended to the system prompt; the
bootstrap adds no topic guidance. Test
`test_semantic_auto_still_used_on_empty` asserts `strategy: semantic_auto`
remains present.

## 11. Memory compatibility

Long-term character memory is untouched. It stays system-prompt-only and
optional; the bootstrap does not force any fact. Test
`test_global_memory_available_on_empty` asserts the memory text still reaches
the system prompt on an empty chat.

## 12. Relationship compatibility

Relationship state is untouched, remains global, and is not reset for a new
chat. Test `test_relationship_available_on_empty` asserts it stays in the
effective prompt.

## 13. Multiple-chat isolation

Each chat has its own agent/`_memory`. The bootstrap attaches only to the
current request's `request_messages`. Test
`test_multiple_chat_isolation` asserts chat A's transcript never appears in
chat B's provider request, B receives the bootstrap, and B's generated output is
stored only in B.

## 14. History behavior

After a successful first proactive generation, Mili's assistant message flows
through the existing pipeline (`_add_message` → `_memory`, then
`single_conversation` persists it to history). A later user reply sees it as
normal recent context (test `test_user_reply_sees_proactive_message`).

## 15. Provider-call count

Exactly **1** provider call per proactive response — for empty chats, normal
chats, and forced ignored-question follow-ups (test
`test_provider_call_count_is_one`). No retry was introduced.

## 16. Empty-response diagnostic behavior

Unchanged. If the provider genuinely returns an empty stream even with the
bootstrap, no assistant text is fabricated: `complete_response` stays empty and
the outcome remains `empty_response` (test `test_empty_stream_still_empty` with
`_EmptyLLM`).

## 17. Latency/logging changes

- One `logger.info` line per empty-chat bootstrap:
  `Proactive generation: empty_chat_proactive_bootstrap=True, request_message_count_after=N`.
- `tracker.message_count` is set to the final provider message count when a
  latency tracker is active, so the LLM TRACE shows the request includes the
  bootstrap turn.
- Nothing sensitive is logged: no trigger text, persona, memory, relationship,
  or user content.

## 18. Files changed

| File | Change |
|---|---|
| `src/open_llm_vtuber/agent/agents/basic_memory_agent.py` | `PROACTIVE_EMPTY_CHAT_BOOTSTRAP` constant, `_has_real_dialogue()` helper, request-only bootstrap injection in `proactive_with_memory` |
| `tests/test_proactive_chat.py` | `_EmptyLLM`, `_make_empty_agent`, `ProactiveEmptyChatBootstrapTests` (12 tests) |

Frontend, scheduler, conversations, memory, relationship, summary, TTS, Live2D:
**untouched.**

## 19. Tests added

`ProactiveEmptyChatBootstrapTests` (12):

1. `test_empty_chat_injects_single_bootstrap_turn` — exactly one bootstrap turn, call count 1
2. `test_bootstrap_not_persisted` — history contains only assistant output
3. `test_non_empty_proactive_unaffected` — no bootstrap on normal chat
4. `test_global_memory_available_on_empty` — memory still in system prompt
5. `test_relationship_available_on_empty` — relationship still in prompt
6. `test_multiple_chat_isolation` — chat A transcript never leaks into B
7. `test_user_reply_sees_proactive_message` — reply sees proactive context
8. `test_provider_call_count_is_one` — empty / normal / ignored = 1 call each
9. `test_no_fake_user_anywhere` — trigger in none of memory/summary/request
10. `test_empty_stream_still_empty` — genuine empty stream still `empty_response`
11. `test_semantic_auto_still_used_on_empty` — semantic_auto remains active
12. `test_has_real_dialogue_helper` — emptiness detection contract

## 20. Targeted test results

```
python -m unittest tests.test_proactive_chat -q        → 100 tests OK
python -m unittest tests.test_proactive_chat.ProactiveEmptyChatBootstrapTests -v → 12 tests OK
```

## 21. Full regression results

```
PYTHONPATH=src python -m unittest discover -s tests -q → Ran 211 tests, OK
```

## 22. ruff / compileall

```
ruff check (changed files)  → All checks passed!
compileall                  → ok
git diff --check            → ok
```

## 23. Git diff summary

- `basic_memory_agent.py`: +~40 lines (constant, helper, guarded injection + log)
- `test_proactive_chat.py`: +~230 lines (fake LLM, empty-agent helper, 12 tests)

## 24. Commit hash

See commit after push: `fix: bootstrap proactive generation in empty chats`
(backend repo, branch `stage6-final-integration`).

## 25. Manual live-test procedure

Restart the backend so the new code is loaded:

```bash
# kill the old server process and start fresh (or: sudo systemctl restart olv-server.service)
```

Then:

1. Create a **completely new empty chat**.
2. Do not send anything. Wait through the initial proactive idle window (45–90 s).
3. Expected: "Thinking" briefly → **Mili sends a natural first message**.
4. Check logs for this turn:

```
request_origin=proactive
strategy=semantic_auto
request_outcome=success
provider_first_token_received=True
empty_chat_proactive_bootstrap=True   (first empty-chat turn)
```

5. Open chat history: first persisted message role must be **assistant** — no
   synthetic user message.
6. Reply normally; your reply should see Mili's proactive message as context.
7. Repeat with at least 3 brand-new empty chats.

Grep for traces:

```bash
grep -E "empty_chat_proactive_bootstrap|request_outcome=" /path/to/server.log | tail -40
```

## 26. Remaining limitations

- Live verification on the actual Ollama Cloud provider was not possible from
  this environment; unit/integration coverage uses a fake provider and the
  exact request shape is asserted.
- The bootstrap only helps when the conversation is truly empty; a chat whose
  only messages are empty/whitespace entries is also treated as empty.
- `message_count` in the trace reflects the final provider message list for the
  bootstrapped turn (system + bootstrap), not the persisted transcript.

## 27. Unrelated observations

- The provided log also contains `Failed to initialize connection for client
  ...: {}` lines. Static inspection found no path from that connection-init
  failure to the empty proactive response: the proactive failure occurred after
  a successful scheduler fire and provider call. It is a separate concern and
  was intentionally left out of this patch's scope.
