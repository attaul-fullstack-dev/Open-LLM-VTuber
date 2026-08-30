# Proactive Turn Cue Fix

**Status: PROACTIVE TURN CUE FIX SELESAI**

## 1. Live failure reproduced conceptually

Production log (after the empty-chat bootstrap fix was deployed) showed a
second proactive generation on the *same* silent conversation returning
`request_outcome=empty_response`:

```
17:00:27  first proactive:  empty_chat_proactive_bootstrap=True  message_count=1  request_outcome=success
17:03:19  second proactive: (no bootstrap)                      message_count=1  request_outcome=empty_response
            provider_started=True  provider_headers_received=True
            provider_first_chunk_received=True  provider_first_token_received=False
            provider_stream_completed=True
```

The scheduler fired, semantic selection ran, the provider accepted the request
and completed its stream — but produced **zero assistant tokens**.

## 2. Confirmed root cause

`BasicMemoryAgent.proactive_with_memory()` builds the provider payload as:

```
[system prompt, *request_messages]
```

where `request_messages` comes from `_prepare_context_with_summary()` over the
persisted transcript. Proactive generation is a **system-initiated turn**; the
transcript is always *history* and never contains a *current* user turn. So the
request ends on whatever the last historical message was.

After the first proactive message succeeds, history is
`[assistant: "<Mili first proactive message>"]` — the next proactive request
ends on an **assistant** message with no current user cue. Observed provider
behavior (Ollama Cloud / gemma4:31b-cloud): a request whose final conversational
message is an assistant turn can complete the stream with no assistant token.

The previous patch only injected a request-only user turn when the transcript
had **zero** real dialogue (`_has_real_dialogue(messages) == False`). That
condition is insufficient: a conversation can have assistant history but still
have no *current* generation cue. Bug 1 (empty chat, system-only request) was
only one manifestation of a broader request-shape issue.

## 3. Why the first empty proactive succeeded

Empty chat → `_has_real_dialogue == False` → one request-only user turn
appended → payload `[system, user-cue]` ends on a user turn → provider
generates Mili's first message.

## 4. Why the second proactive failed

After the first assistant message, `_has_real_dialogue == True` → no bootstrap
→ payload `[system, assistant]` ends on an assistant turn → provider completes
empty. The empty-chat patch did not cover assistant-only or normal
user/assistant history.

## 5. Provider payload before fix

```
SYSTEM: persona + relationship + memory + semantic_auto + speaking rules
ASSISTANT: "<Mili first proactive message>"        # last message = assistant
```

## 6. Provider payload after fix

```
SYSTEM: persona + relationship + memory + semantic_auto + speaking rules
ASSISTANT: "<Mili first proactive message>"        # historical, untouched
USER: "Continue the conversation naturally on your own."   # request-only cue, last
```

## 7. Final cue wording

`PROACTIVE_TURN_CUE = "Continue the conversation naturally on your own."`

Generic, short, model-neutral: no topic instruction, no intent label, no timer,
no silence counter, no memory instruction, no persona name, no
ignored-question wording.

## 8. Exact cue injection condition

Unconditional — **every** `chat_proactively()` request appends exactly one cue
after context selection (`_prepare_context_with_summary`) and before the
provider call. `request_origin == proactive` is inherent to the call site
(`proactive_with_memory` is only reachable via `chat_proactively`).

## 9. Applied to all proactive turns — yes

Empty chat, assistant-only history, normal user/assistant history, ignored
statement, and forced ignored-question follow-ups all get the same single cue.

## 10. Why that condition is safe

- Proactive generation is always a new system-initiated turn, so there is
  never a *current* user turn in the transcript; a historical user message is
  never the current request turn (spec TEST 5 verified: history ending in a
  user message still receives the cue, and the historical message is kept).
- The cue is appended **after** context selection, so summary/trimming logic
  and `protected_start` are unaffected.
- It carries no direction of its own — semantic_auto and the ignored-question
  system block remain authoritative for *what* Mili says; the cue only ensures
  the provider *generates* the turn.
- Consecutive same-role messages are accepted by OpenAI-compatible endpoints;
  the cue content is generic so no extra meaning is injected.
- Zero extra provider calls, timers, network, or storage.

## 11. semantic_auto compatibility

Unchanged. `SEMANTIC_PROACTIVE_INSTRUCTION` still decides the natural
direction; the cue adds no topic/intent wording. Verified by
`test_semantic_auto_still_used_on_empty`.

## 12. Forced ignored-question compatibility

Unchanged. The follow-up system block ("...was ignored by the user...") remains
in the system prompt; the cue carries no ignored wording. Verified by
`test_forced_ignored_question_with_cue`.

## 13. Ignored-statement compatibility

The `semantic_ignored_statement` path (which suppresses the follow-up block for
an ignored *statement* in semantic mode) is untouched; the cue is generic and
does not inject ignored-question wording.

## 14. Memory compatibility

Global long-term memory stays system context only; the cue never enters memory
parsing and is not duplicated into the prompt. Verified by
`test_global_memory_available_on_empty`, `test_cue_exactly_once_with_memory`.

## 15. Relationship compatibility

Relationship state stays system context only; the cue never reaches
relationship detection. Verified by `test_relationship_available_on_empty`.

## 16. Summary compatibility

The cue is appended after summary selection; it can never be summarized or
counted as summary source. `test_no_fake_user_anywhere` asserts the cue text
never appears in `_summary_state.text` or persisted history.

## 17. Multiple-chat isolation

The cue is local to the current request list; chat A's transcript cannot leak
into chat B. Verified by `test_multiple_chat_isolation`.

## 18. Request-only proof

The cue is a local list element appended in `proactive_with_memory()` after
`_prepare_context_with_summary()`. It is not written to `_memory` (only the
generated assistant text is), not stored to history JSON, not passed to the
summary pipeline, not passed through relationship detection. `request_messages`
is a local variable; nothing retains it beyond the provider call.

## 19. No fake-user persistence proof

- `_add_message()` is only called with the accumulated assistant response.
- After two consecutive proactive turns, `_memory` roles are
  `["assistant", "assistant"]` with no cue content (verified by
  `test_second_proactive_after_first_assistant`,
  `test_repeated_proactive_turns_never_empty`).
- `get_history()` stays `[]` for a proactive-only session (verified by
  `test_cue_not_persisted`).

## 20. Provider call count

Exactly one provider call per proactive turn — no retry, no classifier, no
planner, no second attempt on empty. Verified by `test_provider_call_count_is_one`
(empty, normal, forced-ignored) and the new `_GemmaLikeLLM` tests.

## 21. Empty-response behavior

A genuinely empty provider stream (even with the cue present) still yields
`request_outcome=empty_response`; no fallback text is fabricated. Verified by
`test_empty_stream_still_empty` with `_EmptyLLM`.

## 22. message_count behavior

`tracker.message_count` is now always set to `len(request_messages)` after the
cue append (final actual provider request message count):
- first proactive on empty chat → `message_count=1` (the cue)
- second proactive after one assistant → `message_count=2` (assistant + cue)
- normal history `[user, assistant]` → `message_count=3` (user, assistant, cue)

## 23. Logging changes

- Replaced `empty_chat_proactive_bootstrap=True` with the accurate
  `proactive_turn_cue=True` (the mechanism is no longer empty-chat-specific).
- `request_message_count_after=N` kept. No chat text, persona, memory,
  relationship, or cue text is logged.

## 24. Files changed

| File | Change |
|---|---|
| `src/open_llm_vtuber/agent/agents/basic_memory_agent.py` | Cue now appended on every proactive turn; constant renamed |
| `tests/test_proactive_chat.py` | Updated existing tests + new regression tests + `_GemmaLikeLLM` provider-shape simulator |

## 25. Constants/helpers renamed or removed

- `PROACTIVE_EMPTY_CHAT_BOOTSTRAP` → `PROACTIVE_TURN_CUE` (new wording).
- `_has_real_dialogue()` removed (confirmed dead after the new implementation).

## 26. Tests added

- `test_second_proactive_after_first_assistant` — the live regression
- `test_repeated_proactive_turns_never_empty` — 3 consecutive turns, none empty
- `test_historical_user_message_not_current_cue` — history ending in user still
  receives one cue; historical message preserved
- `test_forced_ignored_question_with_cue` — forced ignored-question stays
  authoritative with the cue present
- `test_cue_exactly_once_with_memory` — exactly one cue even with memory present
- `_GemmaLikeLLM` — deterministic provider-shaped simulator: empty completion
  when the request ends on an assistant message, content when it ends on the cue
  (reproduces before/after behavior without provider-specific production code)

## 27. Targeted proactive test results

`python -m unittest tests.test_proactive_chat -q` → **104 tests, OK**.

## 28. Full backend regression

`PYTHONPATH=src python -m unittest discover -s tests -q` → **215 tests, OK**
(was 211 before this patch; +4 new tests).

## 29. ruff

`ruff check` on both changed files → clean.

## 30. compileall

`python -m compileall -q` on the changed module → OK.

## 31. git diff check

`git diff --check` → clean.

## 32. Commit hash

`8e0efd7` — `fix: add ephemeral cue to proactive turns` (branch `stage6-final-integration`, pushed to origin).

## 33. Live Android test procedure

After the backend is restarted with this commit:

- **TEST A** — create a completely empty chat, stay silent. First proactive
  must succeed (`request_outcome=success`, `provider_first_token_received=True`,
  `proactive_turn_cue=True`).
- **TEST B** — do NOT reply. Wait for the next proactive eligibility. The
  second proactive (the exact regression) must succeed.
- **TEST C** — keep staying silent; observe at least 3 consecutive proactive
  generations, all succeeding (or normal ignored/backoff behavior).
- **TEST D** — send a normal reply, go silent again; the next proactive after
  normal dialogue must succeed.
- Verify history contains only assistant messages for proactive-only turns
  (no synthetic user entries).

Log command:

```bash
grep -E "proactive_turn_cue|request_outcome=|request_message_count_after" /tmp/server_setsid.log | tail -40
```

## 34. Remaining limitations

- If the provider genuinely returns an empty stream even with the cue present,
  `request_outcome=empty_response` is preserved and surfaced — this is
  intentional (a real provider failure should not be masked).
- The cue is English wording inside the provider request; it is never shown to
  the user and carries no persona voice, so this is invisible in UX.
- Two consecutive user turns appear in the provider payload when the history
  happens to end with an unanswered user message (e.g. interrupted generation);
  OpenAI-compatible endpoints accept this and Ollama merges same-role messages.

## 35. Unrelated observations

- `tests/test_mili_ui_response_polish.py` imports `open_llm_vtuber` directly
  (no `src.` prefix) and only resolves under `PYTHONPATH=src`; unrelated to
  this patch and left untouched.
- Pre-existing untracked/modified files (`model_dict.json`, `character_state/`,
  `cloudflared`) were not touched and not committed.
- The frontend worktree (`/root/waifu/worktrees/mili-stage3-web`), the frontend
  submodule pointer, and the Stage 3 worktree were not modified.
