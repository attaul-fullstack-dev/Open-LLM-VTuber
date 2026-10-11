# Task G — Live chat and history consistency

Status: **AUDIT + 3 FIXES, TESTED, DEPLOYED.**

## Fixes

1. **`store_message` collapsed transcripts (PROVEN, P0, commit `75dc1ed`).**
   A read failure was swallowed and the whole file was overwritten with the
   single new message. Now refuses the write (CRITICAL log) and preserves the
   file. Tests: `tests/test_history_write_safety.py` (3).
2. **Resync dropped `requestId`/`name`/`avatar`/`timestamp` (PROVEN, P0/P1,
   frontend commit `5656290`).** `applyHistoryData` mapped rows through a
   4-field literal, so a mid-stream resync left the truncated bubble on screen
   (canonical final matched nothing) and every loaded AI bubble lost its
   avatar/speaker. All four fields now survive; `ChatRow` documents them;
   re-inserted pending rows keep `requestId`. Tests:
   `tests/history-field-preservation.test.ts` (5).
3. **Quick-mute wedged the turn lifecycle (PROVEN via Task K, frontend commit
   `5656290`).** Fixed in the footer; behavioural + contract tests in
   `tests/audio-mute-queue.test.mjs` (5).

Related suites green: backend `chat_resync_safety` + `history_backup` (15),
frontend resync/reconcile suites (29).

## Verified correct, not touched

Canonical-final idempotency, `request_id` routing, post-final guard, same-tick
append safety, flush gate, atomic history write, human-row reconcile,
`safe_send`, detached-turn resync, empty-interrupt guard, `cleanChatDisplayText`
applied to both live and loaded text.

## Documented, not fixed (need turn-level reproduction first)

Interrupt-vs-persist duplicate AI rows (F4), group `ai-final` absence (F11),
history-list cleanup of in-flight sessions (F12), proactive `skip_history`
user-row asymmetry (F13), sentence-divider spacing (F9), think-tag display
leak (F10), interruption-marker write-only semantics (F8).
