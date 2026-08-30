# Mili Hidup — Idle State Core v1

## 1. Files Changed

Frontend source:

- `src/renderer/src/utils/avatar-activity-controller.ts` — deterministic event-driven activity state machine.
- `src/renderer/src/context/avatar-activity-context.tsx` — one React runtime provider and `useAvatarActivityState()` consumer API.
- `src/renderer/src/App.tsx` — mounts exactly one activity provider outside the WebSocket lifecycle.
- `src/renderer/src/hooks/footer/use-text-input.tsx` — marks real user activity only after a text message is successfully sent.
- `src/renderer/src/hooks/utils/use-audio-task.ts` — enters `speaking` on actual audio `playing` and keeps it through the complete multi-segment playback queue.
- `src/renderer/src/hooks/utils/use-interrupt.ts` — treats a local explicit interruption as user activity and clears speaking state through the existing audio cancellation flow.
- `src/renderer/src/components/footer/footer.tsx` — clears speaking override when voice playback is muted/stopped.
- `tests/avatar-activity-controller.test.ts` — fake-clock state-machine coverage.
- `tests/avatar-activity-integration.test.mjs` — frontend integration/no-provider-call contracts.

Backend integration:

- `frontend` submodule pointer — deployable production bundle containing the Stage 1 controller.
- `docs/implementation/mili_hidup_idle_core_v1.md` — this implementation report.

No backend Python, persona, provider, TTS configuration, character state, proactive scheduling, memory, relationship, summary, Live2D parameter, or expression-map code changed.

## 2. Controller Location

The state machine lives in:

`Open-LLM-VTuber-Web/src/renderer/src/utils/avatar-activity-controller.ts`

React exposes it from:

`Open-LLM-VTuber-Web/src/renderer/src/context/avatar-activity-context.tsx`

The provider is mounted once above `WebSocketHandler`, so socket reconnects do not recreate the controller.

## 3. Activity States

- `active` — recent meaningful user interaction.
- `idle` — at least 30 seconds of conversational inactivity.
- `long_idle` — at least 120 seconds total conversational inactivity.
- `speaking` — actual Mili audio playback is active.

Stage 1 does not consume these states for autonomous motion. Mili's visual behavior therefore remains unchanged.

## 4. State Priority

`speaking` overrides the time-derived state while one or more playback tokens are active. When playback ends, the controller derives `active`, `idle`, or `long_idle` from the original last-user-activity timestamp.

Multiple audio tokens are reference-counted and cleared together when the complete response playback queue finishes, preventing state flicker between sequential segments.

## 5. Idle Thresholds

Centralized production defaults:

- `AVATAR_ACTIVITY_THRESHOLDS.idleAfterMs = 30_000`
- `AVATAR_ACTIVITY_THRESHOLDS.longIdleAfterMs = 120_000`

No settings UI or scattered timing values were added.

## 6. What Resets User Activity

Stage 1 resets inactivity when:

- a text chat payload is successfully handed to an open WebSocket;
- the local user explicitly interrupts current speech.

A failed/disconnected send does not reset the timer. Passive socket traffic, mouse movement, touch movement, avatar updates, TTS generation, assistant messages, and proactive messages do not reset it.

Voice-transcription activity is not separately wired in Stage 1; explicit text send is the conservative minimum integration.

## 7. Proactive-Chat Behavior

Proactive assistant generation and playback do not call `markUserActivity()`.

Example:

`idle -> speaking -> idle`

or, after enough total inactivity:

`long_idle -> speaking -> long_idle`

No proactive scheduler timing or ignored-question behavior changed.

## 8. Speaking Integration

The existing audio pipeline remains authoritative. State changes to `speaking` only on the HTML audio element's `playing` event—not when text arrives, TTS finishes synthesis, or audio enters the queue.

The existing `frontend-playback-complete` path clears all response playback tokens only after `audioTaskQueue.waitForCompletion()`. Existing stop/mute/interruption paths clear them immediately. This prevents a tiny `speaking -> idle -> speaking` flicker between sequential TTS segments.

## 9. Interruption Behavior

A local interruption is meaningful direct interaction:

`speaking -> active`

because activity is marked before existing audio cancellation clears the speaking override.

A forwarded interruption (`sendSignal=false`) does not impersonate local user activity; it returns to the state derived from elapsed real-user inactivity. Existing subtitle cancellation remains untouched.

## 10. Timer Architecture

The controller uses one pending event-driven `setTimeout` at a time:

1. schedule the remaining delay to `idle`;
2. after reaching `idle`, schedule the remaining delay to `long_idle`;
3. on real user activity, cancel and recalculate.

There is no `setInterval`, render-loop polling, `requestAnimationFrame` timing, worker, daemon, or backend timer.

## 11. Cleanup Behavior

Provider cleanup unsubscribes state updates, cancels the pending timer, and clears speaking tokens. `start()` is idempotent, preventing duplicate timers during development remounts. Playback tokens are idempotently released.

## 12. Reconnect Behavior

The activity provider owns frontend runtime state outside `WebSocketHandler`. A WebSocket reconnect therefore neither creates another controller nor resets `lastUserActivityAt`, preventing reconnect-induced active/idle flicker.

## 13. CPU/RAM/Network Impact

- CPU while waiting: effectively zero; only one browser timeout is pending.
- RAM: one small controller, one timeout handle, a listener set, and short-lived playback symbols; negligible.
- Network: zero additional traffic.
- LLM/provider calls: zero.

## 14. Tests Added

17 dedicated deterministic checks cover:

1. initial active;
2. active to idle threshold;
3. idle to long-idle threshold;
4. idle reset by user;
5. long-idle reset by user;
6. speaking from active;
7. speaking from idle;
8. speaking completion before idle;
9. speaking completion after idle;
10. speaking completion after long idle;
11. proactive playback does not reset user activity;
12. real user send resets activity;
13. cleanup cancels timers;
14. duplicate start does not duplicate transitions;
15. provider location makes reconnect independent;
16. local interruption behavior;
17. no LLM/provider/network/polling call.

Existing subtitle, reconnect delivery, and memory UI contract tests were also rerun.

## 15. Test Results

| Test | Result |
|---|---|
| Avatar activity fake-clock tests (10) | PASS |
| Avatar activity integration contracts (7) | PASS |
| Subtitle playback regression (5) | PASS |
| Chat delivery/reconnect regression (4) | PASS |
| Memory UI regression (5) | PASS |
| `git diff --check` | PASS |
| Production frontend build | PASS |
| Full TypeScript typecheck | BASELINE FAIL — existing Cubism SDK errors; no Stage 1 file reported |
| ESLint | ENVIRONMENT BLOCKED — existing config requires missing `airbnb` package |

Total executed deterministic tests: 31 passed, 0 failed.

## 16. Production Build Result

`npm run build:web`: PASS

- 2,582 modules transformed.
- Output JavaScript: `assets/main-BGcvd8lC.js`
- Existing ONNX `eval` and large-chunk warnings remain unchanged.

## 17. Git Diff Summary

Frontend source commit changes 9 files with 492 insertions and 19 deletions. The large apparent `App.tsx` deletion/insertion count is provider nesting indentation; there is no application-layout rewrite.

Unrelated pre-existing local Live2D/settings/i18n/WebSocket-handler changes were excluded from the commit.

## 18. Commit Hash

- Frontend source checkpoint: `10ac732` — `feat: add avatar idle activity state core`
- Static deployment bundle: `a739bb1` — `build: publish avatar idle state core`
- Backend integration/report: the commit containing this report and the updated frontend submodule pointer.

## 19. Manual Verification Steps

For temporary development verification only:

1. Edit `AVATAR_ACTIVITY_THRESHOLDS` in `avatar-activity-controller.ts` to:
   - `idleAfterMs: 5_000`
   - `longIdleAfterMs: 15_000`
2. Run `npm run dev:web`.
3. Open browser developer console; development-only transitions appear as `[AVATAR ACTIVITY] old -> new`.
4. Verify start is `active`, 5 seconds becomes `idle`, and 15 seconds total becomes `long_idle`.
5. Send a real message and verify `active`.
6. Let it become idle, trigger normal/proactive speech, verify `speaking`, then verify it returns to the elapsed inactivity-derived state.
7. Interrupt speech and verify local interruption returns to `active`.
8. Restore 30,000/120,000 before committing or building production.

No permanent debug badge is shown in normal UI.

## 20. Limitations

- Stage 1 intentionally has no autonomous head, eye, body, expression, or motion behavior.
- Text sends and explicit local interruption are the currently wired direct activity sources; a future checkpoint may mark completed voice-input submission through the same API.
- Development logging shows transitions, but production logging is silent.
- Full typecheck and ESLint remain constrained by pre-existing project/toolchain issues described above.

## 21. Recommended Stage 2 Integration Point

Stage 2 should consume `useAvatarActivityState().activityState` in a small Live2D behavior adapter adjacent to—but separate from—the existing `Live2D` component/model hook.

That adapter should map states to carefully bounded motion eligibility without changing the controller, audio pipeline, or Cubism update loop. `speaking` must continue to suppress autonomous idle motion, and actual parameter/motion selection should remain a separate module.

## Runtime State Changes

Only ephemeral browser runtime state was added:

- `activityState`
- monotonic `lastUserActivityAt`
- one pending transition timeout
- transient speaking tokens

Nothing is persisted to conversation history, character state, local storage, or backend storage.

## Live Verification Command

```bash
cd /root/waifu/Open-LLM-VTuber-Web
node --experimental-strip-types tests/avatar-activity-controller.test.ts
node tests/avatar-activity-integration.test.mjs
npm run build:web
```

## Status

MILI HIDUP — IDLE STATE CORE V1 SELESAI
