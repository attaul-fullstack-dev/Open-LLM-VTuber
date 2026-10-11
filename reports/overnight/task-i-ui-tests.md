# Task I — UI/widget bugs reproducible through automation

Status: **TEST-INFRA FIX + 2 PRODUCT FIXES, 356/356 PASS, DEPLOYED.**

## The 9 silent suites

Plain `node --test`: 268 discovered, 259 pass, 9 fail. The 9 failures were
resolution gaps, not product defects:
- 3 × extensionless relative imports (`./response-face-bus`,
  `./ws-incident-notifier`) — Vite resolves them, Node ESM does not.
- 6 × `@/utils` / `@/theme` tsconfig aliases — unresolvable by Node.

New `tests/resolve-loader.mjs` (failure-first: defers to Node's resolver,
intervenes only on `ERR_MODULE_NOT_FOUND`) + `tests/register-loader.mjs` +
`npm test` script. Result: **356/356 pass, 0 fail** — 88 more tests actually
execute, and no product defect was hiding behind the 9 failures.

## Product fixes verified by the unlocked suites

1. **Quick-mute wedge** (`footer.tsx`): mute now calls `audioTaskQueue.clearQueue()`
   like the interrupt and voice-OFF paths. Behavioural + contract tests
   (`audio-mute-queue.test.mjs`, 5).
2. **Resync field loss** (`chat-history-context.tsx`, `history-reconcile.ts`):
   `requestId`/`name`/`avatar`/`timestamp` survive the server→UI round trip.
   Tests (`history-field-preservation.test.ts`, 5).

## Limits

No drag/pinch/device-gesture acceptance is claimed. `npm run typecheck:web`
still reports the 582 pre-existing vendored-WebSDK errors; zero in changed
files. ESLint remains broken repo-wide (pre-existing).
