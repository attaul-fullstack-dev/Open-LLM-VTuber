# Task A — WebSocket disconnect 1006

Status: **AUDIT COMPLETE, root cause NOT PROVEN.** No code change to the
transport; one reliability fix shipped via Task K (K-3) and the incident
recorder unlocked via Task I.

## Evidence trail

- Searched the full production log for `1006`, `abnormal`, disconnect
  reasons: **zero abnormal-closure records**. Disconnects present are normal
  client closes and mid-turn detaches.
- Audited the lifecycle end to end:
  - `routes.py:29-56` — accept → `handle_new_connection` → `handle_websocket_communication`.
  - `websocket_handler.py:1036-1069` — receive loop; `WebSocketDisconnect` re-raised to the route handler.
  - `websocket_handler.py:1117-1213` — `handle_disconnect`: idempotent, group notify guarded, in-flight single turn detached (never cancelled) + registered by `history_uid`, subscriber binding dropped, context closed.
  - `websocket_handler.py:2092-2099` — heartbeat is client-initiated with server ack; no server ping/timeout mismatch exists to fix.
  - Frontend `websocket-service.tsx:160-367` — exponential-backoff reconnect (1s→10s cap), single retry timer, duplicate-message guards.
- Conclusion: the path is already hardened from prior work; there is no
  evidence-backed defect to fix, and inventing one would violate the mission.

## Regression coverage added

- `tests/ws-incident-recorder.test.ts` previously could not execute (missing
  `ws-incident-notifier` resolution). The Task I loader unlocks it; it passes.
- `tests/audio-mute-queue.test.mjs` proves the one reconnect-adjacent wedge
  actually fixed in this run (mute left queued tasks pending so
  `waitForCompletion()` never settled).

## Limits

The original 1006 was never reproduced here. A transport-level reproduction
(with close codes + proxy logs) is still required before any root-cause claim.
