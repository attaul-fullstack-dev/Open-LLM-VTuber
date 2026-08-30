+# Chat Delivery Reconnect Fix

## Root Cause

Live evidence showed that the user message bubble could appear without a
corresponding backend request:

- the backend was restarted while the mobile page remained open;
- the old WebSocket closed and the frontend did not reconnect automatically;
- `useTextInput()` appended the human bubble and cleared the draft before
  `WebSocketService.sendMessage()` verified that the socket was open;
- therefore a disconnected send looked successful in the UI even though the
  backend never received it.

The backend log after 11:35 contained no user conversation event for the visible
unsent message. Four established connections from the same client address were
also observed, but there was no evidence that the provider or context pipeline
received the missing request.

## Files Changed

Frontend source:

- `src/renderer/src/context/websocket-context.tsx`
- `src/renderer/src/services/websocket-service.tsx`
- `src/renderer/src/hooks/footer/use-text-input.tsx`
- `tests/chat-delivery-reconnect.test.mjs`

Backend/persona:

- `characters/id_mili.yaml`
- `conf.yaml` (active local ignored configuration)
- `tests/test_mili_ui_response_polish.py`
- `frontend` submodule pointer
- `docs/implementation/chat_delivery_reconnect_fix.md`

## Delivery Strategy

`sendMessage()` now returns a boolean:

- `true`: the serialized payload was handed to an open WebSocket;
- `false`: the socket was unavailable or sending threw.

The human message is appended and the draft/attachments are cleared only after
`true`. A failed send keeps the user's draft intact and cannot create another
phantom history bubble.

No unacknowledged payload is queued for replay, avoiding duplicate messages
after reconnect.

## Reconnect Strategy

The WebSocket service now keeps one bounded reconnect timer:

- first retry after 1 second;
- exponential delay with a maximum of 10 seconds;
- one timer at a time;
- successful connection resets the attempt counter;
- stale close/message callbacks from a superseded socket are ignored;
- explicit disconnect cancels retries.

This recovers after backend restarts and temporary mobile-network drops without
a busy loop or reconnect storm.

## Emoji Guidance

The active Mili preset still treats emoji as optional and normally limits usage
to 0--1. Guidance now explicitly asks for emoji often enough in casual chat or
teasing to feel like a real chat habit, while forbidding emoji on every reply,
mechanical suffixes, and decorative chains.

Visible text retains emoji. Existing TTS preprocessing continues stripping
Unicode emoji from spoken text. Live2D markers and their parser are unchanged.

## Runtime State Changes

Only ephemeral frontend WebSocket state was added:

- current connection URL;
- one reconnect timer;
- reconnect attempt count;
- explicit-disconnect flag.

No conversation history, relationship, memory, summary, provider, or
persistence schema changed.

## Tests

| Test | Result |
|---|---|
| Send returns delivery result | PASS |
| Failed send creates no phantom bubble | PASS |
| Failed send preserves draft | PASS |
| Single bounded reconnect timer | PASS |
| Stale socket callbacks ignored | PASS |
| Explicit disconnect cancels retry | PASS |
| Frontend deterministic suite (16 cases / 4 files) | PASS |
| Frontend production build (clean source) | PASS |
| Modified frontend TypeScript files | PASS — 0 new errors |
| Full frontend typecheck | BASELINE FAIL — 584 existing Live2D SDK errors |
| Emoji/Live2D/TTS contract | PASS — 4/4 |
| Full backend regression | PASS — 199/199 |
| Ruff | PASS |
| Python compileall | PASS |
| Config validation | PASS |
| `git diff --check` | PASS |

## Commits

- Frontend source: `9ddeda3`
- Static production bundle: `e2be509`
- Backend parent: recorded by the commit containing this report

## Live Verification Command

```bash
sudo systemctl restart olv-server.service
curl -sS http://127.0.0.1:8880/ | grep -o 'assets/[^" ]*\.js'
sudo systemctl status olv-server.service --no-pager
```

Expected deployed asset:

```text
assets/main-weqfnJxd.js
```

On Android, reload the page once so the new reconnect code is active. Then:

1. send several ordinary messages and verify each visible user bubble receives a
   backend response;
2. restart the backend while leaving the page open;
3. wait for the connection badge to recover automatically;
4. send again without manually refreshing;
5. temporarily disable network, tap send, and verify the draft remains instead
   of creating a phantom bubble;
6. restore network and resend;
7. chat casually for multiple turns and verify emoji appears occasionally, not
   on every response.

## Remaining Limitations

WebSocket `send()` confirms browser-level handoff, not an application-level
server acknowledgement. A future protocol could add per-message ACK IDs, but
that is not required to prevent the observed disconnected phantom-send bug.

## Status

CHAT DELIVERY RECONNECT FIX SELESAI
