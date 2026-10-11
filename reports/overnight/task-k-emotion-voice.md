# Task K — Emotion/voice/avatar synchronization groundwork

Status: **AUDIT + 1 FIX (PROVEN), TESTED, DEPLOYED. Vocabulary preserved.**

## Fix (frontend commit `5656290`, `footer.tsx`)

**Quick-mute path not equivalent (PROVEN).** `audioManager.setMuted(true)`
stopped the current element but, unlike voice-OFF and interrupt, never
cleared `audioTaskQueue`. Queued tasks stayed in `activeTasks`, so
`waitForCompletion()` never settled: no `frontend-playback-complete`, face
never released, turn lifecycle wedged. One-line fix calling the same
primitive; this was the single quick-mute call site. Behavioural proof +
source contracts: `tests/audio-mute-queue.test.mjs` (5).

## Audit (verified correct, not touched)

- `voice_emotion.py` mapping + ElevenLabs-only gating + kill switch
  (50/50 tests); tag only added when synthesising, so voice-OFF never tags.
- Emotion vocabulary identical on both sides (10 labels, `neutral` fallback).
- Audio OFF: backend skips synthesis on all paths; frontend stops + clears +
  per-segment gate + persistence + reconnect sync.
- Lip-sync stop/reset, response-face latch policy, idle-facial subscription
  cleanup, `parseModelUrl`/`isValidModelParts`, chat-avatar `onError`.

## Not changed (need browser-dependent proof first)

In-flight promise resolution under `audio.pause()+src=''` (no other listener
resolves it), stale `ended` face re-claim, completion-effect single claim,
unowned text-only timer. No voice/visual quality claim is made.
