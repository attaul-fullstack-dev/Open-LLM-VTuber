# Mili UI & Response Polish v1

## 1. Files Changed

Frontend source (`/root/waifu/Open-LLM-VTuber-Web`):

- `src/renderer/src/components/sidebar/setting/agent.tsx`
- `src/renderer/src/components/sidebar/setting/character-memory-dialog.tsx`
- `src/renderer/src/components/canvas/subtitle.tsx`
- `src/renderer/src/components/canvas/canvas-styles.tsx`
- `src/renderer/src/components/sidebar/chat-history-panel.tsx`
- `src/renderer/src/components/sidebar/sidebar-styles.tsx`
- `src/renderer/src/hooks/utils/use-audio-task.ts`
- `src/renderer/src/hooks/utils/use-interrupt.ts`
- `src/renderer/src/services/websocket-handler.tsx`
- `src/renderer/src/utils/subtitle-playback.ts`
- `src/renderer/src/utils/clean-display-text.ts`
- `src/renderer/src/App.tsx`
- `src/renderer/src/locales/{en,id,zh}/translation.json`
- `tests/subtitle-playback.test.ts`
- `tests/clean-display-text.test.ts`
- `tests/memory-ui-contract.test.mjs`

Backend/config (`/root/waifu/Open-LLM-VTuber`):

- `characters/id_mili.yaml`
- `conf.yaml` (active local config, ignored by Git)
- `tests/test_mili_ui_response_polish.py`
- `docs/implementation/mili_ui_response_polish_v1.md`

## 2. Long-Term-Memory Launcher Architecture

Agent settings no longer maps every memory into the settings scroll area. It
shows one compact launcher with an icon, label, live count, and chevron. Memory
state and refresh behavior still come from the existing WebSocket events:
`character-memory`, `character-memory-deleted`, `character-memory-reset`, and
`character-state-reset`.

## 3. Dedicated Memory Modal/Sheet Architecture

`CharacterMemoryDialog` is a controlled Chakra dialog portalled above the
settings drawer. At mobile widths it is `100vw × 100dvh`; from `sm` upward it is
a centered responsive dialog capped at 620 × 720 px. Header and footer remain
accessible while only the body scrolls. Safe-area insets are respected.

## 4. Delete-One / Delete-All Behavior

- Delete one reuses `delete-character-memory` with the existing memory text.
- Delete all reuses `reset-character-memory`.
- Both require confirmation.
- Relationship, summary, transcript, storage schema, and memory API are
  unchanged.

## 5. Responsive Memory Behavior

Memory rows wrap with `white-space: pre-wrap` and `overflow-wrap: anywhere`.
Trash and close controls have 44 px touch targets. The list cannot scroll
horizontally and does not sit behind the chat input because it is a viewport
dialog rather than content inside the chat canvas.

## 6. Subtitle Root Cause

The frontend updated subtitle text at the beginning of
`handleAudioPlayback()`, before the generated audio actually began playing.
Synthesis could run ahead and queue subsequent segments, so presentation state
was coupled to task readiness instead of audible playback. The previous code
also accumulated all response segments in one growing subtitle string.

## 7. Old Subtitle Timing Flow

```text
audio payload dequeued
→ append history
→ immediately replace/append subtitle
→ create/load audio element
→ canplaythrough
→ audio.play()
```

## 8. New Playback-Synchronized Flow

```text
audio payload dequeued
→ append history unchanged
→ create subtitle ticket (no visible update)
→ create/load audio element
→ audio `playing`
→ activate ticket and show exactly that segment
→ audio `ended`
→ next queued audio starts
→ show next segment
```

TTS generation and queueing can run ahead without exposing future text in the
subtitle.

## 9. Minimum Readable Timing

No arbitrary millisecond delay was added. A segment stays visible until the
next audio segment actually begins playback. This protects short segments from
queue/synthesis flashes without delaying audio or desynchronizing text. Silent
or muted output uses the explicit no-playback fallback and accumulates text so
segments are not lost.

## 10. Interruption Handling

Every response gets an in-memory generation ID. Interrupt/cancel advances that
ID, clears the visible subtitle, stops audio, and clears the task queue. Late
`playing` events from cancelled audio hold stale tickets and are rejected, so
cancelled future segments cannot appear.

## 11. Emoji Persona Changes

Mili may now use familiar chat emoji when they genuinely improve emotion or a
punchline. The prompt says usually 0–1, occasionally 2, keeps emoji optional,
forbids mechanical suffixes and decorative chains, and requires the wording to
remain meaningful without emoji.

## 12. Emoji / TTS Behavior

Visible response text and stored history retain emoji. Active TTS already uses
`remove_special_char: true`; the existing Unicode-category filter removes emoji
from spoken text while preserving letters and punctuation. No TTS architecture
or provider setting changed.

## 13. Live2D Marker Compatibility

The existing persona instruction still defines `[emotion]` markers as technical
Live2D metadata. Display cleaning removes those markers but preserves emoji.
The actions extractor and expression payload path were not changed.

## 14. Subtitle Bubble Styling

- mobile width capped to 92vw inside a viewport-safe wrapper;
- readable 1.48 mobile line height;
- multi-line text left-aligned on mobile and centered on desktop;
- safe wrapping for URLs/long words;
- 30dvh mobile maximum with internal scrolling only when needed;
- input offset includes bottom safe-area inset;
- consistent padding, 18 px mobile radius, blur, border, and shadow.

## 15. History Bubble Styling

- 1.5 line height and balanced mobile padding;
- responsive max width accounting for the 30 px avatar;
- distinct incoming/outgoing radii and existing gradient preserved;
- `pre-wrap`, `overflow-wrap: anywhere`, and `word-break: break-word`;
- authored paragraph breaks are preserved rather than collapsed into spaces.

## 16. i18n Keys

Added for English, Indonesian, and Chinese:

- `settings.agent.memoryDescription`
- `settings.agent.openMemory`
- `settings.agent.memoryCount`
- `settings.agent.deleteMemoryConfirm`
- `settings.agent.deleteAllMemory`

Existing title, empty-state, delete-one, and reset confirmations are reused.

## 17. Performance Impact

No polling, library, service, background worker, provider call, or persisted
runtime state was added. The subtitle coordinator is one small in-memory object.
The memory list renders only while its dialog is mounted/open.

## 18. Tests Added

- playback activation/order, queued-future protection, short-segment retention,
  duplicate-event protection, cancellation, and silent fallback;
- paragraph/emoji preservation and marker cleanup;
- memory launcher/modal/delete/responsive/i18n source contracts;
- persona emoji optionality, Live2D contract, active-preset parity, and TTS emoji
  cleanup.

## 19. Test Results

| Test | Result |
|---|---|
| Frontend deterministic tests (12 assertions/subtests in 3 files) | PASS |
| Frontend production build | PASS |
| Modified frontend source TypeScript errors | PASS — 0 new errors |
| Full frontend typecheck | BASELINE FAIL — existing Live2D SDK errors |
| Frontend ESLint | BLOCKED — existing missing `eslint-config-airbnb` dependency |
| Backend persona/TTS contract tests | PASS — 4/4 |
| Full backend suite | PASS — 199/199 |
| Ruff | PASS |
| Python compileall | PASS |
| Active config validation | PASS |
| `git diff --check` | PASS |

## 20. Build Result

`npm run build:web` completed successfully from clean source commit `4b3922d`
(2,579 modules). Static production bundle commit: `f660686`. Vite retained the
existing warnings for `eval` inside `onnxruntime-web` and large output chunks;
neither warning was introduced by this patch.

## 21. Git Diff Summary

The patch adds a dedicated memory-view component, two small presentation
utilities, deterministic frontend tests, one backend contract-test module, and
localized strings. It modifies only the playback presentation point, memory UI,
bubble styles, and Mili's emoji guidance. No memory schema, provider, proactive
logic, relationship logic, summary logic, or LLM call count changed.

## 22. Manual Phone Test Checklist

1. Store at least five memories; open Settings → Agent and verify only one
   launcher appears with count 5.
2. Open the memory view at 360/390/412/430 px widths; scroll it, delete one,
   verify count 4, then test cancel/confirm for delete all.
3. Add a 2–3-line memory; verify wrapping and accessible trash button without
   horizontal scrolling.
4. Ask for a 3–4-segment spoken answer; verify each subtitle changes only when
   its matching audio starts.
5. Repeat while TTS synthesis runs ahead; future text must not leap ahead.
6. Interrupt during segment 1; verify current audio/text clear and no queued
   segment later appears.
7. Chat casually for about 20 turns; emoji should occur occasionally, not in
   every answer. Verify TTS does not speak emoji names.
8. Check short, long, paragraph, URL, emoji, and Live2D-marker messages in chat
   history; verify readable wrapping and no clipping.

Live verification:

```bash
sudo systemctl restart olv-server.service
sudo systemctl status olv-server.service --no-pager
```

Then open the normal phone URL, hard-refresh once, and run items 1–8 above.

## 23. Remaining Limitations

- Exact visual/audible behavior still needs the manual Android + real TTS test
  because automated tests use deterministic coordinator state, not a physical
  browser audio decoder.
- Full TypeScript checking remains noisy because of pre-existing Live2D SDK
  strictness errors; modified application source introduced no new error.
- No fixed minimum-duration timer is used by design: unusually tiny real audio
  clips remain visible for their playback duration and until the next clip
  actually starts.

## Runtime State Changes

Only ephemeral subtitle response/segment IDs are added in browser memory. No
conversation, character, relationship, summary, or memory persistence format
changes.

## Status

MILI UI & RESPONSE POLISH V1 SELESAI
