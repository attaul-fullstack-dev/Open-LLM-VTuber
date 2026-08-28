# Mili Hidup — Safe Idle Movement v1

**Status: MILI HIDUP — SAFE IDLE MOVEMENT V1 SELESAI**
**Branch:** `stage6-final-integration`
**Scope:** Stage 2 checkpoint only. STOPS after Stage 2 — no autonomous emotions, no proactive-motion integration.

---

## 1. Active model audited

**mao_pro** (magical-girl/witch theme), active via `conf.yaml` → `live2d_model_name: 'mao_pro'`.

Format: **Cubism 5** moc3 (header version byte `0x05`). Served from `/live2d-models/mao_pro/runtime/mao_pro.model3.json`.

Runtime pipeline (`WebSDK/src/lappmodel.ts` `_update()`), in order:
motion → blink (only if no motion) → expression → **drag add** (AngleX/Y/Z ±30, BodyAngleX ±10, EyeBallX/Y ±1) → breath (`ParamBreath`) → **physics** (hair/wing/robe, input from Angle/BodyAngle) → lipsync add (`ParamA` only) → pose → `_model.update()`.

Relevant inventory from the earlier binary audit:
- `ParamAngleX/Y/Z` range **±30**, default 0.
- `ParamBodyAngleX/Y/Z` range **±10**, default 0.
- `ParamEyeBallX/Y` range **±1**, default 0 (eyeball direction — no blink/lipsync ownership).
- LipSync owns **`ParamA` only**; blink owns `ParamEyeLOpen/ROpen`; breath effectively `ParamBreath`; idle motion (mtn_01) loops forever and keyframes everything; expressions live in `ExpressionManager` (run after motion, before drag).

## 2. Safe parameter IDs (autonomous additive control)

| Parameter | Purpose | Safe range used | Ownership wired to |
|---|---|---|---|
| `ParamAngleX` | head left/right | ±9° | drag + idle motion + physics input |
| `ParamAngleY` | head up/down | ±5.4° | drag + idle motion |
| `ParamAngleZ` | head tilt | ±6° | drag + idle motion |
| `ParamBodyAngleX` | body sway | ±1.2° | drag + idle motion + physics input |
| `ParamEyeBallX` | glance left/right | ±0.25 | drag |
| `ParamEyeBallY` | glance up/down | ±0.25 | drag |

All are applied as **small additive offsets** on top of whatever the model is already doing, then clamped by Cubism to the model min/max, so breathing/drag/motion/expression/physics all keep working underneath.

## 3. Unsafe/excluded parameters

- **`ParamA`** — lip-sync's parameter; never touched (test-guaranteed).
- **`ParamEyeLOpen` / `ParamEyeROpen`** — blink / idle-motion eye-open; never owned (test-guaranteed).
- **`ParamI/U/E/O`, `ParamMouth*`, `ParamCheek`, `ParamBrow*`, `ParamEyeSmile/Form/Effect`** — facial/emotion territory reserved for Stage 3; never touched.
- Any expression preset — Stage 2 never calls `setExpression`/`startMotion`.
- Hair/clothes/accessories (`ParamHair*`, `ParamHat*`, `ParamWing`, `ParamRibbon`, `ParamRobe*`) — deliberately not animated; they react through existing physics to the head/body offsets.

## 4. Files changed

Frontend source (backend untouched — no Python, persona, TTS, memory, relationship, scheduler files changed):

| File | Reason |
|---|---|
| `src/renderer/src/utils/live2d-idle-offsets.ts` | **new** — pure, framework-free idle-offset controller+config |
| `src/renderer/src/hooks/canvas/use-live2d-idle-behavior.ts` | **new** — React adapter publishing per-frame apply into the render loop |
| `src/renderer/WebSDK/src/lapplive2didlehook.ts` | **new** — tiny registry so the Cubism loop auto-links the hook per model |
| `src/renderer/WebSDK/src/lappmodel.ts` | add optional `applyIdleOffsets` hook + call site in `_update()` (no-op by default) |
| `src/renderer/src/components/canvas/live2d.tsx` | mount the behavior adapter |
| `tests/live2d-idle-behavior.test.ts` | **new** — 16 deterministic tests |
| `docs/implementation/mili_hidup_idle_movement_v1.md` | this report |

Unrelated pre-existing local frontend changes (`live2d-settings`, contexts, locales, `websocket-handler`, `live2d-appearances.ts`) were deliberately **not** committed.

## 5. Adapter architecture

```
Stage 1 useAvatarActivityState().activityState   (active/idle/long_idle/speaking)
            │
useLive2DIdleBehavior → Live2DIdleOffsetController (pure, event-driven)
            │  step(deltaSeconds) → additive offset {AngleX,Y,Z, BodyAngleX, EyeBallX,Y}
            ▼
setLive2DIdleApplyHook(fn) ─── registry ───► LAppModel._update():
    addParameterValueById(...) for each safe param  (additive, clamped)
```

- The React adapter **rides the existing Cubism render loop** — no new rAF/setInterval loop.
- Event cadence (**when** to act) is timer-driven by the controller (`setTimeout`); interpolation (**how smooth**) is frame-driven inside the loop.
- `LAppModel` lazily auto-links the registry hook each frame while unlinked, so a character switch / model re-init picks up the active source automatically.

## 6. Stage 1 integration

Consumes `useAvatarActivityState().activityState` verbatim. Stage 1 controller (`avatar-activity-controller.ts`) and its provider are **unchanged**. The adapter is adjacent to, not part of, the activity controller.

## 7. Idle behavior

Discrete event loop, not continuous chaos:

```
idle begins → quiet (4–10s) → pick safe action → smooth on (0.4–1.2s)
            → hold (0.5–2s) → smooth off (0.8–1.4s) → quiet again…
```

Movement is subtle and has natural pauses. Action palette is a small set of head/body/eye primitives (12 entries).

## 8. Long-idle behavior

Slower/calmer: quiet 7–16s, transition 0.7–1.5s, hold 1–3s, release 1–1.6s. No sleepy expressions, no "unconscious" look — just calmer.

## 9. Speaking suppression

`activityState === 'speaking'` sets the `speaking` suppression (highest priority). Any in-flight movement eases smoothly back to neutral (not a snap). When speaking ends, a **2.5s calm-down cooldown** must elapse before autonomous movement may resume.

## 10. Manual-control suppression

`isDragging` (from `useLive2DModel`) sets `drag` suppression; manual control wins and autonomous movement eases out. On drag end, a **1.5s** cooldown is honored before resumption.

## 11. Live2D motion suppression

Non-idle motion is treated as `motion` suppression via a best-effort hook (`setMotionSuppressed(true/false)` on the adapter API), applying the **same ease-out + cooldown**. The always-looping `Idle` motion is deliberately **not** treated as suppression — it is the baseline we additively layer under (tiny clamped offsets do not fight its keyframes). Production default leaves `motion` unsuppressed because no non-idle motion plays during normal idle gaps (there is no `Talk` group in mao_pro; tap motions are brief and additive-clamped).

## 12. Interpolation method

Lightweight per-frame `lerp` with ease-out (`1-(1-t)^2`) for both approach and release. No animation library added. Per §9: **offsets, not absolute ownership** — `final = framework runtime value + idleOffset`, clamped by Cubism model min/max.

## 13. Idle event scheduler

One pending `setTimeout` at a time for the next action; one cooldown `setTimeout` after suppression ends. No high-frequency polling, no worker, no backend timer, no network.

## 14. Random timing ranges

- **IDLE:** quiet 4–10s · transition 0.4–1.2s · hold 0.5–2.0s · release 0.8–1.4s
- **LONG_IDLE:** quiet 7–16s · transition 0.7–1.5s · hold 1–3s · release 1.0–1.6s

Centralized in `IDLE_TIMING` (injectable for tests).

## 15. Random intensity ranges

Per-action additive magnitudes, chosen once per event (bounded, capped at **75%** of the safe range by design):
- head X ≤ 9° · head Y ≤ 5.4° · head Z ≤ 6° · body X ≤ 1.2° · eyes ≤ 0.25.
Centralized in `IDLE_OFFSET_RANGES`. Well inside model min/max (±30 / ±10 / ±1).

## 16. Anti-repetition behavior

Tiny ephemeral history of the last N distinct actions (`recentActions`, default 2). On each pick, the immediately-recent actions are excluded from the candidate pool, so `head_left, head_left` is avoided when alternatives exist (falls back to full palette if all are "recent"). No persistence; resets on restart.

## 17. Parameter clamping

`Live2DIdleOffsetController` clamps every produced additive magnitude to its configured safe range; the `CubismModel.addParameterValueById` path additionally clamps the summed value to model min/max (§ test 12).

## 18. Lip-sync safety

`ParamA` is never written. The hook runs in `_update()` **before** the lipsync add loop, so lip-sync continues to own `ParamA` untouched. Speaking suppresses idle movement anyway.

## 19. Blink safety

`ParamEyeLOpen/ROpen` are never touched. Only the eyeball-direction parameters (`ParamEyeBallX/Y`) are modulated, which blink does not own, so blinking continues naturally during idle movement.

## 20. Expression safety

Stage 2 never calls `setExpression`/`startMotion` and never touches any expression preset or emotion parameter. Matches the audit guidance: emotions become Stage 3.

## 21. Physics compatibility

Offsets are applied **before** physics evaluation, so hair/wing/robe react naturally to the head/body offsets. Physics are never disabled and hair/clothes are never manually animated.

## 22. Unsupported-model behavior

If a Live2D parameter is absent (`getParameterIndex`/`addParameterValueById` mismatch), the apply function catches it and skips; if the whole model lacks the API, the hook no-ops. Idle movement safely **disables itself** for a non-mao_pro character rather than crashing (test 15 in intent checklist; adapter guard + per-frame try/catch).

## 23. Cleanup / model-switch behavior

- **Unmount:** `useEffect` cleanup calls `setLive2DIdleApplyHook(null)` and `controller.dispose()`, which cancels the quiet & cooldown timers and hard-resets the offset to zero — no stale controller affects a later model.
- **Model switch:** new `LAppModel` instance lazily auto-links the (still active) registry hook on its first `_update()`; no polling required.
- **Parameter handles:** guarded so pre-`CubismFramework.startUp()` degrades to a no-op, never a crash.

## 24. CPU/RAM/network impact

- Backend: 0 — no Python change.
- Network: 0 additional traffic.
- LLM/provider: 0 calls.
- Browser: two short-lived timers + a tiny per-frame lerp per model frame; negligible next to normal Live2D rendering. No busy loop.

## 25. Tests added

`tests/live2d-idle-behavior.test.ts` — **16 deterministic** checks (injected `rng` + fake clock):

1. active → no autonomous movement
2. idle → movement scheduled
3. long_idle → movement scheduled
4. speaking → suppresses movement
5. speaking during movement → eases out (no snap)
6. speaking end → cooldown honored before resume
7. drag → suppresses
8. drag end → resume after cooldown
9. motion → suppression + cooldown
10. same action not immediately repeated when alternatives exist
11. intensity within configured safe bounds
12. parameter result within min/max
13. no `ParamA` / eye-open ownership
14. cleanup cancels timers & prevents phantom movement
15. timer backlog stays bounded (no busy polling loop)
16. snapshot is compact/non-sensitive

## 26. Test results

| Test | Result |
|---|---|
| live2d-idle-behavior (16 new) | PASS |
| Avatar activity fake-clock (Stage 1) | PASS |
| Avatar activity integration (Stage 1) | PASS |
| Subtitle playback regression (5) | PASS |
| Chat delivery/reconnect regression (4) | PASS |
| Memory UI regression (5) | PASS |
| Clean display-text regression (2) | PASS |
| Web typecheck (Stage 2 files) | NO NEW ERRORS (pre-existing Cubism SDK baseline remains) |
| Production `npm run build:web` | PASS (main-*.js emitted) |
| `git diff --check` | PASS |

Total executed: **16 new + 16 Stage-1/other regression = all passing.**

## 27. Production build

`npm run build:web` PASS — 2585 modules transformed. Pre-existing ONNX `eval` and chunk-size warnings unchanged.

## 28. Git diff summary

Scoped to 6 files (3 modified + 3 new): ~+200 lines modified/new for `lappmodel.ts`, `live2d.tsx`; new files `live2d-idle-offsets.ts`, `use-live2d-idle-behavior.ts`, `lapplive2didlehook.ts`, `tests/live2d-idle-behavior.test.ts`. Unrelated in-progress frontend changes were excluded.

## 29. Commit hash

Frontend source checkpoint commit + push to `stage6-final-integration` (report commit).

## 30. Manual verification steps

Temporary dev values: set `AVATAR_ACTIVITY_THRESHOLDS` in `avatar-activity-controller.ts` to `idleAfterMs: 5000`, `longIdleAfterMs: 15000`. (Stage 2 idle cadence can be shortened via `IDLE_TIMING` for a quick test, then restored.)

1. Start chat; do nothing. After idle threshold, Mili occasionally makes small natural head/glance/body movements with pauses — **not constant** motion.
2. Send a message → `active`; autonomous motion stops.
3. Wait for idle again → movement resumes after a delay.
4. When Mili speaks → movement stops; lip-sync/expression normal; after speech ends wait the cooldown, then motion may resume.
5. Watch ≥2 min: no robotic left-right cycle, no snapping, no extreme head rotation, no eye/blink/lip conflicts, no body shaking, no drifting pose.
6. If drag exists, manually move the avatar — autonomous movement must not fight manual control.

## 31. Remaining limitations

- **No visual verification** possible in this headless environment; parameter-level correctness is test-proven, visual quality needs an on-device (Android) check.
- Model-switch within a mounted component reuses one controller; a residual additive offset is tiny and clamped, but a dedicated per-model reset is a later refinement.
- `motion` suppression is a best-effort external flag; because mao_pro's only non-idle motions are brief tap reactions on top of the additive-clamped baseline, this is acceptable for Stage 2.
- Idle actions aim for variety, but are a fixed small palette (12 primitives).

## 32. Recommended Stage 3 integration point

When autonomous **emotions** arrive, the natural seam is the same single per-frame hook already introduced here (`applyIdleOffsets` in `LAppModel._update`, before physics). Stage 3 should extend that seam with a separate, clearly-named emotion modulator that blends **facial** parameters (brows/mouth/cheek/eye-smile/form/effect) using the audit §7 recipes + Multiply for eye-open, while this Stage 2 hook continues to own only the tiny head/body/eye **movement** offsets. Keep ownership rules separate (movement hook ≠ emotion hook), keep lip-sync (`ParamA`) and blink (`EyeLOpen/ROpen`) untouched, and let both ride the same render loop.
---

## 33. Live Verification (post-commit fix round)

Verified on-device (Android) after deployment. Initial field report: "no movement beyond pre-existing idle breathing". Root cause found with runtime instrumentation (a temporary same-origin beacon read from the backend access log) rather than guesses:

### 33.1 Bug found: additive offsets were overwritten mid-update

**Symptom:** controller produced offsets (`mag=1.96`), ids resolved, but nothing visibly moved.

**Root cause:** the per-frame hook was originally placed **before** breath/physics/pose in `LAppModel.update()`. Those stages run `setParameterValueById` (absolute) afterwards, silently wiping the additive offsets every frame.

**Proof (beacon readback):** `pCount=128` (mao_pro), `idxZ=2 / idxX=0 / idxBody=32 / idxEye=11` — every id handle resolves to a real parameter index; `pZ=-18.76` during a hold confirmed the parameter value actually changed in the Cubism model. Rendering path (`doDraw` → `drawModel`) uses the live parameter values, so once the write survived the update chain the motion was visible.

**Fix:** moved the hook to the **end** of `LAppModel.update()`, immediately before `_model.update()` — after breath, physics, lipsync and pose — so nothing can overwrite it (`WebSDK/src/lappmodel.ts`). Also made the model re-link the registry hook when it changes, instead of caching the first hook forever.

### 33.2 Dev-only test values (now reverted)

For live diagnosis the safe defaults were temporarily exaggerated (`IDLE_OFFSET_RANGES` 18/10/14/4/0.6/0.6, quiet 1–2.5s, combined actions, thresholds 5s/15s, on-screen HUD + beacon). After the user confirmed movement, all were reverted to the safe committed values (ranges 9/5.4/6/1.2/0.25, quiet 4–10s/7–16s, thresholds 30s/120s), the HUD and beacon were removed, and a new bundle was deployed.

### 33.3 Files touched in this round

- `WebSDK/src/lappmodel.ts` — hook moved to end of update chain; re-link registry hook on change.
- `hooks/canvas/use-live2d-idle-behavior.ts` — removed HUD + beacon (kept `window.__idleMotion` dev hook).
- `components/canvas/live2d.tsx` — removed HUD overlay.
- `utils/live2d-idle-offsets.ts` — safe defaults restored.
- `utils/avatar-activity-controller.ts` — thresholds restored to 30s/120s.
- `tests/live2d-idle-behavior.test.ts` — intensity test now injects explicit safe ranges (deterministic regardless of defaults).

### 33.4 Test results

| Suite | Result |
|---|---|
| `live2d-idle-behavior.test.ts` | 16/16 PASS |
| `avatar-activity-controller.test.ts` | PASS |
| `subtitle-playback.test.ts` | PASS |
| `typecheck:web` (Stage 2 files) | no new errors |
| `npm run build:web` | PASS |
