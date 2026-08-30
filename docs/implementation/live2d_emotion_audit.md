# Live2D Emotion Capability Audit — Mili (mao_pro)

**Status: LIVE2D EMOTION AUDIT SELESAI** (audit only — no model/frontend/backend files modified)
**Date:** 2026-08-28

---

## 1. Current Model Identity

| Item | Value |
|---|---|
| Active model | **mao_pro** (magical-girl/witch theme: hat, wing, ribbon, robe, wand, rabbit familiar, ink-drop & heart effects) |
| Config source | `conf.yaml` → `character_config.live2d_model_name: 'mao_pro'` (`conf_uid: mao_pro_001`) |
| Model directory | `/root/waifu/Open-LLM-VTuber/live2d-models/mao_pro/runtime/` |
| Served URL | `/live2d-models/mao_pro/runtime/mao_pro.model3.json` (`model_dict.json`) |
| Alternate present | `shizuku` (Cubism-2-style `PARAM_*` IDs, **not active**) |

**Format:** `model3.json` says `"Version": 3` (JSON spec version). The **moc3 binary header** is `4D 4F 43 33 05` = `MOC3` + version byte **0x05 → Cubism 5 moc format**. The frontend Core (`WebSDK/Core/live2dcubismcore.d.ts`) exports `MocVersion_50`, so the runtime can load it. **Do not infer "Cubism 3" from the JSON version field.**

## 2. Model File Inventory

| File | Present | Notes |
|---|---|---|
| `mao_pro.moc3` | ✔ | 870,272 bytes, moc3 v5 |
| `mao_pro.4096/texture_00.png` | ✔ | single 4096 texture |
| `mao_pro.physics3.json` | ✔ | 16 settings (§10) |
| `mao_pro.pose3.json` | ✔ | arm part switching: `PartArmLA/LB`, `PartArmRA/RB` (+`PartSketch` in special_03) |
| `mao_pro.cdi3.json` | ✔ | 128 readable parameter names |
| `userData3.json` | ✘ | none |
| `expressions/exp_01..08.exp3.json` | ✔ | 8 expressions (§4) |
| `motions/mtn_01..04, special_01..03.motion3.json` | ✔ | groups: `Idle` (mtn_01), `""` unnamed group (mtn_02–04, special_01–03) (§5) |

Groups in model3.json: `EyeBlink → [ParamEyeLOpen, ParamEyeROpen]`, `LipSync → [ParamA]`. HitAreas: `HitAreaHead`, `HitAreaBody` (tap motions configured in model_dict).

## 3. Parameter Inventory (128 parameters, parsed from moc3 binary)

Min/max/default extracted from the moc3 binary parameter table (layout verified against known Cubism ranges: AngleX ±30, EyeLOpen 0–1.2 def 1.0, ParamA 0–1 def 0). **Control attribution:**

- **LipSync (TTS)**: `ParamA` only (additive, weight 4.0, RMS-driven)
- **EyeBlink / idle motion**: `ParamEyeLOpen`, `ParamEyeROpen`
- **Physics** (16 settings): `ParamHairFront`, `ParamHairSideL/R`, `ParamHairBack`, `ParamHairBackR/L`, `ParamoHairMesh`, `ParamHair*Fuwa`, `ParamHatBrim`, `ParamHatTop`, `ParamRibbon`, `ParamWing`, `ParamString`, `ParamAccessory1/2`, `ParamRobeL/R`, `ParamRobeFuwa` ← driven from Angle/BodyAngle inputs
- **Drag/tracking**: `ParamAngleX/Y/Z` (±30 add), `ParamBodyAngleX` (±10 add)
- **Breath**: `ParamBreath` only (all other breath entries have peak 0.0 → no-op)
- **Motions**: keyframe essentially all parameters
- **Expressions**: facial set only

### Facial parameters (all verified in moc3)

| Parameter | cdi3 name | Min | Max | Default |
|---|---|---|---|---|
| ParamAngleX/Y/Z | Angle_X/Y/Z | -30 | 30 | 0 |
| ParamBodyAngleX/Y/Z | Body Rotation_X/Y/Z | -10 | 10 | 0 |
| ParamEyeLOpen / ROpen | Eye L/R_Open | 0 | **1.2** | 1 |
| ParamEyeLSmile / RSmile | Eye L/R_Smile | 0 | 1 | 0 |
| ParamEyeLForm / RForm | Eye L/R_Deformation | 0 | 1 | 0 |
| ParamEyeBallX / Y | Eyeballs_X/Y | -1 | 1 | 0 |
| ParamEyeBallForm | Eyeballs_Shrink | **-1** | 0 | 0 |
| ParamEyeEffect | Eyes_Effect | 0 | 1 | 0 |
| ParamBrowLY / RY | Eyebrow L/R_Y | -1 | 1 | 0 |
| ParamBrowLX / RX | Eyebrow L/R_X | -1 | 1 | 0 |
| ParamBrowLAngle / RAngle | Eyebrow L/R_Angle | -1 | 1 | 0 |
| ParamBrowLForm / RForm | Eyebrow L/R_Deformation | -1 | 1 | 0 |
| ParamA / I / U / E / O | A/I/U/E/O | 0 | 1 | 0 |
| ParamMouthUp | Mouth Corner Upward | 0 | 1 | **1** |
| ParamMouthDown | Mouth Corner Downward | 0 | 1 | 0 |
| ParamMouthAngry | **Pouting Mouth** | 0 | 1 | 0 |
| ParamMouthAngryLine | Pouting Mouth Line | 0 | 1 | 0 |
| ParamCheek | Blush | 0 | 1 | 0 |
| ParamBreath | Breath | 0 | 1 | 0 |

**Critical finding: there is NO `ParamMouthForm`.** Mouth shape is expressed via
vowels (A/I/U/E/O) + `MouthUp`/`MouthDown` (corner up/down) +
`MouthAngry`/`MouthAngryLine` (**a dedicated pout mouth**, cemberut). Asymmetric
smile is not directly supported (MouthUp/Down are single, both-corner params) but
brows and eye smiles ARE per-side.

### Other notable (custom, model-specific) parameters
Shoulders/arms: `ParamLeftShoulderUp/RightShoulderUp` (±10), `ParamArmL/R A01..03`,
`ParamArmLB01..03`, `ParamHandLA/LB/RA/RB`. Whole-body: `ParamAllX/Y/Rotate` (±10),
`ParamAllColor1/2`. Magic/fun set: `ParamWandRotate`, `ParamWandInk*`,
`ParamInkDrop*` (0–30), `ParamMagicPositionX/Y`, `ParamAura*`, `ParamHeart*`
(heal/miss/light/rainbow), `ParamRabbit*` (X/Y/Rotate/Size/Ear/Light),
`ParamExplosion*`, `ParamSmoke*`, `ParamHealLight*`, `ParamStrengthenLight*`,
`ParamFaceInkOn`, `ParamHat*`, `ParamWing`, `ParamRibbon`, `ParamRobe*`,
`ParamHair*Fuwa`, `ParamoHairMesh`.
**Full 128-ID list is embedded in the appendix of the machine-readable copy below.**

## 4. Existing Expressions (all 8, exact values, Blend mode)

All use `Type: "Live2D Expression"`; `EyeLOpen/ROpen` use **Multiply**, everything
else **Add**.

| File | Key values (non-zero) | Likely emotion | Confidence |
|---|---|---|---|
| exp_01 | all Add = 0; EyeOpen ×1 | neutral / reset baseline | high |
| exp_02 | EyeOpen ×0; EyeSmile +1 (both) | happy closed-eye smile “^^” | high |
| exp_03 | EyeOpen ×0 (smile 0) | eyes closed, neutral mouth (sleepy/blink) | high |
| exp_04 | EyeOpen ×1.2; EyeSmile +1; EyeEffect +1 | joyful sparkle face (big smile, starry eyes) | high |
| exp_05 | BrowAngle −1; BrowForm −1; MouthUp −1; MouthDown +1 | sad / dejected frown | high |
| exp_06 | Cheek +1; BrowAngle −1; BrowForm −1 | embarrassed / shy blush | high |
| exp_07 | EyeOpen ×1.2; EyeBallForm −1; BrowForm +1; MouthUp −1; MouthDown +1 | surprised / shocked (wide eyes, shrink pupils) | medium-high |
| exp_08 | EyeForm +1; MouthUp −1; MouthAngry +1; MouthAngryLine +1 | angry / annoyed / pouting | high |

**emotionMap mismatch (model_dict.json, current):** neutral→exp_01 ✔; joy→exp_04 ✔;
smirk→exp_04 (approx); **anger→exp_03 (closed neutral eyes — weak/wrong)**;
**fear→exp_02 (happy closed smile — wrong)**; **sadness→exp_02 (wrong)**;
**surprise→exp_04 (sparkle smile — wrong)**; disgust→exp_03. **exp_05, exp_06,
exp_07, exp_08 are entirely unmapped.** A future (not this audit) mapping fix has
obvious wins: anger→exp_08, sadness→exp_05, surprise→exp_07, fear→exp_05/07,
disgust→exp_08/06.

## 5. Existing Motions

Groups: `Idle` → mtn_01; unnamed `""` group → mtn_02, mtn_03, mtn_04,
special_01, special_02, special_03 (frontend plays Idle on loop via
`MotionGroupIdle='Idle'`; `startRandomMotion("Talk")` is attempted during
speech but **no Talk group exists** → no-op). Tap motions map HitAreaHead/Body →
index 1 of the `""` group. All motions keyframe the *entire* parameter set plus
arm part opacities.

| Motion | Duration | Character (value ranges, approximate) | Reusable for |
|---|---|---|---|
| mtn_01 | 5.57s loop | gentle idle sway: head ±2–4°, body ±3–6°, breath, shoulders, blink keyframed | base idle |
| mtn_02 | 3.47s loop | small head/body shift, shoulder raise | light reaction |
| mtn_03 | 4.4s | medium head turn (AngleZ to ±15), body turn ±7, shrug | annoyed huff, "whatever" |
| mtn_04 | 4.2s | big nod/look (AngleY −23..26, AngleZ ±15–17), body ±8 | shy look-away, surprised jerk |
| special_01 | 7.8s | large head moves (±25), mouth open (ParamA), mouth down | surprise / excitement |
| special_02 | 9.37s | smile eyes + pout mouth + brows (Angle −0.4/−0.6), big pose | dramatic annoyed/pout |
| special_03 | 9.23s | dramatic full move, smile eyes, eye dart | teasing/story moment |

Caveat: ranges above come from an approximate segment parse of the flat
`Segments` arrays; a few curves mis-decode (time/position confusion), so treat
numbers as indicative. Two referenced IDs (`Param5`, `ParamRabbitEliminationEffect`)
do not exist in the moc3 (harmlessly ignored by the runtime).

## 6. Emotion Capability Matrix (current rig, no re-rig)

| Emotion | Possible? | Quality | Available parameters | Limitation |
|---|---|---|---|---|
| neutral | YES | strong | exp_01 | — |
| happy | YES | strong | exp_02/exp_04, MouthUp, EyeSmile | — |
| small smile | YES | strong | MouthUp 0.3–0.5, EyeSmile 0.2–0.4 | — |
| big smile | YES | strong | exp_04 + MouthUp 1 + Cheek 0.3 | — |
| annoyed | YES | strong | exp_08 (partial intensity), MouthAngry, BrowAngle − | — |
| mildly annoyed | YES | strong | MouthAngry 0.3, BrowAngle −0.3, EyesForm 0.3 | — |
| angry | YES | strong | exp_08, MouthAngryLine, BrowAngle −1 | — |
| pout / cemberut | YES | **strong** | **ParamMouthAngry + MouthAngryLine (dedicated!)**, MouthUp − | rare dedicated pout rig |
| sad | YES | strong | exp_05 (BrowAngle/Form −1, MouthDown) | — |
| embarrassed / shy | YES | strong | exp_06 (Cheek + brows) + mtn_04 look-away | — |
| surprised | YES | strong | exp_07 (wide + shrink + brows) + mtn_04/special_01 | — |
| confused | PARTIAL | acceptable | BrowLForm≠BrowRForm asymmetry, EyeBallX avert, AngleZ tilt, EyeBallForm | no dedicated confusion deform |
| sleepy | PARTIAL | acceptable | EyeOpen ×0.2–0.3 (Multiply), exp_03, AngleY down, BodyAngleX lean | no heavy-lid deform; open×0.2 ≠ true droop |
| suspicious | PARTIAL | subtle | EyeOpen ×0.6, EyeBallX/Y averted, BrowLForm≠RForm, flat mouth (MouthUp −0.2) | subtle; no narrowed-iris param |
| teasing | PARTIAL | acceptable | EyeSmile asym (L1/R0.3), MouthUp 0.5, AngleZ tilt, smirk≈exp_04 soft | no single-corner mouth (true smirk impossible) |
| worried | PARTIAL | acceptable | BrowAngle −0.5, BrowForm −0.5, MouthDown 0.3, EyeOpen ×1.1 | same family as sad — intensity separates them |

## 7. Proposed Parameter Recipes (SAFE, inside min/max — not written to files)

**Format: Add-mode values for Add params; Multiply factor for EyeOpen.**

Emotion: mildly_annoyed — Possible: YES
- ParamMouthAngry: 0.3, ParamMouthUp: −0.3 (Add), ParamBrowLAngle/RAngle: −0.3, ParamEyeLForm/RForm: 0.3, ParamAngleZ: −3

Emotion: annoyed — Possible: YES
- ParamMouthAngry: 0.6, ParamMouthAngryLine: 0.4, ParamMouthUp: −0.6, ParamBrowLAngle/RAngle: −0.7, ParamEyeLForm/RForm: 0.6, ParamAngleZ: −6, ParamBodyAngleX: −3

Emotion: angry — Possible: YES (= exp_08)
- ParamEyeLForm/RForm: +1, ParamMouthUp: −1, ParamMouthAngry: +1, ParamMouthAngryLine: +1, ParamBrowLAngle/RAngle: −1, ParamAngleZ: −8

Emotion: pout_cemberut — Possible: YES (dedicated rig!)
- ParamMouthAngry: 0.8, ParamMouthAngryLine: 0.6, ParamMouthUp: −0.4, ParamBrowLForm/RForm: +0.5, ParamEyeBallY: −0.3, ParamAngleZ: +5 (head turn away)

Emotion: sad — Possible: YES (= exp_05)
- ParamBrowLAngle/RAngle: −1, ParamBrowLForm/RForm: −1, ParamMouthUp: −1, ParamMouthDown: +1, ParamAngleY: −8 (head down), ParamEyeBallY: −0.5

Emotion: embarrassed_shy — Possible: YES (= exp_06 + motion)
- ParamCheek: +1, ParamBrowLAngle/RAngle: −0.8, ParamBrowLForm/RForm: −0.8, ParamEyeBallX: −0.7 (avert), ParamAngleY: −10, ParamAngleZ: −10 (mtn_04 look-away)

Emotion: surprised — Possible: YES (= exp_07)
- EyeOpen ×1.2 (Multiply), ParamEyeBallForm: −1, ParamBrowLForm/RForm: +1, ParamMouthUp: −1, ParamMouthDown: +0.6, ParamA: leave to lipsync, ParamAngleY: +10 (jerk back)

Emotion: small_smile — Possible: YES
- ParamMouthUp: +0.4 (Add on top of default 1 → capped), ParamEyeLSmile/RSmile: +0.3, ParamCheek: 0.15

Emotion: big_smile — Possible: YES (= exp_04 + mouth)
- EyeOpen ×1.15, ParamEyeLSmile/RSmile: +1, ParamEyeEffect: +1, ParamMouthUp: +1, ParamCheek: +0.3

Emotion: worried — Possible: YES
- ParamBrowLAngle/RAngle: −0.5, ParamBrowLForm/RForm: −0.5, ParamMouthDown: +0.3, EyeOpen ×1.1, ParamEyeBallX/Y small darts

Emotion: confused — Possible: PARTIAL
- ParamBrowLForm: +0.6, ParamBrowRForm: −0.4 (asymmetric), ParamBrowLY: +0.4, ParamEyeBallX: +0.8 (look away), ParamAngleZ: −12, ParamMouthDown: +0.2

Emotion: sleepy — Possible: PARTIAL
- EyeOpen ×0.25 (Multiply), ParamEyeLSmile/RSmile: +0.2, ParamAngleY: −10, ParamAngleZ: +8 (head tilt), ParamBodyAngleZ: +4, ParamMouthDown: +0.4, ParamBreath: leave to system

Emotion: suspicious — Possible: PARTIAL
- EyeOpen ×0.6 (Multiply), ParamEyeLForm/RForm: +0.4, ParamEyeBallX: −0.9 (side-eye), ParamBrowLForm: −0.3, ParamBrowRForm: +0.3, ParamMouthUp: −0.3, ParamAngleZ: −4

Emotion: teasing — Possible: PARTIAL (no one-corner mouth)
- ParamEyeLSmile: +1, ParamEyeRSmile: +0.2 (asymmetric eyes), ParamMouthUp: +0.5, ParamAngleZ: −6, ParamBrowLForm: +0.3, ParamEyeBallX: +0.4

## 8. Impossible / Weak Without Re-rig

- **A. Parameter does not exist:** single-corner smirk (no per-corner MouthUp);
  tears/crying (no tear param; `ParamEyeEffect` is "Eyes_Effect" sparkle — visual
  purpose unverified); sweat drop; tongue; nose wrinkle; iris size control
  (`EyeBallForm` is pupil *shrink* −1..0 only, no dilation).
- **B. Technically possible but visually weak:** sleepy (no heavy-lid deform —
  only closed vs open), suspicious (no narrowed-iris), true "smug" face.
- **C. Possible with combination:** confused, teasing, worried (asymmetric
  brow/eye tricks) — acceptable, not perfect.
- **D. Already available as expression:** neutral, happy (2 variants), sad,
  embarrassed, surprised, angry/pout.

## 9. Lip-Sync Conflicts (TTS)

- **`ParamA` — owned by lip sync** during speech: `addParameterValueById(ParamA, rms×1.5 clamped 1, weight 4.0)` (lappmodel.ts `_update`). Emotions must NOT add to `ParamA`; additive lipsync will dominate anyway (which is correct: mouth opens while talking).
- **SAFE DURING SPEECH** (lip sync never touches): `ParamMouthUp/Down/Angry/AngryLine`, `ParamI/U/E/O`, brows, `ParamCheek`, `ParamEye*`, `ParamAngle*` (see §10). → **Emotion mouth shape can coexist with talking** — a pout while speaking works.
- **Must be blended/avoided:** `ParamI/U/E/O` could later be used for viseme refinement, but they are currently free — leave them alone for now.
- `EyeOpen` via **Multiply** is safe during speech.

## 10. Eye Blink / Tracking / Physics Conflicts

Update order in `LAppModel._update()`: **motion → (blink only if no motion) →
expressionManager → drag-tracking add → lipSync add → pose → physics → update**.

- `ParamEyeLOpen/ROpen`: owned by blink manager (`CubismEyeBlink` from model3.json
  group) *only when no motion is playing*; **idle motion actually loops forever and
  keyframes EyeLOpen itself**, so in practice blinking comes from mtn_01 keyframes
  and `CubismEyeBlink` rarely runs. Emotion `EyeOpen` should use **Multiply** (blink
  to 0 stays 0; Add would fight blink by forcing open).
- `ParamEyeSmile/Form/Ball/Brow/Cheek/Mouth*`: **conflict-free** — no system owns them.
- `ParamAngleX/Y/Z`: drag-tracking adds ±30 and idle motion keyframes them; an
  emotion head tilt applied via expression (Add, after motion) or runtime add will
  combine additively and clamp at ±30. Acceptable; large fixed tilts fight mouse
  tracking (user drag wins visually) — keep tilts ≤ ~12° or apply before tracking.
- `ParamBodyAngleX/Y/Z`: same (±10 drag, motion keyframes).
- Physics: reads Angle/BodyAngle only → no direct conflict with facial emotion.
- Breath: effectively only `ParamBreath` (peaks 0 for angles) → no conflict.
- Idle motion keyframes ALL facial params, but expressionManager runs **after**
  motion → expressions override motion facial values (relative Add/Multiply).
  Motion body/head movement continues under an expression — good.

**Conceptual guidance for future implementation (not implemented):** eyes-open
modulation via Multiply; everything else via expression Add; never touch ParamA;
keep Angle tilts small; let blink/motion own their params.

## 11. Frontend Expression/Motion API

- Model instance: `Open-LLM-VTuber-Web/src/renderer/WebSDK/src/lappdelegate.ts`,
  `lapplive2dmanager.ts`, `lappmodel.ts` (official Cubism Web SDK framework, forked
  in-repo; **not** pixi-live2d-display). React glue:
  `hooks/canvas/use-live2d-model.ts`, `use-live2d-expression.ts`,
  `components/canvas/live2d.tsx`, `hooks/utils/use-audio-task.ts`.
- Expression API: `lappAdapter.setExpression(name)` / by index via
  `getExpressionName(index)` (`use-live2d-expression.ts`); reset to default
  (exp_01) when `aiState === IDLE` (`live2d.tsx:56-64`).
- Motion API: `lappAdapter.startMotion(group, index, priority)`,
  `model.startRandomMotion(group, priority)` (`lappadapter.ts:57`,
  `lappmodel.ts:660-742`).
- Expression blending: Cubism `ExpressionManager` (relative Add/Multiply per
  exp3 Blend) — supported natively.
- **Direct runtime parameter access: POSSIBLE** — `window.getLAppAdapter().getModel()._model`
  is a `CubismModel` exposing `setParameterValueById` / `addParameterValueById` /
  `multiplyParameterValueById` / `getParameterValueById` (standard Cubism API; used
  by the framework itself). No adapter wrapper exists yet, but JS access is unblocked.

## 12. Future Emotion → Avatar flow (per-message)

1. LLM text already contains `[emotion]` tags (`live2d_expression_prompt`);
   `actions_extractor` → `Live2dModel.extract_emotion` → emotionMap → **index**
   → WS `actions.expressions` → `use-audio-task.ts` sets it when audio starts.
2. Per-sentence emotions already flow (list, first is applied).

## 13. Recommended Architecture: **HYBRID (D)**

- **Base presets:** new exp3 files per emotion (deterministic, blend-safe,
  matches Cubism semantics) + fix `emotionMap` to actually use exp_05–08.
- **Intensity:** runtime parameter modulation via a small adapter wrapper
  (`setParameterValueById/add/multiply`) applying the §7 recipes scaled by
  intensity, or by selecting preset + additive modulation on top.
- Why: presets alone = only 8 fixed faces and wrong mapping; raw runtime params
  alone = re-implementing expression blending and per-model hardcoding; hybrid
  keeps presets portable, intensity smooth, and lip/blink ownership rules
  centralized in one wrapper.
- Conceptual flow: LLM emotion `{type, intensity}` → wrapper maps to preset +
  param deltas → expressionManager/direct params → optional motion from §5 map.

## 14. Intensity Feasibility

All facial parameters are continuous floats with real ranges (e.g. annoyed 0.2 =
MouthAngry 0.2 vs 0.9) → **smooth intensity is feasible**. Multiply-mode eye
open interpolates poorly at the low end (0.2 open ≈ closed — distinguishes
sleepy from neutral poorly); ParamA is unusable for emotion (lipsync). Add-mode
brow/mouth params interpolate cleanly; asymmetric per-side values (BrowLForm vs
BrowRForm, EyeLSmile vs EyeRSmile) also interpolate smoothly.

## 15. Visual Validation

**Not performed — not possible in this environment** (headless server, no
browser/Chrome). All findings come from file/binary inspection of the actual
model and code. No visual claim is made; §6 "Quality" is a parameter-level
estimate that should be verified in-browser before implementing.

## 16. Source Files / Functions Reference

Backend:
- `src/open_llm_vtuber/live2d_model.py` — `Live2dModel.extract_emotion` (index output), `remove_emotion_keywords`
- `src/open_llm_vtuber/agent/transformers.py` — `actions_extractor`
- `model_dict.json` — emotionMap (name→index), idleMotionGroupName, tapMotions
- `conf.yaml` — `live2d_model_name`, `live2d_expression_prompt`

Frontend:
- `WebSDK/src/lappmodel.ts` — `_update()` pipeline (motion→blink→expression→drag→lipsync→pose→physics), `setExpression`, `startMotion`, wavFileHandler lipSync
- `WebSDK/src/lappadapter.ts` — `setExpression`, `startMotion`, `getExpressionName`
- `WebSDK/src/lappdefine.ts` — `MotionGroupIdle='Idle'`, priorities
- `hooks/utils/use-audio-task.ts` — applies `expressions[0]` + tries "Talk" motion
- `hooks/canvas/use-live2d-expression.ts` — set/reset expression
- `components/canvas/live2d.tsx` — reset expression on AI idle

Model:
- `live2d-models/mao_pro/runtime/*` (model3.json, moc3, cdi3.json, physics3.json, pose3.json, expressions/, motions/)

## 17. Remaining Uncertainties

- Actual on-screen appearance of each expression (no visual test possible here).
- Exact visual meaning of `ParamEyeEffect` ("Eyes_Effect") and `ParamEyeBallForm`
  ("Eyeballs_Shrink") beyond inferred semantics.
- Some motion curve ranges are approximate (flat `Segments` parse; a few curves
  time-contaminated).
- Whether the deployed bundle's Core build is ≥ Cubism 5 (d.ts exposes
  `MocVersion_50`; runtime load of this moc3 is presumed working since the model
  is in daily use, which itself confirms compatibility).
- Physics JSON "Name" fields are null (settings unnamed) — behavior derived from
  Input/Output IDs only.

## 18. Final Classification

**READY WITHOUT MODEL EDIT (implement immediately):**
neutral, happy, small smile, big smile, annoyed, mildly annoyed, angry,
pout/cemberut, sad, embarrassed/shy, surprised, worried
(plus intensity scaling via §7 recipes; plus emotionMap fix to use exp_05–08).

**POSSIBLE BUT LIMITED (look acceptable/subtle, no re-rig):**
confused, sleepy, suspicious, teasing (no single-corner smirk).

**REQUIRES LIVE2D MODEL EDIT / RE-RIG:**
none of the 16 requested emotions. Re-rig only needed for: true one-corner
smirk/smug, tears/crying, sweat drops, tongue, iris dilation.
