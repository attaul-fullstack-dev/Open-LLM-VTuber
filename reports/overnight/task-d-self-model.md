# Task D — Self Model: preferences and desires

Status: **AUDIT + 2 FIXES (PROBABLE defects), TESTED, DEPLOYED. Phase 2A gate
respected: no acceptance claimed, Phase 2B untouched.**

## Audit outcome

- Two preference schemas: inferred activity preferences (`PreferenceCandidate`,
  gated `>=3 items over >=2 days`, only established items render) and stated
  interaction preferences (frame + dimension + polarity required).
- **No "desires" data structure exists anywhere** (only comments mention the
  word). Nothing to derive or gate.
- **No Phase 2A acceptance gate and no Phase 2B feature flag exist at runtime**
  (phase labels are comments/docstrings only; `kind` is an always-on,
  behaviour-neutral label). Per the mission, no acceptance is claimed and no
  2B activation was performed.
- Guards verified correct: single-event/burst rejection, temporary-frame
  suppression, mood never entering the preference path, insufficient-evidence
  default of silence.

## Fixes (commit `75dc1ed`, `interaction_preferences.py`)

1. **No roleplay guard (PROBABLE).** Sibling detectors (`future_intentions`,
   episodic) reject stage-direction framing; the preference path did not, so
   an in-character line meeting every other gate became a permanent user
   preference. Now rejected with the same `_ROLEPLAY_FRAME` shape.
   Test: `test_roleplay_framed_lines_are_not_captured` (4 cases).
2. **Reader trusted the one-per-category invariant (PROBABLE).** The writer
   maintains it, but a legacy/hand-edited file with two active rows of one
   category rendered contradictory guidance. The reader now enforces
   newest-wins. Tests: `test_reader_enforces_one_active_preference_per_category`,
   `test_reader_keeps_distinct_categories`.

Preference suites: 76 tests pass (`interaction_preferences`,
`preference_derivation`, `self_model`, `self_model_integration`).
