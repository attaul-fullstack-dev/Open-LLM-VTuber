# Test Results — overnight mission

Commands are exact. All counts are observed, not estimated.

## Backend (Open-LLM-VTuber)

Command: `./.venv/bin/python -m unittest discover -s tests -p "test_*.py"`

| Checkpoint | Result |
|---|---|
| Baseline (before any change) | 1285 tests, OK, exit 0 |
| After batch 1 (startup crash, goal deferral, timestamps) | 1299 tests, OK, exit 0 |
| Final (after batch 2 + all new tests) | **1313 tests, OK, exit 0** |

New tests added in this run (28 total):
- `tests/test_world_state_integrity.py` — 10 (corrupt start stamp, future-stamp repair + persistence, small-skew fail-soft, on-disk repair)
- `tests/test_adl_proactive_wiring.py` — +4, 1 rewritten (deferral: no pin at trigger, suppression leaves goal pending, dispatch consumes once, concurrent activation survives)
- `tests/test_history_write_safety.py` — 3 (unreadable transcript preserved, normal appends, missing ids refused)
- `tests/test_avatar_startup_safety.py` — 2 (prompt builds with/without a model)
- `tests/test_attachment_memory.py` — +4 (stable id, mime rejected on load, -1 on failed wipe)
- `tests/test_character_memory_commands.py` — +2 (no phantom success)
- `tests/test_interaction_preferences.py` — +4 (roleplay rejection, one-per-category reader)

Ruff: `ruff check src/ tests/` → **All checks passed**. `ruff format --check` → all touched files formatted.

One pre-existing test was rewritten, not weakened: `test_active_goal_with_evidence_becomes_high`
asserted the trigger pinned the anchor (the exact defect fixed). It now asserts
the deferred semantics plus three new tests that fail on the old code path
(goal loss under suppression, anchor visibility before dispatch, concurrent
activation survival).

## Frontend (Open-LLM-VTuber-Web)

Plain runner (baseline): `node --test tests/*.test.ts tests/*.test.mjs`
→ 268 tests, 259 pass, 9 fail (pre-existing infrastructure gaps).

New runner: `node --import ./tests/register-loader.mjs --test tests/*.test.ts tests/*.test.mjs`
(also `npm test`)
→ **356 tests, 356 pass, 0 fail**

New tests: `tests/audio-mute-queue.test.mjs` (5), `tests/history-field-preservation.test.ts` (5).
The 9 previously unrunnable suites execute fully; no product defect was hiding
behind them.

Typecheck (`npm run typecheck:web`): 582 errors, **all confined to vendored
`src/renderer/WebSDK/`** (pre-existing baseline). Zero errors in any file
changed by this run (verified by filename grep over the output).
