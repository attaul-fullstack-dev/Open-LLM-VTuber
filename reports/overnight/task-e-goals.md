# Task E — Goal lifecycle

Status: **AUDIT COMPLETE, NO DEFECT FOUND, NO CHANGE.** (Correct-by-design.)

## Evidence

- Goals are hardcoded seeds (`default_seed_goals`, 3 fixed rows), persisted
  under `character_state/<conf_uid>.json`, never generated
  (`GoalSource = Literal["seed"]`, non-seed forced to `"seed"`).
- State machine: `seed → active → done` only, `from_status` enforced,
  unknown ids fail soft, no auto-complete, no reactivation
  (`test_goal_seeds` transition tests pass).
- Restart safety: `ensure_seed_goals` is idempotent (non-empty list returned
  as-is; seeds emitted only when empty), `_ensure_goals_seeded` skips persist
  when id+status are unchanged (`test_g_restart_preserves_state`,
  `test_h_offline_gap_does_not_alter`, `test_future_lifecycle` round-trip).

One cosmetic note (not a defect, not fixed): duplicate unreachable `except`
in `future_intentions.py:232-235` (dead code, no behavioural impact).
