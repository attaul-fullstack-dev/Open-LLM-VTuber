# Task H — Autonomous Life Decision Layer

Status: **AUDIT + 4 FIXES (PROVEN), TESTED, DEPLOYED.** Three PROBABLE items
documented and deliberately left (see blockers).

## Fixes (commit `47284f4`)

1. **Goal evidence pinned before the gate (PROVEN).** `_goal_evidence_trigger`
   called the anchor inside the trigger, so a gate-suppressed turn consumed
   the dedup anchor without speaking and the goal could never fire again.
   Detection now stashes a deferred anchor; dispatch consumes it once;
   suppression discards it (goal re-detected next cycle). One pre-existing
   test encoded the bug and was rewritten to the corrected semantics.
   Tests: +4 in `test_adl_proactive_wiring.py` (14/14).
2. **Anchor write clobbered concurrent activations (PROVEN).**
   `record_goal_evidence` persisted a whole stale in-memory snapshot.
   New `record_goal_evidence_fresh` does a read-modify-write; the deferred
   consume prefers it. Test: `test_anchor_survives_a_concurrent_activation_by_another_session`.
3. **Unparseable `activity_started_at` froze the life state (PROVEN).**
   Stored verbatim, `_parse_iso` substituted "now", age stuck at 0 forever,
   disabling duration limits/sleep exit/stale-idle decisions. Stamps are
   validated on load (start falls back to last-update). Tests in
   `test_world_state_integrity.py`.
4. **Future `last_update_at` never repaired (PROVEN).** Returned unchanged
   with `changed=False`, freezing state until the wall clock caught up.
   Repaired when skew exceeds `_MAX_BACKWARDS_SKEW` (1h); small skew keeps
   the historical fail-soft no-op; the repair itself forces a persist so disk
   heals. Tests: `test_corrupt_future_stamp_is_repaired_and_persisted`,
   `test_repaired_state_stops_being_frozen`, `test_small_backwards_skew_stays_fail_soft`,
   plus on-disk integration tests.

Decision suites after change: 185 + 235 + 10 tests, all pass.

## Verified correct, not touched

UTC-only persistence + user-timezone hours, duration clamping, goal
`last_evidence_id` dedup, single-reason trigger priority, `decide=False`
widget path, restart reconstruction from real timestamps, `by="decision"`
tags, SELF reality boundary, `not proactive` memory guards.

## Left for follow-ups (PROBABLE, need execution proof)

Save-failure retry/idempotency, cross-process lost-update (state_version
exists but unused), per-client proactive budget keying.
