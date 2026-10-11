# Task C — Memory Manager

Status: **AUDIT + 1 FIX (phantom success), TESTED, DEPLOYED.**

## Audit outcome

- Character-memory entries: `{text, added_at, explicit, kind}` — no stable
  id, no source, `kind` never written by any caller; character-scoped.
  There is no delete-by-id API (addressed by text overlap).
- Deletion writes through to the authoritative file (`character_state.py:
  load → filter → atomic save`) and the agent rebinds from disk — correct.
- Attachment deletion paths are idempotent and honest (except the two fixed
  below). `memory_receipt.py` itself is correct; its failure branch was
  unreachable because the wired `forget_fn` could never return False.

## Fix (commit `75dc1ed`)

**Forgetting nothing reported success (PROVEN).** `remove_character_memory`
returned the unchanged state when zero rows matched, and the agent mapped any
non-None state to `True`, so the receipt asserted "HAS been removed" for a
removal that never happened. New `remove_character_memory_count` returns
`(state, removed)`; the agent returns True only on a real removal (a no-match
falls into the safe `stored=False` receipt direction, never a phantom claim).
Tests: `test_forget_reports_real_removal_not_phantom_success`,
`test_forget_returns_false_when_nothing_is_stored` (18/18 in the file).

## Not changed

Cross-client snapshot staleness (each connection holds its own agent state):
same-agent paths re-read/write through disk, but no cross-connection
invalidation was introduced (architectural, see blockers). Memory dedup
threshold (`suka`-style short facts) and `reset_character_memory` shape:
audited, left alone.
