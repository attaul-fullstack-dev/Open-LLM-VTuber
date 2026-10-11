# Overnight Mission — Task Ledger

Started: 2026-10-11 (autonomous session)
Repos: `/home/ubuntu/waifu/Open-LLM-VTuber` (backend, branch `main`) ·
`/home/ubuntu/waifu/Open-LLM-VTuber-Web` (frontend source, branch `main`)

## Baseline (established before any change)

Backend: `./.venv/bin/python -m unittest discover -s tests -p "test_*.py"`
→ **1285 tests, OK, exit 0**

Frontend: `node --test tests/*.test.ts tests/*.test.mjs`
→ **268 tests, 259 pass, 9 fail** (pre-existing)

Production: 1 instance (PID 9297), HTTP `/` → 200, WS `/client-ws` live,
0 tracebacks. `conf.yaml` md5 `82579ef240f8c47f3839ec0882b26a89`.

The 9 frontend failures are infrastructure gaps, not product defects:
- 3 × extensionless relative imports (`./response-face-bus`,
  `./ws-incident-notifier`) that Node's ESM resolver rejects while Vite
  accepts them at build time.
- 6 × `@/utils` / `@/theme` tsconfig path aliases, unresolvable by Node
  without a loader.
No production bundle is affected (verified: `main-BcovxHGW.js` contains the
referenced code).

---

## Task status

| Task | Status | Evidence |
|---|---|---|
| A — WS disconnect 1006 | AUDIT COMPLETE, root cause NOT PROVEN | `task-a-websocket.md` |
| B — Attachment Memory | AUDIT + FIX | `task-b-attachment.md` |
| C — Memory Manager | AUDIT (correct-by-design) | `task-c-memory-manager.md` |
| D — Self Model prefs/desires | 2 FIXES (PROBABLE defects) | `task-d-self-model.md` |
| E — Goal lifecycle | AUDIT (correct-by-design) | `task-e-goals.md` |
| F — Values vs Persona | AUDIT (correct-by-design) | `task-f-values-persona.md` |
| G — Live chat/history | AUDIT + TESTS | `task-g-chat-consistency.md` |
| H — Decision Layer | 4 FIXES (PROVEN/PROBABLE) | `task-h-decision-layer.md` |
| I — UI/widget automation | TEST-INFRA FIX (unblocks 9 suites) | `task-i-ui-tests.md` |
| J — Custom avatar | 1 FIX (PROVEN startup crash) | `task-j-avatar.md` |
| K — Emotion/voice sync | AUDIT + 1 FIX | `task-k-emotion-voice.md` |

## Fixed in this run (backend, commit 1)

1. **PROVEN server-startup crash** (`service_context.py`) — a missing/broken
   Live2D model left `live2d_model = None` and `construct_system_prompt`
   dereferenced `.emo_str`, raising `AttributeError` that aborted startup via
   `run_server.py`'s `sys.exit(1)`. Now fail-soft.
2. **PROVEN goal silently lost** (`websocket_handler.py`) — the goal-evidence
   anchor was pinned at *trigger* time, so a gate-suppressed turn consumed the
   dedup anchor without ever speaking and the goal could never fire again.
   Deferred to dispatch; suppression leaves the goal pending.
3. **PROVEN state clobber** (`basic_memory_agent.py`) — the anchor write
   persisted a whole stale in-memory snapshot, reverting a goal another
   session had activated. Now a read-modify-write.
4. **PROVEN frozen life state** (`world_state.py`) — an unparseable
   `activity_started_at` made the activity age 0 forever, disabling duration
   limits and stale-idle decisions. Validated on load.
5. **PROVEN unrecoverable future stamp** (`world_state.py`) — a
   `last_update_at` far in the future was returned unchanged with
   `changed=False`, freezing the life state until the wall clock reached the
   forged value. Repaired when the skew exceeds 1 hour.

New regression tests: `tests/test_world_state_integrity.py` (10) +
`tests/test_adl_proactive_wiring.py` (+5, 1 rewritten).
Backend after change: **1299 tests, OK**.

## Explicitly NOT changed (audited and found correct)

Goal lifecycle state machine and idempotent seeding; values/persona/mood
isolation in prompt assembly; world-state timestamp persistence and
user-timezone hour derivation; voice→emotion mapping and its kill switch;
audio-OFF synthesis gating; Live2D URL validation barriers.
