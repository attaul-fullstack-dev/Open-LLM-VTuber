# Git Delivery — overnight mission

No force-push anywhere. Untracked pre-existing junk
(`PHASE14_FUTURE_INTENTION_CROSS_SESSION_REPORT.md`, `Open-LLM-VTuber-Web/assets/`,
`Open-LLM-VTuber/attachment_memory/` data dir) was left untouched.

## Backend — `git@github.com:attaul-fullstack-dev/Open-LLM-VTuber.git`, branch `main`

| Commit | Message | Verified on remote |
|---|---|---|
| `47284f4` | fix: startup crash on missing avatar, goal-evidence loss, and frozen life state | yes, `main` |
| `75dc1ed` | fix: honest deletion reporting, attachment identity, and interaction-preference guards | yes, `main` |
| `9f05a17` | build: deploy quick-mute fix and resync field preservation (submodule pointer) | yes, `main` (HEAD) |

Batch 1 files: `service_context.py`, `world_state.py`, `websocket_handler.py`,
`agent/agents/basic_memory_agent.py`, `test_adl_proactive_wiring.py`,
`test_world_state_integrity.py` (new), `reports/overnight/task-ledger.md` (new).

Batch 2 files: `agent/agents/basic_memory_agent.py`, `attachment_memory.py`,
`character_state.py`, `chat_history_manager.py`, `interaction_preferences.py`,
`websocket_handler.py`, `test_attachment_memory.py`,
`test_character_memory_commands.py`, `test_interaction_preferences.py`,
`test_avatar_startup_safety.py` (new), `test_history_write_safety.py` (new).

## Frontend source — `git@github.com:attaul-fullstack-dev/Open-LLM-VTuber-Web.git`, branch `main`

| Commit | Message | Verified on remote |
|---|---|---|
| `5656290` | fix: quick-mute queue wedge, resync field loss, and test-loader unlock | yes, `main` (HEAD) |

Files: `package.json` (`test` script), `footer.tsx`, `chat-history-context.tsx`,
`history-reconcile.ts`, `tests/resolve-loader.mjs` (new),
`tests/register-loader.mjs` (new), `tests/audio-mute-queue.test.mjs` (new),
`tests/history-field-preservation.test.ts` (new).

## Static bundle — submodule `frontend`, branch `stage7-static-life-widget`

| Commit | Message | Verified on remote |
|---|---|---|
| `a919774` | build: publish quick-mute fix and resync field preservation | yes |

Bundle `main-DsPS_peT.js` built by `./scripts/build_frontend.sh` (vite,
11.94s), synchronized into `frontend/`, committed via the backend
pointer-bump commit `9f05a17`.
