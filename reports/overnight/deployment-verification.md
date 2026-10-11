# Deployment Verification — overnight mission

Target: production backend `:12393` (+ static frontend) on this host.
Procedure: official workflow (source push → `build_frontend.sh` → static
submodule push → pointer-bump push → kill-by-PID → `setsid start_server.sh` →
health checks). No improvisation.

## Revisions deployed

- Backend HEAD: `9f05a17` (contains `75dc1ed` + `47284f4`)
- Web source HEAD: `5656290`
- Static submodule HEAD: `a919774`
- Served bundle: `assets/main-DsPS_peT.js` (confirmed in served `index.html`)

## Post-deploy health (PID 16904, single instance)

| Check | Result |
|---|---|
| Exactly 1 `run_server.py` instance | PASS |
| `GET /` HTTP | 200 |
| WS `/client-ws` first frame | `group-update` (live) |
| Startup log | `Started server process [16904]`, `Application startup complete`, `Uvicorn running` |
| Tracebacks since restart | 0 |
| Deployed code active | process born from worktree at `9f05a17` after restart |

## Data integrity (read-only checks, no writes)

`conf.yaml` md5 unchanged (`82579ef240f8c47f3839ec0882b26a89`).
Observed counters moved only through live user activity during the run
(new turn, user purge of one attachment at 16:45, auto-extracted facts):
histories 18→19, episodic 26 (unchanged), character memories 0→2,
intentions 32 (unchanged), attachments 1→0. No mission operation wrote to
`chat_history/`, `character_state/`, `episodic/`, `attachment_memory/` or
`conf.yaml`; every automated test ran in an isolated temporary directory
(verified: no test references the production `conf_uid id_mili_01`).

## Verification limits

- No production write-path smoke test was run (deliberate: destructive smoke
  tests against live user data are forbidden by the mission). New behaviour is
  covered by 28 new backend tests + 10 new frontend tests, all passing.
- Visual/browser confirmation of the new bundle was not performed (no
  visual-browser tooling in this environment).
- Rollback: `git revert` of the three backend commits + rebuild/republish of
  the previous static bundle; not needed (health green).
