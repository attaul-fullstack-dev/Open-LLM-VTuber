# Phase 14 — Future Intention Cross-Session Recall

## 1. Initial failure

Production, `conf_uid=id_mili_01`, new chat `2026-10-06_06-44-19_...`:

- User: `Mil, kemarin aku bilang hari ini/besok mau nanya apa?`
- Mili: `Mana aku tahu? Kamu kan yang ngomong, masa aku yang harus ingat. Lagian kamu
  nggak ada kasih tahu spesifik mau nanya apa, cuma bilang "nanti" atau "besok".`

Session A (long history `2026-10-04_17-44-08_...`, same character) did contain the
statements — e.g. `iya, nanti jam 7 aku nanya lagi, siap siap jangan lupa`
(2026-10-04T18:05:40Z) and `bentar, nanti hari Selasa besok aku bakal nanya
terakhir aku crot lagi` (2026-10-05T00:13:03Z).

## 2. Exact reproduction

Isolated (temp dir `/tmp/p14-*/`, temp conf `p14recall`, temp history UIDs):

- Session A agent: `set_memory_from_history(conf, history-A)` →
  `observe_character_events("Besok jam 07.00 aku mau nanya lagi sesuatu.", ...)`
  → 1 row persisted.
- Session B agent: fresh `BasicMemoryAgent`, `history-B`, same conf, transcript
  verified free of Session A text → `list_future_intentions()` returns the row;
  `_relationship_system_prompt()` renders
  `- [upcoming due Oct 07 07:00 WIB, stated Oct 06] Besok jam 07.00 ...`.
- Restart (agents dropped, disk reload) + third session: still rendered.
- Result: **PASS** — full E2E output `E2E RESULT: PASS`.

## 3. Root cause

ROOT CAUSE: Future intentions stated in Session A (Oct 4 17:47 UTC, Oct 4 18:05 UTC,
Oct 5 00:13 UTC) were never persisted, because the capture path
(`character_state.record_future_intention` via
`BasicMemoryAgent._observe_future_intention`) did not exist until `d146a56`
(Oct 5 14:57 UTC; natural plans `e4ab958` Oct 6 00:35 UTC). The observer runs only
on new turns — old transcripts are never rescanned. Session B (new history_uid,
same conf_uid, Oct 6 06:44 UTC) correctly retrieved what was stored and the model
answered from no-data.

Proof:

- Commit timestamps (`d146a56` Oct 5 14:57 UTC) are AFTER all Session A
  message timestamps (verified per-message from the history JSON).
- `character_state/id_mili_01.json` contains zero rows (pending OR done) matching
  Session A text. Consume marks rows `done`, never deletes — no `done` rows exist,
  so the statements were never stored (not stored-then-consumed).
- Current code detects ALL THREE historical Session A phrasings plus the task
  example (verified programmatically).
- Writer proven innocent: the 06:12 UTC Oct 6 plan-kind capture (`mau tdur`)
  proves the plan writer was already deployed/running before Session B.

## 4. Files inspected

- `src/open_llm_vtuber/future_intentions.py` (schema, detect, persist helpers,
  retrieval, context render, lifecycle)
- `src/open_llm_vtuber/character_state.py` (`record/complete/consume`,
  atomic store, `future_intentions` field)
- `src/open_llm_vtuber/agent/agents/basic_memory_agent.py`
  (`_observe_future_intention`, `_load_character_state`,
  `set_memory_from_history`, `_relationship_system_prompt`, prompt wiring)
- `src/open_llm_vtuber/conversations/single_conversation.py` (observer call site)
- `src/open_llm_vtuber/websocket_handler.py` (`_handle_create_history`,
  consume-on-dispatch only)
- `chat_history/id_mili_01/*.json`, `character_state/id_mili_01.json` (read-only)

## 5. Files changed

- `src/open_llm_vtuber/future_intentions.py` (+45/-10, functional only; no
  unrelated reformat)
- `tests/test_future_cross_session_recall.py` (new, 10 acceptance tests)
- `PHASE14_FUTURE_INTENTION_CROSS_SESSION_REPORT.md` (this file, untracked)

## 6. Minimal fix

1. `build_future_intention_context`: each row now carries a
   `stated <Mon DD>` tag derived per turn from persisted `created_at` (user tz),
   e.g. `- [upcoming due Oct 07 07:00 WIB, stated Oct 06] ...`. Rows without a
   parseable `created_at` render exactly as before (fail-soft). Same rows, same
   8-row/240-token bounds. This grounds recall questions of the exact failing
   shape (`kemarin aku bilang ... apa?`), which previously only showed the
   future due date.
2. `add_future_intention` overflow: undated pending rows are evicted before dated
   ones (oldest first within each group; done/cancelled never evicted; same
   20-cap). A dated intention (`besok jam 07.00`) now survives accumulation of
   undated immediate-desire rows instead of being evicted first as the oldest.

Detection grammar, schema, persistence, Context Builder placement, temporal
model, and bounds are unchanged (bare-`mau` undated plans stay recallable per
approved `test_4`; first-person subject requirement kept).

## 7. Persistence verification

- Session A store → `character_state/<conf>.json` contains the exact row
  (`kind=plan`, absolute UTC `due_at`, `created_at`, `tz=Asia/Jakarta`).
- Disk reload (`load_character_state`) returns the identical row. PASS.

## 8. Session-boundary test

Test A (`test_A_session_a_to_session_b`): new agent + different history_uid,
query mentions no topic → intention found; query itself creates no row. PASS.
Test C (`test_C_...`): Session B `_memory` contains no Session A text while the
context block carries it — mechanism proven to be the persistent store, not
transcript or summary. PASS.

## 9. Restart test

Test B: agents dropped, state reloaded from disk, third session recalls. PASS.
Agent-level E2E post-restart prompt render. PASS.

## 10. Timezone test

Test E: `tomorrow 07:00` in `Asia/Jakarta` vs `UTC` resolves 7h apart; block
renders `07:00 WIB`. PASS.

## 11. No-fabrication test

Test F: never-stored topics absent from block; empty store renders `""`. PASS.

## 12. Multiple-intention test

Test G: two distinct dated rows both retrieved, due-soonest first
(`Besok...` before `lusa...`), both rendered verbatim. PASS.

## 13. E2E result

Agent-level E2E through production wiring (`set_memory_from_history` →
`observe_character_events` → new history → `_relationship_system_prompt` →
restart → third history): PASS (3/3 checks: stored, B-prompt renders with
stated tag, post-restart renders).

Live-browser E2E with a real model response: NOT TESTED (no live LLM call from
an isolated env; production chat must not be used as a test bench).

## 14. Full regression result

- New suite `test_future_cross_session_recall`: 10/10 PASS.
- Targeted: future intentions/plans/lifecycle/recall + temporal awareness/session
  + self-model/integration + preference derivation + goal seeds + memory-kind +
  episodic continuity + interaction preferences: 243/243 PASS.
- Full backend `python -m unittest discover -s tests`: 1142/1142 PASS.
- `ruff check` on touched files: PASS. (`ruff format` churn on pre-existing
  sections was reverted to keep the diff minimal.)

## 15. Commit

- `d42e671` — `fix: restore future intention cross-session recall`
- 2 files: `src/open_llm_vtuber/future_intentions.py`,
  `tests/test_future_cross_session_recall.py`

## 16. Push

- `b8ed5f1..d42e671 main -> main`
  (github.com:attaul-fullstack-dev/Open-LLM-VTuber.git)
- `HEAD == origin/main` verified.

## 17. Deployment

- SOURCE `d42e671` → COMMIT → PUSH → BACKEND RESTART (no frontend change, no
  rebuild): prod PID `368679` → `374225` (`setsid`, cwd repo root).
- Staging `:18192` PID `325934` untouched.

## 18. Production verification

- `GET /` → 200. PASS.
- WS smoke `history-list` → `n=6` ok. PASS.
- Prod log `/tmp/p14_prod_restart.log`: 0 errors/tracebacks. PASS.
- `character_state/id_mili_01.json` md5 identical before/after
  (`88f94b899d1b03a76f20e0d5de6092ac`). PASS.
- `chat_history/id_mili_01/` newest mtimes unchanged by deploy actions. PASS.

## 19. Remaining limitations

- Intentions stated before a capture feature is deployed are unrecoverable by
  design (observer runs on new turns only; no transcript rescan). The Oct 4–5
  statements fall in this class.
- Undated immediate-desire rows (`aku mau tidur`) are stored as recallable
  undated plans by approved design (`test_4`); dated rows are now eviction-
  protected, but a 20-cap store can still rotate undated rows.
- Pronoun-drop phrasings (`besok mau nanya lagi`, no subject) are rejected by
  design (false positive worse than false negative).
- `_parse_due_at` takes the first `jam` mention, so `sekarang jam 00.46 ...
  ntar jam 7` resolves to the now-reference jam (still future-valid, still
  recalled; due hour may be off). Left untouched as out of scope.
- Live model answer quality for recall questions: NOT TESTED (no live call).

## 20. Final verdict

PASS (code-level recall across session boundary + restart + timezone +
no-fabrication + multiple + regression + verified deployment). Live-model
wording remains NOT TESTED by policy.
