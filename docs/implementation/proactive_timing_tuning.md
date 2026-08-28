# Proactive Timing Tuning

**Status: PROACTIVE TIMING TUNING SELESAI**

## 1. Previous Timing

| Window | Previous min | Previous max |
|---|---|---|
| Initial (user inactive) | 45s | 90s |
| Follow-up (1 ignored proactive) | 90s | 240s |
| Backoff (3 consecutive ignored) | 180s | 360s |

## 2. New Timing

| Window | New min | New max |
|---|---|---|
| Initial (user inactive) | 45s (unchanged) | 90s (unchanged) |
| Follow-up (1 ignored proactive) | 25s | 50s |
| Backoff (3 consecutive ignored) | 45s | 60s |

No proactive follow-up window exceeds **60 seconds**.

## 3. Files Changed

- `src/open_llm_vtuber/proactive_chat.py` — `ProactiveChatConfig` default values
- `src/open_llm_vtuber/config_manager/agent.py` — config schema defaults + `DESCRIPTIONS` docs
- `tests/test_proactive_chat.py` — updated default-contract + timing assertions, added new timing tests

## 4. Centralized Timing Location

Two centralized sources, updated identically:

- `ProactiveChatConfig` dataclass (`proactive_chat.py:70`) — runtime defaults
- `BasicMemoryAgentConfig` pydantic fields (`config_manager/agent.py:48-54`) — config schema defaults the scheduler is built from

`websocket_handler.py` reads these fields when constructing the config, so there are no scattered hard-coded intervals. `proactive_chat.py`'s `record_proactive_sent()` picks the follow-up vs backoff window based on the unchanged `ignored_before_backoff` threshold.

## 5. Initial Proactive Timing Unchanged

`initial_idle_min_seconds = 45`, `initial_idle_max_seconds = 90` — untouched. The first proactive after real inactivity still fires in 45–90s.

## 6. Ignored Follow-up Timing

`followup_idle_min_seconds = 25`, `followup_idle_max_seconds = 50`. Applies to any proactive send below the ignored threshold (all 1st/2nd consecutive ignored messages).

## 7. Backoff Timing

`backoff_min_seconds = 45`, `backoff_max_seconds = 60`. Applies once `consecutive_ignored_proactive >= ignored_before_backoff (3)`.

## 8. Ignored Threshold Unchanged

`ignored_before_backoff = 3` — exactly as before. Only the time window changed.

## 9. User-Activity Reset Behavior

`record_user_activity()` is untouched. A real user message still resets the ignored counter to 0, bumps `activity_revision`, cancels pending eligibility, and restarts the timer at the initial **45–90s** window. The shorter 25–50s follow-up only applies after Mili already proactively spoke and the user stayed silent.

## 10. Turn-Cue Compatibility

`PROACTIVE_TURN_CUE`, `chat_proactively()`, ephemeral cue injection, provider request shape, `message_count` logging, `semantic_auto`, and `forced_ignored_question` are all untouched. This patch only decides *when* Mili speaks again, not *what* she says.

## 11. Semantic Compatibility

Semantic Proactive Selection v3, ignored-question priority, ignored-statement handling, topic selection, memory relevance, relationship context, and prompt wording are unchanged.

## 12. Tests Updated

- `test_configuration_defaults_match_runtime_contract` — expected defaults tuple now `(45, 90, 25, 50, 3, 45, 60)`
- `test_followup_and_three_ignored_backoff_ranges` — now uses a dedicated machine with mocked values inside the new ranges; asserts 25–50 for the first two follow-ups and 45–60 for backoff

## 13. Tests Added

- `test_initial_proactive_min_max_unchanged` — first proactive stays exactly 45 and 90s
- `test_no_followup_window_exceeds_60s` — follow-up/backoff windows both cap at 60s

## 14. Targeted Test Results

`python -m unittest tests.test_proactive_chat -q` → **106 OK**

## 15. Full Regression Results

`PYTHONPATH=src python -m unittest discover -s tests -q` → **217 OK**

## 16. Ruff

All checks passed.

## 17. Compileall

`compileall -q src/open_llm_vtuber` — no output (clean).

## 18. Git Diff Check

`git diff --check` — clean.

## 19. Commit Hash

(see git log; commit below)

## 20. Live Test Procedure

1. **Test A** — Send a normal message, then go silent → first proactive in **45–90s**.
2. **Test B** — Don't reply → second proactive in **25–50s**.
3. **Test C** — Stay silent → third proactive in **25–50s**.
4. **Test D** — Stay silent past the ignored threshold → next backoff proactive in **45–60s**.
5. **Test E** — Reply normally, then go silent → next proactive back to **45–90s**.

## 21. Remaining Limitations

- Randomized timing means individual observed intervals vary within the configured ranges; the ranges themselves are what this patch changed.
- No change to ignored-counter semantics, semantic direction, or the turn-cue fix — those shipped in earlier patches.