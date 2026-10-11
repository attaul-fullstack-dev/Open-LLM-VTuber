# Task J — Custom-avatar pipeline groundwork

Status: **AUDIT + 1 FIX (PROVEN startup crash), TESTED, DEPLOYED.**

## Fix (commit `47284f4`, `service_context.py`)

A missing/broken Live2D model left `live2d_model = None` while
`construct_system_prompt` dereferenced `.emo_str` unconditionally; the
`AttributeError` propagated into `run_server.py`, which exits the process —
**one bad avatar name bricked the whole launch** despite `init_live2d`'s
"proceed without Live2D" promise. Now fail-soft: the emomap placeholder is
dropped and the remaining instructions stand. Tests:
`tests/test_avatar_startup_safety.py` (2: with and without a model).

## Audit (not changed, see blockers for follow-ups)

- Resolution is name → `model_dict.json` → URL string; no server-side
  existence/extension check; frontend has three correct barriers and an
  `onError` letter fallback (not touched).
- Same crash exists on mid-session config switch with a half-applied engine
  swap (J-2): noted, not fixed (needs a session-state transaction design).
- Shipped `characters/en_nuke_debate.yaml` points at non-existent
  `shizuku-local`: still crashes-by-config at selection time (now without
  taking the server down at startup only if it is the boot character).
- Settings-open scale inflation (J-5): noted, unpatched.

No claim is made about custom avatars or Visual Self-Identity being complete;
only the crash groundwork was laid.
