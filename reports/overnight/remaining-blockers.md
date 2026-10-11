# Remaining Blockers — overnight mission

Nothing in this list blocks the deployed work; all of it is explicitly
out-of-scope verification or follow-up.

1. **Visual/browser acceptance of the new bundle — UNVERIFIED.**
   `main-DsPS_peT.js` is built, published, and served (200 + correct asset
   reference), but no visual-browser run confirmed the footer mute toggle,
   the attachment dialog, or avatar/avatar-scale behaviour on screen. The
   footer fix, however, is covered by behavioural + contract tests. Next: a
   manual or automated browser pass when tooling is available.

2. **Preference Derivation Phase 2A manual acceptance — NOT PASSED, NOT BYPASSED.**
   The mission gate holds: no claim is made that Phase 2A passed, Phase 2B is
   not activated, and there is no Phase 2B feature flag in the code (the audit
   confirmed the phase labels are comments/docstrings only). Only
   independently verifiable detector robustness fixes (roleplay rejection,
   one-per-category reader) were shipped.

3. **WS 1006 root cause — NOT PROVEN.** Zero abnormal-closure evidence exists
   in the accessible logs; the disconnect/reconnect path is already hardened
   (detached turns, subscriber registry, exponential-backoff reconnect). Only
   independently justified reliability work was shipped (the mute-wedge fix
   and the incident-recorder test unlock). A 1006 reproduction with transport
   logs is still needed before any deeper claim.

4. **Cross-client memory staleness — PARTIAL.** Each WS connection holds its
   own agent snapshot; a deletion through client A is not visible to client B
   until B reloads history. Same-agent paths were fixed (refetch on write in
   the modules touched). A broadcast/invalidation across live connections is a
   larger architectural change and was deliberately left out.

5. **Derived-data deletion limits — DOCUMENTED, NOT CHANGED.** Purge removes
   the attachment record + its episodic events by id and reports the rest
   honestly (`complete=false`). Transcript rows, rolling summaries, character
   facts, and world-state quotes of a deleted attachment intentionally survive
   (the UI must surface `purge.complete`). Widening deletion to those domains
   would be a destructive migration and is out of scope.

6. **Group-turn `ai-final` emission (F11), history-list cleanup of in-flight
   sessions (F12), proactive `skip_history` user-row asymmetry (F13) — PROBABLE,
   unverified.** Found by audit, not reproduced with an executable turn, so no
   code was changed. Each needs a turn-level reproduction before a fix.

7. **Proactive budget keyed per-client (H-ISSUE 5), cross-process lost-update
   on world state (H-ISSUE 4), save-failure retry semantics (H-ISSUE 3) —
   PROBABLE, unverified by execution.** Documented with file:line evidence;
   left untouched because the fixes (budget re-keying, compare-and-swap on
   state_version) change live accounting/concurrency semantics and deserve a
   dedicated, user-visible test window rather than an overnight push.

8. **Avatar pipeline follow-ups — NOT DONE.** The startup crash is fixed and
   guarded, but the shipped character `en_nuke_debate.yaml` still points at a
   non-existent model (`shizuku-local`), and the settings-scale inflation
   (J-5) is unpatched. Both are safe to fix next: point the character at an
   existing model (or add the missing one) and make `setModelInfo` idempotent
   to scale.
