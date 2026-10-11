# Task B — Attachment Memory persistence

Status: **AUDIT + 3 FIXES, TESTED, DEPLOYED, PRODUCTION_VERIFIED (health).**

## Audit outcome

- Authoritative store: `attachment_memory/<conf_uid>.json` (atomic + RLock,
  fail-soft). Record schema `{id,filename,mime_type,source,content_hash,
  received_at,session_uid,request_id,status,summary,summary_model,created_at,tz}`.
- Image bytes are never persisted (only the transport string is hashed).
- Cross-session retrieval works (key is `conf_uid`; session only prioritises
  the 900s fallback). There is deliberately **no per-user ownership** — the
  store is character-scoped.
- Purge deletes the record + episodic events by id and reports the rest
  honestly (`complete=false`); transcripts/summaries/facts/world quotes
  intentionally survive (documented limit, not changed).

## Fixes (commit `75dc1ed`)

1. **Unstable record identity (PROVEN).** `_valid_record` minted a fresh
   random uuid on every load for a missing id, making the row undeletable.
   Now a stable content-derived id. Tests: `test_row_missing_an_id_gets_a_stable_identity`
   (stable across loads + actually deletable).
2. **Load validation weaker than write validation (PROVEN).** Any `mime_type`
   loaded while appends require `image/*`. Load now mirrors write.
   Test: `test_row_with_non_image_mime_is_rejected_on_load`.
3. **Failed wipe reported as success (PROVEN).** `clear_attachment_memories`
   returned 0 for both empty and failed-save; the handler always sent
   `success:true`. Now -1 on failure, handler maps to `success:false`.
   Test: `test_failed_wipe_reports_minus_one_not_zero`.
4. **Stale selection cache after purge (PROVEN at runtime by audit).**
   The proactive path reused the last user turn's cached selection, so a
   purged summary was re-injected. purge/delete/clear now invalidate.
   Test: `test_selections_empty_after_purge` asserts the purge itself clears
   the cache (manual reset removed).

`tests/test_attachment_memory.py`: 75 → 79, all pass.

## Not changed

Interrupted-capture rows (no record on `CancelledError`), provider-absent
paths, attachment↔summary index matching, episodic substring fallback width:
audited, left for dedicated follow-ups with turn-level reproductions.
