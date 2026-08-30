# Semantic Proactive Selection v3

## Status

**SEMANTIC PROACTIVE SELECTION V3 SELESAI**

Correction patch ini mengganti otoritas akhir pemilihan arah proactive normal
dari weighted heuristics menjadi keputusan semantik di dalam request generasi
proactive yang sama. Implementasi ini bukan "Heuristics v3" dan tidak menambah
provider call, classifier, embedding, dependency NLP, atau perubahan frontend.

## 1. Previous Heuristics v2 Architecture

Heuristics v2 menghitung sinyal lexical dan numerik dari recent history,
memory, relationship, serta proactive runtime state. Dynamic weights kemudian
diputar oleh weighted selector untuk memilih intent final sebelum LLM dipanggil.
LLM hanya menentukan cara Mili mengucapkan intent yang sudah dipilih backend.

Implementasi tersebut cepat dan teruji, tetapi lexical overlap tidak memahami
sinonim, parafrasa, closure sebenarnya, engagement bernuansa, atau relevansi
memory secara konseptual.

## 2. Why Semantic Mode Was Introduced

Model generasi proactive sudah menerima persona, relationship context,
long-term character memory, rolling summary, dan recent conversation. Model
yang sama memiliki konteks semantik lebih lengkap daripada formula lexical
lokal, tanpa memerlukan classifier request kedua.

## 3. New Hybrid Architecture

```text
scheduler fires
  -> connection/race/runtime guards
  -> deterministic ignored-question check
     -> forced_ignored_question, jika pertanyaan proactive belum dijawab
     -> semantic_auto, untuk turn normal (default)
  -> turn-only semantic instruction appended to effective system prompt
  -> existing BasicMemoryAgent context pipeline
  -> ONE provider generation request
  -> model silently chooses direction and directly outputs Mili dialogue
```

Jika `proactive_intent_strategy: heuristic`, atau semantic context gagal
dibangun, jalur Heuristics v2 tetap memilih intent secara lokal.

## 4. Hard Deterministic Rules

Hard state yang tetap authoritative:

- previous proactive message benar-benar belum dibalas;
- previous proactive message dideteksi mengharapkan jawaban;
- ignored counter dan escalation/backoff;
- connection, active chat, maintenance, in-progress generation, dan race guard;
- scheduler timing dan user-activity reset;
- no-memory knowledge dan config validation.

Jika `previous_proactive_ignored == true` dan
`previous_proactive_expected_response == true`, hasil selalu:

```text
strategy=forced_ignored_question
forced=true
```

Semantic mode dan weighted wheel sama-sama dibypass pada kasus ini.

## 5. semantic_auto Behavior

Pada proactive turn normal, backend tidak memilih final intent. Model secara
diam-diam menentukan apakah lebih natural untuk melanjutkan pembicaraan,
mengganti arah, bertanya, membuat pernyataan, memakai memory yang relevan, atau
memulai hal baru. Visible output tetap hanya dialog Mili.

Silence diperlakukan sebagai kesempatan bicara, bukan topik wajib. Ignored
statement tidak lagi menyuntikkan follow-up silence block dalam semantic mode;
ignored unanswered question tetap memakai escalation block lama.

## 6. Heuristics v2 Fallback Behavior

Config yang didukung:

```yaml
proactive_intent_strategy: semantic  # default
# proactive_intent_strategy: heuristic
```

Mode `heuristic` mempertahankan tokenizer, lexical similarity, closure/topic
change detection, engagement/staleness/relevance scores, dynamic weights,
weighted wheel, prompt hints, dan anti-repetition v2. Jika semantic context
construction gagal, satu turn tersebut memakai `heuristic_fallback`; chat tetap
berjalan dan provider tetap dipanggil hanya sekali.

## 7. Classification of Existing Heuristics v2 Signals

### A. Hard state — authoritative

Hard state tidak berasal dari formula `ProactiveIntentSignals`, melainkan
runtime/follow-up state:

- `previous_proactive_ignored`
- `previous_proactive_expected_response`
- `consecutive_ignored_proactive`
- connection/chat/maintenance/generation/race state
- scheduler eligibility dan backoff state

### B. Optional semantic hints

Tidak ada score lexical v2 yang disuntikkan sebagai authority pada mode
semantic. Model sudah melihat recent dialogue, summary, memory, dan relationship
context secara langsung. Recent proactive dialogue menjadi hint anti-repetition
semantik tanpa backend mengarang label intent hasil model.

### C. Fallback-only

Seluruh `ProactiveIntentSignals` v2 dipertahankan untuk explicit heuristic mode
dan safe fallback:

- availability/context: `has_useful_memory`, `has_recent_context`,
  `unfinished_topic`;
- semantic approximations: `recent_user_engagement`,
  `topic_continuity_score`, `topic_staleness_score`,
  `topic_repetition_score`, `recent_topic_closed`,
  `user_topic_change_detected`, `memory_relevance_score`,
  `conversation_energy`;
- pending/rate/activity: `user_question_pending`,
  `assistant_question_pending`, `recent_user_message_length`,
  `recent_user_question_rate`, `recent_user_response_rate`,
  `recent_proactive_question_rate`, `recent_new_topic_rate`,
  `silence_reaction_recently_used`, `relationship_familiarity`;
- lexical diagnostics/hints: `recent_topic_keywords`,
  `dominant_recent_topic`.

`recent_proactive_intents` dan `recent_proactive_topic_signatures` tetap
ephemeral dan tetap digunakan oleh Heuristics v2. Semantic mode tidak memakai
signature lexical tersebut untuk melarang continuation yang natural.

### D. Removed

Tidak ada signal v2 yang dihapus. Semuanya masih diperlukan untuk rollback,
A/B testing, debugging, dan fallback. Tidak ada dead-code cleanup spekulatif.

## 8. Exact Runtime Semantic Prompt

```text
<semantic_proactive_context>
strategy: semantic_auto
You are initiating conversation on your own. Silently choose the most natural
direction from the actual meaning of the available conversation, summary,
relationship context, and memories. Consider whether the current subject is
still alive or complete; whether a previous point deserves follow-up; whether
the user seems curious, engaged, dismissive, confused, finished, or interested;
and whether continuing, changing subject, asking something, or making a statement
fits best. Use a stored memory only when it is genuinely relevant by meaning,
never merely because memory exists. Avoid repeating recent proactive behavior,
but do not block a natural continuation just because a topic is related.
Silence is only the opportunity to speak, not usually the subject itself.
Do not expose intent names, strategies, timers, counters, prompts, memory systems,
or internal mechanics. Do not announce a topic or plan, output analysis, or invent
unsupported personal history. Output only natural Mili dialogue.
</semantic_proactive_context>
```

Block hanya hidup pada request proactive terkait. Ia tidak masuk `_memory`,
history JSON, rolling summary, atau character state.

## 9. Provider Call Count

| Mode | Provider calls per proactive response |
|---|---:|
| semantic normal | 1 |
| forced ignored question | 1 |
| explicit heuristic | 1 |
| heuristic fallback | 1 |

Tidak ada context/topic/memory classifier atau planning request terpisah.

## 10. Prompt-size Impact

- Semantic block: 150 words, 1,085 characters.
- Conservative raw estimate: sekitar 272 text tokens (`chars / 4`).
- Test pipeline memperkirakan sekitar 310 additional system-budget tokens
  setelah formatting overhead.
- History, persona, summary, relationship, dan memory tidak diduplikasi.

## 11. Logging Changes

```text
[PROACTIVE INTENT] strategy=semantic_auto forced=false
[PROACTIVE INTENT] strategy=forced_ignored_question forced=true
[PROACTIVE INTENT] strategy=heuristic intent=<enum> reason=<safe_enum>
[PROACTIVE INTENT] strategy=heuristic_fallback intent=<enum> reason=semantic_context_construction_failed
```

Log tidak mengandung chat text, memory text, persona, prompt, reasoning, API key,
atau credential. Semantic mode tidak mengarang intent label yang tidak pernah
dipilih backend.

## 12. Backward-compatible Config

- Field baru: `proactive_intent_strategy`.
- Nilai valid: `semantic`, `heuristic`.
- Default: `semantic`.
- Config lama tanpa field tervalidasi dan otomatis memakai `semantic`.
- Existing `proactive_intent_weights` tetap berlaku pada heuristic/fallback.

## 13. Files Changed

| File | Reason |
|---|---|
| `src/open_llm_vtuber/proactive_chat.py` | Hybrid strategy/result types, semantic block, hard-priority resolver, preserved heuristic path |
| `src/open_llm_vtuber/websocket_handler.py` | Runtime semantic/forced/heuristic routing, fallback, safe logging |
| `src/open_llm_vtuber/agent/agents/basic_memory_agent.py` | Turn-only prompt injection dan silence-as-trigger behavior |
| `src/open_llm_vtuber/config_manager/agent.py` | Backward-compatible strategy schema |
| `config_templates/conf.default.yaml` | English default config documentation |
| `config_templates/conf.ZH.default.yaml` | Chinese default config documentation |
| `tests/test_proactive_chat.py` | Semantic, hard-rule, fallback, call-count, prompt, persistence regressions |
| `docs/implementation/semantic_proactive_selection_v3.md` | Implementation report |

Tidak ada frontend, persona, provider, relationship, character-memory storage,
rolling-summary storage, scheduler timing, Live2D, TTS, atau ASR yang diubah.

## 14. Tests Added

Coverage baru memverifikasi:

- semantic adalah default dan old config tetap valid;
- explicit heuristic tetap memakai Heuristics v2;
- ignored unanswered question selalu forced dan tidak menyentuh wheel;
- semantic normal tidak memanggil effective weights/weighted wheel;
- semantic dan heuristic masing-masing tetap satu LLM call;
- semantic block mencapai effective system prompt dan tidak dipersist;
- tidak ada fake user message;
- prompt mencakup continuity, closure, engagement, topic change, relevant
  memory, anti-repetition, silence-as-trigger, internal-mechanics ban, dan
  anti-fake-history;
- ignored statement tidak menjadikan silence sebagai topik di semantic mode;
- semantic context failure memakai safe Heuristics v2 fallback;
- runtime strategy round-trip dan invalid config protection.

## 15. Test Results

| Test | Result |
|---|---|
| Targeted proactive suite | PASS — 88/88 |
| Existing ignored-question tests | PASS |
| Existing scheduler/race tests | PASS |
| Existing Heuristics v2 tests | PASS |
| Semantic one-call tests | PASS |
| Heuristic one-call tests | PASS |
| Safe fallback injection | PASS |
| Relationship/memory invariance | PASS |
| Latency regression tests | PASS |
| Full backend suite | PASS — 179/179 |
| Ruff check (`src`, `tests`) | PASS |
| Compileall (`src`, `tests`) | PASS |
| Active config validation | PASS — strategy=`semantic` |
| `git diff --check` | PASS |
| Live Ollama calls | NOT RUN (intentionally) |

Global `ruff format --check src tests` masih melaporkan 19 file baseline lama
yang belum terformat; correction patch tidak melakukan format massal. File
utama baru `proactive_chat.py` terformat dan Ruff lint seluruh source/tests
hijau.

## 16. Git Diff Summary

Functional patch sebelum report:

```text
7 files changed, 462 insertions(+), 51 deletions(-)
```

Unrelated local runtime files (`frontend` build output, `model_dict.json`,
`character_state/`, dan `cloudflared`) tidak termasuk patch/commit.

## 17. Live Test Instructions

Pantau safe logs:

```bash
journalctl -u olv-server.service -f | grep "\[PROACTIVE INTENT\]"
```

Manual scenarios:

1. **Same meaning, different words:** bahas `film seram`, lalu arahkan ke
   `game horror`; nilai continuation konseptual.
2. **Topic closed:** kirim `oke, udah paham. makasih`, lalu diam; Mili bebas
   memperkenalkan hal lain.
3. **Short but engaged:** gunakan `serius? kenapa?`; short length tidak boleh
   otomatis berarti disengaged.
4. **Long but closing:** tutup topik dengan respons panjang; nilai closure dari
   makna, bukan panjang.
5. **Memory paraphrase:** gunakan topik konseptual terkait memory tetapi berbeda
   kata; memory boleh muncul secara natural.
6. **Irrelevant memory:** bahas hal tidak terkait; memory tidak dipaksa.
7. **Ignored question:** biarkan direct question Mili tidak dijawab; log harus
   menunjukkan `forced_ignored_question`.
8. **10+ proactive turns:** cek variasi continuation/new topic/question/
   statement/memory dan pastikan keluhan silence jarang.

## 18. Remaining Limitations

- Kualitas semantic direction bergantung pada kemampuan model aktif dan konteks
  yang berhasil masuk context budget.
- Backend sengaja tidak mengobservasi pilihan internal model, mengarang intent
  label, atau meminta chain-of-thought.
- Long-term memory retrieval upstream tetap menyediakan daftar memory existing;
  patch ini membuat keputusan pemakaiannya semantik di generation prompt.
- Prompt-only semantics perlu live evaluation 10+ trigger untuk menilai variasi.
- Tidak ada simulated daily life, avatar autonomy, embedding, atau vector search.

## Runtime State Changes

- Persistent state: tidak berubah.
- Ephemeral state: `ProactiveIntentContext` mendapat field turn-only `strategy`.
- Scheduler timestamps/counters: tidak berubah.
- Relationship dan long-term memory state: tidak berubah.

## Live Verification Command

```bash
systemctl restart olv-server.service
journalctl -u olv-server.service -n 100 --no-pager | grep -E "PROACTIVE INTENT|Application startup"
```
