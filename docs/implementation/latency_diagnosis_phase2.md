# Latency Diagnosis Phase 2

Patch ini murni **diagnosis**: tidak ada perubahan model, provider, persona,
sampling, context window, relationship/memory, rolling summary, MCP, Live2D,
TTS provider, atau visual frontend. Tidak ada timeout/retry yang diubah.

## Files Changed

Backend (`Open-LLM-VTuber`):

| File | Perubahan |
|---|---|
| `src/open_llm_vtuber/request_latency.py` | Ditulis ulang: phase marks (monotonic), attempt tracking, outcome, `unattributed_ms`, `largest_gap_*`, bottleneck classifier v2, `AttemptTrackingTransport` (httpx) |
| `src/open_llm_vtuber/agent/stateless_llm/openai_compatible_llm.py` | Pakai transport pelacak attempt, mark `provider_prepare`/`first_chunk`/`stream_completed`, kategorisasi error (timeout/connection/rate_limit/api), audit timeout+retry saat init |
| `src/open_llm_vtuber/agent/agents/basic_memory_agent.py` | Mark `context_build_*`, `summary_*`, `tool_*`; ukur `metadata_save_ms` (summary metadata) & `character_state_save_ms` |
| `src/open_llm_vtuber/conversations/single_conversation.py` | Mark `agent_*`, `websocket_first/final_output`; ukur `history_save_ms`; inject `request_id` ke payload tool; outcome interrupt/error; `tts_enqueue_ms` |
| `src/open_llm_vtuber/conversations/conversation_utils.py` | Ukur `playback_wait_ms` di sekitar `wait_for_response("frontend-playback-complete")`; inject `request_id` ke audio path |
| `src/open_llm_vtuber/conversations/tts_manager.py` | Ukur **synthesis aktual** (`tts_synthesis_ms`) di sekitar `_generate_audio`; inject `request_id` ke payload audio |
| `tests/test_request_latency.py` | Test sintetis A–J + regression |

Frontend (`Open-LLM-VTuber-Web`):

| File | Perubahan |
|---|---|
| `src/renderer/src/utils/chat-latency.ts` | Mark `firstVisibleTextPerfMs` / `firstAudioPerfMs`; metrik `frontend_send_to_first_token_ms`, `first_token_to_first_visible_text_ms`, `first_token_to_first_audio_ms` |
| `src/renderer/src/services/websocket-handler.tsx` | Korelasi payload audio per `request_id` (`markFrontendPayload`) |

Bundle production di-deploy ke `frontend/` (submodule).

## Request Lifecycle Instrumented

Setiap request kini punya `request_id` tunggal (`chat-YYYYMMDD-HHMMSS-xxxxxx`
atau UUID dari frontend) yang sama dari `user_send` → WebSocket → conversation
handler → agent → provider → TTS → frontend `response_complete`. Semua durasi
memakai `time.perf_counter()` (monotonic); wall-clock hanya untuk pembacaan
manusia.

Fase yang dicatat (yang tidak terpakai = `None`, bukan `0.0`):

```
request_received
  ├─ conversation_prep_ms     (received → agent_start: signals, ASR, store user msg)
  ├─ agent_stream_ms          (agent_start → agent_end)
  │    ├─ context_build_ms    (budgeting + inject summary)
  │    ├─ summary_ms          (rolling summary, terisolasi)
  │    ├─ tool_ms             (MCP/tools)
  │    ├─ provider_prepare_ms
  │    ├─ provider_headers_ms → provider_first_chunk → provider_first_content_token
  │    └─ provider_generation_ms (first content → stream end)
  ├─ tts_enqueue_ms + tts_synthesis_ms + tts_wait_ms   (audio pipeline)
  ├─ playback_wait_ms         (tunggu frontend-playback-complete)
  └─ post_processing_ms       (history_save, character_event, metadata_save, cleanup)
```

Plus flag boolean: `provider_call_expected`, `provider_started`,
`provider_headers_received`, `provider_first_chunk_received`,
`provider_first_token_received`, `provider_stream_completed`, dan
`provider_attempt_count` + rincian per attempt.

## New Metrics

Per request (log `[LLM TRACE]` satu baris + event `latency-event` ke
frontend):

- `request_id`, `request_outcome`, `error_category`, `interrupted`,
  `client_disconnected`
- `conversation_prep_ms`, `agent_stream_ms`, `context_build_ms`,
  `summary_ms`, `tool_ms`, `character_event_ms`
- `tts_enqueue_ms`, `tts_synthesis_ms`, `tts_wait_ms`, `playback_wait_ms`,
  `tts_total_ms`
- `history_save_ms`, `metadata_save_ms`, `character_state_save_ms`
- `response_processing_ms`
- `provider_prepare_ms`, `ollama_request_to_headers_ms`,
  `ollama_request_to_first_token_ms` (**TTFT provider**),
  `provider_first_chunk_to_content_ms`, `ollama_generation_ms`
  (`provider_first_chunk` vs `provider_first_content_token` dipisah — TTFT UX
  memakai content token)
- `provider_attempt_count`, `provider_retry_overhead_ms`
- `known_pipeline_ms`, `unattributed_ms`, `largest_gap_ms`,
  `largest_gap_from`, `largest_gap_to`
- `bottleneck_hint` (v2), `output_tokens_per_second_estimate`,
  `message_count`, `estimated_input_tokens`, `total_response_ms`

Rolling stats in-memory 20 request: `[LLM LATENCY STATS]` (backend) dan
`[FRONTEND LATENCY STATS]` (frontend).

Tidak ada isi chat, respons, persona, summary, memory, API key, atau header
yang dicetak.

## Provider Zero-Metric Bug

Data live:

```
total_response_ms=24858.73
ollama_request_to_first_token_ms=0.0
ollama_request_to_headers_ms=0.0
ollama_generation_ms=0.0
```

**Hasil inspeksi statis (belum dikonfirmasi live):** di Phase 1, semua nilai
provider yang "tidak pernah terjadi" dipaksa menjadi `0.0` — tidak bisa
dibedakan dari "provider menjawab dalam 0 ms". Kombinasi `total ≈ 24.8 s` +
ketiga metrik `0.0` paling konsisten dengan: **request provider dimulai
(`provider_request_ms` ter-set) tapi `create()` tidak pernah menerima headers**
— yaitu koneksi gagal/hang di lapisan OpenAI SDK (timeout koneksi default
`connect=5s`, `read=600s`, `max_retries=2` + backoff), lalu teks error di-yield.
Dengan 3 attempt × ±8 s ≈ 24.8 s. Path alternatif: interrupt sebelum headers
tiba, atau fase `summary` yang gagal sebelum tercatat — keduanya butuh trace
live untuk dipastikan.

Perbaikan instrumentasi: nilai yang tidak terjadi kini `None` (bukan `0.0`),
flag `provider_started/headers_received/first_chunk/first_token/stream_completed`,
`provider_attempt_count`, dan `request_outcome=provider_error/provider_timeout/
provider_cancelled/retry`. Setelah 5–10 chat live, trace akan langsung
menunjukkan path mana yang benar.

## Unattributed Latency

`unattributed_ms = total_response_ms − known_pipeline_ms`, di mana
`known_pipeline_ms` adalah jumlah fase eksklusif
(`conversation_prep + agent_stream + tts_wait + playback_wait + post`).
Karena fase-fase itu kontigu, `unattributed_ms` akan mendekati 0 — kekuatan
diagnosisnya justru di **`largest_gap_ms` + `largest_gap_from/to`**: gap
terbesar antara dua boundary yang tercatat, yang menunjuk langsung ke
"koridor gelap".

**Temuan inspeksi statis untuk gap 2–3 s pada request normal:** Phase 1 tidak
mengukur bagian `total_response_ms` ini:

1. **`wait_for_response(client_uid, "frontend-playback-complete")` tanpa
   timeout** — backend memblokir sampai browser selesai memutar audio (2–3 s
   audio ≈ gap yang diamati). Sekarang diukur sebagai `playback_wait_ms`.
2. **Tunggu synthesis TTS** (`asyncio.gather(*tts_manager.task_list)`) —
   sekarang `tts_synthesis_ms`/`tts_wait_ms`.
3. Penulisan history ke disk (`store_message` atomic JSON) — sekarang
   `history_save_ms`.

Dengan data live, angka 2–3 s itu harusnya muncul persis di
`playback_wait_ms`/`tts_wait_ms` — bukan lagi "unattributed".

## TTS Timing Audit

Benar dugaan: `tts_ms ≈ 1 ms` Phase 1 hanya mengukur **enqueue**
(`process_agent_output` → men-schedule task background), bukan synthesis.
Synthesis berjalan async (`TTSTaskManager._process_tts`), dan waktu tunggunya
(`asyncio.gather`) ikut masuk `total_response_ms` tanpa diukur.

Sekarang dipisah tegas:

- `tts_enqueue_ms` — biaya men-schedule (dulu `tts_ms`)
- `tts_synthesis_ms` — durasi aktual `async_generate_audio` (diukur di dalam
  task, bukan di main loop)
- `tts_wait_ms` — berapa lama main loop menunggu task synthesis selesai
- `playback_wait_ms` — tunggu frontend memutar audio
- `tts_total_ms` — gabungan

Boundary: `total_response_ms` **termasuk** synthesis + playback (request
dianggap selesai setelah audio habis diputar). Ini perilaku lama yang tidak
diubah; sekarang terlihat eksplisit.

## Retry / Timeout Behavior

Audit (tidak diubah):

- `AsyncOpenAI` (openai 2.15.0) default `timeout = Timeout(connect=5, read=600,
  write=600, pool=600)` detik, `max_retries = 2`. Nilai ini dilaporkan saat
  init dan tidak disentuh.
- Retry diimplementasikan **di lapisan SDK** (re-issue request). Karena itu
  tiap attempt melewati transport httpx lagi → `AttemptTrackingTransport`
  mencatat tiap attempt (`attempt=1/2/3`, `phase`, `started`, `headers`,
  `stream_end`, `error`) sebagai baris `[LLM ATTEMPT]`.
- Interpretasi data live Phase 1: spike `ollama_request_to_headers_ms=8291.76`
  konsisten dengan attempt pertama gagal (±8 s) lalu retry sukses; kasus
  `total=24858` + metrik 0 konsisten dengan semua attempt gagal (error
  koneksi/timeout) → kini akan tercatat `provider_attempt_count=3`,
  `request_outcome=provider_error`, `bottleneck_hint=provider_retry` atau
  `provider_timeout`.

## Tests

| Test | Hasil |
|---|---|
| TEST A — provider normal (context 5ms, TTFT 500ms, gen 300ms) | PASS — `unattributed_ms` kecil, bukan `unattributed` |
| TEST B — provider spike TTFT 8s | PASS — `bottleneck_hint=provider_ttft` |
| TEST C — gap tersembunyi 2.5s setelah provider | PASS — `largest_gap≈2500`, `bottleneck_hint=unattributed` |
| TEST D — retry (attempt 1 timeout 10s, attempt 2 sukses 500ms) | PASS — `provider_attempt_count=2`, `outcome=retry`, `hint=provider_retry` |
| TEST E — provider tidak pernah start (total 20s) | PASS — metrik provider `None` (bukan 0), fase pre-provider terlihat |
| TEST F — client disconnect | PASS — `request_outcome=client_disconnect` |
| TEST G — interrupt | PASS — `request_outcome=interrupted` |
| TEST H — TTS async | PASS — enqueue (1.2ms) vs synthesis (850ms) tidak tercampur |
| TEST I — history persistence lambat | PASS — `history_save_ms` terlihat |
| TEST J — streaming regression | PASS — event `first-token` dikirim seketika, tidak di-buffer |
| Regression backend (test context window, summary, final integration, ollama cloud, relationship, architecture) | PASS — 82/82 |
| Backend compile / ruff | PASS |
| Frontend typecheck | PASS — 587 error baseline (0 error baru di file yang disentuh) |
| Frontend production build + deploy ke `frontend/` | PASS — bundle memuat marker baru |

## Live Data Analysis (log 2026-08-27, 14 request, instrumentasi Phase 1)

### Kasus 24.8 s terpecahkan: interrupt + provider hang

Konteks log di sekitar request `total_response_ms=24858.73`:

```
12:13:04.831 | 🛑 Conversation task was successfully interrupted
12:13:04.835 | New Conversation Chain started! (pesan 1 karakter)
12:13:04.851 | 🤡👍 Conversation cancelled because interrupted
12:13:04.852 | [LLM LATENCY] ... total_response_ms=24858.73 ollama_*=0.0
```

Request 24.8 s itu **bukan** error backend/TTS/context: pengguna mengirim pesan,
provider `create()` menggantung menunggu headers ±24.8 detik (default read
`timeout=600s` tidak pernah tercapai), lalu pengguna interrupt. Karena headers
tak pernah diterima, nilai provider di Phase 1 dirender `0.0` (None dipaksa
jadi 0) — persis ambiguitas yang dilaporkan. Interrupt dikonfirmasi oleh baris
"Conversation cancelled because interrupted" 1 ms sebelum log latency.

### Spike 8.29 s = cold start / model load Ollama Cloud

11 detik setelah request di atas, request berikutnya menerima headers setelah
8.29 s (`ollama_request_to_headers_ms=8291.09`) lalu generation hanya 482 ms.
Artinya model baru selesai dimuat/keluar antrian provider dalam jendela itu.
Request pertama (12:11:52, TTFT 1396 ms) juga lebih lambat dari rata-rata —
pola cold start/queue, bukan backend.

### Gap 2–3 s pada request normal = pipeline audio

11 request normal (12:19–12:56):

| TTFT (ms) | Gen (ms) | Total (ms) | Gap tak terukur ≈ audio (ms) |
|---|---|---|---|
| 786.8 | 213.4 | 3635.5 | 2629 |
| 546.9 | 200.3 | 3081.3 | 2332 |
| 522.3 | 212.4 | 4174.4 | 3438 |
| 532.9 | 428.7 | 3085.8 | 2122 |
| 539.9 | 238.2 | 3545.3 | 2766 |
| 475.0 | 175.2 | 2978.5 | 2327 |
| 695.4 | 198.6 | 4753.3 | 3858 |
| 506.5 | 396.3 | 2688.3 | 1784 |
| 767.2 | 377.8 | 4001.6 | 2855 |
| 542.8 | 359.8 | 3260.7 | 2356 |
| 521.8 | 397.5 | 2626.0 | 1705 |

Rata-rata: TTFT **565 ms**, generation **280 ms**, total **3439 ms**, gap
±**2.5 s**. Provider hanya ±25% dari total request; ±75% adalah waktu setelah
token terakhir: **TTS synthesis + tunggu frontend memutar audio** (dua hal yang
dulu tidak diukur; kini `tts_synthesis_ms`/`tts_wait_ms`/`playback_wait_ms`).
`tts_ms` Phase 1 (±1 ms) memang hanya enqueue, bukan synthesis.

### Kesimpulan live

1. **Bottleneck utama request normal = audio pipeline (±2.5 s/request)**, bukan
   model/provider — LLM hanya ±0.85 s.
2. **Kasus lambat ekstrem (24.8 s / 8.29 s) = provider (Ollama Cloud) cold
   start / model load / queue**; yang 24.8 s diperparah user interrupt sebelum
   headers tiba.
3. TTFT normal 475–767 ms (warm) — bukan penyebab "Sedang berpikir..." lama.

Untuk membuktikan poin 1 dengan angka persis, jalankan kembali test dengan
kode Phase 2 (log `[LLM TRACE]` memisahkan `tts_synthesis_ms`,
`tts_wait_ms`, `playback_wait_ms`).

## Live Test Command

Setelah 5–10 chat normal (plus 1 chat setelah diam ±10 menit untuk deteksi
cold start), jalankan:

```bash
cd /root/waifu/Open-LLM-VTuber && grep -E "\[LLM TRACE|\[LLM ATTEMPT|\[LLM LATENCY STATS" logs/*.log* | tail -120
```

Cara membaca cepat:

- `largest_gap_from/to` + `playback_wait_ms` / `tts_wait_ms` besar →
  bottleneck audio pipeline (bukan provider).
- `ollama_request_to_first_token_ms` besar → TTFT provider (cold start /
  queue / load model).
- `provider_attempt_count > 1` + `[LLM ATTEMPT] error=...` → retry provider.
- `request_outcome=provider_error` / `provider_timeout` dengan metrik provider
  `None` → request yang dulu tampak "24.8s misterius".
- `unattributed_ms` besar → masih ada fase yang belum tercatat; kirim trace-nya.

## Status

**LATENCY DIAGNOSIS PHASE 2 SELESAI**

Kesimpulan statis (perlu konfirmasi live): gap 2–3 detik pada request normal
hampir pasti = TTS synthesis + **tunggu frontend memutar audio**
(`frontend-playback-complete` tanpa timeout), bukan model. Kasus `24.8 s`
dengan metrik provider 0 = request provider gagal/timeout di lapisan SDK
(headers tidak pernah diterima) yang dulu tersembunyi karena nilai `None`
dipaksa jadi `0.0`.
