# Thinking Latency Patch — Laporan

Tanggal: 2026-08-27
Lingkup: (1) Lokalisasi UI ke Bahasa Indonesia, (2) Instrumentasi latency per tahap untuk mendiagnosis status "Thinking..." yang kadang sangat lama.
Tidak mengubah: model Ollama Cloud, persona, temperature, top_p, max_tokens, context window. Patch ini fokus mengukur, bukan memperbaiki dengan tebakan.

---

## 1. UI Translation

Semua string lewat sistem i18n yang sudah ada (`src/renderer/src/locales/{id,en,zh}/translation.json`). Tidak ada hardcode di komponen.

| Key | en | id |
|---|---|---|
| `aiState.thinking-speaking` | thinking/speaking | **Sedang berpikir...** |
| `history.compact` | Compact conversation | **Ringkas percakapan** |
| `notification.compactSuccess` | Conversation compacted | **Percakapan berhasil diringkas** |
| `settings.agent.resetRelationship` | Reset relationship with Mili | **Reset hubungan dengan Mili** |
| `settings.agent.resetMemory` | Reset Mili's memory | **Hapus ingatan Mili** |
| `settings.agent.resetCharacterState` | Reset relationship and memory | **Reset hubungan dan ingatan** |
| `settings.agent.characterMemory` | Mili's Long-Term Memory | **Ingatan Jangka Panjang Mili** |
| `history.newChat` | New Conversation | **Percakapan Baru** |
| `toolCall.using` / `toolCall.used` (key baru) | X is using tool Y / used tool Y | **X sedang memakai Y / X memakai Y** |
| `history.loadOlder` (key baru) | Load older messages | **Muat pesan lama** |

Juga termasuk seluruh toast success/fail terkait conversation, relationship, dan memory — paritas key en↔id↔zh diverifikasi 100% via script.

Bug i18n yang diperbaiki:
- `use-interrupt.ts` sebelumnya membandingkan subtitle dengan hardcoded `'Thinking...'` sehingga deteksi gagal saat bahasa bukan Inggris. Kini pakai `t('aiState.thinking-speaking')`.
- Indikator tool call di panel chat (`chat-history-panel.tsx`) dan tombol "Muat pesan lama" kini pakai `t()` dengan interpolasi.

File locale yang ditambah/diperbaiki: key zh yang hilang (`imageCompressionQualityPlaceholder`, `imageMaxWidthPlaceholder`, `about.license`).

---

## 2. Runtime Model

- Provider: **Ollama Cloud** (`https://ollama.com/v1`)
- Model: **`gemma4:31b-cloud`**
- Agent: `basic_memory_agent`
- Rolling summary: aktif (`summary_min_new_messages=4`, target 320 / cap 384 token)
- MCP/tools: `time`, `ddg-search` aktif
- Temperature 0.8, top_p 0.9, max_tokens 384 — tidak disentuh patch ini.

Catatan: endpoint remote membuat preload/unload lokal Ollama dilewati otomatis (`OllamaLLM.is_local=False`), jadi pola cold start yang terukur adalah milik provider, bukan host lokal.

---

## 3. Instrumentation — Metrik yang Kini Tersedia

Modul utama:
- Backend: `src/open_llm_vtuber/request_latency.py` (+ wiring di `single_conversation.py`, `basic_memory_agent.py`, `openai_compatible_llm.py`, `conversation_handler.py`)
- Frontend: `src/renderer/src/utils/chat-latency.ts` (+ wiring di `use-text-input.tsx`, `websocket-service.tsx`, `websocket-handler.tsx`)

Metrik per request, dicetak sebagai log `[LLM LATENCY]` dan dikirim ke frontend sebagai event `latency-event`:

Frontend:
- timestamp `user_send`, `websocket_send`, `first_backend_status`, `first_token_received`, `response_complete`
- `frontend_user_send_to_backend_ms`, `frontend_to_backend_ms`, `frontend_to_first_backend_status_ms`

Backend per tahap:
- `context_build_ms`, `character_event_ms`, `tts_ms`
- `summary_ms` + `summary_triggered=true/false` (fase summary diisolasi lewat contextvar `latency_phase`, jadi tidak mencampur TTFT chat)
- `tool_ms` + `tool_used=true/false`
- `ollama_request_to_headers_ms`, `ollama_request_to_first_token_ms` (**TTFT provider**)
- `ollama_generation_ms`, `output_tokens_per_second_estimate`
- `message_count`, `estimated_input_tokens`, `total_response_ms`, `bottleneck_hint`

Keamanan logging:
- Hanya angka, status, model, provider, jumlah pesan, estimasi token.
- Tidak ada isi chat, respons, persona, ringkasan, karakter memory, API key, atau credential yang dicetak.

Streaming:
- Tidak ada buffering baru; token pertama tetap langsung di-yield.
- Event `first-token` dikirim ke frontend SEBELUM yield konten pertama.

Rolling stats in-memory (tanpa database, window 20 request):
- Backend: `[LLM LATENCY STATS] requests=… average_ttft_ms=… min_ttft_ms=… max_ttft_ms=… average_total_ms=…`
- Frontend console: `[FRONTEND LATENCY STATS]` dengan avg/min/max TTFT dan avg total.
- Manual compact dicatat terpisah: `[COMPACT LATENCY] total_ms=… messages_compacted=…` — tidak dicampur latency chat normal.

TTFT vs Generation Speed dipisah tegas:
- `ollama_request_to_first_token_ms` = waktu tunggu token pertama (request → first content chunk).
- `ollama_generation_ms` = kecepatan menulis setelah token pertama.
- `output_tokens_per_second_estimate` = estimasi dari panjang output / generation time.

---

## 4. Thinking Indicator

Sebelum:
- Subtitle memakai teks English; deteksi interrupt banding string hardcoded `'Thinking...'`.
- Tidak ada sinyal first-token dari backend → indikator bisa tampak bertahan lama meski model sudah mulai menjawab.

Sesudah:
- Subtitle "Sedang berpikir..." (locale `id`).
- Backend mengirim `latency-event first-token` begitu byte pertama konten model tiba → frontend langsung menghapus indikator dan mulai streaming.
- Target alur tercapai: send → Sedang berpikir... → FIRST TOKEN → indikator hilang → streaming text/audio tampil.

Status teknis tambahan ("Building context", dst.) sengaja tidak ditampilkan ke user; detail tetap tersedia di developer log/console.

---

## 5. Bottleneck Analysis

Belum dapat menentukan apakah bottleneck berasal dari model atau Ollama Cloud sampai user melakukan beberapa request live. Cara membacanya setelah data terkumpul:

| Pola | Interpretasi |
|---|---|
| `context_build_ms` kecil, `ollama_request_to_first_token_ms` besar | Delay di provider/model (queue/cold start/TTFT) |
| Request pertama sesudah idle punya TTFT jauh lebih besar dari berikutnya | Cold start provider |
| TTFT bervariasi drastis untuk context serupa | Provider queue/load |
| `summary_triggered=true` pada request lambat saja | Penyebab summarization, bukan chat provider |
| `tool_used=true` + `tool_ms` besar | MCP/tool pending response |
| `frontend_to_backend_ms` besar | Masalah WebSocket/jaringan lokal |
| TTFT kecil tapi generation lambat | Kecepatan output model |

---

## 6. Bug Tambahan yang Ditemukan & Diperbaiki (Regresi Test F)

Saat menjalankan test penuh, ditemukan bug pre-existing (bukan akibat instrumentasi):

- Lokasi: `src/open_llm_vtuber/agent/context_window.py` → `select_messages_for_context()`
- Gejala: rolling summary bisa TERBUANG dari request final saat budget context mepet, karena summary diurutkan sebagai turn paling tua dan greedy trimming `break` berhenti sebelum sempat menyertakannya (`summary_included=False`).
- Dampak nyata: Mili kehilangan ingatan jangka panjang persis pada percakapan paling panjang — momen ketika summary paling dibutuhkan.
- Perbaikan minimal: ruang untuk unit internal (summary) direservasi dulu sebelum pengisian turn transkrip (greedy terbaru-dulu + `break` dipertahankan), sehingga boundary ringkasan inkremental tidak berubah.
- Verifikasi: test integrasi stress konteks (yang semula FAIL) kini PASS; `test_conversation_summary` dan `test_context_window` tetap PASS semua.

---

## 7. File yang Diubah

Backend (Open-LLM-VTuber):
- `src/open_llm_vtuber/request_latency.py` — modul tracker (sudah ada dari sesi sebelumnya, kini terverifikasi ter-wire penuh)
- `src/open_llm_vtuber/conversations/single_conversation.py`, `conversation_handler.py`, `conversation_utils.py`, `group_conversation.py` — wiring metadata & lifecycle tracker
- `src/open_llm_vtuber/agent/stateless_llm/openai_compatible_llm.py` — provider_started / headers / first_token / generation / fase summary
- `src/open_llm_vtuber/agent/agents/basic_memory_agent.py` — timing context, summary, tool; isolasi fase
- `src/open_llm_vtuber/agent/context_window.py` — PERBAIKAN eviksi rolling summary
- `tests/test_request_latency.py` — test sintetis A/B/C/D(parsial)/generation separation

Frontend (Open-LLM-VTuber-Web) + hasil build:
- `locales/{en,id,zh}/translation.json` — translasi + key baru
- `utils/chat-latency.ts` — util latency frontend
- `hooks/footer/use-text-input.tsx`, `hooks/utils/use-interrupt.ts` — start timestamp, perbaikan compare subtitle
- `services/websocket-service.tsx`, `services/websocket-handler.tsx` — mark websocket send, handle `latency-event`
- `components/sidebar/chat-history-panel.tsx` — tool call indicator & load older via i18n
- Build web production tersinkron ke `Open-LLM-VTuber/frontend/` via `scripts/build_frontend.sh` (bundle memuat semua string id).

---

## 8. Cara Menguji Live (5–10 Chat)

1. Jalankan server seperti biasa (`run_server.py`) dan pastikan frontend bahasa Indonesia.
2. Lakukan 5–10 chat normal dengan Mili.
3. Tambahkan 1 chat SETELAH diam ±10 menit (untuk mendeteksi cold start provider).
4. Kirim hasil command ini untuk analisis:
   ```bash
   cd /root/waifu/Open-LLM-VTuber && grep -E "\[LLM LATENCY|\[LLM LATENCY STATS" logs/*.log* | tail -40
   ```
5. Opsional (sisi browser): buka DevTools console, filter `[FRONTEND LATENCY]`.

---

## 9. Tests

| Test | Hasil |
|---|---|
| Indonesian Thinking UI | PASS |
| First-token indicator (Test D) | PASS |
| Backend latency measurement | PASS |
| Provider TTFT measurement (Test A) | PASS |
| Backend-context bottleneck (Test B) | PASS |
| Generation latency measurement & pemisahan TTFT/generation | PASS |
| Summary latency separation (Test C) | PASS |
| Tool latency separation | PASS |
| Streaming regression (Test E) | PASS |
| Full regression multi-conversation / global relationship / character memory / manual compact / auto rolling summary / Live2D / TTS / reconnect / persona Mili (Test F) | PASS (74/74 unittest OK; termasuk perbaikan bug eviksi summary) |
| Frontend production build + deploy | PASS (typecheck 0 error baru; baseline 588→587) |
| Backend compile + ruff + `git diff --check` | PASS |

Perintah verifikasi ulang:
```bash
cd /root/waifu/Open-LLM-VTuber && .venv/bin/python -m unittest discover -s tests
cd /root/waifu/Open-LLM-VTuber && bash scripts/build_frontend.sh
```

---

## Status

**THINKING LATENCY PATCH SELESAI**

Belum dapat menyimpulkan penyebab utama delay sebelum ada data live. Setelah kamu kirim log `[LLM LATENCY …]` dari 5–10 chat, bottleneck bisa dipastikan dengan angka — bukan tebakan.
