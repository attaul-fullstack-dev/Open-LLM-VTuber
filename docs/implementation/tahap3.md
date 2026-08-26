# Tahap 3 — Hasil Implementasi

Tanggal: 26 Agustus 2026

## Context Limit

- Model aktif: `mistral-small-latest`
- Context limit: `256000` token
- Sumber/penentuan limit: mapping terpusat mengacu pada dokumentasi resmi Mistral Small 4 (`mistral-small-2603`) yang menyatakan context window 256k. Alias `mistral-small-latest` saat ini diarahkan ke batas tersebut. Dokumentasi: <https://docs.mistral.ai/models/mistral-small-4-0-26-03> dan <https://docs.mistral.ai/resources/known-limitations>.
- Model lama `mistral-small-2506` dipetakan ke `128000` token.
- Model/provider yang tidak dikenal memakai fallback konservatif `32768` token.
- Karena alias `latest` dapat berubah, `context_window_override` tersedia pada config sebagai sumber kebenaran manual tanpa mengubah source.

Context limit tidak menjadi angka global tunggal. Resolusi limit berada dalam satu mapping terpusat dan dapat dioverride per konfigurasi `basic_memory_agent`.

## Budget

- Reserved output: `384` token, dibaca dari `max_tokens` provider aktif.
- Safety margin: `1024` token.
- Maximum input budget aktif: `256000 - 384 - 1024 = 254592` token estimasi.
- Jika provider tidak mengekspos `max_tokens`, fallback reserved output adalah `1024` token.
- Jika system prompt, tool schema, dan current turn saja melampaui maximum input budget, request ditolak sebelum API provider dipanggil. System/persona dan pesan user tidak dipotong diam-diam.

## Token Counting

- Metode: estimasi konservatif terpusat `ceil(UTF-8 bytes / 3)` ditambah overhead framing per message.
- Status: **estimated**, bukan exact dan bukan angka billing.
- System prompt aktual yang sudah mengandung persona, Live2D prompt, interruption instruction, dan prompt tekstual MCP ikut dihitung.
- Native tool schema ikut diestimasi sebagai bagian budget.
- Dependency tambahan: **tidak ada**.

Environment tidak memiliki tokenizer resmi Mistral maupun tokenizer umum yang sudah menjadi dependency (`mistral-common`, `tiktoken`, `transformers`). Dependency besar tidak ditambahkan hanya untuk Tahap 3. Estimasi tiga byte per token dipilih agar lebih berhati-hati untuk campuran bahasa Indonesia, Unicode, dan JSON dibanding asumsi empat karakter per token.

## Trimming

- Strategi: pertahankan system/persona dan seluruh current turn, lalu pilih suffix history terbaru yang masih muat.
- Unit trimming: turn yang dimulai dari message `user` dan mencakup respons `assistant`/tool terkait sampai user berikutnya.
- History lama dibuang dari request mulai dari yang paling lama.
- Turn lama yang terlalu besar tidak dipotong sebagian; hanya suffix turn lengkap yang dipertahankan.
- Pada loop MCP/tool, current user message beserta tool-call/tool-result aktif diproteksi agar tidak hilang di request lanjutan.
- Transcript disk/RAM berubah: **Tidak**. `_memory` tetap lengkap; context manager hanya menghasilkan daftar message baru untuk satu request LLM.
- Context management dapat dimatikan dengan `context_management_enabled: false` tanpa mengubah API agent.

## Debug Stats

Log developer sekarang menyediakan statistik berikut tanpa menampilkan isi persona/chat:

- `model`
- `context_limit`
- `reserved_output`
- `safety_margin`
- `maximum_input_budget`
- `system_tokens`
- `tool_tokens`
- `history_tokens_before`
- `history_tokens_after`
- `messages_before`
- `messages_after`
- `trimmed`
- `estimated_input_tokens`
- `fallback_limit`

Logging system prompt dan request message pada jalur LLM aktif juga diubah menjadi panjang/jumlah saja. Credential tetap tidak dicetak.

## Tests

| Test | Hasil |
|---|---|
| Short history | PASS — semua message tetap dikirim dan `trimmed=false` |
| Long history | PASS — turn lama dipangkas, suffix recent tetap utuh, tidak melebihi budget |
| Transcript preservation | PASS — input asli dan `_memory` penuh tidak berubah karena trimming |
| Recent recall | PASS — turn `kode saya mangga` tetap masuk bersama pertanyaan terbaru |
| Persona preservation | PASS — system/persona dihitung terpisah dan tidak menjadi kandidat trimming |
| Reserved output | PASS — `384` token serta safety margin benar-benar mengurangi input budget |
| Oversized user message | PASS — request ditolak dengan pesan yang dapat ditangani dan provider tidak dipanggil |
| Unknown model fallback | PASS — fallback `32768` digunakan tanpa crash |
| Tool/MCP budget | PASS — native tool schema dihitung; current tool chain diproteksi |
| Regression | PASS dengan catatan frontend typecheck lama di Temuan Tambahan |

Rincian verifikasi:

- 10 unit/integration test context window: PASS.
- Agent aktif hasil factory: `basic_memory_agent`, model `mistral-small-latest`, temperature `0.8`, top-p `0.9`, max tokens `384`: PASS.
- Dua session tetap memiliki instance agent, LLM, dan `_memory` berbeda: PASS.
- Active config dan kedua default config template tervalidasi: PASS.
- Backend compile: PASS.
- Ruff lint: PASS.
- `git diff --check`: PASS.
- Lightweight backend app construction dan route `/client-ws`: PASS; server tidak ditinggalkan hidup.
- Frontend production build: PASS.
- Conversation resume, invalid-history fallback, manual new conversation, dan history switching melalui source/runtime test Tahap 1: PASS.
- Live2D system-prompt composition dan marker parser: PASS.
- Tidak ada live API call Mistral; synthetic history dan fake provider digunakan agar tidak memakai quota.

## Regression Tahap 1 dan 2

| Area | Hasil |
|---|---|
| Session isolation | PASS |
| Conversation resume | PASS |
| Manual new conversation | PASS |
| Invalid/deleted last conversation fallback | PASS |
| History switching | PASS |
| Live2D prompt/parser | PASS |
| Temperature `0.8` | PASS |
| Top P `0.9` | PASS |
| Max output token `384` | PASS |
| Persona Mili tidak berubah | PASS |
| Backend compile/config/startup smoke | PASS |
| Frontend production build | PASS |

## File yang Diubah

- `src/open_llm_vtuber/agent/context_window.py` — resolver context limit, estimasi token, budget, coherent-turn trimming, statistik, dan oversized-input exception.
- `src/open_llm_vtuber/agent/agents/basic_memory_agent.py` — menerapkan budget tepat sebelum tiap request simple chat, Claude tool loop, dan OpenAI/MCP loop tanpa mengubah `_memory`.
- `src/open_llm_vtuber/config_manager/agent.py` — schema optional dan backward-compatible untuk enable/disable, context override, serta safety margin.
- `src/open_llm_vtuber/agent/agent_factory.py` — meneruskan config context management ke tiap instance `BasicMemoryAgent`.
- `config_templates/conf.default.yaml` — dokumentasi default context management Inggris.
- `config_templates/conf.ZH.default.yaml` — dokumentasi default context management Mandarin.
- `src/open_llm_vtuber/agent/stateless_llm/openai_compatible_llm.py` — request log hanya statistik, tidak lagi isi message.
- `src/open_llm_vtuber/agent/stateless_llm/claude_llm.py` — mengekspos reserved output existing `1024` untuk budgeting dan mengamankan log request.
- `src/open_llm_vtuber/agent/stateless_llm/llama_cpp_llm.py` — log hanya jumlah message.
- `src/open_llm_vtuber/agent/stateless_llm/stateless_llm_with_template.py` — log hanya jumlah message.
- `src/open_llm_vtuber/service_context.py` — persona/system prompt di debug output diganti statistik panjang, credential tetap direduksi.
- `tests/test_context_window.py` — unit dan integration tests untuk trimming, preservation, recent recall, tools, fallback, dan oversized input.
- `docs/implementation/tahap3.md` — laporan Tahap 3 di repository.
- `/root/waifu/tahap3.md` — salinan laporan mudah ditemukan bersama `audit.md`, `tahap1.md`, dan `tahap2.md`.

Tidak ada file persona maupun frontend yang diubah pada Tahap 3.

## Temuan Tambahan

- `mistral-small-latest` adalah alias dinamis. Mapping harus ditinjau jika Mistral mengubah target alias; `context_window_override` sudah tersedia agar operasional tidak bergantung pada perubahan source.
- Token count adalah estimasi konservatif karena tokenizer resmi tidak tersedia sebagai dependency existing. Akurasi dapat ditingkatkan pada pekerjaan terpisah bila tokenizer ringan/resmi kemudian menjadi dependency project.
- Full frontend typecheck masih gagal pada banyak deklarasi Live2D SDK dan beberapa error UI yang sudah ada sebelum Tahap 3. Frontend production build tetap PASS dan Tahap 3 tidak menyentuh frontend.
- Source frontend masih berada di `/tmp/olv-mobile` dan belum menjadi lokasi permanen.
- Ada warning lama tentang komponen MCP saat `use_mcpp=false`; warning tersebut tidak menghalangi request dan tidak diubah karena di luar scope.
- Beberapa jalur aplikasi lama di luar context manager masih mempunyai log isi percakapan/TTS. Log baru Tahap 3 dan jalur provider/system aktif sudah aman; audit logging menyeluruh tidak dilakukan karena di luar scope.
- Verifikasi kualitas provider secara live tetap perlu dilakukan manual bila diinginkan; tidak diperlukan untuk membuktikan algoritma trimming.

## Status

**TAHAP 3 SELESAI**

Rolling summary, relationship state, dan Tahap 4 belum dimulai.
