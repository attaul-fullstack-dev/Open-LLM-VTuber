# Conversation Architecture v2 — Hasil

Tanggal: 26 Agustus 2026

## Character State

- **Storage**: `character_state/<conf_uid>.json` (satu file JSON per karakter). Ditulis atomically (temp file + `os.replace`) dengan per-file `RLock`, sama seperti mekanisme history metadata existing. Tidak ada database, vector index, embeddings, Letta, Mem0, atau knowledge graph.
- **Relationship scope**: character-level (`conf_uid`). Relationship dimiliki Mili, bukan satu history. Semua chat (Chat A/B/C) berbagi state yang sama.
- **Persistent**: setelah refresh, reconnect, backend restart, new conversation, dan conversation switching. File tetap aman saat dua session menulis karena atomic write + lock per file.
- Field: `relationship_status`, `relationship_updated_at`, `relationship_reason`, `relationship_migrated`, `memories`.

## Character Memory

- **Storage**: digabung dalam `character_state/<conf_uid>.json` (field `memories`). Setiap entri: `{text, added_at, explicit}`.
- **Cara memory ditambahkan**:
  - Manual via chat: aturan regex lokal konservatif, tanpa LLM/API call tambahan. Hanya permintaan eksplisit yang dipenuhi, contoh: `Ingat ya, makanan favoritku ramen.`, `Jangan lupa kalau aku suka kopi.`, `Tolong ingat ...`, `Catat ...`.
  - Backend API/method: `BasicMemoryAgent.add_character_memory(text)` (untuk UI/integrasi lain).
  - Auto-detection sengaja tidak ada untuk pernyataan sesaat (`aku lagi makan mie`, `aku ngantuk`, `hari ini hujan`, `wkwk`, `makasih` → tidak pernah jadi memory permanen). Pertanyaan yang diawali "Ingat nggak ...?" juga tidak dianggap permintaan menyimpan.
- **Cara memory dihapus**:
  - Manual via chat: `Lupakan kalau makanan favoritku ramen.`, `Hapus dari ingatan ...`.
  - Backend API: `remove_character_memory(text)`, `reset_character_memory()`, dan `reset_character_state()` (relationship + memory).
  - UI: tombol delete per fakta dan tombol reset memory di Settings → Agent.
- **Budget**: blok memory dibatasi maksimum **600 estimated tokens** (`CHARACTER_MEMORY_MAX_TOKENS`, dalam rentang konservatif 500–1000 dari spec). Prioritaskan memory eksplisit (manual) lalu yang terbaru. Memory tidak pernah dilempar utuh ke model.
- Dedup: teks yang sama tidak ditambahkan dua kali.

## Conversation Isolation

- Transcript, rolling summary, dan recent context tetap per history: `chat_history/<conf_uid>/<history_uid>.json` (tidak berubah dari Tahap 4).
- Agent memuat `_memory` dan `_summary_state` dari history aktif. Chat A tidak pernah menerima transcript/summary Chat B.
- Hanya relationship dan character memory yang bersifat lintas-chat. Test switching A→B→A→B membuktikan summary dan recent messages tidak bocor; character memory dan relationship tetap sama.

## Manual Compact

- **Backend flow**: WebSocket `compact-conversation` → `WebSocketHandler._handle_compact_conversation` → `BasicMemoryAgent.compact_conversation()` (async):
  1. Ambil semua pesan yang belum dirangkum (`_memory[summarized_through:]`).
  2. Ringkas dengan `IncrementalSummarizer` yang sama dengan rolling summary otomatis (system prompt netral, target/max token sama).
  3. Persist lewat `update_summary_metadata` dengan `expected_summarized_through` (guard stale write) → `summarized_through` maju ke `len(_memory)`.
  4. Balas `compact-result {success, error}`.
- **UI flow**: tombol archive (Compact conversation) per item di history drawer → konfirmasi → toaster "Conversation compacted" (tanpa istilah teknis). Tersedia juga untuk chat yang sedang aktif.
- **Transcript tetap utuh**: **Ya** — compact tidak menghapus/mengganti transcript, tidak menghapus message dari UI, tidak menghapus conversation. UI tetap menampilkan message 1–500.
- **Compact failure**: jika summarizer gagal atau persist gagal → transcript aman, summary lama aman, `summarized_through` tidak berubah, chat tetap bisa dipakai, UI menampilkan error terkontrol.

## Auto + Manual Compact

- Manual compact memakai pipeline rolling summary yang sama; `summarized_through_message_index` adalah satu-satunya boundary (tidak ada sistem compact kedua).
- Setelah manual compact, auto compact menghitung `start = summarized_through` → hanya pesan setelah boundary yang dirangkum. Test membuktikan prompt summarizer kedua berisi "pesanbaru ..." dan tidak mengandung "pesan lama nomor 0" (tidak ada re-summarize).

## Conversation UI

- **New Chat**: tombol `+` existing di sidebar → `create-new-history` → `history_uid` baru, transcript & summary kosong, tetap memuat persona Mili + relationship global + memory global. `history_uid` terakhir disimpan untuk resume.
- **Rename**: tombol pencil di history drawer → prompt manual → `rename-history` → tersimpan di metadata conversation (`title`), persistent setelah restart/reconnect. Tidak mengubah transcript, summary, relationship, atau memory. Fallback tampilan: `title` metadata → preview pesan pertama → "New Conversation". Tidak ada auto-title via LLM.
- **Compact**: tombol archive per item di history drawer (dengan konfirmasi).
- **Delete**: menghapus hanya transcript + summary + metadata lokal conversation tersebut. Relationship dan character memory global tetap ada (perubahan dari Tahap 5).
- Sidebar menampilkan daftar chat terpisah dengan title + preview, sesuai prinsip "daftar chat seperti ChatGPT".

## Context Composition

Urutan request setelah v2:

1. **system/persona** (persona Mili tidak diubah)
2. **relationship** (internal guidance, disisipkan ke system prompt setelah persona)
3. **character memory** (blok "Known long-term context", setelah relationship, maks 600 token)
4. **conversation summary** (rolling summary chat aktif, sebagai message berlabel)
5. **recent conversation** (pesan utuh chat aktif)
6. **current user message** (terakhir, protected)

Relationship + memory lintas-chat; summary + recent history hanya untuk chat aktif. Seluruh system prompt dihitung oleh context budget Tahap 3 (reserved 384 untuk output).

## Migration

- Strategi: **migrasi sekali jalan, backward-compatible, tanpa tebakan**.
- Saat pertama kali character state dimuat (`migrate_relationship_if_needed`):
  - Jika character-level relationship sudah ada atau `relationship_migrated` sudah true → tidak melakukan apa-apa (tidak pernah rescan).
  - Jika belum: scan metadata semua conversation di `chat_history/<conf_uid>/*.json`.
  - Pilih state **terkuat + terbaru** yang valid (ranking `stranger < familiar < close < dating`; tie-break `relationship_updated_at`).
  - `stranger` default tidak pernah di-migrate menjadi tebakan; `dating` yang eksplisit di metadata lama dipertahankan apa adanya (beserta `relationship_reason`).
  - Hasil ditulis ke character state; flag `relationship_migrated` di-set; setelah itu character state menjadi source of truth.
- Setelah migrasi, semua update relationship baru menulis ke character state (bukan metadata conversation).

## Tests

Total automated test: **64** — semuanya PASS (fake LLM + mock summarizer + synthetic conversations; tanpa live Mistral call).

Test baru `test_conversation_architecture_v2.py` (19):

| Test | Hasil |
|---|---|
| New chat isolated transcript | PASS |
| New chat isolated summary | PASS |
| No topic leakage A/B/A/B | PASS |
| Global relationship cross-chat | PASS |
| Relationship reset global | PASS |
| Delete conversation preserves global relationship | PASS |
| Character memory cross-chat | PASS |
| Remember/forget via chat + konservatif | PASS |
| Memory budget bounded (≤600 token) | PASS |
| Character memory reset (relationship aman) | PASS |
| Reset character state (transcript aman) | PASS |
| Manual compact (summary update, index maju, transcript utuh) | PASS |
| Auto compact setelah manual compact (incremental, tidak re-summarize) | PASS |
| Manual compact failure (terkontrol, chat tetap bisa dipakai) | PASS |
| Rename conversation persistence | PASS |
| WebSocket compact/rename/memory handlers | PASS |
| Migrasi relationship legacy → character | PASS |
| Stranger tidak pernah di-migrasi jadi tebakan | PASS |
| Context composition order + budget | PASS |
| Restart persistence character state | PASS |

Test existing diperbarui ke semantik v2:
- `test_relationship_context.py` (17): isolation → global; persistence/reset/delete memeriksa character state.
- `test_final_integration.py` (7): switching global, write-failure (patch `save_character_state`), restart, invalid uid (relationship global tetap dating, memory/summary kosong).
- `test_context_window.py` (10) dan `test_conversation_summary.py` (8): tidak diubah, tetap PASS.

## Regression Tahap 1–6

PASS.

- Session isolation, reconnect/resume, history switching: PASS (tidak diubah).
- Live2D, TTS, ASR: PASS (tidak disentuh).
- Persona Mili: PASS (tidak diubah).
- Generation config `mistral-small-latest / temperature 0.8 / top_p 0.9 / max_tokens 384`: PASS (tidak diubah).
- Context budgeting + reserved 384: PASS (context stress test tetap hijau).
- Automatic rolling summary incremental + persistence + isolation: PASS.
- Relationship detection existing (regex konservatif): PASS, kini persist ke character level.
- Reset relationship UI: PASS, kini global.
- Frontend production build: PASS (`npm run build:web`, 0 error TS baru: 588 sebelum = 588 sesudah).
- Backend compile + ruff + `git diff --check`: PASS.
- Logging security: tidak ada isi transcript/summary/memory/relationship-reason/API key yang dilog; hanya statistik (`character_memory_count`, `character_memory_updated`, `relationship_status`, `compact_trigger`, `summary_updated`).

## File yang Diubah

Backend (main repo):
- `src/open_llm_vtuber/character_state.py` (**baru**) — storage character-level: relationship + long-term memory, atomic write, migration, memory context builder.
- `src/open_llm_vtuber/agent/agents/basic_memory_agent.py` — load character state, relationship set/reset → character-level, memory add/remove/list/reset, injection relationship+memory ke system prompt, remember/forget konservatif, `compact_conversation()`, `observe_character_events()`.
- `src/open_llm_vtuber/websocket_handler.py` — handler baru: `compact-conversation`, `rename-history`, `fetch-character-memory`, `delete-character-memory`, `reset-character-memory`, `reset-character-state`; `reset-relationship` kini global.
- `src/open_llm_vtuber/chat_history_manager.py` — `get_history_list` menyertakan `title` dari metadata.
- `src/open_llm_vtuber/conversations/single_conversation.py` — memanggil `observe_character_events` (fallback ke `observe_relationship_event`).
- `tests/test_relationship_context.py`, `tests/test_final_integration.py` — diperbarui ke semantik relationship global.
- `tests/test_conversation_architecture_v2.py` (**baru**) — 19 test v2.
- `docs/implementation/tahap7.md` (**baru**) — laporan ini.
- `frontend` — submodule static: bundle baru hasil sync `scripts/build_frontend.sh`.

Frontend source (`Open-LLM-VTuber-Web`):
- `src/renderer/src/components/sidebar/history-drawer.tsx` — daftar chat dengan title, tombol rename + compact + delete.
- `src/renderer/src/components/sidebar/sidebar-styles.tsx` — style title/action button drawer.
- `src/renderer/src/hooks/sidebar/use-history-drawer.ts` — `renameHistory`, `compactConversation`.
- `src/renderer/src/services/websocket-handler.tsx` — handle `compact-result`, `history-renamed`, `character-memory*`.
- `src/renderer/src/services/websocket-service.tsx` — tipe `title`, `memories`, `error` pada `MessageEvent`.
- `src/renderer/src/context/websocket-context.tsx` — `HistoryInfo.title`.
- `src/renderer/src/components/sidebar/setting/agent.tsx` — seksi "Mili's Long-Term Memory" (list + delete + reset memory), tombol reset relationship (global) + reset character state.
- `src/renderer/src/locales/en/translation.json`, `locales/zh/translation.json` — string baru (history, memory, compact, rename, reset global).

## Temuan Tambahan

- `git status` di main repo menunjukkan `cloudflared` sebagai untracked — bukan bagian dari perubahan ini (artefak lama di luar scope).
- Warning build frontend (chunk size + `eval` onnxruntime-web) tetap sama seperti baseline; bukan regression.
- Boot server penuh membutuhkan >12 detik karena inisialisasi MCP (`time`, `ddg-search`); verifikasi route dilakukan via import router/handler, bukan boot penuh (sesuai praktik Tahap 6).
- Manual remember v1 hanya mengenali permintaan eksplisit berbahasa Indonesia (`ingat`, `jangan lupa`, `catat`, `lupakan`, `hapus dari ingatan`). Parafrase lain belum dikenali — sengaja konservatif.
- Auto-summary tetap hanya berjalan saat trimming benar-benar meng-evict pesan; setelah manual compact, pesan baru yang masih muat di context tidak langsung diringkas (perilaku yang benar).

## FINAL STATUS

**CONVERSATION ARCHITECTURE V2 SELESAI**

Menunggu manual test user. Tidak ada fitur baru yang dimulai.
