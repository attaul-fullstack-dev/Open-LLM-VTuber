# Tahap 5 — Hasil Implementasi

Tanggal: 26 Agustus 2026

## Relationship States

State yang tersedia:

- `stranger` — lebih menjaga jarak dan sedikit lebih reserved.
- `familiar` — lebih santai dan nyaman dengan user yang sudah dikenali.
- `close` — lebih terbuka, perhatian, nyaman bercanda, dan tidak terlalu defensif.
- `dating` — hubungan romantis sudah disepakati secara mutual dalam conversation; Mili tetap tsundere tetapi tidak bertindak seolah kesepakatan itu tidak pernah terjadi.

Default conversation baru: `stranger`.

Relationship hanya mengubah tingkat familiaritas, openness, dan defensiveness. Persona Mili tetap menjadi sumber utama kepribadian, gaya bahasa, dan cara merespons. Tidak ada point, XP, mood meter, decay, atau kenaikan berdasarkan jumlah message.

## Storage

- Lokasi: metadata file conversation existing, `chat_history/<conf_uid>/<history_uid>.json`.
- Metadata fields:
  - `relationship_status`
  - `relationship_updated_at`
  - `relationship_reason`
- Nilai `relationship_reason` hanya trigger generik seperti `explicit_relationship_event`, bukan transcript atau isi percakapan.
- Persistent setelah restart: **Ya**.
- Refresh/reconnect dan agent recreation memuat state dari history aktif.
- History A dan History B menyimpan state masing-masing.
- New conversation dimulai dari `stranger`.
- Delete conversation menghapus relationship state bersama file conversation yang sama; tidak ada orphan database/state.
- Database atau dependency baru: tidak ada.

## Update Logic

Relationship diperiksa setelah satu turn visible selesai dan respons final Mili tersedia. Pemeriksaan memakai rule lokal sempit terhadap pasangan user–assistant tersebut; tidak ada classifier model dan tidak ada API call tambahan.

Perubahan otomatis yang tersedia:

- `stranger -> familiar`: hanya event returning-user/recognition yang eksplisit dan diakui oleh Mili.
- `stranger/familiar -> close`: hanya pernyataan trust/kedekatan yang kuat dan mendapat acknowledgement mutual.
- state apa pun selain `dating -> dating`: user mengusulkan hubungan/pacaran secara eksplisit dan respons Mili menunjukkan penerimaan yang jelas.

Explicit dating event diperlakukan paling kuat. Marker teknis Live2D di awal respons diabaikan saat klasifikasi agar tidak mengganggu deteksi.

False positive dicegah dengan cara berikut:

- `Aku suka kamu` tanpa penerimaan relationship tidak mengubah state menjadi `dating`.
- Pujian biasa seperti `kamu lucu` tidak menaikkan state.
- Sering chat, banyak message, curhat, atau sentiment positif tidak otomatis mengubah state.
- Respons penolakan seperti `nggak mau`, `cuma teman`, atau `aku tolak` membatalkan kandidat dating.
- Jika bukti tidak cukup, state lama dipertahankan.
- Tidak ada downgrade atau decay otomatis.

Manual backend update tersedia melalui `BasicMemoryAgent.set_relationship_status()`. Reset sederhana tersedia melalui:

- method backend `BasicMemoryAgent.reset_relationship()`;
- WebSocket message `reset-relationship` untuk history yang sedang aktif;
- response backend `relationship-reset` berisi status keberhasilan dan history UID.

UI reset tidak ditambahkan karena itu scope Tahap 6.

## Request Injection

Urutan konseptual request:

1. system rules dan persona Mili
2. relationship context internal
3. rolling summary, jika tersedia dan masuk budget
4. recent conversation utuh
5. current user message

Relationship context disisipkan sebagai bagian system prompt efektif setelah persona, sehingga selalu dihitung oleh context budget Tahap 3 dan tidak menjadi message user. Isinya singkat, faktual, dan menegaskan:

- state hanya memengaruhi familiarity/openness;
- persona inti Mili tidak berubah;
- internal state name dan mekanismenya tidak boleh disebut kepada user;
- jika user bertanya tentang hubungan, Mili harus menjawab secara natural, bukan menyebut label internal.

Relationship state terbaru lebih otoritatif untuk status hubungan daripada rolling summary lama. Summary dan relationship state tetap merupakan dua data terpisah.

## API Cost

Relationship detection membutuhkan additional LLM request: **Tidak**.

Mekanisme:

- satu pemeriksaan regex lokal yang murah setelah completed turn;
- update metadata hanya dilakukan ketika event valid ditemukan;
- tidak ada pemeriksaan API per message;
- tidak ada provider/model baru;
- tidak ada live Mistral call selama testing.

## Debug Stats

Log aman yang tersedia:

- `relationship_status`
- `relationship_updated`
- `relationship_update_trigger`

Trigger yang dilog hanya kategori generik seperti:

- `load_history`
- `skipped`
- `returning_user_event`
- `mutual_trust_event`
- `explicit_relationship_event`
- `manual_reset`

Logging baru tidak mencetak transcript, isi percakapan romantis, persona, API key, atau credential.

## Tests

| Test | Hasil |
|---|---|
| Default stranger | PASS — conversation baru memiliki metadata dan agent state `stranger` |
| Persistence | PASS — state `close` termuat kembali setelah agent recreation |
| Conversation isolation | PASS — History A `dating`, History B `familiar`, switching tetap benar |
| Explicit dating event | PASS — proposal user + penerimaan eksplisit Mili menghasilkan `dating` |
| One-sided romantic message | PASS — pengakuan suka satu arah tidak menghasilkan `dating` |
| Compliment false-positive | PASS — pujian biasa tidak menaikkan relationship |
| Long conversation | PASS — state `dating` tetap tersedia saat old history di-trim dan summary aktif |
| Restart | PASS — load metadata setelah simulasi restart tetap memulihkan state |
| Context injection | PASS — request nyata mengandung relationship context kecil setelah persona |
| Persona stability | PASS — base persona identik pada keempat state; hanya guidance familiarity berbeda |
| Internal label leakage | PASS — prompt melarang penyebutan internal state dan meminta jawaban natural |
| Manual reset | PASS — method dan WebSocket backend reset mengembalikan serta menyimpan `stranger` |
| Delete conversation | PASS — relationship metadata hilang bersama file history |
| Regression Tahap 1–4 | PASS |

Rincian regression:

- 14 relationship continuity tests: PASS.
- 8 rolling-summary tests Tahap 4: PASS.
- 10 context-window tests Tahap 3: PASS.
- Total automated test discovery: **32 PASS**.
- Session isolation: PASS.
- Reconnect/resume dan conversation switching tidak diubah: PASS.
- History persistence dan summary isolation: PASS.
- Live2D marker tetap bekerja; marker teknis juga aman untuk detector: PASS.
- Temperature `0.8`, top-p `0.9`, max tokens `384`, model `mistral-small-latest`: PASS.
- Persona Mili tidak diubah: PASS.
- Context budgeting, trimming, recent recall, rolling summary, dan oversized-input protection: PASS.
- Config aktif dan kedua config template tervalidasi: PASS.
- Backend compile: PASS.
- Ruff lint: PASS.
- Lightweight backend construction dan route `/client-ws`: PASS.
- Frontend production build: PASS.
- `git diff --check`: PASS.
- Server tidak ditinggalkan hidup.

## File yang Diubah

- `src/open_llm_vtuber/agent/relationship_context.py` — empat state, compact context formatter, dan explicit-event detector konservatif tanpa LLM.
- `src/open_llm_vtuber/agent/agents/basic_memory_agent.py` — load/persist state, context injection, update/reset API backend, event observation, dan safe debug stats.
- `src/open_llm_vtuber/chat_history_manager.py` — metadata default `stranger` pada conversation baru.
- `src/open_llm_vtuber/conversations/single_conversation.py` — menjalankan detector hanya setelah turn visible selesai dan respons final tersedia.
- `src/open_llm_vtuber/websocket_handler.py` — backend WebSocket reset dan pembersihan active agent state ketika history aktif dihapus.
- `tests/test_relationship_context.py` — synthetic tests untuk state, persistence, isolation, detection, summary/context integration, leakage guidance, reset, dan delete.
- `docs/implementation/tahap5.md` — laporan Tahap 5 di repository.
- `/root/waifu/tahap5.md` — salinan laporan mudah ditemukan bersama laporan sebelumnya.

Tidak ada perubahan pada persona Mili, model, temperature, top-p, max tokens, Live2D prompt, TTS, ASR, atau frontend.

## Temuan Tambahan

- Detector sengaja sempit dan saat ini berfokus pada frasa relationship bahasa Indonesia yang eksplisit. Parafrase yang sangat berbeda mungkin tidak otomatis mengubah state; ini lebih aman daripada false positive. Backend setter tetap tersedia untuk koreksi eksplisit.
- Relationship bersifat per conversation sesuai scope. Membuka new conversation akan kembali `stranger`; global character-level relationship belum dibuat.
- UI reset belum dibuat. Backend method dan WebSocket event sudah tersedia untuk integrasi UI pada Tahap 6.
- Frontend production build masih mengeluarkan warning existing tentang ukuran chunk dan `eval` dari `onnxruntime-web`; bukan regression Tahap 5.
- Source frontend masih berada di `/tmp/olv-mobile` dan belum permanen.
- Warning MCP existing ketika `use_mcpp=false` tetap tidak diubah karena di luar scope.

## Status

**TAHAP 5 SELESAI**

Tahap 6 belum dimulai.
