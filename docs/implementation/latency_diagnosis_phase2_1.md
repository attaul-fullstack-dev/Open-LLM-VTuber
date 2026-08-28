# Latency Diagnosis Phase 2.1

Classifier, TTS metric semantics, dan AssertionError fix.

Patch ini **tidak** mengubah: model, provider, persona, sampling, context window,
relationship, character memory, rolling summary, MCP, TTS provider/config,
Live2D, UI, timeout, atau retry. Tidak ada optimasi performa yang disengaja —
hanya memperbaiki 3 masalah yang ditemukan setelah pengujian live Phase 2.

---

## 1. Bottleneck Classifier Bug

### Root cause

`classify_bottleneck` (Phase 2) menggunakan **`largest_gap_ms`** sebagai dasar
label `unattributed`:

```python
if largest_gap >= 2000:
    ...
    if not inside_provider:
        return "unattributed"
```

Aturan itu salah: `largest_gap_ms` bisa menunjuk ke lorong waktu yang **sudah
diketahui dan terukur** — misalnya span `tts_wait_start → tts_wait_end` yang
persis = `tts_wait_ms`. Trace live membuktikannya:

```
largest_gap_ms=2668.71  largest_gap_from=tts_wait  largest_gap_to=tts_wait
unattributed_ms=0.2
bottleneck_hint=unattributed   # SALAH — gap itu adalah tts_wait yang terukur
```

Karena `unattributed_ms` hanya 0.2 ms dan gap terbesar berasal dari phase
`tts_wait` yang sudah diketahui, hint seharusnya `tts`.

### Sebelum

- `unattributed` diputuskan dari `largest_gap_ms >= 2000` di luar provider marks.
- `tts` di classifier memakai `tts_total_ms` lama (jumlah durasi async yang
  overlap — angka misleading, lihat section 2).
- Waktu agent-side di dalam `agent_stream` tidak pernah ikut dihitung
  (provider time dihitung ganda vs `agent_stream_ms`).

### Sesudah

- **`unattributed` hanya dipicu oleh `unattributed_ms`** (sisa waktu setelah
  seluruh exclusive segments), dengan threshold `_UNATTRIBUTED_THRESHOLD_MS = 500`.
  Gap besar di dalam phase yang diketahui (tts_wait, playback, TTFT, dsb.)
  **bukan** unattributed — itu latency terukur.
- `_exclusive_segments` kini memetakan jendela `agent_end → tts_wait_start`
  sebagai `response_processing_ms` (phase yang diketahui), jadi waktu itu tidak
  lagi dobel dihitung sebagai unattributed.
- Kontribusi `tts` = `tts_total_ms` (wall-clock blocking) + `playback_wait_ms`.
- Kontribusi `response_processing` baru: `agent_stream_ms − provider_wallclock_ms`
  (waktu agent-side di luar panggilan provider), sehingga provider time tidak
  dihitung ganda.

### Contoh trace

Trace live 13:31 (request `fc435125-...`) yang tadinya `bottleneck_hint=tts`
kini diklasifikasikan **`provider_ttft`** — karena TTFT 4034 ms memang
bottleneck sebenarnya:

```
provider_ttft=4034.11  provider_generation=919.73  tts_wait=2047.04
unattributed_ms=0.01   bottleneck_hint=provider_ttft
```

Trace dengan bentuk `tts_wait=2500, unattributed=0.2, largest_gap=tts_wait`
→ kini `bottleneck_hint=tts` (bukan unattributed).

---

## 2. TTS Metric Semantics

Arti tepat setiap metrik (per kode aktual `tts_manager.py` /
`single_conversation.py`):

| Metrik | Semantik | Boleh overlap? | Wall-clock? |
|---|---|---|---|
| `tts_enqueue_ms` | Biaya enqueue/scheduling per chunk (`speak()` → queue), dijumlahkan | Ya (cumulative work) | Tidak |
| `tts_synthesis_ms` | **Cumulative work** seluruh task sintesis async per chunk (`_process_tts`), dijumlahkan | Ya — task task berjalan paralel, jumlah bisa melebihi wall-clock | Tidak |
| `tts_wait_ms` | Span wall-clock `asyncio.gather(*task_list)` — waktu request benar-benar tertahan menunggu sintesis | Tidak (serial dengan main flow) | **Ya** |
| `tts_blocking_ms` | **Baru** — alias wall-clock `tts_wait_ms` (diverifikasi dari kode: gather = blocking utama) | Tidak | **Ya** |
| `playback_wait_ms` | Tunggu `frontend-playback-complete` (audio selesai diputar browser), setelah TTS | Tidak (serial setelah TTS) | **Ya** |
| `tts_total_ms` | **Definisi diubah**: kini = wall-clock blocking (`tts_wait_ms`), **bukan** jumlah enqueue+synthesis+wait | Tidak | **Ya** |
| `audio_blocking_ms` | **Baru** = `tts_blocking_ms + playback_wait_ms` (dua phase serial, tidak overlap — diverifikasi dari kode) | Tidak | **Ya** |

Aturan yang dipatuhi:

- Metrik berlabel "total"/wall-clock (`tts_total_ms`, `audio_blocking_ms`)
  **tidak pernah melebihi `total_response_ms`** (ada invariant test).
- Cumulative work (`tts_synthesis_ms`) **boleh** melebihi wall-clock — itu
  work time task async yang overlap, bukan latency request. Contoh live yang
  tadinya misleading: `tts_synthesis_ms=6046.8, tts_wait_ms=2537.88,
  tts_total_ms=8585.82 > total_response_ms=4122.16` — sekarang
  `tts_total_ms=2537.88` (wall-clock) dan `tts_synthesis_ms=6046.8` tetap
  dilaporkan sebagai cumulative work.
- `tts_total_ms` dipertahankan untuk kompatibilitas, tetapi definisinya
  diubah ke wall-clock blocking time (bukan dihapus, tidak ada dashboard/test
  lama yang rusak).

---

## 3. AssertionError Investigation

Request: `fc435125-8ac0-4b46-b76c-447f64e14186` (2026-08-27 13:31:43)

- `request_outcome=internal_error`, `error_category=AssertionError`, padahal
  provider sukses penuh (`provider_started=True ... provider_stream_completed=True`)
  dan TTS berjalan (`tts_synthesis_ms=3017.09`, `tts_wait_ms=2047.04`).
- Error muncul **di akhir request sukses** dengan pesan kosong:
  `Error in conversation chain:  ` — tanda tangan bare `assert` tanpa message.
- Terjadi 4× (12:11, 12:19, 12:22, 13:31), selalu berdekatan dengan send
  `backend-synth-complete`/`latency-event` saat sender task TTS masih aktif.

### Root cause (terbukti, bukan tebakan)

- **File**: `.venv/lib/python3.10/site-packages/websockets/legacy/protocol.py`
- **Function**: `_drain_helper` (line 308)
- **Assertion**: `assert waiter is None or waiter.cancelled()` — bare assert,
  pesan kosong (cocok persis dengan log).
- **Trigger**: transport write buffer sedang *paused* (`_paused=True`, client
  lambat membaca / payload audio besar) dan **dua coroutine memanggil drain
  bersamaan**. Drain kedua melihat `_drain_waiter` milik drain pertama yang
  belum selesai → assert gagal.

Rantai lengkap di aplikasi kita: `websocket.send_text` (starlette, tanpa queue)
→ uvicorn `asgi_send` → websockets legacy `send()` → `write_frame()` →
`transport.write()` (+ kemungkinan `pause_writing`) → `await drain()` →
`_drain_helper()` assert.

**Tiga sender konkuren** pada satu koneksi (semua memakai `websocket.send_text`
yang sama):

1. Task background conversation (`process_single_conversation` via
   `handle_conversation_trigger` → `asyncio.create_task`) — status/teks/
   `backend-synth-complete`/latency events.
2. Task sender payload TTS (`TTSTaskManager._process_payload_queue`) — chunk
   audio.
3. Receive loop (`_receive_loop`) — control/interrupt/error.

### Bukti deterministik

Test `test_library_drain_helper_asserts_on_concurrent_drains` mereproduksi
library race secara langsung: dua `_drain_helper()` konkuren saat `_paused=True`
→ `AssertionError()` pesan kosong (PASS).

### Fix

1. **`create_locked_send_text`** di `websocket_handler.py`: lock
   `asyncio.Lock` per koneksi, di-assign sebagai `websocket.send_text` di
   `handle_new_connection`. Semua jalur send (task conversation, sender TTS,
   receive loop, dan callable yang diteruskan ke ServiceContext) terserialisasi
   lewat lock yang sama → race drain hilang. Tidak mengubah urutan pesan di
   luar lock, tidak menyentuh streaming/TTS.
2. **`complete()`** di `request_latency.py`: kegagalan mengirim event
   `response-complete` (mis. client disconnect tepat di akhir) kini di-catch,
   dicatat sebagai warning post-processing, dan **tidak** mengubah
   `request_outcome`. Request sukses tidak lagi berubah jadi `internal_error`
   karena instrumentation gagal (spec item 10). Setelah assert pertama, dulu
   `_drain_waiter` tertinggal dalam keadaan rusak sehingga send berikutnya
   (emit `response-complete`) ikut assert — itu sebabnya ada dua log kegagalan
   per request.

### Regression test

- `test_locked_send_text_serializes_concurrent_senders`: simulasi websocket
  yang melempar `AssertionError` saat dua send overlap → unlocked FAIL,
  locked PASS, semua pesan terkirim.
- `test_complete_emit_failure_does_not_raise`: emit gagal → `complete()`
  tetap return, `request_outcome=success`.

### Temuan tambahan (bukan fix, butuh trace live Phase 2.2)

Matematika trace 13:31 memperlihatkan anomali: mark `provider_request_start`
berada ~3.5 detik **sebelum** entry transport attempt
(`[LLM ATTEMPT] started_rel_ms=3520.88` vs `ollama_request_to_headers_ms=4026.71`).
Artinya ada jendela ±3.5 s di dalam SDK OpenAI antara `provider_started()` dan
saat request benar-benar mencapai transport (headers transport hanya 531 ms).
Belum dijelaskan dari inspeksi statis; tidak ada fix spekulatif yang dipasang.
Trace `[LLM TRACE]`/`[LLM ATTEMPT]` Phase 2.1 sudah cukup untuk mengukur
jendela ini secara live.

---

## 4. Tests

| Test | Hasil |
|---|---|
| TEST A — tts dominan, unattributed 0.2 ms → `tts` | PASS |
| TEST B — unattributed nyata 2500 ms → `unattributed` | PASS |
| TEST C — provider TTFT 8 s vs tts 1 s → `provider_ttft` | PASS |
| TEST D — playback dominan → `tts` (taxonomy existing, didokumentasikan) | PASS |
| TTS invariant — `tts_total_ms` ≤ `total_response_ms` | PASS |
| Cumulative vs wall-clock — 2×3 s synthesis overlap → work 6 s, block 3.5 s | PASS |
| response_processing window = known phase (bukan unattributed) | PASS |
| Gap setelah provider → `response_processing`, bukan `unattributed` | PASS |
| Library race `_drain_helper` → `AssertionError` (bukti root cause) | PASS |
| Send lock — concurrent sends aman, tanpa lock gagal | PASS |
| `complete()` emit gagal → outcome tetap `success` | PASS |
| Full suite backend (`unittest discover`) | **91/91 PASS** |
| ruff + `git diff --check` | PASS |
| Frontend | tidak disentuh (backend-only patch) |

Tidak ada perubahan behavior: streaming, TTS, summary, MCP, Live2D, reconnect,
interrupt, multi-conversation tidak diubah.

---

## 5. Live Verification

Jalankan 3–5 chat normal, lalu kirim:

```bash
cd /root/waifu/Open-LLM-VTuber && grep -E "\[LLM TRACE|\[LLM ATTEMPT|\[LLM LATENCY STATS" logs/debug_2026-08-27.log | tail -40
```

Yang harus diperiksa setelah patch ini:

- `bottleneck_hint=tts` untuk request dengan `largest_gap_from=tts_wait` dan
  `unattributed_ms` kecil (sebelumnya salah `unattributed`).
- `tts_total_ms` = wall-clock (`tts_wait_ms`), tidak pernah > `total_response_ms`;
  `tts_blocking_ms` dan `audio_blocking_ms` terisi.
- Tidak ada lagi `request_outcome=internal_error` + `error_category=AssertionError`
  di akhir request sukses (send lock + emit catch).
- `request_outcome=interrupted` untuk request yang di-interrupt.

## 6. Status

**LATENCY DIAGNOSIS PHASE 2.1 SELESAI**
