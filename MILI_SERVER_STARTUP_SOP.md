# SOP WAJIB STARTUP & PEMULIHAN SERVER MILI

> **ATURAN MUTLAK:** Dokumen ini WAJIB dibaca dan dijalankan setiap kali ada permintaan menghidupkan, me-restart, atau memulihkan server Mili ("hidupkan server Mili", "start server Mili", "recover server Mili").

---

## 1. Alur Kerja Wajib (10 Langkah Mutlak)

Ketika pengguna meminta "hidupkan server Mili":

1. **Audit kondisi Git backend dan frontend:**
   - Periksa working tree, branch aktif, dan status commit pada kedua repository (`Open-LLM-VTuber` dan `Open-LLM-VTuber-Web`).
2. **Periksa commit lokal dan origin/main:**
   - Bandingkan status HEAD lokal dengan `origin/main` untuk mendeteksi apakah lokal tertinggal atau mendahului remote.
3. **Deteksi perubahan lokal yang belum di-commit atau di-push:**
   - Periksa staged, unstaged, dan untracked files.
   - Jika ditemukan perubahan lokal yang belum diamankan di Git maupun backup arsip, **BERHENTI dan laporkan terlebih dahulu**. Jangan mengambil keputusan sepihak.
4. **Jangan menimpa atau membuang perubahan lokal:**
   - Dilarang keras melakukan `git reset --hard`, `git clean -fd`, `git checkout .`, `git restore .`, atau `git stash drop`.
5. **Sinkronisasi aman:**
   - Jika GitHub memiliki perubahan terbaru yang aman diambil (clean fast-forward), sinkronkan terlebih dahulu. Jika berisiko konflik, berhenti dan laporkan.
6. **Pastikan frontend build dan static bundle sesuai dengan source terbaru:**
   - Periksa apakah source frontend memerlukan build (`npm run build:web`).
   - Pastikan bundle statis di backend (`Open-LLM-VTuber/frontend/`) identik dan tersinkronisasi dengan hasil build terbaru dari `Open-LLM-VTuber-Web/dist/web/`.
7. **Jalankan backend dengan konfigurasi produksi yang sudah ada:**
   - Gunakan environment Python virtualenv (`.venv/bin/python run_server.py --verbose`).
   - Jalankan di dalam sesi tmux yang terisolasi (misal: `waifu:<window>`).
8. **Jangan mengubah conf.yaml, credential, character_state, atau chat_history:**
   - Dilarang memodifikasi file konfigurasi produksi `conf.yaml`.
   - Dilarang mereset atau memanipulasi database runtime `character_state/` dan `chat_history/`.
9. **Verifikasi HTTP, WebSocket, dan koneksi frontend:**
   - Verifikasi port 12393 listening (Uvicorn).
   - Verifikasi port 80 listening dan Nginx meneruskan reverse proxy dengan benar.
   - Verifikasi respons HTTP 200 pada root (`/`) dan metadata Live2D (`/live2d-models/info`).
   - Verifikasi WebSocket handshake (`101 Switching Protocols`) dan penerimaan 4 frame inisialisasi awal.
10. **Verifikasi bahwa versi kode yang berjalan sesuai dengan commit yang dituju:**
    - Cocokkan hash commit proses backend yang berjalan dengan HEAD target.
    - Pastikan penanganan normal disconnect (`WebSocketDisconnect`) tercatat sebagai INFO (0 error log).

---

## 2. Larangan Keras (Strict Boundaries)

- **DILARANG** mengklaim deploy atau startup berhasil sebelum verifikasi runtime (HTTP & WebSocket) selesai diuji secara riil.
- **DILARANG** menghapus atau menimpa berkas pengguna yang belum di-commit.
- **DILARANG** melakukan deploy ulang atau restart jika server sudah berjalan normal dan melayani traffic dengan benar, kecuali secara eksplisit diminta oleh pengguna.

---

## 3. Format Laporan Standar Pasca-Startup

Setiap startup wajib menghasilkan laporan berformat tabel rapi berikut:

```text
================================================================================
                    LAPORAN STARTUP & STATUS SERVER MILI
================================================================================
+----------------------------+-------------------------------------------------+
| Parameter Audit            | Status Aktual                                   |
+----------------------------+-------------------------------------------------+
| Backend Commit             | <hash> — <pesan commit>                         |
| Frontend Commit            | <hash> — <pesan commit>                         |
| GitHub Sync Status         | SINKRON (0 ahead / 0 behind)                    |
| Deployment Status          | AKTIF (PID: <pid>, Port: 12393 & 80)            |
| HTTP Status                | 200 OK (Aplikasi Web & Metadata Live2D)         |
| WebSocket Status           | CONNECTED (Direct 12393 & Nginx 80)             |
| Audio / TTS Status         | READY (Provider aktif & kredit terverifikasi)   |
| Perubahan Belum Diamankan  | NIHIL / <Daftar file jika ada>                  |
+----------------------------+-------------------------------------------------+
================================================================================
```
