# Catatan Solusi (Lessons Learned)

> **Aturan pakai:** File ini adalah sumber kebenaran habit. Setiap kali mulai bekerja, BACA file ini
> dulu. Setiap kali menemukan MASALAH + SOLUSI yang terkonfirmasi, TAMBAHKAN di sini. Jangan ulangi
> kesalahan yang sudah tercatat. Server ini: `Open-LLM-VTuber` (backend) + `Open-LLM-VTuber-Web`
> (frontend), model `mao_pro`.

---

## 1. Manajemen RAM Server

**Masalah:** Server backend makan banyak RAM kalau nyala terus.
**Solusi:** Sebelum tiap perubahan/build/test/deploy, **Matiin server dulu**. Nyalakan lagi HANYA saat
perlu live test, dan matiin lagi setelah selesai.

- Matiin: `pkill -f run_server.py` (atau cek `pgrep -af run_server.py` dulu).
- Cek nyala: `pgrep -af run_server.py` + `ss -ltnp | grep :8880`.

## 2. Cara NYALAKAN server yang benar

**Masalah:** `cmd > log 2>&1 & disown` di dalam tool SYNC sering mati saat command selesai (child shell ikut diterminasi).
**Solusi:** Relaunch dengan detach penuh lalu tunggu boot:

```bash
cd /root/waifu/Open-LLM-VTuber && setsid .venv/bin/python run_server.py >> /tmp/server_setsid.log 2>&1 < /dev/null & disown; sleep 1; echo done
```

- **Boot lama (±45–60 detik)** karena init MCP (`time`, `ddg-search`) + pengecekan model. Jangan langsung panik kalau port belum up.
- Cara verifikasi (tunggu ~45s):
```bash
pgrep -af "run_server.py"       # proses jalan
ss -ltnp | grep :8880           # port LISTEN
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8880/   # → 200
grep -o 'main-[A-Za-z0-9_-]*\.js' frontend/index.html | head -1    # bundle live
```
- **Background mode tool (`process_type: BACKGROUND`) TIDAK diimplementasikan** di lingkungan ini — jangan dipakai.

## 3. Deploy bundle frontend

**Masalah:** Mau ganti bundle tapi harus bisa rollback.
**Solusi:**
```bash
cp -r frontend frontend_backup_<nama>_$(date +%Y%m%d_%H%M%S)   # backup
cp <dest>/frontend/dist/...js frontend/assets/                   # copy bundle baru
# edit index.html: ganti nama file js/css ke yang baru
# verifikasi: curl GET /, GET /assets/<bundle>.js → 200, cek penanda fitur di bundle
```
- Selalu verifikasi bundle benar-benar disajikan (bukan cache) & cek penanda fitur (`grep` penanda di JS).
- Jangan merge ke main kecuali diminta. Jangan deploy tanpa izin live test.

## 4. Debug bridge ke server log (untuk tes dari Android tanpa DevTools)

**Masalah:** User tes dari Chrome Android, sulit buka DevTools Console.
**Solusi:** Buat endpoint HTTP debug kecil di backend yang nulis metadata ke `/tmp/server_setsid.log`,
lalu frontend kirim metadata via GET/POST fire-and-forget. User cukup grep.

```bash
grep "S4TRACE" /tmp/server_setsid.log | tail -100      # contoh pattern trace
grep -E "EMODIAG" /tmp/server_setsid.log | tail -80
```

**Aturan payload debug:**
- BOLEH: timestamp relatif, event name, label emotion, resolved faceId, claim/keep/release, reason, source, indeks sentence.
- JANGAN PERNAH: isi chat user, isi response, memory, persona, API key.
- Harus HAPUS semua instrumentation sementara sebelum final (sudah tercatat sebagai aturan di Stage 3/4/5).

## 5. Runtime-order Live2D (bug yang pernah menggigit)

**Masalah:** Stage 2 dulu kena bug: offset parameter di-generate benar tapi di-overwrite sistem Live2D lain di frame yang sama (physics, pose, lip sync, dll) sebelum `_model.update()`.
**Solusi:** Jangan asumsi menulis parameter = frame akhir menyimpannya. Audit urutan update nyata di
`WebSDK/src/lappmodel.ts`. Susunan final yang TERVERIFIKASI:
```
motion → expression → blink → breath → physics → lip sync (ParamA) → pose
→ Hook Stage 2 (movement) → Hook Stage 3/4 (facial) → _model.update()
```
- **Hook facial Stage 3/4 harus jadi PENULIS TERAKHIR** sebelum `_model.update()` biar tidak di-overwrite.
- **Harus ada regression test runtime-order** biar bug overwrite gak balik lagi.

## 6. Ownership parameter (jangan saling tabrak)

| Pemilik | Parameter |
|---|---|
| Lip sync | **ParamA** (jangan pernah disentuh Stage 3/4/5) |
| Stage 2 (movement) | ParamAngleX/Y/Z, ParamBodyAngleX, ParamEyeBallX/Y |
| Stage 3/4 (facial) | mouth form, brow, eye smile/form, EyeOpen (MULTIPLY only), blush/cheek |
| Stage 5 | HANYA koordinasi ownership — TIDAK BOLEH menulis parameter Live2D |

- `EyeOpen` hanya boleh pakai semantik MULTIPLY (netral = 1.0), jangan absolute-set biar blink aman. Kalau dilepas, smooth balik ke 1.0.

## 7. Stage 4 contextual face — timing & lifecycle yang sudah benar

Urutan yang TERVERIFIKASI ABIS di live test:
- **Marker emotion di START response** (sebelum teks terlihat): `[smirk] text ...` → jangan di tengah/akhir.
- Marker tetap di-strip dari: teks tampil, TTS, history.
- **Turn-level latch:** sekali non-neutral emotion masuk → claim face dan TAHAN sepanjang turn. Sentence berikutnya yg tak punya marker = "no new emotion", BUKAN reset ke neutral. Jangan release dgn `reason=neutral_sentence`.
- **Audio ON:** claim sebelum audio → audio_start → face_keep (refresh tiap aktivitas) → playback_complete (SEKALI) → release `reason=turn_end`. JANGAN ada text_only_hold di turn yg benar-benar ada audio_played.
- **Muted/teks-only:** claim dari emotion METADATA (bukan dari audio sukses) di saat task tiba → text_only_hold ±2.5–6s → release. Jangan release instan cuma karena gak ada audio.
- **Audio-played latch:** sekali `audio_start` nyata terjadi → `turnHadAudioPlayback=true` tetap sampai turn itu selesai. Reset hanya di turn baru/interruption.
- **Safety timeout** itu FALLBACK buat lifecycle nyangkut, bukan durasi normal. Refresh/keep saat ada aktivitas turn. Jangan pakai fixed 6s dari claim sebagai lifetime utama.
- **Interruption/session switch** harus langsung release ownership sementara.
- **Playback_complete harus ONE authoritative per turn.** Kalau ada duplicate emitter, dedupe di sumbernya + jaga idempotensi (guard `already_released`) sebagai defense-in-depth.

## 8. Stage 3 idle facial — pola yang dipakai

- Palette IDLE final (subtle, joy/gembira paling umum): `neutral, small_smile, squint_smile, pout_small, angry_pout, sad_soft`. LONG_IDLE: tambah `sleepy_soft/relaxed`.
- Anti-repeat (jangan langsung repeat state sebelumnya). Weighted random, bukan random penuh.
- Interpolasi smooth, NO snapping. Suppress saat: active, speaking, drag, motion, response-face aktif.
- **Keterbatasan rig mao_pro (ACCEPTED):** perbedaan halus antar state mulut (pout vs sulk vs angry) susah dibedain jelas di rig ini. Jangan tuning berlebihan demi kesempurnaan visual yg gak mungkin.
- ParamMouthUp rig ini ada baseline ~1.0 dari motion idle → offset aditif kecil tak terlihat. Jangan andalkan MouthDown untuk angry_pout (terbaca sedih). Sad → MouthDown boleh, angry → jangan.

## 9. Stage 5 orchestrator (ownership)

- Satu `behavior-orchestrator.ts` (pure) jadi sumber kebenaran kepemilikan. Query: `canRunIdleFace()`, `canRunIdleMovement()`, `isResponseOwned()`, `shouldSuppressAutonomous()`.
- Priority: `interruption > response > speaking > user_active > drag > intentional_motion > idle_face > idle_movement > long_idle > neutral`. Channel-aware (face/movement/lip/lifecycle).
- Idle face & movement TIDAK boleh jalan saat response ownership aktif (termasuk jendela proactive generasi).
- Stage 2 movement BIJAK BOLEH terus saat response-face aktif (spec D) — jangan freeze seluruh avatar.
- Session switch → `clearTransientOwnership()`.
- **Jangan redeploy kalau cuma test yang berubah** — cek apakah bundle hash berubah dulu. Kalau byte-identical, cukup commit test.

## 10. Jalankan tests (quirk environment)

**Frontend (`/root/waifu/worktrees/mili-stage5-web`):**
- **Path alias `@/*` HARUS di-resolve dgn `--tsconfig tsconfig.web.json`** — kalau tanpa itu, `@/utils/...` gagal resolve di beberapa file.
- Kalau runner glob satu proses gagal resolve all files → jalankan **per-file** (loop `for f` atau manual).
- Build: `npm run build` (butuh `node_modules`; worktree baru kadang perlu `ln -s` dari worktree lain yg punya).

**Backend (`/root/waifu/Open-LLM-VTuber`):**
- Ada module test yg error karena env-import pra-ada (`test_mili_ui_response_polish` + `test_proactive_chat` iniitive) — bedakan error PRA-ADA vs dari perubahan kita sebelum report gagal.
- Verifikasi: `ruff`, `compileall`, `git diff --check`.

## 11. Pelajaran kerja umum

- **GIT PATH ALIAS / `code_search` (`rg`) di beberapa cwd bisa error `ENOTDIR`** — fallback ke `grep` via terminal kalau search tool gagal.
- Saat laporan: kasih tabel empirical (cek → hasil), jangan klaim sukses cuma dari unit test. Stage 2/3/4 membuktikan perlu live test visual Android.
- Kalau user bilang "matiin server biar hemat RAM" → jalankan mati dulu SEBELUM kerja, nyalain cuma pas live test.
- Jangan mulai stage fitur berikutnya (Stage 6 dst) tanpa izin & tanpa finalisasi stage sekarang.
- Debug instrumentasi TEMPORER boleh dipasang utk buktikan hipotesis, tapi WAJIB dibersihkan sebelum final.

---

## 12. Bias emosi Signature (smirk default) & jalur latch

**Masalah live:** konteks sedih & netral sama-sama dapat `emotions=smirk` → wajah melenceng. Penyebabnya bukan mapping/ownership — frontend menerima metadata & latch dengan benar. Akar masalah = **prompt LLM tidak menjelaskan semantik per label**, jadi LLM persona tsundere gampang pakai `[smirk]` sebagai default.

**Solusi (TASK 1):** Tambah guidline per-label di `prompts/utils/live2d_expression_prompt.txt`:
- Default = TANPA marker (atau `[neutral]`) utk respons faktual/datar/tenang — ini kasus paling umum.
- `[smirk]` HANYA utk tone teasing/sly/smug/playful yang jelas; JANGAN utk tsundere default, candaan receh, atau respons sarkas ringan. Saat ragu → no marker / `[neutral]`.
- `[sadness]` hanya utk response yang benar-benar sedih/kecewa/wistful/mirip. `[anger]` hanya utk kesal/sakit hati yang nyata. `[joy]` utk senang/cerah.
- Pilih berdasar EMOSI AKTUAL dari response yang ditulis, bukan kata-kata user, bukan habit kepribadian.

**TASK 2 (latch):** Konfirmasi bahwa sentence tak-bertekanan marker TIDAK menimpa latched face. Alur `decideResponseFace`:
- `incoming === null` → release (turn end/interruption).
- `incoming === 'neutral'/''` → `refresh` (KEEP latch, refresh watchdog) — JANGAN release.
- Face valid lain → `claim` (ganti face utk turn).
- Subscriber `use-live2d-idle-facial` & `use-audio-task` sudah terikat dgn biner ini.

**Trap diagnostik:** JANGAN tambah `emoDiag({ claim:true })` tanpa syarat setelah publish — itu MENYESATKAN. Decision otentik ada di subscriber (`decision:${kind}`). Kalau perlu trace, pakai field jujur (`incoming`/`on`) dan serahkan penentuan ke subscribe.

## Log perubahan catatan ini

- `2026-08-30` — Inisialisasi: catat 11 pelajaran terkonfirmasi dari sesi Stage 3/4/5.
- `2026-08-30` — Tambah #12: bias emosi `[smirk]` default & jalur turn-level latch (Case A terkonfirmasi; prompt-guard + diagnosa jujur).