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

## 13. Capability-test wajah (rig mao_pro) — neutral/sad/angry

**Akar masalah "neutral tampak senyum":** idle motion `mtn_01.motion3.json` megang `ParamMouthUp = 1.0` (pose mulut netral-nya sendiri, loop motion 5.57s), DAN max asli param itu juga 1.0. Tapi palette Stage 3 `neutral` additive-nya KOSONG → tidak ada yang nge-counter baseline → mulut tetap ke-clamp di 1.0 = senyum permanen. Ini masalah NILAI, bukan writer-order (Time mode facial Stage 3/4 memang writer terakhir sebelum `model.update()`).

**Writer order terkonfirmasi (`lappmodel.ts update`):** `loadParameters → motion(set ParamMouthUp=1.0) → blink → expression → drag → breath → physics → lip sync(ParamA) → pose → Stage2 movement hook → Stage3 facial hook → model.update()`. Stage 3 API (brows/mouth/eye-smile) jadi penulis terakhir → tidak di-overwrite physics/pose/lipsync. Bukan masalah ordering.

**Solusi sementara (developer-only), aktivasi via URL `?capface=`:**
- `neutral`: `MouthUp:-1.0` (counter baseline 1.0 → ratakan senyum), brows/eyes netral, `Cheek:0` (clear blush residue).
- `sad`: mirror exp_05 — `MouthUp:-1.0, MouthDown:0.8, BrowLAngle/Form:-0.8, eyeOpen 0.92`.
- `angry`: mirror exp_08 — `MouthUp:-1.0, MouthAngry:1.0, MouthAngryLine:1.0, MouthDown:0 (jangan, biar gak jadi sad), EyeForm:1.0, brow furrow -0.9, eyeOpen 0.85`.
- Saat cap active: `controller.step()` TIDAK dipanggil → Stage 3 idle & Stage 4 contextual face di-bypass; Stage 2 body/head tetap jalan (hook terpisah).
- File: `use-live2d-idle-facial.ts` (TEMP_CAPABILITY_FACES + readCapabilityFace). Gampang dibalik (satu diff) setelah live-test.

**Kesimpulan kemampuan rig (belum tuning final):** semua 3 ekspresi secara teori bisa dibuat jelas — neutral butuh counter MouthUp -1.0; sad bisa lewat mulut turun + brows; angry bisa lewat pout line + EyeForm + brow furrow tanpa MouthDown. Belum ada satupun yang terbukti mustahil TANPA rerig kecuali live-test membuktikan.

## 14. Proactive follow-up: ganti bias "complain silence" jadi variasi + anti-repetisi

**Masalah live:** kalau user diem, follow-up proactive jadi repetitive/self-conscious ("kok malah nyuruh aku lanjutin sendiri?", "capek nungguin kamu", "mau lihat aku ngomong sendirian ya?") — terasa mekanis.

**Akar penyebab (2 sumber):**
1. `format_followup_instruction` di `proactive_chat.py` — instruksi lama: "strongly prefer reacting to the unanswered question first ... notice the silence with mild irritation ... escalating naturally ... (first: mild confusion; second: impatient; third: resigned/sulking)" → menjadikan komplain silence DEFAULT di tiap follow-up yang diabaikan.
2. `resolve_proactive_intent_decision` — SELALU paksa `FORCED_IGNORED_QUESTION → REACT_TO_IGNORED_QUESTION` saat question diabaikan, tanpa anti-repetisi.

**Solusi (permanen, tidak ubah persona/tuning/mapping):**
- **Softkan instruksi follow-up** (`format_followup_instruction`): model disuruh BIASANYA ganti arah (ganti topik, komentar, pemikiran, pertanyaan beda, lanjut sendiri), cuma sekali-kali akui keheningan (tease / "yaudah deh"), jangan komplain berulang, jangan ulang pertanyaan yang sama. Statement yang diabaikan → "not replying is completely normal, do NOT make an issue of the silence".
- **Guard deterministic anti-repetisi** (`_silence_complaint_recently_used(state)`): kalau `REACT_TO_IGNORED_QUESTION` sudah ada di `state.recent_proactive_intents` (sudah pernah komplain), turn berikutnya TIDAK dipaksa force lagi → jatuh ke seleksi semantic/bervariasi. Komplain pertama tetap diizinkan (state fresh = recent intents kosong).
- `record_proactive_sent` sudah melacak intent → menjaganya di window 3 turn.

**Cara kerja:** user diem → proactive #1 (boleh komplain sekali) → intent tercatat → proactive #2/#3 tidak dipaksa komplain → model otomatis pilih topik/observasi/pertanyaan beda. Proactive setelah user akhirnya balas → `record_user_activity` reset counters → normal lagi.

**Kontrak yang dipertahankan:** backoff/ignored limit tidak berubah, satu provider call, tidak ada eval LLM, casing teks assertion di test menyesuaikan frase baru.

**Catatan test quirk:** `context_window_override=2000` di `_make_agent` — teks follow-up sedikit lebih panjang bisa bikin test yang sekaligus kirim `followup_context` + `intent_context` lewat budget → "Proactive generation skipped because context does not fit" → output kosong. Jaga teks tetap ringkas.

## 15. State wajah intensitas-tinggi: angry_strong + strong_blush

**Tujuan:** dua state kontekstual high-intensity tambahan tanpa ngerusak state yang udah verified (neutral/sad_soft/angry_pout/small_smile/squint_smile). **Bukan** random idle.

**Cara menambah label semantic baru (hati-hati):**
- Label emotion yang valid = `emo_map.keys()` mao_pro/shizuku di `model_dict.json`. Biar parser (`extract_emotion_keys`) + prompt (`<insert_emomap_keys>`) kenal, label BARU harus ditambah ke emo_map DI KEDUA model (mao_pro & shizuku). Index = nomor preset exp_0X (exp_06 embarrassed, exp_08 angry: pakai index itu di emoMap, biar legacy extract juga wajar).
- `embarrassed`: index 6. `anger_strong`: index 8.
- Prompt guard: `[anger_strong]` HANYA utk kemarahan benar-benar kuat (fury/outburst), ordinary anger/tsundere/pout TETAP `[anger]`. `[embarrassed]` HANYA utk genuinely flustered/shy, bukan sekadar playful. Ini mencegah terulang bias overuse label (pelajaran #12).
- Frontend `contextual-emotion.ts`: `embarrassed -> strong_blush`, `anger_strong -> angry_strong`. Generic `joy/smirk/anger` TIDAK di-upgrade ke state kuat.

**Palette Stage 3:**
- `angry_strong`: (weight 0 = tak pernah dipilih random idle). Stack semua cue anger ke extreme SAFE-nya TANPA MouthDown (biar gak jadi sedih). Beda dari angry_pout: brow angle/form -1.0 (lebih mengerut dari -0.9) + eyeOpen 0.78 (lebih sipit dari 0.85). Cheek 0 (pipi merah + alis marah kayak malu, bukan marah).
- `strong_blush`: (weight 0). Ikuti resep shy rig (exp_06): Cheek 1.0 (max rig, jauh > small_smile 0.45 / squint 0.5), EyeSmile 0.35 (senyum malu kecil), brow naik 0.2, eyeOpen normal. MouthDown/Angry 0 (bukan sedih/marah).
- Karena `weight:0` tanpa `longIdleWeight`, `pickWeighted` (yang hanya tambah bobot `w>0`) otomatis skip → tidak muncul di idle/long_idle. Tapi MAP `CONTEXTUAL_EMOTION_MAP` tetap di-validasi "semua face id harus ada di palette" — jadi state baru WAJIB ada di palette biar response face bisa di-claim.

**Cara trigger live test:** minta Mili benaran marah keras (bukan kesal biasa) → model keluar `[anger_strong]`. Atau minta Mili nge-bangun terkejut/malu beneran → `[embarrassed]`. Cek mapping di jawaban.

**Catatan testing quirk:** test `strong_blush` "release clears Cheek" perlu toleransi epsilon (`Math.abs(cheek) < 0.01`) karena interpolation meninggalkan residu floating ~1e-11, bukan 0 persis.

## 16. angry_strong visual distinction — headroom yang tersisa cuma mata

**Masalah live:** `[anger_strong]` sudah benar (label), tapi visual nyaris sama dengan `angry_pout` (bedanya cuma sedikit).

**Temuan audit:** sebagian besar cue anger SUDAH JENUH (mentok real max):
- `ParamMouthUp: -1.0` (min), `ParamMouthAngry: 1.0` / `ParamMouthAngryLine: 1.0` (max),
- `ParamEyeForm: 1.0` (max), `ParamBrowAngle/Form: -1.0` (min, sudah -0.9 → -1.0 = beda kecil).
- `model3.json` TIDAK mencantumkan range param (Live2D default -1..1; EyeOpen 0..1) — yang klasik `FACIAL_RANGES` di frontend cuma DOKUMENTASI, bukan clamp penulis. Jadi nilai additive langsung sampai rig; CubismModel yg clamp ke min/max asli param.

**Headroom visual yang TERSISA (belum disentuh angry_pout):**
1. **`EyeOpen` multiply** — bisa dikecilkan jauh (0.78 → 0.62).
2. **`EyeLSmile/EyeRSmile` NEGATIF** — menarik sudut mata tegangan/turun. angry_pout sengaja EyeSmile = 0, jadi ini pembeda utama "cute pout → garang".

**Fix angry_strong** (tanpa sentuh angry_pout, tanpa rerig, tanpa MouthDown, Cheek tetap 0): `EyeOpen x0.62` + `EyeSmile ±(-0.6)`, sisanya tetap (MouthAngry/Line 1.0, MouthUp -1, brow angle/form -1.0, EyeForm 1.0, MouthDown 0, Cheek 0).

**Kesimpulan rig:** kalau ini masih belum cukup beda, kemungkinan besar rig mao_pro udah mentok (mouth/brow/eye-form all saturated) → butuh rerig / ekspresi, bukan tuning byar.

## Log perubahan catatan ini

- `2026-08-30` — Inisialisasi: catat 11 pelajaran terkonfirmasi dari sesi Stage 3/4/5.
- `2026-08-30` — Tambah #12: bias emosi `[smirk]` default & jalur turn-level latch (Case A terkonfirmasi; prompt-guard + diagnosa jujur).
- `2026-08-30` — Tambah #13: capability-test neutral/sad/angry via `?capface=`, akar smile-baseline MouthUp=1.0, writer order terkonfirmasi.
- `2026-08-30` — Tambah #14: proactive follow-up kena bias "complain silence" (2 sumber); soft-kan instruksi + guard anti-repetisi.
- `2026-08-30` — Tambah #15: state intensitas-tinggi angry_strong + strong_blush (label emo_map baru embarrassed/anger_strong, weight 0, mapping contextual).
- `2026-08-30` — Tambah #16: angry_strong visual — mouth/brow/eye-form jenuh; headroom cuma EyeOpen + EyeSmile negatif.