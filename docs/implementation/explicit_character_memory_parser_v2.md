+# Explicit Character Memory Parser v2

## 1. Root Cause

Parser lama memakai dua regex sempit langsung di `BasicMemoryAgent`. Pola
`ya|yah` juga tidak memiliki abstraksi partikel yang dapat dipakai ulang,
sehingga variasi chat yang normal seperti `ingat yah`, `inget yaa`,
`catet`, atau `simpen` tidak tertangani secara konsisten. Ekstraksi payload
juga hanya memotong teks dari akhir regex tanpa klasifikasi hasil yang jelas.

Regression live berikut sekarang dikenali:

```text
ok, ingat yah, gw juga suka film tentang detektif
```

Hasil parse:

```text
action=remember
payload=gw juga suka film tentang detektif
matched_trigger=remember_ingat
```

## 2. Files Changed

- `src/open_llm_vtuber/character_memory_commands.py`
  - parser lokal terstruktur, normalization, trigger families, guards, dan
    payload extraction.
- `src/open_llm_vtuber/agent/agents/basic_memory_agent.py`
  - memakai hasil parser dan meneruskannya ke API character-memory existing.
- `tests/test_character_memory_commands.py`
  - matriks deterministik remember/forget, false-positive, persistence, dan
    invariance.
- `docs/implementation/explicit_character_memory_parser_v2.md`
  - laporan implementasi ini.

Salinan laporan tersedia di
`/root/waifu/explicit_character_memory_parser_v2.md`.

## 3. Normalization Strategy

Matching memakai salinan teks yang:

- di-trim;
- spasi berulang di-collapse;
- dibandingkan case-insensitive;
- menerima pemisah koma, titik dua, dash, ellipsis, beberapa tanda baca, dan
  emoji di batas command/payload;
- hanya menghapus scaffolding command dari payload.

Kapitalisasi dan kata pada fakta tetap dipertahankan. Pronoun seperti `gw`,
`gue`, `aku`, `saya`, `gua`, `ane`, dan `user` tidak ditulis ulang.

## 4. Remember Trigger Families

Parser mendukung:

- `ingat` / `inget`;
- `tolong ingat`, `tolong inget`, `tolong diingat`, `tolong diinget`;
- `jangan lupa`, `jgn lupa`, `jangan lupain`;
- `catat` / `catet`;
- `simpan` / `simpen` dengan qualifier eksplisit;
- `masukkan` / `masukin` ke ingatan;
- bentuk `buat/untuk ke depannya` dan `mulai sekarang`;
- English sederhana `remember` / `please remember`.

## 5. Forget Trigger Families

Parser mendukung:

- `lupakan` / `lupain`;
- `hapus dari ingatan`, termasuk `ingatan kamu` / `ingatanmu`;
- `hapus memory tentang`;
- `jangan ingat` / `jangan inget`, termasuk bentuk `udah ... lagi`;
- English sederhana `forget`.

## 6. Safe Conversational Prefixes

Command hanya dicari di awal pesan atau setelah satu prefix aman:

- `eh`
- `eh iya`
- `oh iya`
- `btw`
- `eh btw`
- `ngomong-ngomong`
- `ok`

Parser tidak mencari kata `ingat` secara arbitrer di tengah pesan panjang.

## 7. Supported Particles

Pattern reusable menerima:

- `ya`, `yaa`, `yaaa`
- `yah`
- `yak`
- `yap`
- `dong`
- `deh`

Tidak ada fuzzy matching atau edit distance; typo acak seperti `ingtt` dan
`cattt` tetap ditolak.

## 8. False-Positive Guards

Proteksi yang ditambahkan:

- command harus berada di posisi aman;
- payload wajib non-empty dan bermakna;
- recall question ditolak;
- ordinary statement seperti `gw jadi inget` tidak cocok;
- quoted/third-party use tidak cocok;
- `simpan file ini` tidak dianggap memory command;
- `catatannya ada di meja` tidak cocok dengan verb command;
- payload berbentuk pertanyaan ditolak secara konservatif.

## 9. Recall-Question Protection

Bentuk seperti berikut tidak memutasi memory:

- `ingat/inget gak|nggak|ga|ngga ...`
- `kamu masih ingat/inget ...`
- `ingat?` / `inget?`

Recall question tetap diproses sebagai chat biasa.

## 10. Temporary-Reminder Protection

Family `jangan lupa` diberi guard khusus. Task sementara seperti makan,
tidur, mematikan lampu, membalas, atau mengecek server tidak otomatis menjadi
long-term memory. Bentuk faktual eksplisit seperti `jangan lupa kalau gw suka
kopi` atau `game favorit gw Silent Hill` tetap diterima.

## 11. Remember-vs-Forget Priority

Urutan parser bersifat eksplisit:

1. forget;
2. remember;
3. none.

Karena itu `jangan ingat lagi kalau gw suka kopi` selalu menghasilkan
`forget`, bukan salah dikenali sebagai remember.

## 12. Payload Extraction

Scaffolding seperti `bahwa`, `kalau`, `kalo`, dan `kalo misalnya`
dihapus jika jelas merupakan connector command. Forget juga membersihkan
`soal`, `tentang`, `yang tentang`, dan `yang tadi tentang`.
Punctuation di tepi dibersihkan, sedangkan isi fakta tetap human-readable.

## 13. Backward Compatibility dan Runtime State

- API add/remove character-memory existing tetap digunakan.
- Dedup exact-normalized existing tetap digunakan.
- Schema `character_state/<conf_uid>.json` tidak berubah.
- Memory scope, token budget, persistence, relationship, history, dan rolling
  summary tidak berubah.
- Tidak ada state runtime baru yang dipersist.
- Tidak ada perubahan frontend.

## 14. Provider Call Count

`0` call tambahan. Parser hanya menjalankan normalization + regex lokal.
Satu test memasang fake LLM yang gagal jika dipanggil dan memastikan call
count tetap nol.

## 15. Tests Added

Suite baru mempunyai 16 test methods dengan lebih dari 100 deterministic
subcases, mencakup seluruh matriks command, bug live, normalization,
false-positive, empty payload, dedup, persistence lintas chat, API visibility,
relationship invariance, transcript invariance, dan zero-LLM-call.

| Test | Result |
|---|---|
| Remember variants | PASS |
| Forget variants | PASS |
| Live `ingat yah` regression | PASS |
| Recall-question negatives | PASS |
| Temporary-reminder negatives | PASS |
| Capitalization/spacing/punctuation/emoji | PASS |
| Empty/useless payload | PASS |
| Forget priority | PASS |
| Exact dedup | PASS |
| Cross-chat persistence | PASS |
| Existing memory API visibility | PASS |
| Relationship unchanged | PASS |
| Transcript unchanged | PASS |
| Additional LLM calls | PASS — 0 |
| Targeted parser + architecture tests | PASS — 35/35 |
| Full backend test suite | PASS — 195/195 |
| Ruff (`src`, `tests`) | PASS |
| Compileall (`src`, `tests`) | PASS |
| Active config validation | PASS |
| `git diff --check` | PASS |

Tidak ada live provider call yang dijalankan.

## 16. Git Diff Summary

Patch fungsional menambah satu parser utility dan satu test module, lalu
mengganti dua regex inline di `BasicMemoryAgent` dengan structured parse
result. Tidak ada perubahan pada storage, frontend, model, atau subsystem lain.

Unrelated runtime files `frontend`, `model_dict.json`, `character_state/`,
dan `cloudflared` tidak dimasukkan ke commit.

## 17. Live Verification Commands

Jalankan aplikasi, lalu uji:

```text
ok, ingat yah, gw juga suka film tentang detektif
```

Buka kontrol Ingatan Jangka Panjang Mili dan pastikan fakta bersih muncul.
Kemudian kirim:

```text
Lupain soal gw juga suka film tentang detektif
```

Pastikan fakta hilang. Negative smoke test:

```text
Inget gak kita tadi bahas apa?
Jangan lupa makan.
Simpan file ini.
```

Ketiganya tidak boleh menambah long-term memory.

Untuk regression lokal:

```bash
cd /root/waifu/Open-LLM-VTuber
LOGURU_LEVEL=ERROR UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_character_memory_commands
LOGURU_LEVEL=ERROR UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests
```

## 18. Remaining Limitations

Parser sengaja deterministik dan konservatif. Ia tidak memahami paraphrase
semantik di luar family yang didukung, tidak melakukan semantic deduplication,
dan tidak menerima typo arbitrer. Batas ini menjaga false-positive rendah tanpa
dependency atau provider call baru.

## Status

EXPLICIT CHARACTER MEMORY PARSER V2 SELESAI
