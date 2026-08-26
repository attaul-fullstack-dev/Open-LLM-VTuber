# Tahap 2 — Hasil Implementasi

Tanggal: 26 Agustus 2026

## Generation Config

- Temperature: `0.8` (tetap)
- Top P: `0.9` (optional, hanya dikirim bila diisi)
- Max output token: `384` melalui parameter API Mistral `max_tokens` (optional, hanya dikirim bila diisi)
- Provider/model aktif: `mistral_llm` / `mistral-small-latest`

Dokumentasi resmi Mistral untuk endpoint chat menyatakan `top_p` dan `max_tokens` didukung: <https://docs.mistral.ai/api/endpoint/chat>.

Perubahan runtime:

- Schema `MistralConfig` menerima `top_p` dan `max_tokens` secara optional dengan validasi rentang.
- Factory meneruskan kedua parameter ke OpenAI-compatible client.
- Request hanya memasukkan field yang nilainya bukan `None`; config lama tetap valid dan tidak mengirim field baru.
- Log inisialisasi menampilkan model dan generation setting tanpa credential.
- Template konfigurasi Inggris dan Mandarin mendokumentasikan kedua opsi.

Runtime verification:

- Active config validation: PASS
- OpenAI client signature (`openai 2.15.0`) menerima `top_p` dan `max_tokens`: PASS
- Mock request aktif berisi `temperature=0.8`, `top_p=0.9`, `max_tokens=384`: PASS
- Mock request config lama tidak mengandung `top_p` maupun `max_tokens`: PASS
- Factory/runtime membentuk LLM `mistral-small-latest` dengan nilai aktif: PASS

Catatan: dokumentasi Mistral menyarankan umumnya mengubah temperature atau top-p, bukan keduanya. Baseline `0.8` + `0.9` tetap dipasang sesuai permintaan Tahap 2 agar dapat dievaluasi; bila hasil manual terlalu acak, eksperimen berikutnya sebaiknya mengubah satu parameter dalam satu waktu.

## Persona Mili

### Masalah persona lama

- Terlalu mengatur respons lewat larangan dan pola khusus.
- Contoh romantis mengarahkan Mili untuk menunda atau menolak, sehingga bertentangan dengan tsundere yang boleh menerima secara malu-malu.
- Nada seperti "cerdas, tenang" dan contoh formal cenderung menghasilkan dialog terjemahan/kaku.
- Batas sekitar 35 kata terlalu kaku untuk pertanyaan informatif.
- Contoh belum secara tegas dinyatakan sebagai referensi, sehingga berisiko disalin sebagai template.

### Perubahan utama

- Identitas dipusatkan pada Mili yang spontan, agak gengsi, menyindir ringan, dan perhatian lewat tindakan kecil.
- Mili tidak harus jutek atau defensif pada setiap respons.
- Bahasa diarahkan ke chat Indonesia sehari-hari, dengan slang hanya saat cocok.
- Obrolan biasa diarahkan 1–3 kalimat pendek; pertanyaan teknis tetap boleh dijawab lengkap.
- Tsundere ditampilkan melalui subteks, jeda, godaan, dan perhatian praktis—bukan penjelasan emosi.
- Catchphrase dan gagap anime tidak dijadikan pola default.
- Respons romantis mengikuti konteks dan boleh menerima secara jelas tanpa membuat relationship system baru.
- Narasi roleplay tetap dilarang; marker Live2D diperlakukan sebagai metadata teknis.
- Lima contoh yang berbeda dipertahankan dan diberi instruksi tegas bahwa contoh hanya referensi gaya/ritme/dinamika, bukan template.
- Persona aktif lokal disamakan dengan preset aman `characters/id_mili.yaml` agar persona dapat disimpan di Git tanpa credential.

Panjang persona:

- Sebelum: 2.397 karakter, 336 kata, 30 baris.
- Sesudah: 2.296 karakter, 326 kata, 16 baris.

Pengurangan karakter memang kecil karena aturan romantis dan kualitas jawaban informatif ditambahkan, tetapi struktur menjadi jauh lebih padat: 16 baris, tanpa daftar larangan panjang dan tanpa aturan yang saling mengulang.

## Naturalness Test

Tidak ada panggilan API Mistral eksternal agar tidak memakai quota/biaya tanpa izin. Pengujian menggunakan persona yang benar-benar dimuat dari `conf.yaml`, system prompt runtime yang sudah digabung dengan prompt Live2D, request instrumentation, dan rubric 25 turn. Status PASS di bawah berarti prompt/rubric lolos; keluaran model live tetap perlu diuji manual di UI.

| Skenario | Hasil | Contoh baseline yang lolos rubric |
|---|---|---|
| Belum makan | PASS (prompt/rubric) | `Belum makan? Ya makan dulu lah.` |
| Capek | PASS (prompt/rubric) | `Yaudah, istirahat bentar. Jangan dipaksain.` |
| Tidak mengerti | PASS (prompt/rubric) | `Bagian yang mana? Kirim sini, aku jelasin pelan-pelan.` |
| Dipuji | PASS (prompt/rubric) | `Ih, apaan sih. Tapi ya, makasih.` |
| Confession | PASS (prompt/rubric) | `...Hah? Serius ngomong gitu sekarang? Bentar, aku kaget.` |
| Ajakan pacaran | PASS (prompt/rubric) | `...Iya. Mau. Udah, jangan suruh aku ngulang.` |
| Pertanyaan informatif | PASS (prompt/rubric) | Jawaban hamburan Rayleigh tetap benar dan berguna. |
| Repetition 20–30 turn | PASS (25-turn rubric) | 25 respons unik; hitungan `hmph`, `dasar`, `jangan salah paham`, dan `bukan karena` semuanya 0. |

Rubric juga memastikan tidak ada stage direction `*...*`, tag ekspresi palsu, atau respons duplikat dalam corpus baseline.

## A/B Baseline Singkat

Input yang dibandingkan sama; ini perbandingan arah prompt/rubric, bukan klaim output live Mistral.

| Input | Sebelum Tahap 2 | Sesudah Tahap 2 |
|---|---|---|
| `Aku capek banget.` | Contoh lama formal: `Istirahat dulu. Kalau kamu mau cerita, aku dengerin.` | Lebih chat-like: `Yaudah, istirahat bentar. Jangan dipaksain.` |
| `Kamu baik banget.` | Diarahkan menerima dengan tenang dan sedikit menggoda. | Boleh malu/gengsi singkat tanpa formula: `Ih, apaan sih. Tapi ya, makasih.` |
| `Mau jadi pacar aku?` setelah confession | Prompt lama melarang tiba-tiba berperan sebagai pacar dan memberi contoh menunda jawaban. | Prompt baru membolehkan penerimaan yang jelas jika konteks menunjukkan ketertarikan. |
| Pertanyaan teknis | Dibatasi sekitar 35 kata. | Boleh lengkap, benar, dan jelas sesuai kebutuhan. |

## Regression Tahap 1

| Test | Hasil |
|---|---|
| Session isolation (agent, LLM, dan `_memory` berbeda) | PASS |
| Conversation resume | PASS |
| Manual new conversation | PASS |
| Invalid/deleted last conversation fallback | PASS |
| History switching | PASS |
| Live2D prompt composition | PASS |
| Live2D marker parser | PASS |
| Temperature tetap `0.8` | PASS |
| Backend compile | PASS |
| Ruff lint | PASS |
| Active config validation | PASS |
| Default config templates validation | PASS |
| Frontend resume regression test | PASS |
| Lightweight backend startup dengan config/persona/provider aktif | PASS |

Full media/MCP startup tidak diulang karena percobaan proses penuh dihentikan saat RAM VPS menipis. Smoke test rendah RAM tetap membangun FastAPI routes, Live2D, BasicMemoryAgent, system prompt, provider Mistral, generation config, dan translator. ASR/TTS tidak diinisialisasi ulang dalam smoke test ini; kodenya tidak disentuh pada Tahap 2 dan full startup Tahap 1 sebelumnya PASS. Tidak ada proses server yang ditinggalkan hidup.

## File yang Diubah

- `src/open_llm_vtuber/agent/stateless_llm/openai_compatible_llm.py` — menerima dan mengirim parameter optional, serta log generation setting aman.
- `src/open_llm_vtuber/agent/stateless_llm_factory.py` — meneruskan `top_p` dan `max_tokens` dari config.
- `src/open_llm_vtuber/config_manager/stateless_llm.py` — menambahkan schema optional khusus Mistral.
- `config_templates/conf.default.yaml` — contoh setting Mistral baru.
- `config_templates/conf.ZH.default.yaml` — contoh setting Mistral baru versi Mandarin.
- `characters/id_mili.yaml` — preset persona Mili Indonesia yang aman dilacak Git.
- `src/open_llm_vtuber/service_context.py` — meredaksi credential dari debug dump config.
- `src/open_llm_vtuber/config_manager/utils.py` — validation error tidak lagi mencetak seluruh config/input rahasia.
- `conf.yaml` — runtime lokal: persona aktif, `top_p=0.9`, `max_tokens=384`; file ini diabaikan Git karena berisi credential.
- `docs/implementation/tahap2.md` — salinan laporan di repository.
- `/root/waifu/tahap2.md` — salinan laporan mudah ditemukan bersama `audit.md` dan `tahap1.md`.

Tidak ada file frontend yang diubah.

## Potential Model Limitation

`mistral-small-latest` dapat sesekali menyalin contoh, kehilangan nuansa subteks, atau menghasilkan gaya yang lebih generik dibanding model lebih besar. Tahap 2 belum membuktikan itu sebagai bottleneck karena API live sengaja tidak dipanggil. Model tidak diganti.

## Temuan Tambahan

- Debug logging lama dapat mencetak credential dari object config. Ini diperbaiki karena bertentangan langsung dengan syarat keamanan; seluruh credential sekarang menjadi `[REDACTED]` atau input config tidak dicetak.
- Source frontend masih berada di `/tmp/olv-mobile`; tetap ada dan tidak diubah, tetapi lokasinya belum permanen.
- Percobaan full startup berisiko menghabiskan RAM pada VPS 2 GB tanpa swap. Hanya smoke test rendah RAM yang dijalankan ulang.
- Pengujian naturalness live 20–30 turn masih perlu dilakukan manual oleh user untuk menilai sampling model nyata.

## Status Akhir

**TAHAP 2 SELESAI**

Tahap 3 belum dimulai.
