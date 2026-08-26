# Audit Open-LLM-VTuber AI Waifu

Audit ini hanya memeriksa implementasi dan konfigurasi yang ada. Tidak ada source project yang diubah sebagai bagian dari audit.

Backend tidak sedang berjalan ketika audit dilakukan. Oleh karena itu, istilah **aktif** di laporan ini berarti konfigurasi yang akan dimuat oleh entrypoint ketika `run_server.py` dijalankan.

## Current Setup

**Character file:**

`/root/waifu/Open-LLM-VTuber/conf.yaml`, bukan file di folder `characters/`.

**Character aktif:**

`mao_pro`

**Character UID:**

`mao_pro_001`

**Nama karakter:**

`Mili`

**Persona prompt:**

Persona Bahasa Indonesia di `conf.yaml`, mulai sekitar baris 42. Ini konfigurasi nyata, bukan example config.

**Agent:**

`basic_memory_agent`

**LLM provider:**

`mistral_llm`

**LLM model percakapan:**

`mistral-small-latest`

Ini adalah model yang berbicara sebagai Mili. Model Codex yang mengedit repository sepenuhnya terpisah dan tidak digunakan untuk percakapan karakter.

**Endpoint:**

`https://api.mistral.ai/v1`

**Temperature:**

`1.0`

**Top P:**

Tidak tersedia dan tidak dikirim ke provider.

**Max output tokens:**

Tidak tersedia dan tidak dikirim ke provider.

**Parameter generation lain:**

Implementasi Mistral/OpenAI-compatible saat ini hanya mengirim:

- `model`
- `stream=True`
- `temperature`
- `tools`

Implementasinya berada di:

`src/open_llm_vtuber/agent/stateless_llm/openai_compatible_llm.py`

**Streaming dan segmentasi:**

- `faster_first_response: true`
- `segment_method: pysbd`
- MCP aktif dengan tool `time` dan `ddg-search`

## Persona Implementation

Alur prompt saat ini:

```text
conf.yaml persona_prompt
    + Live2D expression prompt
    + instruksi interruption
    + MCP tools/prompt bila diperlukan
    -> system message
    -> conversation history
    -> pesan user terbaru
```

Temuan detail:

- `run_server.py` benar-benar membaca `conf.yaml` pada startup, bukan example config.
- Persona diproses oleh `ServiceContext.construct_system_prompt()`.
- Hasilnya dikirim sebagai message dengan `role: system` pada setiap request LLM.
- Persona tidak menjadi bagian dari `_memory`, sehingga persona tidak ikut hilang ketika message history berubah.
- `BasicMemoryAgent` menambahkan instruksi interruption setelah persona.
- MCP native tools dikirim lewat parameter `tools`.
- Prompt MCP tekstual baru ditambahkan jika provider jatuh ke prompt-mode.
- Prompt group conversation dan proactive speaking hanya dimasukkan ketika fitur tersebut digunakan.

### Konflik prompt yang ditemukan

Persona Mili melarang tag seperti `[smirk]`, tetapi prompt Live2D yang ditempel setelah persona menyuruh model memakai tag ekspresi secara rutin.

Prompt tersebut berada di:

`prompts/utils/live2d_expression_prompt.txt`

Konflik ini dapat melemahkan persona dan menjelaskan kemunculan tag atau narasi ekspresi pada jawaban.

## Context Handling

Saat ini praktis belum ada context manager.

Setiap request berisi:

1. System prompt lengkap.
2. Seluruh isi `BasicMemoryAgent._memory`.
3. Pesan user terbaru.

Tidak ditemukan:

- batas jumlah message
- penghitungan token
- tokenizer
- pemotongan history
- context limit per model
- reserved output tokens
- summary history lama
- fallback saat context overflow

Hal positifnya: system/persona selalu ditambahkan terpisah pada setiap request, sehingga persona tidak dipotong oleh kode existing.

Namun, seluruh history terus bertambah sampai akhirnya provider berpotensi menolak request karena context terlalu besar.

## Conversation History

History berada di dua tempat:

- RAM: `BasicMemoryAgent._memory`
- Disk: `chat_history/<conf_uid>/<history_uid>.json`

Penyimpanan JSON diimplementasikan di:

`src/open_llm_vtuber/chat_history_manager.py`

Saat audit dilakukan, terdapat 12 file conversation untuk `mao_pro_001`. Masing-masing berisi sekitar 1 sampai 13 message.

Kemampuan existing:

- membuat conversation
- menampilkan daftar conversation
- memilih dan memuat conversation lama
- menghapus conversation
- menyimpan user dan assistant message
- menyimpan metadata per conversation
- memuat kembali history ke `BasicMemoryAgent`

Ketika conversation dipilih, seluruh message JSON dimuat kembali ke `_memory`. Tidak ada limit maupun summary ketika file dimuat.

## Persistence

| Kejadian | Hasil saat ini |
|---|---|
| Refresh browser | File lama tetap ada, tetapi frontend otomatis membuat conversation baru |
| Reconnect WebSocket | Membuat context sesi baru dan conversation baru |
| Ganti conversation manual | History yang dipilih dimuat kembali dari JSON |
| Restart backend | File JSON bertahan |
| Tutup dan buka aplikasi | File tetap ada, tetapi conversation terakhir tidak otomatis dipulihkan |
| Ganti karakter | Menggunakan namespace `conf_uid` karakter tersebut |

Frontend menjalankan `fetch-history-list` lalu langsung `create-new-history` setiap koneksi.

Lokasi implementasi frontend:

`/tmp/olv-mobile/src/renderer/src/services/websocket-service.tsx`

Karena itu, hubungan dapat terasa reset meskipun history lama sebenarnya masih tersimpan.

### Masalah isolasi sesi

Context WebSocket baru menerima `agent_engine` by-reference dari default context.

Artinya, satu instance `_memory` berpotensi digunakan lintas reconnect atau client sampai conversation tertentu dimuat. Ini perlu diperbaiki karena dapat mencampur state sesi.

Lokasi terkait:

- `src/open_llm_vtuber/websocket_handler.py`
- `src/open_llm_vtuber/service_context.py`

## Long-Term Memory

Belum ada long-term memory pada konfigurasi aktif.

Project memang memiliki pilihan agent lain seperti Letta dan Mem0, tetapi keduanya tidak aktif dan tidak perlu dipasang untuk kebutuhan tahap pertama.

`BasicMemoryAgent` hanya menyimpan transcript mentah. Belum ada:

- rolling summary
- fakta penting terpisah
- memory consolidation
- selective long-term memory

## Relationship State

Belum ada.

Tidak ditemukan state:

- `stranger`
- `familiar`
- `close`
- `dating`

Metadata JSON sebenarnya sudah mendukung field tambahan melalui `update_metadate()`, tetapi `BasicMemoryAgent` belum memanfaatkannya.

Metadata existing ini merupakan tempat integrasi paling kecil untuk relationship state tanpa database baru.

## Sudah Ada

- Persona prompt sebagai system message.
- Persona dikirim ulang pada setiap request.
- History percakapan persistent berbasis JSON.
- Multiple conversation.
- Character switching.
- Provider dan model switching melalui config.
- Streaming response.
- Live2D expression pipeline.
- MCP/tool calling.
- TTS, ASR, WebSocket, dan frontend existing.
- Metadata per conversation yang dapat diperluas.
- Debug log file level `DEBUG`, rotation 10 MB, retention 30 hari.

## Bermasalah / Kurang

- Tidak ada context-window management.
- Seluruh history dikirim terus-menerus.
- Tidak ada output-token reservation.
- Tidak ada summary history lama.
- `top_p` dan `max_tokens` belum didukung config Mistral existing.
- `temperature: 1.0` agak tinggi untuk persona stabil; target awal `0.8` lebih masuk akal.
- Prompt Live2D bertentangan dengan larangan tag ekspresi dalam persona.
- Example dialogue dan banyak instruksi negatif berisiko dijadikan pola atau template.
- Reconnect selalu membuat conversation baru.
- Conversation terakhir tidak otomatis dipulihkan.
- Agent instance dibagi by-reference antar sesi.
- Debug logging belum menampilkan token dan context statistics.
- Debug log saat ini dapat merekam system prompt dan seluruh message.
- Representasi config dalam debug berpotensi ikut mencetak credential.
- API key tersimpan plaintext dalam `conf.yaml`.

API key yang pernah dibagikan sebaiknya dirotasi dan dipindahkan ke environment variable.

## Belum Ada

- Token-aware context trimming.
- Ringkasan incremental history lama.
- Persistent relationship continuity.
- Automatic resume conversation terakhir.
- Context limit berdasarkan model/provider.
- Input/output token usage logging.
- Proteksi eksplisit agar credential tidak pernah masuk log.
- Test khusus persona, context panjang, relationship, restart, dan reset.

## Rencana Perubahan Minimum

### 1. Perbaiki isolasi sesi

Setiap WebSocket harus mendapat instance `BasicMemoryAgent` sendiri, bukan berbagi `_memory`.

### 2. Resume conversation dengan benar

Simpan dan pulihkan `history_uid` terakhir. Jangan otomatis membuat conversation baru setiap reconnect jika history valid tersedia.

### 3. Tambahkan context budget kecil di BasicMemoryAgent

Tetap pertahankan system/persona, recent messages, dan ruang output. Limit dibuat per provider/model dengan fallback estimasi aman.

### 4. Tambahkan rolling summary incremental

Hanya meringkas bagian lama ketika threshold tercapai. Summary dan fakta penting dapat disimpan dalam metadata JSON existing.

### 5. Relationship state di metadata existing

Gunakan satu field `relationship_status`, tanpa database atau sistem poin. Reset conversation tidak otomatis menghapusnya kecuali user memilih reset memory.

### 6. Perluas generation config existing

Tambahkan optional `top_p` dan `max_tokens` pada satu config/provider pipeline.

Untuk percobaan awal:

- `temperature: 0.8`
- `top_p: 0.9`

### 7. Hilangkan konflik persona dan Live2D

Ekspresi tetap boleh bekerja, tetapi instruksinya tidak boleh memengaruhi dialog yang terlihat atau memaksa tag terlalu sering.

### 8. Tambahkan debug summary yang aman

Log informasi berikut:

- character aktif
- agent aktif
- provider
- model
- temperature
- top_p
- max output tokens
- jumlah history message
- estimasi input token
- output token
- context limit
- apakah trimming dilakukan
- apakah summary digunakan
- relationship state

Jangan mencetak prompt lengkap, chat sensitif, API key, password, atau credential.

### 9. Tambahkan test terarah

- Persona selama 20 sampai 30 turn.
- Recent-context recall.
- Relationship berubah menjadi `dating`.
- Context panjang sampai trimming atau summary aktif.
- Restart backend.
- Reconnect WebSocket.
- Conversation switching.
- Reset conversation dan reset memory.

## Kesimpulan

Fondasi Open-LLM-VTuber yang ada sudah cukup. Tidak dibutuhkan Letta, Mem0, vector database, embedding server, atau rewrite besar.

Perubahan utama cukup dilakukan pada:

- lifecycle dan isolasi `BasicMemoryAgent`
- metadata history existing
- context budgeting dan summary incremental
- relationship state sederhana
- generation settings
- perilaku reconnect frontend
- debug logging yang aman
