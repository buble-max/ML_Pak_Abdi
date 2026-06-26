# Sistem Deteksi Bahasa Isyarat BISINDO

Menjalankan API:

```bash
python -m api.main
```

Proyek ini adalah sistem deteksi bahasa isyarat BISINDO berbasis Machine
Learning dan Computer Vision. Sistem menerima input dari webcam atau API,
mengekstraksi landmark tangan menggunakan MediaPipe Hand Landmarker, lalu
memprediksi kelas gesture dengan model CNN + LSTM.

Target kelas aktif saat ini berjumlah 76 kelas:

- 26 huruf alfabet: `a` sampai `z`
- 10 digit: `digit_0` sampai `digit_9`
- 40 kata/frasa, misalnya `word_halo`, `word_makan`, `word_minum`,
  `word_terima_kasih`, dan lain-lain

Daftar label lengkap tersimpan di `model/saved/labels2.json` dan dibuat dari
konfigurasi pusat di `config.py`.

## Ringkasan Fitur

- Deteksi gesture real-time dari webcam.
- Ekstraksi 2 tangan, masing-masing 21 landmark dengan koordinat `(x, y, z)`.
- Normalisasi landmark agar lebih tahan terhadap perubahan posisi, ukuran
  tangan, dan jarak kamera.
- Sequence temporal sepanjang 30 frame untuk mengenali gesture statis maupun
  gerakan.
- Model CNN + LSTM dengan output softmax.
- Smoothing prediksi menggunakan EMA, majority vote, dan confidence threshold
  untuk mengurangi flicker.
- Pipeline lengkap: download dataset, rekam dataset, preprocessing,
  augmentation, training, inference, dan API.
- FastAPI server dengan endpoint REST dan WebSocket.

## Arsitektur Singkat

```text
Webcam / Image / Landmark
        |
        v
OpenCV + MediaPipe Hand Landmarker
        |
        v
Landmark tangan: max 2 tangan x 21 titik x 3 koordinat
        |
        v
Normalisasi + padding tangan kosong
        |
        v
Frame feature: 126 nilai
        |
        v
SequenceBuffer: 30 frame x 126 fitur
        |
        v
CNN per frame + LSTM temporal
        |
        v
Softmax 76 kelas
        |
        v
Smoothing + threshold
        |
        v
Label gesture + confidence
```

Format input model:

```text
(batch, 30, 126)
```

Keterangan:

- `30` adalah panjang sequence (`SEQUENCE_LENGTH`).
- `126` berasal dari `2 tangan x 21 landmark x 3 koordinat`.

## Struktur Folder

```text
Hand_sign/
|-- api/
|   |-- main.py               # FastAPI server
|   `-- example_client.py     # Contoh client REST dan WebSocket
|-- augmentation/
|   `-- augment.py            # Augmentasi sequence landmark dan class balancing
|-- dataset/
|   |-- download_dataset.py   # Download dataset alfabet BISINDO dari GitHub
|   |-- download_from_drive.py# Download ZIP dataset dari Google Drive
|   |-- record_word_gestures.py
|   |-- record_landmarks_live.py
|   |-- raw/                  # Gambar mentah alfabet/digit
|   |-- raw_words/            # Frame gesture kata per clip
|   `-- processed/            # File .npy hasil preprocessing
|-- docs/
|   `-- ARCHITECTURE.md       # Dokumentasi teknis arsitektur
|-- inference/
|   |-- predictor.py          # Predictor reusable + anti-flicker
|   `-- realtime_webcam.py    # Inference webcam real-time
|-- model/
|   |-- architecture.py       # Definisi model CNN + LSTM
|   `-- saved/                # Model dan label hasil training
|-- preprocessing/
|   |-- landmark_extractor.py # Build dataset X.npy dan y.npy
|   |-- mp_hand_landmarker.py # Wrapper MediaPipe Tasks API
|   |-- normalizer.py         # Normalisasi landmark
|   |-- sequence_builder.py   # Buffer sequence real-time
|   `-- video_to_frames.py    # Konversi video mentah menjadi frame
|-- training/
|   |-- train.py              # Training lokal
|   `-- train_colab.ipynb     # Training via Google Colab
|-- utils/
|   `-- labels.py             # Mapping label ke index dan sebaliknya
|-- config.py                 # Path, kelas, hyperparameter, dan setting global
|-- requirements.txt          # Dependency Python
`-- run_pipeline.py           # Orchestrator pipeline end-to-end
```

## Kebutuhan Sistem

Disarankan:

- Python 3.10 sampai 3.12
- Webcam untuk perekaman dataset dan real-time inference
- GPU opsional untuk training lebih cepat
- Git, jika ingin menjalankan `dataset/download_dataset.py`

Dependency utama:

- TensorFlow / Keras
- NumPy
- OpenCV
- MediaPipe
- scikit-learn
- FastAPI
- Uvicorn
- gdown

Semua dependency tercantum di `requirements.txt`.

## Instalasi

Masuk ke root proyek:

```bash
cd "D:\Kuliah\P.Computer V\Hand_sign"
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment di Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependency:

```bash
pip install -r requirements.txt
```

Catatan: saat MediaPipe Hand Landmarker pertama kali dipakai, file
`hand_landmarker.task` akan diunduh otomatis ke `model/mp_assets/`.

## Konfigurasi Utama

Seluruh konfigurasi penting ada di `config.py`.

Beberapa nilai penting:

| Konfigurasi | Nilai default | Fungsi |
| --- | ---: | --- |
| `SEQUENCE_LENGTH` | `30` | Jumlah frame per sample |
| `MAX_HANDS` | `2` | Maksimal tangan yang diproses |
| `NUM_LANDMARKS` | `21` | Jumlah landmark per tangan |
| `FEATURES_PER_FRAME` | `126` | `2 x 21 x 3` |
| `NUM_CLASSES` | `76` | Total kelas gesture |
| `MODEL_PATH` | `model/saved/bisindo_new.keras` | Model utama |
| `LABELS_PATH` | `model/saved/labels2.json` | File label utama |
| `CONFIDENCE_THRESHOLD` | `0.70` | Ambang confidence inference |
| `BUFFER_SIZE` | `10` | Buffer voting prediksi |
| `SMOOTHING_MIN_VOTES` | `6` | Minimum vote agar label dianggap stabil |

## Dataset

Proyek mendukung beberapa sumber dataset.

### 1. Dataset Gambar Alfabet BISINDO

Script `dataset/download_dataset.py` mengunduh dataset dari repository:

```text
https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset
```

Jalankan:

```bash
python -m dataset.download_dataset
```

Output akan disalin ke:

```text
dataset/raw/<label>/
```

Contoh:

```text
dataset/raw/a/
dataset/raw/b/
dataset/raw/c/
```

### 2. Dataset Gesture Kata dari Webcam

Untuk merekam gesture kata dalam bentuk frame gambar:

```bash
python -m dataset.record_word_gestures
```

Kontrol keyboard:

| Tombol | Fungsi |
| --- | --- |
| `SPACE` | Rekam 1 batch clip untuk label aktif |
| `n` | Label berikutnya |
| `p` | Label sebelumnya |
| `q` | Keluar |

Output:

```text
dataset/raw_words/<LABEL>/clip_0000/frame_000.jpg
dataset/raw_words/<LABEL>/clip_0000/frame_001.jpg
...
```

Di workspace saat ini sudah ada folder contoh seperti:

- `HALO`
- `MAKAN`
- `MINUM`
- `TERIMA_KASIH`
- `TOLONG`

### 3. Dataset Landmark Live dari Webcam

Untuk merekam langsung sequence landmark ke `.npy`:

```bash
python -m dataset.record_landmarks_live
```

Mode ini lebih dekat dengan kondisi inference real-time karena data langsung
diambil dari webcam, diekstrak menjadi landmark, dinormalisasi, lalu disimpan
sebagai sequence.

Kontrol keyboard:

| Tombol | Fungsi |
| --- | --- |
| `SPACE` | Mulai/stop rekam label aktif |
| `n` | Label berikutnya |
| `p` | Label sebelumnya |
| `s` | Simpan dataset |
| `d` | Tampilkan statistik dataset |
| `r` | Reset buffer |
| `q` | Simpan dan keluar |

Output:

```text
dataset/processed/live/X_live.npy
dataset/processed/live/y_live.npy
```

Saat preprocessing dijalankan, dataset live ini akan digabung otomatis jika
file tersebut tersedia.

### 4. Dataset dari Google Drive

Untuk mengunduh ZIP dataset dari Google Drive:

```bash
python -m dataset.download_from_drive ^
  --url "https://drive.google.com/file/d/<FILE_ID>/view" ^
  --out dataset ^
  --expect raw_words
```

Atau secara programatis:

```python
from dataset.download_from_drive import download_and_extract

download_and_extract(
    file_id_or_url="https://drive.google.com/file/d/<FILE_ID>/view",
    extract_to="dataset",
    expected_subdirs=["raw_words"],
)
```

## Preprocessing

Preprocessing mengubah gambar/frame menjadi dataset training dalam format
NumPy.

Jalankan:

```bash
python -m preprocessing.landmark_extractor
```

Tahapan yang dilakukan:

1. Membaca gambar dari `dataset/raw/`.
2. Membaca clip gesture kata dari `dataset/raw_words/`.
3. Mengekstraksi landmark tangan dengan MediaPipe Tasks API.
4. Menormalisasi landmark.
5. Membuat sequence sepanjang 30 frame.
6. Menggabungkan dataset live dari `dataset/processed/live/` jika ada.
7. Menyimpan output ke `dataset/processed/`.

Output:

```text
dataset/processed/X.npy
dataset/processed/y.npy
```

Shape yang diharapkan:

```text
X.npy: (N, 30, 126)
y.npy: (N,)
```

## Normalisasi Landmark

Normalisasi dilakukan di `preprocessing/normalizer.py`.

Untuk setiap tangan:

1. Semua titik dikurangi titik wrist/index `0`.
2. Semua koordinat dibagi jarak wrist ke middle-MCP/index `9`.
3. Jika hanya satu tangan terdeteksi, tangan kedua diisi nol.
4. Jika dua tangan terdeteksi, tangan diurutkan dari kiri ke kanan berdasarkan
   posisi wrist.

Tujuannya agar model tidak terlalu sensitif terhadap:

- posisi tangan di frame,
- ukuran tangan,
- jarak tangan ke kamera,
- variasi pengguna.

## Augmentation

Jalankan:

```bash
python -m augmentation.augment
```

Augmentasi yang dilakukan:

- Gaussian noise pada landmark
- Scaling acak
- Rotasi 3D kecil
- Translasi landmark
- Oversampling kelas minoritas sampai seimbang

Penjelasan tiap augmentasi:

- **Gaussian noise pada landmark**: menambahkan gangguan kecil secara acak ke
  koordinat landmark `(x, y, z)`. Teknik ini meniru jitter atau pergeseran kecil
  dari hasil deteksi MediaPipe, sehingga model tidak terlalu sensitif terhadap
  landmark yang sedikit bergeser.
- **Scaling acak**: memperbesar atau memperkecil susunan landmark secara
  proporsional. Ini meniru kondisi tangan yang terlihat lebih besar saat dekat
  kamera atau lebih kecil saat jauh dari kamera.
- **Rotasi 3D kecil**: memutar landmark sedikit pada sumbu 3D. Teknik ini
  membantu model mengenali gesture walaupun tangan pengguna agak miring,
  berputar, atau tidak selalu tepat menghadap kamera.
- **Translasi landmark**: menggeser seluruh landmark ke arah tertentu tanpa
  mengubah bentuk tangan. Ini membuat model tetap mengenali gesture walaupun
  posisi tangan berada agak ke kiri, kanan, atas, atau bawah frame.

Output:
 
```text
dataset/processed/X_aug.npy
dataset/processed/y_aug.npy
```

Jika `X_aug.npy` tersedia, training akan memakai file tersebut secara default.
Jika tidak tersedia, `training/train.py` akan memuat `X.npy` dan melakukan
augmentasi saat training dimulai.

## Training

Training lokal:

```bash
python -m training.train
```

Contoh dengan parameter:

```bash
python -m training.train --epochs 50 --batch 32
```

Opsi penting:

| Opsi | Fungsi |
| --- | --- |
| `--epochs` | Jumlah epoch training |
| `--batch` | Batch size |
| `--seed` | Random seed |
| `--use_raw` | Paksa memakai `X.npy` dan augmentasi ulang |
| `--no_balance` | Matikan class balancing |
| `--no_plot` | Tidak membuat plot confusion matrix |

Output training:

```text
model/saved/bisindo_new.keras
model/saved/labels2.json
logs/training.csv
logs/history.json
logs/classification_report.txt
logs/confusion_matrix.npy
logs/confusion_matrix.png
```

Model lama `model/saved/bisindo_model.h5` juga ada di folder saved, tetapi
konfigurasi proyek saat ini memakai `bisindo_new.keras` sebagai model utama.

## Arsitektur Model

Model didefinisikan di `model/architecture.py`.

Ringkasan:

```text
Input: (30, 126)
  |
Reshape: (30, 42, 3)
  |
TimeDistributed frame CNN:
  Conv1D(64) + BatchNorm + ReLU
  Conv1D(128) + BatchNorm + ReLU
  GlobalAveragePooling1D
  |
LSTM(128, return_sequences=True)
  |
BatchNormalization
  |
LSTM(64)
  |
BatchNormalization
  |
Dense(128, ReLU)
  |
Dropout(0.4)
  |
Dense(76, softmax)
```

Alasan desain:

- CNN membaca pola spasial antar-landmark pada setiap frame.
- LSTM membaca perubahan temporal antar-frame.
- Dropout dan L2 regularization membantu mengurangi overfitting.
- Output softmax menghasilkan confidence untuk setiap kelas.

## Real-Time Inference Webcam

Jalankan:

```bash
python -m inference.realtime_webcam
```

Kontrol:

| Tombol | Fungsi |
| --- | --- |
| `q` | Keluar |
| `r` | Reset buffer prediksi |

Alur real-time:

1. Webcam dibaca dengan OpenCV.
2. Frame dibalik horizontal agar terasa seperti cermin.
3. MediaPipe mendeteksi landmark tangan.
4. Landmark dinormalisasi dan dimasukkan ke `SequenceBuffer`.
5. Setelah 30 frame terkumpul, model melakukan prediksi.
6. Hasil diperhalus dengan smoothing dan majority vote.
7. Label dan confidence ditampilkan di layar.

## Menjalankan API

Jalankan server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Atau:

```bash
python -m api.main
```

Setelah server berjalan, buka:

```text
http://localhost:8000/docs
```

### Endpoint

| Method | Endpoint | Fungsi |
| --- | --- | --- |
| `GET` | `/` | Info singkat API |
| `GET` | `/health` | Status API, model, jumlah session |
| `GET` | `/labels` | Daftar label |
| `GET` | `/classes` | Alias daftar label |
| `GET` | `/model/classes` | Alias daftar label |
| `POST` | `/predict/landmarks` | Prediksi dari sequence landmark |
| `POST` | `/predict/frame` | Prediksi dari frame base64 dengan session buffer |
| `WS` | `/ws/realtime` | Streaming frame/landmark real-time |

### Contoh Request `/predict/landmarks`

Input harus berbentuk matrix `(30, 126)`. Default-nya `sequence` dianggap
sudah siap-model/ternormalisasi seperti data training. Jika frontend mengirim
landmark mentah MediaPipe, set `"normalized": false`.

```json
{
  "normalized": true,
  "sequence": [
    [0.0, 0.1, -0.02]
  ]
}
```

Contoh di atas hanya ilustrasi. Request sebenarnya harus berisi 30 baris dan
setiap baris berisi 126 angka.

Response:

```json
{
  "label": "word_halo",
  "confidence": 0.87,
  "top_k": [
    {"word_halo": 0.87},
    {"word_kamu": 0.05}
  ],
  "is_stable": true
}
```

### Contoh Request `/predict/frame`

Endpoint ini menerima satu frame gambar base64 per request. Gunakan
`session_id` yang sama agar server dapat mengumpulkan 30 frame.

```json
{
  "session_id": "demo-session-1",
  "image_base64": "data:image/jpeg;base64,...",
  "reset": false
}
```

Prediksi stabil baru muncul setelah buffer session terisi 30 frame.

### Kontrak Input dan Sinkronisasi Frontend

Catatan perbaikan yang sudah diterapkan:

- `/predict/landmarks` tidak lagi menormalisasi ulang input secara default.
  Gunakan `"normalized": true` untuk sequence yang sudah siap-model, atau
  `"normalized": false` untuk landmark mentah MediaPipe.
- Frame tanpa tangan tidak lagi mengisi buffer prediksi API. Jika tidak ada
  tangan, server mengosongkan buffer session dan mengembalikan label `"..."`.
- Preprocessing melewati gambar/frame yang gagal mendeteksi tangan, sehingga
  dataset training tidak tercampur sample all-zero dengan label tertentu.
- Contoh client REST/WebSocket sudah menyertakan flag `normalized`.

Cara sinkronisasi dari frontend:

1. Saat aplikasi dibuka, ambil daftar label dari API:

```js
const labelsRes = await fetch("http://localhost:8000/model/classes");
const { labels, num_classes } = await labelsRes.json();
```

2. Buat `session_id` stabil untuk satu stream kamera. Gunakan ID yang sama
   selama user masih memakai sesi kamera yang sama.

```js
const sessionId = crypto.randomUUID();
```

3. Untuk mode frontend hanya mengirim gambar, kirim frame base64 berurutan ke
   `/predict/frame`. Kirim `reset: true` pada frame pertama atau saat user
   menekan tombol reset/ganti kamera.

```js
await fetch("http://localhost:8000/predict/frame", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    session_id: sessionId,
    image_base64: "data:image/jpeg;base64,...",
    reset: false
  })
});
```

4. Untuk mode frontend sudah menjalankan MediaPipe sendiri, kirim sequence ke
   `/predict/landmarks` dengan `normalized: true` jika landmark sudah
   dinormalisasi seperti pipeline training. Jika masih landmark mentah
   MediaPipe, kirim `normalized: false`.

```js
await fetch("http://localhost:8000/predict/landmarks", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    normalized: true,
    sequence
  })
});
```

5. Untuk WebSocket, kirim salah satu bentuk pesan berikut:

```js
socket.send(JSON.stringify({
  type: "frame",
  image_base64: "data:image/jpeg;base64,..."
}));

socket.send(JSON.stringify({
  type: "landmarks",
  frame,
  normalized: true
}));
```

Frontend sebaiknya menampilkan status menunggu saat `label` bernilai `"..."`.
Itu berarti buffer belum penuh 30 frame valid, confidence belum stabil, atau
tangan sedang tidak terdeteksi.

### Contoh Client

Jalankan server API terlebih dahulu, lalu:

```bash
python api/example_client.py landmarks
python api/example_client.py frame
python api/example_client.py ws
```

Mode:

- `landmarks`: mengirim random landmark untuk menguji kontrak API.
- `frame`: membaca webcam dan mengirim frame base64 via REST.
- `ws`: streaming landmark webcam melalui WebSocket.

## Pipeline Otomatis

Script `run_pipeline.py` dapat menjalankan beberapa tahap sekaligus.

Melihat bantuan:

```bash
python run_pipeline.py
```

Menjalankan download, preprocessing, augmentasi, dan training:

```bash
python run_pipeline.py --all
```

Menjalankan tahap tertentu:

```bash
python run_pipeline.py --download
python run_pipeline.py --preprocess
python run_pipeline.py --augment
python run_pipeline.py --train
```

Tahap interaktif:

```bash
python run_pipeline.py --record
python run_pipeline.py --record-live
```

Catatan: `--all` tidak menjalankan `--record` atau `--record-live` karena dua
tahap tersebut membutuhkan interaksi webcam.

## Urutan Kerja yang Disarankan

Untuk memulai dari nol:

```bash
pip install -r requirements.txt
python -m dataset.download_dataset
python -m dataset.record_word_gestures
python -m preprocessing.landmark_extractor
python -m augmentation.augment
python -m training.train
python -m inference.realtime_webcam
```

Jika hanya ingin memakai model yang sudah ada:

```bash
pip install -r requirements.txt
python -m inference.realtime_webcam
```

Jika ingin menjalankan sebagai service:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## File Penting

| File | Fungsi |
| --- | --- |
| `config.py` | Semua path, label, dan hyperparameter utama |
| `model/architecture.py` | Arsitektur model CNN + LSTM |
| `model/saved/bisindo_new.keras` | Model utama hasil training |
| `model/saved/labels2.json` | Label utama berjumlah 76 kelas |
| `preprocessing/mp_hand_landmarker.py` | Wrapper MediaPipe Tasks API |
| `preprocessing/landmark_extractor.py` | Build dataset `.npy` |
| `inference/predictor.py` | Predictor reusable untuk webcam dan API |
| `api/main.py` | FastAPI server |
| `run_pipeline.py` | Orchestrator pipeline |

## Troubleshooting

### Webcam tidak terbuka

Pastikan:

- webcam tidak sedang dipakai aplikasi lain,
- permission kamera aktif,
- index kamera OpenCV `cv2.VideoCapture(0)` sesuai perangkat.

Jika punya lebih dari satu kamera, ubah index menjadi `1` atau `2` di script
yang memakai webcam.

### Model belum tersedia saat API dijalankan

Jika muncul pesan model belum tersedia di `model/saved/bisindo_new.keras`,
jalankan training dulu:

```bash
python -m training.train
```

Atau pastikan file model hasil training sudah berada di:

```text
model/saved/bisindo_new.keras
```

### Jumlah label tidak cocok dengan output model

`inference/predictor.py` akan memvalidasi jumlah output model terhadap
`labels2.json`. Jika jumlahnya berbeda, gunakan model dan label dari training
run yang sama, atau latih ulang model setelah mengubah daftar kelas di
`config.py`.

### MediaPipe gagal mengunduh `hand_landmarker.task`

File model MediaPipe diunduh otomatis ke:

```text
model/mp_assets/hand_landmarker.task
```

Jika koneksi gagal, unduh manual dari URL yang ada di
`preprocessing/mp_hand_landmarker.py`, lalu letakkan file tersebut di folder
di atas.

### Dataset kosong saat preprocessing

Cek salah satu sumber data berikut sudah tersedia:

```text
dataset/raw/
dataset/raw_words/
dataset/processed/live/X_live.npy
dataset/processed/live/y_live.npy
```

Lalu jalankan ulang:

```bash
python -m preprocessing.landmark_extractor
```

### Training lambat

Training TensorFlow di CPU bisa lambat. Untuk dataset besar, gunakan GPU atau
Google Colab melalui notebook:

```text
training/train_colab.ipynb
```

## Catatan Pengembangan

- Proyek ini memakai MediaPipe Tasks API, bukan API lama
  `mp.solutions.hands`.
- Label disusun dari `ALL_CLASSES` di `config.py`.
- Semua output preprocessing dan training sebaiknya dibuat ulang jika daftar
  kelas berubah.
- Dokumentasi teknis tambahan ada di `docs/ARCHITECTURE.md`.

## Lisensi

Proyek ini dibuat untuk riset dan pembelajaran. Dataset eksternal mengikuti
lisensi masing-masing sumber dataset.
