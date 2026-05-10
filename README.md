# Sistem Deteksi BISINDO Real-Time (ML + Computer Vision)

Sistem Artificial Intelligence berbasis Machine Learning dan Computer Vision yang mampu
mendeteksi bahasa isyarat **BISINDO (Bahasa Isyarat Indonesia)** secara real-time
menggunakan webcam. Sistem mengenali gesture tangan berdasarkan huruf alfabet BISINDO
(A–Z) serta 5 gesture kata tambahan: **Halo, Makan, Minum, Terima Kasih, Tolong**.

## Arsitektur Sistem

```
Webcam ─► OpenCV ─► MediaPipe Hand Landmarker ─► Normalisasi Landmark
                                                        │
                                                        ▼
                              Sliding Window Sequence (T x 21 x 3)
                                                        │
                                                        ▼
                              CNN (TimeDistributed) + LSTM + Dense
                                                        │
                                                        ▼
                    Prediction Buffer + Smoothing + Confidence Threshold
                                                        │
                                                        ▼
                                Output Gesture (Alfabet / Kata)
```

## Struktur Folder

| Folder            | Fungsi                                                           |
| ----------------- | ---------------------------------------------------------------- |
| `dataset/`        | Script pengambilan dataset BISINDO & perekaman gesture kata      |
| `preprocessing/`  | Ekstraksi 21 landmark MediaPipe, normalisasi, sliding window     |
| `augmentation/`   | Noise, scaling, rotasi, variasi posisi, class balancing          |
| `model/`          | Arsitektur CNN + LSTM (TimeDistributed, Dropout, BatchNorm)      |
| `training/`       | Training script + notebook Google Colab (GPU)                    |
| `inference/`      | Real-time webcam inference dengan anti-flicker                   |
| `api/`            | FastAPI server (endpoint predict, health, landmarks)             |
| `utils/`          | Label mapping & helper functions                                 |

## Pipeline

### 1. Dataset
Tiga sumber dataset yang didukung sistem:
1. **Gambar statis BISINDO** (alfabet A–Z) — diunduh otomatis dari repository
   [Indonesian Sign Language BISINDO Hand Sign Detection Dataset](https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset)
   via `dataset/download_dataset.py`. Dataset tidak tersedia sebagai `.zip`
   terpisah sehingga script akan melakukan `git clone` repo penuh lalu
   mengambil folder `collectedimages/` sebagai sumber utama.
2. **Klip video gesture kata** — direkam mandiri dari webcam (frame gambar)
   via `dataset/record_word_gestures.py`.
3. **Landmark langsung dari webcam** — dataset manual yang ditangkap real-time
   dari kamera laptop, diproses MediaPipe + normalisasi + sequence buffer,
   lalu disimpan langsung sebagai `.npy` via
   `dataset/record_landmarks_live.py`. Pendekatan ini menghasilkan dataset
   yang paling realistis karena kondisi pencahayaan, sudut kamera, dan
   karakteristik tangan pengguna persis sama dengan saat inference.

**Import dataset dari Google Drive (Colab).**
Di Google Colab, dataset tambahan (misal klip gesture kata) tidak perlu
di-upload manual lewat `files.upload()`. Notebook
`training/train_colab.ipynb` melakukan `drive.mount('/content/drive')`
lalu memanggil helper `dataset/download_from_drive.py` yang:
- Mengunduh ZIP dari link Google Drive via `gdown` (file ID auto-terdeteksi
  dari URL `drive.google.com/file/d/<FILE_ID>/view`).
- Membuat folder dataset kalau belum ada.
- Mengekstrak ZIP ke direktori target.
- Menghapus ZIP sementara setelah ekstraksi.
- Memvalidasi subfolder wajib (mis. `raw_words/`) sudah terbentuk.

Pemakaian programatis:
```python
from dataset.download_from_drive import download_and_extract
download_and_extract(
    file_id_or_url="https://drive.google.com/file/d/<FILE_ID>/view",
    extract_to="dataset",
    expected_subdirs=["raw_words"],
)
```
Atau CLI:
```bash
python -m dataset.download_from_drive \
    --url "https://drive.google.com/file/d/<FILE_ID>/view" \
    --out dataset --expect raw_words
```

### 2. Preprocessing
- Ekstraksi **21 landmark tangan** (x, y, z) menggunakan **MediaPipe Tasks API**
  (`HandLandmarker` + `hand_landmarker.task`). API lama `mp.solutions.hands`
  sudah dihapus pada MediaPipe terbaru (Python 3.12); pipeline kami
  menggunakan `HandLandmarker.detect()` (mode IMAGE) dan
  `HandLandmarker.detect_for_video()` (mode VIDEO) — lihat
  `preprocessing/mp_hand_landmarker.py`.
- Model asset `hand_landmarker.task` diunduh otomatis ke `model/mp_assets/`
  saat pertama kali dibutuhkan.
- **Normalisasi** terhadap wrist (titik 0) dan jarak wrist↔middle-MCP untuk
  menghilangkan pengaruh posisi, ukuran tangan, dan jarak kamera.
- Konversi gambar statis → sequence temporal dengan **sliding window** (default T=30).
- Output disimpan dalam `.npy` (`X.npy`, `y.npy`) agar training efisien.

### 3. Augmentation
- Gaussian noise pada koordinat landmark.
- Scaling acak (0.9–1.1).
- Rotasi landmark pada sumbu kamera (±15°).
- Translasi posisi tangan.
- **Class balancing** via oversampling untuk kelas minoritas.

### 4. Model
Kombinasi **CNN + LSTM**:
- `TimeDistributed(Conv1D + BN + Pool)` per frame landmark.
- `LSTM(128) → LSTM(64)` untuk learning temporal.
- `Dropout` + `BatchNormalization` untuk regularisasi.
- `Dense(softmax)` output 31 kelas (26 alfabet + 5 kata).

### 5. Training
- Framework: TensorFlow / Keras.
- Environment: Google Colab GPU (lihat `training/train_colab.ipynb`).
- Callback: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`.
- Output: `model/bisindo_model.h5`.

### 6. Real-Time Inference
- OpenCV capture webcam → MediaPipe → landmark.
- **Prediction buffer** (deque) + majority vote + confidence threshold (default 0.7)
  untuk mencegah flicker.
- Overlay label gesture + confidence pada frame.

### 7. API
FastAPI server dengan endpoint:
| Method | Endpoint         | Deskripsi                                            |
| ------ | ---------------- | ---------------------------------------------------- |
| GET    | `/health`        | Status API & model                                   |
| POST   | `/predict/landmarks` | Input: array landmark (T x 21 x 3) → JSON hasil  |
| POST   | `/predict/frame`     | Input: base64 image → JSON hasil                  |
| WS     | `/ws/realtime`   | Streaming prediksi real-time                         |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset BISINDO
python dataset/download_dataset.py

# 3. Rekam gesture kata tambahan (opsional, jika dataset belum ada)
python dataset/record_word_gestures.py

# 3b. ATAU buat dataset mandiri langsung dari webcam (landmark → .npy)
#     Mendukung semua kelas (alfabet + kata), append mode.
python -m dataset.record_landmarks_live

# 4. Preprocessing (ekstraksi landmark + sliding window → .npy)
#    Akan otomatis menggabungkan data live dari langkah 3b.
python -m preprocessing.landmark_extractor

# 5. Training (lokal / Colab)
python -m training.train
#   atau buka training/train_colab.ipynb di Google Colab (GPU)

# 6. Real-time webcam inference
python -m inference.realtime_webcam

# 7. Jalankan API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Konfigurasi
Semua hyperparameter & path terpusat di `config.py`.

## Lisensi
Proyek riset / pembelajaran. Dataset BISINDO mengikuti lisensi masing-masing sumber.
