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
Dataset utama berasal dari repository
[Indonesian Sign Language BISINDO Hand Sign Detection Dataset](https://github.com/).
Gambar statis alfabet A–Z ditambah 5 gesture kata yang direkam mandiri
(`dataset/record_word_gestures.py`).

### 2. Preprocessing
- Ekstraksi **21 landmark tangan** (x, y, z) menggunakan MediaPipe Hand Landmarker.
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

# 4. Preprocessing (ekstraksi landmark + sliding window → .npy)
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
