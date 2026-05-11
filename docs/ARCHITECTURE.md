# Arsitektur Sistem BISINDO

Dokumentasi teknis lengkap pipeline, format data, keputusan desain, dan
kriteria evaluasi sistem.

---

## 1. Alur Data End-to-End

```
Dataset Mentah (gambar / klip video / landmark live .npy)
        │
        ▼
MediaPipe Tasks API (HandLandmarker)  ──►  21 titik (x,y,z) per tangan
        │
        ▼
Normalisasi  (wrist-centered + scale(wrist ↔ middle_MCP))
        │
        ▼
Padding hingga 2 tangan  ──►  (42, 3) per frame  ──►  flatten 126-dim
        │
        ▼
┌── Static (alfabet+angka) ──► Sliding window (configurable) ──┐
│                                                              │
├── Klip video (kata+angka) ─► Temporal sampling (uniform/rng) ┤
│                                                              │
└── Landmark live .npy ──────► SequenceBuffer webcam ──────────┘
                                                         ▼
                               X.npy (N, T, 126) + y.npy (N,)
                                                         │
                                                         ▼
                Augment (spatial + temporal jitter) + Class balancing
                                                         │
                                                         ▼
                        Train/Val/Test stratified (configurable ratio)
                                                         │
                                                         ▼
                    CNN+LSTM (TimeDistributed → LSTM → Dense)
                                                         │
                                                         ▼
                                bisindo_model.h5
                                                         │
                                                         ▼
                   Real-time inference (webcam) / API (FastAPI)
```

---

## 2. Kelas Gesture yang Didukung

| Kategori   | Kelas                          | Jumlah |
| ---------- | ------------------------------ | ------ |
| Alfabet    | A – Z                          | 26     |
| Angka      | 0 – 9                          | 10     |
| Kata       | HALO, MAKAN, MINUM, TERIMA_KASIH, TOLONG | 5 |
| **Total**  |                                | **41** |

Sistem mendukung **auto-detect label**: cukup tambah folder baru di
`dataset/raw/<LABEL>/` atau `dataset/raw_words/<LABEL>/` dan label baru
otomatis terdeteksi tanpa edit `config.py` (selama `AUTO_DETECT_LABELS=True`).

---

## 3. Format Landmark per Frame

```
frame_vector (126,) =
  [hand_0_lm_0_x, hand_0_lm_0_y, hand_0_lm_0_z,
   hand_0_lm_1_x, ..., hand_0_lm_20_z,
   hand_1_lm_0_x, ..., hand_1_lm_20_z]
```

Tangan diurutkan kiri→kanan berdasarkan x-wrist. Slot kosong diisi nol.

---

## 4. Normalisasi & Ekstraksi Landmark

### MediaPipe Tasks API

API lama `mp.solutions.hands` dihapus di MediaPipe terbaru (Python 3.12).
Sistem menggunakan `HandLandmarker` dari `mediapipe.tasks.python.vision`:

| Konteks              | Running Mode | Method                          |
| -------------------- | ------------ | ------------------------------- |
| Batch preprocessing  | IMAGE        | `landmarker.detect(mp_img)`     |
| Webcam streaming     | VIDEO        | `landmarker.detect_for_video()` |

Model `hand_landmarker.task` auto-download ke `model/mp_assets/`.

### Normalisasi Koordinat

1. **Translasi**: origin di wrist (landmark[0]).
2. **Skala**: dibagi `||landmark[9]||` (jarak wrist ↔ middle MCP).

---

## 5. Konfigurasi Sequence Training

Semua parameter berada di `config.py` dan dapat diubah fleksibel:

| Parameter            | Default   | Deskripsi                                     |
| -------------------- | --------- | --------------------------------------------- |
| `SEQUENCE_LENGTH`    | 30        | Jumlah frame per sample (T)                   |
| `WINDOW_STRIDE`      | 5         | Hop antar sliding window                      |
| `SEQUENCE_OVERLAP`   | 0.83      | Overlap ratio (menggantikan WINDOW_STRIDE jika > 0) |
| `FRAME_STRIDE`       | 1         | Interval pengambilan frame (1=semua, 2=skip 1, dst) |
| `TEMPORAL_SAMPLING`  | "uniform" | Metode sampling klip video ("uniform" / "random") |
| `VAL_SPLIT`          | 0.15      | Rasio validasi                                |
| `TEST_SPLIT`         | 0.15      | Rasio test                                    |

**Contoh konfigurasi rasio**:
- 70/15/15: `VAL_SPLIT=0.15, TEST_SPLIT=0.15` (default)
- 80/10/10: `VAL_SPLIT=0.10, TEST_SPLIT=0.10`
- 30/70 (no test): `VAL_SPLIT=0.70, TEST_SPLIT=0.0`

---

## 6. Sliding Window & Dynamic Sequence Management

### Gambar Statis (alfabet + angka)
1. Ekstrak landmark → array (M, 126).
2. Jika `M < T`: repeat + jitter (σ=1e-3).
3. Sliding window `arr[s : s+T]` dengan stride dari `SEQUENCE_OVERLAP` atau
   `WINDOW_STRIDE`.

### Klip Video (kata + angka dinamis)
1. Frame di-skip setiap `FRAME_STRIDE` frame.
2. Jika total frame > T:
   - `TEMPORAL_SAMPLING="uniform"`: linspace sampling.
   - `TEMPORAL_SAMPLING="random"`: random N frame (sorted).
3. Jika total < T: padding repeat frame terakhir.

### Live Webcam
1. `SequenceBuffer` (deque maxlen=T) mengumpulkan frame real-time.
2. Saat penuh → simpan / predict; buffer direset otomatis.

---

## 7. Augmentasi Temporal & Spasial

| Teknik            | Parameter            | Deskripsi                                  |
| ----------------- | -------------------- | ------------------------------------------ |
| Temporal jitter   | `AUG_TEMPORAL_JITTER=2` | Shift frame ±N posisi (clamp [0,T-1])  |
| Gaussian noise    | `AUG_NOISE_STD=0.01`   | Noise pada koordinat landmark            |
| Scaling           | `AUG_SCALE_RANGE=(0.9,1.1)` | Zoom seragam                       |
| Rotasi 3D         | `AUG_ROTATION_DEG=15` | ±15° (z), ±7.5° (x,y)                    |
| Translasi         | `AUG_TRANSLATION=0.05` | Shift posisi tangan                      |
| Class balancing   | automatic             | Oversample kelas minoritas via augment   |

Temporal jitter diterapkan **sebelum** spatial augmentation untuk
mensimulasikan variasi kecepatan gesture antar pengguna.

---

## 8. Arsitektur Model CNN + LSTM

```
Input (B, 30, 126)
└─ Reshape (B, 30, 42, 3)
└─ TimeDistributed(
      Conv1D(64, 3, same) → BN → ReLU
      Conv1D(128, 3, same) → BN → ReLU
      GlobalAveragePooling1D
   )                         # (B, 30, 128) - fitur spasial per frame
└─ LSTM(128, return_sequences=True, dropout=0.2)
└─ BatchNormalization
└─ LSTM(64, dropout=0.2)
└─ BatchNormalization
└─ Dense(128, ReLU, l2=1e-4)
└─ Dropout(0.4)
└─ Dense(NUM_CLASSES, softmax)
```

- **TimeDistributed CNN**: ekstraksi fitur spasial (hubungan antar landmark
  dalam satu frame).
- **LSTM layers**: pemodelan temporal (hubungan antar frame dalam sequence).
- **Dropout + BN**: regularisasi, mencegah overfitting.
- Output dinamis berdasarkan `NUM_CLASSES` (auto-computed dari label).

---

## 9. Training & Callbacks

| Callback              | Parameter              | Fungsi                          |
| --------------------- | ---------------------- | ------------------------------- |
| EarlyStopping         | patience=15            | Stop jika val_acc tidak naik    |
| ReduceLROnPlateau     | patience=5, factor=0.5 | Kurangi LR saat val_loss stagnan |
| ModelCheckpoint       | save_best_only=True    | Simpan model terbaik ke .h5     |
| CSVLogger             |                        | Log metrics per epoch           |

---

## 10. Anti-Flicker Real-Time

```
probs = model.predict(sequence)[0]
prob_ema = α * probs + (1-α) * prob_ema         # EMA smoothing
top_idx  = argmax(prob_ema)
pred_buf.append(top_idx)                         # buffer deque
vote, cnt = most_common(pred_buf)

if cnt >= MIN_VOTES and prob_ema[top_idx] ≥ THRESHOLD:
    emit label
else:
    emit "..."
```

Default: `BUFFER_SIZE=10`, `SMOOTHING_MIN_VOTES=6`, `CONFIDENCE_THRESHOLD=0.70`, `EMA α=0.6`.

---

## 11. API Contract (FastAPI)

### REST Endpoints

| Method | Endpoint             | Input                          | Output           |
| ------ | -------------------- | ------------------------------ | ---------------- |
| GET    | `/health`            | -                              | status + model info |
| GET    | `/labels`            | -                              | list of classes  |
| POST   | `/predict/landmarks` | sequence (T×F floats)          | PredictionResponse |
| POST   | `/predict/frame`     | session_id + base64 image      | PredictionResponse |

### WebSocket `/ws/realtime`
- Client → `{"type":"frame","image_base64":"..."}`
- Client → `{"type":"landmarks","frame":[126 floats]}`
- Client → `{"type":"reset"}`
- Server → `{"type":"prediction","label":...,"confidence":...,"top_k":[...]}`

---

## 12. Kriteria Evaluasi Sistem

Evaluasi menyeluruh mencakup:

### A. Pipeline Dataset
- [x] Multi-source: gambar statis, klip video, webcam live recording
- [x] Auto-detect label dari folder tanpa mapping manual
- [x] Import dataset dari Google Drive via gdown (tanpa upload manual)
- [x] Rekam multi-sample langsung ke .npy (append mode)

### B. Preprocessing & Temporal Processing
- [x] MediaPipe Tasks API (`HandLandmarker`) stabil di Python 3.12
- [x] 21 landmark (x,y,z) per tangan, 2 tangan terdeteksi
- [x] Normalisasi wrist-centered + scale invariant
- [x] Sliding window sequence (T × 21 × 3) configurable
- [x] FRAME_STRIDE untuk temporal downsampling
- [x] TEMPORAL_SAMPLING: uniform atau random
- [x] SEQUENCE_OVERLAP sebagai alternatif WINDOW_STRIDE

### C. Augmentasi
- [x] Spatial: noise, scaling, rotasi 3D, translasi
- [x] Temporal: jitter ±N frame
- [x] Class balancing otomatis (oversample minoritas)

### D. Model & Training
- [x] CNN + LSTM dengan TimeDistributed, BN, Dropout
- [x] Multi-kategori: alfabet + angka + kata (41 kelas)
- [x] Configurable train/val/test split ratio
- [x] EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
- [x] Training GPU via Google Colab notebook
- [x] Export .h5 + labels.json

### E. Real-Time Inference
- [x] Deteksi gesture langsung dari webcam
- [x] Anti-flicker: prediction buffer + EMA + confidence threshold
- [x] FPS overlay + landmark visualization
- [x] Reset buffer via keyboard

### F. API Deployment
- [x] FastAPI: /health, /labels, /predict/landmarks, /predict/frame
- [x] WebSocket /ws/realtime streaming
- [x] CORS enabled, session-based buffer management
- [x] Kompatibel untuk integrasi web & mobile

### G. Full AI Pipeline Architecture
- [x] Dataset generation (multi-source + auto-label)
- [x] Video-based temporal preprocessing (configurable)
- [x] Configurable sequence training parameters
- [x] Deep Learning training (GPU accelerated)
- [x] Real-time webcam inference (anti-flicker)
- [x] API deployment berbasis AI service

---

## 13. Keputusan Desain

- **Landmark bukan pixel**: robust terhadap latar, pencahayaan.
- **2 tangan**: BISINDO banyak gesture dua-tangan (berbeda dari SIBI).
- **CNN sebelum LSTM**: Conv1D melihat tetangga landmark (struktur spasial),
  LSTM melihat temporal → separation of concerns.
- **Temporal jitter**: mensimulasikan variasi kecepatan gesture.
- **EMA + vote buffer**: EMA menghaluskan probabilitas kontinu, vote buffer
  memberi keputusan diskrit stabil.
- **Auto-detect labels**: scalable — tambah kelas cukup buat folder baru.
- **Configurable split/overlap/stride**: memudahkan eksperimen tanpa edit code.
