# Arsitektur Sistem BISINDO

Dokumentasi teknis lengkap pipeline, format data, dan keputusan desain.

## 1. Alur Data End-to-End

```
Dataset Mentah (gambar / klip video)
        │
        ▼
MediaPipe Hand Landmarker  ──►  21 titik (x,y,z) per tangan
        │
        ▼
Normalisasi  (wrist-centered + scale(wrist ↔ middle_MCP))
        │
        ▼
Padding hingga 2 tangan  ──►  (42, 3) per frame  ──►  flatten 126-dim
        │
        ▼
┌── Static (alfabet) ──► Sliding window stride 5, T=30 ──┐
│                                                        │
└── Klip video (kata)  ─► Uniform resample ke T=30 ──────┘
                                                         ▼
                               X.npy (N, 30, 126) + y.npy (N,)
                                                         │
                                                         ▼
                      Augment × AUG_MULTIPLIER  +  Class balancing
                                                         │
                                                         ▼
                           X_aug.npy, y_aug.npy
                                                         │
                                                         ▼
                        Train/Val/Test stratified 70/15/15
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

## 2. Format Landmark per Frame

```
frame_vector (126,) =
  [hand_0_landmark_0_x,   hand_0_landmark_0_y,   hand_0_landmark_0_z,
   hand_0_landmark_1_x,   hand_0_landmark_1_y,   hand_0_landmark_1_z,
   ...
   hand_0_landmark_20_x,  hand_0_landmark_20_y,  hand_0_landmark_20_z,
   hand_1_landmark_0_x,   ..., hand_1_landmark_20_z]
```

Tangan diurutkan dari kiri ke kanan berdasarkan x-wrist. Jika hanya satu
tangan terdeteksi, slot tangan kedua diisi nol (padding) dan dilewati dari
augmentasi agar tetap nol.

## 3. Normalisasi

Invarian terhadap:
- **Posisi tangan**: semua titik dikurangi `landmark[0]` (wrist).
- **Ukuran tangan / jarak kamera**: dibagi `||landmark[9]||` (jarak
  wrist ↔ middle MCP) pasca-translasi. Jika < 1e-6, diganti 1.
- **Perbedaan antar pengguna**: kombinasi dua normalisasi di atas sudah
  menyerap mayoritas variasi ukuran/posisi.

## 4. Sliding Window (static images)

Dataset BISINDO sumber berupa gambar statis. Kita:
1. Ekstrak vektor landmark untuk setiap gambar.
2. Susun vektor-vektor tersebut berurutan (per kelas) menjadi array (M, 126).
3. Jika `M < T`, replikasi + tambahkan jitter halus (σ=1e-3) agar tetap
   bervariasi.
4. Ambil sliding window `arr[s : s+T]` dengan stride `WINDOW_STRIDE`.

Hasil: satu kelas alfabet menghasilkan banyak sequence "pseudo-temporal"
yang dapat dipelajari LSTM.

## 5. Augmentation

Satu set parameter acak **per sequence** (konsisten antar frame) supaya
konsistensi temporal tetap terjaga:

| Teknik      | Range                   | Tujuan                          |
| ----------- | ----------------------- | ------------------------------- |
| Noise       | N(0, 0.01)              | Menyimulasikan jitter MediaPipe |
| Scaling     | Uniform [0.9, 1.1]      | Ukuran tangan bervariasi        |
| Rotasi 3D   | ±15° (z), ±7.5° (x,y)   | Rotasi di bidang kamera + yaw   |
| Translasi   | ±0.05                   | Posisi tangan di frame          |

Kelas minoritas di-oversample sampai setara kelas terbesar (augment baru
pada tiap sampel). Padding tangan kosong tidak diaugment.

## 6. Arsitektur Model

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
└─ Dense(31, softmax)
```

Parameter ~200K, kompatibel dengan Colab GPU T4/A100 & inference real-time CPU.

## 7. Anti-Flicker Real-Time

```
         probs = model.predict(sequence)[0]
         prob_ema = α * probs + (1-α) * prob_ema         # smoothing
         top_idx  = argmax(prob_ema)
         pred_buf.append(top_idx)
         vote, cnt = most_common(pred_buf)

         if cnt >= MIN_VOTES and prob_ema[top_idx] ≥ THRESHOLD:
             emit label  else emit "..."
```

Parameter default:
- `BUFFER_SIZE = 10`
- `SMOOTHING_MIN_VOTES = 6`
- `CONFIDENCE_THRESHOLD = 0.70`
- `EMA α = 0.6`

## 8. API Contract

### REST

**POST /predict/landmarks**
```json
{ "sequence": [[...126 floats...], ... (30 baris) ...] }
```
→
```json
{ "label": "A", "confidence": 0.87, "is_stable": true,
  "top_k": [{"label":"A","confidence":0.87}, ...] }
```

**POST /predict/frame**
```json
{ "session_id": "uuid", "image_base64": "data:image/jpeg;base64,...",
  "reset": false }
```

### WebSocket `/ws/realtime`
- Client → `{"type":"frame","image_base64":"..."}`
- Client → `{"type":"landmarks","frame":[... 126 floats ...]}`
- Client → `{"type":"reset"}`
- Server → `{"type":"prediction", "label":..., "confidence":..., "top_k":[...]}`

## 9. Keputusan Desain
- **Landmark bukan pixel**: jauh lebih robust terhadap latar, pencahayaan.
- **2 tangan**: BISINDO banyak gesture dua-tangan, berbeda dari SIBI.
- **CNN sebelum LSTM**: Conv1D melihat tetangga landmark (struktur tangan),
  LSTM melihat temporal → pemisahan concerns yang bersih.
- **EMA + vote buffer**: EMA menghaluskan probabilitas kontinu, vote buffer
  memberi keputusan diskrit yang stabil → mengatasi flicker.
- **SequenceBuffer per API session**: memungkinkan frontend mengirim frame
  satu-per-satu tanpa harus mengelola buffer di sisi klien.
