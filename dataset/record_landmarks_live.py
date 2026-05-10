"""
Pembuatan dataset BISINDO secara manual langsung dari webcam laptop.

Berbeda dari record_word_gestures.py yang menyimpan gambar frame mentah,
script ini:
  1. Menangkap frame real-time via OpenCV.
  2. Mengekstraksi 21 landmark tangan menggunakan MediaPipe Hand Landmarker.
  3. Menormalisasi landmark (wrist-centered + scale).
  4. Memasukkan landmark ke dalam SequenceBuffer secara otomatis.
  5. Saat buffer penuh (T frame), menyimpan sequence (T, F) beserta label
     ke dalam file .npy yang siap training.

Output disimpan di:
    dataset/processed/live/
        X_live.npy  shape (N, SEQUENCE_LENGTH, FEATURES_PER_FRAME)
        y_live.npy  shape (N,)

File output bersifat APPEND: setiap kali script dijalankan, sample baru
ditambahkan ke dataset yang sudah ada.

Mendukung SEMUA kelas (alfabet A-Z + 5 kata), sehingga pengguna dapat
membuat dataset lengkap secara mandiri hanya dari kamera laptop.

Kontrol Keyboard:
    SPACE = mulai/stop perekaman sequence untuk label aktif
    n     = label berikutnya
    p     = label sebelumnya
    s     = simpan dataset ke .npy
    d     = tampilkan statistik dataset saat ini
    r     = reset buffer (ulang sample terakhir)
    q     = simpan & keluar

Workflow:
    1. Jalankan:  python -m dataset.record_landmarks_live
    2. Tekan n/p untuk memilih label.
    3. Posisikan tangan di depan kamera.
    4. Tekan SPACE untuk mulai merekam; buffer terisi otomatis.
    5. Saat T frame terkumpul, sequence tersimpan dan buffer direset
       sehingga sample berikutnya bisa langsung direkam (multi-sample).
    6. Pindah label, ulangi. Tekan 's' atau 'q' untuk menyimpan ke .npy.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    ALL_CLASSES,
    FEATURES_PER_FRAME,
    MAX_HANDS,
    MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    PROCESSED_DIR,
    SEQUENCE_LENGTH,
)
from preprocessing.normalizer import flatten_frame, normalize_two_hands  # noqa: E402
from preprocessing.sequence_builder import SequenceBuffer  # noqa: E402
from utils.labels import build_label_maps  # noqa: E402

# Output directory
LIVE_DIR = PROCESSED_DIR / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

X_LIVE_PATH = LIVE_DIR / "X_live.npy"
Y_LIVE_PATH = LIVE_DIR / "y_live.npy"

LABEL_TO_IDX, IDX_TO_LABEL = build_label_maps()
NUM_LABELS = len(ALL_CLASSES)


def _load_existing() -> Tuple[List[np.ndarray], List[int]]:
    """Muat dataset live yang sudah ada (append mode)."""
    sequences: List[np.ndarray] = []
    labels: List[int] = []
    if X_LIVE_PATH.exists() and Y_LIVE_PATH.exists():
        X = np.load(X_LIVE_PATH)
        y = np.load(Y_LIVE_PATH)
        sequences = list(X)
        labels = list(y)
        print(f"[info] Dataset live existing dimuat: {len(sequences)} sample")
    return sequences, labels


def _save_dataset(sequences: List[np.ndarray], labels: List[int]) -> None:
    """Simpan dataset live ke .npy."""
    if not sequences:
        print("[warn] Tidak ada data untuk disimpan.")
        return
    X = np.stack(sequences, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    np.save(X_LIVE_PATH, X)
    np.save(Y_LIVE_PATH, y)
    print(f"[saved] X_live: {X.shape}, y_live: {y.shape} -> {LIVE_DIR}")


def _print_stats(labels: List[int]) -> None:
    """Tampilkan statistik jumlah sample per kelas."""
    counts: Dict[int, int] = defaultdict(int)
    for lbl in labels:
        counts[int(lbl)] += 1
    print("\n" + "=" * 50)
    print("  STATISTIK DATASET LIVE")
    print("=" * 50)
    for idx in sorted(counts.keys()):
        name = IDX_TO_LABEL[idx]
        print(f"    {name:>14} : {counts[idx]:4d} sample")
    print(f"{'TOTAL':>18} : {len(labels):4d} sample")
    print("=" * 50 + "\n")


def _extract_landmarks(
    rgb_frame: np.ndarray, hands
) -> Tuple[Optional[np.ndarray], object]:
    """
    Return (feature_vector, mp_result).
    feature_vector: (FEATURES_PER_FRAME,) atau None jika tangan tidak terdeteksi.
    mp_result     : hasil MediaPipe untuk kebutuhan drawing (menghindari
                    proses dua kali pada frame yang sama).
    """
    res = hands.process(rgb_frame)
    hand_arrays = []
    if res.multi_hand_landmarks:
        for hand_lms in res.multi_hand_landmarks:
            arr = np.array(
                [[p.x, p.y, p.z] for p in hand_lms.landmark],
                dtype=np.float32,
            )
            hand_arrays.append(arr)

    if not hand_arrays:
        return None, res

    normed = normalize_two_hands(hand_arrays, max_hands=MAX_HANDS)
    return flatten_frame(normed), res


def _draw_ui(
    frame: np.ndarray,
    label: str,
    label_idx: int,
    is_recording: bool,
    buf_fill: int,
    buf_max: int,
    session_count: int,
    total_count: int,
    hand_detected: bool,
    fps: float,
) -> None:
    """Gambar overlay UI pada frame."""
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)

    color_label = (0, 255, 0) if not is_recording else (0, 0, 255)
    cv2.putText(
        frame,
        f"Label: [{label_idx+1}/{NUM_LABELS}] {label}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_label, 2,
    )

    status = "REC" if is_recording else "IDLE"
    status_color = (0, 0, 255) if is_recording else (200, 200, 200)
    cv2.putText(
        frame, status, (w - 100, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2,
    )

    # Buffer progress bar
    bar_x, bar_y, bar_w, bar_h = 15, 45, w - 30, 15
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), -1)
    fill_w = int(bar_w * buf_fill / max(buf_max, 1))
    bar_color = (0, 200, 255) if is_recording else (100, 100, 100)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      bar_color, -1)
    cv2.putText(
        frame, f"Buffer: {buf_fill}/{buf_max}",
        (bar_x + 5, bar_y + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
    )

    # Info bawah
    hand_str = "HAND OK" if hand_detected else "NO HAND"
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(
        frame,
        f"Session: {session_count}  Total: {total_count}  {fps:.0f}FPS",
        (15, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )
    cv2.putText(
        frame, hand_str, (w - 130, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1,
    )

    # Bottom help
    cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)
    cv2.putText(
        frame,
        "SPACE=rec  n/p=label  s=save  d=stats  r=reset  q=quit",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
    )


def main() -> None:
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=MP_MAX_NUM_HANDS,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    )
    mp_draw = mp.solutions.drawing_utils
    mp_connections = mp.solutions.hands.HAND_CONNECTIONS

    sequences, labels = _load_existing()
    total_count = len(sequences)

    label_idx = 0
    is_recording = False
    buf = SequenceBuffer()
    session_count = 0
    t_prev = time.time()
    fps = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Webcam tidak dapat dibuka. Pastikan kamera tersedia.")
        return

    print("\n" + "=" * 60)
    print("  BISINDO LIVE DATASET RECORDER")
    print("  Rekam landmark langsung dari webcam -> .npy")
    print("=" * 60)
    print(f"  Output          : {LIVE_DIR}")
    print(f"  Sequence length : {SEQUENCE_LENGTH} frame")
    print(f"  Features/frame  : {FEATURES_PER_FRAME}")
    print(f"  Total kelas     : {NUM_LABELS}")
    print(f"  Data existing   : {total_count} sample")
    print("=" * 60)
    print("\nKontrol:")
    print("  SPACE = mulai/stop perekaman")
    print("  n/p   = next/prev label")
    print("  s     = simpan dataset ke .npy")
    print("  d     = statistik dataset")
    print("  r     = reset buffer")
    print("  q     = simpan & keluar\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[error] Gagal membaca frame dari webcam")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            feat, mp_res = _extract_landmarks(rgb, hands)
            hand_detected = feat is not None

            if is_recording and hand_detected:
                buf.push(feat)
                if buf.is_ready():
                    sequence = buf.get()  # (T, F)
                    current_label = LABEL_TO_IDX[ALL_CLASSES[label_idx]]
                    sequences.append(sequence)
                    labels.append(current_label)
                    session_count += 1
                    total_count += 1
                    buf.reset()
                    print(
                        f"  [+] Sample #{total_count} tersimpan: "
                        f"{ALL_CLASSES[label_idx]} (session: {session_count})"
                    )

            # Draw landmarks
            if mp_res.multi_hand_landmarks:
                for hand_lms in mp_res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_connections)

            # FPS (EMA)
            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.8 * fps + 0.2 * (1.0 / dt)

            _draw_ui(
                frame,
                label=ALL_CLASSES[label_idx],
                label_idx=label_idx,
                is_recording=is_recording,
                buf_fill=len(buf._buf),
                buf_max=buf.seq_len,
                session_count=session_count,
                total_count=total_count,
                hand_detected=hand_detected,
                fps=fps,
            )

            cv2.imshow("BISINDO Live Recorder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                _save_dataset(sequences, labels)
                break
            elif key == ord(" "):
                is_recording = not is_recording
                if is_recording:
                    buf.reset()
                    print(f"[rec] Mulai merekam: {ALL_CLASSES[label_idx]}")
                else:
                    print("[stop] Berhenti merekam. Buffer direset.")
                    buf.reset()
            elif key == ord("n"):
                label_idx = (label_idx + 1) % NUM_LABELS
                buf.reset()
                is_recording = False
                print(f"[label] -> {ALL_CLASSES[label_idx]}")
            elif key == ord("p"):
                label_idx = (label_idx - 1) % NUM_LABELS
                buf.reset()
                is_recording = False
                print(f"[label] -> {ALL_CLASSES[label_idx]}")
            elif key == ord("s"):
                _save_dataset(sequences, labels)
            elif key == ord("d"):
                _print_stats(labels)
            elif key == ord("r"):
                buf.reset()
                print("[reset] Buffer direset.")

    except KeyboardInterrupt:
        print("\n[interrupt] Menyimpan data...")
        _save_dataset(sequences, labels)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("[done] Recorder selesai.")


if __name__ == "__main__":
    main()
