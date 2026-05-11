"""
Pipeline preprocessing utama.

1. Membaca gambar statis dari `dataset/raw/<LABEL>/*.{jpg,png}` (alfabet + angka)
   dan klip video dari `dataset/raw_words/<LABEL>/clip_*/frame_*.jpg` (kata)
   serta `dataset/raw_numbers/<LABEL>/clip_*/frame_*.jpg` (angka video).
2. Mengekstraksi 21 landmark tangan (x, y, z) per frame menggunakan
   MediaPipe **Tasks API** (`HandLandmarker` + `hand_landmarker.task`).
3. Menormalisasi landmark (lihat normalizer.normalize_two_hands).
4. Mengubah gambar statis menjadi sequence temporal via SLIDING WINDOW
   (duplikasi frame + jitter kecil) atau langsung dari klip video.
5. Menyimpan hasil sebagai:
       dataset/processed/X.npy  shape (N, T, FEATURES_PER_FRAME)
       dataset/processed/y.npy  shape (N,)  int label

Fitur:
- **Auto-detect labels**: jika `AUTO_DETECT_LABELS=True` di config,
  label baru ditemukan otomatis dari nama folder tanpa perlu edit config.
- **Configurable sequence**: SEQUENCE_LENGTH, WINDOW_STRIDE/SEQUENCE_OVERLAP,
  FRAME_STRIDE, TEMPORAL_SAMPLING (uniform/random).
- **Multi-source**: gambar statis, klip video, dan live .npy digabung.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    FEATURES_PER_FRAME,
    FRAME_STRIDE,
    MAX_HANDS,
    NUMBER_RAW_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SEQUENCE_LENGTH,
    SEQUENCE_OVERLAP,
    TEMPORAL_SAMPLING,
    WINDOW_STRIDE,
    WORD_RAW_DIR,
)
from preprocessing.mp_hand_landmarker import HandLandmarkerWrapper  # noqa: E402
from preprocessing.normalizer import flatten_frame, normalize_two_hands  # noqa: E402
from utils.labels import build_label_maps, save_labels  # noqa: E402


def _get_landmarker(running_mode: str = "image") -> HandLandmarkerWrapper:
    """Factory HandLandmarker (MediaPipe Tasks API)."""
    return HandLandmarkerWrapper(running_mode=running_mode)


def _compute_stride(seq_len: int, overlap: float, fallback_stride: int) -> int:
    """Hitung stride dari overlap ratio atau gunakan fallback."""
    if overlap > 0:
        stride = max(1, int(round(seq_len * (1.0 - overlap))))
    else:
        stride = max(1, fallback_stride)
    return stride


def extract_landmarks_from_image(
    image_bgr: np.ndarray,
    landmarker: HandLandmarkerWrapper,
) -> np.ndarray:
    """
    Ekstraksi 21 landmark per tangan → normalisasi → padding MAX_HANDS.
    Return shape: (FEATURES_PER_FRAME,).
    """
    hand_arrays: List[np.ndarray] = landmarker.detect_bgr(image_bgr)
    normed = normalize_two_hands(hand_arrays, max_hands=MAX_HANDS)
    return flatten_frame(normed)


# ---------------------------------------------------------------
# Gambar statis → sequence via SLIDING WINDOW
# ---------------------------------------------------------------
def static_images_to_sequences(
    image_paths: List[Path],
    landmarker: HandLandmarkerWrapper,
    seq_len: int = SEQUENCE_LENGTH,
    stride: Optional[int] = None,
    frame_stride: int = FRAME_STRIDE,
) -> np.ndarray:
    """
    Kumpulkan vektor landmark dari gambar statis 1 kelas,
    potong dengan sliding window menjadi banyak sequence (N, T, F).
    """
    if stride is None:
        stride = _compute_stride(seq_len, SEQUENCE_OVERLAP, WINDOW_STRIDE)

    # Apply frame_stride: skip setiap ke-N gambar
    if frame_stride > 1:
        image_paths = image_paths[::frame_stride]

    feats: List[np.ndarray] = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        feats.append(extract_landmarks_from_image(img, landmarker))

    if not feats:
        return np.zeros((0, seq_len, FEATURES_PER_FRAME), dtype=np.float32)

    arr = np.stack(feats, axis=0)  # (M, F)
    M = arr.shape[0]

    # Jika terlalu pendek → repeat + jitter halus
    if M < seq_len:
        rng = np.random.default_rng(42)
        reps = int(np.ceil(seq_len * 2 / M))
        arr = np.tile(arr, (reps, 1))
        arr = arr + rng.normal(0, 1e-3, size=arr.shape).astype(np.float32)
        M = arr.shape[0]

    sequences = []
    for start in range(0, M - seq_len + 1, stride):
        sequences.append(arr[start : start + seq_len])
    if not sequences:
        sequences.append(arr[:seq_len])
    return np.stack(sequences, axis=0).astype(np.float32)


# ---------------------------------------------------------------
# Klip video → 1 sequence per klip
# ---------------------------------------------------------------
def clip_to_sequence(
    clip_dir: Path,
    landmarker: HandLandmarkerWrapper,
    seq_len: int = SEQUENCE_LENGTH,
    frame_stride: int = FRAME_STRIDE,
    temporal_sampling: str = TEMPORAL_SAMPLING,
) -> Optional[np.ndarray]:
    """
    Klip video → (T, F) sequence.
    Mendukung FRAME_STRIDE dan TEMPORAL_SAMPLING (uniform/random).
    """
    frames = sorted(clip_dir.glob("frame_*.jpg"))
    if not frames:
        return None

    # Apply frame_stride
    if frame_stride > 1:
        frames = frames[::frame_stride]

    feats = []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        feats.append(extract_landmarks_from_image(img, landmarker))
    if not feats:
        return None

    arr = np.stack(feats, axis=0)  # (M, F)
    M = arr.shape[0]

    if M < seq_len:
        pad = np.repeat(arr[-1:], seq_len - M, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif M > seq_len:
        if temporal_sampling == "random":
            rng = np.random.default_rng()
            idx = sorted(rng.choice(M, size=seq_len, replace=False))
        else:  # uniform
            idx = np.linspace(0, M - 1, seq_len).astype(int)
        arr = arr[idx]

    return arr[:seq_len].astype(np.float32)


# ---------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------
def build_dataset(
    raw_dir: Path = RAW_DIR,
    word_dir: Path = WORD_RAW_DIR,
    number_dir: Path = NUMBER_RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    """
    Pipeline utama: scan semua folder dataset → preprocessing → .npy.

    Auto-detect labels: label baru dari folder otomatis terdeteksi
    tanpa perlu mengedit config.py (selama AUTO_DETECT_LABELS=True).
    """
    # Build labels (auto-detect dari folder jika diaktifkan)
    label_to_idx, _ = build_label_maps(auto_detect=True)
    all_classes = list(label_to_idx.keys())
    save_labels(classes=all_classes)
    print(f"[info] Total kelas terdeteksi: {len(all_classes)}")

    X: List[np.ndarray] = []
    y: List[int] = []

    stride = _compute_stride(SEQUENCE_LENGTH, SEQUENCE_OVERLAP, WINDOW_STRIDE)

    # ---- SEMUA FOLDER GAMBAR STATIS (alfabet + angka + auto-detected) ----
    landmarker_static = _get_landmarker(running_mode="image")
    try:
        # Scan raw_dir: setiap subfolder = 1 label
        if raw_dir.exists():
            for class_dir in sorted(raw_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                label = class_dir.name.strip().upper()
                if label not in label_to_idx:
                    continue  # skip folder yang bukan kelas valid

                imgs = sorted(
                    [p for p in class_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
                )
                if not imgs:
                    continue

                seqs = static_images_to_sequences(
                    imgs, landmarker_static,
                    seq_len=SEQUENCE_LENGTH, stride=stride,
                    frame_stride=FRAME_STRIDE,
                )
                if seqs.shape[0] > 0:
                    X.append(seqs)
                    y.extend([label_to_idx[label]] * seqs.shape[0])
                    print(f"  {label:>14}: {len(imgs):4d} img → {seqs.shape[0]:4d} seq")

        # ---- KLIP VIDEO: kata + angka + auto-detected ----
        for video_dir in (word_dir, number_dir):
            if not video_dir.exists():
                continue
            for class_dir in sorted(video_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                label = class_dir.name.strip().upper()
                if label not in label_to_idx:
                    continue

                clips = sorted([d for d in class_dir.iterdir() if d.is_dir()])
                class_seqs = []
                for clip in tqdm(clips, desc=f"  {label}", leave=False):
                    seq = clip_to_sequence(
                        clip, landmarker_static,
                        seq_len=SEQUENCE_LENGTH,
                        frame_stride=FRAME_STRIDE,
                        temporal_sampling=TEMPORAL_SAMPLING,
                    )
                    if seq is not None:
                        class_seqs.append(seq)
                if class_seqs:
                    seqs = np.stack(class_seqs, axis=0)
                    X.append(seqs)
                    y.extend([label_to_idx[label]] * seqs.shape[0])
                    print(f"  {label:>14}: {len(clips):4d} clip → {seqs.shape[0]:4d} seq")
    finally:
        landmarker_static.close()

    # ---- LIVE (dataset manual dari webcam → .npy) ----
    live_dir = out_dir / "live"
    x_live_path = live_dir / "X_live.npy"
    y_live_path = live_dir / "y_live.npy"
    if x_live_path.exists() and y_live_path.exists():
        X_live = np.load(x_live_path)
        y_live = np.load(y_live_path)
        if X_live.shape[1:] == (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
            X.append(X_live)
            y.extend(y_live.tolist())
            print(f"  {'LIVE':>14}: {X_live.shape[0]:4d} seq (webcam recorder)")
        else:
            print(f"[warn] X_live shape {X_live.shape} tidak kompatibel, dilewati.")

    if not X:
        print("[error] Tidak ada data terbangun. Cek dataset/raw, dataset/raw_words,")
        print("        dataset/raw_numbers, atau: python -m dataset.record_landmarks_live")
        return

    X_arr = np.concatenate(X, axis=0).astype(np.float32)
    y_arr = np.array(y, dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X_arr)
    np.save(out_dir / "y.npy", y_arr)
    print(f"\n[done] X: {X_arr.shape}  y: {y_arr.shape}")
    print(f"[done] Disimpan ke {out_dir}")
    print(f"[done] Config: T={SEQUENCE_LENGTH}, stride={stride}, "
          f"frame_stride={FRAME_STRIDE}, sampling={TEMPORAL_SAMPLING}")


if __name__ == "__main__":
    build_dataset()
