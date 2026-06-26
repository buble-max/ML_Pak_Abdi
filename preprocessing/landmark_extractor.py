"""
Pipeline preprocessing utama.

1. Membaca gambar statis dari `dataset/raw/<LABEL>/*.{jpg,png}` (alfabet)
   dan klip video dari `dataset/raw_words/<LABEL>/clip_*/frame_*.jpg` (kata).
2. Mengekstraksi 21 landmark tangan (x, y, z) per frame menggunakan
   MediaPipe **Tasks API** (`HandLandmarker` + `hand_landmarker.task`),
   menggantikan API lama `mp.solutions.hands.Hands` yang sudah dihapus
   pada MediaPipe terbaru (Python 3.12).
3. Menormalisasi landmark (lihat normalizer.normalize_two_hands).
4. Mengubah gambar statis menjadi sequence temporal via SLIDING WINDOW
   (duplikasi frame + jitter kecil) atau langsung dari klip video.
5. Menyimpan hasil sebagai:
       dataset/processed/X.npy  shape (N, T, FEATURES_PER_FRAME)
       dataset/processed/y.npy  shape (N,)  int label
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
    ALPHABET_CLASSES,
    DIGIT_CLASSES,
    FEATURES_PER_FRAME,
    MAX_HANDS,
    PROCESSED_DIR,
    RAW_DIR,
    SEQUENCE_LENGTH,
    WINDOW_STRIDE,
    WORD_CLASSES,
    WORD_RAW_DIR,
)
from preprocessing.mp_hand_landmarker import HandLandmarkerWrapper  # noqa: E402
from preprocessing.normalizer import flatten_frame, normalize_two_hands  # noqa: E402
from utils.labels import build_label_maps, save_labels  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _label_dir(root: Path, label: str) -> Path:
    candidates = [
        label,
        label.upper(),
        label.lower(),
    ]
    if label.startswith("word_"):
        word = label.removeprefix("word_")
        candidates.extend([word, word.upper(), word.lower()])

    for candidate in dict.fromkeys(candidates):
        path = root / candidate
        if path.exists():
            return path
    return root / label


def _get_landmarker(running_mode: str = "image") -> HandLandmarkerWrapper:
    """Factory HandLandmarker (MediaPipe Tasks API)."""
    return HandLandmarkerWrapper(running_mode=running_mode)


def extract_landmarks_from_image(
    image_bgr: np.ndarray,
    landmarker: HandLandmarkerWrapper,
) -> Optional[np.ndarray]:
    """
    Ekstraksi 21 landmark per tangan (x, y, z) menggunakan MediaPipe
    Tasks API, lalu normalisasi dan padding ke MAX_HANDS.

    Return shape: (FEATURES_PER_FRAME,) = (MAX_HANDS * 21 * 3,).
    """
    hand_arrays: List[np.ndarray] = landmarker.detect_bgr(image_bgr)
    if not hand_arrays:
        return None
    normed = normalize_two_hands(hand_arrays, max_hands=MAX_HANDS)   # (MAX_HANDS,21,3)
    return flatten_frame(normed)                                     # (FEATURES_PER_FRAME,)


def _extract_features_from_paths(
    image_paths: List[Path],
    landmarker: HandLandmarkerWrapper,
) -> Optional[np.ndarray]:
    feats: List[np.ndarray] = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        feat = extract_landmarks_from_image(img, landmarker)
        if feat is None:
            continue
        feats.append(feat)

    if not feats:
        return None
    return np.stack(feats, axis=0).astype(np.float32)


def _pad_or_sample_sequence(frames: np.ndarray, seq_len: int) -> np.ndarray:
    """Ubah frame landmark (M, F) menjadi sequence fixed (T, F)."""
    frame_count = frames.shape[0]
    if frame_count < seq_len:
        pad = np.repeat(frames[-1:], seq_len - frame_count, axis=0)
        return np.concatenate([frames, pad], axis=0).astype(np.float32)

    idx = np.linspace(0, frame_count - 1, seq_len).astype(int)
    return frames[idx].astype(np.float32)


# ---------------------------------------------------------------
# Gambar statis (alfabet) → sequence via SLIDING WINDOW
# ---------------------------------------------------------------
def static_images_to_sequences(
    image_paths: List[Path],
    landmarker: HandLandmarkerWrapper,
    seq_len: int = SEQUENCE_LENGTH,
    stride: int = WINDOW_STRIDE,
) -> np.ndarray:
    """
    Kumpulkan vektor landmark dari semua gambar statis 1 kelas,
    lalu potong dengan sliding window menjadi banyak sequence (N, T, F).
    Jika total frame < seq_len, gambar akan diulang dan diberi jitter kecil.
    """
    arr = _extract_features_from_paths(image_paths, landmarker)
    if arr is None:
        return np.zeros((0, seq_len, FEATURES_PER_FRAME), dtype=np.float32)

    # Jika terlalu pendek → repeat + jitter halus
    if arr.shape[0] < seq_len:
        rng = np.random.default_rng(42)
        reps = int(np.ceil(seq_len * 2 / arr.shape[0]))
        arr = np.tile(arr, (reps, 1))
        arr = arr + rng.normal(0, 1e-3, size=arr.shape).astype(np.float32)

    sequences = []
    for start in range(0, arr.shape[0] - seq_len + 1, max(stride, 1)):
        sequences.append(arr[start : start + seq_len])
    if not sequences:
        sequences.append(arr[:seq_len])
    return np.stack(sequences, axis=0).astype(np.float32)  # (N, T, F)


# ---------------------------------------------------------------
# Klip video (gesture kata) → 1 sequence per klip
# ---------------------------------------------------------------
def clip_to_sequence(
    clip_dir: Path,
    landmarker: HandLandmarkerWrapper,
    seq_len: int = SEQUENCE_LENGTH,
) -> Optional[np.ndarray]:
    frames = sorted(clip_dir.glob("frame_*.jpg"))
    if not frames:
        return None

    arr = _extract_features_from_paths(frames, landmarker)
    if arr is None:
        return None

    return _pad_or_sample_sequence(arr, seq_len)


# ---------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------
def build_dataset(
    raw_dir: Path = RAW_DIR,
    word_dir: Path = WORD_RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    label_to_idx, _ = build_label_maps()
    save_labels()

    X: List[np.ndarray] = []
    y: List[int] = []

    # ---- ALFABET + DIGIT (gambar statis) ----
    #  Tasks API mode IMAGE: cocok untuk preprocessing batch.
    landmarker_static = _get_landmarker(running_mode="image")
    try:
        for label in ALPHABET_CLASSES + DIGIT_CLASSES:
            class_dir = _label_dir(raw_dir, label)
            if not class_dir.exists():
                print(f"[skip] {label}: folder tidak ditemukan")
                continue

            imgs = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not imgs:
                print(f"[skip] {label}: kosong")
                continue

            seqs = static_images_to_sequences(imgs, landmarker_static)
            X.append(seqs)
            y.extend([label_to_idx[label]] * seqs.shape[0])
            print(f"  {label:>3}: {len(imgs):4d} img → {seqs.shape[0]:4d} seq")

        # ---- KATA (klip video) ----
        #  Kita tetap gunakan mode IMAGE karena frame dari klip diproses
        #  satu per satu sebagai still image, bukan stream.
        for label in WORD_CLASSES:
            class_dir = _label_dir(word_dir, label)
            if not class_dir.exists():
                print(f"[skip] {label}: folder tidak ditemukan")
                continue

            clips = sorted([d for d in class_dir.iterdir() if d.is_dir()])
            class_seqs = []
            for clip in tqdm(clips, desc=f"  {label}", leave=False):
                seq = clip_to_sequence(clip, landmarker_static)
                if seq is not None:
                    class_seqs.append(seq)
            if class_seqs:
                seqs = np.stack(class_seqs, axis=0)  # (N, T, F)
                X.append(seqs)
                y.extend([label_to_idx[label]] * seqs.shape[0])
                print(f"  {label:>12}: {len(clips):4d} clip → {seqs.shape[0]:4d} seq")
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
            print(f"  {'LIVE':>12}: {X_live.shape[0]:4d} seq (dari webcam recorder)")
        else:
            print(f"[warn] X_live shape {X_live.shape} tidak kompatibel, dilewati.")

    if not X:
        print("[error] Tidak ada data terbangun. Cek dataset/raw, dataset/raw_words,")
        print("        atau jalankan: python -m dataset.record_landmarks_live")
        return

    X_arr = np.concatenate(X, axis=0).astype(np.float32)
    y_arr = np.array(y, dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X_arr)
    np.save(out_dir / "y.npy", y_arr)
    print(f"\n[done] X: {X_arr.shape}  y: {y_arr.shape}")
    print(f"[done] Disimpan ke {out_dir}")


if __name__ == "__main__":
    build_dataset()
