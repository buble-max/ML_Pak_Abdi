"""
Pipeline preprocessing utama:

1. Membaca gambar statis dari `dataset/raw/<LABEL>/*.{jpg,png}` (alfabet)
   dan klip video dari `dataset/raw_words/<LABEL>/clip_*/frame_*.jpg` (kata).
2. Mengekstraksi 21 landmark tangan (x, y, z) per frame menggunakan
   MediaPipe Hand Landmarker (hingga MAX_HANDS tangan).
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
    FEATURES_PER_FRAME,
    MAX_HANDS,
    MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    PROCESSED_DIR,
    RAW_DIR,
    SEQUENCE_LENGTH,
    WINDOW_STRIDE,
    WORD_CLASSES,
    WORD_RAW_DIR,
)
from preprocessing.normalizer import flatten_frame, normalize_two_hands  # noqa: E402
from utils.labels import build_label_maps, save_labels  # noqa: E402

# MediaPipe diinisialisasi lazy supaya testing tanpa dependency lebih mudah
_mp_hands = None


def _get_hands(static: bool):
    global _mp_hands
    import mediapipe as mp

    _mp_hands = mp.solutions.hands.Hands(
        static_image_mode=static,
        max_num_hands=MP_MAX_NUM_HANDS,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    )
    return _mp_hands


def extract_landmarks_from_image(image_bgr: np.ndarray, hands) -> np.ndarray:
    """Return shape (MAX_HANDS*21*3,) - normalized & flattened."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    hand_arrays: List[np.ndarray] = []
    if res.multi_hand_landmarks:
        for hand_lms in res.multi_hand_landmarks:
            arr = np.array(
                [[p.x, p.y, p.z] for p in hand_lms.landmark],
                dtype=np.float32,
            )  # (21, 3)
            hand_arrays.append(arr)

    normed = normalize_two_hands(hand_arrays, max_hands=MAX_HANDS)  # (MAX_HANDS,21,3)
    return flatten_frame(normed)                                    # (FEATURES_PER_FRAME,)


# ---------------------------------------------------------------
# Gambar statis (alfabet) → sequence via SLIDING WINDOW
# ---------------------------------------------------------------
def static_images_to_sequences(
    image_paths: List[Path],
    hands,
    seq_len: int = SEQUENCE_LENGTH,
    stride: int = WINDOW_STRIDE,
) -> np.ndarray:
    """
    Kumpulkan vektor landmark dari semua gambar statis 1 kelas,
    lalu potong dengan sliding window menjadi banyak sequence (N, T, F).
    Jika total frame < seq_len, gambar akan diulang dan diberi jitter kecil.
    """
    feats: List[np.ndarray] = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        feats.append(extract_landmarks_from_image(img, hands))

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
    for start in range(0, M - seq_len + 1, max(stride, 1)):
        sequences.append(arr[start : start + seq_len])
    if not sequences:
        sequences.append(arr[:seq_len])
    return np.stack(sequences, axis=0).astype(np.float32)  # (N, T, F)


# ---------------------------------------------------------------
# Klip video (gesture kata) → 1 sequence per klip
# ---------------------------------------------------------------
def clip_to_sequence(
    clip_dir: Path, hands, seq_len: int = SEQUENCE_LENGTH
) -> Optional[np.ndarray]:
    frames = sorted(clip_dir.glob("frame_*.jpg"))
    if not frames:
        return None

    feats = []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        feats.append(extract_landmarks_from_image(img, hands))
    if not feats:
        return None

    arr = np.stack(feats, axis=0)  # (M, F)
    # Padding / truncating ke seq_len
    M = arr.shape[0]
    if M < seq_len:
        pad = np.repeat(arr[-1:], seq_len - M, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    else:
        # downsample uniform
        idx = np.linspace(0, M - 1, seq_len).astype(int)
        arr = arr[idx]
    return arr.astype(np.float32)  # (T, F)


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

    # ---- ALFABET (gambar statis) ----
    hands_static = _get_hands(static=True)
    for label in ALPHABET_CLASSES:
        class_dir = raw_dir / label
        if not class_dir.exists():
            print(f"[skip] {label}: folder tidak ditemukan")
            continue

        imgs = sorted(
            [p for p in class_dir.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        if not imgs:
            print(f"[skip] {label}: kosong")
            continue

        seqs = static_images_to_sequences(imgs, hands_static)
        X.append(seqs)
        y.extend([label_to_idx[label]] * seqs.shape[0])
        print(f"  {label:>3}: {len(imgs):4d} img → {seqs.shape[0]:4d} seq")

    # ---- KATA (klip video) ----
    hands_video = _get_hands(static=False)
    for label in WORD_CLASSES:
        class_dir = word_dir / label
        if not class_dir.exists():
            print(f"[skip] {label}: folder tidak ditemukan")
            continue

        clips = sorted([d for d in class_dir.iterdir() if d.is_dir()])
        class_seqs = []
        for clip in tqdm(clips, desc=f"  {label}", leave=False):
            seq = clip_to_sequence(clip, hands_video)
            if seq is not None:
                class_seqs.append(seq)
        if class_seqs:
            seqs = np.stack(class_seqs, axis=0)  # (N, T, F)
            X.append(seqs)
            y.extend([label_to_idx[label]] * seqs.shape[0])
            print(f"  {label:>12}: {len(clips):4d} clip → {seqs.shape[0]:4d} seq")

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
