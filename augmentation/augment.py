"""
Data augmentation untuk sequence landmark BISINDO.

Input shape : (N, T, F) dengan F = MAX_HANDS * 21 * 3
Output shape: (N', T, F) setelah augment + balancing.

Teknik:
- Gaussian noise pada koordinat.
- Scaling seragam (zoom tangan).
- Rotasi 3D (sumbu z ≈ rotasi di bidang kamera), opsional yaw kecil di sumbu y.
- Translasi landmark (shift posisi tangan).
- Class balancing: oversample kelas minoritas sampai seimbang.

Catatan: sample dengan tangan yang hilang (padding nol) TIDAK diaugment
supaya padding tetap nol.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    AUG_MULTIPLIER,
    AUG_NOISE_STD,
    AUG_ROTATION_DEG,
    AUG_SCALE_RANGE,
    AUG_TRANSLATION,
    FEATURES_PER_FRAME,
    MAX_HANDS,
    NUM_LANDMARKS,
    PROCESSED_DIR,
    RANDOM_SEED,
)


def _reshape_frames(seq: np.ndarray) -> np.ndarray:
    """(T, F) -> (T, MAX_HANDS, 21, 3)"""
    if seq.ndim != 2 or seq.shape[1] != FEATURES_PER_FRAME:
        raise ValueError(
            f"Expected sequence shape (T, {FEATURES_PER_FRAME}), got {seq.shape}"
        )
    return seq.reshape(seq.shape[0], MAX_HANDS, NUM_LANDMARKS, 3)


def _flatten_frames(arr: np.ndarray) -> np.ndarray:
    """(T, MAX_HANDS, 21, 3) -> (T, F)"""
    return arr.reshape(arr.shape[0], -1)


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def augment_sequence(
    seq: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = AUG_NOISE_STD,
    scale_range: Tuple[float, float] = AUG_SCALE_RANGE,
    rotation_deg: float = AUG_ROTATION_DEG,
    translation: float = AUG_TRANSLATION,
) -> np.ndarray:
    """Augment satu sequence (T, F). Parameter diambil sekali per sequence
    (konsisten antar-frame)."""
    frames = _reshape_frames(seq.copy())  # (T, H, 21, 3)

    # parameter tetap untuk seluruh sequence
    scale = rng.uniform(*scale_range)
    rz = np.deg2rad(rng.uniform(-rotation_deg, rotation_deg))
    ry = np.deg2rad(rng.uniform(-rotation_deg * 0.5, rotation_deg * 0.5))
    rx = np.deg2rad(rng.uniform(-rotation_deg * 0.5, rotation_deg * 0.5))
    R = _rotation_matrix(rx, ry, rz)
    tvec = rng.uniform(-translation, translation, size=(3,)).astype(np.float32)

    # Mask tangan valid (bukan padding nol). Shape (H,).
    hand_valid = np.abs(frames).sum(axis=(0, 2, 3)) > 0

    for h in range(MAX_HANDS):
        if not hand_valid[h]:
            continue  # biarkan padding nol
        hand = frames[:, h, :, :]  # (T, 21, 3)
        noise = rng.normal(0, noise_std, size=hand.shape).astype(np.float32)
        frames[:, h, :, :] = (hand @ R.T * scale) + tvec + noise

    return _flatten_frames(frames).astype(np.float32)


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    multiplier: int = AUG_MULTIPLIER,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Setiap sample asli ditambah `multiplier` sample hasil augment.
    Output mencakup sample asli + hasil augment."""
    if multiplier < 0:
        raise ValueError("multiplier harus >= 0")
    rng = np.random.default_rng(seed)
    aug_X = [X]
    aug_y = [y]
    for _ in range(multiplier):
        new_X = np.stack([augment_sequence(s, rng) for s in X], axis=0)
        aug_X.append(new_X)
        aug_y.append(y.copy())
    return (
        np.concatenate(aug_X, axis=0).astype(np.float32),
        np.concatenate(aug_y, axis=0).astype(np.int64),
    )


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oversample kelas minoritas sampai setara dengan kelas mayoritas
    (random sampling dengan augmentation sekaligus)."""
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(y, return_counts=True)
    target = int(counts.max())

    X_out = [X]
    y_out = [y]
    for cls, cnt in zip(unique, counts):
        deficit = target - int(cnt)
        if deficit <= 0:
            continue
        idx_cls = np.where(y == cls)[0]
        pick = rng.choice(idx_cls, size=deficit, replace=True)
        aug_X = np.stack([augment_sequence(X[i], rng) for i in pick], axis=0)
        aug_y = np.full(deficit, cls, dtype=y.dtype)
        X_out.append(aug_X)
        y_out.append(aug_y)
    return (
        np.concatenate(X_out, axis=0).astype(np.float32),
        np.concatenate(y_out, axis=0).astype(np.int64),
    )


def main() -> None:
    X_path = PROCESSED_DIR / "X.npy"
    y_path = PROCESSED_DIR / "y.npy"
    if not (X_path.exists() and y_path.exists()):
        print(f"[error] {X_path} atau {y_path} tidak ditemukan. "
              f"Jalankan preprocessing.landmark_extractor terlebih dahulu.")
        return

    X = np.load(X_path)
    y = np.load(y_path)
    print(f"[info] Loaded X {X.shape}, y {y.shape}")

    assert X.shape[-1] == FEATURES_PER_FRAME, \
        f"Expected last dim {FEATURES_PER_FRAME}, got {X.shape[-1]}"

    X_aug, y_aug = augment_dataset(X, y)
    print(f"[info] Augmented → X {X_aug.shape}, y {y_aug.shape}")

    X_bal, y_bal = balance_classes(X_aug, y_aug)
    print(f"[info] Balanced  → X {X_bal.shape}, y {y_bal.shape}")

    np.save(PROCESSED_DIR / "X_aug.npy", X_bal)
    np.save(PROCESSED_DIR / "y_aug.npy", y_bal)
    print(f"[done] Disimpan ke {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
