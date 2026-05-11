"""
Augmentasi temporal khusus video sequence BISINDO.

Menambahkan variasi temporal pada sequence `(T, F)` agar model lebih
generalis terhadap kecepatan dan dinamika gesture:

- speed variation      : ubah kecepatan via resampling (faktor s in [smin, smax])
- temporal jitter      : shift indeks setiap frame +-N posisi
- random frame drop    : hilangkan sebagian frame lalu resample ke T
- motion noise         : random walk halus antar frame (integrated noise)
- sequence reversing   : balik urutan frame (opsional, per-kelas)
- temporal shift       : geser seluruh sequence +-N frame (circular)

Seluruh fungsi tetap mempertahankan shape (T, F) dan tidak mengubah
padding nol (tangan hilang) kecuali untuk motion noise yang hanya
diterapkan pada tangan valid.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    AUG_FRAME_DROP_PROB,
    AUG_MOTION_NOISE_STD,
    AUG_REVERSE_PROB,
    AUG_SPEED_RANGE,
    AUG_TEMPORAL_JITTER,
    AUG_TEMPORAL_SHIFT,
    MAX_HANDS,
    NUM_LANDMARKS,
)
from preprocessing.temporal_resampler import interpolate_to_length


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _hand_valid_mask(seq: np.ndarray) -> np.ndarray:
    """Return bool array (MAX_HANDS,): True jika tangan tsb punya data non-nol."""
    frames = seq.reshape(seq.shape[0], MAX_HANDS, NUM_LANDMARKS, 3)
    return np.array(
        [np.abs(frames[:, h, :, :]).sum() > 0 for h in range(MAX_HANDS)]
    )


# ---------------------------------------------------------------
# Temporal augmentations
# ---------------------------------------------------------------
def speed_variation(
    seq: np.ndarray,
    rng: np.random.Generator,
    speed_range: Tuple[float, float] = AUG_SPEED_RANGE,
) -> np.ndarray:
    """
    Simulasikan variasi kecepatan gesture dengan meng-resample sequence:
    speed > 1 -> lebih cepat (frame dipadatkan), speed < 1 -> lebih lambat.
    Panjang output tetap T (re-interpolate kembali).
    """
    T = seq.shape[0]
    s = float(rng.uniform(*speed_range))
    # Jumlah frame "sampled" dengan kecepatan s
    M = max(2, int(round(T / s)))
    idx = np.linspace(0, T - 1, M).astype(np.int64)
    compressed = seq[idx]
    return interpolate_to_length(compressed, T)


def temporal_jitter(
    seq: np.ndarray,
    rng: np.random.Generator,
    max_shift: int = AUG_TEMPORAL_JITTER,
) -> np.ndarray:
    """Geser setiap frame +-max_shift posisi (clamp). Simulasi variasi kecepatan mikro."""
    if max_shift <= 0:
        return seq
    T = seq.shape[0]
    offsets = rng.integers(-max_shift, max_shift + 1, size=T)
    new_idx = np.clip(np.arange(T) + offsets, 0, T - 1)
    return seq[new_idx]


def random_frame_drop(
    seq: np.ndarray,
    rng: np.random.Generator,
    drop_prob: float = AUG_FRAME_DROP_PROB,
    max_drop_ratio: float = 0.10,
) -> np.ndarray:
    """
    Hilangkan sebagian frame secara acak lalu resample kembali ke T.
    Simulasi frame skip / kehilangan frame pada kamera real.
    """
    T = seq.shape[0]
    if drop_prob <= 0:
        return seq
    mask = rng.random(T) > drop_prob
    # Jangan drop terlalu banyak frame
    if mask.sum() < T * (1 - max_drop_ratio):
        keep_n = int(T * (1 - max_drop_ratio))
        keep_idx = np.sort(rng.choice(T, size=keep_n, replace=False))
        mask = np.zeros(T, dtype=bool)
        mask[keep_idx] = True
    if mask.sum() < 2:
        return seq
    kept = seq[mask]
    return interpolate_to_length(kept, T)


def motion_noise(
    seq: np.ndarray,
    rng: np.random.Generator,
    std: float = AUG_MOTION_NOISE_STD,
) -> np.ndarray:
    """
    Tambahkan smooth drift (random walk terintegrasi) ke tangan valid.
    Berbeda dari Gaussian noise per-frame: drift ini terkorelasi antar
    frame sehingga menyerupai guncangan kamera atau goyangan tangan.
    """
    if std <= 0:
        return seq
    T, F = seq.shape
    steps = rng.normal(0, std, size=(T, F)).astype(np.float32)
    drift = np.cumsum(steps, axis=0)
    # Kurangi mean supaya drift tidak bias ke satu arah
    drift -= drift.mean(axis=0, keepdims=True)

    # Terapkan hanya pada tangan valid (agar padding tetap nol)
    hand_valid = _hand_valid_mask(seq)
    frames = seq.reshape(T, MAX_HANDS, NUM_LANDMARKS, 3).copy()
    drift_frames = drift.reshape(T, MAX_HANDS, NUM_LANDMARKS, 3)
    for h in range(MAX_HANDS):
        if hand_valid[h]:
            frames[:, h, :, :] = frames[:, h, :, :] + drift_frames[:, h, :, :]
    return frames.reshape(T, F).astype(np.float32)


def sequence_reverse(
    seq: np.ndarray,
    rng: np.random.Generator,
    prob: float = AUG_REVERSE_PROB,
) -> np.ndarray:
    """Balik urutan frame dengan probabilitas `prob`. Default 0 (disabled)."""
    if prob <= 0:
        return seq
    if rng.random() < prob:
        return seq[::-1].copy()
    return seq


def temporal_shift(
    seq: np.ndarray,
    rng: np.random.Generator,
    max_shift: int = AUG_TEMPORAL_SHIFT,
) -> np.ndarray:
    """Circular-shift seluruh sequence +-max_shift frame."""
    if max_shift <= 0:
        return seq
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return seq
    return np.roll(seq, shift, axis=0)


# ---------------------------------------------------------------
# Composed temporal augment
# ---------------------------------------------------------------
def temporal_augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Komposisi augmentasi temporal sesuai konfigurasi."""
    out = seq
    out = speed_variation(out, rng)
    out = temporal_jitter(out, rng)
    out = random_frame_drop(out, rng)
    out = temporal_shift(out, rng)
    out = sequence_reverse(out, rng)
    out = motion_noise(out, rng)
    return out.astype(np.float32)
