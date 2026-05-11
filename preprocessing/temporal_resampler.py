"""
Temporal sequence resampler.

Mengubah sequence landmark dengan panjang `M` menjadi panjang tetap `T`
(=SEQUENCE_LENGTH) menggunakan salah satu strategi:

- `interpolate` : linear interpolation antar frame (default untuk M != T).
- `pad_last`    : repeat frame terakhir (hanya jika M < T).
- `pad_zero`    : zero-padding (hanya jika M < T; cocok untuk gesture diam).
- `trim_center` : ambil T frame tengah (hanya jika M > T).
- `trim_start`  : ambil T frame pertama.
- `trim_end`    : ambil T frame terakhir.

Fungsi-fungsi ini bersifat pure (tanpa side-effect) dan menerima array
shape `(M, F)` lalu mengembalikan `(T, F)`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    FEATURES_PER_FRAME,
    SEQUENCE_LENGTH,
    TEMPORAL_PADDING,
)

PadMode = Literal["pad_last", "pad_zero"]
TrimMode = Literal["trim_center", "trim_start", "trim_end"]


# ---------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------
def interpolate_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Linear interpolation di dimensi waktu untuk menghasilkan `target_len`
    frame. Input shape (M, F), output (target_len, F).
    """
    M = arr.shape[0]
    if M == target_len:
        return arr.astype(np.float32, copy=False)
    if M <= 1:
        return np.tile(arr, (target_len, 1)).astype(np.float32)

    src = np.linspace(0.0, 1.0, M, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, target_len, dtype=np.float64)

    out = np.empty((target_len, arr.shape[1]), dtype=np.float32)
    for f in range(arr.shape[1]):
        out[:, f] = np.interp(dst, src, arr[:, f])
    return out


# ---------------------------------------------------------------
# Padding
# ---------------------------------------------------------------
def pad_sequence(arr: np.ndarray, target_len: int, mode: PadMode = "pad_last") -> np.ndarray:
    """Padding sequence ke `target_len` (hanya jika M < target_len)."""
    M = arr.shape[0]
    if M >= target_len:
        return arr[:target_len].astype(np.float32, copy=False)

    pad_n = target_len - M
    if mode == "pad_zero":
        pad = np.zeros((pad_n, arr.shape[1]), dtype=np.float32)
    else:  # pad_last
        if M == 0:
            pad = np.zeros((pad_n, arr.shape[1]), dtype=np.float32)
        else:
            pad = np.repeat(arr[-1:], pad_n, axis=0)
    return np.concatenate([arr, pad], axis=0).astype(np.float32)


# ---------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------
def trim_sequence(arr: np.ndarray, target_len: int, mode: TrimMode = "trim_center") -> np.ndarray:
    """Trim sequence ke `target_len` (hanya jika M > target_len)."""
    M = arr.shape[0]
    if M <= target_len:
        return arr.astype(np.float32, copy=False)

    if mode == "trim_start":
        return arr[:target_len].astype(np.float32)
    if mode == "trim_end":
        return arr[-target_len:].astype(np.float32)
    # trim_center
    start = (M - target_len) // 2
    return arr[start : start + target_len].astype(np.float32)


# ---------------------------------------------------------------
# High-level: resample to fixed length
# ---------------------------------------------------------------
def resample_to_fixed_length(
    arr: np.ndarray,
    target_len: int = SEQUENCE_LENGTH,
    padding_mode: PadMode = "pad_last",
    long_mode: Literal["interpolate", "trim_center", "trim_start", "trim_end"] = "interpolate",
    short_mode: Literal["interpolate", "pad_last", "pad_zero"] = "interpolate",
) -> np.ndarray:
    """
    Normalisasi panjang sequence ke `target_len`.

    - Jika M == T: return apa adanya.
    - Jika M >  T: `long_mode` (default interpolate).
    - Jika M <  T: `short_mode` (default interpolate; alternatif padding).
    """
    if arr.ndim != 2:
        raise ValueError(f"expected (M, F), got {arr.shape}")

    M = arr.shape[0]
    if M == 0:
        return np.zeros((target_len, arr.shape[1] or FEATURES_PER_FRAME), dtype=np.float32)

    if M == target_len:
        return arr.astype(np.float32, copy=False)

    if M > target_len:
        if long_mode == "interpolate":
            return interpolate_to_length(arr, target_len)
        return trim_sequence(arr, target_len, mode=long_mode)  # type: ignore[arg-type]

    if short_mode == "interpolate":
        return interpolate_to_length(arr, target_len)
    return pad_sequence(arr, target_len, mode=short_mode)  # type: ignore[arg-type]


def resample_from_config(arr: np.ndarray, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Convenience wrapper yang menggunakan `TEMPORAL_PADDING` dari config."""
    mode = TEMPORAL_PADDING  # "interpolate" | "pad_last" | "pad_zero"
    if mode == "interpolate":
        return resample_to_fixed_length(arr, target_len)
    if mode == "pad_zero":
        return resample_to_fixed_length(
            arr, target_len,
            short_mode="pad_zero", long_mode="interpolate",
        )
    # pad_last
    return resample_to_fixed_length(
        arr, target_len,
        short_mode="pad_last", long_mode="interpolate",
    )
