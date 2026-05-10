"""
Normalisasi landmark tangan agar invariant terhadap:
- Posisi tangan (translasi)  → dikurangi titik wrist (index 0)
- Ukuran tangan / jarak kamera → dibagi jarak wrist → middle-MCP (index 9)
- Perbedaan antar pengguna

Input  : np.ndarray shape (21, 3)
Output : np.ndarray shape (21, 3) ter-normalisasi
"""
from __future__ import annotations

import numpy as np

WRIST = 0
MIDDLE_MCP = 9


def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """Normalisasi 1 tangan (21, 3)."""
    if landmarks is None or landmarks.size == 0:
        return np.zeros((21, 3), dtype=np.float32)

    lm = landmarks.astype(np.float32).copy()

    # 1) Translasi: origin di wrist
    lm = lm - lm[WRIST]

    # 2) Skala: bagi dengan jarak wrist ↔ middle finger MCP (titik 9)
    ref_len = np.linalg.norm(lm[MIDDLE_MCP])
    if ref_len < 1e-6:
        ref_len = 1.0
    lm = lm / ref_len

    return lm.astype(np.float32)


def normalize_two_hands(hands: list[np.ndarray] | None, max_hands: int = 2) -> np.ndarray:
    """
    Pastikan output berukuran tetap (max_hands, 21, 3). Jika tangan < max_hands,
    isi dengan nol. Urutan tangan diurutkan berdasarkan x wrist (kiri→kanan).
    """
    out = np.zeros((max_hands, 21, 3), dtype=np.float32)
    if not hands:
        return out

    # sort berdasarkan x wrist sebelum normalisasi (lebih stabil)
    hands_sorted = sorted(hands, key=lambda h: float(h[WRIST, 0]))[:max_hands]
    for i, h in enumerate(hands_sorted):
        out[i] = normalize_hand(h)
    return out


def flatten_frame(frame_hands: np.ndarray) -> np.ndarray:
    """(max_hands, 21, 3) → (max_hands*21*3,)"""
    return frame_hands.reshape(-1).astype(np.float32)
