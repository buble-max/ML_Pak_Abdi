"""
Utility tambahan untuk sliding window di inference real-time.
Menyediakan buffer yang menahan T frame landmark terakhir.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np

from config import FEATURES_PER_FRAME, SEQUENCE_LENGTH


class SequenceBuffer:
    """
    Rolling window frame landmark untuk real-time inference.
    Tambahkan satu vektor (FEATURES_PER_FRAME,) per frame; ambil sequence
    (T, F) yang siap dikonsumsi model hanya ketika buffer sudah penuh.
    """

    def __init__(
        self,
        seq_len: int = SEQUENCE_LENGTH,
        feat_dim: int = FEATURES_PER_FRAME,
    ):
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self._buf: Deque[np.ndarray] = deque(maxlen=seq_len)

    def push(self, frame_feat: np.ndarray) -> None:
        if frame_feat.shape != (self.feat_dim,):
            raise ValueError(
                f"Expected ({self.feat_dim},), got {frame_feat.shape}"
            )
        self._buf.append(frame_feat.astype(np.float32))

    def is_ready(self) -> bool:
        return len(self._buf) == self.seq_len

    @property
    def fill_level(self) -> int:
        return len(self._buf)

    def __len__(self) -> int:
        return len(self._buf)

    def get(self) -> Optional[np.ndarray]:
        if not self.is_ready():
            return None
        return np.stack(self._buf, axis=0)  # (T, F)

    def reset(self) -> None:
        self._buf.clear()
