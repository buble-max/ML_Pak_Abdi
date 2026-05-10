"""
Predictor BISINDO yang reusable untuk real-time webcam & API.

Fitur anti-flicker:
- Prediction buffer (deque) + majority vote
- Confidence threshold
- Smoothing exponential terhadap vektor probabilitas
"""
from __future__ import annotations

import sys
from collections import Counter, deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    BUFFER_SIZE,
    CONFIDENCE_THRESHOLD,
    FEATURES_PER_FRAME,
    MODEL_PATH,
    SEQUENCE_LENGTH,
    SMOOTHING_MIN_VOTES,
)
from utils.labels import load_labels  # noqa: E402


class BisindoPredictor:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        labels: Optional[List[str]] = None,
        buffer_size: int = BUFFER_SIZE,
        min_votes: int = SMOOTHING_MIN_VOTES,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        ema_alpha: float = 0.6,
    ):
        import tensorflow as tf  # import lazy
        self._tf = tf
        self.model = tf.keras.models.load_model(str(model_path))
        self.labels = labels if labels is not None else load_labels()
        self.buffer_size = buffer_size
        self.min_votes = min_votes
        self.confidence_threshold = confidence_threshold
        self.ema_alpha = ema_alpha

        self._pred_buf: Deque[int] = deque(maxlen=buffer_size)
        self._prob_ema: Optional[np.ndarray] = None

    # ---------- Inference primitif ----------
    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """sequence: (T, F) atau (B, T, F). Return softmax probabilities."""
        if sequence.ndim == 2:
            sequence = sequence[None, ...]
        if sequence.shape[1:] != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
            raise ValueError(
                f"Expected ( ..., {SEQUENCE_LENGTH}, {FEATURES_PER_FRAME}), "
                f"got {sequence.shape}"
            )
        probs = self.model.predict(sequence.astype(np.float32), verbose=0)
        return probs  # (B, C)

    # ---------- Anti-flicker ----------
    def predict_smooth(self, sequence: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Prediksi 1 sequence + terapkan EMA pada probabilitas dan majority vote
        pada buffer prediksi diskrit. Return (label, confidence, probs_ema).
        """
        probs = self.predict_proba(sequence)[0]

        # EMA smoothing pada vektor probabilitas
        if self._prob_ema is None:
            self._prob_ema = probs.copy()
        else:
            self._prob_ema = (
                self.ema_alpha * probs + (1 - self.ema_alpha) * self._prob_ema
            )

        top_idx = int(np.argmax(self._prob_ema))
        top_conf = float(self._prob_ema[top_idx])

        # Push ke buffer & majority vote
        self._pred_buf.append(top_idx)
        vote_idx, vote_count = Counter(self._pred_buf).most_common(1)[0]

        label = "..."
        conf = top_conf
        if (
            vote_count >= self.min_votes
            and top_conf >= self.confidence_threshold
            and top_idx == vote_idx
        ):
            label = self.labels[top_idx]

        return label, conf, self._prob_ema

    def reset(self) -> None:
        self._pred_buf.clear()
        self._prob_ema = None
