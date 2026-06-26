"""
Predictor BISINDO yang reusable untuk real-time webcam & API.

Fitur anti-flicker:
- Prediction buffer (deque) + majority vote
- Confidence threshold
- Smoothing exponential terhadap vektor probabilitas
"""
from __future__ import annotations

import sys
import os
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

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class ModelLabelMismatchError(RuntimeError):
    """Raised when the loaded model output does not match the label file."""


class PredictionSmoother:
    """State anti-flicker untuk satu stream prediksi."""

    def __init__(
        self,
        buffer_size: int = BUFFER_SIZE,
        min_votes: int = SMOOTHING_MIN_VOTES,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        ema_alpha: float = 0.6,
    ) -> None:
        self.buffer_size = buffer_size
        self.min_votes = min_votes
        self.confidence_threshold = confidence_threshold
        self.ema_alpha = ema_alpha
        self._pred_buf: Deque[int] = deque(maxlen=buffer_size)
        self._prob_ema: Optional[np.ndarray] = None

    def update(self, probs: np.ndarray, labels: List[str]) -> Tuple[str, float, np.ndarray]:
        if len(probs) != len(labels):
            raise ModelLabelMismatchError(
                f"Model output has {len(probs)} classes, but the label file has "
                f"{len(labels)} labels."
            )

        if self._prob_ema is None:
            self._prob_ema = probs.astype(np.float32, copy=True)
        else:
            self._prob_ema = (
                self.ema_alpha * probs + (1.0 - self.ema_alpha) * self._prob_ema
            )

        top_idx = int(np.argmax(self._prob_ema))
        top_conf = float(self._prob_ema[top_idx])

        self._pred_buf.append(top_idx)
        vote_idx, vote_count = Counter(self._pred_buf).most_common(1)[0]

        label = "..."
        if (
            vote_count >= self.min_votes
            and top_conf >= self.confidence_threshold
            and top_idx == vote_idx
        ):
            label = labels[top_idx]

        return label, top_conf, self._prob_ema

    def reset(self) -> None:
        self._pred_buf.clear()
        self._prob_ema = None


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
        tf.get_logger().setLevel("ERROR")
        self._tf = tf
        self.model = tf.keras.models.load_model(str(model_path))
        self.labels = labels if labels is not None else load_labels()
        self._validate_model_contract(model_path)
        self.smoother = PredictionSmoother(
            buffer_size=buffer_size,
            min_votes=min_votes,
            confidence_threshold=confidence_threshold,
            ema_alpha=ema_alpha,
        )

    def _validate_model_contract(self, model_path: Path) -> None:
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        expected_input = (SEQUENCE_LENGTH, FEATURES_PER_FRAME)
        actual_input = tuple(input_shape[-2:]) if input_shape else None
        if actual_input != expected_input:
            raise ModelLabelMismatchError(
                f"Model {model_path} memiliki input shape {input_shape}, "
                f"tetapi config mengharapkan (None, {SEQUENCE_LENGTH}, "
                f"{FEATURES_PER_FRAME})."
            )

        output_shape = self.model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        output_classes = output_shape[-1] if output_shape else None
        if output_classes is None:
            raise ModelLabelMismatchError(
                f"Tidak bisa membaca jumlah output class dari model {model_path}."
            )

        if int(output_classes) != len(self.labels):
            raise ModelLabelMismatchError(
                f"Model {model_path} memiliki {int(output_classes)} output class, "
                f"tetapi file label memiliki {len(self.labels)} label. "
                "Pastikan model dan file label berasal dari training run yang sama, "
                "atau latih ulang model dengan daftar kelas saat ini."
            )

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
    def predict_smooth(
        self,
        sequence: np.ndarray,
        smoother: Optional[PredictionSmoother] = None,
    ) -> Tuple[str, float, np.ndarray]:
        """
        Prediksi 1 sequence + terapkan EMA pada probabilitas dan majority vote
        pada buffer prediksi diskrit. Return (label, confidence, probs_ema).
        """
        probs = self.predict_proba(sequence)[0]
        return (smoother or self.smoother).update(probs, self.labels)

    def reset(self) -> None:
        self.smoother.reset()
