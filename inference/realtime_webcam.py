"""
Real-time webcam inference BISINDO.

Loop:
    Capture frame → MediaPipe Tasks API (HandLandmarker, mode VIDEO)
    → 21 landmark → normalisasi → SequenceBuffer
    → (jika T frame penuh) predict_smooth → overlay label + confidence.

Migrasi: menggunakan MediaPipe Tasks API (`HandLandmarker`) karena
`mp.solutions.hands` sudah tidak tersedia pada MediaPipe terbaru
(Python 3.12).

Keyboard:
    q   : quit
    r   : reset buffer prediksi
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import MAX_HANDS  # noqa: E402
from inference.predictor import BisindoPredictor  # noqa: E402
from preprocessing.mp_hand_landmarker import (  # noqa: E402
    HandLandmarkerWrapper,
    draw_hand_landmarks,
)
from preprocessing.normalizer import flatten_frame, normalize_two_hands  # noqa: E402
from preprocessing.sequence_builder import SequenceBuffer  # noqa: E402


def _draw_overlay(frame, label: str, conf: float, fps: float) -> None:
    h, w = frame.shape[:2]
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    color = (0, 255, 0) if label not in ("...", "") else (0, 200, 255)
    cv2.putText(
        frame, f"{label}",
        (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3,
    )
    cv2.putText(
        frame, f"conf {conf*100:5.1f}%",
        (w - 260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    cv2.putText(
        frame, f"{fps:4.1f} FPS",
        (w - 260, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
    )
    cv2.putText(
        frame, "q=quit  r=reset",
        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )


def main() -> None:
    predictor = BisindoPredictor()
    buf = SequenceBuffer()

    # MediaPipe Tasks API mode VIDEO untuk streaming webcam.
    landmarker = HandLandmarkerWrapper(running_mode="video")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Webcam tidak dapat dibuka")
        landmarker.close()
        return

    label, conf = "...", 0.0
    t_prev = time.time()
    t_start = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            # Timestamp monotonic yang diperlukan mode VIDEO
            ts_ms = int((time.time() - t_start) * 1000)
            hand_arrays = landmarker.detect_bgr(frame, timestamp_ms=ts_ms)

            # Gambar landmark (helper manual karena solutions.drawing_utils
            # tidak tersedia pada Tasks API).
            draw_hand_landmarks(frame, hand_arrays)

            normed = normalize_two_hands(hand_arrays, max_hands=MAX_HANDS)
            buf.push(flatten_frame(normed))

            if buf.is_ready():
                sequence = buf.get()
                label, conf, _ = predictor.predict_smooth(sequence)

            # FPS (EMA)
            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            _draw_overlay(frame, label, conf, fps)
            cv2.imshow("BISINDO Real-Time", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                buf.reset()
                predictor.reset()
                label, conf = "...", 0.0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
