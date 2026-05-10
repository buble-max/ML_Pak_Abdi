"""
Real-time webcam inference BISINDO.

Loop:
    Capture frame → MediaPipe → 21 landmark → normalisasi → SequenceBuffer
    → (jika T frame penuh) predict_smooth → overlay label + confidence.

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

from config import (  # noqa: E402
    MAX_HANDS,
    MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
)
from inference.predictor import BisindoPredictor  # noqa: E402
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
    import mediapipe as mp

    predictor = BisindoPredictor()
    buf = SequenceBuffer()

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=MP_MAX_NUM_HANDS,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    )
    drawer = mp.solutions.drawing_utils
    connections = mp.solutions.hands.HAND_CONNECTIONS

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Webcam tidak dapat dibuka")
        return

    label, conf = "...", 0.0
    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            hand_arrays = []
            if res.multi_hand_landmarks:
                for hand_lms in res.multi_hand_landmarks:
                    arr = np.array(
                        [[p.x, p.y, p.z] for p in hand_lms.landmark],
                        dtype=np.float32,
                    )
                    hand_arrays.append(arr)
                    drawer.draw_landmarks(frame, hand_lms, connections)

            normed = normalize_two_hands(hand_arrays, max_hands=MAX_HANDS)
            buf.push(flatten_frame(normed))

            if buf.is_ready():
                sequence = buf.get()
                label, conf, _ = predictor.predict_smooth(sequence)

            # FPS
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
        hands.close()


if __name__ == "__main__":
    main()
