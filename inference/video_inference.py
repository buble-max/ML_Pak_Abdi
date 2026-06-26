"""
Inference dari file video utuh (CLI):

    python -m inference.video_inference path/to/video.mp4
    python -m inference.video_inference path/to/video.mp4 --sliding

Mode default: 1 video -> 1 prediksi (resample ke T).
Mode --sliding: sliding window di ruang landmark -> list prediksi per window
(cocok untuk video kalimat / panjang).

Output: JSON ke stdout + opsional overlay video dengan --save-overlay.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    FRAME_STRIDE, SEQUENCE_LENGTH, TEMPORAL_SAMPLING,
)
from inference.predictor import BisindoPredictor  # noqa: E402
from preprocessing.landmark_extractor import (  # noqa: E402
    _get_landmarker, video_to_sequence, video_to_sequences_sliding,
)


def infer_video(video_path: Path, sliding: bool = False) -> dict:
    predictor = BisindoPredictor()
    landmarker = _get_landmarker(running_mode="image")

    try:
        if sliding:
            seqs = video_to_sequences_sliding(
                video_path, landmarker,
                seq_len=SEQUENCE_LENGTH, frame_stride=FRAME_STRIDE,
            )
            windows = []
            for i, s in enumerate(seqs):
                probs = predictor.predict_proba(s)[0]
                top = int(np.argmax(probs))
                windows.append({
                    "window": i,
                    "label": predictor.labels[top],
                    "confidence": float(probs[top]),
                    "top_k": [
                        {"label": predictor.labels[int(j)],
                         "confidence": float(probs[int(j)])}
                        for j in np.argsort(probs)[::-1][:5]
                    ],
                })
            return {
                "video": str(video_path),
                "mode": "sliding",
                "num_windows": len(windows),
                "windows": windows,
            }

        seq = video_to_sequence(
            video_path, landmarker,
            seq_len=SEQUENCE_LENGTH, frame_stride=FRAME_STRIDE,
            mode=TEMPORAL_SAMPLING,
        )
        if seq is None:
            return {"video": str(video_path), "error": "landmark extraction failed"}

        probs = predictor.predict_proba(seq)[0]
        top = int(np.argmax(probs))
        return {
            "video": str(video_path),
            "mode": "single",
            "label": predictor.labels[top],
            "confidence": float(probs[top]),
            "top_k": [
                {"label": predictor.labels[int(j)],
                 "confidence": float(probs[int(j)])}
                for j in np.argsort(probs)[::-1][:5]
            ],
        }
    finally:
        landmarker.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", help="Path file video .mp4/.avi/...")
    p.add_argument("--sliding", action="store_true",
                   help="Sliding window di ruang landmark (cocok untuk kalimat panjang).")
    args = p.parse_args()

    result = infer_video(Path(args.video), sliding=args.sliding)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
