"""
Contoh klien API BISINDO.

Mendemonstrasikan 3 cara pemanggilan:
    1. POST /predict/landmarks  - kirim sequence (T, F) langsung
    2. POST /predict/frame       - kirim frame base64 berurutan per session
    3. WS   /ws/realtime         - streaming landmark real-time dari webcam

Jalankan server terlebih dulu:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Kemudian:
    python api/example_client.py landmarks
    python api/example_client.py frame
    python api/example_client.py ws
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    API_HOST,
    API_PORT,
    FEATURES_PER_FRAME,
    SEQUENCE_LENGTH,
)

BASE = f"http://{API_HOST if API_HOST != '0.0.0.0' else 'localhost'}:{API_PORT}"


def demo_landmarks() -> None:
    """Kirim random sequence (hanya uji kontrak API)."""
    seq = np.random.randn(SEQUENCE_LENGTH, FEATURES_PER_FRAME).astype(np.float32) * 0.1
    r = requests.post(
        f"{BASE}/predict/landmarks",
        json={"sequence": seq.tolist(), "normalized": True},
        timeout=30,
    )
    print("status:", r.status_code)
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))


def demo_frame_session() -> None:
    """Webcam → base64 → kirim ~T frame dengan session_id yang sama."""
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Webcam tidak tersedia")
        return
    sid = str(uuid.uuid4())
    try:
        for i in range(SEQUENCE_LENGTH + 5):
            ok, frame = cap.read()
            if not ok:
                break
            _, buf = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            r = requests.post(
                f"{BASE}/predict/frame",
                json={"session_id": sid, "image_base64": b64, "reset": i == 0},
                timeout=30,
            )
            data = r.json()
            print(f"[{i:02d}] label={data['label']:<12} "
                  f"conf={data['confidence']:.2f} stable={data['is_stable']}")
    finally:
        cap.release()


async def demo_ws_landmarks() -> None:
    """Streaming landmark real-time via WebSocket (MediaPipe Tasks API)."""
    import time

    import cv2
    import websockets

    from preprocessing.mp_hand_landmarker import HandLandmarkerWrapper
    from preprocessing.normalizer import flatten_frame, normalize_two_hands

    uri = f"ws://{API_HOST if API_HOST != '0.0.0.0' else 'localhost'}:{API_PORT}/ws/realtime"
    landmarker = HandLandmarkerWrapper(running_mode="video")
    cap = cv2.VideoCapture(0)
    t_start = time.time()

    async with websockets.connect(uri) as ws:
        try:
            for _ in range(200):
                ok, frame = cap.read()
                if not ok:
                    break
                ts_ms = int((time.time() - t_start) * 1000)
                hand_arrays = landmarker.detect_bgr(frame, timestamp_ms=ts_ms)
                feat = flatten_frame(
                    normalize_two_hands(hand_arrays, max_hands=2)
                )
                await ws.send(json.dumps({
                    "type": "landmarks",
                    "frame": feat.tolist(),
                    "normalized": True,
                }))
                resp = json.loads(await ws.recv())
                if resp.get("type") == "prediction":
                    print(f"label={resp['label']:<12} conf={resp['confidence']:.2f}")
        finally:
            cap.release()
            landmarker.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "landmarks"
    if mode == "landmarks":
        demo_landmarks()
    elif mode == "frame":
        demo_frame_session()
    elif mode == "ws":
        asyncio.run(demo_ws_landmarks())
    else:
        print(__doc__)
