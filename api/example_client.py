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
        json={"sequence": seq.tolist()},
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
    """Streaming landmark real-time via WebSocket."""
    import cv2
    import mediapipe as mp
    import websockets

    from preprocessing.normalizer import flatten_frame, normalize_two_hands

    uri = f"ws://{API_HOST if API_HOST != '0.0.0.0' else 'localhost'}:{API_PORT}/ws/realtime"
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(0)

    async with websockets.connect(uri) as ws:
        try:
            for _ in range(200):
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                arrs = []
                if res.multi_hand_landmarks:
                    for h in res.multi_hand_landmarks:
                        arrs.append(np.array(
                            [[p.x, p.y, p.z] for p in h.landmark], dtype=np.float32
                        ))
                feat = flatten_frame(normalize_two_hands(arrs, max_hands=2))
                await ws.send(json.dumps({"type": "landmarks", "frame": feat.tolist()}))
                resp = json.loads(await ws.recv())
                if resp.get("type") == "prediction":
                    print(f"label={resp['label']:<12} conf={resp['confidence']:.2f}")
        finally:
            cap.release()


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
