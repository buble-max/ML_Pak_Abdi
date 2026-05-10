"""
FastAPI server BISINDO.

Endpoints:
    GET  /                     → info singkat
    GET  /health               → status API & model
    GET  /labels               → daftar kelas
    POST /predict/landmarks    → input sequence landmark (T x F) → prediksi
    POST /predict/frame        → input 1 frame gambar (base64) → ekstrak + buffer
                                 → prediksi (stateless, perlu kirim banyak frame
                                 berurutan via session_id yang sama)
    WS   /ws/realtime          → streaming prediksi real-time (frame-by-frame)

Menjalankan:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    API_HOST,
    API_PORT,
    FEATURES_PER_FRAME,
    MAX_HANDS,
    MODEL_PATH,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
)
from inference.predictor import BisindoPredictor  # noqa: E402
from preprocessing.mp_hand_landmarker import HandLandmarkerWrapper  # noqa: E402
from preprocessing.normalizer import flatten_frame, normalize_two_hands  # noqa: E402
from preprocessing.sequence_builder import SequenceBuffer  # noqa: E402
from utils.labels import load_labels  # noqa: E402

# --------------------------------------------------------------- App
app = FastAPI(
    title="BISINDO Sign Language API",
    version="1.0.0",
    description="API deteksi bahasa isyarat BISINDO (alfabet + 5 kata).",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------- Singletons
_predictor: Optional[BisindoPredictor] = None
_session_buffers: Dict[str, SequenceBuffer] = {}


def _get_predictor() -> BisindoPredictor:
    global _predictor
    if _predictor is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model belum tersedia di {MODEL_PATH}. "
                    "Jalankan training terlebih dahulu."
                ),
            )
        _predictor = BisindoPredictor()
    return _predictor


def _get_landmarker() -> HandLandmarkerWrapper:
    """
    Lazy init HandLandmarker (MediaPipe Tasks API) dalam mode IMAGE karena
    setiap request HTTP membawa frame independen (bukan stream).
    """
    if not hasattr(_get_landmarker, "_inst"):
        _get_landmarker._inst = HandLandmarkerWrapper(running_mode="image")
    return _get_landmarker._inst


def _decode_image(b64: str) -> np.ndarray:
    """data URI atau base64 murni → BGR np.ndarray."""
    import cv2
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Gagal decode gambar base64")
    return img


def _frame_to_landmarks(img_bgr: np.ndarray) -> np.ndarray:
    """
    Deteksi tangan → normalisasi → padding ke MAX_HANDS → flatten.
    Return shape: (FEATURES_PER_FRAME,).
    """
    landmarker = _get_landmarker()
    hand_arrays = landmarker.detect_bgr(img_bgr)
    normed = normalize_two_hands(hand_arrays, max_hands=MAX_HANDS)
    return flatten_frame(normed)


# --------------------------------------------------------------- Schemas
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    num_classes: int
    sequence_length: int
    features_per_frame: int
    uptime_s: float


class LabelsResponse(BaseModel):
    labels: List[str]
    num_classes: int


class LandmarkPredictRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description=(
            f"Shape ({SEQUENCE_LENGTH}, {FEATURES_PER_FRAME}). "
            "Landmark sudah harus dinormalisasi."
        ),
    )


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    top_k: List[Dict[str, float]]
    is_stable: bool


class FramePredictRequest(BaseModel):
    session_id: str = Field(..., description="ID sesi untuk menjaga buffer frame")
    image_base64: str
    reset: bool = False


_START_TIME = time.time()


# --------------------------------------------------------------- Routes
@app.get("/")
def root():
    return {
        "name": "BISINDO Sign Language API",
        "endpoints": [
            "/health", "/labels",
            "/predict/landmarks", "/predict/frame",
            "/ws/realtime",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = False
    try:
        _get_predictor()
        loaded = True
    except HTTPException:
        loaded = False
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_path=str(MODEL_PATH),
        num_classes=NUM_CLASSES,
        sequence_length=SEQUENCE_LENGTH,
        features_per_frame=FEATURES_PER_FRAME,
        uptime_s=time.time() - _START_TIME,
    )


@app.get("/labels", response_model=LabelsResponse)
def get_labels():
    lab = load_labels()
    return LabelsResponse(labels=lab, num_classes=len(lab))


def _format_prediction(
    probs: np.ndarray, label: str, conf: float, labels: List[str], k: int = 5
) -> PredictionResponse:
    top_idx = np.argsort(probs)[::-1][:k]
    top_k = [{"label": labels[i], "confidence": float(probs[i])} for i in top_idx]
    return PredictionResponse(
        label=label,
        confidence=float(conf),
        top_k=top_k,
        is_stable=label != "...",
    )


@app.post("/predict/landmarks", response_model=PredictionResponse)
def predict_landmarks(req: LandmarkPredictRequest):
    arr = np.array(req.sequence, dtype=np.float32)
    if arr.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Shape sequence harus ({SEQUENCE_LENGTH}, {FEATURES_PER_FRAME}), "
                f"diterima {arr.shape}"
            ),
        )
    pred = _get_predictor()
    label, conf, probs_ema = pred.predict_smooth(arr)
    return _format_prediction(probs_ema, label, conf, pred.labels)


@app.post("/predict/frame", response_model=PredictionResponse)
def predict_frame(req: FramePredictRequest):
    """
    Kirim frame satu-per-satu dengan `session_id` yang sama. Server memelihara
    SequenceBuffer per session; prediksi mulai muncul setelah buffer terisi T frame.
    """
    pred = _get_predictor()

    buf = _session_buffers.get(req.session_id)
    if buf is None or req.reset:
        buf = SequenceBuffer()
        _session_buffers[req.session_id] = buf
        pred.reset()

    img = _decode_image(req.image_base64)
    feat = _frame_to_landmarks(img)
    buf.push(feat)

    if not buf.is_ready():
        labels = pred.labels
        zeros = np.zeros(len(labels), dtype=np.float32)
        return _format_prediction(zeros, "...", 0.0, labels)

    sequence = buf.get()
    label, conf, probs = pred.predict_smooth(sequence)
    return _format_prediction(probs, label, conf, pred.labels)


@app.websocket("/ws/realtime")
async def websocket_realtime(ws: WebSocket):
    """
    Protokol:
        Client → {"type": "frame", "image_base64": "..."}
        Client → {"type": "landmarks", "frame": [F floats]}  # alternatif
        Client → {"type": "reset"}
        Server → PredictionResponse JSON per pesan frame
    """
    await ws.accept()
    pred = _get_predictor()
    buf = SequenceBuffer()
    pred.reset()
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "reset":
                buf.reset()
                pred.reset()
                await ws.send_json({"type": "reset_ack"})
                continue

            if mtype == "frame":
                img = _decode_image(msg["image_base64"])
                feat = _frame_to_landmarks(img)
            elif mtype == "landmarks":
                feat = np.array(msg["frame"], dtype=np.float32)
                if feat.shape != (FEATURES_PER_FRAME,):
                    await ws.send_json(
                        {"type": "error",
                         "detail": f"frame shape harus ({FEATURES_PER_FRAME},)"}
                    )
                    continue
            else:
                await ws.send_json({"type": "error", "detail": f"type tidak dikenal: {mtype}"})
                continue

            buf.push(feat)
            if not buf.is_ready():
                await ws.send_json(
                    {"type": "prediction", "label": "...", "confidence": 0.0,
                     "is_stable": False, "top_k": []}
                )
                continue

            sequence = buf.get()
            label, conf, probs = pred.predict_smooth(sequence)
            resp = _format_prediction(probs, label, conf, pred.labels)
            await ws.send_json({"type": "prediction", **resp.model_dump()})
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
