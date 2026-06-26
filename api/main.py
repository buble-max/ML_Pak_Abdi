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
import sys
import time
from dataclasses import dataclass
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
    NUM_LANDMARKS,
    SEQUENCE_LENGTH,
)
from inference.predictor import (  # noqa: E402
    BisindoPredictor,
    ModelLabelMismatchError,
    PredictionSmoother,
)
from preprocessing.mp_hand_landmarker import HandLandmarkerWrapper  # noqa: E402
from preprocessing.normalizer import flatten_frame, normalize_two_hands  # noqa: E402
from preprocessing.sequence_builder import SequenceBuffer  # noqa: E402
from utils.labels import load_labels  # noqa: E402

# --------------------------------------------------------------- App
app = FastAPI(
    title="BISINDO Sign Language API",
    version="1.0.0",
    description="API deteksi bahasa isyarat BISINDO (alfabet, digit, dan kata).",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------- Singletons
_predictor: Optional[BisindoPredictor] = None


@dataclass
class SessionState:
    buffer: SequenceBuffer
    smoother: PredictionSmoother
    last_seen: float


_sessions: Dict[str, SessionState] = {}
SESSION_TTL_S = 30 * 60


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
        try:
            _predictor = BisindoPredictor()
        except ModelLabelMismatchError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _predictor


def _new_session_state() -> SessionState:
    return SessionState(
        buffer=SequenceBuffer(),
        smoother=PredictionSmoother(),
        last_seen=time.time(),
    )


def _prune_sessions(now: Optional[float] = None) -> None:
    now = now or time.time()
    expired = [
        session_id
        for session_id, state in _sessions.items()
        if now - state.last_seen > SESSION_TTL_S
    ]
    for session_id in expired:
        _sessions.pop(session_id, None)


def _get_session(session_id: str, reset: bool = False) -> SessionState:
    _prune_sessions()
    state = _sessions.get(session_id)
    if state is None or reset:
        state = _new_session_state()
        _sessions[session_id] = state
    else:
        state.last_seen = time.time()
    return state


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
    active_sessions: int
    uptime_s: float


class LabelsResponse(BaseModel):
    labels: List[str]
    num_classes: int


class LandmarkPredictRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description=(
            f"Shape ({SEQUENCE_LENGTH}, {FEATURES_PER_FRAME}). "
            "Default-nya landmark sudah dalam format siap-model/ternormalisasi."
        ),
    )
    normalized: bool = Field(
        True,
        description=(
            "True jika sequence sudah dinormalisasi seperti data training. "
            "False jika sequence masih landmark mentah MediaPipe dan perlu "
            "dinormalisasi server."
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
            "/health", "/labels", "/classes", "/model/classes",
            "/predict/landmarks", "/predict/frame",
            "/ws/realtime",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    _prune_sessions()
    loaded = False
    num_classes = NUM_CLASSES
    try:
        pred = _get_predictor()
        loaded = True
        num_classes = len(pred.labels)
    except HTTPException:
        loaded = False
        try:
            num_classes = len(load_labels())
        except Exception:
            num_classes = NUM_CLASSES
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_path=str(MODEL_PATH),
        num_classes=num_classes,
        sequence_length=SEQUENCE_LENGTH,
        features_per_frame=FEATURES_PER_FRAME,
        active_sessions=len(_sessions),
        uptime_s=time.time() - _START_TIME,
    )


@app.get("/labels", response_model=LabelsResponse)
def get_labels():
    lab = load_labels()
    return LabelsResponse(labels=lab, num_classes=len(lab))


@app.get("/classes", response_model=LabelsResponse)
@app.get("/model/classes", response_model=LabelsResponse)
def get_classes():
    return get_labels()


def _format_prediction(
    probs: np.ndarray, label: str, conf: float, labels: List[str], k: int = 5
) -> PredictionResponse:
    if not np.all(np.isfinite(probs)):
        raise HTTPException(
            status_code=500,
            detail="Model menghasilkan probabilitas tidak valid.",
        )
    if len(probs) != len(labels):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model menghasilkan {len(probs)} probabilitas, tetapi file label "
                f"memiliki {len(labels)} label. Pastikan model dan file label "
                "berasal dari training run yang sama."
            ),
        )

    top_idx = np.argsort(probs)[::-1][: min(k, len(labels))]
    top_k = [{labels[i]: float(probs[i])} for i in top_idx]
    return PredictionResponse(
        label=label,
        confidence=float(conf),
        top_k=top_k,
        is_stable=label != "...",
    )


def _empty_prediction() -> PredictionResponse:
    return PredictionResponse(
        label="...",
        confidence=0.0,
        top_k=[],
        is_stable=False,
    )


def _validate_landmark_sequence(arr: np.ndarray) -> None:
    if arr.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Shape sequence harus ({SEQUENCE_LENGTH}, {FEATURES_PER_FRAME}), "
                f"diterima {arr.shape}"
            ),
        )
    if not np.all(np.isfinite(arr)):
        raise HTTPException(
            status_code=400,
            detail="Sequence hanya boleh berisi angka finite, bukan NaN/Infinity.",
        )
    if np.count_nonzero(arr) == 0:
        raise HTTPException(
            status_code=400,
            detail="Sequence tidak boleh all-zero. Kirim landmark tangan yang valid.",
        )


def _normalize_landmark_sequence(arr: np.ndarray) -> np.ndarray:
    frames = arr.reshape(SEQUENCE_LENGTH, MAX_HANDS, NUM_LANDMARKS, 3)
    normed = [
        normalize_two_hands(_valid_hands(frame), max_hands=MAX_HANDS)
        for frame in frames
    ]
    return np.stack(normed, axis=0).reshape(SEQUENCE_LENGTH, FEATURES_PER_FRAME)


def _normalize_landmark_frame(frame: np.ndarray) -> np.ndarray:
    hands = frame.reshape(MAX_HANDS, NUM_LANDMARKS, 3)
    return normalize_two_hands(_valid_hands(hands), max_hands=MAX_HANDS).reshape(-1)


def _valid_hands(frame_hands: np.ndarray) -> List[np.ndarray]:
    return [hand for hand in frame_hands if np.count_nonzero(hand) > 0]


def _landmark_sequence_to_array(sequence: List[List[float]]) -> np.ndarray:
    try:
        return np.array(sequence, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Sequence harus berupa matrix angka berukuran "
                f"({SEQUENCE_LENGTH}, {FEATURES_PER_FRAME})."
            ),
        ) from exc


@app.post("/predict/landmarks", response_model=PredictionResponse)
def predict_landmarks(req: LandmarkPredictRequest):
    arr = _landmark_sequence_to_array(req.sequence)
    _validate_landmark_sequence(arr)
    if not req.normalized:
        arr = _normalize_landmark_sequence(arr)
    pred = _get_predictor()
    try:
        probs = pred.predict_proba(arr)[0]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Gagal menjalankan prediksi model.",
        ) from exc

    top_idx = int(np.argmax(probs))
    if top_idx >= len(pred.labels):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model menghasilkan index kelas {top_idx}, tetapi file label "
                f"hanya memiliki {len(pred.labels)} label. Pastikan model dan "
                "file label berasal dari training run yang sama."
            ),
        )
    label = pred.labels[top_idx]
    conf = float(probs[top_idx])
    return _format_prediction(probs, label, conf, pred.labels)


@app.post("/predict/frame", response_model=PredictionResponse)
def predict_frame(req: FramePredictRequest):
    """
    Kirim frame satu-per-satu dengan `session_id` yang sama. Server memelihara
    SequenceBuffer per session; prediksi mulai muncul setelah buffer terisi T frame.
    """
    pred = _get_predictor()
    state = _get_session(req.session_id, reset=req.reset)

    img = _decode_image(req.image_base64)
    feat = _frame_to_landmarks(img)
    if np.count_nonzero(feat) == 0:
        state.buffer.reset()
        state.smoother.reset()
        return _empty_prediction()

    state.buffer.push(feat)

    if not state.buffer.is_ready():
        return _empty_prediction()

    sequence = state.buffer.get()
    label, conf, probs = pred.predict_smooth(sequence, smoother=state.smoother)
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
    smoother = PredictionSmoother()
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "reset":
                buf.reset()
                smoother.reset()
                await ws.send_json({"type": "reset_ack"})
                continue

            if mtype == "frame":
                img = _decode_image(msg["image_base64"])
                feat = _frame_to_landmarks(img)
                if np.count_nonzero(feat) == 0:
                    buf.reset()
                    smoother.reset()
                    await ws.send_json(
                        {"type": "prediction", **_empty_prediction().model_dump()}
                    )
                    continue
            elif mtype == "landmarks":
                feat = np.array(msg["frame"], dtype=np.float32)
                if feat.shape != (FEATURES_PER_FRAME,):
                    await ws.send_json(
                        {"type": "error",
                         "detail": f"frame shape harus ({FEATURES_PER_FRAME},)"}
                    )
                    continue
                if not np.all(np.isfinite(feat)):
                    await ws.send_json(
                        {"type": "error",
                         "detail": "frame hanya boleh berisi angka finite."}
                    )
                    continue
                if np.count_nonzero(feat) == 0:
                    await ws.send_json(
                        {"type": "error",
                         "detail": "frame tidak boleh all-zero."}
                    )
                    continue
                if not bool(msg.get("normalized", True)):
                    feat = _normalize_landmark_frame(feat)
            else:
                await ws.send_json({"type": "error", "detail": f"type tidak dikenal: {mtype}"})
                continue

            buf.push(feat)
            if not buf.is_ready():
                await ws.send_json(
                    {"type": "prediction", **_empty_prediction().model_dump()}
                )
                continue

            sequence = buf.get()
            label, conf, probs = pred.predict_smooth(sequence, smoother=smoother)
            resp = _format_prediction(probs, label, conf, pred.labels)
            await ws.send_json({"type": "prediction", **resp.model_dump()})
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
