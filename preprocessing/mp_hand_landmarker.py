"""
Wrapper MediaPipe Tasks API (HandLandmarker).

MIGRASI: Pipeline lama menggunakan `mp.solutions.hands.Hands` yang
sudah dihapus pada MediaPipe terbaru (Python 3.12) sehingga muncul:

    AttributeError: module 'mediapipe' has no attribute 'solutions'

Module ini menggantikan pendekatan tersebut dengan MediaPipe Tasks API
(`mediapipe.tasks.python.vision.HandLandmarker`) menggunakan model
`hand_landmarker.task`. API ini lebih modern, resmi, dan lebih stabil
untuk deployment production.

Fitur:
- Auto-download `hand_landmarker.task` dari Google Cloud Storage.
- Mode IMAGE (untuk preprocessing batch) dan VIDEO (untuk real-time).
- Ekstraksi seragam: list of np.ndarray shape (21, 3) per tangan.
- Helper drawing (21 titik + koneksi jari) karena `mp.solutions.drawing_utils`
  juga tidak tersedia pada Tasks API.

Pemakaian:
    from preprocessing.mp_hand_landmarker import HandLandmarkerWrapper

    hl = HandLandmarkerWrapper(running_mode="image")
    hands = hl.detect_bgr(frame_bgr)     # list[np.ndarray(21, 3)]

    # real-time
    hl_vid = HandLandmarkerWrapper(running_mode="video")
    hands = hl_vid.detect_bgr(frame_bgr, timestamp_ms=int(time.time() * 1000))
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Literal, Optional
from urllib.request import urlretrieve

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    ROOT_DIR,
)

# ---------------------------------------------------------------
# Model asset
# ---------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_DIR = ROOT_DIR / "model" / "mp_assets"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


def ensure_model(model_path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Unduh `hand_landmarker.task` jika belum ada."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        print(f"[mp-tasks] Mengunduh model: {url}")
        try:
            urlretrieve(url, model_path)
            print(f"[mp-tasks] Model tersimpan di {model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Gagal mengunduh hand_landmarker.task dari {url}. "
                f"Unduh manual dan taruh di {model_path}. Error: {e}"
            )
    return model_path


# ---------------------------------------------------------------
# Hand connections (untuk drawing manual karena solutions.drawing_utils
# juga tidak tersedia pada Tasks API).
# 21 titik landmark MediaPipe Hand (indeks standar).
# ---------------------------------------------------------------
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (0, 17),
]


# ---------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------
RunningMode = Literal["image", "video"]


class HandLandmarkerWrapper:
    """
    Wrapper tipis di atas `mediapipe.tasks.python.vision.HandLandmarker`.

    Running modes:
      - "image": panggil `detect()`. Cocok untuk preprocessing gambar statis.
      - "video": panggil `detect_for_video()`. Cocok untuk webcam real-time
                 (lebih cepat dan hasil temporal-consistent).
    """

    def __init__(
        self,
        running_mode: RunningMode = "image",
        num_hands: int = MP_MAX_NUM_HANDS,
        min_hand_detection_confidence: float = MP_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence: float = MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MP_MIN_TRACKING_CONFIDENCE,
        model_path: Optional[Path] = None,
    ):
        # Import ditunda supaya testing tanpa mediapipe lebih mudah.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._mp_vision = mp_vision
        self.running_mode = running_mode

        model_path = ensure_model(model_path or MODEL_PATH)

        mode_enum = (
            mp_vision.RunningMode.IMAGE
            if running_mode == "image"
            else mp_vision.RunningMode.VIDEO
        )

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode_enum,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # ------------------------------------------------------------
    # Deteksi
    # ------------------------------------------------------------
    def detect_rgb(
        self,
        rgb: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Input : RGB uint8 (H, W, 3).
        Output: list of np.ndarray (21, 3) – satu per tangan terdeteksi.
        """
        mp_img = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        if self.running_mode == "image":
            result = self._landmarker.detect(mp_img)
        else:
            if timestamp_ms is None:
                raise ValueError(
                    "running_mode='video' membutuhkan timestamp_ms (int, ms)."
                )
            result = self._landmarker.detect_for_video(mp_img, int(timestamp_ms))

        hands_out: List[np.ndarray] = []
        if result and result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                arr = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks],
                    dtype=np.float32,
                )  # (21, 3)
                hands_out.append(arr)
        return hands_out

    def detect_bgr(
        self,
        bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Convenience untuk frame OpenCV (BGR)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.detect_rgb(rgb, timestamp_ms=timestamp_ms)

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---------------------------------------------------------------
# Drawing helper (menggantikan mp.solutions.drawing_utils)
# ---------------------------------------------------------------
def draw_hand_landmarks(
    bgr: np.ndarray,
    hands_norm: List[np.ndarray],
    point_color=(0, 255, 0),
    line_color=(200, 200, 200),
    point_radius: int = 4,
    line_thickness: int = 2,
) -> None:
    """
    Gambar 21 titik + koneksi untuk setiap tangan pada `bgr` (in-place).
    `hands_norm` adalah list of (21, 3) dengan x, y ternormalisasi [0, 1]
    (sesuai output mentah MediaPipe sebelum normalisasi wrist-scale kita).
    """
    if not hands_norm:
        return
    h, w = bgr.shape[:2]
    for lm in hands_norm:
        if lm is None or lm.size == 0:
            continue
        pts = [(int(round(float(x) * w)), int(round(float(y) * h)))
               for x, y, _ in lm]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(bgr, pts[a], pts[b], line_color, line_thickness)
        for p in pts:
            cv2.circle(bgr, p, point_radius, point_color, -1)


if __name__ == "__main__":
    # Smoke test: pastikan model bisa di-load.
    ensure_model()
    hl = HandLandmarkerWrapper(running_mode="image")
    print(f"[ok] HandLandmarker siap. Model: {MODEL_PATH}")
    hl.close()
