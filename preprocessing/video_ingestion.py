"""
Video ingestion pipeline untuk BISINDO.

Membaca file video utuh (.mp4, .avi, .mov, .mkv, .webm) menggunakan
OpenCV `VideoCapture`, melakukan decoding frame, dan mengembalikan
array frame yang siap diproses pipeline landmark.

Empat strategi temporal sampling:
    - "uniform"  : sampling dengan interval tetap (linspace).
    - "stride"   : ambil setiap frame ke-N (sesuai FRAME_STRIDE).
    - "random"   : pilih T frame acak (sorted) dari total frame.
    - "adaptive" : gabungan stride + uniform.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Tuple

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    FRAME_STRIDE,
    SEQUENCE_LENGTH,
    TEMPORAL_SAMPLING,
    VIDEO_EXTENSIONS,
)

SamplingMode = Literal["uniform", "stride", "random", "adaptive"]


def list_videos(directory: Path) -> List[Path]:
    """Enumerasi semua file video di `directory` (non-rekursif)."""
    if not directory.exists() or not directory.is_dir():
        return []
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}"
            for e in VIDEO_EXTENSIONS}
    return sorted(
        [p for p in directory.iterdir()
         if p.is_file() and p.suffix.lower() in exts]
    )


def probe_video(video_path: Path) -> Tuple[int, float, Tuple[int, int]]:
    """Return (frame_count, fps, (width, height))."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Tidak dapat membuka video: {video_path}")
    try:
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return nframes, fps, (w, h)
    finally:
        cap.release()


def iter_video_frames(video_path: Path) -> Iterator[np.ndarray]:
    """Generator frame BGR dari video (in-order)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Tidak dapat membuka video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield frame
    finally:
        cap.release()


def _compute_sampling_indices(
    total: int,
    seq_len: int,
    mode: SamplingMode,
    frame_stride: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Hitung daftar indeks frame sesuai mode sampling."""
    if total <= 0:
        return np.zeros(0, dtype=np.int64)

    stride = max(1, int(frame_stride))

    if mode == "stride":
        idx = np.arange(0, total, stride, dtype=np.int64)
        if idx.size >= seq_len:
            pick = np.linspace(0, idx.size - 1, seq_len).astype(np.int64)
            return idx[pick]
        return idx

    if mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        k = min(seq_len, total)
        return np.sort(rng.choice(total, size=k, replace=False).astype(np.int64))

    if mode == "adaptive":
        strided = np.arange(0, total, stride, dtype=np.int64)
        n = strided.size
        if n >= seq_len:
            pick = np.linspace(0, n - 1, seq_len).astype(np.int64)
            return strided[pick]
        return strided

    # default: uniform
    if total <= seq_len:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, seq_len).astype(np.int64)


def read_video_frames(
    video_path: Path,
    seq_len: int = SEQUENCE_LENGTH,
    mode: SamplingMode = TEMPORAL_SAMPLING,
    frame_stride: int = FRAME_STRIDE,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Decode video → pilih frame berdasarkan mode sampling → return list BGR
    frames (panjang bisa < seq_len untuk video pendek; resampler hilir akan
    melakukan padding/interpolasi).
    """
    total, _, _ = probe_video(video_path)
    indices = _compute_sampling_indices(
        total=total, seq_len=seq_len, mode=mode,
        frame_stride=frame_stride, rng=rng,
    )
    if indices.size == 0:
        return []

    wanted = set(int(i) for i in indices)
    max_idx = int(indices.max())
    picked: dict[int, np.ndarray] = {}

    for frame_idx, frame in enumerate(iter_video_frames(video_path)):
        if frame_idx in wanted:
            picked[frame_idx] = frame
        if frame_idx >= max_idx:
            break

    return [picked[i] for i in indices.tolist() if i in picked]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path ke file video")
    p.add_argument("--mode", default=TEMPORAL_SAMPLING,
                   choices=["uniform", "stride", "random", "adaptive"])
    args = p.parse_args()

    video = Path(args.path)
    n, fps, (w, h) = probe_video(video)
    print(f"[probe] {video.name}: {n} frames, {fps:.2f} fps, {w}x{h}")

    frames = read_video_frames(video, mode=args.mode)
    print(f"[sampled] {len(frames)} frames (mode={args.mode}, "
          f"seq_len={SEQUENCE_LENGTH}, stride={FRAME_STRIDE})")
