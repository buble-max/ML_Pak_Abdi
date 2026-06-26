from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2

RAW_VIDEO_DIR = Path("dataset/raw_videos")
OUT_DIR = Path("dataset/raw_words")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

FRAME_STEP = 2


def iter_videos(label_dir: Path) -> Iterable[Path]:
    return sorted(p for p in label_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)


def video_to_frames(
    video_path: Path,
    output_clip_dir: Path,
    frame_step: int = FRAME_STEP,
) -> int:
    if frame_step <= 0:
        raise ValueError("frame_step harus lebih besar dari 0")

    output_clip_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[skip] gagal membuka video: {video_path}")
        return 0

    frame_idx = 0
    saved_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_step == 0:
                out_path = output_clip_dir / f"frame_{saved_idx:04d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved_idx += 1

            frame_idx += 1
    finally:
        cap.release()

    print(f"[done] {video_path.name} -> {saved_idx} frame")
    return saved_idx


def convert_all_videos() -> None:
    if not RAW_VIDEO_DIR.exists():
        print(f"[error] folder tidak ditemukan: {RAW_VIDEO_DIR}")
        return

    for label_dir in sorted(RAW_VIDEO_DIR.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        videos = list(iter_videos(label_dir))
        if not videos:
            print(f"[skip] {label}: tidak ada video")
            continue

        for idx, video_path in enumerate(videos, start=1):
            clip_dir = OUT_DIR / label / f"clip_{idx:03d}"
            video_to_frames(video_path, clip_dir)


if __name__ == "__main__":
    convert_all_videos()
