"""
Webcam temporal recorder: rekam gesture BISINDO langsung sebagai video
utuh (.mp4) DAN simultan menghasilkan landmark sequence (.npy) yang
siap langsung masuk pipeline training video-based.

Alur per clip:
  1. User tekan SPACE -> countdown 3 detik.
  2. Rekam N frame dari webcam, simpan ke .mp4 (VideoWriter).
     Secara simultan, setiap frame diproses MediaPipe untuk menghasilkan
     landmark (T, F) yang disimpan ke X_live / dataset/processed/live.
  3. Jika landmark berhasil diekstraksi pada minimal sebagian frame,
     sequence dipush ke buffer live dan siap dipakai oleh pipeline
     `preprocessing.landmark_extractor` (cabang LIVE) tanpa preprocessing
     tambahan.

Output:
  dataset/raw_videos/<LABEL>/clip_XXXX.mp4       (video utuh)
  dataset/processed/live/{X_live.npy, y_live.npy} (landmark append)

Kontrol keyboard:
  SPACE = mulai rekam 1 klip (countdown 3s)
  n/p   = next/prev label
  s     = simpan dataset live (.npy)
  d     = statistik dataset live
  r     = hapus klip terakhir (video + landmark)
  q     = simpan & keluar
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    ALL_CLASSES,
    FEATURES_PER_FRAME,
    MAX_HANDS,
    PROCESSED_DIR,
    SEQUENCE_LENGTH,
    VIDEO_RAW_DIR,
)
from preprocessing.mp_hand_landmarker import (  # noqa: E402
    HandLandmarkerWrapper, draw_hand_landmarks,
)
from preprocessing.normalizer import flatten_frame, normalize_two_hands  # noqa: E402
from preprocessing.temporal_resampler import resample_from_config  # noqa: E402
from utils.labels import build_label_maps  # noqa: E402

LIVE_DIR = PROCESSED_DIR / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
X_LIVE_PATH = LIVE_DIR / "X_live.npy"
Y_LIVE_PATH = LIVE_DIR / "y_live.npy"

LABEL_TO_IDX, IDX_TO_LABEL = build_label_maps()
NUM_LABELS = len(ALL_CLASSES)

# Default frame per clip (2x sequence length agar memiliki resolusi temporal lebih).
FRAMES_PER_CLIP = SEQUENCE_LENGTH * 2
COUNTDOWN_SECONDS = 3


def _load_existing() -> Tuple[List[np.ndarray], List[int]]:
    seqs: List[np.ndarray] = []
    labels: List[int] = []
    if X_LIVE_PATH.exists() and Y_LIVE_PATH.exists():
        X = np.load(X_LIVE_PATH)
        y = np.load(Y_LIVE_PATH)
        seqs = list(X)
        labels = list(y)
        print(f"[info] Dataset live existing: {len(seqs)} sample")
    return seqs, labels


def _save_dataset(seqs: List[np.ndarray], labels: List[int]) -> None:
    if not seqs:
        print("[warn] Tidak ada data untuk disimpan.")
        return
    X = np.stack(seqs, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    np.save(X_LIVE_PATH, X)
    np.save(Y_LIVE_PATH, y)
    print(f"[saved] X_live: {X.shape}, y_live: {y.shape} -> {LIVE_DIR}")


def _print_stats(labels: List[int]) -> None:
    counts: Dict[int, int] = defaultdict(int)
    for lbl in labels:
        counts[int(lbl)] += 1
    print("\n" + "=" * 50)
    print("  STATISTIK DATASET LIVE")
    print("=" * 50)
    for idx in sorted(counts.keys()):
        name = IDX_TO_LABEL[idx]
        print(f"    {name:>14} : {counts[idx]:4d} sample")
    print(f"{'TOTAL':>18} : {len(labels):4d} sample")
    print("=" * 50 + "\n")


def _countdown(cap: cv2.VideoCapture, label: str, seconds: int = COUNTDOWN_SECONDS) -> bool:
    for s in range(seconds, 0, -1):
        end = time.time() + 1
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                return False
            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame, f"{label}  REC in {s}...",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3,
            )
            cv2.imshow("BISINDO Video Recorder", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
    return True


def _make_video_writer(path: Path, frame: np.ndarray, fps: float = 20.0) -> cv2.VideoWriter:
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (w, h))


def _record_clip(
    cap: cv2.VideoCapture,
    landmarker: HandLandmarkerWrapper,
    label: str,
    label_idx: int,
    video_path: Path,
    frames_per_clip: int = FRAMES_PER_CLIP,
    t_start: float = 0.0,
) -> Optional[np.ndarray]:
    """Rekam satu klip video + ekstraksi landmark simultan.
    Return (T, F) atau None bila gagal."""
    if not _countdown(cap, label):
        return None

    writer: Optional[cv2.VideoWriter] = None
    lm_frames: List[np.ndarray] = []

    try:
        for t in range(frames_per_clip):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            if writer is None:
                writer = _make_video_writer(video_path, frame)
            writer.write(frame)

            ts_ms = int((time.time() - t_start) * 1000)
            hands = landmarker.detect_bgr(frame, timestamp_ms=ts_ms)
            normed = normalize_two_hands(hands, max_hands=MAX_HANDS)
            lm_frames.append(flatten_frame(normed))

            disp = frame.copy()
            draw_hand_landmarks(disp, hands)
            cv2.putText(
                disp, f"REC {label}  {t+1}/{frames_per_clip}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
            )
            cv2.imshow("BISINDO Video Recorder", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None
    finally:
        if writer is not None:
            writer.release()

    if not lm_frames:
        return None

    arr = np.stack(lm_frames, axis=0)
    return resample_from_config(arr, target_len=SEQUENCE_LENGTH)


def _draw_hud(
    frame: np.ndarray, label: str, label_idx: int,
    session: int, total: int, hand_ok: bool, fps: float,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.putText(
        frame, f"[{label_idx+1}/{NUM_LABELS}] {label}",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    cv2.putText(
        frame, f"Session: {session}  Total: {total}  {fps:.0f} FPS",
        (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )
    hc = (0, 255, 0) if hand_ok else (0, 0, 255)
    cv2.putText(
        frame, "HAND OK" if hand_ok else "NO HAND",
        (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hc, 2,
    )
    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(
        frame, "SPACE=rec  n/p=label  s=save  d=stats  r=undo  q=quit",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
    )


def main() -> None:
    landmarker = HandLandmarkerWrapper(running_mode="video")
    seqs, labels = _load_existing()
    total = len(seqs)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Webcam tidak dapat dibuka.")
        landmarker.close()
        return

    label_idx = 0
    session = 0
    last_video_path: Optional[Path] = None
    t_start = time.time()
    t_prev = time.time()
    fps = 0.0

    print("\n" + "=" * 60)
    print("  BISINDO VIDEO RECORDER (.mp4 + landmark .npy)")
    print("=" * 60)
    print(f"  Output video     : {VIDEO_RAW_DIR}/<LABEL>/clip_XXXX.mp4")
    print(f"  Output landmark  : {LIVE_DIR}/X_live.npy")
    print(f"  Frames per clip  : {FRAMES_PER_CLIP}")
    print(f"  Resample -> T    : {SEQUENCE_LENGTH}")
    print("=" * 60)
    print("  SPACE=rec  n/p=label  s=save  d=stats  r=undo  q=quit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            ts_ms = int((time.time() - t_start) * 1000)
            hands = landmarker.detect_bgr(frame, timestamp_ms=ts_ms)
            draw_hand_landmarks(frame, hands)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.8 * fps + 0.2 * (1.0 / dt)

            _draw_hud(
                frame, ALL_CLASSES[label_idx], label_idx,
                session, total, bool(hands), fps,
            )
            cv2.imshow("BISINDO Video Recorder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                _save_dataset(seqs, labels)
                break
            elif key == ord("n"):
                label_idx = (label_idx + 1) % NUM_LABELS
                print(f"[label] -> {ALL_CLASSES[label_idx]}")
            elif key == ord("p"):
                label_idx = (label_idx - 1) % NUM_LABELS
                print(f"[label] -> {ALL_CLASSES[label_idx]}")
            elif key == ord("s"):
                _save_dataset(seqs, labels)
            elif key == ord("d"):
                _print_stats(labels)
            elif key == ord("r") and last_video_path is not None:
                # Undo sample terakhir
                try:
                    last_video_path.unlink(missing_ok=True)
                except Exception:
                    pass
                if seqs:
                    seqs.pop()
                    labels.pop()
                    total = max(0, total - 1)
                    session = max(0, session - 1)
                print(f"[undo] Klip terakhir dihapus: {last_video_path}")
                last_video_path = None
            elif key == ord(" "):
                label = ALL_CLASSES[label_idx]
                class_video_dir = VIDEO_RAW_DIR / label
                class_video_dir.mkdir(parents=True, exist_ok=True)
                existing = len(list(class_video_dir.glob("clip_*.mp4")))
                video_path = class_video_dir / f"clip_{existing:04d}.mp4"

                seq = _record_clip(
                    cap, landmarker, label, label_idx,
                    video_path=video_path, t_start=t_start,
                )
                if seq is not None:
                    seqs.append(seq)
                    labels.append(LABEL_TO_IDX[label])
                    total += 1
                    session += 1
                    last_video_path = video_path
                    print(f"  [+] {label} clip #{existing:04d} saved "
                          f"(total={total}, session={session})")
                else:
                    if video_path.exists():
                        video_path.unlink(missing_ok=True)
                    print("  [warn] Rekaman dibatalkan.")
    except KeyboardInterrupt:
        print("\n[interrupt] Menyimpan data...")
        _save_dataset(seqs, labels)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("[done] Recorder selesai.")


if __name__ == "__main__":
    main()
