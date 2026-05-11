"""
Pipeline preprocessing utama (video-based temporal training).

Mendukung tiga sumber dataset dengan struktur berbasis folder label:

  1. Gambar statis      : dataset/raw/<LABEL>/*.jpg
  2. Klip frame folder  : dataset/raw_words/<LABEL>/clip_*/frame_*.jpg
                          dataset/raw_numbers/<LABEL>/clip_*/frame_*.jpg
  3. Video utuh         : dataset/raw_videos/<LABEL>/*.mp4|.avi|.mov|...

Alur tiap sample:
  a. Dekode video atau baca frame klip.
  b. Temporal sampling (uniform/stride/random/adaptive) -> list frame.
  c. MediaPipe Tasks API HandLandmarker -> 21 titik (x,y,z) per tangan.
  d. Normalisasi (wrist-centered + scale).
  e. Resample ke panjang tetap T (interpolate/pad_last/pad_zero).
  f. Simpan ke (N, T, F) .npy.

Label dibaca otomatis dari nama folder (`AUTO_DETECT_LABELS`), tidak
perlu edit config saat menambah gesture baru.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    FEATURES_PER_FRAME,
    FRAME_STRIDE,
    MAX_HANDS,
    NUMBER_RAW_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SEQUENCE_LENGTH,
    SEQUENCE_OVERLAP,
    TEMPORAL_SAMPLING,
    VIDEO_RAW_DIR,
    VIDEO_SLIDING_WINDOW,
    WINDOW_STRIDE,
    WORD_RAW_DIR,
)
from preprocessing.mp_hand_landmarker import HandLandmarkerWrapper  # noqa: E402
from preprocessing.normalizer import flatten_frame, normalize_two_hands  # noqa: E402
from preprocessing.temporal_resampler import resample_from_config  # noqa: E402
from preprocessing.video_ingestion import (  # noqa: E402
    list_videos, probe_video, read_video_frames,
)
from utils.labels import build_label_maps, save_labels  # noqa: E402


# ---------------------------------------------------------------
# Landmark extraction per frame
# ---------------------------------------------------------------
def _get_landmarker(running_mode: str = "image") -> HandLandmarkerWrapper:
    return HandLandmarkerWrapper(running_mode=running_mode)


def _compute_stride(seq_len: int, overlap: float, fallback_stride: int) -> int:
    if overlap > 0:
        stride = max(1, int(round(seq_len * (1.0 - overlap))))
    else:
        stride = max(1, fallback_stride)
    return stride


def extract_landmarks_from_image(
    image_bgr: np.ndarray,
    landmarker: HandLandmarkerWrapper,
) -> np.ndarray:
    """BGR frame -> (FEATURES_PER_FRAME,) vector ternormalisasi."""
    hand_arrays: List[np.ndarray] = landmarker.detect_bgr(image_bgr)
    normed = normalize_two_hands(hand_arrays, max_hands=MAX_HANDS)
    return flatten_frame(normed)


def _frames_to_landmark_array(
    frames: List[np.ndarray],
    landmarker: HandLandmarkerWrapper,
) -> np.ndarray:
    """List[BGR frame] -> (M, F) landmark array."""
    feats = [extract_landmarks_from_image(f, landmarker) for f in frames if f is not None]
    if not feats:
        return np.zeros((0, FEATURES_PER_FRAME), dtype=np.float32)
    return np.stack(feats, axis=0).astype(np.float32)


# ---------------------------------------------------------------
# Sources -> (T, F) sequences
# ---------------------------------------------------------------
def static_images_to_sequences(
    image_paths: List[Path],
    landmarker: HandLandmarkerWrapper,
    seq_len: int = SEQUENCE_LENGTH,
    stride: Optional[int] = None,
    frame_stride: int = FRAME_STRIDE,
) -> np.ndarray:
    """
    Gambar statis (folder A/, B/, ...) -> banyak sequence via sliding window.
    """
    if stride is None:
        stride = _compute_stride(seq_len, SEQUENCE_OVERLAP, WINDOW_STRIDE)

    if frame_stride > 1:
        image_paths = image_paths[::frame_stride]

    feats: List[np.ndarray] = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        feats.append(extract_landmarks_from_image(img, landmarker))

    if not feats:
        return np.zeros((0, seq_len, FEATURES_PER_FRAME), dtype=np.float32)

    arr = np.stack(feats, axis=0)
    M = arr.shape[0]

    if M < seq_len:
        rng = np.random.default_rng(42)
        reps = int(np.ceil(seq_len * 2 / M))
        arr = np.tile(arr, (reps, 1))
        arr = arr + rng.normal(0, 1e-3, size=arr.shape).astype(np.float32)
        M = arr.shape[0]

    sequences = []
    for start in range(0, M - seq_len + 1, stride):
        sequences.append(arr[start : start + seq_len])
    if not sequences:
        sequences.append(arr[:seq_len])
    return np.stack(sequences, axis=0).astype(np.float32)


def clip_to_sequence(
    clip_dir: Path,
    landmarker: HandLandmarkerWrapper,
    seq_len: int = SEQUENCE_LENGTH,
    frame_stride: int = FRAME_STRIDE,
) -> Optional[np.ndarray]:
    """
    Folder klip frame (frame_*.jpg) -> satu sequence (T, F).
    Menggunakan temporal_resampler untuk menormalkan panjang.
    """
    frames = sorted(clip_dir.glob("frame_*.jpg"))
    if not frames:
        return None
    if frame_stride > 1:
        frames = frames[::frame_stride]

    feats: List[np.ndarray] = []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        feats.append(extract_landmarks_from_image(img, landmarker))
    if not feats:
        return None

    arr = np.stack(feats, axis=0)
    return resample_from_config(arr, target_len=seq_len)


def video_to_sequence(
    video_path: Path,
    landmarker: HandLandmarkerWrapper,
    seq_len: int = SEQUENCE_LENGTH,
    frame_stride: int = FRAME_STRIDE,
    mode: str = TEMPORAL_SAMPLING,
) -> Optional[np.ndarray]:
    """
    Video utuh -> satu sequence (T, F).
    1 video = 1 training sample (re-sampled ke T frame).
    """
    try:
        frames = read_video_frames(
            video_path, seq_len=seq_len, mode=mode,  # type: ignore[arg-type]
            frame_stride=frame_stride,
        )
    except IOError as e:
        print(f"[warn] {video_path.name}: {e}")
        return None
    if not frames:
        return None

    arr = _frames_to_landmark_array(frames, landmarker)
    if arr.shape[0] == 0:
        return None
    return resample_from_config(arr, target_len=seq_len)


def video_to_sequences_sliding(
    video_path: Path,
    landmarker: HandLandmarkerWrapper,
    seq_len: int = SEQUENCE_LENGTH,
    frame_stride: int = FRAME_STRIDE,
    stride: Optional[int] = None,
) -> np.ndarray:
    """
    Video panjang (kalimat) -> banyak sequence via sliding window di ruang landmark.
    Berguna untuk gesture kalimat / video panjang.
    """
    if stride is None:
        stride = _compute_stride(seq_len, SEQUENCE_OVERLAP, WINDOW_STRIDE)

    # Ambil SEMUA frame (stride saja, tidak sampling ke T) supaya sliding
    # window di ruang temporal masih bermakna.
    try:
        n_total, _, _ = probe_video(video_path)
    except IOError as e:
        print(f"[warn] {video_path.name}: {e}")
        return np.zeros((0, seq_len, FEATURES_PER_FRAME), dtype=np.float32)

    # stream seluruh frame dengan FRAME_STRIDE
    from preprocessing.video_ingestion import iter_video_frames
    feats: List[np.ndarray] = []
    for i, frame in enumerate(iter_video_frames(video_path)):
        if frame_stride > 1 and (i % frame_stride) != 0:
            continue
        feats.append(extract_landmarks_from_image(frame, landmarker))
    if not feats:
        return np.zeros((0, seq_len, FEATURES_PER_FRAME), dtype=np.float32)

    arr = np.stack(feats, axis=0)
    M = arr.shape[0]
    if M < seq_len:
        return resample_from_config(arr, target_len=seq_len)[None, ...]

    sequences = []
    for start in range(0, M - seq_len + 1, stride):
        sequences.append(arr[start : start + seq_len])
    return np.stack(sequences, axis=0).astype(np.float32)


# ---------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------
def build_dataset(
    raw_dir: Path = RAW_DIR,
    word_dir: Path = WORD_RAW_DIR,
    number_dir: Path = NUMBER_RAW_DIR,
    video_dir: Path = VIDEO_RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    """
    Pipeline utama: scan semua folder dataset -> preprocessing -> .npy.
    """
    label_to_idx, _ = build_label_maps(auto_detect=True)
    all_classes = list(label_to_idx.keys())
    save_labels(classes=all_classes)
    print(f"[info] Total kelas terdeteksi: {len(all_classes)}")

    X: List[np.ndarray] = []
    y: List[int] = []

    stride = _compute_stride(SEQUENCE_LENGTH, SEQUENCE_OVERLAP, WINDOW_STRIDE)

    landmarker = _get_landmarker(running_mode="image")
    try:
        # ---- 1) Gambar statis ----
        if raw_dir.exists():
            for class_dir in sorted(raw_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                label = class_dir.name.strip().upper()
                if label not in label_to_idx:
                    continue
                imgs = sorted([p for p in class_dir.iterdir()
                               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
                if not imgs:
                    continue
                seqs = static_images_to_sequences(
                    imgs, landmarker,
                    seq_len=SEQUENCE_LENGTH, stride=stride,
                    frame_stride=FRAME_STRIDE,
                )
                if seqs.shape[0] > 0:
                    X.append(seqs)
                    y.extend([label_to_idx[label]] * seqs.shape[0])
                    print(f"  [img ] {label:>14}: {len(imgs):4d} img -> {seqs.shape[0]:4d} seq")

        # ---- 2) Klip frame folder (legacy: words + numbers) ----
        for clip_root in (word_dir, number_dir):
            if not clip_root.exists():
                continue
            for class_dir in sorted(clip_root.iterdir()):
                if not class_dir.is_dir():
                    continue
                label = class_dir.name.strip().upper()
                if label not in label_to_idx:
                    continue
                clips = sorted([d for d in class_dir.iterdir() if d.is_dir()])
                class_seqs = []
                for clip in tqdm(clips, desc=f"  [clip] {label}", leave=False):
                    seq = clip_to_sequence(
                        clip, landmarker,
                        seq_len=SEQUENCE_LENGTH, frame_stride=FRAME_STRIDE,
                    )
                    if seq is not None:
                        class_seqs.append(seq)
                if class_seqs:
                    seqs = np.stack(class_seqs, axis=0)
                    X.append(seqs)
                    y.extend([label_to_idx[label]] * seqs.shape[0])
                    print(f"  [clip] {label:>14}: {len(clips):4d} clip -> {seqs.shape[0]:4d} seq")

        # ---- 3) Video utuh (.mp4/.avi/...) ----
        if video_dir.exists():
            for class_dir in sorted(video_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                label = class_dir.name.strip().upper()
                if label not in label_to_idx:
                    continue
                videos = list_videos(class_dir)
                if not videos:
                    continue
                class_seqs: List[np.ndarray] = []
                for v in tqdm(videos, desc=f"  [vid ] {label}", leave=False):
                    if VIDEO_SLIDING_WINDOW:
                        seqs = video_to_sequences_sliding(
                            v, landmarker,
                            seq_len=SEQUENCE_LENGTH, frame_stride=FRAME_STRIDE,
                            stride=stride,
                        )
                        for s in seqs:
                            class_seqs.append(s)
                    else:
                        seq = video_to_sequence(
                            v, landmarker,
                            seq_len=SEQUENCE_LENGTH, frame_stride=FRAME_STRIDE,
                            mode=TEMPORAL_SAMPLING,
                        )
                        if seq is not None:
                            class_seqs.append(seq)
                if class_seqs:
                    seqs = np.stack(class_seqs, axis=0)
                    X.append(seqs)
                    y.extend([label_to_idx[label]] * seqs.shape[0])
                    print(f"  [vid ] {label:>14}: {len(videos):4d} video -> {seqs.shape[0]:4d} seq")
    finally:
        landmarker.close()

    # ---- 4) Live (dataset manual webcam -> .npy) ----
    live_dir = out_dir / "live"
    x_live_path = live_dir / "X_live.npy"
    y_live_path = live_dir / "y_live.npy"
    if x_live_path.exists() and y_live_path.exists():
        X_live = np.load(x_live_path)
        y_live = np.load(y_live_path)
        if X_live.shape[1:] == (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
            X.append(X_live)
            y.extend(y_live.tolist())
            print(f"  [live] {'LIVE':>14}: {X_live.shape[0]:4d} seq (webcam recorder)")
        else:
            print(f"[warn] X_live shape {X_live.shape} tidak kompatibel, dilewati.")

    if not X:
        print("[error] Tidak ada data terbangun. Cek:")
        print("        - dataset/raw/<LABEL>/*.jpg")
        print("        - dataset/raw_words/<LABEL>/clip_*/frame_*.jpg")
        print("        - dataset/raw_videos/<LABEL>/*.mp4")
        print("        - atau jalankan: python -m dataset.record_landmarks_live")
        return

    X_arr = np.concatenate(X, axis=0).astype(np.float32)
    y_arr = np.array(y, dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X_arr)
    np.save(out_dir / "y.npy", y_arr)
    print(f"\n[done] X: {X_arr.shape}  y: {y_arr.shape}")
    print(f"[done] Disimpan ke {out_dir}")
    print(f"[done] Config: T={SEQUENCE_LENGTH}, stride={stride}, "
          f"frame_stride={FRAME_STRIDE}, sampling={TEMPORAL_SAMPLING}, "
          f"sliding_video={VIDEO_SLIDING_WINDOW}")


if __name__ == "__main__":
    build_dataset()
