"""
Orchestrator: menjalankan seluruh pipeline BISINDO end-to-end.

Tahapan (dapat dilewati via flag):
    1. --download       : unduh dataset BISINDO alfabet dari GitHub
    2. --record         : rekam klip gesture kata (frame .jpg per clip)
    3. --record-live    : rekam landmark langsung dari webcam -> .npy
                          (mendukung semua kelas, append mode)
    4. --record-video   : rekam gesture sebagai video .mp4 + landmark .npy
                          (full video-based temporal pipeline)
    5. --preprocess     : video/klip/gambar -> landmark -> .npy
    6. --augment        : spatial + temporal augmentation + class balancing
    7. --train          : training CNN+LSTM dan ekspor .h5
    8. --all            : download -> preprocess -> augment -> train
                          (tidak termasuk tahap --record* yang interaktif)

Contoh:
    python run_pipeline.py --all
    python run_pipeline.py --record-video      # rekam gesture .mp4 + .npy
    python run_pipeline.py --preprocess --augment --train
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(module_or_script: str) -> None:
    print(f"\n{'='*60}\n>>> {module_or_script}\n{'='*60}")
    cmd = [sys.executable, "-m", module_or_script] if "/" not in module_or_script \
        else [sys.executable, module_or_script]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument("--record", action="store_true",
                   help="rekam klip frame .jpg per clip")
    p.add_argument("--record-live", action="store_true", dest="record_live",
                   help="rekam landmark langsung dari webcam -> .npy")
    p.add_argument("--record-video", action="store_true", dest="record_video",
                   help="rekam video .mp4 + landmark .npy")
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--all", action="store_true",
                   help="jalankan download -> preprocess -> augment -> train")
    args = p.parse_args()

    if not any(vars(args).values()):
        p.print_help()
        return

    if args.all or args.download:
        _run("dataset.download_dataset")
    if args.record:
        _run("dataset.record_word_gestures")
    if args.record_live:
        _run("dataset.record_landmarks_live")
    if args.record_video:
        _run("dataset.record_video_gestures")
    if args.all or args.preprocess:
        _run("preprocessing.landmark_extractor")
    if args.all or args.augment:
        _run("augmentation.augment")
    if args.all or args.train:
        _run("training.train")

    print("\n[done] Pipeline selesai.")


if __name__ == "__main__":
    main()
