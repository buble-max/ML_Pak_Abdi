"""
Orchestrator: menjalankan seluruh pipeline BISINDO end-to-end.

Tahapan (dapat dilewati via flag):
    1. --download      : unduh dataset BISINDO alfabet dari GitHub
    2. --record        : rekam 5 gesture kata sebagai klip video
                         (HALO, MAKAN, MINUM, TERIMA_KASIH, TOLONG)
    3. --record-live   : rekam dataset manual landmark dari webcam → .npy
                         (mendukung semua kelas, append mode)
    4. --preprocess    : ekstraksi landmark + sliding window + .npy
    5. --augment       : data augmentation + class balancing
    6. --train         : training CNN+LSTM dan ekspor .h5
    7. --all           : jalankan download→preprocess→augment→train
                         (tidak termasuk tahap --record* yang interaktif)

Contoh:
    python run_pipeline.py --all
    python run_pipeline.py --download --preprocess --augment --train
    python run_pipeline.py --record          # rekam klip gesture kata
    python run_pipeline.py --record-live     # rekam landmark langsung → .npy
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
                   help="rekam klip video gesture kata")
    p.add_argument("--record-live", action="store_true", dest="record_live",
                   help="rekam landmark langsung dari webcam → .npy")
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--all", action="store_true",
                   help="jalankan download→preprocess→augment→train")
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
    if args.all or args.preprocess:
        _run("preprocessing.landmark_extractor")
    if args.all or args.augment:
        _run("augmentation.augment")
    if args.all or args.train:
        _run("training.train")

    print("\n[done] Pipeline selesai.")


if __name__ == "__main__":
    main()
