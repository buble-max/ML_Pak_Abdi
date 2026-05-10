"""
Orchestrator: menjalankan seluruh pipeline BISINDO end-to-end.

Tahapan (dapat dilewati via flag):
    1. --download         : unduh dataset BISINDO alfabet dari GitHub
    2. --record            : rekam 5 gesture kata (HALO, MAKAN, MINUM, TERIMA_KASIH, TOLONG)
    3. --preprocess        : ekstraksi landmark + sliding window + .npy
    4. --augment           : data augmentation + class balancing
    5. --train             : training CNN+LSTM dan ekspor .h5
    6. --all               : jalankan semua kecuali --record (interaktif)

Contoh:
    python run_pipeline.py --all
    python run_pipeline.py --download --preprocess --augment --train
    python run_pipeline.py --record          # hanya merekam gesture kata
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
    p.add_argument("--record", action="store_true")
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
    if args.all or args.preprocess:
        _run("preprocessing.landmark_extractor")
    if args.all or args.augment:
        _run("augmentation.augment")
    if args.all or args.train:
        _run("training.train")

    print("\n[done] Pipeline selesai.")


if __name__ == "__main__":
    main()
