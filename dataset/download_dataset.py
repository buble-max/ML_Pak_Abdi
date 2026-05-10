"""
Mengunduh dataset BISINDO (Indonesian Sign Language BISINDO Hand Sign Detection
Dataset) dari repository GitHub:

    https://github.com/rhiosutoyo/
        Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset

Dataset TIDAK disediakan sebagai arsip .zip, melainkan langsung tersusun
sebagai folder di `collectedimages/` pada branch `master`. Setiap subfolder
merepresentasikan label gesture (umumnya huruf alfabet BISINDO A-Z) dan
berisi file gambar .jpg / .png.

Strategi pengambilan:
1. `git clone` repository secara penuh (shallow depth=1 untuk efisiensi).
   Jika tidak bisa clone, fallback download ZIP branch master.
2. Ambil folder `collectedimages/` sebagai sumber dataset utama.
3. Salin setiap subfolder label ke layout standar proyek:
       dataset/raw/<LABEL>/<image>.jpg

Setelah tahap ini, `preprocessing/landmark_extractor.py` dapat memproses
`dataset/raw/` seperti biasa (MediaPipe → normalisasi → sliding window → .npy).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

# Supaya bisa dijalankan langsung: `python dataset/download_dataset.py`
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    ALPHABET_CLASSES,
    BISINDO_DATASET_SUBDIR,
    BISINDO_REPO_BRANCH,
    BISINDO_REPO_URL,
    RAW_DIR,
)


def _git_clone(url: str, dest: Path, branch: str) -> bool:
    if dest.exists():
        shutil.rmtree(dest)
    try:
        subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--branch", branch,
                url, str(dest),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # fallback tanpa --branch (mis. kalau nama branch default berbeda)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                check=True, capture_output=True,
            )
            return True
        except Exception:
            print(f"[warn] git clone gagal: {e}")
            return False


def _download_zip(url: str, dest_zip: Path) -> bool:
    try:
        urlretrieve(url, dest_zip)
        return True
    except Exception as e:
        print(f"[warn] download zip gagal: {e}")
        return False


def _copy_dataset_from_subdir(
    source_root: Path,
    raw_dir: Path,
    subdir: str = BISINDO_DATASET_SUBDIR,
) -> int:
    """
    Ambil folder `<source_root>/**/<subdir>` (mis. `collectedimages/`) sebagai
    sumber dataset. Salin setiap subfolder label ke `raw_dir/<LABEL>/`.

    Label dinormalisasi menjadi uppercase. Jika label tidak termasuk alfabet
    A-Z, folder tetap disalin (mendukung label tambahan di masa depan).
    """
    # Cari folder `collectedimages` di mana pun di dalam tree (repo ZIP sering
    # ter-ekstrak di bawah folder bernama `<repo>-<branch>/`).
    candidates = [p for p in source_root.rglob(subdir) if p.is_dir()]
    if not candidates:
        print(f"[error] Folder '{subdir}/' tidak ditemukan di {source_root}")
        return 0

    src = candidates[0]
    print(f"[info] Menggunakan sumber dataset: {src}")

    count = 0
    for label_dir in sorted(src.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name.strip().upper()
        target = raw_dir / label
        target.mkdir(parents=True, exist_ok=True)

        for img in label_dir.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                shutil.copy2(img, target / img.name)
                count += 1
    return count


def download_bisindo(
    repo_url: Optional[str] = None,
    branch: str = BISINDO_REPO_BRANCH,
    subdir: str = BISINDO_DATASET_SUBDIR,
    raw_dir: Path = RAW_DIR,
) -> None:
    repo_url = repo_url or BISINDO_REPO_URL
    raw_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = raw_dir.parent / "_tmp_bisindo"
    print(f"[info] Mengunduh dataset BISINDO dari: {repo_url} (branch={branch})")
    print(f"[info] Sumber dataset: {subdir}/")

    # --- 1. Coba git clone ---
    ok = _git_clone(repo_url, tmp_dir, branch)

    # --- 2. Fallback: download ZIP ---
    if not ok:
        zip_url = repo_url.rstrip("/") + f"/archive/refs/heads/{branch}.zip"
        zip_path = raw_dir.parent / "_bisindo.zip"
        print(f"[info] Mencoba fallback ZIP: {zip_url}")
        if _download_zip(zip_url, zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
            zip_path.unlink(missing_ok=True)
        else:
            print(
                "[error] Tidak dapat mengunduh dataset secara otomatis.\n"
                f"        Silakan clone manual ke {tmp_dir}, atau salin\n"
                f"        folder {subdir}/<LABEL>/ ke dataset/raw/<LABEL>/."
            )
            return

    # --- 3. Salin dari collectedimages/ ke dataset/raw/ ---
    copied = _copy_dataset_from_subdir(tmp_dir, raw_dir, subdir=subdir)
    print(f"[info] Disalin {copied} gambar ke {raw_dir}")

    # Bersihkan working copy
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- 4. Ringkasan per kelas ---
    print("\n[summary] Jumlah gambar per kelas:")
    for label_dir in sorted(raw_dir.iterdir()):
        if label_dir.is_dir():
            n = sum(
                1 for p in label_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            marker = "   " if label_dir.name in ALPHABET_CLASSES else "  +"
            print(f"  {marker} {label_dir.name:>10}: {n} gambar")


if __name__ == "__main__":
    download_bisindo()
