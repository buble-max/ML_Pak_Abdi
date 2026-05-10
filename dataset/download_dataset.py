"""
Mengunduh dataset BISINDO (Indonesian Sign Language Hand Sign Detection Dataset).

Dataset diharapkan tersusun dalam struktur:
    dataset/raw/<LABEL>/<image>.jpg

Script ini mencoba beberapa strategi:
1. `git clone` repository sumber (fast path).
2. Fallback: download ZIP archive.
3. Jika struktur sumber berbeda, script akan memindai subfolder per huruf/label
   (A..Z) lalu menyalinnya ke layout standar di `dataset/raw/`.
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

from config import ALPHABET_CLASSES, BISINDO_REPO_URL, RAW_DIR  # noqa: E402


def _git_clone(url: str, dest: Path) -> bool:
    if dest.exists():
        shutil.rmtree(dest)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[warn] git clone gagal: {e}")
        return False


def _download_zip(url: str, dest_zip: Path) -> bool:
    try:
        urlretrieve(url, dest_zip)
        return True
    except Exception as e:
        print(f"[warn] download zip gagal: {e}")
        return False


def _flatten_to_raw(source_root: Path, raw_dir: Path) -> int:
    """
    Pindai `source_root` secara rekursif. Jika menemukan folder dengan nama
    satu karakter alfabet (A-Z), salin isinya ke `raw_dir/<LABEL>/`.
    """
    count = 0
    for subdir in source_root.rglob("*"):
        if not subdir.is_dir():
            continue
        name = subdir.name.strip().upper()
        if name in ALPHABET_CLASSES:
            target = raw_dir / name
            target.mkdir(parents=True, exist_ok=True)
            for img in subdir.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    shutil.copy2(img, target / img.name)
                    count += 1
    return count


def download_bisindo(repo_url: Optional[str] = None, raw_dir: Path = RAW_DIR) -> None:
    repo_url = repo_url or BISINDO_REPO_URL
    raw_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = raw_dir.parent / "_tmp_bisindo"
    print(f"[info] Mengunduh dataset BISINDO dari: {repo_url}")

    ok = _git_clone(repo_url, tmp_dir)
    if not ok:
        zip_url = repo_url.rstrip("/") + "/archive/refs/heads/main.zip"
        zip_path = raw_dir.parent / "_bisindo.zip"
        if _download_zip(zip_url, zip_path):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
            zip_path.unlink(missing_ok=True)
        else:
            print(
                "[error] Tidak dapat mengunduh dataset secara otomatis.\n"
                "        Silakan clone manual ke folder dataset/raw/<LABEL>/."
            )
            return

    copied = _flatten_to_raw(tmp_dir, raw_dir)
    print(f"[info] Disalin {copied} gambar ke {raw_dir}")

    # Bersihkan
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Ringkasan
    for c in ALPHABET_CLASSES:
        d = raw_dir / c
        n = len(list(d.glob("*"))) if d.exists() else 0
        print(f"  {c}: {n} gambar")


if __name__ == "__main__":
    download_bisindo()
