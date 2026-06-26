"""
Auto-download dataset Kaggle BISINDO 40 Kata MP4.

Alur end-to-end:
  1. Autentikasi kaggle.json (Colab: files.upload(); lokal: manual).
  2. Download ZIP via Kaggle API.
  3. Ekstrak + deteksi root <LABEL>/*.mp4.
  4. Salin ke dataset/raw_videos/<LABEL>/.
  5. Validasi (min N video per kelas).
  6. Print statistik.

CLI: python -m dataset.download_kaggle [--slug ...] [--force]
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    DATASET_DIR,
    KAGGLE_DATASET_SLUG,
    MIN_VIDEOS_PER_CLASS,
    VIDEO_EXTENSIONS,
    VIDEO_RAW_DIR,
)


def _kaggle_json_path() -> Path:
    env = os.environ.get("KAGGLE_CONFIG_DIR")
    if env:
        return Path(env).expanduser() / "kaggle.json"
    return Path.home() / ".kaggle" / "kaggle.json"


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def ensure_kaggle_auth(force_upload: bool = False) -> Path:
    """Pastikan kaggle.json tersedia. Di Colab minta upload jika belum ada."""
    token = _kaggle_json_path()
    if token.exists() and not force_upload:
        token.chmod(stat.S_IRUSR | stat.S_IWUSR)
        print(f"[auth] kaggle.json ditemukan: {token}")
    else:
        if _is_colab():
            from google.colab import files  # type: ignore
            print("[auth] Upload kaggle.json (Account -> Create New API Token)")
            uploaded = files.upload()
            name = next((k for k in uploaded if "kaggle" in k.lower()), None)
            if name is None:
                raise FileNotFoundError("kaggle.json tidak terdeteksi.")
            token.parent.mkdir(parents=True, exist_ok=True)
            token.write_bytes(uploaded[name])
            token.chmod(stat.S_IRUSR | stat.S_IWUSR)
            print(f"[auth] Tersimpan: {token}")
        else:
            raise FileNotFoundError(
                f"kaggle.json tidak ditemukan di {token}.\n"
                "Download dari https://www.kaggle.com/settings/account "
                "-> Create New API Token."
            )
    os.environ["KAGGLE_CONFIG_DIR"] = str(token.parent)
    # Validasi
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    api = KaggleApi()
    api.authenticate()
    print("[auth] Token valid.")
    return token


def _video_exts() -> set:
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in VIDEO_EXTENSIONS}


def _find_label_root(base: Path) -> Optional[Path]:
    """Cari folder yang punya subfolder berisi video."""
    exts = _video_exts()
    best, best_score = None, 0
    for d in base.rglob("*"):
        if not d.is_dir():
            continue
        subs = [s for s in d.iterdir() if s.is_dir()]
        score = sum(1 for s in subs if any(
            f.suffix.lower() in exts for f in s.iterdir() if f.is_file()
        ))
        if score > best_score:
            best, best_score = d, score
    return best


def validate_dataset(root: Path, min_vids: int = MIN_VIDEOS_PER_CLASS) -> dict:
    """Return {label: count}. Raise jika gagal."""
    if not root.exists():
        raise FileNotFoundError(f"Folder tidak ada: {root}")
    exts = _video_exts()
    stats: dict = {}
    problems: List[str] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        n = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in exts)
        stats[d.name] = n
        if n < min_vids:
            problems.append(f"  {d.name}: {n} (min {min_vids})")
    if not stats:
        raise RuntimeError(f"Tidak ada folder label di {root}")
    if problems:
        raise RuntimeError("Validasi GAGAL:\n" + "\n".join(problems))
    return stats


def ensure_kaggle_bisindo_videos(
    slug: str = KAGGLE_DATASET_SLUG,
    target: Path = VIDEO_RAW_DIR,
    min_vids: int = MIN_VIDEOS_PER_CLASS,
    force_auth: bool = False,
    skip_if_exists: bool = True,
) -> Path:
    """End-to-end: auth -> download -> extract -> validate."""
    if skip_if_exists:
        try:
            stats = validate_dataset(target, min_vids)
            print(f"[skip] Dataset sudah valid ({sum(stats.values())} video, "
                  f"{len(stats)} kelas)")
            return target
        except (FileNotFoundError, RuntimeError):
            pass

    ensure_kaggle_auth(force_upload=force_auth)

    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    api = KaggleApi()
    api.authenticate()

    staging = DATASET_DIR / "_kaggle_tmp"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    print(f"[download] {slug} ...")
    api.dataset_download_files(slug, path=str(staging), unzip=False, quiet=False)

    zips = sorted(staging.glob("*.zip"))
    if not zips:
        raise RuntimeError("Download gagal: tidak ada ZIP.")
    zip_path = zips[-1]
    print(f"[download] OK: {zip_path.name} ({zip_path.stat().st_size/1e6:.1f} MB)")

    print(f"[extract] {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging)
    zip_path.unlink(missing_ok=True)

    src = _find_label_root(staging)
    if src is None:
        raise RuntimeError(f"Tidak menemukan <LABEL>/*.mp4 di {staging}")
    print(f"[info] Root dataset: {src}")

    exts = _video_exts()
    target.mkdir(parents=True, exist_ok=True)
    total = 0
    for label_dir in sorted(src.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name.strip().replace(" ", "_").replace("-", "_").upper()
        dst = target / label
        dst.mkdir(parents=True, exist_ok=True)
        for f in label_dir.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                shutil.copy2(f, dst / f.name)
                total += 1
    print(f"[info] {total} video -> {target}")

    shutil.rmtree(staging, ignore_errors=True)

    stats = validate_dataset(target, min_vids)
    print("\n" + "=" * 50)
    print("  STATISTIK DATASET KAGGLE")
    print("=" * 50)
    for label, n in sorted(stats.items()):
        print(f"    {label:>20} : {n:4d} video")
    print(f"    {'TOTAL':>20} : {sum(stats.values()):4d} ({len(stats)} kelas)")
    print("=" * 50 + "\n")
    return target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slug", default=KAGGLE_DATASET_SLUG)
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-auth", action="store_true")
    args = p.parse_args()
    ensure_kaggle_bisindo_videos(
        slug=args.slug,
        force_auth=args.force_auth,
        skip_if_exists=not args.force,
    )


if __name__ == "__main__":
    main()
