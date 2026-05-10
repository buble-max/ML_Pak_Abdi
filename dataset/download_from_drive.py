"""
Unduh dataset (ZIP) dari Google Drive menggunakan `gdown`, ekstrak otomatis,
bersihkan file ZIP, lalu validasi struktur folder hasil ekstraksi.

Alur pemakaian utama (dari Google Colab atau lokal):

    from dataset.download_from_drive import download_and_extract

    download_and_extract(
        file_id_or_url="https://drive.google.com/file/d/<FILE_ID>/view",
        extract_to="dataset",                    # relatif terhadap repo root
        expected_subdirs=["raw_words"],          # validasi setelah ekstrak
    )

CLI:
    python -m dataset.download_from_drive \
        --url "https://drive.google.com/file/d/<FILE_ID>/view" \
        --out dataset --expect raw_words

Menggantikan pendekatan `google.colab.files.upload()` yang:
  - Mengharuskan pengguna men-drag file ZIP manual di browser.
  - Tidak auto-resume saat koneksi putus.
  - Tidak tersinkron dengan Google Drive pengguna.

Dengan helper ini, Colab cukup melakukan `drive.mount()` sekali, dan semua
dataset (alfabet BISINDO, klip gesture kata, dataset live) dapat diunduh
otomatis dari link Google Drive menggunakan file ID.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR  # noqa: E402


# ---------------------------------------------------------------
# File-ID extraction
# ---------------------------------------------------------------
_ID_PATTERNS = [
    re.compile(r"/file/d/([a-zA-Z0-9_-]{10,})"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]{10,})"),
    re.compile(r"/folders/([a-zA-Z0-9_-]{10,})"),
]


def extract_file_id(file_id_or_url: str) -> str:
    """
    Terima file ID mentah atau URL Google Drive dan kembalikan file ID.
    Contoh input yang didukung:
      - "1aBcDeFgHiJkLmNoPqRsTuVwXyZ"
      - "https://drive.google.com/file/d/<ID>/view?usp=sharing"
      - "https://drive.google.com/open?id=<ID>"
      - "https://drive.google.com/uc?id=<ID>"
    """
    s = file_id_or_url.strip()
    # Sudah berupa ID murni?
    if "/" not in s and "?" not in s and "=" not in s and len(s) >= 10:
        return s
    for pat in _ID_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)
    raise ValueError(
        f"Tidak dapat menemukan Google Drive file ID pada: {file_id_or_url!r}"
    )


# ---------------------------------------------------------------
# Download + extract + validate
# ---------------------------------------------------------------
def _ensure_gdown():
    try:
        import gdown  # noqa: F401
        return
    except ImportError:
        # Auto-install di Colab/local (best effort).
        import subprocess
        print("[info] Menginstal gdown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"],
                       check=True)


def download_zip_from_drive(
    file_id_or_url: str,
    dest_zip: Path,
    quiet: bool = False,
) -> Path:
    """Unduh file ZIP dari Google Drive ke `dest_zip`. Return path ZIP."""
    _ensure_gdown()
    import gdown

    file_id = extract_file_id(file_id_or_url)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[info] Mengunduh dari Google Drive: {file_id} -> {dest_zip}")
    out = gdown.download(url=url, output=str(dest_zip), quiet=quiet, fuzzy=True)
    if out is None or not Path(out).exists():
        raise RuntimeError(
            f"Gagal mengunduh file dari Google Drive (id={file_id}). "
            "Pastikan link sudah di-share 'Anyone with the link'."
        )
    return Path(out)


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Ekstrak ZIP ke `extract_to` (dibuat otomatis)."""
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[info] Mengekstrak {zip_path.name} -> {extract_to}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)


def validate_dataset(
    base_dir: Path,
    expected_subdirs: Optional[Iterable[str]] = None,
) -> List[Path]:
    """
    Pastikan `base_dir` ada dan (opsional) setiap `expected_subdirs` sudah
    ada di dalamnya. Return list path yang divalidasi.
    """
    checked: List[Path] = []
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(
            f"Folder dataset '{base_dir}' tidak terbentuk setelah ekstraksi."
        )

    if expected_subdirs:
        missing = []
        for name in expected_subdirs:
            p = base_dir / name
            if p.exists() and p.is_dir():
                checked.append(p)
            else:
                missing.append(p)
        if missing:
            raise FileNotFoundError(
                "Folder berikut tidak ditemukan setelah ekstraksi: "
                + ", ".join(str(p) for p in missing)
            )
    return checked


def download_and_extract(
    file_id_or_url: str,
    extract_to: str | Path = "dataset",
    expected_subdirs: Optional[Iterable[str]] = None,
    remove_zip: bool = True,
    zip_name: str = "dataset.zip",
    base_dir: Path = ROOT_DIR,
    quiet: bool = False,
) -> Path:
    """
    End-to-end: download → extract → (cleanup ZIP) → validate.

    Args:
        file_id_or_url   : file ID Drive mentah atau URL.
        extract_to       : folder tujuan ekstraksi, relatif ke `base_dir`
                           (atau absolut).
        expected_subdirs : daftar nama folder yang WAJIB ada setelah
                           ekstraksi (mis. ["raw_words"]). Jika tidak ada,
                           raise FileNotFoundError.
        remove_zip       : hapus file ZIP setelah ekstrak berhasil.
        zip_name         : nama sementara file ZIP yang diunduh.
        base_dir         : root repo (default: ROOT_DIR dari config.py).

    Return:
        Path absolut folder hasil ekstraksi.
    """
    extract_path = Path(extract_to)
    if not extract_path.is_absolute():
        extract_path = base_dir / extract_path
    extract_path.mkdir(parents=True, exist_ok=True)

    tmp_zip = base_dir / zip_name

    try:
        download_zip_from_drive(file_id_or_url, tmp_zip, quiet=quiet)
        extract_zip(tmp_zip, extract_path)
    finally:
        if remove_zip and tmp_zip.exists():
            try:
                tmp_zip.unlink()
                print(f"[info] ZIP sementara dihapus: {tmp_zip}")
            except OSError as e:
                print(f"[warn] Gagal menghapus ZIP sementara: {e}")

    validated = validate_dataset(extract_path, expected_subdirs)
    print(f"[done] Dataset siap di: {extract_path}")
    if validated:
        print("[done] Subfolder terverifikasi:")
        for p in validated:
            print(f"       - {p.relative_to(base_dir)}")
    return extract_path


# ---------------------------------------------------------------
# Optional: mount Google Drive (Colab only, no-op di environment lain)
# ---------------------------------------------------------------
def mount_drive(mount_point: str = "/content/drive") -> Optional[str]:
    """
    Mount Google Drive jika sedang berjalan di Colab. Di environment lain,
    fungsi ini tidak melakukan apa-apa.
    """
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        print("[info] Bukan environment Google Colab - skip mount Drive.")
        return None

    print(f"[info] Mounting Google Drive di {mount_point} ...")
    drive.mount(mount_point, force_remount=False)
    return mount_point


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--url", required=True,
                   help="File ID atau URL Google Drive (share: Anyone with the link)")
    p.add_argument("--out", default="dataset",
                   help="Folder tujuan ekstraksi (relatif ke repo root)")
    p.add_argument("--expect", action="append", default=[],
                   help="Nama subfolder yang wajib ada setelah ekstraksi "
                        "(boleh ulang, mis. --expect raw_words --expect raw)")
    p.add_argument("--keep-zip", action="store_true",
                   help="Jangan hapus file ZIP setelah ekstrak")
    p.add_argument("--mount-drive", action="store_true",
                   help="Mount Google Drive dulu (hanya berlaku di Colab)")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.mount_drive:
        mount_drive()
    download_and_extract(
        file_id_or_url=args.url,
        extract_to=args.out,
        expected_subdirs=args.expect or None,
        remove_zip=not args.keep_zip,
    )


if __name__ == "__main__":
    main()
