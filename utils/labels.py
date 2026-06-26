"""Helper untuk mapping label ↔ index & persistensi ke JSON.

Mendukung:
- Static class list dari config (ALL_CLASSES).
- Auto-detect class baru dari folder dataset tanpa perlu mapping manual.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from config import ALL_CLASSES, LABELS_PATH, RAW_DIR, WORD_RAW_DIR, NUMBER_RAW_DIR


def discover_labels_from_folders(
    *dirs: Path,
    existing: Optional[List[str]] = None,
) -> List[str]:
    """
    Scan folder dataset dan temukan label baru (nama subfolder) yang belum
    ada di `existing`. Mengembalikan daftar lengkap (existing + baru), sorted.

    Ini memungkinkan pengguna menambahkan gesture baru cukup dengan membuat
    folder baru tanpa mengubah config.py.
    """
    existing_set: Set[str] = set(existing or ALL_CLASSES)
    discovered: Set[str] = set()

    for d in dirs:
        if not d.exists():
            continue
        for sub in d.iterdir():
            if sub.is_dir():
                name = sub.name.strip().upper()
                if name and name not in existing_set:
                    discovered.add(name)

    if discovered:
        # Gabung existing + discovered (sorted agar deterministik)
        all_classes = list(existing or ALL_CLASSES) + sorted(discovered)
        return all_classes
    return list(existing or ALL_CLASSES)


def build_label_maps(
    classes: Optional[List[str]] = None,
    auto_detect: bool = True,
) -> tuple:
    """
    Bangun mapping label↔index. Jika `auto_detect=True`, scan folder
    dataset untuk menemukan label tambahan secara otomatis.
    """
    if classes is None:
        classes = ALL_CLASSES
    if auto_detect:
        from config import AUTO_DETECT_LABELS
        if AUTO_DETECT_LABELS:
            classes = discover_labels_from_folders(
                RAW_DIR, WORD_RAW_DIR, NUMBER_RAW_DIR,
                existing=classes,
            )
    label_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}
    idx_to_label: Dict[int, str] = {i: c for i, c in enumerate(classes)}
    return label_to_idx, idx_to_label


def save_labels(classes: Optional[List[str]] = None, path: Path = LABELS_PATH) -> None:
    if classes is None:
        classes = discover_labels_from_folders(
            RAW_DIR, WORD_RAW_DIR, NUMBER_RAW_DIR,
            existing=ALL_CLASSES,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    label_to_idx, idx_to_label = build_label_maps(classes)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_classes": len(classes),
                "labels": classes,
                "label_to_idx": label_to_idx,
                "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_labels(path: Path = LABELS_PATH) -> List[str]:
    if not path.exists():
        save_labels()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if "classes" in data:
        labels = data["classes"]
    elif "labels" in data:
        labels = data["labels"]
    elif "label_to_idx" in data:
        labels = [
            label
            for label, _idx in sorted(
                data["label_to_idx"].items(),
                key=lambda item: int(item[1]),
            )
        ]
    else:
        raise KeyError(
            f"File label {path} harus berisi key 'classes', 'labels', "
            "atau 'label_to_idx'."
        )

    if "num_classes" in data and int(data["num_classes"]) != len(labels):
        raise ValueError(
            f"File label {path} menyatakan num_classes={data['num_classes']}, "
            f"tetapi berisi {len(labels)} label."
        )

    return labels


if __name__ == "__main__":
    save_labels()
    classes = load_labels()
    print(f"Saved {len(classes)} labels → {LABELS_PATH}")
    for i, c in enumerate(classes):
        print(f"  [{i:2d}] {c}")
