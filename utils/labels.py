"""Helper untuk mapping label ↔ index & persistensi ke JSON."""
import json
from pathlib import Path
from typing import List, Dict

from config import ALL_CLASSES, LABELS_PATH


def build_label_maps(classes: List[str] = ALL_CLASSES):
    label_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}
    idx_to_label: Dict[int, str] = {i: c for i, c in enumerate(classes)}
    return label_to_idx, idx_to_label


def save_labels(classes: List[str] = ALL_CLASSES, path: Path = LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)


def load_labels(path: Path = LABELS_PATH) -> List[str]:
    if not path.exists():
        save_labels()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["classes"]


if __name__ == "__main__":
    save_labels()
    print(f"Saved {len(ALL_CLASSES)} labels → {LABELS_PATH}")
