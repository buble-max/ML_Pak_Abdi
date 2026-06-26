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
    print(f"Saved {len(ALL_CLASSES)} labels → {LABELS_PATH}")
