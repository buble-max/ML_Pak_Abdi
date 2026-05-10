"""
Training script BISINDO CNN+LSTM.

Usage:
    python -m training.train
    python -m training.train --epochs 50 --batch 32 --use_raw

Output:
    model/saved/bisindo_model.h5
    model/saved/labels.json
    logs/history.json
    logs/classification_report.txt
    logs/confusion_matrix.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))

from augmentation.augment import augment_dataset, balance_classes  # noqa: E402
from config import (  # noqa: E402
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    LOG_DIR,
    MODEL_DIR,
    MODEL_PATH,
    PROCESSED_DIR,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    TEST_SPLIT,
    VAL_SPLIT,
)
from model.architecture import build_model  # noqa: E402
from utils.labels import load_labels, save_labels  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--use_raw",
        action="store_true",
        help="Gunakan X.npy + augment on-the-fly, bukan X_aug.npy yang sudah disimpan.",
    )
    p.add_argument("--no_balance", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()


def _load_data(use_raw: bool, do_balance: bool):
    if use_raw or not (PROCESSED_DIR / "X_aug.npy").exists():
        X = np.load(PROCESSED_DIR / "X.npy")
        y = np.load(PROCESSED_DIR / "y.npy")
        print(f"[info] Loaded raw X {X.shape}, y {y.shape}")
        X, y = augment_dataset(X, y)
        if do_balance:
            X, y = balance_classes(X, y)
        print(f"[info] After aug/balance → X {X.shape}, y {y.shape}")
    else:
        X = np.load(PROCESSED_DIR / "X_aug.npy")
        y = np.load(PROCESSED_DIR / "y_aug.npy")
        print(f"[info] Loaded X_aug {X.shape}, y_aug {y.shape}")
    return X.astype(np.float32), y.astype(np.int64)


def _split(X, y, seed: int):
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=seed
    )
    val_frac = VAL_SPLIT / (1.0 - TEST_SPLIT)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=seed
    )
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def _plot_confusion(cm: np.ndarray, labels, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix – BISINDO")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] Plot confusion matrix gagal: {e}")


def main() -> None:
    args = _parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    # --- Data ---
    save_labels()
    labels = load_labels()
    X, y = _load_data(use_raw=args.use_raw, do_balance=not args.no_balance)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = _split(X, y, args.seed)
    print(f"[info] train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    # --- Model ---
    model = build_model()
    model.summary()

    # --- Callbacks ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(LOG_DIR / "training.csv")),
    ]

    # --- Train ---
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1,
    )

    # Pastikan model terakhir (best) disimpan ke .h5
    model.save(MODEL_PATH)
    print(f"[done] Model tersimpan: {MODEL_PATH}")

    # --- Evaluate ---
    test_loss, test_acc, *_rest = model.evaluate(X_te, y_te, verbose=0)
    print(f"[test] loss={test_loss:.4f}  acc={test_acc:.4f}")

    y_pred = model.predict(X_te, batch_size=args.batch, verbose=0).argmax(1)
    report = classification_report(y_te, y_pred, target_names=labels, zero_division=0)
    (LOG_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)

    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(labels))))
    np.save(LOG_DIR / "confusion_matrix.npy", cm)
    if not args.no_plot:
        _plot_confusion(cm, labels, LOG_DIR / "confusion_matrix.png")

    (LOG_DIR / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals]
                    for k, vals in history.history.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"[done] Log tersimpan di {LOG_DIR}")


if __name__ == "__main__":
    main()
