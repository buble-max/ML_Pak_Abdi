"""
Merekam gesture kata BISINDO tambahan sesuai WORD_CLASSES di config.py
langsung dari webcam.

Setiap kata direkam sebanyak N klip × T frame dan disimpan sebagai gambar:
    dataset/raw_words/<LABEL>/clip_<k>/frame_<t>.jpg

Kontrol:
    SPACE = mulai merekam 1 klip untuk label saat ini
    n     = pindah ke label berikutnya
    p     = pindah ke label sebelumnya
    q     = keluar
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import SEQUENCE_LENGTH, WORD_CLASSES, WORD_RAW_DIR  # noqa: E402

CLIPS_PER_LABEL = 30  # jumlah klip per kata


def record_label(cap, label: str, out_dir: Path, clips: int = CLIPS_PER_LABEL) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("clip_*")))
    print(f"[info] Mulai merekam '{label}'. Klip existing: {existing}")

    k = existing
    while k < existing + clips:
        # Countdown
        for s in (3, 2, 1):
            end = time.time() + 1
            while time.time() < end:
                ok, frame = cap.read()
                if not ok:
                    return
                frame = cv2.flip(frame, 1)
                cv2.putText(
                    frame, f"{label}  clip {k+1}/{existing+clips}  in {s}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                )
                cv2.imshow("Recorder", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return

        clip_dir = out_dir / f"clip_{k:04d}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        for t in range(SEQUENCE_LENGTH):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.imwrite(str(clip_dir / f"frame_{t:03d}.jpg"), frame)
            disp = frame.copy()
            cv2.putText(
                disp, f"REC {label}  {t+1}/{SEQUENCE_LENGTH}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
            )
            cv2.imshow("Recorder", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        print(f"  ✓ clip_{k:04d} tersimpan")
        k += 1


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Webcam tidak dapat dibuka")
        return

    idx = 0
    print("Kontrol: SPACE=rekam, n=next, p=prev, q=quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            label = WORD_CLASSES[idx]
            cv2.putText(
                frame, f"[{idx+1}/{len(WORD_CLASSES)}] {label}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.putText(
                frame, "SPACE=record  n=next  p=prev  q=quit",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )
            cv2.imshow("Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                idx = (idx + 1) % len(WORD_CLASSES)
            elif key == ord("p"):
                idx = (idx - 1) % len(WORD_CLASSES)
            elif key == ord(" "):
                record_label(cap, label, WORD_RAW_DIR / label)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
