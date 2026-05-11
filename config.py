"""
Konfigurasi global untuk sistem deteksi BISINDO.
Seluruh hyperparameter dan path terpusat di sini.
Pengguna dapat mengubah konfigurasi di sini untuk menyesuaikan
eksperimen (sequence length, frame stride, rasio split, dll).
"""
from pathlib import Path

# ============================================================
# PATH
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"                # Gambar mentah BISINDO
WORD_RAW_DIR = DATASET_DIR / "raw_words"     # Rekaman gesture kata
NUMBER_RAW_DIR = DATASET_DIR / "raw_numbers" # Rekaman gesture angka
PROCESSED_DIR = DATASET_DIR / "processed"    # Output .npy
MODEL_DIR = ROOT_DIR / "model" / "saved"
LOG_DIR = ROOT_DIR / "logs"

for d in (RAW_DIR, WORD_RAW_DIR, NUMBER_RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "bisindo_model.h5"
LABELS_PATH = MODEL_DIR / "labels.json"

# ============================================================
# DATASET
# ============================================================
# Repository sumber dataset BISINDO (Indonesian Sign Language Hand Sign
# Detection Dataset). Struktur: collectedimages/<LABEL>/*.jpg
BISINDO_REPO_URL = (
    "https://github.com/rhiosutoyo/"
    "Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset"
)
BISINDO_REPO_BRANCH = "master"
BISINDO_DATASET_SUBDIR = "collectedimages"

# Kelas alfabet BISINDO (A–Z)
ALPHABET_CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# Kelas angka (0–9)
NUMBER_CLASSES = [str(n) for n in range(10)]

# 5 gesture kata tambahan (bebas dipilih)
WORD_CLASSES = ["HALO", "MAKAN", "MINUM", "TERIMA_KASIH", "TOLONG"]

# Gabungan seluruh kelas: alfabet + angka + kata = 41 kelas
ALL_CLASSES = ALPHABET_CLASSES + NUMBER_CLASSES + WORD_CLASSES
NUM_CLASSES = len(ALL_CLASSES)  # 41

# Auto-label: jika True, build_dataset() akan mendeteksi label baru
# dari folder tanpa perlu menambahkan ke daftar di atas secara manual.
AUTO_DETECT_LABELS = True

# ============================================================
# MEDIAPIPE
# ============================================================
MP_STATIC_IMAGE_MODE = False
MP_MAX_NUM_HANDS = 2
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

NUM_LANDMARKS = 21      # MediaPipe hand landmark count
LANDMARK_DIMS = 3       # x, y, z
# BISINDO sering menggunakan 2 tangan → kita flatten max 2 tangan
MAX_HANDS = 2
FEATURES_PER_FRAME = MAX_HANDS * NUM_LANDMARKS * LANDMARK_DIMS  # 2*21*3 = 126

# ============================================================
# SEQUENCE / SLIDING WINDOW / TEMPORAL SAMPLING
# ============================================================
SEQUENCE_LENGTH = 30    # T frames per sample
WINDOW_STRIDE = 5       # sliding window stride (hop antar sequence)
FRAME_STRIDE = 1        # interval pengambilan frame dari video/stream
                        # (=1 ambil semua frame, =2 skip setiap frame kedua, dst)
SEQUENCE_OVERLAP = 0.83 # rasio overlap antar sequence (0.0–1.0).
                        # Digunakan hanya jika > 0; WINDOW_STRIDE diabaikan
                        # saat SEQUENCE_OVERLAP > 0.
                        # Default 0.83 ≈ stride 5 dari length 30.
TEMPORAL_SAMPLING = "uniform"  # "uniform" | "random"
                               # uniform: ambil frame dengan interval tetap
                               # random: ambil T frame secara acak dari klip

# ============================================================
# AUGMENTATION
# ============================================================
AUG_NOISE_STD = 0.01        # Gaussian noise std-dev
AUG_SCALE_RANGE = (0.9, 1.1)
AUG_ROTATION_DEG = 15.0
AUG_TRANSLATION = 0.05
AUG_TEMPORAL_JITTER = 2     # max frame shift untuk temporal jitter (±N)
                             # 0 = disabled. Mengacak urutan frame sedikit
                             # untuk meningkatkan robustness temporal.
AUG_MULTIPLIER = 3          # jumlah sample hasil augment per sample asli

# ============================================================
# TRAINING
# ============================================================
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3

# Rasio pembagian dataset (train/val/test). Jumlah harus ≤ 1.0.
# Contoh: 70/15/15, 80/10/10, 30/70 (no test), dst.
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
# TRAIN_SPLIT dihitung otomatis: 1.0 - VAL_SPLIT - TEST_SPLIT

RANDOM_SEED = 42

EARLY_STOP_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ============================================================
# INFERENCE / REAL-TIME
# ============================================================
CONFIDENCE_THRESHOLD = 0.70
BUFFER_SIZE = 10            # prediction buffer untuk smoothing
SMOOTHING_MIN_VOTES = 6     # minimum majority votes

# ============================================================
# API
# ============================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
