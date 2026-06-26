"""
Konfigurasi global untuk sistem deteksi BISINDO.
Seluruh hyperparameter dan path terpusat di sini.
"""
from pathlib import Path

# ============================================================
# PATH
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"                # Gambar mentah BISINDO
WORD_RAW_DIR = DATASET_DIR / "raw_words"     # Rekaman gesture kata
PROCESSED_DIR = DATASET_DIR / "processed"    # Output .npy
MODEL_DIR = ROOT_DIR / "model" / "saved"
LOG_DIR = ROOT_DIR / "logs"

for d in (RAW_DIR, WORD_RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "bisindo_new.keras"
LABELS_PATH = MODEL_DIR / "labels2.json"

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
ABJAD_CLASSES = [chr(c) for c in range(ord("a"), ord("z") + 1)]
ALPHABET_CLASSES = ABJAD_CLASSES
DIGIT_CLASSES = [f"digit_{i}" for i in range(10)]
WORD_CLASSES = [
    "word_apa",
    "word_apa_kabar",
    "word_bagaimana",
    "word_baik",
    "word_belajar",
    "word_berapa",
    "word_berdiri",
    "word_bingung",
    "word_dia",
    "word_dimana",
    "word_duduk",
    "word_halo",
    "word_kalian",
    "word_kami",
    "word_kamu",
    "word_kapan",
    "word_kemana",
    "word_kita",
    "word_makan",
    "word_mandi",
    "word_marah",
    "word_melihat",
    "word_membaca",
    "word_menulis",
    "word_mereka",
    "word_minum",
    "word_pendek",
    "word_ramah",
    "word_sabar",
    "word_saya",
    "word_sedih",
    "word_selamat_malam",
    "word_selamat_pagi",
    "word_selamat_siang",
    "word_selamat_sore",
    "word_senang",
    "word_siapa",
    "word_terima_kasih",
    "word_tidur",
    "word_tinggi",
]

ALL_CLASSES = (
    ABJAD_CLASSES[:4]
    + DIGIT_CLASSES
    + ABJAD_CLASSES[4:23]
    + WORD_CLASSES
    + ABJAD_CLASSES[23:]
)
NUM_CLASSES = len(ALL_CLASSES)  # 76

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
# SEQUENCE / SLIDING WINDOW
# ============================================================
SEQUENCE_LENGTH = 30    # T frames per sample
WINDOW_STRIDE = 5       # sliding window stride

# ============================================================
# AUGMENTATION
# ============================================================
AUG_NOISE_STD = 0.01        # Gaussian noise std-dev
AUG_SCALE_RANGE = (0.9, 1.1)
AUG_ROTATION_DEG = 15.0
AUG_TRANSLATION = 0.05
AUG_MULTIPLIER = 3          # jumlah sample hasil augment per sample asli

# ============================================================
# TRAINING
# ============================================================
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
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
