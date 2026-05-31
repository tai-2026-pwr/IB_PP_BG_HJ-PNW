from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_BALANCED_FILE = "dataset_CPP_1to1_train.csv"
TRAIN_IMBALANCED_FILE = "dataset_CPP_1to2_train.csv"
TEST_BALANCED_FILE = "dataset_CPP_1to1_test.csv"
TEST_IMBALANCED_FILE = "dataset_CPP_1to2_test.csv"

TRAIN_BALANCED_PATH = ROOT / PROCESSED_DIR / TRAIN_BALANCED_FILE
TRAIN_IMBALANCED_PATH = ROOT / PROCESSED_DIR / TRAIN_IMBALANCED_FILE
TEST_BALANCED_PATH = ROOT / PROCESSED_DIR / TEST_BALANCED_FILE
TEST_IMBALANCED_PATH = ROOT / PROCESSED_DIR / TEST_IMBALANCED_FILE

EMBEDDINGS_MEAN_DIR = ROOT / "data/embeddings/mean"
EMBEDDINGS_PER_TOKEN_DIR = ROOT / "data/embeddings/per_token"

EMBEDDINGS_MEAN_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_PER_TOKEN_DIR.mkdir(parents=True, exist_ok=True)
