from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer, logging

logging.set_verbosity_error()

RANDOM_STATE = 50
ROOT = Path(__file__).resolve().parents[2]

TRAIN_BALANCED_PATH   = ROOT / "data/processed/dataset_CPP_1to1_train.csv"
TRAIN_IMBALANCED_PATH = ROOT / "data/processed/dataset_CPP_1to2_train.csv"
TEST_BALANCED_PATH    = ROOT / "data/processed/dataset_CPP_1to1_test.csv"
TEST_IMBALANCED_PATH  = ROOT / "data/processed/dataset_CPP_1to2_test.csv"

EMBEDDINGS_MEAN_DIR      = ROOT / "data/embeddings/mean"
EMBEDDINGS_PER_TOKEN_DIR = ROOT / "data/embeddings/per_token"
EMBEDDINGS_MEAN_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_PER_TOKEN_DIR.mkdir(parents=True, exist_ok=True)

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 32
MAX_LEN        = 50


def load_data(path):
    df = pd.read_csv(path)
    df = df[["Seq", "CPP"]].dropna().copy()
    df["Seq"] = df["Seq"].astype(str).str.strip().str.upper()
    df = df[df["Seq"] != ""].copy()
    return df


def get_embeddings(sequences, tokenizer, model):
    all_mean      = []
    all_per_token = []
    model.eval()

    for i in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[i : i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN + 2,  # +2 for [CLS] and [EOS] tokens
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state  # (batch, MAX_LEN+2, hidden_size)

        # mean pooling → (batch, hidden_size)
        mean_emb = hidden.mean(dim=1).cpu().numpy()

        # per-token: strip [CLS] and [EOS] → (batch, MAX_LEN, hidden_size)
        per_token_emb = hidden[:, 1:MAX_LEN + 1, :].cpu().numpy()

        all_mean.append(mean_emb)
        all_per_token.append(per_token_emb)

        print(f"  {min(i + BATCH_SIZE, len(sequences))}/{len(sequences)}")

    return np.vstack(all_mean), np.vstack(all_per_token)


print(f"Loading {ESM_MODEL_NAME} on {DEVICE}...")
tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(DEVICE)

datasets = {
    "train_balanced":   load_data(TRAIN_BALANCED_PATH),
    "train_imbalanced": load_data(TRAIN_IMBALANCED_PATH),
    "test_balanced":    load_data(TEST_BALANCED_PATH),
    "test_imbalanced":  load_data(TEST_IMBALANCED_PATH),
}

for name, df in datasets.items():
    print(f"\nGenerating embeddings: {name} ({len(df)} sequences)...")
    mean_emb, per_token_emb = get_embeddings(df["Seq"].tolist(), tokenizer, esm_model)
    labels = df["CPP"].astype(int).to_numpy()

    np.savez_compressed(EMBEDDINGS_MEAN_DIR      / f"{name}_X.npz", X=mean_emb)
    np.savez_compressed(EMBEDDINGS_PER_TOKEN_DIR / f"{name}_X.npz", X=per_token_emb)
    np.save(EMBEDDINGS_MEAN_DIR      / f"{name}_y.npy", labels)
    np.save(EMBEDDINGS_PER_TOKEN_DIR / f"{name}_y.npy", labels)

    print(f"  mean:      {name}_X.npz {mean_emb.shape} ({mean_emb.dtype})")
    print(f"  per_token: {name}_X.npz {per_token_emb.shape} ({per_token_emb.dtype})")

print("\nDone.")
print("Mean embeddings saved in:      ", EMBEDDINGS_MEAN_DIR)
print("Per-token embeddings saved in: ", EMBEDDINGS_PER_TOKEN_DIR)