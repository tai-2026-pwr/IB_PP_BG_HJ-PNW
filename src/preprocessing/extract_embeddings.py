import numpy as np
import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer, logging

from src.constants import DEVICE
from src.paths import (
    EMBEDDINGS_MEAN_DIR,
    EMBEDDINGS_PER_TOKEN_DIR,
    TEST_BALANCED_PATH,
    TEST_IMBALANCED_PATH,
    TRAIN_BALANCED_PATH,
    TRAIN_IMBALANCED_PATH,
)
from src.preprocessing.constatns import BATCH_SIZE, ESM_MODEL_NAME, MAX_LEN
from src.utils.load_data import load_data

logging.set_verbosity_error()


def get_embeddings(
    sequences: list[str], tokenizer: EsmTokenizer, model: EsmModel
) -> tuple[np.ndarray, np.ndarray]:
    all_mean: list[np.ndarray] = []
    all_per_token: list[np.ndarray] = []
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
        per_token_emb = hidden[:, 1 : MAX_LEN + 1, :].cpu().numpy()

        all_mean.append(mean_emb)
        all_per_token.append(per_token_emb)

        print(f"  {min(i + BATCH_SIZE, len(sequences))}/{len(sequences)}")

    return np.vstack(all_mean), np.vstack(all_per_token)


if __name__ == "__main__":
    print(f"Loading {ESM_MODEL_NAME} on {DEVICE}...")
    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME)

    esm_model = esm_model.to(DEVICE)  # type: ignore

    datasets: dict[str, pd.DataFrame] = {
        "train_balanced": load_data(TRAIN_BALANCED_PATH),
        "train_imbalanced": load_data(TRAIN_IMBALANCED_PATH),
        "test_balanced": load_data(TEST_BALANCED_PATH),
        "test_imbalanced": load_data(TEST_IMBALANCED_PATH),
    }

    for name, df in datasets.items():
        print(f"\nGenerating embeddings: {name} ({len(df)} sequences)...")
        mean_emb, per_token_emb = get_embeddings(df["Seq"].tolist(), tokenizer, esm_model)
        labels = df["CPP"].astype(int).to_numpy()

        np.savez_compressed(EMBEDDINGS_MEAN_DIR / f"{name}_X.npz", X=mean_emb)
        np.savez_compressed(EMBEDDINGS_PER_TOKEN_DIR / f"{name}_X.npz", X=per_token_emb)
        np.save(EMBEDDINGS_MEAN_DIR / f"{name}_y.npy", labels)
        np.save(EMBEDDINGS_PER_TOKEN_DIR / f"{name}_y.npy", labels)

        print(f"  mean:      {name}_X.npz {mean_emb.shape} ({mean_emb.dtype})")
        print(f"  per_token: {name}_X.npz {per_token_emb.shape} ({per_token_emb.dtype})")

    print("\nDone.")
    print("Mean embeddings saved in:      ", EMBEDDINGS_MEAN_DIR)
    print("Per-token embeddings saved in: ", EMBEDDINGS_PER_TOKEN_DIR)
