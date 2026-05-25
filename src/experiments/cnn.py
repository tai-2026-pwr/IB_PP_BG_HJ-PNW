from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 50
ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_DIR = ROOT / "data/embeddings/per_token"
OUTPUT_DIR     = ROOT / "results/cnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT    = 0.3
LR         = 1e-4
EPOCHS     = 50
BATCH_SIZE = 64

torch.manual_seed(RANDOM_STATE)


class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim) → (batch, input_dim, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return self.classifier(x).squeeze(1)


def load_embeddings(dataset_type):
    X_train = np.load(EMBEDDINGS_DIR / f"train_{dataset_type}_X.npy")
    y_train = np.load(EMBEDDINGS_DIR / f"train_{dataset_type}_y.npy")
    X_test  = np.load(EMBEDDINGS_DIR / f"test_{dataset_type}_X.npy")
    y_test  = np.load(EMBEDDINGS_DIR / f"test_{dataset_type}_y.npy")
    return X_train, y_train, X_test, y_test


def calculate_metrics(y_true, y_pred, y_score):
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1":                f1_score(y_true, y_pred),
        "mcc":               matthews_corrcoef(y_true, y_pred),
        "roc_auc":           roc_auc_score(y_true, y_score),
    }


def evaluate_model(model, X, y):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
    )
    all_logits = []
    with torch.no_grad():
        for X_batch, _ in loader:
            all_logits.append(model(X_batch.to(DEVICE)).cpu())

    logits  = torch.cat(all_logits).numpy()
    y_score = torch.sigmoid(torch.tensor(logits)).numpy()
    y_pred  = (y_score >= 0.5).astype(int)
    return calculate_metrics(y, y_pred, y_score)


def run_single_experiment(dataset_type):
    X_train, y_train, X_test, y_test = load_embeddings(dataset_type)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []

    for fold_number, (train_index, val_index) in enumerate(
        cv.split(X_train, y_train), start=1
    ):
        X_fold_train, y_fold_train = X_train[train_index], y_train[train_index]
        X_fold_val,   y_fold_val   = X_train[val_index],   y_train[val_index]

        input_dim = X_fold_train.shape[2]  # (n, seq_len, hidden_size)
        model     = CNN1D(input_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()
        loader    = DataLoader(
            TensorDataset(torch.tensor(X_fold_train, dtype=torch.float32),
                          torch.tensor(y_fold_train, dtype=torch.float32)),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        model.train()
        for _ in range(EPOCHS):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                criterion(model(X_batch), y_batch).backward()
                optimizer.step()

        metrics = evaluate_model(model, X_fold_val, y_fold_val)

        row = {
            "model":          "1D-CNN",
            "representation": "ESM-2",
            "dataset":        dataset_type,
            "fold":           fold_number,
        }
        row.update(metrics)
        fold_rows.append(row)

    fold_results = pd.DataFrame(fold_rows)
    metric_names = ["accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc"]

    mean_row = {"model": "1D-CNN", "representation": "ESM-2",
                "dataset": dataset_type, "stage": "cv_mean"}
    std_row  = {"model": "1D-CNN", "representation": "ESM-2",
                "dataset": dataset_type, "stage": "cv_std"}

    for metric in metric_names:
        mean_row[metric] = fold_results[metric].mean()
        std_row[metric]  = fold_results[metric].std()

    input_dim   = X_train.shape[2]
    final_model = CNN1D(input_dim).to(DEVICE)
    optimizer   = torch.optim.Adam(final_model.parameters(), lr=LR)
    criterion   = nn.BCEWithLogitsLoss()
    loader      = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    final_model.train()
    for _ in range(EPOCHS):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            criterion(final_model(X_batch), y_batch).backward()
            optimizer.step()

    test_metrics = evaluate_model(final_model, X_test, y_test)
    test_row = {
        "model": "1D-CNN", "representation": "ESM-2",
        "dataset": dataset_type, "stage": "test",
    }
    test_row.update(test_metrics)

    summary = pd.DataFrame([mean_row, std_row, test_row])
    return fold_results, summary


print(f"Device: {DEVICE}")

# Model 21: 1D-CNN, ESM-2, balanced
# Model 22: 1D-CNN, ESM-2, imbalanced
configs = ["balanced", "imbalanced"]

all_folds     = []
all_summaries = []

for dataset_type in configs:
    print(f"\nRunning 1D-CNN [{dataset_type}]...")
    fold_results, summary = run_single_experiment(dataset_type)
    all_folds.append(fold_results)
    all_summaries.append(summary)

final_folds   = pd.concat(all_folds,     ignore_index=True)
final_summary = pd.concat(all_summaries, ignore_index=True)

final_folds.to_csv(OUTPUT_DIR / "cv_folds.csv",  index=False)
final_summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

print("\nModels 21-22 (1D-CNN) experiments completed.")
print(final_summary)
print("Results saved in:", OUTPUT_DIR)