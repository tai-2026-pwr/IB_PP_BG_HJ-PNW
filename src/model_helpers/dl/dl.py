from __future__ import annotations
from tqdm import tqdm

import dataclasses
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

from src.constants import DEVICE, RANDOM_STATE
from src.model_helpers.dl.constants import BATCH_SIZE, EPOCHS, LR
from src.model_helpers.dl.models.cnn import CNN1D
from src.model_helpers.dl.models.mlp import MLP
from src.utils.metrics import Metrics, calculate_metrics
from src.utils.representations import Representation

if TYPE_CHECKING:
    import torch.utils.data


class DLModel(StrEnum):
    MLP = "MLP"
    CNN1D = "1D-CNN"


def load_embeddings(
    embeddings_dir: Path, dataset_type: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    with np.load(embeddings_dir / f"train_{dataset_type}_X.npz") as data:
        x_train = data["X"]

    with np.load(embeddings_dir / f"test_{dataset_type}_X.npz") as data:
        x_test = data["X"]

    # Labels are standard uncompressed .npy files, so load them normally
    y_train = np.load(embeddings_dir / f"train_{dataset_type}_y.npy")
    y_test = np.load(embeddings_dir / f"test_{dataset_type}_y.npy")

    return x_train, y_train, x_test, y_test


def train_fold_dl(
    model_name: DLModel,
    x_fold_train: np.ndarray,
    y_fold_train: np.ndarray,
    x_fold_val: np.ndarray,
    y_fold_val: np.ndarray,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
) -> Metrics:
    """Instantiate a DL model and run the training pipeline for a single CV fold."""
    model = _make_pytorch_model(model_name, input_dim=x_fold_train.shape[-1]).to(DEVICE)

    _train_model(model, x_fold_train, y_fold_train, epochs, lr, batch_size)

    return _evaluate_pytorch_model(model, x_fold_val, y_fold_val)


def evaluate_test_set_dl(
    model_name: DLModel,
    representation: Representation,
    dataset_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Train the final DL model on the full train set and evaluate on the test set."""
    model = _train_final_model_pytorch(model_name=model_name, x_train=x_train, y_train=y_train)

    metrics = _evaluate_pytorch_model(model, x_test, y_test)

    row = {
        "model": model_name,
        "representation": representation,
        "dataset": dataset_type,
        "stage": "test_score",
    }
    row.update(dataclasses.asdict(metrics))

    return pd.DataFrame([row])


def _train_final_model_pytorch(
    model_name: DLModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
) -> nn.Module:
    """Train the final production PyTorch model on the full training set."""
    model = _make_pytorch_model(model_name, input_dim=x_train.shape[-1]).to(DEVICE)
    _train_model(model, x_train, y_train, epochs, lr, batch_size)
    return model


def _train_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """Unified engine to execute backpropagation loops without duplication."""
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = _build_loaders(x, y, batch_size, shuffle=True)

    model.train()
    for _ in tqdm(range(epochs), desc="  Training Epochs", leave=False):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x_batch)

            # Removed .unsqueeze(1) here to preserve 1D shape alignment with your models
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()


def _build_loaders(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    """Build a standard PyTorch DataLoader."""
    from torch.utils.data import DataLoader, TensorDataset

    return DataLoader(
        TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _evaluate_pytorch_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
) -> Metrics:
    """Evaluate a trained model and accurately assemble a Metrics block."""
    model.eval()
    tensor_x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor_x)
        # Handle 0-dim tensor edges cleanly by forcing a 1D array view
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        y_score = torch.sigmoid(logits).cpu().numpy()

    y_pred = (y_score >= 0.5).astype(int)

    # Map raw evaluation dict to your custom structural Dataclass
    metrics_dict = calculate_metrics(y, y_pred, y_score)
    return Metrics(**metrics_dict)


def _make_pytorch_model(model_name: DLModel, input_dim: int) -> torch.nn.Module:
    """Factory module ensuring consistent seed generation per pipeline invocation."""
    torch.manual_seed(RANDOM_STATE)

    models = {
        DLModel.MLP: lambda: MLP(input_dim=input_dim),
        DLModel.CNN1D: lambda: CNN1D(input_dim=input_dim),
    }
    return models[model_name]()
