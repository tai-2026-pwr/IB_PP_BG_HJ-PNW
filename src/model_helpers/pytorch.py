from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.constants import DEVICE
from src.utils.metrics import Metrics, calculate_metrics

if TYPE_CHECKING:
    import torch.utils.data


def train_and_evaluate_fold_pytorch(
    model_class: type[nn.Module],
    x_fold_train: np.ndarray,
    y_fold_train: np.ndarray,
    x_fold_val: np.ndarray,
    y_fold_val: np.ndarray,
    device: torch.device | None,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Metrics:

    if device is None:
        device = DEVICE

    model = model_class(input_dim=x_fold_train.shape[1]).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    loader = _build_loaders(x_fold_train, y_fold_train, batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    return _evaluate_pytorch_model(model, x_fold_val, y_fold_val, device)


def train_final_model_pytorch(
    model_class: type[nn.Module],
    x_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device | None,
    epochs: int,
    lr: float,
    batch_size: int,
) -> nn.Module:
    """Train final PyTorch model on full training set."""

    if device is None:
        device = DEVICE

    model = model_class(input_dim=x_train.shape[1]).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    loader = _build_loaders(x_train, y_train, batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

    return model


def _build_loaders(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    """Build a PyTorch DataLoader."""
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
    device: torch.device | None,
) -> Metrics:
    """Evaluate a trained PyTorch model."""
    if device is None:
        device = DEVICE

    model.eval()
    tensor_x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(tensor_x).squeeze()
        y_score = torch.sigmoid(logits).cpu().numpy()

    y_pred = (y_score >= 0.5).astype(int)
    return calculate_metrics(y, y_pred, y_score)
