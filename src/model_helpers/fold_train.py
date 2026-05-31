import dataclasses
from collections import defaultdict

import numpy as np
import pandas as pd

from src.model_helpers.classic_ml import ClassicalModel, train_fold_classic_ml
from src.model_helpers.one_class import OneClassModels, train_fold_one_class
from src.utils.load_data import get_cv_folds
from src.utils.metrics import Metrics
from src.utils.representations import Representation
from src.utils.summary import fold_to_mean_row


def fold_train(
    model_name: OneClassModels | ClassicalModel,
    dataset_type: str,
    representation: Representation,
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run cross-validation for a single config."""
    cv = get_cv_folds(x_train, y_train, n_splits)

    fold_rows: list[dict] = []
    metrics: defaultdict[str, list[float]] = defaultdict(list)

    for fold_number, (train_index, val_index) in enumerate(cv, start=1):
        fold_metrics = _get_fold_metrics(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            train_index=train_index,
            val_index=val_index,
        )

        row = {
            "model": model_name,
            "representation": representation,
            "dataset": dataset_type,
            "fold": fold_number,
        }
        for metric, value in dataclasses.asdict(fold_metrics).items():
            row[metric] = value
            metrics[metric].append(value)

        fold_rows.append(row)

    mean_row, std_row = fold_to_mean_row(
        model=model_name,
        representation=representation,
        dataset=dataset_type,
        metrics=metrics,
    )
    summary = pd.DataFrame([mean_row, std_row])
    return pd.DataFrame(fold_rows), summary


def _get_fold_metrics(
    model_name: OneClassModels | ClassicalModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_index: np.ndarray,
    val_index: np.ndarray,
) -> Metrics:
    match model_name:
        case OneClassModels():
            return _get_fold_metrics_one_class(model_name, x_train, y_train, train_index, val_index)
        case ClassicalModel():
            return _get_fold_metrics_classic_ml(
                model_name, x_train, y_train, train_index, val_index
            )


def _get_fold_metrics_one_class(
    model_name: OneClassModels,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_index: np.ndarray,
    val_index: np.ndarray,
) -> Metrics:
    x_fold_train, _ = x_train[train_index], y_train[train_index]
    x_fold_val, y_fold_val = x_train[val_index], y_train[val_index]

    return train_fold_one_class(
        model_name=model_name,
        x_fold_train=x_fold_train,
        x_fold_val=x_fold_val,
        y_fold_val=y_fold_val,
    )


def _get_fold_metrics_classic_ml(
    model_name: ClassicalModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_index: np.ndarray,
    val_index: np.ndarray,
) -> Metrics:
    x_fold_train, y_fold_train = x_train[train_index], y_train[train_index]
    x_fold_val, y_fold_val = x_train[val_index], y_train[val_index]

    return train_fold_classic_ml(
        model_name=model_name,
        x_fold_train=x_fold_train,
        y_fold_train=y_fold_train,
        x_fold_val=x_fold_val,
        y_fold_val=y_fold_val,
    )
