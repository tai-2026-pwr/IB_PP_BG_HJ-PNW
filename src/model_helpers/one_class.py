from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.constants import RANDOM_STATE
from src.utils.metrics import Metrics, calculate_metrics

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class OneClassModel(StrEnum):
    SVM = "One-Class SVM"
    IFOREST = "Isolation Forest"


def evaluate_test_set_one_class(
    model_name: OneClassModel,
    representation: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    model = fit_final_one_class(model_name, x_train)
    metrics = evaluate_fitted_one_class(model, x_test, y_test)

    return pd.DataFrame(
        [
            {
                "model": model_name,
                "representation": representation,
                "dataset": "test",
                "stage": "test_score",
                **metrics,
            }
        ]
    )


def fit_final_one_class(
    model_name: OneClassModel,
    x_train: np.ndarray,
) -> BaseEstimator:
    """Fit a one-class model on the full training set."""
    model = _make_one_class_model(model_name)
    model.fit(x_train)
    return model


def evaluate_fitted_one_class(
    model: BaseEstimator,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Metrics:
    """Evaluate an already fitted one-class model."""
    raw_preds = model.predict(x_test)
    y_pred = _convert_one_class_predictions(raw_preds)

    y_score = -model.decision_function(x_test)

    return calculate_metrics(y_test, y_pred, y_score)


def train_fold_one_class(
    model_name: OneClassModel,
    x_fold_train: np.ndarray,
    x_fold_val: np.ndarray,
    y_fold_val: np.ndarray,
) -> Metrics:
    model = fit_final_one_class(model_name, x_fold_train)
    return evaluate_fitted_one_class(model, x_fold_val, y_fold_val)


def _convert_one_class_predictions(raw_predictions: np.ndarray) -> np.ndarray:
    """Convert one-class predictions to binary (1 for anomaly, 0 for normal)."""
    return np.where(raw_predictions == -1, 1, 0)


def _make_one_class_model(model_name: OneClassModel) -> BaseEstimator:
    models = {
        OneClassModel.SVM: lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", OneClassSVM()),
            ]
        ),
        OneClassModel.IFOREST: lambda: IsolationForest(random_state=RANDOM_STATE),
    }

    return models[model_name]()
