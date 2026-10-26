from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.constants import RANDOM_STATE
from src.utils.metrics import Metrics, calculate_metrics

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class ClassicalModel(StrEnum):
    SVM = "SVM"
    RANDOM_FOREST = "Random Forest"
    GRADIENT_BOOSTING = "Gradient Boosting"


def evaluate_test_set_classic_ml(
    model_name: ClassicalModel,
    representation: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    model = fit_final_classic_ml(model_name, x_train, y_train)
    metrics = evaluate_fitted_classic_ml(model, x_test, y_test)

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


def fit_final_classic_ml(
    model_name: ClassicalModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> BaseEstimator:
    """Fit a classical model on the full training set."""
    model = _make_sklearn_model(model_name)
    model.fit(x_train, y_train)
    return model


def evaluate_fitted_classic_ml(
    model: BaseEstimator,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Metrics:
    """Evaluate an already fitted classical model."""
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    return calculate_metrics(y_test, y_pred, y_score)


def train_fold_classic_ml(
    model_name: ClassicalModel,
    x_fold_train: np.ndarray,
    y_fold_train: np.ndarray,
    x_fold_val: np.ndarray,
    y_fold_val: np.ndarray,
) -> Metrics:
    """Train an sklearn model on a fold and evaluate it."""
    model = fit_final_classic_ml(model_name, x_fold_train, y_fold_train)
    return evaluate_fitted_classic_ml(model, x_fold_val, y_fold_val)


def _make_sklearn_model(model_name: ClassicalModel) -> BaseEstimator:
    models = {
        ClassicalModel.SVM: lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, random_state=RANDOM_STATE)),
            ]
        ),
        ClassicalModel.RANDOM_FOREST: lambda: RandomForestClassifier(random_state=RANDOM_STATE),
        ClassicalModel.GRADIENT_BOOSTING: lambda: GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }

    return models[model_name]()
