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
    model = _make_sklearn_model(model_name)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_score)

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


def train_fold_classic_ml(
    model_name: ClassicalModel,
    x_fold_train: np.ndarray,
    y_fold_train: np.ndarray,
    x_fold_val: np.ndarray,
    y_fold_val: np.ndarray,
) -> Metrics:
    """Train an sklearn model on a fold and evaluate it."""
    model = _make_sklearn_model(model_name)
    model.fit(x_fold_train, y_fold_train)

    y_pred = model.predict(x_fold_val)
    y_score = model.predict_proba(x_fold_val)[:, 1]

    return calculate_metrics(y_fold_val, y_pred, y_score)


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
