import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    balanced_accuracy: float
    f1: float
    mcc: float
    roc_auc: float

    def __iter__(self) -> Iterator[float]:
        return iter(dataclasses.astuple(self))

    @classmethod
    def keys(cls) -> list[str]:
        return [field.name for field in dataclasses.fields(cls)]

    def __getitem__(self, key: str) -> float:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Metrics:
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        mcc=matthews_corrcoef(y_true, y_pred),
        roc_auc=roc_auc_score(y_true, y_score),
    )
