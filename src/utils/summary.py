from typing import Any

import numpy as np

from src.utils.metrics import Metrics
from src.utils.representations import Representation


def fold_to_mean_row(
    *,
    model: str,
    representation: Representation,
    dataset: str,
    metrics: dict[str, list[float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create mean and std summary rows for cross-validation."""
    mean_row = _make_base_row(model, representation, dataset, "cv_mean")
    std_row = _make_base_row(model, representation, dataset, "cv_std")

    for metric in Metrics.keys():
        if metric in metrics:
            mean_row[metric] = float(np.mean(metrics[metric]))
            std_row[metric] = float(np.std(metrics[metric]))

    return mean_row, std_row


def _make_base_row(
    model: str,
    representation: Representation,
    dataset: str,
    stage: str,
) -> dict[str, Any]:
    """Create the base metadata row for cv_mean, cv_std, or test output."""
    return {
        "model": model,
        "representation": representation,
        "dataset": dataset,
        "stage": stage,
    }
