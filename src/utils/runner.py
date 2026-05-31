from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.utils.metrics import calculate_metrics

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def build_output_dir(output_dir: str | Path) -> Path:
    """Create output directory if it doesn't exist."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def save_results(
    folds_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    out_dir = Path(output_path)
    folds_df.to_csv(out_dir / "cv_folds.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
