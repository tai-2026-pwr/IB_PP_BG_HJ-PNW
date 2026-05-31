from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.constants import RANDOM_STATE


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["Seq", "CPP"]].dropna().copy()
    df["Seq"] = df["Seq"].astype(str).str.strip().str.upper()
    df = df[df["Seq"] != ""].copy()
    return df


def get_cv_folds(
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate stratified K-fold splits."""
    return list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE).split(x, y)
    )
