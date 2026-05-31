from collections.abc import Callable
from enum import StrEnum

import numpy as np
import pandas as pd
import peptides

from src.constants import AMINO_ACIDS, DIPEPTIDES


class Representation(StrEnum):
    AAC = "AAC"
    DPC = "DPC"
    PHYSICOCHEMICAL = "Physicochemical"
    ESM2 = "ESM-2"


def extract_aac(seq: str) -> list[float]:
    length = len(seq)
    return [seq.count(aa) / length for aa in AMINO_ACIDS]


def extract_dpc_features(seq: str) -> list[float]:
    length = len(seq)
    if length < 2:
        return [0.0] * 400

    dp_counts = {dp: 0.0 for dp in DIPEPTIDES}
    for i in range(length - 1):
        dp = seq[i : i + 2]
        if dp in DIPEPTIDES:
            dp_counts[dp] += 1.0

    denominator = length - 1
    return [dp_counts[dp] / denominator for dp in DIPEPTIDES]


def extract_physicochemical(seq: str) -> list[float]:
    pep = peptides.Peptide(seq)
    return [
        len(seq),
        pep.molecular_weight(),
        pep.hydrophobicity(),
        pep.isoelectric_point(),
        pep.instability_index(),
        pep.aliphatic_index(),
        pep.boman(),
        pep.charge(pH=7.0),
    ]


def extract_features_dataframe(
    df: pd.DataFrame,
    feature_extractor: Callable[[str], list[float]],
) -> np.ndarray:
    """Extract features from DataFrame using a generic function."""

    return np.array([feature_extractor(seq) for seq in df["Seq"]])
