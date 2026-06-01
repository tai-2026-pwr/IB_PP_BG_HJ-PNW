from pathlib import Path

import pandas as pd
from Bio import SeqIO

from src.preprocessing.constatns import STANDARD_AA


def load_positive_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.dropna(subset=["Seq"]).copy()
    df["Seq"] = df["Seq"].astype(str).str.strip()
    df = df[df["Seq"] != ""]

    df["len"] = df["Seq"].str.len()
    df = df[df["len"] <= 50]

    df = df[df["Seq"].apply(lambda x: set(x).issubset(STANDARD_AA))]

    df = df.drop_duplicates(subset="Seq").copy()
    df["CPP"] = 1
    return df


def load_uniprot_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    df = df.dropna(subset=["Seq"]).copy()
    df["Seq"] = df["Seq"].astype(str).str.strip()
    df = df[df["Seq"] != ""]

    df["len"] = df["Seq"].str.len()
    df = df[df["len"] <= 50]

    df = df[df["Seq"].apply(lambda x: set(x).issubset(STANDARD_AA))]

    df = df.drop_duplicates(subset="Seq").copy()
    df["CPP"] = 0
    df["source"] = "uniprot"

    return df


def load_peptipedia_dataset(path: str | Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []

    for record in SeqIO.parse(path, "fasta"):
        records.append({"ID": str(record.id), "Seq": str(record.seq)})

    df = pd.DataFrame(records)
    df["Seq"] = df["Seq"].astype(str).str.strip()

    df["len"] = df["Seq"].str.len()
    df = df[df["len"] <= 50]

    df = df[df["Seq"].apply(lambda x: set(x).issubset(STANDARD_AA))]

    df = df.drop_duplicates(subset="Seq").copy()
    df["CPP"] = 0
    df["source"] = "peptipedia"

    return df


def build_negative_dataset(
    pos: pd.DataFrame,
    neg_uni: pd.DataFrame,
    neg_pep: pd.DataFrame,
    multiplier: int,
    random_state: int,
) -> pd.DataFrame:
    target_counts = pos["len"].value_counts()
    final_parts: list[pd.DataFrame] = []

    for length, count in target_counts.items():
        target = count * multiplier

        n_uni = target // 2
        n_pep = target - n_uni

        uni_subset = neg_uni[neg_uni["len"] == length]

        pep_subset = neg_pep[neg_pep["len"] == length]

        uni_sample = uni_subset.sample(n=min(n_uni, len(uni_subset)), random_state=random_state)

        pep_sample = pep_subset.sample(n=min(n_pep, len(pep_subset)), random_state=random_state)

        final_parts.append(uni_sample)
        final_parts.append(pep_sample)

    final_df = pd.concat(final_parts)

    final_df = final_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return final_df


def remove_overlaps(
    pos: pd.DataFrame, neg_uni: pd.DataFrame, neg_pep: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos_set = set(pos["Seq"])

    neg_uni = neg_uni[~neg_uni["Seq"].isin(pos_set)].copy()

    neg_pep = neg_pep[~neg_pep["Seq"].isin(pos_set)].copy()

    uni_set = set(neg_uni["Seq"])

    neg_pep = neg_pep[~neg_pep["Seq"].isin(uni_set)].copy()

    return neg_uni, neg_pep
