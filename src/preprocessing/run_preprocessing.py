from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.plots.plots import plot_length_distribution, plot_positive_vs_negative
from src.preprocessing.dataset_preprocessing import (
    load_positive_dataset,
    load_uniprot_dataset,
    load_peptipedia_dataset,
    remove_overlaps,
    build_negative_dataset,
)

RANDOM_STATE = 50
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

POSITIVE_PATH = RAW_DIR / "cpp.csv"
UNIPROT_PATH = RAW_DIR / "uniprot.xlsx"
PEPTIPEDIA_PATH = RAW_DIR / "peptipedia.fasta"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

pos = load_positive_dataset(POSITIVE_PATH)
neg_uni = load_uniprot_dataset(UNIPROT_PATH)
neg_pep = load_peptipedia_dataset(PEPTIPEDIA_PATH)

neg_uni, neg_pep = remove_overlaps(pos, neg_uni, neg_pep)

neg_dataset_1 = build_negative_dataset(
    pos=pos,
    neg_uni=neg_uni,
    neg_pep=neg_pep,
    multiplier=1,
    random_state=RANDOM_STATE,
)

neg_dataset_2 = build_negative_dataset(
    pos=pos,
    neg_uni=neg_uni,
    neg_pep=neg_pep,
    multiplier=2,
    random_state=RANDOM_STATE,
)

pos_final = pos[["Seq", "len", "CPP"]].copy()
pos_final["source"] = "cppsite3"

if "id" in pos.columns:
    pos_final["ID"] = pos["id"]
else:
    pos_final["ID"] = range(len(pos_final))

pos_final = pos_final[["Seq", "len", "CPP", "source", "ID"]]
neg_dataset_1 = neg_dataset_1[["Seq", "len", "CPP", "source", "ID"]].copy()
neg_dataset_2 = neg_dataset_2[["Seq", "len", "CPP", "source", "ID"]].copy()

dataset_1to1 = pd.concat([pos_final, neg_dataset_1], ignore_index=True)
dataset_1to1 = dataset_1to1.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

dataset_1to2 = pd.concat([pos_final, neg_dataset_2], ignore_index=True)
dataset_1to2 = dataset_1to2.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

pos_final.to_csv(PROCESSED_DIR / "positive_dataset_clean.csv", index=False)
neg_dataset_1.to_csv(PROCESSED_DIR / "negative_dataset_1.csv", index=False)
neg_dataset_2.to_csv(PROCESSED_DIR / "negative_dataset_2.csv", index=False)
dataset_1to1.to_csv(PROCESSED_DIR / "dataset_CPP_1to1.csv", index=False)
dataset_1to2.to_csv(PROCESSED_DIR / "dataset_CPP_1to2.csv", index=False)

train_1to1, test_1to1 = train_test_split(
    dataset_1to1,
    test_size=0.2,
    stratify=dataset_1to1["CPP"],
    random_state=RANDOM_STATE,
)

train_1to2, test_1to2 = train_test_split(
    dataset_1to2,
    test_size=0.2,
    stratify=dataset_1to2["CPP"],
    random_state=RANDOM_STATE,
)

PLOTS_DIR = PROCESSED_DIR / "plots"

plot_length_distribution(
    pos_final,
    "Positive dataset length distribution",
    PLOTS_DIR / "positive_lengths.png",
)

plot_length_distribution(
    neg_dataset_1,
    "Negative dataset 1 length distribution",
    PLOTS_DIR / "negative_dataset_1_lengths.png",
)

plot_length_distribution(
    neg_dataset_2,
    "Negative dataset 2 length distribution",
    PLOTS_DIR / "negative_dataset_2_lengths.png",
)

plot_positive_vs_negative(
    pos_final,
    neg_dataset_1,
    "negative 1",
    "Positive vs negative dataset 1",
    PLOTS_DIR / "positive_vs_negative_1.png",
)

plot_positive_vs_negative(
    pos_final,
    neg_dataset_2,
    "negative 2",
    "Positive vs negative dataset 2",
    PLOTS_DIR / "positive_vs_negative_2.png",
)

print("Preprocessing completed.")
print("Positive:", len(pos_final))
print("Negative 1:", len(neg_dataset_1))
print("Negative 2:", len(neg_dataset_2))
print("Final 1:1:", dataset_1to1["CPP"].value_counts().to_dict())
print("Final 1:2:", dataset_1to2["CPP"].value_counts().to_dict())

train_1to1.to_csv(
    PROCESSED_DIR / "dataset_CPP_1to1_train.csv",
    index=False,
)

test_1to1.to_csv(
    PROCESSED_DIR / "dataset_CPP_1to1_test.csv",
    index=False,
)

train_1to2.to_csv(
    PROCESSED_DIR / "dataset_CPP_1to2_train.csv",
    index=False,
)

test_1to2.to_csv(
    PROCESSED_DIR / "dataset_CPP_1to2_test.csv",
    index=False,
)