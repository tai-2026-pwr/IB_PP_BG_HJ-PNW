from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.constants import AMINO_ACIDS, RANDOM_STATE
from src.paths import (
    TEST_BALANCED_PATH,
    TEST_IMBALANCED_PATH,
    TRAIN_BALANCED_PATH,
    TRAIN_IMBALANCED_PATH,
)

OUTPUT_DIR = Path("results/aac")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    df = df[["Seq", "CPP"]].dropna().copy()
    df["Seq"] = df["Seq"].astype(str).str.strip().str.upper()
    df = df[df["Seq"] != ""].copy()
    return df


def extract_aac(seq):
    length = len(seq)
    return [seq.count(aa) / length for aa in AMINO_ACIDS]


def build_features(df):
    return np.array([extract_aac(seq) for seq in df["Seq"]])


MODELS = {
    "SVM": SVC,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
}


def make_model(model_name):
    if model_name == "SVM":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MODELS[model_name](probability=True, random_state=RANDOM_STATE)),
            ]
        )
    return MODELS[model_name](random_state=RANDOM_STATE)


def calculate_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    return calculate_metrics(y, y_pred, y_score)


def run_single_experiment(train_df, test_df, model_name, dataset_type):
    X_train = build_features(train_df)
    y_train = train_df["CPP"].astype(int).to_numpy()
    X_test = build_features(test_df)
    y_test = test_df["CPP"].astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []

    for fold_number, (train_index, val_index) in enumerate(cv.split(X_train, y_train), start=1):
        X_fold_train, y_fold_train = X_train[train_index], y_train[train_index]
        X_fold_val, y_fold_val = X_train[val_index], y_train[val_index]

        model = make_model(model_name)
        model.fit(X_fold_train, y_fold_train)
        metrics = evaluate_model(model, X_fold_val, y_fold_val)

        row = {
            "model": model_name,
            "representation": "AAC",
            "dataset": dataset_type,
            "fold": fold_number,
        }
        row.update(metrics)
        fold_rows.append(row)

    fold_results = pd.DataFrame(fold_rows)
    metric_names = ["accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc"]

    mean_row = {
        "model": model_name,
        "representation": "AAC",
        "dataset": dataset_type,
        "stage": "cv_mean",
    }
    std_row = {
        "model": model_name,
        "representation": "AAC",
        "dataset": dataset_type,
        "stage": "cv_std",
    }

    for metric in metric_names:
        mean_row[metric] = fold_results[metric].mean()
        std_row[metric] = fold_results[metric].std()

    final_model = make_model(model_name)
    final_model.fit(X_train, y_train)

    test_metrics = evaluate_model(final_model, X_test, y_test)
    test_row = {
        "model": model_name,
        "representation": "AAC",
        "dataset": dataset_type,
        "stage": "test",
    }
    test_row.update(test_metrics)

    summary = pd.DataFrame([mean_row, std_row, test_row])
    return fold_results, summary


train_balanced = load_data(TRAIN_BALANCED_PATH)
train_imbalanced = load_data(TRAIN_IMBALANCED_PATH)
test_balanced = load_data(TEST_BALANCED_PATH)
test_imbalanced = load_data(TEST_IMBALANCED_PATH)

# Model 1:  SVM,               AAC, balanced
# Model 2:  SVM,               AAC, imbalanced
# Model 3:  Random Forest,     AAC, balanced
# Model 4:  Random Forest,     AAC, imbalanced
# Model 5:  Gradient Boosting, AAC, balanced
# Model 6:  Gradient Boosting, AAC, imbalanced
configs = [
    ("SVM", "balanced", train_balanced, test_balanced),
    ("SVM", "imbalanced", train_imbalanced, test_imbalanced),
    ("Random Forest", "balanced", train_balanced, test_balanced),
    ("Random Forest", "imbalanced", train_imbalanced, test_imbalanced),
    ("Gradient Boosting", "balanced", train_balanced, test_balanced),
    ("Gradient Boosting", "imbalanced", train_imbalanced, test_imbalanced),
]

all_folds = []
all_summaries = []

for model_name, dataset_type, train_df, test_df in configs:
    fold_results, summary = run_single_experiment(train_df, test_df, model_name, dataset_type)
    all_folds.append(fold_results)
    all_summaries.append(summary)

final_folds = pd.concat(all_folds, ignore_index=True)
final_summary = pd.concat(all_summaries, ignore_index=True)

final_folds.to_csv(OUTPUT_DIR / "cv_folds.csv", index=False)
final_summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

print("Models 1-6 (AAC) experiments completed.")
print(final_summary)
print("Results saved in:", OUTPUT_DIR)
