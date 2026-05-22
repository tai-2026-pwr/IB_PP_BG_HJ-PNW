from pathlib import Path
import numpy as np
import pandas as pd
import peptides

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

RANDOM_STATE = 50
TRAIN_PATH = Path("data/processed/dataset_CPP_1to1_train.csv")
TEST_PATH = Path("data/processed/dataset_CPP_1to1_test.csv")
OUTPUT_DIR = Path("results/one_class")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def load_data(path):
    df = pd.read_csv(path)
    df = df[["Seq", "CPP"]].dropna().copy()
    df["Seq"] = df["Seq"].astype(str).str.strip().str.upper()
    df = df[df["Seq"] != ""].copy()
    return df

def extract_aac(seq):
    features = []
    length = len(seq)

    for aa in AMINO_ACIDS:
        value = seq.count(aa) / length
        features.append(value)
    return features

def extract_physicochemical(seq):
    pep = peptides.Peptide(seq)
    return [
        len(seq),
        pep.molecular_weight(),
        pep.hydrophobicity(),
        pep.isoelectric_point(),
        pep.instability_index(),
        pep.aliphatic_index(),
        pep.boman(),
        pep.charge(pH=7.0),]

def build_features(df, representation):
    features = []
    for seq in df["Seq"]:
        if representation == "AAC":
            row = extract_aac(seq)
        else:
            row = extract_physicochemical(seq)
        features.append(row)
    return np.array(features)

def make_model(model_name):
    if model_name == "One-Class SVM":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)),])
        return model

    if model_name == "Isolation Forest":
        model = IsolationForest(
            n_estimators=200,
            contamination=0.1,
            random_state=RANDOM_STATE,
        )
        return model
    raise ValueError("Unknown model name")

def convert_predictions(raw_predictions):
    return np.where(raw_predictions == 1, 1, 0)

def calculate_metrics(y_true, y_pred, y_score):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),}
    return results

def evaluate_model(model, X, y):
    raw_predictions = model.predict(X)
    y_pred = convert_predictions(raw_predictions)
    y_score = model.decision_function(X)
    metrics = calculate_metrics(y, y_pred, y_score)

    return metrics

def train_one_class_model(model, X_train, y_train):
    X_positive = X_train[y_train == 1]
    model.fit(X_positive)
    return model

def run_single_experiment(train_df, test_df, representation, model_name):
    X_train = build_features(train_df, representation)
    y_train = train_df["CPP"].astype(int).to_numpy()
    X_test = build_features(test_df, representation)
    y_test = test_df["CPP"].astype(int).to_numpy()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,)
    fold_rows = []

    for fold_number, split in enumerate(cv.split(X_train, y_train), start=1):
        train_index = split[0]
        val_index = split[1]

        X_fold_train = X_train[train_index]
        y_fold_train = y_train[train_index]
        X_fold_val = X_train[val_index]
        y_fold_val = y_train[val_index]

        model = make_model(model_name)
        model = train_one_class_model(model, X_fold_train, y_fold_train)
        metrics = evaluate_model(model, X_fold_val, y_fold_val)

        row = {
            "model": model_name,
            "representation": representation,
            "fold": fold_number,}
        row.update(metrics)
        fold_rows.append(row)

    fold_results = pd.DataFrame(fold_rows)

    mean_row = {
        "model": model_name,
        "representation": representation,
        "stage": "cv_mean",
    }
    std_row = {
        "model": model_name,
        "representation": representation,
        "stage": "cv_std",
    }
    metric_names = ["accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc"]

    for metric in metric_names:
        mean_row[metric] = fold_results[metric].mean()
        std_row[metric] = fold_results[metric].std()

    final_model = make_model(model_name)
    final_model = train_one_class_model(final_model, X_train, y_train)

    test_metrics = evaluate_model(final_model, X_test, y_test)
    test_row = {
        "model": model_name,
        "representation": representation,
        "stage": "test",
    }

    test_row.update(test_metrics)
    summary = pd.DataFrame([mean_row, std_row, test_row])
    return fold_results, summary

train_df = load_data(TRAIN_PATH)
test_df = load_data(TEST_PATH)

configs = [
    ("AAC", "One-Class SVM"),
    ("Physicochemical", "One-Class SVM"),
    ("AAC", "Isolation Forest"),
    ("Physicochemical", "Isolation Forest"),]

all_folds = []
all_summaries = []

for representation, model_name in configs:
    fold_results, summary = run_single_experiment(
        train_df,
        test_df,
        representation,
        model_name,
    )
    all_folds.append(fold_results)
    all_summaries.append(summary)

final_folds = pd.concat(all_folds, ignore_index=True)
final_summary = pd.concat(all_summaries, ignore_index=True)

final_folds.to_csv(OUTPUT_DIR / "one_class_cv_folds.csv", index=False)
final_summary.to_csv(OUTPUT_DIR / "one_class_summary.csv", index=False)

print("One-class experiments completed.")
print(final_summary)
print("Results saved in:", OUTPUT_DIR)