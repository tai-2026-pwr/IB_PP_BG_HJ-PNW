from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAIN_METRIC = "balanced_accuracy"
CV_FILES = [
    RESULTS_DIR / "aac" / "cv_folds.csv",
    RESULTS_DIR / "physicochemical" / "cv_folds.csv",
    RESULTS_DIR / "one_class" / "one_class_cv_folds.csv",
    RESULTS_DIR / "cnn" / "cv_folds.csv",]

all_data = []

for path in CV_FILES:
    if path.exists():
        df_part = pd.read_csv(path)

        if "dataset" not in df_part.columns:
            df_part["dataset"] = "balanced"

        all_data.append(df_part)

df = pd.concat(all_data, ignore_index=True)
configs = df[["model", "representation", "dataset"]].drop_duplicates().values
results = []

for i in range(len(configs)):
    model_1, representation_1, dataset_1 = configs[i]

    df_1 = df[
        (df["model"] == model_1)
        & (df["representation"] == representation_1)
        & (df["dataset"] == dataset_1)
    ].sort_values("fold")

    scores_1 = df_1[MAIN_METRIC].values
    for j in range(i + 1, len(configs)):
        model_2, representation_2, dataset_2 = configs[j]

        df_2 = df[(df["model"] == model_2)
            & (df["representation"] == representation_2)
            & (df["dataset"] == dataset_2)].sort_values("fold")

        scores_2 = df_2[MAIN_METRIC].values
        if len(scores_1) != len(scores_2):
            continue
        if len(scores_1) < 2 or len(scores_2) < 2:
            continue
        statistic, p_value = wilcoxon(scores_1, scores_2)
        mean_1 = scores_1.mean()
        mean_2 = scores_2.mean()

        if mean_1 >= mean_2:
            better_model = f"{model_1} | {representation_1} | {dataset_1}"
        else:
            better_model = f"{model_2} | {representation_2} | {dataset_2}"

        row = {"model_1": model_1,
            "representation_1": representation_1,
            "dataset_1": dataset_1,
            "model_2": model_2,
            "representation_2": representation_2,
            "dataset_2": dataset_2,
            "metric": MAIN_METRIC,
            "mean_1": mean_1,
            "mean_2": mean_2,
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "better_model": better_model,}

        results.append(row)
results_df = pd.DataFrame(results)

if len(results_df) > 0:
    results_df = results_df.sort_values("p_value")

output_path = OUTPUT_DIR / "wilcoxon_results_balanced_accuracy.csv"
results_df.to_csv(output_path, index=False)

#to do: maybe use more folds to get significant results