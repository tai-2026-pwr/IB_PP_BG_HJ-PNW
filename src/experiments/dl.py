from __future__ import annotations

import pandas as pd

from src.model_helpers.dl.dl import DLModel, evaluate_test_set_dl, load_embeddings
from src.model_helpers.fold_train import fold_train
from src.paths import EMBEDDINGS_MEAN_DIR, EMBEDDINGS_PER_TOKEN_DIR
from src.utils.representations import Representation
from src.utils.runner import build_output_dir, save_results

output_dir_dl = build_output_dir("results/dl")

# Cleaned up configs: Just the Enum, the dataset type, and the embeddings path
dl_model_configs = [
    (DLModel.CNN1D, "balanced", EMBEDDINGS_PER_TOKEN_DIR),
    (DLModel.CNN1D, "imbalanced", EMBEDDINGS_PER_TOKEN_DIR),
    (DLModel.MLP, "balanced", EMBEDDINGS_MEAN_DIR),
    (DLModel.MLP, "imbalanced", EMBEDDINGS_MEAN_DIR),
]

if __name__ == "__main__":
    all_folds = []
    all_summaries = []

    for model_name, dataset_type, embeddings_dir in dl_model_configs:
        print(f"Running {model_name} on {dataset_type} dataset...")

        # Load pre-computed embeddings
        x_train, y_train, x_test, y_test = load_embeddings(
            embeddings_dir=embeddings_dir, dataset_type=dataset_type
        )

        fold_results, summary = fold_train(
            model_name=model_name,
            dataset_type=dataset_type,
            representation=Representation.ESM2,
            x_train=x_train,
            y_train=y_train,
            n_splits=10,
        )

        test_row = evaluate_test_set_dl(
            model_name=model_name,
            representation=Representation.ESM2,
            dataset_type=dataset_type,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

        summary = pd.concat([summary, test_row], ignore_index=True)
        all_folds.append(fold_results)
        all_summaries.append(summary)

    # Aggregate and save everything
    final_folds = pd.concat(all_folds, ignore_index=True)
    final_summary = pd.concat(all_summaries, ignore_index=True)

    save_results(final_folds, final_summary, output_dir_dl)

    print(f"\nDL {Representation.ESM2} experiments completed.")
    print(final_summary)
    print("Results saved in:", output_dir_dl)
