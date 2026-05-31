from __future__ import annotations

import pandas as pd

from src.experiments.configs import classical_ml_configs, representations_configs_classic_ml
from src.model_helpers.fold_train import fold_train
from src.model_helpers.sklearn import evaluate_test_set_classic_ml
from src.utils import load_data as data_loader
from src.utils.representations import extract_aac, extract_features_dataframe
from src.utils.runner import (
    save_results,
)

if __name__ == "__main__":
    for representation, output_dir in representations_configs_classic_ml:
        all_folds = []
        all_summaries = []

        for model_name, dataset_type, train_path, test_path in classical_ml_configs:
            train_df = data_loader.load_data(train_path)
            test_df = data_loader.load_data(test_path)

            x_train = extract_features_dataframe(train_df, extract_aac)
            y_train = train_df["CPP"].astype(int).to_numpy()
            x_test = extract_features_dataframe(test_df, extract_aac)
            y_test = test_df["CPP"].astype(int).to_numpy()

            fold_results, summary = fold_train(
                model_name=model_name,
                dataset_type=dataset_type,
                representation=representation,
                x_train=x_train,
                y_train=y_train,
                n_splits=10,
            )

            test_row = evaluate_test_set_classic_ml(
                model_name=model_name,
                representation=representation,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )

            summary = pd.concat([summary, test_row], ignore_index=True)
            all_folds.append(fold_results)
            all_summaries.append(summary)

        final_folds = pd.concat(all_folds, ignore_index=True)
        final_summary = pd.concat(all_summaries, ignore_index=True)

        save_results(final_folds, final_summary, output_dir)

        print(f"{representation} experiments completed.")
        print(final_summary)
        print("Results saved in:", output_dir)
