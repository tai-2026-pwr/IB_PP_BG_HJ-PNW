from __future__ import annotations

from dataclasses import asdict
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from src.constants import DEVICE
from src.experiments.classical_ml import (
    classical_ml_configs,
    representations_configs_classic_ml,
)
from src.experiments.dl import dl_model_configs
from src.experiments.one_class import (
    one_class_ml_configs,
    representations_configs_one_class,
)
from src.model_helpers.classic_ml import (
    ClassicalModel,
    evaluate_fitted_classic_ml,
    fit_final_classic_ml,
)
from src.model_helpers.dl.constants import BATCH_SIZE, EPOCHS, LR
from src.model_helpers.dl.dl import (
    DLModel,
    evaluate_dl_model,
    fit_final_dl,
    load_embeddings,
)
from src.model_helpers.export import (
    build_metadata,
    load_dl_model,
    load_sklearn_model,
    metadata_path_for,
    model_slug,
    save_dl_model,
    save_sklearn_model,
    write_manifest,
)
from src.model_helpers.one_class import (
    OneClassModel,
    evaluate_fitted_one_class,
    fit_final_one_class,
)
from src.paths import (
    EMBEDDINGS_PER_TOKEN_DIR,
    MODELS_DIR,
)
from src.preprocessing.constatns import ESM_MODEL_NAME, MAX_LEN
from src.utils import load_data as data_loader
from src.utils.metrics import Metrics
from src.utils.representations import Representation, extract_features_dataframe


def main() -> None:
    entries: list[dict[str, Any]] = []

    print("Exporting DL models...")
    for model_name, dataset_type, embeddings_dir in dl_model_configs:
        entries.append(export_dl_model(model_name, dataset_type, embeddings_dir))

    print("\nExporting classical ML models...")
    for representation, _ in representations_configs_classic_ml:
        for model_name, dataset_type, train_path, test_path in classical_ml_configs:
            entries.append(
                export_classical_model(
                    model_name,
                    dataset_type,
                    train_path,
                    test_path,
                    representation,
                )
            )

    print("\nExporting one-class models...")
    for representation, _ in representations_configs_one_class:
        for model_name, dataset_type, train_path, test_path in one_class_ml_configs:
            entries.append(
                export_one_class_model(
                    model_name,
                    dataset_type,
                    train_path,
                    test_path,
                    representation,
                )
            )

    manifest_path = write_manifest(MODELS_DIR, entries)

    print(f"\nExported {len(entries)} models.")
    print(f"Manifest: {manifest_path}")


def export_dl_model(
    model_name: DLModel,
    dataset_type: str,
    embeddings_dir: Path,
) -> dict[str, Any]:
    x_train, y_train, x_test, y_test = load_embeddings(
        embeddings_dir=embeddings_dir,
        dataset_type=dataset_type,
    )

    pooling = "per_token" if embeddings_dir == EMBEDDINGS_PER_TOKEN_DIR else "mean"
    model_id = (
        f"dl.{model_slug(Representation.ESM2.value)}.{model_slug(pooling)}."
        f"{model_slug(model_name.value)}.{dataset_type}"
    )
    artifact_path = (
        MODELS_DIR
        / "dl"
        / f"{model_slug(model_name.value)}_esm-2_{model_slug(pooling)}_{dataset_type}.pt"
    )

    print(f"  {model_id}", flush=True)
    model = fit_final_dl(model_name=model_name, x_train=x_train, y_train=y_train)
    metrics = evaluate_dl_model(model, x_test, y_test)

    metadata = build_metadata(
        model_id=model_id,
        artifact_type="torch_state_dict",
        model_name=model_name.value,
        representation=Representation.ESM2.value,
        dataset=dataset_type,
        input_shape=x_train.shape[1:],
        test_metrics=asdict(metrics),
        extra={
            "input_dim": int(x_train.shape[-1]),
            "hyperparameters": {
                "epochs": EPOCHS,
                "lr": LR,
                "batch_size": BATCH_SIZE,
            },
            "pooling": pooling,
            "esm_model_name": ESM_MODEL_NAME,
            "max_len": MAX_LEN,
            "threshold": 0.5,
            "positive_class": 1,
            "output_kind": "probability",
            "score_transform": "sigmoid",
            "training_set_size": len(y_train),
        },
    )

    save_dl_model(model, artifact_path, metadata)

    reloaded = load_dl_model(artifact_path).to(DEVICE)
    _assert_metrics_match(metrics, evaluate_dl_model(reloaded, x_test, y_test), model_id)

    return _manifest_entry(metadata, artifact_path)


def export_classical_model(
    model_name: ClassicalModel,
    dataset_type: str,
    train_path: Path,
    test_path: Path,
    representation: Representation,
) -> dict[str, Any]:
    x_train, y_train, x_test, y_test = _load_features(
        train_path=train_path,
        test_path=test_path,
        representation=representation,
    )

    representation_slug = model_slug(representation.value)
    model_id = f"classical.{representation_slug}.{model_slug(model_name.value)}.{dataset_type}"
    artifact_path = (
        MODELS_DIR
        / "classical_ml"
        / f"{model_slug(model_name.value)}_{representation_slug}_{dataset_type}.joblib"
    )

    print(f"  {model_id}", flush=True)
    model = fit_final_classic_ml(model_name, x_train, y_train)
    metrics = evaluate_fitted_classic_ml(model, x_test, y_test)

    metadata = build_metadata(
        model_id=model_id,
        artifact_type="sklearn_joblib",
        model_name=model_name.value,
        representation=representation.value,
        dataset=dataset_type,
        input_shape=(int(x_train.shape[1]),),
        test_metrics=asdict(metrics),
        extra={
            "feature_dim": int(x_train.shape[1]),
            "classes": _classes_of(model),
            "positive_class": 1,
            "threshold": 0.5,
            "output_kind": "probability",
            "score_transform": "predict_proba",
            "training_set_size": len(y_train),
        },
    )

    save_sklearn_model(model, artifact_path, metadata)

    reloaded = load_sklearn_model(artifact_path)
    _assert_metrics_match(
        metrics,
        evaluate_fitted_classic_ml(reloaded, x_test, y_test),
        model_id,
    )

    return _manifest_entry(metadata, artifact_path)


def export_one_class_model(
    model_name: OneClassModel,
    dataset_type: str,
    train_path: Path,
    test_path: Path,
    representation: Representation,
) -> dict[str, Any]:
    x_train, _, x_test, y_test = _load_features(
        train_path=train_path,
        test_path=test_path,
        representation=representation,
    )

    representation_slug = model_slug(representation.value)
    model_id = f"one-class.{representation_slug}.{model_slug(model_name.value)}.{dataset_type}"
    artifact_path = (
        MODELS_DIR
        / "one_class"
        / f"{model_slug(model_name.value)}_{representation_slug}_{dataset_type}.joblib"
    )

    print(f"  {model_id}", flush=True)
    model = fit_final_one_class(model_name, x_train)
    metrics = evaluate_fitted_one_class(model, x_test, y_test)

    metadata = build_metadata(
        model_id=model_id,
        artifact_type="sklearn_joblib",
        model_name=model_name.value,
        representation=representation.value,
        dataset=dataset_type,
        input_shape=(int(x_train.shape[1]),),
        test_metrics=asdict(metrics),
        extra={
            "feature_dim": int(x_train.shape[1]),
            "output_kind": "anomaly_score",
            "score_transform": "negated_decision_function",
            "anomaly_label": 1,
            "normal_label": 0,
            "training_set_size": len(x_train),
        },
    )

    save_sklearn_model(model, artifact_path, metadata)

    reloaded = load_sklearn_model(artifact_path)
    _assert_metrics_match(
        metrics,
        evaluate_fitted_one_class(reloaded, x_test, y_test),
        model_id,
    )

    return _manifest_entry(metadata, artifact_path)


@cache
def _load_features(
    train_path: Path,
    test_path: Path,
    representation: Representation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = data_loader.load_data(train_path)
    test_df = data_loader.load_data(test_path)

    x_train = extract_features_dataframe(train_df, representation)
    y_train = train_df["CPP"].astype(int).to_numpy()
    x_test = extract_features_dataframe(test_df, representation)
    y_test = test_df["CPP"].astype(int).to_numpy()

    return x_train, y_train, x_test, y_test


def _assert_metrics_match(expected: Metrics, actual: Metrics, model_id: str) -> None:
    mismatches = [
        f"{metric}: {expected[metric]!r} != {actual[metric]!r}"
        for metric in Metrics.keys()
        if not np.isclose(expected[metric], actual[metric], rtol=1e-6, atol=1e-9)
    ]

    if mismatches:
        raise RuntimeError(
            f"Reloaded model {model_id!r} does not reproduce its metrics: " + "; ".join(mismatches)
        )


def _classes_of(model: Any) -> list[int] | None:
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "steps"):
        classes = getattr(model.steps[-1][1], "classes_", None)

    return [int(cls) for cls in classes] if classes is not None else None


def _manifest_entry(metadata: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    return {
        "model_id": metadata["model_id"],
        "artifact": artifact_path.relative_to(MODELS_DIR).as_posix(),
        "metadata": metadata_path_for(artifact_path).relative_to(MODELS_DIR).as_posix(),
        "artifact_type": metadata["artifact_type"],
        "model_name": metadata["model_name"],
        "representation": metadata["representation"],
        "dataset": metadata["dataset"],
    }


if __name__ == "__main__":
    main()
