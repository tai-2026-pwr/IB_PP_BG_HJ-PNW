from __future__ import annotations

import json
import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import torch
import torch.nn as nn

from src.model_helpers.dl.dl import DLModel, build_dl_model

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

SCHEMA_VERSION = 1


def model_slug(value: str) -> str:
    """Convert a model or representation name into a filesystem-friendly slug."""
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def metadata_path_for(artifact_path: str | Path) -> Path:
    """Return the JSON metadata path that belongs to an artifact."""
    return Path(artifact_path).with_suffix(".json")


def build_metadata(
    *,
    model_id: str,
    artifact_type: str,
    model_name: str,
    representation: str,
    dataset: str,
    input_shape: Sequence[int],
    test_metrics: Mapping[str, float],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the metadata sidecar shared by every exported model."""
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "artifact_type": artifact_type,
        "model_name": model_name,
        "representation": representation,
        "dataset": dataset,
        "input_shape": list(input_shape),
        "test_metrics": dict(test_metrics),
        "versions": _versions(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    if extra:
        metadata.update(extra)

    return metadata


def read_metadata(artifact_path: str | Path) -> dict[str, Any]:
    """Read the metadata sidecar belonging to an artifact."""
    return json.loads(metadata_path_for(artifact_path).read_text(encoding="utf-8"))


def save_dl_model(
    model: nn.Module,
    artifact_path: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Save a PyTorch state dict and its metadata sidecar."""
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"state_dict": model.state_dict()}, path)
    _write_metadata(path, metadata)

    return path


def load_dl_model(artifact_path: str | Path) -> nn.Module:
    """Rebuild a PyTorch model from an exported state dict."""
    path = Path(artifact_path)
    metadata = read_metadata(path)

    model = build_dl_model(
        DLModel(metadata["model_name"]),
        input_dim=int(metadata["input_dim"]),
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def save_sklearn_model(
    model: BaseEstimator,
    artifact_path: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Save a fitted scikit-learn estimator and its metadata sidecar."""
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)
    _write_metadata(path, metadata)

    return path


def load_sklearn_model(artifact_path: str | Path) -> BaseEstimator:
    """Load a fitted scikit-learn estimator."""
    return joblib.load(artifact_path)


def write_manifest(
    models_dir: str | Path,
    entries: Sequence[Mapping[str, Any]],
) -> Path:
    """Write the manifest that maps stable model IDs to exported artifacts."""
    directory = Path(models_dir)
    directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "models": sorted(entries, key=lambda entry: entry["model_id"]),
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return manifest_path


def _write_metadata(artifact_path: Path, metadata: Mapping[str, Any]) -> None:
    metadata_path_for(artifact_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "numpy": _package_version("numpy"),
        "scikit-learn": _package_version("scikit-learn"),
        "transformers": _package_version("transformers"),
        "peptides": _package_version("peptides"),
    }


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"
