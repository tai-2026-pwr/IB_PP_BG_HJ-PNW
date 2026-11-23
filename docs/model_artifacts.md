# Exported Model Artifacts

This document describes the artifacts produced by:

```bash
python -m src.experiments.export_final_models
```

The export is implemented in `src/experiments/export_final_models.py` and the
serialization helpers are in `src/model_helpers/export.py`.

## What is exported

For every configured model, representation, and dataset type, the export
trains one **final model on the full training split** and writes it to disk.

The exported model is:

- trained on `train_*` only,
- evaluated once against the held-out test split before being saved,
- stored with its test metrics in a metadata sidecar.

The `test_score` metrics stored in the metadata therefore remain a valid
held-out estimate for that exported model.

## Artifact formats

### PyTorch `.pt` files

The `.pt` file contains only a state dictionary:

```python
{"state_dict": model.state_dict()}
```

It does **not** contain:

- the `MLP` or `CNN1D` architecture definition,
- the ESM-2 model or tokenizer,
- the embedding preprocessing,
- the threshold or metadata.

The JSON sidecar and the repository code are required to reconstruct a usable
model.

The state dict is device-independent when loaded with `map_location="cpu"`.
The caller can then move the model to the serving device.

### scikit-learn `.joblib` files

The `.joblib` file contains the complete fitted scikit-learn estimator or
pipeline.

- `SVM` is a `Pipeline(StandardScaler, SVC)`; the scaler is included.
- `One-Class SVM` is a `Pipeline(StandardScaler, OneClassSVM)`; the scaler is
  included.
- `Random Forest`, `Gradient Boosting`, and `Isolation Forest` are stored
  directly.

The artifact is a Python pickle under the hood and depends on compatible
`scikit-learn` and NumPy versions.

### JSON metadata sidecars

Every artifact has a matching `.json` file with the same stem:

```text
svm_aac_balanced.joblib
svm_aac_balanced.json
```

The metadata is the contract for using the artifact. It records the model
family, representation, input shape, preprocessing assumptions, threshold,
test metrics, and package versions.

### `manifest.json`

The manifest maps stable model IDs to artifact and metadata paths:

```json
{
  "schema_version": 1,
  "models": [
    {
      "model_id": "classical.aac.svm.balanced",
      "artifact": "classical_ml/svm_aac_balanced.joblib",
      "metadata": "classical_ml/svm_aac_balanced.json",
      "artifact_type": "sklearn_joblib",
      "model_name": "SVM",
      "representation": "AAC",
      "dataset": "balanced"
    }
  ]
}
```

Paths in the manifest are relative to `artifacts/models/`. Use the manifest for
model discovery rather than hardcoding filenames.

## Model IDs

Model IDs follow these patterns:

```text
dl.esm-2.per-token.1d-cnn.balanced
dl.esm-2.mean.mlp.balanced
classical.aac.svm.balanced
classical.dpc.random-forest.imbalanced
classical.physicochemical.gradient-boosting.balanced
one-class.aac.one-class-svm.balanced
one-class.aac.isolation-forest.balanced
```

The components are:

1. Model family: `dl`, `classical`, or `one-class`.
2. Representation: `esm-2`, `aac`, `dpc`, or `physicochemical`.
3. For DL only: pooling mode, `mean` or `per-token`.
4. Model name.
5. Dataset type: `balanced` or `imbalanced`.

There is currently no `best` flag in the manifest. If an application needs a
single production model, choose it explicitly from the test metrics or from an
external model-selection decision.

## Metadata reference

### Common fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Metadata schema version, currently `1`. |
| `model_id` | Stable identifier used by the manifest and backend. |
| `artifact_type` | `torch_state_dict` or `sklearn_joblib`. |
| `model_name` | Human-readable model name, such as `1D-CNN` or `SVM`. |
| `representation` | `ESM-2`, `AAC`, `DPC`, or `Physicochemical`. |
| `dataset` | `balanced` or `imbalanced`. |
| `input_shape` | Shape of one model input after preprocessing. |
| `test_metrics` | Accuracy, balanced accuracy, F1, MCC, and ROC AUC on the test split. |
| `versions` | Python, torch, NumPy, scikit-learn, transformers, and peptides versions. |
| `created_at` | Export timestamp in UTC. |

### DL-specific fields

| Field | Meaning |
| --- | --- |
| `input_dim` | Size of the final embedding dimension, normally `1280`. |
| `hyperparameters` | `epochs`, `lr`, and `batch_size` used for training. |
| `pooling` | `mean` for the MLP, `per_token` for the 1D-CNN. |
| `esm_model_name` | Hugging Face ESM-2 checkpoint, normally `facebook/esm2_t33_650M_UR50D`. |
| `max_len` | Maximum tokenized sequence length, `50`. |
| `threshold` | Classification threshold, `0.5`. |
| `positive_class` | `1`, corresponding to CPP. |
| `output_kind` | `probability`. |
| `score_transform` | `sigmoid`. |

For DL:

- Mean-pooled input shape is `(1280,)`.
- Per-token input shape is `(50, 1280)`.
- The model returns logits. Apply `sigmoid` to obtain the CPP probability.

### Classical ML fields

| Field | Meaning |
| --- | --- |
| `feature_dim` | Number of input features. |
| `classes` | Class order recorded by the fitted estimator, normally `[0, 1]`. |
| `positive_class` | `1`, corresponding to CPP. |
| `threshold` | Classification threshold, `0.5`. |
| `output_kind` | `probability`. |
| `score_transform` | `predict_proba`. |

Feature dimensions:

- AAC: `20`
- DPC: `400`
- Physicochemical: `8`

### One-class fields

| Field | Meaning |
| --- | --- |
| `feature_dim` | Number of input features. |
| `output_kind` | `anomaly_score`. |
| `score_transform` | `negated_decision_function`. |
| `anomaly_label` | `1`. |
| `normal_label` | `0`. |

One-class models do not produce calibrated probabilities. The score is:

```python
anomaly_score = -model.decision_function(x)
```

Higher scores mean more anomalous / more CPP-like. The model's own
`predict()` output determines the label:

```python
raw = model.predict(x)
label = (raw == -1).astype(int)
```

Note that the current experiments fit one-class models on the full training set,
including negative examples, and use the labels only for evaluation. The
exported artifact reflects that existing training behavior.

## Inference contract

The model artifacts contain only the model. A backend must reproduce the same
preprocessing that was used during training.

```text
raw amino-acid sequence
        |
        v
input validation
        |
        v
representation extraction
  - AAC / DPC / physicochemical features, or
  - ESM-2 mean / per-token embeddings
        |
        v
exported model
        |
        v
score and label
```

### Input validation

Training data was restricted to:

- uppercase amino-acid sequences,
- standard amino acids `ACDEFGHIKLMNPQRSTVWY`,
- length at most 50.

Reject or normalize invalid input before calling the model. ESM tokenization
truncates to `max_len=50`, but the classical feature extractors do not truncate
automatically.

### Classical ML inference

```python
import json
from pathlib import Path

import joblib
import pandas as pd

from src.utils.representations import (
    Representation,
    extract_features_dataframe,
)

artifact_path = Path("artifacts/models/classical_ml/svm_aac_balanced.joblib")
metadata = json.loads(artifact_path.with_suffix(".json").read_text())

model = joblib.load(artifact_path)
representation = Representation(metadata["representation"])

sequences = ["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"]
x = extract_features_dataframe(
    pd.DataFrame({"Seq": sequences}),
    representation,
)

probabilities = model.predict_proba(x)[:, list(model.classes_).index(1)]
labels = (probabilities >= metadata["threshold"]).astype(int)
```

If the estimator is a pipeline, `model.classes_` may be delegated to the final
estimator. If necessary, read the classes from
`model.steps[-1][1].classes_`.

### DL inference

The DL artifact contains only the small classification head. ESM-2 must be
loaded separately.

```python
import json
from pathlib import Path

import torch
from transformers import EsmModel, EsmTokenizer

from src.constants import DEVICE
from src.model_helpers.dl.dl import DLModel, build_dl_model
from src.preprocessing.extract_embeddings import get_embeddings

artifact_path = Path(
    "artifacts/models/dl/1d-cnn_esm-2_per-token_balanced.pt"
)
metadata = json.loads(artifact_path.with_suffix(".json").read_text())

device = DEVICE

head = build_dl_model(
    DLModel(metadata["model_name"]),
    input_dim=metadata["input_dim"],
)
checkpoint = torch.load(
    artifact_path,
    map_location="cpu",
    weights_only=True,
)
head.load_state_dict(checkpoint["state_dict"])
head.to(device).eval()

tokenizer = EsmTokenizer.from_pretrained(metadata["esm_model_name"])
esm = EsmModel.from_pretrained(metadata["esm_model_name"]).to(device).eval()

mean_embeddings, per_token_embeddings = get_embeddings(
    sequences,
    tokenizer,
    esm,
)
x = (
    per_token_embeddings
    if metadata["pooling"] == "per_token"
    else mean_embeddings
)

with torch.inference_mode():
    logits = head(torch.tensor(x, dtype=torch.float32, device=device))
    probabilities = torch.sigmoid(logits).cpu().numpy()

labels = (probabilities >= metadata["threshold"]).astype(int)
```

All DL artifacts share the same ESM-2 backbone, so load the tokenizer and ESM
model once and reuse them across the MLP and CNN heads.

`get_embeddings` moves its inputs to `src.constants.DEVICE`. Keep the ESM model
and classification head on that same device, or replace the helper with an
embedder that accepts an explicit device.

### One-class inference

```python
import joblib

model = joblib.load(
    "artifacts/models/one_class/one-class-svm_aac_balanced.joblib"
)

anomaly_scores = -model.decision_function(x)
raw_predictions = model.predict(x)
labels = (raw_predictions == -1).astype(int)
```

Do not call `predict_proba` on a one-class artifact.

## Using the artifacts from FastAPI or Flama

Load models once during application startup. Do not load joblib or PyTorch
weights inside each request.

### FastAPI lifespan example

`build_runtime` below is the loader for the selected artifact(s): it should read
the manifest, load the model(s) once, and preload ESM-2 if a DL model is served.

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool


class PredictRequest(BaseModel):
    sequences: list[str]


class PredictResponse(BaseModel):
    label: int
    probability: float


runtime = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime
    runtime = build_runtime("artifacts/models")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=list[PredictResponse])
async def predict(request: PredictRequest):
    # ESM/torch inference is blocking; keep it off the event loop.
    return await run_in_threadpool(runtime.predict, request.sequences)
```

The same pattern applies to Flama: initialize the runtime in the Flama startup
hook, keep it on the application state, and call it from the route handler.
Flama can also mount the FastAPI application if the same endpoint should be
reused.

### Runtime responsibilities

The backend runtime should:

1. Read `manifest.json`.
2. Resolve the requested `model_id` to its artifact and metadata paths.
3. Load the selected model once at startup.
4. Load ESM-2 once if any DL model is enabled.
5. Keep PyTorch models in `.eval()` mode.
6. Run inference under `torch.inference_mode()`.
7. Move inputs and models to the same device.
8. Apply the representation-specific preprocessing.
9. Return the score and derived label.

If the API exposes multiple models, put `model_id` in the request or define a
fixed default model in configuration. Loading all 28 artifacts is usually
unnecessary; load only the models the service actually serves.

## Deployment checklist

- Copy `artifacts/models/` into the deployment image or artifact store.
- Install the runtime dependencies: `torch`, `transformers`, `scikit-learn`,
  `joblib`, `numpy`, `pandas`, and `peptides`.
- Make the repository's model definitions and feature extractors available to
  the backend. A `state_dict` cannot be loaded without the matching `MLP` or
  `CNN1D` class.
- Pre-download or pre-populate the Hugging Face ESM-2 checkpoint if the backend
  has no internet access.
- Keep one ESM-2 instance per process. A 650M-parameter model is roughly 2.5 GB
  in fp32 and dominates startup time and memory.
- Use one backend worker per GPU if memory is limited. Multiple workers each
  load their own copy of ESM-2.
- Keep `scikit-learn`, NumPy, and torch versions close to the versions recorded
  in each artifact's metadata.
- Treat `test_metrics` as evaluation data, not as calibrated probabilities.

## Known limitations

### Mean ESM pooling is batch-sensitive

`src/preprocessing/extract_embeddings.py` computes the mean embedding with:

```python
hidden.mean(dim=1)
```

This includes `[CLS]`, `[EOS]`, and padding positions. Training used
`BATCH_SIZE = 2`, so the mean embedding of a sequence can depend on which other
sequence shared its batch. A backend prediction on one sequence at a time may
therefore differ slightly from the offline embedding for that sequence.

The per-token/CNN path is less affected because per-token embeddings are padded
explicitly to `MAX_LEN = 50` after ESM.

A long-term fix would be masked mean pooling plus regenerating embeddings and
retraining. That is a separate experiment change; the current artifacts retain
the existing behavior.

### One-class training semantics

The current one-class experiments fit on the mixed training set and use labels
only for evaluation. The artifacts are therefore anomaly models over that
training distribution, not positive-only CPP models.

### Artifact versions

- `state_dict` files are portable across devices when loaded with
  `map_location="cpu"`, but the architecture code must match.
- `joblib` files are Python/sklearn-version-sensitive.
- Metadata records the versions used for export.

## Regenerating artifacts

Standard command:

```bash
python -m src.experiments.export_final_models
```

On the current machine, the bundled ROCm runtime may crash on the RX 6700
(`gfx1031`) target. If PyTorch segfaults on a CUDA operation, use:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run python -m src.experiments.export_final_models
```

CPU fallback:

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m src.experiments.export_final_models
```

The CPU path is valid but much slower for the four DL models.

The export script verifies each artifact by reloading it and recomputing the test
metrics before adding it to the manifest. Re-running the command overwrites
existing artifacts with the same model IDs.
