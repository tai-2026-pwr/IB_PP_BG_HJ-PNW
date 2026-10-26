# Projekt Naukowo Wdrożeniowy

## Autorzy

- Ignacy Berent
- Pola Polańska
- Bartosz Gałczyśnki
- Hubert Janaszczyk

## Exporting final models

After preprocessing and embedding generation, export the full-train models with:

```bash
python -m src.experiments.export_final_models
```

Artifacts are written to `artifacts/models/` (gitignored). The directory contains:

- `dl/*.pt` — PyTorch `state_dict` files and their JSON metadata sidecars.
- `classical_ml/*.joblib` — fitted scikit-learn pipelines and their JSON metadata sidecars.
- `one_class/*.joblib` — fitted one-class models and their JSON metadata sidecars.
- `manifest.json` — model IDs mapped to artifact and metadata paths.

The exported models are trained on the full training split only. Their test metrics are stored in the metadata and match the `test_score` rows in `results/**/summary.csv`.
