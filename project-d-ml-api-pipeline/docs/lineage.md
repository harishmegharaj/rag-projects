# Data and model lineage (Project D)

## What is versioned

| Asset | Location | Notes |
| --- | --- | --- |
| **Training table** | `data/raw/iris.csv` | Small public iris extract; replace with your dataset and keep a checksum in the registry. |
| **Model artifact** | `models/artifacts/classifier_*.joblib` | sklearn `Pipeline` (scaler + logistic regression). Not committed by default (see `.gitignore`). |
| **Registry** | `models/registry.json` | Points to the **active** artifact and keeps a history list with metrics and hashes. |

## Registry fields

Each training run appends a record with:

- `data_sha256` — SHA-256 of the CSV used for training.
- `metrics.accuracy_holdout` — holdout accuracy (stratified split).
- `library_versions` — sklearn / numpy / pandas at train time.
- `git_sha` — commit when available (`GITHUB_SHA` in CI).
- `trained_at_utc` — ISO timestamp.

## Optional: DVC

If you adopt [DVC](https://dvc.org/), track `data/raw/` and `models/artifacts/` as DVC outputs and store them in S3/GCS/Azure Blob. Keep `registry.json` in Git or generate it in CI after `dvc pull`. This repo uses a **registry file** so you can stay DVC-free and still document lineage.

## Reproducing a train

```bash
python scripts/train.py
```

Set `DATA_RAW_PATH` to point at another CSV with columns `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `target` if you extend the schema (update `src/train_pipeline.py` accordingly).
