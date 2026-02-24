# Triplet ML Pipeline (ROOT -> BDT/TabPFN)

Minimal, reproducible pipeline for:
1. Reading ROOT TTrees.
2. Building triplet-level features.
3. Preparing train/val/test datasets.
4. Training XGBoost (BDT) or TabPFN models.
5. Running inference on prepared data.

## Inputs

Primary input is one or more ROOT files with at least these branches:
- `N_genjet`, `genjet_pt`, `genjet_eta`, `genjet_phi`
- `truth_triplet_0..3`
- event id branch (default: `Number`, configurable)

Sample file for testing top quark reconstruction:
- `ttbar.root`: `https://portal.nersc.gov/project/atlas/haichenwang/artifacts/ttbar.root`

See `docs/branch_interpretation.md` and `docs/schema.md`.

## Outputs

- Stage 1 (`dataset_build`): `triplets_raw.parquet`
- Stage 2 (`dataset_prepare`): `train.parquet`, `val.parquet`, `test.parquet`
- Stage 3 (`train`): model file (`model_xgb.json` or `model_tabpfn.pkl`) + training report
- Stage 4 (`infer`): inference parquet with score column (`score` or `score_tabpfn`)

## Install

```bash
python -m pip install -e .
# Optional for TabPFN backend
python -m pip install -e .[tabpfn]
```

## Optional Makefile Workflow

`Makefile` is included to run the exact same commands with consistent variables.

```bash
# Full XGBoost pipeline
make pipeline_xgb ROOT_INPUT=data/input.root MAX_EVENTS=40000

# Full TabPFN pipeline
make pipeline_tabpfn ROOT_INPUT=data/input.root MAX_EVENTS=40000
```

Direct CLI commands remain fully supported (shown below).

## Run Preprocessing

```bash
# Stage 1: ROOT -> triplets_raw.parquet
python -m triplet_ml dataset_build \
  --inputs data/input.root \
  --output-dir artifacts/dataset_build \
  --max-events 40000

# Stage 2: split + balancing
python -m triplet_ml dataset_prepare \
  --input artifacts/dataset_build/triplets_raw.parquet \
  --output-dir artifacts/dataset_prepare
```

## Train

```bash
# BDT (XGBoost)
python -m triplet_ml train \
  --model xgb \
  --xgb-config configs/xgb_hyperparameters.json \
  --train artifacts/dataset_prepare/train.parquet \
  --val artifacts/dataset_prepare/val.parquet \
  --test artifacts/dataset_prepare/test.parquet \
  --output-dir artifacts/train

# TabPFN
python -m triplet_ml train \
  --model tabpfn \
  --train artifacts/dataset_prepare/train.parquet \
  --val artifacts/dataset_prepare/val.parquet \
  --test artifacts/dataset_prepare/test.parquet \
  --output-dir artifacts/train \
  --max-training-samples 10000
```

For XGBoost, explicit CLI hyperparameter flags (for example `--eta`, `--max-depth`) override values from `--xgb-config`.

## Inference

```bash
python -m triplet_ml infer \
  --model xgb \
  --test artifacts/dataset_prepare/test.parquet \
  --train-output-dir artifacts/train \
  --output-dir artifacts/infer
```

## Notes

- For a minimal/non-plot workflow, pass `--skip-plots` in `dataset_prepare`, `train`, and `infer`.
- Live progress is shown on interactive terminals; disable with `--no-progress`.
- TypePFN is not implemented in the current codebase; only `xgb` and `tabpfn` backends exist.
