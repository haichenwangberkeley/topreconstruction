# Triplet ML Pipeline Run Instructions

This document provides reproducible commands for a standard 40k-event run.

## 0. Prerequisites

From repository root:

```bash
python -m pip install -e .
# optional for TabPFN runs
python -m pip install -e .[tabpfn]
```

Set an input ROOT file:

```bash
export ROOT_INPUT=data/input.root
```

The default XGBoost hyperparameter config is:

```bash
export XGB_CONFIG=configs/xgb_hyperparameters.json
```

## 1. Recommended (Makefile) Workflow

Run the full pipeline with one command:

```bash
make pipeline_xgb ROOT_INPUT=$ROOT_INPUT MAX_EVENTS=40000
```

TabPFN variant:

```bash
make pipeline_tabpfn ROOT_INPUT=$ROOT_INPUT MAX_EVENTS=40000
```

Default outputs are written to:
- `artifacts/run_40000/dataset_build/`
- `artifacts/run_40000/dataset_prepare/`
- `artifacts/run_40000/train/`
- `artifacts/run_40000/infer/`

## 2. Equivalent Direct CLI Commands

```bash
mkdir -p artifacts/run_40000/dataset_build \
         artifacts/run_40000/dataset_prepare \
         artifacts/run_40000/train \
         artifacts/run_40000/infer
```

Stage 1:

```bash
python -m triplet_ml dataset_build \
  --inputs "$ROOT_INPUT" \
  --output-dir artifacts/run_40000/dataset_build \
  --max-events 40000 \
  --seed 42
```

Stage 2:

```bash
python -m triplet_ml dataset_prepare \
  --input artifacts/run_40000/dataset_build/triplets_raw.parquet \
  --output-dir artifacts/run_40000/dataset_prepare \
  --skip-plots \
  --seed 42
```

Stage 3 (XGBoost):

```bash
python -m triplet_ml train \
  --model xgb \
  --xgb-config "$XGB_CONFIG" \
  --train artifacts/run_40000/dataset_prepare/train.parquet \
  --val artifacts/run_40000/dataset_prepare/val.parquet \
  --test artifacts/run_40000/dataset_prepare/test.parquet \
  --output-dir artifacts/run_40000/train \
  --skip-plots \
  --seed 42
```

CLI overrides still work when explicitly set, for example:

```bash
python -m triplet_ml train \
  --model xgb \
  --xgb-config "$XGB_CONFIG" \
  --eta 0.03 \
  --max-depth 8 \
  --use-sample-weights \
  --train artifacts/run_40000/dataset_prepare/train.parquet \
  --val artifacts/run_40000/dataset_prepare/val.parquet \
  --test artifacts/run_40000/dataset_prepare/test.parquet \
  --output-dir artifacts/run_40000/train \
  --skip-plots \
  --seed 42
```

Stage 4 (XGBoost inference):

```bash
OMP_NUM_THREADS=8 python -m triplet_ml infer \
  --model xgb \
  --test artifacts/run_40000/dataset_prepare/test.parquet \
  --train-output-dir artifacts/run_40000/train \
  --output-dir artifacts/run_40000/infer \
  --skip-plots \
  --seed 42
```

## 3. Progress Output

- By default, stages show live progress in interactive terminals.
- To disable progress (batch/log mode), add `--no-progress` to `dataset_build`, `dataset_prepare`, `train`, and `infer`.

## 4. Quick Verification

```bash
python -c "import json; print(json.load(open('artifacts/run_40000/dataset_build/dataset_build_report.json')))"
python -c "import json; print(json.load(open('artifacts/run_40000/dataset_prepare/dataset_prepare_report.json')))"
python -c "import json; d=json.load(open('artifacts/run_40000/train/training_report_xgb.json')); print({'best_iteration': d.get('best_iteration'), 'val_auc': d['metrics']['val_auc'], 'val_logloss': d['metrics']['val_logloss']})"
python -c "import json; print(json.load(open('artifacts/run_40000/infer/inference_report_xgb.json')))"
```

## 5. Scope Note

TabPFN is implemented in this repository. Supported backends are `xgb` and `tabpfn`.
