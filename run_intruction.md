# Triplet ML Pipeline Run Instructions (40k Events)

This document records the exact commands used to run the full 4-stage pipeline on 40,000 events.

## 0. Prerequisites

Run from project root:

```bash
cd /global/homes/h/haichen/disk/top
```

Install dependencies (if needed):

```bash
python -m pip install --user pyarrow xgboost uproot awkward
```

## 1. Prepare output directories

```bash
mkdir -p artifacts/run_40000/dataset_build \
         artifacts/run_40000/dataset_prepare \
         artifacts/run_40000/train \
         artifacts/run_40000/infer
```

## 2. Stage 1: dataset_build (ROOT -> triplets_raw.parquet)

```bash
python main.py dataset_build \
  --inputs ttbar.root \
  --output-dir artifacts/run_40000/dataset_build \
  --max-events 40000 \
  --seed 42
```

Expected outputs:
- `artifacts/run_40000/dataset_build/triplets_raw.parquet`
- `artifacts/run_40000/dataset_build/dataset_build_report.json`
- `artifacts/run_40000/dataset_build/config_snapshot.json`

## 3. Stage 2: dataset_prepare (split + balancing)

```bash
python main.py dataset_prepare \
  --input artifacts/run_40000/dataset_build/triplets_raw.parquet \
  --output-dir artifacts/run_40000/dataset_prepare \
  --seed 42
```

Expected outputs:
- `artifacts/run_40000/dataset_prepare/train.parquet`
- `artifacts/run_40000/dataset_prepare/val.parquet`
- `artifacts/run_40000/dataset_prepare/test.parquet`
- `artifacts/run_40000/dataset_prepare/dataset_prepare_report.json`
- `artifacts/run_40000/dataset_prepare/config_snapshot.json`

## 4. Stage 3: train (train+val -> model)

```bash
python main.py train \
  --train artifacts/run_40000/dataset_prepare/train.parquet \
  --val artifacts/run_40000/dataset_prepare/val.parquet \
  --output-dir artifacts/run_40000/train \
  --use-sample-weights \
  --seed 42
```

Expected outputs:
- `artifacts/run_40000/train/model_xgb.json`
- `artifacts/run_40000/train/training_report.json`
- `artifacts/run_40000/train/config_snapshot.json`

## 5. Stage 4: infer (model+test -> scored triplets)

Recommended stable run (limited threads):

```bash
OMP_NUM_THREADS=8 python main.py infer \
  --model artifacts/run_40000/train/model_xgb.json \
  --test artifacts/run_40000/dataset_prepare/test.parquet \
  --output-dir artifacts/run_40000/infer \
  --batch-size 25000 \
  --seed 42
```

Expected outputs:
- `artifacts/run_40000/infer/inference_test.parquet`
- `artifacts/run_40000/infer/inference_report.json`
- `artifacts/run_40000/infer/config_snapshot.json`

## 6. Quick verification

```bash
python -c "import json; print(json.load(open('artifacts/run_40000/dataset_build/dataset_build_report.json')))"
python -c "import json; print(json.load(open('artifacts/run_40000/dataset_prepare/dataset_prepare_report.json')))"
python -c "import json; d=json.load(open('artifacts/run_40000/train/training_report.json')); print({'best_iteration': d['best_iteration'], 'val_auc': d['metrics']['val_auc'], 'val_logloss': d['metrics']['val_logloss']})"
python -c "import json; print(json.load(open('artifacts/run_40000/infer/inference_report.json')))"
```

