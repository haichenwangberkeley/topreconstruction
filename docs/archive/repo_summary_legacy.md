# REPO SUMMARY

## TL;DR
- This workspace contains a **Top triplet ML pipeline** that reads `ttbar.root`, builds triplet features, writes Parquet, trains (`xgb` and `tabpfn`), runs inference, and makes plots (`main.py:15-35`, `dataset_build.py:67-228`, `train.py:383-431`, `infer.py:156-185`, `plotting.py:790-831`).
- Your filename memory check: `ttbar.root` is used everywhere; `TPBar.root` does not appear in the repo (`rg` search; examples in `run_intruction.md:31-33`, `README.md:10`, `validate_triplet_interpretation.py:62`).
- There are **two parallel code layouts**: script-style at repo root and packaged `clean_repo/` with the same stages (`main.py:8-23`, `clean_repo/src/triplet_ml/cli.py:8-23`).
- Fastest reproducible interface is now `python -m triplet_ml` / `triplet-ml` after editable install (`clean_repo/pyproject.toml:23-24`, `clean_repo/README.md:28-32`).
- Core data path is `ROOT -> triplets_raw.parquet -> train/val/test.parquet -> model -> inference parquet` (`schema.md:27-33`, `schema.md:49-61`, `schema.md:69-77`, `schema.md:85-93`).
- BDT backend is **XGBoost** (`models/xgb_model.py:24-33`, `models/xgb_model.py:54-57`); TabPFN backend is implemented (`models/tabpfn_model.py:26-35`, `models/tabpfn_model.py:46-49`).
- Splitting is deterministic event-level hashing; train/val get background capping, test is unmodified (`triplet_io.py:75-89`, `dataset_prepare.py:37-40`, `dataset_prepare.py:207-215`, `dataset_prepare.py:324-327`).
- Environment mismatch: system `python` is 2.7 (fails on type hints), while `.venv` is Python 3.11 and works (`main.py` fails under `/usr/bin/python`, CLI works under `.venv/bin/python`).
- No test suite is present and `pytest` is not installed in this environment.
- Smoke validation done: Stage 1+2 succeeded on tiny subset; tiny run can yield empty validation split and fail training (`artifacts/smoke_fast2/dataset_prepare/dataset_prepare_report.json:2-6`, `train.py:171-173`); inference succeeded when reusing an existing trained model (`artifacts/smoke_fast2/infer_from_pretrained/inference_report.json:3-11`).

## 1) Quickstart map (fastest way to run)

### Primary entry points

| Entry point | Command | Inputs | Outputs | Output location |
|---|---|---|---|---|
| Root CLI (script layout) | `.venv/bin/python main.py <subcommand> ...` | ROOT/parquet/model depending on stage | Stage artifacts + reports + config snapshots | Per `--output-dir` (`main.py:15-35`) |
| Packaged CLI module | `.venv/bin/python -m triplet_ml <subcommand> ...` | Same as above | Same as above | Per `--output-dir` (`clean_repo/src/triplet_ml/__main__.py:1-4`) |
| Installed console script | `.venv/bin/triplet-ml <subcommand> ...` | Same as above | Same as above | Per `--output-dir` (`clean_repo/pyproject.toml:23-24`) |
| Makefile orchestration | `cd clean_repo && make pipeline_xgb ROOT_INPUT=... MAX_EVENTS=...` | ROOT file path | All 4 stages chained | `artifacts/run_<MAX_EVENTS>/...` (`clean_repo/Makefile:7-12`, `clean_repo/Makefile:95-97`) |
| Optional diagnostics/legacy scripts | `print_ttree_branches.py`, `plot_jets.py`, `cutflow_and_store.py`, `triplet_reco.py`, `validate_triplet_interpretation.py` | ROOT or NPY | branch dumps / PNG+HTML / NPY+NPZ / validation plots | local dirs (`README.md:1-210`) |

No notebooks were found (`find . -name '*.ipynb'` -> empty).

### Fastest end-to-end commands (packaged path)

```bash
cd /global/homes/h/haichen/disk/top
.venv/bin/python -m pip install -e clean_repo

# Stage 1
.venv/bin/python -m triplet_ml dataset_build \
  --inputs ttbar.root \
  --output-dir artifacts/run_40000/dataset_build \
  --max-events 40000 \
  --seed 42

# Stage 2
.venv/bin/python -m triplet_ml dataset_prepare \
  --input artifacts/run_40000/dataset_build/triplets_raw.parquet \
  --output-dir artifacts/run_40000/dataset_prepare \
  --skip-plots \
  --seed 42

# Stage 3 (XGBoost)
.venv/bin/python -m triplet_ml train \
  --model xgb \
  --train artifacts/run_40000/dataset_prepare/train.parquet \
  --val artifacts/run_40000/dataset_prepare/val.parquet \
  --test artifacts/run_40000/dataset_prepare/test.parquet \
  --output-dir artifacts/run_40000/train \
  --use-sample-weights \
  --skip-plots \
  --seed 42

# Stage 4 (inference)
OMP_NUM_THREADS=8 .venv/bin/python -m triplet_ml infer \
  --model xgb \
  --test artifacts/run_40000/dataset_prepare/test.parquet \
  --train-output-dir artifacts/run_40000/train \
  --output-dir artifacts/run_40000/infer \
  --skip-plots \
  --seed 42
```

Equivalent root-script commands are documented in `run_intruction.md:31-99`.

### What is missing / gotchas for full run
- Use `.venv/bin/python` (or any Python >=3.9). System `/usr/bin/python` is 2.7 and fails on current code (`main.py` type hints).
- Dependencies must be installed (`clean_repo/pyproject.toml:10-21` or `run_intruction.md:16`).
- Very small `--max-events` can produce no validation events and make training fail (`artifacts/smoke_fast2/dataset_prepare/dataset_prepare_report.json:2-6`, `train.py:171-173`).
- Stage 3 can be compute-heavy on large parquet inputs.

## 2) Repository architecture overview

### Top-down data flow

```text
ttbar.root (ROOT TTree)
  -> Stage 1: dataset_build.py (triplet enumeration + feature calc)
       -> triplets_raw.parquet
  -> Stage 2: dataset_prepare.py (event split + bg cap)
       -> train.parquet, val.parquet, test.parquet
  -> Stage 3: train.py (xgb or tabpfn)
       -> model_xgb.json or model_tabpfn.pkl
       -> training_report*.json, config_snapshot.json
  -> Stage 4: infer.py (score test triplets)
       -> inference_test.parquet / inference_test_tabpfn.parquet
       -> inference_report.json, config_snapshot.json
  -> plotting.py / plot subcommands
       -> feature/training/inference PNGs + metrics JSON
```

### Key files/directories by stage
- CLI orchestration: `main.py:15-35`, packaged `clean_repo/src/triplet_ml/cli.py:15-35`.
- Stage 1 build: `dataset_build.py:67-228`.
- Stage 2 split/prepare: `dataset_prepare.py:159-332`.
- Stage 3 train: `train.py:151-431`.
- Stage 4 infer: `infer.py:37-185`.
- Feature math/sanity: `features.py:11-184`.
- Shared I/O/schema/splitting: `triplet_io.py:15-185`.
- Model backends: `models/xgb_model.py`, `models/tabpfn_model.py`, registry `models/__init__.py:12-112`.
- Plots: `plotting.py:213-831`.
- Packaged mirror: `clean_repo/src/triplet_ml/*` (mostly same logic, relative imports).

### Mismatch callout
- Root `README.md` is mostly plotting/cutflow focused (`README.md:31-210`) and does **not** document the modular 4-stage parquet ML CLI end-to-end.
- End-to-end pipeline docs are in `run_intruction.md:1-101` and `clean_repo/docs/run_instructions.md:1-107`.

## 3) Data schema and artifacts

### ROOT branches actually used by the pipeline
- Stage 1 explicitly reads:
  - `N_genjet`, `genjet_pt`, `genjet_eta`, `genjet_phi`, `truth_triplet_0..3` (`dataset_build.py:20-33`).
  - Optional event id branch `Number` by default if present (`dataset_build.py:81-90`, `dataset_build.py:220`).
- Tree is auto-discovered as first TTree unless `--tree-name` is set (`triplet_io.py:31-46`, `dataset_build.py:76`).
- If event id branch missing, fallback is entry-index counter (`dataset_build.py:113-117`, `dataset_build.py:193-194`).

### Parquet schema (current code)
- Canonical schema (no score):
  - `event_id`, `i`, `j`, `k`, 6 feature columns, 7 observables (`m123`, `mij_*`, `triplet_*`), `is_truth` (`triplet_io.py:97-116`).
- Inference schema appends backend-specific score column (`triplet_io.py:117-119`, `models/__init__.py:50-54`).
- Dtypes are explicit Arrow types (`triplet_io.py:98-116`).

### Observed schema drift in existing artifacts
- `artifacts/run_10000_fresh/*` uses 18-column schema (features + observables + label), matching current `triplet_io.py`.
- `artifacts/run_40000/*` raw/split parquet files are older 11-column schema (no `m123/mij_*/triplet_*`) from prior pipeline revision (observed via pyarrow schema inspection).

### Intermediate artifacts found
- Stage reports/config snapshots:
  - `dataset_build_report.json`, `dataset_prepare_report.json`, `training_report*.json`, `inference_report.json`, `config_snapshot.json` (written in stage code: `dataset_build.py:187-213`, `dataset_prepare.py:280-317`, `train.py:277-380`, `infer.py:125-153`).
- Models:
  - `model_xgb.json`, `model_tabpfn.pkl` (`models/__init__.py:29-33`).
- Plot metrics:
  - `feature_plot_summary.json`, `training_plot_metrics.json`, `inference_plot_metrics.json`, `inference_comparison_metrics.json` (`plotting.py:305-313`, `plotting.py:491`, `plotting.py:592`, `plotting.py:726`).
- Existing artifact directories:
  - `artifacts/run_40000`, `artifacts/run_10000_fresh`, `artifacts/smoke_visual`, plus smoke outputs from this discovery pass.
- Notable artifact anomalies:
  - `artifacts/run_10000_fresh/infer/inference_test_xgb_rerun.parquet` is not readable parquet (footer/magic failure from pyarrow).
  - `artifacts/run_40000/infer/inference_test_tabpfn.parquet` exists but has 0 rows (pyarrow row count).

## 4) Training & inference details

### TabPFN
- Config location: CLI args in `train` subparser (no YAML config file for this pipeline) (`train.py:383-431`).
- Data prep: loads parquet columns `FEATURE_COLUMNS + is_truth` (`train.py:54-67`).
- Split source: uses Stage 2 event-hash split from `dataset_prepare.py` + `triplet_io.assign_split` (`dataset_prepare.py:207-215`, `triplet_io.py:82-89`).
- TabPFN hyperparameters:
  - `--tabpfn-device`, `--tabpfn-n-ensemble-configurations`, `--tabpfn-balance-classes`, optional `--max-training-samples` (`train.py:415-425`, `train.py:90-128`).
- Model save format:
  - pickle (`model_tabpfn.pkl`) (`models/__init__.py:29-33`, `models/tabpfn_model.py:88-95`).
- Outputs:
  - `training_report_tabpfn.json`, `training_statistics.md`, `config_snapshot.json` (`models/__init__.py:36-40`, `train.py:319-380`).

### BDT (XGBoost)
- Library: `xgboost` wrapper around `xgboost.train` (`models/xgb_model.py:24-33`, `models/xgb_model.py:80-88`).
- Feature list: six engineered features from `features.FEATURE_COLUMNS` (`features.py:11-18`, `train.py:54`, `train.py:281`).
- Hyperparameters (CLI):
  - `num_boost_round=400`, `early_stopping_rounds=30`, `eta=0.05`, `max_depth=6`, `min_child_weight=1.0`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `tree_method=hist` (`train.py:404-413`).
- Class imbalance weighting (optional): `--use-sample-weights` computes positive class weight from train/val counts (`train.py:199-203`, `train.py:413`).
- Model save format: `model_xgb.json` (`models/__init__.py:29-33`, `models/xgb_model.py:113-117`).

### Inference (both backends)
- Invocation: `infer` subcommand with backend or explicit model path (`infer.py:156-173`).
- Backend/path resolution and score column naming:
  - score columns: `score` (xgb), `score_tabpfn` (tabpfn) (`models/__init__.py:50-54`, `infer.py:57`).
- Reads full test parquet in batches, predicts probabilities, writes scored parquet (`infer.py:74-99`).
- Consistency check: input/output row counts must match (`infer.py:102-103`).
- Metrics computed:
  - Training stage computes AUC/logloss from internal helpers (`train.py:26-45`, `train.py:309-313`).
  - Plotting stage computes ROC/AUC and threshold curves (`plotting.py:50-60`, `plotting.py:63-99`, `plotting.py:519-593`).

## 5) Plotting outputs

### Pipeline-generated plots (`plotting.py`)
- Feature validation (train/test):
  - per-feature shapes, observables, `mij`, train-only correlation matrices (`plotting.py:213-313`).
  - Output dir: `plots/features/<split>/` (`plotting.py:214`).
- Training diagnostics:
  - `roc_train_val.png`, `score_train_signal_background.png`, `overtraining_score_comparison.png`, `efficiency_curves.png`, `training_plot_metrics.json` (`plotting.py:400`, `plotting.py:410`, `plotting.py:463`, `plotting.py:478`, `plotting.py:491`).
  - Output dir: `plots/training/<backend>/` (`plotting.py:366`).
- Inference diagnostics:
  - `roc_test.png`, `score_distribution_test.png`, `tpr_vs_threshold.png`, `fpr_vs_threshold.png`, optional `score_vs_m123.png`, plus `inference_plot_metrics.json` (`plotting.py:530`, `plotting.py:540`, `plotting.py:557`, `plotting.py:567`, `plotting.py:578`, `plotting.py:592`).
  - Output dir: `plots/inference/<backend>/` (`plotting.py:503`).
- Inference comparison (xgb vs tabpfn):
  - `roc_comparison.png`, `score_distribution_signal_background_comparison.png`, `score_distribution_model_overlay.png`, `inference_comparison_metrics.json` (`plotting.py:638`, `plotting.py:672`, `plotting.py:707`, `plotting.py:726`).
  - Output dir: `plots/inference/comparison/` (`plotting.py:622`).

### Additional plotting scripts outside the modular pipeline
- `plot_jets.py`: reconstructed/genjet kinematic and multiplicity PNGs + `index.html` (`plot_jets.py:5-10`, `plot_jets.py:313-320`).
- `triplet_reco.py`: `triplet_pt/eta/phi/mass.png` + `index.html` from `selected_jets.npy` (`triplet_reco.py:85-131`, `triplet_reco.py:167-174`).
- `validate_triplet_interpretation.py`: overlay/zoom plots, default PNG with optional PDF (`validate_triplet_interpretation.py:531-535`, `validate_triplet_interpretation.py:670-680`).

## 6) Configuration & dependencies

### Configuration style
- Main pipeline configuration is via **argparse flags**, not Hydra/Click/Tyro/YAML for run-time stage control (`main.py:15-35`, `dataset_build.py:216-228`, `dataset_prepare.py:320-332`, `train.py:383-431`, `infer.py:156-185`).
- Determinism controls:
  - seed passed to each stage and written to `config_snapshot.json` (`triplet_io.py:54-72`).
  - deterministic split hash on `event_id` (`triplet_io.py:75-89`).
- Makefile orchestration exists in packaged repo (`clean_repo/Makefile:41-97`).

### Dependencies / environment
- Packaged requirements (`clean_repo/pyproject.toml:10-21`):
  - Python `>=3.9`, `numpy`, `uproot`, `awkward`, `pyarrow`, `xgboost`, `matplotlib`, optional `tabpfn` extra.
- Root-run instructions mention direct pip install for core deps (`run_intruction.md:16`).
- Practical environment state observed:
  - System `python` is 2.7 (incompatible with pipeline syntax).
  - `.venv/bin/python` is 3.11 and contains required libs including `tabpfn`.

### Multiple environments
- Script-style root workflow: `.venv/bin/python main.py ...`.
- Package workflow: install `clean_repo` editable, then use `python -m triplet_ml` / `triplet-ml`.

## 7) What I should do next (5 concrete tasks)

1. **Pick one canonical interface and document it in one place.**
- Problem prevented: user/agent confusion from dual entrypoints and split docs.
- Why it matters: wrong command path (`main.py` vs `triplet-ml`) causes reproducibility drift.
- Files: `README.md`, `run_intruction.md`, `clean_repo/README.md`, `clean_repo/docs/run_instructions.md`.
- Effort: **low**.

2. **Add a smoke test command that guarantees non-empty train/val splits.**
- Problem prevented: tiny runs fail with `Training and validation datasets must both be non-empty`.
- Why it matters: agents need a fast deterministic sanity check before long jobs.
- Files: `dataset_prepare.py` (small-data split safeguards), `train.py` (clearer pre-check message), `clean_repo/Makefile` (add `smoke` target).
- Evidence: `artifacts/smoke_fast2/dataset_prepare/dataset_prepare_report.json:2-6`, `train.py:171-173`.
- Effort: **low/medium**.

3. **Add schema-compat checks between dataset files and model/inference stages.**
- Problem prevented: silent breakage when old 11-column parquet artifacts are mixed with new 18-column schema expectations.
- Why it matters: reproducibility across historical runs (`run_40000` vs `run_10000_fresh`) and agent-safe reuse.
- Files: `triplet_io.py`, `train.py`, `infer.py`, `dataset_prepare.py`.
- Effort: **medium**.

4. **Add artifact integrity checks (parquet readability + row counts) as a utility.**
- Problem prevented: corrupted or zero-row inference files propagating downstream.
- Why it matters: avoids training/evaluation on broken outputs.
- Files: new `artifact_check.py` (or extend `triplet_io.py`), integrate in `infer.py`/`train.py` post-write checks.
- Evidence: unreadable `artifacts/run_10000_fresh/infer/inference_test_xgb_rerun.parquet`; zero-row `artifacts/run_40000/infer/inference_test_tabpfn.parquet`.
- Effort: **medium**.

5. **Add explicit training runtime controls for large datasets in docs + CLI defaults for smoke mode.**
- Problem prevented: very long CPU-bound training during routine checks.
- Why it matters: student and agent workflows need predictable turnaround on NERSC/CERN nodes.
- Files: `train.py` (document/optionally expose thread control), `run_intruction.md`, `clean_repo/docs/run_instructions.md`, `clean_repo/Makefile` (smoke-friendly rounds/samples target).
- Effort: **medium**.

## Explicit hypothesis check (your initial recollection)
- `ttbar.root` hypothesis: **confirmed** (`ttbar.root` symlink exists; refs in `run_intruction.md:31-33`).
- `TPBar.root` hypothesis: **not found** (no references from `rg`).
- ROOT -> Parquet conversion hypothesis: **confirmed** (`dataset_build.py:2`, `dataset_build.py:74`, `triplet_io.py:122-158`).
- TabPFN and BDT training hypothesis: **confirmed** (`MODEL_BACKENDS = ("xgb", "tabpfn")` in `models/__init__.py:12`, and backend wrappers).
- Inference + plotting hypothesis: **confirmed** (`infer.py:2`, `plotting.py:495-831`).
