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
- Stage 3 (`train`):
  - XGBoost: `model_xgb.json`, `training_report_xgb.json`, `training_statistics_xgb.md`
  - TabPFN: `model_tabpfn.pkl`, `training_report_tabpfn.json`, `training_statistics_tabpfn.md`
- Stage 4 (`infer`):
  - XGBoost: `inference_test_xgb.parquet`, `inference_report_xgb.json`
  - TabPFN: `inference_test_tabpfn.parquet`, `inference_report_tabpfn.json`
  - score columns: `score_xgb` / `score_tabpfn`

## Install

```bash
python -m pip install -e .
# Optional for TabPFN backend
python -m pip install -e .[tabpfn]
```

## Command Style

All command examples below follow one style:
- Run from repository root.
- Use `python -m triplet_ml ...` for CLI stages.
- Put one argument per line using trailing `\` for readability.

Recent development summary:
- `docs/changes_since_last_push.md` (changes since last pushed commit)

## Optional Makefile Workflow

`Makefile` is included to run the exact same commands with consistent variables.

```bash
# Full XGBoost pipeline
make pipeline_xgb \
  ROOT_INPUT=data/input.root \
  MAX_EVENTS=40000

# Full TabPFN pipeline
make pipeline_tabpfn \
  ROOT_INPUT=data/input.root \
  MAX_EVENTS=40000
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

## Select Reconstructed Top Candidates (Stage 5)

Select variable-count top candidates per event from scored triplets.
Default cap is 4 selected triplets per event.

Outputs:
- `selected_triplets.parquet`
  - `event_id`, `selected_rank`, `i`, `j`, `k`, `score`
  - selected triplet four-vector: `triplet_pt`, `triplet_eta`, `triplet_phi`, `triplet_mass`
- `event_selection.parquet`
  - `n_top_selected`
  - fixed top-candidate slots `top1_*` ... `top4_*` (`pt`, `eta`, `phi`, `mass`)
  - dummy placeholder for missing slots (default `-999.0`)
  - `m_top1_top2` for the two leading selected candidates
- `plots/`
  - `n_top_selected_<strategy>.png`
  - `m_top1_top2_<strategy>.png`
  - `top_pt_by_rank_<strategy>.png`, `top_eta_by_rank_<strategy>.png`, `top_phi_by_rank_<strategy>.png`, `top_mass_by_rank_<strategy>.png`
  - `selection_plot_metrics_<strategy>.json`

```bash
python -m triplet_ml select_triplets \
  --inference artifacts/infer/inference_test_xgb.parquet \
  --output-dir artifacts/select_triplets \
  --strategy greedy_disjoint \
  --min-score 0.5 \
  --max-top-per-event 4
```

Plot notes:
- Selection plots are generated automatically by default.
- To disable them, add `--skip-plots`.
- To control binning, use `--plot-bins` (default: `20`).
- To change plot location, use `--plot-root` (default: `<output-dir>/plots`).

Available strategies:
- `greedy_disjoint`: sequential highest-score selection with non-overlapping jets
- `top1`: highest-score triplet only
- `topk`: top-k by score (`--top-k`)
- `threshold`: all score-passing triplets up to cap
- `best_pair_avg_disjoint`: for events with inferred jet count `>= 6`, selects exactly two non-overlapping triplets whose pair has the highest average score; for `< 6` jets no candidates are selected (dummy placeholders remain)

Two-top strategy example:

```bash
python -m triplet_ml select_triplets \
  --inference artifacts/infer/inference_test_xgb.parquet \
  --output-dir artifacts/select_triplets_pair \
  --strategy best_pair_avg_disjoint \
  --min-score 0.5 \
  --max-top-per-event 4 \
  --dummy-value -999.0
```

For this strategy, `top1_*` and `top2_*` are filled only when a valid pair exists; `top3_*` and `top4_*` remain dummy by construction.

## Plot From Persistent Histogram Cache

This flow follows the plotting invariants: histogram production is separate from rendering.

```bash
# Step 1: build persistent histogram cache from inference parquet (single pass)
python -m triplet_ml build_m123_hist_cache \
  --inference artifacts/infer/inference_test_xgb.parquet \
  --output-hist artifacts/plots/inference/xgb/m123_score_gt_0p5_hist_cache.npz \
  --score-cut 0.5 \
  --pt-bins 20 \
  --pt-min 0 \
  --pt-max 1000 \
  --eta-bins 10 \
  --eta-min -5 \
  --eta-max 5 \
  --observable-bins 80 \
  --observable-min 0 \
  --observable-max 500

# Step 2: render true-vs-fake m123 from cached histogram only
python -m triplet_ml plot_m123_hist_cache \
  --histogram-cache artifacts/plots/inference/xgb/m123_score_gt_0p5_hist_cache.npz \
  --output-png artifacts/plots/inference/xgb/m123_score_gt_0p5_truth_vs_fake_from_cache.png \
  --title "m123 after score > 0.5 (true vs fake)"
```

## Plot 4-Way Cut Comparison For All Features

Generate invariant-style overlays for all numeric columns in inference parquet:
- true + pass (`score > cut`)
- true + fail (`score <= cut`)
- fake + pass
- fake + fail

Each plot includes a ratio panel in `[0.5, 1.5]`.

```bash
python -m triplet_ml plot_cut_comparison \
  --inference artifacts/infer/inference_test_xgb.parquet \
  --output-root artifacts/plots \
  --score-cut 0.5 \
  --bins 20 \
  --nominal true_pass
```

To run on selected columns only, add:

```bash
python -m triplet_ml plot_cut_comparison \
  --inference artifacts/infer/inference_test_xgb.parquet \
  --output-root artifacts/plots \
  --score-cut 0.5 \
  --bins 20 \
  --columns m123 triplet_pt triplet_eta
```

## Notes

- For a minimal/non-plot workflow, pass `--skip-plots` in `dataset_prepare`, `train`, `infer`, and `select_triplets`.
- Live progress is shown on interactive terminals; disable with `--no-progress`.
- TabPFN is implemented; only `xgb` and `tabpfn` backends exist.
