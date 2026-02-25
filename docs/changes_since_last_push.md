# Changes Since Last Push

Base pushed commit: `878c2aa` (`origin/main`)  
Date documented: 2026-02-25

## Summary

This repository now includes a full Stage 5 triplet-selection flow, new plotting commands that follow histogram invariants, and updated run/documentation content for the last-10k analysis workflow.

## Code Changes

### 1. New Stage 5 selector command

Added `python -m triplet_ml select_triplets` with strategy-based top-candidate selection:

- `greedy_disjoint`
- `top1`
- `topk`
- `threshold`

Implemented in:

- `src/triplet_ml/select_triplets.py` (new)
- `src/triplet_ml/cli.py` (subcommand registration)

Key behavior:

- Input: scored inference parquet.
- Output cap: configurable selected candidates per event (`--max-top-per-event`, capped at 4).
- Score threshold: `--min-score`.
- Strategy parameters: `--top-k`.

### 2. Stage 5 output schema expansion

`selected_triplets.parquet` now stores:

- `event_id`, `selected_rank`, `i`, `j`, `k`, `score`, `strategy`
- selected triplet four-vector:
  - `triplet_pt`
  - `triplet_eta`
  - `triplet_phi`
  - `triplet_mass`

`event_selection.parquet` now stores:

- `event_id`
- `n_triplets_total`
- `n_top_selected`
- fixed top-candidate slots `top1_*` to `top4_*` (`pt`, `eta`, `phi`, `mass`)
- placeholder values for missing slots via `--dummy-value`
- `m_top1_top2` (invariant mass of leading two selected candidates)

### 3. Stage 5 built-in plotting outputs

`select_triplets` can now produce selection-level distributions in the same run.

Default outputs under `<output-dir>/plots/`:

- `n_top_selected.png`
- `m_top1_top2.png`
- `top_pt_by_rank.png`
- `top_eta_by_rank.png`
- `top_phi_by_rank.png`
- `top_mass_by_rank.png`
- `selection_plot_metrics.json`

New options:

- `--plot-root`
- `--plot-bins` (default `20`)
- `--skip-plots`

## Plotting Module Enhancements

Updated `src/triplet_ml/plotting.py` with:

### 1. 4-way cut-comparison plotting

New `plot_cut_comparison` command to compare:

- true/pass
- true/fail
- fake/pass
- fake/fail

For each numeric feature with:

- common binning
- statistical error bars
- ratio pane and nominal uncertainty band
- fixed ratio axis range `[0.5, 1.5]`

### 2. Cached histogram flow for `m123`

Commands:

- `build_m123_hist_cache`
- `plot_m123_hist_cache`

Purpose:

- single-pass histogram cache creation from parquet
- render true-vs-fake `m123` from cache without rescanning events

### 3. Inference comparison plotting refinements

`plot_inference` now supports model-comparison mode (`xgb` vs `tabpfn`) with ratio-aware overlays and saved metrics.

## Documentation Updates

### 1. README updates

`README.md` now includes:

- uniform command-style guidance
- Stage 5 usage and outputs
- Stage 5 plotting options
- histogram-cache flow commands
- 4-way cut-comparison commands

### 2. Schema updates

`docs/schema.md` now documents Stage 5 inputs, outputs, schema contents, and plotting outputs.

### 3. New plotting invariant spec

`docs/plottinginvariants.md` defines mandatory 1D histogram invariants:

- line-only histograms
- uncertainty handling
- ratio-panel requirements
- error propagation and edge cases

### 4. Last-10k runbook

`docs/run_last10k_inference.md` includes command lines to:

- build a last-10k ROOT slice
- run dataset build
- run inference
- run Stage 5 selection
- perform quick checks

## Files Added/Modified (Since Last Push)

- Added:
  - `src/triplet_ml/select_triplets.py`
  - `docs/plottinginvariants.md`
  - `docs/run_last10k_inference.md`
  - `docs/changes_since_last_push.md`
- Modified:
  - `src/triplet_ml/cli.py`
  - `src/triplet_ml/plotting.py`
  - `README.md`
  - `docs/schema.md`

