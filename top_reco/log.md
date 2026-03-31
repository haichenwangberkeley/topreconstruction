2026-02-25 (Wed) 20:51 America/Los_Angeles

### 3.1 Objective (conceptual, non-technical)
Improve traceability and reproducibility of the reconstruction workflow so it is easy to tell what was run, what outputs were produced, and how to compare results across model backends and selection strategies. The goal was to make outputs self-identifying, reduce ambiguity in reports and plots, and document the workflow clearly for future analysis. A secondary goal was to run and validate the last-10k and TabPFN workflows enough to establish a stable baseline and known limitations.

### 3.2 Work summary
- Implemented and integrated Stage 5 triplet selection into the CLI, including event-level and selected-triplet outputs.
- Added a second Stage 5 strategy for two-top reconstruction (`best_pair_avg_disjoint`) with non-overlapping triplet pair selection.
- Expanded Stage 5 outputs to include triplet four-vectors, fixed top candidate slots, dummy placeholders, and `m_top1_top2`.
- Added Stage 5 plotting outputs and made plot titles/filenames include selection strategy for clarity.
- Added/updated documentation for runbooks, schema, plotting invariants, and a repository change summary.
- Standardized model-specific artifact naming (XGBoost vs TabPFN) for train/infer outputs and report files.
- Executed last-10k XGBoost inference and both selection strategies, then validated counts from report files.
- Ran TabPFN training reruns; completed a CPU run with reduced sample cap and produced a valid training report.
- Attempted full TabPFN inference on CPU; runtime was very long and run was stopped before completion.

### 3.3 Changes to code and configuration
#### A) Files created
- `src/triplet_ml/select_triplets.py` — New Stage 5 command for strategy-based triplet selection and selection plotting.
- `docs/plottinginvariants.md` — Repository-local plotting invariant specification used to enforce consistent histogram behavior.
- `docs/run_last10k_inference.md` — Last-10k runbook with dataset build, inference, and selection commands.
- `docs/changes_since_last_push.md` — High-level summary of changes since the previous pushed commit.
- `log.md` — Ongoing session log file (this file).

#### B) Files modified
- `src/triplet_ml/cli.py` — Registered `select_triplets` subcommand in main CLI.
- `src/triplet_ml/plotting.py` — Added/extended plotting workflows (cut comparison, histogram-cache plotting, comparison behavior updates).
- `src/triplet_ml/train.py` — Wrote model-specific training stats filenames and model-specific config snapshot copy.
- `src/triplet_ml/infer.py` — Wrote model-specific inference report filename and model-specific config snapshot copy.
- `src/triplet_ml/models/__init__.py` — Updated defaults to model-explicit names (`training_report_xgb.json`, `inference_test_xgb.parquet`, `score_xgb`).
- `README.md` — Updated commands/output naming and Stage 5 behavior/docs.
- `docs/schema.md` — Updated schema text for Stage 5 and model-specific train/infer artifact names.
- `docs/run_instructions.md` — Updated verification commands to model-specific report names.
- `docs/run_last10k_inference.md` — Updated inference input names and report checks.

#### C) Files renamed/moved/removed
- `docs/plottinginvariants.md` (symlink) → `docs/plottinginvariants.md` (regular tracked markdown file) — Removed external absolute symlink dependency and made docs portable in git.

### 3.4 Commands and runs executed
Working directory for all runs:
- `/global/homes/h/haichen/disk/top/clean_repo`

Key pipeline and validation commands:
```bash
# Last-10k inference (XGBoost) and selection
python -m triplet_ml dataset_build \
  --inputs "${LAST10K_ROOT}" \
  --output-dir "${RUN_DIR}/dataset_build" \
  --max-events 10000

python -m triplet_ml infer \
  --model xgb \
  --model-path "${MODEL_PATH}" \
  --test "${RUN_DIR}/dataset_build/triplets_raw.parquet" \
  --output-dir "${RUN_DIR}/infer" \
  --plot-root "${RUN_DIR}/plots"

python -m triplet_ml select_triplets \
  --inference "${RUN_DIR}/infer/inference_test.parquet" \
  --output-dir "${RUN_DIR}/select_triplets" \
  --strategy greedy_disjoint \
  --min-score 0.5 \
  --max-top-per-event 4 \
  --dummy-value -999.0

python -m triplet_ml select_triplets \
  --inference "${RUN_DIR}/infer/inference_test.parquet" \
  --output-dir "${RUN_DIR}/select_triplets_pair" \
  --strategy best_pair_avg_disjoint \
  --min-score 0.5 \
  --max-top-per-event 4 \
  --dummy-value -999.0

# TabPFN environment preparation and runs
conda run -p /global/homes/h/haichen/disk/llm_for_analysis/conda_env \
  python -m pip install pyarrow

# Failed (CUDA runtime error during plotting path)
PYTHONPATH=src conda run -p /global/homes/h/haichen/disk/llm_for_analysis/conda_env \
  python -m triplet_ml train \
  --model tabpfn \
  --train study/artifacts/run_10000/dataset_prepare/train.parquet \
  --val study/artifacts/run_10000/dataset_prepare/val.parquet \
  --test study/artifacts/run_10000/dataset_prepare/test.parquet \
  --output-dir study/artifacts/run_10000/train \
  --max-training-samples 10000 \
  --plot-root study/artifacts/run_10000/plots

# Failed (CPU >1000 restriction)
PYTHONPATH=src conda run -p /global/homes/h/haichen/disk/llm_for_analysis/conda_env \
  python -m triplet_ml train \
  --model tabpfn \
  --tabpfn-device cpu \
  --train study/artifacts/run_10000/dataset_prepare/train.parquet \
  --val study/artifacts/run_10000/dataset_prepare/val.parquet \
  --test study/artifacts/run_10000/dataset_prepare/test.parquet \
  --output-dir study/artifacts/run_10000/train \
  --max-training-samples 10000 \
  --plot-root study/artifacts/run_10000/plots

# Successful reduced CPU training
PYTHONPATH=src conda run -p /global/homes/h/haichen/disk/llm_for_analysis/conda_env \
  python -m triplet_ml train \
  --model tabpfn \
  --tabpfn-device cpu \
  --train study/artifacts/run_10000/dataset_prepare/train.parquet \
  --val study/artifacts/run_10000/dataset_prepare/val.parquet \
  --test study/artifacts/run_10000/dataset_prepare/test.parquet \
  --output-dir study/artifacts/run_10000/train \
  --max-training-samples 500 \
  --plot-root study/artifacts/run_10000/plots \
  --skip-plots \
  --no-progress

# Compile checks for modified scripts
python3.11 -m compileall -q src/triplet_ml/models/__init__.py src/triplet_ml/train.py src/triplet_ml/infer.py src/triplet_ml/select_triplets.py
```

### 3.5 Tests and validation
- Verified Stage 5 selection counts from report JSONs:
  - `study/artifacts/run_last10k/select_triplets/selection_report.json`
  - `study/artifacts/run_last10k/select_triplets_pair/selection_report.json`
  - Outcome: pass (reports present and internally consistent).
- Verified last-10k inference row accounting from `study/artifacts/run_last10k/infer/inference_report.json`.
  - Outcome: pass (`rows_input == rows_output == 104892`).
- Verified TabPFN training output from `study/artifacts/run_10000/train/training_report_tabpfn.json`.
  - Outcome: partial (training completed for reduced sample cap `500`; not full 10k CPU run).
- Performed syntax checks on modified Python files using `compileall`.
  - Outcome: pass.
- Known limitation:
  - Full TabPFN inference on CPU over `run_10000` test set was too slow for this session and was interrupted; resulting parquet may be partial.

### 3.6 Output artifacts
- Artifact type: parquet file (triplet dataset)
  - Path: `study/artifacts/run_last10k/dataset_build/triplets_raw.parquet`
  - Produced by: `dataset_build` command in §3.4.
  - Notes:
    - Contains candidate triplets for last-10k event slice.

- Artifact type: parquet file (inference output, XGBoost)
  - Path: `study/artifacts/run_last10k/infer/inference_test.parquet`
  - Produced by: XGBoost `infer` command in §3.4.
  - Notes:
    - `rows_input = rows_output = 104892`.
    - Score column is `score` in this older artifact.

- Artifact type: parquet files (selection outputs, greedy strategy)
  - Paths:
    - `study/artifacts/run_last10k/select_triplets/selected_triplets.parquet`
    - `study/artifacts/run_last10k/select_triplets/event_selection.parquet`
  - Produced by: `select_triplets --strategy greedy_disjoint` in §3.4.
  - Notes:
    - `selected_rows_total = 7414`, `events_with_selection = 6361`.

- Artifact type: parquet files (selection outputs, pair strategy)
  - Paths:
    - `study/artifacts/run_last10k/select_triplets_pair/selected_triplets.parquet`
    - `study/artifacts/run_last10k/select_triplets_pair/event_selection.parquet`
  - Produced by: `select_triplets --strategy best_pair_avg_disjoint` in §3.4.
  - Notes:
    - `selected_rows_total = 2832`, `events_with_selection = 1416`.
    - `events_with_lt6_jets_inferred = 6440`, `events_with_ge6_jets_inferred = 2277`.

- Artifact type: plot metrics and PNG plots (selection)
  - Paths:
    - `study/artifacts/run_last10k/select_triplets/plots/selection_plot_metrics.json`
    - `study/artifacts/run_last10k/select_triplets_pair/plots/selection_plot_metrics_best_pair_avg_disjoint.json`
    - `study/artifacts/run_last10k/select_triplets_pair/plots/*_best_pair_avg_disjoint.png`
  - Produced by: Stage 5 plotting in `select_triplets`.
  - Notes:
    - Plot set includes `n_top_selected`, `m_top1_top2`, and top kinematics by rank.
    - Newer files include strategy in title and filename.

- Artifact type: model and report (TabPFN training)
  - Paths:
    - `study/artifacts/run_10000/train/model_tabpfn.pkl`
    - `study/artifacts/run_10000/train/training_report_tabpfn.json`
    - `study/artifacts/run_10000/train/training_statistics_tabpfn.md`
  - Produced by: reduced-cap TabPFN train command in §3.4.
  - Notes:
    - Completed at `max_training_samples = 500`.
    - Validation AUC recorded in report: `0.7654` (approx).

Storage context note:
- `study/` is a symlink in repo root to an external directory (`/global/homes/h/haichen/disk/top/study`), so these artifacts are outside git-tracked repository contents.

### 3.7 Issues, surprises, and decisions
- Error: TabPFN GPU training path hit `torch.AcceleratorError: CUDA error: invalid configuration argument`.
  - Decision: switch to CPU for reliability.
- Error: TabPFN CPU run with `>1000` samples failed due built-in limit (`Running on CPU with more than 1000 samples is not allowed by default`).
  - Decision: run with reduced training sample cap for successful completion.
- Surprise: Full TabPFN inference on CPU is significantly slower than XGBoost for this test split.
  - Decision: stop long-running CPU inference in this session and keep completed training artifacts.
- Decision: make output names model-specific (`xgb`/`tabpfn`) and include strategy in selection plot names to reduce ambiguity.

### 3.8 Next steps
- [ ] Complete a full TabPFN inference run (prefer GPU if stable; otherwise CPU with managed runtime window) and produce `inference_report_tabpfn.json`.
- [ ] Regenerate `run_last10k` inference/selection using the new model-specific output naming (`*_xgb`, `*_tabpfn`) for consistency.
- [ ] Remove legacy ambiguous files (or archive them) after confirming new named outputs are present.
- [ ] Update Stage 5 report/metrics references to use only strategy-tagged plot metric files.
- [ ] Add a short helper script/Make target to run and validate last-10k pipeline end-to-end with one command.
- [ ] Add a small validation script that checks required artifact existence and schema fields after each stage.
