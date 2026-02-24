# Future Development Tasks (Agent-Facing)

This file is a backlog of implementation tasks intended for coding agents.
Each item should be treated as a concrete engineering requirement.

## Item 1: Externalize BDT Hyperparameters to a Config File

### Goal
The BDT (XGBoost) hyperparameters must be read from a configuration file instead of being hardcoded in Python defaults.

### Why
Hardcoded hyperparameters make experiments harder to reproduce, compare, and tune.

### Requirements
- Add a config file (YAML or JSON) under `clean_repo/docs/` or another clearly documented config location.
- Move current XGBoost training hyperparameters from code defaults into that config file.
- Ensure training can run by loading the config file values at runtime.
- Keep CLI overrides possible (CLI values should override config values when explicitly provided).
- Persist the resolved hyperparameters in `config_snapshot.json` for reproducibility.
- Update `clean_repo/docs/run_instructions.md` with the new usage pattern.

### Scope
- This item applies to the BDT/XGBoost training path only.
- Do not change model logic or feature definitions in this item.

### Acceptance Criteria
- No BDT hyperparameter defaults remain hardcoded as the primary source in the training code path.
- A single config file can fully define the BDT hyperparameters for a run.
- Running training with that config reproduces the same parameter values in the output snapshot/report.
- Documentation clearly shows how to run training with config-based hyperparameters.

## Item 2: Add Progress Bars for Pipeline Stages

### Goal
Show progress bars in the terminal for data processing/preparation, training, and inference stages.

### Why
Long-running jobs need transparent runtime feedback so users and agents can track status and estimate remaining time.

### Requirements
- Add on-screen progress indicators for:
  - dataset build (`dataset_build`)
  - dataset preparation (`dataset_prepare`)
  - training (`train`)
  - inference (`infer`)
- Progress output must work in normal terminal sessions on NERSC/CERN-style environments.
- Progress should report processed work units and completion fraction (or equivalent).
- Keep a quiet/non-interactive mode available for logs or batch scripts.
- Do not remove existing summary/report outputs.

### Scope
- This item is about user-visible progress reporting only.
- Do not change model behavior, splitting logic, or output schema in this item.

### Acceptance Criteria
- Each listed stage shows a live progress bar (or equivalent incremental progress) during execution.
- Output remains readable and does not break existing JSON/markdown artifact writing.
- Users can disable progress output when needed (for batch/non-interactive workflows).
- `clean_repo/docs/run_instructions.md` documents progress behavior and any relevant flags.
