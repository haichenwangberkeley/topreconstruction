# Automation Recipes

This directory contains machine-readable recipes for running pipeline workflows.

- `full_chain.recipe.yaml`: run build -> prepare -> train -> infer -> analysis, then write manifest.
- `infer_plus_analysis.recipe.yaml`: run inference and analysis on existing artifacts, then write manifest.
- `analysis_only.recipe.yaml`: run analysis and manifest from existing inference artifacts.
- `strategy_compare.recipe.yaml`: sweep multiple selection strategies on one inference sample and summarize efficiency comparison.
- `compare_strategies.py`: strategy sweep runner producing JSON/CSV comparison summaries.
- `run_manifest.schema.json`: schema for generated run manifests.
- `write_manifest.py`: manifest writer utility used by top-level Makefile.

## Status Semantics

- `complete`: inference report exists under the run artifact root.
- `incomplete`: completion artifacts are missing.
- `failed`: `error.log` or `failed.marker` exists under the run artifact root.
