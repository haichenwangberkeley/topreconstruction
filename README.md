# Top Reconstruction Monorepo

This repository is organized as a monorepo for top quark triplet reconstruction,
from ROOT input processing through classifier training/inference and downstream
triplet-score analysis.

The canonical git repository root is this directory. The `top_reco/` folder is
kept as the canonical pipeline package subproject within this single repo.

## Requirements

- Python 3.9+
- A project virtual environment (recommended: `.venv`)
- Dependencies from `top_reco/pyproject.toml`

## Subprojects

- `top_reco/`: Canonical reconstruction and ML pipeline package (active).
- `data_processing/`: Input sample processing and diagnostics scripts (active).
- `analysis/`: Triplet-score driven top-candidate analysis scripts (active).
- `docs/`: Unified runbooks and architecture docs (scaffolded).
- `automation/`: Agent/workflow recipes and run manifest schema.

## Quick Start

Install the canonical package from monorepo root:

```bash
.venv/bin/pip install -e top_reco
```

Run a full reconstruction chain (build -> prepare -> train -> infer):

```bash
make pipeline ROOT_INPUT=$(pwd)/ttbar.root MAX_EVENTS=40000 PYTHON=.venv/bin/python
```

Run the full chain including score-based candidate selection analysis:

```bash
make pipeline_full ROOT_INPUT=$(pwd)/ttbar.root MAX_EVENTS=40000 PYTHON=.venv/bin/python \
	SELECT_STRATEGY=greedy_disjoint SELECT_MIN_SCORE=0.5 SELECT_MAX_TOP=4
```

Write a structured run manifest for the run:

```bash
make manifest MAX_EVENTS=40000 PYTHON=.venv/bin/python
```

Manifest schema:
- `automation/run_manifest.schema.json`

Migration notes for the single-repo cutover:
- `docs/MIGRATION.md`

Legacy script-focused overview has been preserved in:
- `README_legacy_scripts.md`

## Compatibility

Several root-level script names are preserved as compatibility wrappers and now
forward to canonical files under `data_processing/` and `analysis/`.
