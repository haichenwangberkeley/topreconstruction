# Last-10k Inference Runbook

This runbook provides command lines to:
1. Build a **last 10k event** ROOT sample from `ttbar.root`
2. Run `dataset_build` on that sample
3. Run `infer` with an existing trained XGBoost model

Run from repository root:

```bash
cd /global/homes/h/haichen/disk/top
export PYTHONPATH=top_reco/src
```

Set paths:

```bash
PYTHON=.venv/bin/python
ROOT_SRC=/global/cfs/projectdirs/atlas/www/haichenwang/artifacts/ttbar.root
LAST10K_ROOT=study/data/ttbar_last_10000.root
RUN_DIR=study/artifacts/run_last10k
MODEL_PATH=study/artifacts/run_10000/train/model_xgb.json
```

## Option A (Recommended): Write last-10k as TTree

This avoids the RNTuple/TTree auto-detection issue.

```bash
${PYTHON} - <<'PY'
from pathlib import Path
import uproot

src = "/global/cfs/projectdirs/atlas/www/haichenwang/artifacts/ttbar.root"
dst = "study/data/ttbar_last_10000.root"
Path("study/data").mkdir(parents=True, exist_ok=True)

with uproot.open(src) as f:
    tree_name = next(k.split(";")[0] for k, cls in f.classnames().items() if "TTree" in cls)
    t = f[tree_name]
    n = t.num_entries
    start = max(0, n - 10000)

    branches = [
        "N_genjet", "genjet_pt", "genjet_eta", "genjet_phi",
        "truth_triplet_0", "truth_triplet_1", "truth_triplet_2", "truth_triplet_3",
    ]
    if "Number" in t.keys():
        branches.append("Number")

    arr = t.arrays(branches, entry_start=start, entry_stop=n, library="ak")
    branch_types = {b: arr[b].type for b in branches}

    with uproot.recreate(dst) as out:
        out.mktree(tree_name, branch_types)
        out[tree_name].extend({b: arr[b] for b in branches})

print(f"wrote {dst} from entries [{start}, {n})")
PY
```

Then run build + inference:

```bash
${PYTHON} -m triplet_ml dataset_build \
  --inputs "${LAST10K_ROOT}" \
  --output-dir "${RUN_DIR}/dataset_build" \
  --max-events 10000

${PYTHON} -m triplet_ml infer \
  --model xgb \
  --model-path "${MODEL_PATH}" \
  --test "${RUN_DIR}/dataset_build/triplets_raw.parquet" \
  --output-dir "${RUN_DIR}/infer" \
  --plot-root "${RUN_DIR}/plots"

# Stage 5: greedy sequential triplet selection (cap 4/event)
${PYTHON} -m triplet_ml select_triplets \
  --inference "${RUN_DIR}/infer/inference_test_xgb.parquet" \
  --output-dir "${RUN_DIR}/select_triplets" \
  --strategy greedy_disjoint \
  --min-score 0.5 \
  --max-top-per-event 4 \
  --dummy-value -999.0

# Stage 5 (alternative): two-top pair strategy
${PYTHON} -m triplet_ml select_triplets \
  --inference "${RUN_DIR}/infer/inference_test_xgb.parquet" \
  --output-dir "${RUN_DIR}/select_triplets_pair" \
  --strategy best_pair_avg_disjoint \
  --min-score 0.5 \
  --max-top-per-event 4 \
  --dummy-value -999.0
```

## Option B: Keep current RNTuple file and force tree name

If your `LAST10K_ROOT` already exists and is `ROOT::RNTuple` with key `output`, run:

```bash
${PYTHON} -m triplet_ml dataset_build \
  --inputs "${LAST10K_ROOT}" \
  --tree-name output \
  --output-dir "${RUN_DIR}/dataset_build" \
  --max-events 10000

${PYTHON} -m triplet_ml infer \
  --model xgb \
  --model-path "${MODEL_PATH}" \
  --test "${RUN_DIR}/dataset_build/triplets_raw.parquet" \
  --output-dir "${RUN_DIR}/infer" \
  --plot-root "${RUN_DIR}/plots"

${PYTHON} -m triplet_ml select_triplets \
  --inference "${RUN_DIR}/infer/inference_test_xgb.parquet" \
  --output-dir "${RUN_DIR}/select_triplets" \
  --strategy greedy_disjoint \
  --min-score 0.5 \
  --max-top-per-event 4 \
  --dummy-value -999.0

${PYTHON} -m triplet_ml select_triplets \
  --inference "${RUN_DIR}/infer/inference_test_xgb.parquet" \
  --output-dir "${RUN_DIR}/select_triplets_pair" \
  --strategy best_pair_avg_disjoint \
  --min-score 0.5 \
  --max-top-per-event 4 \
  --dummy-value -999.0
```

Selection plots are created by default under:
- `${RUN_DIR}/select_triplets/plots`

To disable plots, add:
- `--skip-plots`

## Quick checks

```bash
${PYTHON} - <<'PY'
import json
print(json.load(open("study/artifacts/run_last10k/dataset_build/dataset_build_report.json"))["processed_events"])
print(json.load(open("study/artifacts/run_last10k/infer/inference_report_xgb.json"))["rows_output"])
print(json.load(open("study/artifacts/run_last10k/select_triplets/selection_report.json"))["selected_rows_total"])
PY
```
