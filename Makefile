PYTHON ?= .venv/bin/python
ROOT_INPUT ?= $(CURDIR)/ttbar.root
MAX_EVENTS ?= 40000
SEED ?= 42
OMP_THREADS ?= 8
SELECT_STRATEGY ?= greedy_disjoint
SELECT_MIN_SCORE ?= 0.5
SELECT_MAX_TOP ?= 4
COMPARE_STRATEGIES ?= greedy_disjoint top1 topk threshold best_pair_avg_disjoint

# Normalize interpreter path so delegated make in top_reco can execute it.
# Keep bare commands (e.g. "python") unchanged; rewrite relative paths.
ABS_PYTHON := $(if $(findstring /,$(PYTHON)),$(if $(filter /%,$(PYTHON)),$(PYTHON),$(CURDIR)/$(PYTHON)),$(PYTHON))

ARTIFACT_ROOT ?= top_reco/artifacts/run_$(MAX_EVENTS)
PLOT_FLAGS ?= --skip-plots

.PHONY: help check_python pipeline build prepare train infer analyze analysis_only compare_strategies pipeline_full manifest

help:
	@echo "Monorepo targets (delegates to top_reco):"
	@echo "  make pipeline      # full build->prepare->train->infer"
	@echo "  make build         # stage 1: dataset_build"
	@echo "  make prepare       # stage 2: dataset_prepare"
	@echo "  make train         # stage 3: train_xgb"
	@echo "  make infer         # stage 4: infer_xgb"
	@echo "  make analyze       # stage 5: select_triplets analysis"
	@echo "  make analysis_only # run analysis + manifest on existing inference artifacts"
	@echo "  make compare_strategies # compare strategy efficiencies on existing inference artifacts"
	@echo "  make pipeline_full # full build->prepare->train->infer->analyze"
	@echo "  make manifest      # write run manifest JSON"
	@echo ""
	@echo "Configurable vars:"
	@echo "  ROOT_INPUT=$(ROOT_INPUT)"
	@echo "  MAX_EVENTS=$(MAX_EVENTS)"
	@echo "  SEED=$(SEED)"
	@echo "  SELECT_STRATEGY=$(SELECT_STRATEGY)"
	@echo "  SELECT_MIN_SCORE=$(SELECT_MIN_SCORE)"
	@echo "  SELECT_MAX_TOP=$(SELECT_MAX_TOP)"
	@echo "  COMPARE_STRATEGIES=$(COMPARE_STRATEGIES)"
	@echo "  PYTHON=$(PYTHON)"

check_python:
	@$(ABS_PYTHON) -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" || \
	  (echo "Error: Python 3.9+ required. Interpreter: $(ABS_PYTHON)" && exit 1)

RECO_MAKE = $(MAKE) -C top_reco \
	PYTHON=$(ABS_PYTHON) \
	ROOT_INPUT=$(ROOT_INPUT) \
	MAX_EVENTS=$(MAX_EVENTS) \
	SEED=$(SEED) \
	OMP_NUM_THREADS=$(OMP_THREADS) \
	PLOT_FLAGS="$(PLOT_FLAGS)"

build: check_python
	$(RECO_MAKE) dataset_build

prepare: check_python
	$(RECO_MAKE) dataset_prepare

train: check_python
	$(RECO_MAKE) train_xgb

infer: check_python
	$(RECO_MAKE) infer_xgb

analyze: check_python
	@test -f "$(ARTIFACT_ROOT)/infer/inference_test_xgb.parquet" || (echo "Error: missing inference artifact $(ARTIFACT_ROOT)/infer/inference_test_xgb.parquet" && exit 1)
	@$(ABS_PYTHON) -m triplet_ml select_triplets \
	  --inference "$(ARTIFACT_ROOT)/infer/inference_test_xgb.parquet" \
	  --output-dir "$(ARTIFACT_ROOT)/select_triplets" \
	  --strategy "$(SELECT_STRATEGY)" \
	  --min-score $(SELECT_MIN_SCORE) \
	  --max-top-per-event $(SELECT_MAX_TOP)

pipeline: check_python
	$(RECO_MAKE) pipeline_xgb

pipeline_full: pipeline analyze

analysis_only: analyze manifest

compare_strategies: check_python
	@test -f "$(ARTIFACT_ROOT)/infer/inference_test_xgb.parquet" || (echo "Error: missing inference artifact $(ARTIFACT_ROOT)/infer/inference_test_xgb.parquet" && exit 1)
	@$(ABS_PYTHON) automation/compare_strategies.py \
	  --artifact-root "$(ARTIFACT_ROOT)" \
	  --strategies $(COMPARE_STRATEGIES) \
	  --min-score $(SELECT_MIN_SCORE) \
	  --max-top-per-event $(SELECT_MAX_TOP) \
	  --top-k $(SELECT_MAX_TOP)

manifest: check_python
	@command -v "$(ABS_PYTHON)" >/dev/null 2>&1 || (echo "Error: Python interpreter not found: $(ABS_PYTHON)" && exit 1)
	@$(ABS_PYTHON) automation/write_manifest.py \
	  --root-input "$(ROOT_INPUT)" \
	  --max-events $(MAX_EVENTS) \
	  --seed $(SEED) \
	  --artifact-root "$(ARTIFACT_ROOT)" \
	  --pipeline top_reco \
	  --select-strategy "$(SELECT_STRATEGY)" \
	  --select-min-score $(SELECT_MIN_SCORE) \
	  --select-max-top $(SELECT_MAX_TOP)