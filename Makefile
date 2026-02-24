PYTHON ?= python
ROOT_INPUT ?= data/input.root
MAX_EVENTS ?= 40000
SEED ?= 42
OMP_NUM_THREADS ?= 8

ARTIFACT_ROOT ?= artifacts/run_$(MAX_EVENTS)
BUILD_DIR := $(ARTIFACT_ROOT)/dataset_build
PREP_DIR := $(ARTIFACT_ROOT)/dataset_prepare
TRAIN_DIR := $(ARTIFACT_ROOT)/train
INFER_DIR := $(ARTIFACT_ROOT)/infer

XGB_CONFIG ?= configs/xgb_hyperparameters.json
PLOT_FLAGS ?= --skip-plots
PROGRESS_FLAGS ?=

.PHONY: help install dirs dataset_build dataset_prepare train_xgb train_tabpfn infer_xgb infer_tabpfn pipeline_xgb pipeline_tabpfn

help:
	@echo "Targets:"
	@echo "  make pipeline_xgb      # run full ROOT->prepare->train(XGB)->infer pipeline"
	@echo "  make pipeline_tabpfn   # run full ROOT->prepare->train(TabPFN)->infer pipeline"
	@echo "  make dataset_build     # stage 1"
	@echo "  make dataset_prepare   # stage 2"
	@echo "  make train_xgb         # stage 3 (XGBoost)"
	@echo "  make train_tabpfn      # stage 3 (TabPFN)"
	@echo "  make infer_xgb         # stage 4 (XGBoost model)"
	@echo "  make infer_tabpfn      # stage 4 (TabPFN model)"
	@echo ""
	@echo "Configurable vars:"
	@echo "  ROOT_INPUT=$(ROOT_INPUT)"
	@echo "  MAX_EVENTS=$(MAX_EVENTS)"
	@echo "  SEED=$(SEED)"
	@echo "  ARTIFACT_ROOT=$(ARTIFACT_ROOT)"
	@echo "  XGB_CONFIG=$(XGB_CONFIG)"
	@echo "  PLOT_FLAGS='$(PLOT_FLAGS)'"
	@echo "  PROGRESS_FLAGS='$(PROGRESS_FLAGS)'"

install:
	$(PYTHON) -m pip install -e .

dirs:
	mkdir -p $(BUILD_DIR) $(PREP_DIR) $(TRAIN_DIR) $(INFER_DIR)

dataset_build: dirs
	$(PYTHON) -m triplet_ml dataset_build \
	  --inputs $(ROOT_INPUT) \
	  --output-dir $(BUILD_DIR) \
	  --max-events $(MAX_EVENTS) \
	  $(PROGRESS_FLAGS) \
	  --seed $(SEED)

dataset_prepare: dataset_build
	$(PYTHON) -m triplet_ml dataset_prepare \
	  --input $(BUILD_DIR)/triplets_raw.parquet \
	  --output-dir $(PREP_DIR) \
	  $(PLOT_FLAGS) \
	  $(PROGRESS_FLAGS) \
	  --seed $(SEED)

train_xgb: dataset_prepare
	$(PYTHON) -m triplet_ml train \
	  --model xgb \
	  --xgb-config $(XGB_CONFIG) \
	  --train $(PREP_DIR)/train.parquet \
	  --val $(PREP_DIR)/val.parquet \
	  --test $(PREP_DIR)/test.parquet \
	  --output-dir $(TRAIN_DIR) \
	  $(PLOT_FLAGS) \
	  $(PROGRESS_FLAGS) \
	  --seed $(SEED)

train_tabpfn: dataset_prepare
	$(PYTHON) -m triplet_ml train \
	  --model tabpfn \
	  --train $(PREP_DIR)/train.parquet \
	  --val $(PREP_DIR)/val.parquet \
	  --test $(PREP_DIR)/test.parquet \
	  --output-dir $(TRAIN_DIR) \
	  --max-training-samples 10000 \
	  $(PLOT_FLAGS) \
	  $(PROGRESS_FLAGS) \
	  --seed $(SEED)

infer_xgb: train_xgb
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) $(PYTHON) -m triplet_ml infer \
	  --model xgb \
	  --test $(PREP_DIR)/test.parquet \
	  --train-output-dir $(TRAIN_DIR) \
	  --output-dir $(INFER_DIR) \
	  $(PLOT_FLAGS) \
	  $(PROGRESS_FLAGS) \
	  --seed $(SEED)

infer_tabpfn: train_tabpfn
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) $(PYTHON) -m triplet_ml infer \
	  --model tabpfn \
	  --test $(PREP_DIR)/test.parquet \
	  --train-output-dir $(TRAIN_DIR) \
	  --output-dir $(INFER_DIR) \
	  $(PLOT_FLAGS) \
	  $(PROGRESS_FLAGS) \
	  --seed $(SEED)

pipeline_xgb: infer_xgb

pipeline_tabpfn: infer_tabpfn
