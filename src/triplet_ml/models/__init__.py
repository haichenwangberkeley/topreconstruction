#!/usr/bin/env python3
"""Model backend registry for triplet ML pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from .tabpfn_model import TabPFNTripletModel
from .xgb_model import XGBTripletModel

MODEL_BACKENDS = ("xgb", "tabpfn")


def normalize_model_backend(model_backend: str) -> str:
    model = str(model_backend).strip().lower()
    if model not in MODEL_BACKENDS:
        raise ValueError(f"Unsupported model backend '{model_backend}'. Valid: {', '.join(MODEL_BACKENDS)}")
    return model


def infer_model_backend_from_path(path: str) -> str:
    name = Path(path).name.lower()
    if "tabpfn" in name or name.endswith(".pkl"):
        return "tabpfn"
    return "xgb"


def default_model_filename(model_backend: str) -> str:
    model = normalize_model_backend(model_backend)
    if model == "tabpfn":
        return "model_tabpfn.pkl"
    return "model_xgb.json"


def default_training_report_filename(model_backend: str) -> str:
    model = normalize_model_backend(model_backend)
    if model == "tabpfn":
        return "training_report_tabpfn.json"
    return "training_report.json"


def default_inference_filename(model_backend: str) -> str:
    model = normalize_model_backend(model_backend)
    if model == "tabpfn":
        return "inference_test_tabpfn.parquet"
    return "inference_test.parquet"


def inference_score_column(model_backend: str) -> str:
    model = normalize_model_backend(model_backend)
    if model == "tabpfn":
        return "score_tabpfn"
    return "score"


def create_model(model_backend: str, feature_columns: Sequence[str], **kwargs: Any):
    model = normalize_model_backend(model_backend)
    if model == "xgb":
        return XGBTripletModel(
            feature_columns=feature_columns,
            params=kwargs.get("params", {}),
            num_boost_round=int(kwargs.get("num_boost_round", 400)),
            early_stopping_rounds=kwargs.get("early_stopping_rounds", 30),
        )
    return TabPFNTripletModel(
        feature_columns=feature_columns,
        random_state=int(kwargs.get("random_state", 42)),
        device=str(kwargs.get("device", "auto")),
        n_ensemble_configurations=kwargs.get("n_ensemble_configurations"),
    )


def load_model(model_backend: str, path: str, feature_columns: Sequence[str]):
    model = normalize_model_backend(model_backend)
    if model == "xgb":
        return XGBTripletModel.load(path=path, feature_columns=feature_columns)
    return TabPFNTripletModel.load(path=path, feature_columns=feature_columns)


def resolve_backend_and_path(
    model_arg: str,
    model_path: str | None,
    default_train_dir: str = "artifacts/train",
    test_dataset: str | None = None,
) -> tuple[str, str]:
    lowered = str(model_arg).strip().lower()
    if lowered in MODEL_BACKENDS:
        backend = lowered
        if model_path is not None:
            return backend, model_path

        default_name = default_model_filename(backend)
        candidates = [
            Path(default_train_dir) / default_name,
        ]
        if test_dataset is not None:
            test_path = Path(test_dataset)
            if len(test_path.parents) >= 2:
                candidates.append(test_path.parents[1] / "train" / default_name)

        for candidate in candidates:
            if candidate.exists():
                return backend, str(candidate)

        return backend, str(candidates[0])

    legacy_path = str(model_arg)
    if model_path is not None:
        raise ValueError("Provide either --model as a backend/path OR --model-path, not both path forms together.")
    return infer_model_backend_from_path(legacy_path), legacy_path

