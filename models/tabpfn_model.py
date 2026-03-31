#!/usr/bin/env python3
"""TabPFN model wrapper exposing a common fit/predict_proba interface."""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


def _as_float32_matrix(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()  # pandas DataFrame path
    return np.asarray(x, dtype=np.float32)


def _as_int8_vector(y: Any) -> np.ndarray:
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()  # pandas Series path
    return np.asarray(y, dtype=np.int8)


class TabPFNTripletModel:
    """Wrapper around official TabPFNClassifier API."""

    def __init__(
        self,
        feature_columns: Sequence[str],
        random_state: int = 42,
        device: str = "auto",
        n_ensemble_configurations: Optional[int] = None,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.random_state = int(random_state)
        self.device = str(device)
        self.n_ensemble_configurations = (
            int(n_ensemble_configurations) if n_ensemble_configurations is not None else None
        )
        self.classifier = None

    def _build_classifier(self):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise RuntimeError("tabpfn is required for TabPFN backend") from exc

        signature = inspect.signature(TabPFNClassifier)
        kwargs: Dict[str, Any] = {}

        if "random_state" in signature.parameters:
            kwargs["random_state"] = self.random_state
        elif "seed" in signature.parameters:
            kwargs["seed"] = self.random_state

        if "device" in signature.parameters:
            kwargs["device"] = self.device

        if self.n_ensemble_configurations is not None and "n_ensemble_configurations" in signature.parameters:
            kwargs["n_ensemble_configurations"] = self.n_ensemble_configurations

        return TabPFNClassifier(**kwargs)

    def fit(self, x: Any, y: Any, **_: Any) -> Dict[str, Any]:
        x_train = _as_float32_matrix(x)
        y_train = _as_int8_vector(y)
        self.classifier = self._build_classifier()
        self.classifier.fit(x_train, y_train)
        return {}

    def predict_proba(self, x: Any) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Model is not fitted")

        matrix = _as_float32_matrix(x)
        proba = np.asarray(self.classifier.predict_proba(matrix), dtype=np.float64)
        if proba.ndim == 1:
            pos = np.clip(proba, 0.0, 1.0)
        elif proba.shape[1] >= 2:
            pos = np.clip(proba[:, 1], 0.0, 1.0)
        else:
            pos = np.clip(proba[:, 0], 0.0, 1.0)
        neg = 1.0 - pos
        return np.column_stack([neg, pos])

    def save(self, path: str) -> None:
        if self.classifier is None:
            raise RuntimeError("Model is not fitted")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str, feature_columns: Sequence[str]) -> "TabPFNTripletModel":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, cls):
            payload.feature_columns = list(feature_columns)
            return payload
        raise RuntimeError(f"Unexpected pickle payload type for TabPFN model: {type(payload)!r}")
