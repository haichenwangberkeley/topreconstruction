#!/usr/bin/env python3
"""XGBoost model wrapper exposing a common fit/predict_proba interface."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


def _as_float32_matrix(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()  # pandas DataFrame path
    return np.asarray(x, dtype=np.float32)


def _as_int8_vector(y: Any) -> np.ndarray:
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()  # pandas Series path
    return np.asarray(y, dtype=np.int8)


class XGBTripletModel:
    """Thin wrapper over xgboost.Booster with sklearn-like API shape."""

    def __init__(
        self,
        feature_columns: Sequence[str],
        params: Dict[str, Any],
        num_boost_round: int = 400,
        early_stopping_rounds: Optional[int] = 30,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.params = dict(params)
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds) if early_stopping_rounds is not None else None

        self.booster = None
        self.evals_result: Dict[str, Dict[str, list[float]]] = {}
        self.best_iteration = self.num_boost_round - 1
        self.best_score = float("nan")

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[Tuple[Any, Any]] = None,
        eval_sample_weight: Optional[np.ndarray] = None,
        iteration_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("xgboost is required for XGBoost backend") from exc

        x_train = _as_float32_matrix(x)
        y_train = _as_int8_vector(y)

        dtrain = xgb.DMatrix(
            x_train,
            label=y_train,
            weight=sample_weight,
            feature_names=list(self.feature_columns),
        )

        evals = [(dtrain, "train")]
        if eval_set is not None:
            x_eval, y_eval = eval_set
            deval = xgb.DMatrix(
                _as_float32_matrix(x_eval),
                label=_as_int8_vector(y_eval),
                weight=eval_sample_weight,
                feature_names=list(self.feature_columns),
            )
            evals.append((deval, "val"))

        callbacks = []
        if iteration_callback is not None:
            callback_fn = iteration_callback

            class _IterationCallback(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch: int, evals_log) -> bool:  # type: ignore[override]
                    callback_fn(int(epoch) + 1)
                    return False

            callbacks.append(_IterationCallback())

        self.evals_result = {}
        self.booster = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            evals_result=self.evals_result,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
            callbacks=callbacks,
        )

        self.best_iteration = int(getattr(self.booster, "best_iteration", self.num_boost_round - 1))
        self.best_score = float(getattr(self.booster, "best_score", math.nan))
        return {
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "eval_history": self.evals_result,
        }

    def predict_proba(self, x: Any) -> np.ndarray:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("xgboost is required for XGBoost backend") from exc

        if self.booster is None:
            raise RuntimeError("Model is not fitted")

        matrix = _as_float32_matrix(x)
        dmatrix = xgb.DMatrix(matrix, feature_names=list(self.feature_columns))
        pos = self.booster.predict(dmatrix, iteration_range=(0, int(self.best_iteration) + 1)).astype(np.float64)
        neg = 1.0 - pos
        return np.column_stack([neg, pos])

    def save(self, path: str) -> None:
        if self.booster is None:
            raise RuntimeError("Model is not fitted")
        self.booster.save_model(path)

    @classmethod
    def load(cls, path: str, feature_columns: Sequence[str]) -> "XGBTripletModel":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("xgboost is required for XGBoost backend") from exc

        model = cls(
            feature_columns=feature_columns,
            params={},
            num_boost_round=1,
            early_stopping_rounds=None,
        )
        booster = xgb.Booster()
        booster.load_model(path)
        model.booster = booster
        model.best_iteration = int(getattr(booster, "best_iteration", 0))
        model.best_score = float(getattr(booster, "best_score", math.nan))
        return model
