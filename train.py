#!/usr/bin/env python3
"""Stage 3: train classifier (XGBoost or TabPFN) from train/val parquet files."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import features
import plotting
import triplet_io as pio
from models import (
    MODEL_BACKENDS,
    create_model,
    default_model_filename,
    default_training_report_filename,
    normalize_model_backend,
)


def _binary_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(y_score, eps, 1.0 - eps)
    y = y_true.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(np.int64)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_pos = float(np.sum(ranks[y == 1]))
    auc = (sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc)


def _load_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for loading parquet training data") from exc

    cols = list(features.FEATURE_COLUMNS) + ["is_truth"]
    table = pq.read_table(path, columns=cols)
    data = table.to_pydict()

    x = np.column_stack([np.asarray(data[col], dtype=np.float32) for col in features.FEATURE_COLUMNS])
    y = np.asarray(data["is_truth"], dtype=np.int8)

    if x.shape[0] != y.shape[0]:
        raise RuntimeError(f"Feature/label row mismatch in {path}")
    if x.shape[1] != len(features.FEATURE_COLUMNS):
        raise RuntimeError(f"Unexpected feature width in {path}: {x.shape[1]}")

    features.assert_feature_batch_sane({name: x[:, i] for i, name in enumerate(features.FEATURE_COLUMNS)})
    return x, y


def _subsample_training_rows(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_training_samples: Optional[int],
    seed: int,
    balance_classes: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    original_rows = int(x_train.shape[0])
    sampling: Dict[str, Any] = {
        "enabled": bool(max_training_samples is not None),
        "seed": int(seed),
        "max_training_samples": None if max_training_samples is None else int(max_training_samples),
        "strategy": "balanced_50_50" if balance_classes else "random",
        "subsampled": False,
        "original_train_rows": original_rows,
        "used_train_rows": original_rows,
    }

    rng = np.random.default_rng(seed)

    if balance_classes:
        if max_training_samples is None:
            raise ValueError("--tabpfn-balance-classes requires --max-training-samples")
        if max_training_samples <= 0:
            raise ValueError("--max-training-samples must be > 0")

        signal_idx = np.where(y_train == 1)[0]
        background_idx = np.where(y_train == 0)[0]
        if signal_idx.size == 0 or background_idx.size == 0:
            raise RuntimeError("Balanced sampling requires both classes in the training data")

        n_signal_target = int(max_training_samples) // 2
        n_background_target = int(max_training_samples) - n_signal_target

        signal_replace = bool(signal_idx.size < n_signal_target)
        background_replace = bool(background_idx.size < n_background_target)

        sampled_signal = rng.choice(signal_idx, size=n_signal_target, replace=signal_replace)
        sampled_background = rng.choice(background_idx, size=n_background_target, replace=background_replace)
        indices = np.concatenate([sampled_signal, sampled_background])
        rng.shuffle(indices)

        sampled_x = x_train[indices]
        sampled_y = y_train[indices]
        sampling.update(
            {
                "enabled": True,
                "subsampled": True,
                "used_train_rows": int(sampled_x.shape[0]),
                "target_signal_rows": int(n_signal_target),
                "target_background_rows": int(n_background_target),
                "sampled_signal_rows": int(np.sum(sampled_y == 1)),
                "sampled_background_rows": int(np.sum(sampled_y == 0)),
                "signal_sampling_with_replacement": signal_replace,
                "background_sampling_with_replacement": background_replace,
                "unique_source_rows_used": int(np.unique(indices).size),
            }
        )
        return sampled_x, sampled_y, sampling

    if max_training_samples is None:
        return x_train, y_train, sampling
    if max_training_samples <= 0:
        raise ValueError("--max-training-samples must be > 0")
    if original_rows <= max_training_samples:
        return x_train, y_train, sampling

    indices = rng.choice(original_rows, size=int(max_training_samples), replace=False)
    indices.sort()

    sampled_x = x_train[indices]
    sampled_y = y_train[indices]
    sampling.update(
        {
            "subsampled": True,
            "used_train_rows": int(sampled_x.shape[0]),
        }
    )
    return sampled_x, sampled_y, sampling


def run(args: argparse.Namespace) -> None:
    model_backend = normalize_model_backend(args.model)
    use_balanced_tabpfn_sampling = bool(model_backend == "tabpfn" and args.tabpfn_balance_classes)
    if args.tabpfn_balance_classes and model_backend != "tabpfn":
        raise ValueError("--tabpfn-balance-classes is only valid when --model tabpfn")

    output_dir = pio.ensure_dir(args.output_dir)
    model_path = Path(args.model_out) if args.model_out else output_dir / default_model_filename(model_backend)
    report_path = (
        Path(args.report_out) if args.report_out else output_dir / default_training_report_filename(model_backend)
    )

    x_train_full, y_train_full = _load_xy(args.train)
    x_val, y_val = _load_xy(args.val)
    resolved_test: Optional[str] = args.test
    if resolved_test is None:
        candidate = Path(args.val).with_name("test.parquet")
        if candidate.exists():
            resolved_test = str(candidate)

    if x_train_full.shape[0] == 0 or x_val.shape[0] == 0:
        raise RuntimeError("Training and validation datasets must both be non-empty")

    x_train, y_train, sampling = _subsample_training_rows(
        x_train=x_train_full,
        y_train=y_train_full,
        max_training_samples=args.max_training_samples,
        seed=args.seed,
        balance_classes=use_balanced_tabpfn_sampling,
    )

    pos_train = int(np.sum(y_train == 1))
    neg_train = int(np.sum(y_train == 0))
    if pos_train == 0 or neg_train == 0:
        raise RuntimeError(
            "Training set must contain both classes after sampling. "
            "Increase --max-training-samples or disable subsampling."
        )

    pos_val = int(np.sum(y_val == 1))
    neg_val = int(np.sum(y_val == 0))
    if pos_val == 0 or neg_val == 0:
        raise RuntimeError("Validation set must contain both classes")

    weight_train = None
    weight_val = None
    pos_weight = 1.0

    if model_backend == "xgb" and args.use_sample_weights:
        pos_weight = float(neg_train / max(pos_train, 1))
        weight_train = np.where(y_train == 1, pos_weight, 1.0).astype(np.float32)
        weight_val = np.where(y_val == 1, float(neg_val / pos_val), 1.0).astype(np.float32)

    model_kwargs: Dict[str, Any] = {}
    if model_backend == "xgb":
        model_kwargs.update(
            {
                "params": {
                    "objective": "binary:logistic",
                    "eval_metric": ["logloss", "auc"],
                    "eta": args.eta,
                    "max_depth": args.max_depth,
                    "min_child_weight": args.min_child_weight,
                    "subsample": args.subsample,
                    "colsample_bytree": args.colsample_bytree,
                    "lambda": args.reg_lambda,
                    "seed": args.seed,
                    "tree_method": args.tree_method,
                },
                "num_boost_round": args.num_boost_round,
                "early_stopping_rounds": args.early_stopping_rounds,
            }
        )
    else:
        model_kwargs.update(
            {
                "random_state": args.seed,
                "device": args.tabpfn_device,
                "n_ensemble_configurations": args.tabpfn_n_ensemble_configurations,
            }
        )

    model = create_model(
        model_backend=model_backend,
        feature_columns=list(features.FEATURE_COLUMNS),
        **model_kwargs,
    )

    fit_kwargs: Dict[str, Any] = {}
    if model_backend == "xgb":
        fit_kwargs = {
            "sample_weight": weight_train,
            "eval_set": (x_val, y_val),
            "eval_sample_weight": weight_val,
        }

    train_start = time.perf_counter()
    fit_info = model.fit(x_train, y_train, **fit_kwargs)
    training_time_seconds = time.perf_counter() - train_start

    model.save(str(model_path))

    pred_train_start = time.perf_counter()
    train_pred = model.predict_proba(x_train)[:, 1]
    prediction_time_train_seconds = time.perf_counter() - pred_train_start

    pred_val_start = time.perf_counter()
    val_pred = model.predict_proba(x_val)[:, 1]
    prediction_time_val_seconds = time.perf_counter() - pred_val_start

    auc_train = _binary_auc(y_train, train_pred)
    auc_val = _binary_auc(y_val, val_pred)

    plot_metrics: Dict[str, float] = {}
    if not args.skip_plots and resolved_test is None:
        raise RuntimeError("Could not resolve test dataset for diagnostics. Pass --test explicitly.")
    if not args.skip_plots and resolved_test is not None:
        plot_metrics = plotting.generate_training_diagnostics(
            model=str(model_path),
            model_name=model_backend,
            train_dataset=args.train,
            val_dataset=args.val,
            test_dataset=resolved_test,
            output_root=args.plot_root,
        )

    report = {
        "schema_version": pio.SCHEMA_VERSION,
        "model_backend": model_backend,
        "model_output": str(model_path),
        "feature_columns": list(features.FEATURE_COLUMNS),
        "train_rows_original": int(x_train_full.shape[0]),
        "train_rows": int(x_train.shape[0]),
        "val_rows": int(x_val.shape[0]),
        "number_training_samples_used": int(x_train.shape[0]),
        "sampling": sampling,
        "class_balance": {
            "train_signal": pos_train,
            "train_background": neg_train,
            "val_signal": pos_val,
            "val_background": neg_val,
        },
        "sample_weights": {
            "enabled": bool(model_backend == "xgb" and args.use_sample_weights),
            "train_positive_weight": float(pos_weight),
        },
        "best_iteration": int(fit_info.get("best_iteration", -1)) if model_backend == "xgb" else None,
        "best_score": float(fit_info.get("best_score", math.nan)) if model_backend == "xgb" else float("nan"),
        "AUC_train": float(auc_train),
        "AUC_val": float(auc_val),
        "training_time_seconds": float(training_time_seconds),
        "prediction_time_train_seconds": float(prediction_time_train_seconds),
        "prediction_time_val_seconds": float(prediction_time_val_seconds),
        "timing": {
            "training_time_seconds": float(training_time_seconds),
            "prediction_time_train_seconds": float(prediction_time_train_seconds),
            "prediction_time_val_seconds": float(prediction_time_val_seconds),
        },
        "metrics": {
            "train_auc": float(auc_train),
            "val_auc": float(auc_val),
            "val_logloss": _binary_logloss(y_val, val_pred),
            "eval_history": fit_info.get("eval_history", {}),
            "plot_metrics": plot_metrics,
        },
        "test_dataset_for_diagnostics": resolved_test,
    }

    pio.write_json(report_path, report)
    with open(output_dir / "training_statistics.md", "w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "# Training Statistics",
                    "",
                    f"- Model backend: `{model_backend}`",
                    f"- Train rows (original): `{x_train_full.shape[0]}`",
                    f"- Train rows (used): `{x_train.shape[0]}`",
                    f"- Validation rows: `{x_val.shape[0]}`",
                    f"- Train signal/background: `{pos_train}` / `{neg_train}`",
                    f"- Validation signal/background: `{pos_val}` / `{neg_val}`",
                    f"- AUC train: `{report['metrics']['train_auc']:.6f}`",
                    f"- AUC val: `{report['metrics']['val_auc']:.6f}`",
                    f"- Validation logloss: `{report['metrics']['val_logloss']:.6f}`",
                    f"- Training time [s]: `{training_time_seconds:.4f}`",
                    f"- Prediction time train [s]: `{prediction_time_train_seconds:.4f}`",
                    f"- Prediction time val [s]: `{prediction_time_val_seconds:.4f}`",
                ]
            )
            + "\n"
        )

    snapshot_parameters: Dict[str, Any] = {
        "model": model_backend,
        "max_training_samples": args.max_training_samples,
        "tabpfn_balance_classes": bool(args.tabpfn_balance_classes),
        "plot_root": args.plot_root,
        "skip_plots": args.skip_plots,
        "test_dataset_for_diagnostics": resolved_test,
    }
    if model_backend == "xgb":
        snapshot_parameters.update(
            {
                "num_boost_round": args.num_boost_round,
                "early_stopping_rounds": args.early_stopping_rounds,
                "eta": args.eta,
                "max_depth": args.max_depth,
                "min_child_weight": args.min_child_weight,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "reg_lambda": args.reg_lambda,
                "tree_method": args.tree_method,
                "use_sample_weights": args.use_sample_weights,
            }
        )
    else:
        snapshot_parameters.update(
            {
                "tabpfn_device": args.tabpfn_device,
                "tabpfn_n_ensemble_configurations": args.tabpfn_n_ensemble_configurations,
            }
        )

    pio.write_config_snapshot(
        output_dir=output_dir,
        stage="train",
        input_files=[args.train, args.val] + ([resolved_test] if resolved_test is not None else []),
        parameters=snapshot_parameters,
        seed=args.seed,
    )


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Stage 3: train model")
    parser.add_argument("--train", required=True, help="Train parquet path")
    parser.add_argument("--val", required=True, help="Validation parquet path")
    parser.add_argument("--test", default=None, help="Test parquet path used for diagnostics")
    parser.add_argument(
        "--model",
        choices=list(MODEL_BACKENDS),
        default="xgb",
        help="Model backend (xgb or tabpfn)",
    )
    parser.add_argument("--output-dir", default="artifacts/train", help="Output directory")
    parser.add_argument("--model-out", default=None, help="Model output path (default depends on --model)")
    parser.add_argument("--report-out", default=None, help="Training report path (default depends on --model)")
    parser.add_argument(
        "--max-training-samples",
        type=int,
        default=None,
        help="Optional cap for training rows (reproducible random subsampling)",
    )

    parser.add_argument("--num-boost-round", type=int, default=400, help="Max XGBoost boosting rounds")
    parser.add_argument("--early-stopping-rounds", type=int, default=30, help="XGBoost early stopping rounds")
    parser.add_argument("--eta", type=float, default=0.05, help="XGBoost learning rate")
    parser.add_argument("--max-depth", type=int, default=6, help="XGBoost tree max depth")
    parser.add_argument("--min-child-weight", type=float, default=1.0, help="XGBoost min_child_weight")
    parser.add_argument("--subsample", type=float, default=0.8, help="XGBoost row subsample")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="XGBoost feature subsample")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="XGBoost L2 regularization")
    parser.add_argument("--tree-method", default="hist", help="XGBoost tree_method")
    parser.add_argument("--use-sample-weights", action="store_true", help="Enable XGBoost class-imbalance weighting")

    parser.add_argument("--tabpfn-device", default="auto", help="TabPFN device hint")
    parser.add_argument(
        "--tabpfn-n-ensemble-configurations",
        type=int,
        default=None,
        help="Optional TabPFN n_ensemble_configurations",
    )
    parser.add_argument(
        "--tabpfn-balance-classes",
        action="store_true",
        help="For TabPFN: enforce 50/50 signal/background train sampling using --max-training-samples",
    )

    parser.add_argument("--plot-root", default="plots", help="Root directory for generated training diagnostic plots")
    parser.add_argument("--skip-plots", action="store_true", help="Skip automatic training diagnostic plotting")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.set_defaults(func=run)
