#!/usr/bin/env python3
"""Reusable plotting and statistical utilities for pipeline stages."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import features
import triplet_io as pio
from models import (
    MODEL_BACKENDS,
    infer_model_backend_from_path,
    inference_score_column,
    load_model,
    normalize_model_backend,
)


def _require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for plotting from parquet artifacts") from exc
    return pq


def _read_parquet_columns(path: str, columns: Sequence[str]) -> Dict[str, np.ndarray]:
    pq = _require_pyarrow_parquet()
    table = pq.read_table(path, columns=list(columns))
    as_dict = table.to_pydict()
    return {name: np.asarray(as_dict[name]) for name in columns}


def _parquet_column_names(path: str) -> Sequence[str]:
    pq = _require_pyarrow_parquet()
    return list(pq.read_schema(path).names)


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(np.int64)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_pos = float(np.sum(ranks[y == 1]))
    return float((sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg))


def _roc_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = y_true.astype(np.int64)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(-y_score)
    y_sorted = y[order]
    s_sorted = y_score[order]

    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)

    change = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    idx = np.where(change)[0]
    if idx[-1] != len(s_sorted) - 1:
        idx = np.r_[idx, len(s_sorted) - 1]

    tpr = cum_tp[idx] / float(pos)
    fpr = cum_fp[idx] / float(neg)
    return np.r_[0.0, fpr, 1.0], np.r_[0.0, tpr, 1.0]


def _threshold_rates(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = y_true.astype(np.int64)
    sig = y_score[y == 1]
    bkg = y_score[y == 0]

    tpr = np.zeros_like(thresholds, dtype=np.float64)
    fpr = np.zeros_like(thresholds, dtype=np.float64)

    if sig.size > 0:
        tpr = np.asarray([(sig >= t).mean() for t in thresholds], dtype=np.float64)
    if bkg.size > 0:
        fpr = np.asarray([(bkg >= t).mean() for t in thresholds], dtype=np.float64)
    return tpr, fpr


def _norm_weights(values: np.ndarray) -> np.ndarray:
    return np.ones(values.shape[0], dtype=np.float64) / float(values.shape[0])


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _bin_edges(
    signal: np.ndarray,
    background: np.ndarray,
    n_bins: int = 60,
    fixed_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    sig = _finite(signal)
    bkg = _finite(background)
    if sig.size == 0 and bkg.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)

    if fixed_range is not None:
        low, high = fixed_range
    else:
        merged = np.concatenate([sig, bkg]) if sig.size and bkg.size else (sig if sig.size else bkg)
        low = float(np.quantile(merged, 0.001))
        high = float(np.quantile(merged, 0.999))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(merged))
            high = float(np.max(merged))
        if high <= low:
            high = low + 1.0

    return np.linspace(low, high, n_bins + 1)


def _plot_shape_overlay(
    signal: np.ndarray,
    background: np.ndarray,
    bins: np.ndarray,
    outpath: Path,
    title: str,
    xlabel: str,
    signal_label: str = "Signal",
    background_label: str = "Background",
    yscale_log: bool = False,
) -> None:
    sig = _finite(signal)
    bkg = _finite(background)

    plt.figure(figsize=(8, 5))
    if bkg.size > 0:
        plt.hist(
            bkg,
            bins=bins,
            weights=_norm_weights(bkg),
            histtype="stepfilled",
            alpha=0.40,
            label=background_label,
        )
    if sig.size > 0:
        plt.hist(
            sig,
            bins=bins,
            weights=_norm_weights(sig),
            histtype="step",
            linewidth=2.0,
            label=signal_label,
        )
    if yscale_log:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Normalized entries")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _plot_correlation_matrix(matrix: np.ndarray, labels: Sequence[str], outpath: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    im = plt.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _model_label(model_name: str) -> str:
    model = normalize_model_backend(model_name)
    if model == "xgb":
        return "XGBoost"
    return "TabPFN"


def _resolve_score_column(path: str, preferred: Optional[str]) -> str:
    names = set(_parquet_column_names(path))
    candidates = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(["score_tabpfn", "score_xgb", "score"])
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise RuntimeError(f"Could not find score column in {path}; available columns: {sorted(names)}")


def generate_feature_validation_plots(dataset: str, split_name: str, output_root: str = "plots") -> Dict[str, float]:
    out_dir = pio.ensure_dir(Path(output_root) / "features" / split_name)

    required = list(features.FEATURE_COLUMNS) + list(features.OBSERVABLE_COLUMNS) + ["is_truth"]
    values = _read_parquet_columns(dataset, required)

    y = values["is_truth"].astype(np.int64)
    sig_mask = y == 1
    bkg_mask = y == 0

    for name in features.FEATURE_COLUMNS:
        sig = values[name][sig_mask]
        bkg = values[name][bkg_mask]
        bins = _bin_edges(sig, bkg, n_bins=70)
        _plot_shape_overlay(
            sig,
            bkg,
            bins,
            out_dir / f"{name}.png",
            title=f"{split_name}: {name} (shape)",
            xlabel=name,
        )

    observable_specs = [
        ("m123", "Triplet invariant mass m123 [GeV]", "m123"),
        ("triplet_pt", "Triplet pT [GeV]", "triplet_pt"),
        ("triplet_eta", "Triplet eta", "triplet_eta"),
        ("triplet_phi", "Triplet phi", "triplet_phi"),
    ]

    for column, xlabel, stem in observable_specs:
        sig = values[column][sig_mask]
        bkg = values[column][bkg_mask]
        fixed = (-math.pi, math.pi) if column == "triplet_phi" else None
        bins = _bin_edges(sig, bkg, n_bins=70, fixed_range=fixed)
        _plot_shape_overlay(
            sig,
            bkg,
            bins,
            out_dir / f"{stem}.png",
            title=f"{split_name}: {column} (shape)",
            xlabel=xlabel,
        )

    mij_sig = np.concatenate([
        values["mij_ab"][sig_mask],
        values["mij_ac"][sig_mask],
        values["mij_bc"][sig_mask],
    ])
    mij_bkg = np.concatenate([
        values["mij_ab"][bkg_mask],
        values["mij_ac"][bkg_mask],
        values["mij_bc"][bkg_mask],
    ])
    bins_mij = _bin_edges(mij_sig, mij_bkg, n_bins=70)
    _plot_shape_overlay(
        mij_sig,
        mij_bkg,
        bins_mij,
        out_dir / "mij.png",
        title=f"{split_name}: pair invariant mass mij (shape)",
        xlabel="mij [GeV]",
    )

    if split_name == "train":
        names = list(features.FEATURE_COLUMNS)
        sig_matrix = np.column_stack([values[name][sig_mask].astype(np.float64) for name in names])
        bkg_matrix = np.column_stack([values[name][bkg_mask].astype(np.float64) for name in names])

        if sig_matrix.shape[0] >= 2:
            corr_sig = np.nan_to_num(np.corrcoef(sig_matrix, rowvar=False), nan=0.0)
        else:
            corr_sig = np.zeros((len(names), len(names)), dtype=np.float64)

        if bkg_matrix.shape[0] >= 2:
            corr_bkg = np.nan_to_num(np.corrcoef(bkg_matrix, rowvar=False), nan=0.0)
        else:
            corr_bkg = np.zeros((len(names), len(names)), dtype=np.float64)

        _plot_correlation_matrix(
            corr_sig,
            names,
            out_dir / "correlation_signal.png",
            title="Train feature correlations (signal)",
        )
        _plot_correlation_matrix(
            corr_bkg,
            names,
            out_dir / "correlation_background.png",
            title="Train feature correlations (background)",
        )

    summary = {
        "dataset": dataset,
        "split": split_name,
        "rows": int(y.shape[0]),
        "signal_rows": int(np.sum(sig_mask)),
        "background_rows": int(np.sum(bkg_mask)),
    }
    pio.write_json(out_dir / "feature_plot_summary.json", summary)
    return summary


def _load_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    cols = list(features.FEATURE_COLUMNS) + ["is_truth"]
    values = _read_parquet_columns(path, cols)
    x = np.column_stack([values[c].astype(np.float32) for c in features.FEATURE_COLUMNS])
    y = values["is_truth"].astype(np.int8)
    return x, y


def _resolve_training_inputs(
    model_path: str,
    train_path: Optional[str],
    val_path: Optional[str],
    test_path: Optional[str],
) -> Tuple[str, str, str]:
    resolved_train = train_path
    resolved_val = val_path
    resolved_test = test_path

    config_path = Path(model_path).parent / "config_snapshot.json"
    if (resolved_train is None or resolved_val is None) and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        input_files = payload.get("input_files", [])
        if len(input_files) >= 2:
            resolved_train = resolved_train or input_files[0]
            resolved_val = resolved_val or input_files[1]

    if resolved_train is None or resolved_val is None:
        raise RuntimeError("Could not resolve train/val datasets. Pass --train and --val explicitly.")

    if resolved_test is None:
        candidate = str(Path(resolved_val).with_name("test.parquet"))
        if Path(candidate).exists():
            resolved_test = candidate
        else:
            raise RuntimeError("Could not resolve test dataset. Pass --test explicitly.")

    return resolved_train, resolved_val, resolved_test


def generate_training_diagnostics(
    model: str,
    model_name: str,
    train_dataset: str,
    val_dataset: str,
    test_dataset: str,
    output_root: str = "plots",
) -> Dict[str, float]:
    backend = normalize_model_backend(model_name)
    label = _model_label(backend)
    out_dir = pio.ensure_dir(Path(output_root) / "training" / backend)

    x_train, y_train = _load_xy(train_dataset)
    x_val, y_val = _load_xy(val_dataset)
    x_test, y_test = _load_xy(test_dataset)

    trained_model = load_model(
        model_backend=backend,
        path=model,
        feature_columns=list(features.FEATURE_COLUMNS),
    )

    t_pred_start = time.perf_counter()
    p_train = trained_model.predict_proba(x_train)[:, 1]
    p_val = trained_model.predict_proba(x_val)[:, 1]
    p_test = trained_model.predict_proba(x_test)[:, 1]
    prediction_time_seconds = time.perf_counter() - t_pred_start

    auc_train = _auc(y_train, p_train)
    auc_val = _auc(y_val, p_val)
    auc_test = _auc(y_test, p_test)

    fpr_train, tpr_train = _roc_points(y_train, p_train)
    fpr_val, tpr_val = _roc_points(y_val, p_val)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_train, tpr_train, label=f"Train (AUC={auc_train:.4f})")
    plt.plot(fpr_val, tpr_val, label=f"Validation (AUC={auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{label} ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_train_val.png", dpi=160)
    plt.close()

    sig_train = p_train[y_train == 1]
    bkg_train = p_train[y_train == 0]
    bins_score = np.linspace(0.0, 1.0, 80)
    _plot_shape_overlay(
        sig_train,
        bkg_train,
        bins_score,
        out_dir / "score_train_signal_background.png",
        title=f"{label}: training score distribution (shape)",
        xlabel=f"{label} score",
        yscale_log=False,
    )

    sig_test = p_test[y_test == 1]
    bkg_test = p_test[y_test == 0]

    plt.figure(figsize=(8, 5))
    if sig_train.size > 0:
        plt.hist(
            sig_train,
            bins=bins_score,
            weights=_norm_weights(sig_train),
            histtype="step",
            linewidth=2.0,
            label="Signal train",
        )
    if sig_test.size > 0:
        plt.hist(
            sig_test,
            bins=bins_score,
            weights=_norm_weights(sig_test),
            histtype="step",
            linewidth=2.0,
            linestyle="--",
            label="Signal test",
        )
    if bkg_train.size > 0:
        plt.hist(
            bkg_train,
            bins=bins_score,
            weights=_norm_weights(bkg_train),
            histtype="step",
            linewidth=2.0,
            label="Background train",
        )
    if bkg_test.size > 0:
        plt.hist(
            bkg_test,
            bins=bins_score,
            weights=_norm_weights(bkg_test),
            histtype="step",
            linewidth=2.0,
            linestyle="--",
            label="Background test",
        )
    plt.xlabel(f"{label} score")
    plt.ylabel("Normalized entries")
    plt.title(f"{label}: overtraining check score shapes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "overtraining_score_comparison.png", dpi=160)
    plt.close()

    thresholds = np.linspace(0.0, 1.0, 201)
    tpr, fpr = _threshold_rates(y_val, p_val, thresholds)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, tpr, label="TPR")
    plt.plot(thresholds, fpr, label="FPR")
    plt.xlabel("Score threshold")
    plt.ylabel("Rate")
    plt.ylim(0.0, 1.0)
    plt.title(f"{label}: efficiency curves (validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "efficiency_curves.png", dpi=160)
    plt.close()

    metrics = {
        "model": backend,
        "train_rows": int(y_train.shape[0]),
        "val_rows": int(y_val.shape[0]),
        "test_rows": int(y_test.shape[0]),
        "auc_train": float(auc_train),
        "auc_val": float(auc_val),
        "auc_test": float(auc_test),
        "prediction_time_seconds": float(prediction_time_seconds),
    }
    pio.write_json(out_dir / "training_plot_metrics.json", metrics)
    return metrics


def generate_inference_plots(
    inference_dataset: str,
    model_name: str,
    output_root: str = "plots",
    score_column: Optional[str] = None,
) -> Dict[str, float]:
    backend = normalize_model_backend(model_name)
    label = _model_label(backend)
    out_dir = pio.ensure_dir(Path(output_root) / "inference" / backend)

    resolved_score_column = _resolve_score_column(
        path=inference_dataset,
        preferred=score_column if score_column is not None else inference_score_column(backend),
    )
    available_columns = set(_parquet_column_names(inference_dataset))
    required_columns = [resolved_score_column, "is_truth"]
    if "m123" in available_columns:
        required_columns.append("m123")

    values = _read_parquet_columns(inference_dataset, required_columns)
    score = values[resolved_score_column].astype(np.float64)
    y = values["is_truth"].astype(np.int64)
    m123 = values["m123"].astype(np.float64) if "m123" in values else None

    auc = _auc(y, score)
    fpr, tpr = _roc_points(y, score)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"Test ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{label} Inference ROC (test)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_test.png", dpi=160)
    plt.close()

    sig = score[y == 1]
    bkg = score[y == 0]
    bins_score = np.linspace(0.0, 1.0, 80)
    _plot_shape_overlay(
        sig,
        bkg,
        bins_score,
        out_dir / "score_distribution_test.png",
        title=f"{label}: inference score distribution (shape)",
        xlabel=f"{label} score",
    )

    thresholds = np.linspace(0.0, 1.0, 201)
    tpr_thr, fpr_thr = _threshold_rates(y, score, thresholds)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, tpr_thr, label="TPR")
    plt.xlabel("Score threshold")
    plt.ylabel("True Positive Rate")
    plt.ylim(0.0, 1.0)
    plt.title(f"{label}: TPR vs threshold (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tpr_vs_threshold.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, fpr_thr, label="FPR")
    plt.xlabel("Score threshold")
    plt.ylabel("False Positive Rate")
    plt.ylim(0.0, 1.0)
    plt.title(f"{label}: FPR vs threshold (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fpr_vs_threshold.png", dpi=160)
    plt.close()

    if m123 is not None:
        plt.figure(figsize=(8, 5))
        plt.hexbin(m123, score, gridsize=90, bins="log", mincnt=1)
        plt.colorbar(label="log10(N)")
        plt.xlabel("Triplet invariant mass m123 [GeV]")
        plt.ylabel(f"{label} score")
        plt.title(f"{label}: score vs triplet invariant mass")
        plt.tight_layout()
        plt.savefig(out_dir / "score_vs_m123.png", dpi=160)
        plt.close()

    metrics = {
        "model": backend,
        "score_column": resolved_score_column,
        "has_m123": bool(m123 is not None),
        "rows": int(score.shape[0]),
        "signal_rows": int(np.sum(y == 1)),
        "background_rows": int(np.sum(y == 0)),
        "auc_test": float(auc),
        "score_mean_signal": float(np.mean(sig)) if sig.size else float("nan"),
        "score_mean_background": float(np.mean(bkg)) if bkg.size else float("nan"),
    }
    pio.write_json(out_dir / "inference_plot_metrics.json", metrics)
    return metrics


def generate_inference_comparison_plots(
    inference_datasets: Dict[str, str],
    output_root: str = "plots",
) -> Dict[str, float]:
    resolved: Dict[str, Dict[str, np.ndarray | float | str]] = {}
    for backend, path in inference_datasets.items():
        model = normalize_model_backend(backend)
        score_col = _resolve_score_column(path=path, preferred=inference_score_column(model))
        values = _read_parquet_columns(path, [score_col, "is_truth"])
        score = values[score_col].astype(np.float64)
        y = values["is_truth"].astype(np.int64)
        auc = _auc(y, score)
        fpr, tpr = _roc_points(y, score)
        resolved[model] = {
            "path": path,
            "score_column": score_col,
            "score": score,
            "y": y,
            "auc": float(auc),
            "fpr": fpr,
            "tpr": tpr,
        }

    if "xgb" not in resolved or "tabpfn" not in resolved:
        raise RuntimeError("Comparison plots require both xgb and tabpfn inference datasets")

    out_dir = pio.ensure_dir(Path(output_root) / "inference" / "comparison")

    plt.figure(figsize=(6, 6))
    for backend in ("xgb", "tabpfn"):
        payload = resolved[backend]
        plt.plot(
            payload["fpr"],
            payload["tpr"],
            label=f"{_model_label(backend)} (AUC={float(payload['auc']):.4f})",
        )
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison (test)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_comparison.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, backend in zip(axes, ("xgb", "tabpfn")):
        payload = resolved[backend]
        score = np.asarray(payload["score"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.int64)
        sig = score[y == 1]
        bkg = score[y == 0]
        bins = np.linspace(0.0, 1.0, 80)
        if bkg.size > 0:
            axis.hist(
                bkg,
                bins=bins,
                weights=_norm_weights(bkg),
                histtype="stepfilled",
                alpha=0.35,
                label="Background",
            )
        if sig.size > 0:
            axis.hist(
                sig,
                bins=bins,
                weights=_norm_weights(sig),
                histtype="step",
                linewidth=2.0,
                label=f"Signal (AUC={float(payload['auc']):.4f})",
            )
        axis.set_title(f"{_model_label(backend)} score")
        axis.set_xlabel(f"{_model_label(backend)} score")
        axis.set_ylabel("Normalized entries")
        axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution_signal_background_comparison.png", dpi=160)
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    for backend in ("xgb", "tabpfn"):
        payload = resolved[backend]
        score = np.asarray(payload["score"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.int64)
        sig = score[y == 1]
        bkg = score[y == 0]
        bins = np.linspace(0.0, 1.0, 80)
        if sig.size > 0:
            plt.hist(
                sig,
                bins=bins,
                weights=_norm_weights(sig),
                histtype="step",
                linewidth=2.0,
                label=f"{_model_label(backend)} signal (AUC={float(payload['auc']):.4f})",
            )
        if bkg.size > 0:
            plt.hist(
                bkg,
                bins=bins,
                weights=_norm_weights(bkg),
                histtype="step",
                linewidth=1.8,
                linestyle="--",
                label=f"{_model_label(backend)} background",
            )
    plt.xlabel("Model score")
    plt.ylabel("Normalized entries")
    plt.title("Score Distribution Comparison (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_distribution_model_overlay.png", dpi=160)
    plt.close()

    metrics = {
        "models": {
            "xgb": {
                "dataset": str(resolved["xgb"]["path"]),
                "score_column": str(resolved["xgb"]["score_column"]),
                "auc_test": float(resolved["xgb"]["auc"]),
                "rows": int(np.asarray(resolved["xgb"]["score"]).shape[0]),
            },
            "tabpfn": {
                "dataset": str(resolved["tabpfn"]["path"]),
                "score_column": str(resolved["tabpfn"]["score_column"]),
                "auc_test": float(resolved["tabpfn"]["auc"]),
                "rows": int(np.asarray(resolved["tabpfn"]["score"]).shape[0]),
            },
        }
    }
    pio.write_json(out_dir / "inference_comparison_metrics.json", metrics)
    return {
        "auc_test_xgb": float(resolved["xgb"]["auc"]),
        "auc_test_tabpfn": float(resolved["tabpfn"]["auc"]),
    }


def run_plot_features(args: argparse.Namespace) -> None:
    split_name = args.split_name
    if split_name is None:
        name = Path(args.dataset).name.lower()
        if "train" in name:
            split_name = "train"
        elif "test" in name:
            split_name = "test"
        else:
            split_name = "dataset"

    generate_feature_validation_plots(
        dataset=args.dataset,
        split_name=split_name,
        output_root=args.output_root,
    )


def run_plot_training(args: argparse.Namespace) -> None:
    model_name = normalize_model_backend(args.model_name) if args.model_name else infer_model_backend_from_path(args.model)
    train_path, val_path, test_path = _resolve_training_inputs(
        model_path=args.model,
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
    )
    generate_training_diagnostics(
        model=args.model,
        model_name=model_name,
        train_dataset=train_path,
        val_dataset=val_path,
        test_dataset=test_path,
        output_root=args.output_root,
    )


def run_plot_inference(args: argparse.Namespace) -> None:
    if args.inference_xgb is not None or args.inference_tabpfn is not None:
        if args.inference_xgb is None or args.inference_tabpfn is None:
            raise RuntimeError("Provide both --inference-xgb and --inference-tabpfn for comparison plotting")
        generate_inference_comparison_plots(
            inference_datasets={"xgb": args.inference_xgb, "tabpfn": args.inference_tabpfn},
            output_root=args.output_root,
        )
        return

    if args.inference is None:
        raise RuntimeError("Provide --inference for single-model plotting")

    model_name = normalize_model_backend(args.model_name) if args.model_name else infer_model_backend_from_path(args.inference)
    generate_inference_plots(
        inference_dataset=args.inference,
        model_name=model_name,
        output_root=args.output_root,
    )


def register_plot_subparsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    plot_features = subparsers.add_parser(
        "plot_features",
        help="Generate feature/non-training shape-validation plots from a parquet dataset",
    )
    plot_features.add_argument("--dataset", required=True, help="Dataset parquet path (train.parquet or test.parquet)")
    plot_features.add_argument("--split-name", default=None, help="Split label used in output path/name (train/test)")
    plot_features.add_argument("--output-root", default="plots", help="Root directory for plot outputs")
    plot_features.set_defaults(func=run_plot_features)

    plot_training = subparsers.add_parser(
        "plot_training",
        help="Generate training diagnostics from model + parquet datasets",
    )
    plot_training.add_argument("--model", required=True, help="Trained model path")
    plot_training.add_argument(
        "--model-name",
        choices=list(MODEL_BACKENDS),
        default=None,
        help="Model backend (auto-inferred from model path when omitted)",
    )
    plot_training.add_argument("--train", default=None, help="Train parquet path (auto-resolved if omitted)")
    plot_training.add_argument("--val", default=None, help="Validation parquet path (auto-resolved if omitted)")
    plot_training.add_argument("--test", default=None, help="Test parquet path (auto-resolved if omitted)")
    plot_training.add_argument("--output-root", default="plots", help="Root directory for plot outputs")
    plot_training.set_defaults(func=run_plot_training)

    plot_inference = subparsers.add_parser(
        "plot_inference",
        help="Generate inference evaluation plots from inference parquet",
    )
    plot_inference.add_argument("--inference", default=None, help="Single-model inference parquet path")
    plot_inference.add_argument(
        "--model-name",
        choices=list(MODEL_BACKENDS),
        default=None,
        help="Model backend for single-model inference plotting",
    )
    plot_inference.add_argument("--inference-xgb", default=None, help="XGBoost inference parquet for comparison")
    plot_inference.add_argument("--inference-tabpfn", default=None, help="TabPFN inference parquet for comparison")
    plot_inference.add_argument("--output-root", default="plots", help="Root directory for plot outputs")
    plot_inference.set_defaults(func=run_plot_inference)
