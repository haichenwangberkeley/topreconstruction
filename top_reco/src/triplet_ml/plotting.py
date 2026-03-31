#!/usr/bin/env python3
"""Reusable plotting and statistical utilities for pipeline stages."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import features
from . import triplet_io as pio
from .models import (
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


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "feature"


def _range_from_mean_std(distributions: Sequence[np.ndarray]) -> Tuple[float, float]:
    lowers: List[float] = []
    uppers: List[float] = []
    finite_all: List[np.ndarray] = []

    for dist in distributions:
        finite = _finite(dist)
        if finite.size == 0:
            continue
        finite_all.append(finite)
        mu = float(np.mean(finite))
        sigma = float(np.std(finite))
        if not np.isfinite(mu):
            continue
        if not np.isfinite(sigma):
            sigma = 0.0
        lowers.append(mu - 3.0 * sigma)
        uppers.append(mu + 3.0 * sigma)

    if not lowers or not uppers:
        return 0.0, 1.0

    low = float(min(lowers))
    high = float(max(uppers))
    if high <= low:
        merged = np.concatenate(finite_all) if finite_all else np.asarray([0.0, 1.0], dtype=np.float64)
        min_v = float(np.min(merged))
        max_v = float(np.max(merged))
        if max_v > min_v:
            low, high = min_v, max_v
        else:
            low, high = min_v - 0.5, max_v + 0.5
    return low, high


def _normalized_hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(_finite(values), bins=edges)[0].astype(np.float64)
    total = float(np.sum(counts))
    if total > 0.0:
        counts /= total
    return counts


def _normalized_hist_and_var(values: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.histogram(_finite(values), bins=edges)[0].astype(np.float64)
    # Unweighted case: variance per bin is N before normalization.
    var = counts.copy()
    total = float(np.sum(counts))
    if total > 0.0:
        counts /= total
        var /= (total * total)
    return counts, var


def _default_numeric_feature_columns(path: str, excluded: Sequence[str]) -> List[str]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for feature discovery from parquet schema") from exc

    excluded_set = set(excluded)
    schema = pq.read_schema(path)
    columns: List[str] = []
    for field in schema:
        if field.name in excluded_set:
            continue
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            columns.append(field.name)
    return columns


def generate_cut_comparison_plots(
    inference_dataset: str,
    output_root: str = "plots",
    score_cut: float = 0.5,
    score_column: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    bins: int = 20,
    nominal: str = "true_pass",
) -> Dict[str, float]:
    """Create invariant-compliant 1D overlays: true/fake x pass/fail(score-cut) for each feature."""

    if bins <= 0:
        raise RuntimeError("bins must be a positive integer")

    resolved_score_column = _resolve_score_column(path=inference_dataset, preferred=score_column)
    excluded = [resolved_score_column, "is_truth", "event_id"]

    selected_columns = list(columns) if columns is not None and len(columns) > 0 else None
    if selected_columns is None:
        selected_columns = _default_numeric_feature_columns(path=inference_dataset, excluded=excluded)
    if len(selected_columns) == 0:
        raise RuntimeError("No numeric feature columns available for cut-comparison plotting")

    available_columns = set(_parquet_column_names(inference_dataset))
    missing_columns = [name for name in selected_columns if name not in available_columns]
    if missing_columns:
        raise RuntimeError(
            f"Requested columns are not present in {inference_dataset}: {missing_columns}. "
            f"Available columns: {sorted(available_columns)}"
        )

    required_columns = [resolved_score_column, "is_truth"] + selected_columns
    values = _read_parquet_columns(inference_dataset, required_columns)
    score = np.asarray(values[resolved_score_column], dtype=np.float64)
    y = np.asarray(values["is_truth"], dtype=np.int64)
    valid_base = np.isfinite(score) & (y >= 0) & (y <= 1)
    pass_cut = score > float(score_cut)

    distributions = [
        ("true_pass", f"True, score > {score_cut:g}", "tab:blue", "-"),
        ("true_fail", f"True, score <= {score_cut:g}", "tab:blue", "--"),
        ("fake_pass", f"Fake, score > {score_cut:g}", "tab:red", "-"),
        ("fake_fail", f"Fake, score <= {score_cut:g}", "tab:red", "--"),
    ]
    keys = [k for k, _, _, _ in distributions]
    if nominal not in keys:
        raise RuntimeError(f"Invalid nominal '{nominal}'. Supported values: {keys}")

    # Invariant policy: if user did not explicitly choose nominal, first provided distribution is nominal.
    nominal_idx = keys.index(nominal)
    ordered = [distributions[nominal_idx]] + [item for idx, item in enumerate(distributions) if idx != nominal_idx]

    out_dir = pio.ensure_dir(Path(output_root) / "cut_comparison")
    summary: Dict[str, object] = {
        "inference_dataset": inference_dataset,
        "score_column": resolved_score_column,
        "score_cut": float(score_cut),
        "bins": int(bins),
        "nominal_distribution": ordered[0][0],
        "features": {},
    }

    plotted_features = 0
    skipped_features = 0
    for feature in selected_columns:
        feature_values = np.asarray(values[feature], dtype=np.float64)
        finite_feature = np.isfinite(feature_values)

        masks = {
            "true_pass": valid_base & finite_feature & (y == 1) & pass_cut,
            "true_fail": valid_base & finite_feature & (y == 1) & (~pass_cut),
            "fake_pass": valid_base & finite_feature & (y == 0) & pass_cut,
            "fake_fail": valid_base & finite_feature & (y == 0) & (~pass_cut),
        }

        slices = {key: feature_values[mask] for key, mask in masks.items()}
        nonempty = [arr for arr in slices.values() if arr.size > 0]
        if len(nonempty) == 0:
            skipped_features += 1
            continue

        xlow, xhigh = _range_from_mean_std([slices[key] for key, _, _, _ in ordered])
        if not np.isfinite(xlow) or not np.isfinite(xhigh) or xhigh <= xlow:
            skipped_features += 1
            continue
        edges = np.linspace(xlow, xhigh, bins + 1, dtype=np.float64)

        hist: Dict[str, np.ndarray] = {}
        var_hist: Dict[str, np.ndarray] = {}
        for key, _, _, _ in ordered:
            hist[key], var_hist[key] = _normalized_hist_and_var(slices[key], edges)

        nominal_key = ordered[0][0]
        nominal_hist = hist[nominal_key]
        nominal_var = var_hist[nominal_key]
        nominal_err = np.sqrt(np.maximum(nominal_var, 0.0))
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig, (ax_top, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(8, 6),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
        )

        for key, label, color, linestyle in ordered:
            yvals = hist[key]
            yerr = np.sqrt(np.maximum(var_hist[key], 0.0))
            step = np.r_[yvals, yvals[-1] if yvals.size > 0 else 0.0]
            ax_top.step(edges, step, where="post", label=label, color=color, linestyle=linestyle, linewidth=1.8)
            ax_top.errorbar(
                centers,
                yvals,
                yerr=yerr,
                fmt="none",
                ecolor=color,
                elinewidth=0.9,
                alpha=0.9,
                capsize=0,
            )
        ax_top.set_ylabel("Normalized entries")
        ax_top.set_title(f"{feature}: true/fake x pass/fail")
        ax_top.legend(fontsize=8)

        ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        denom_mask = nominal_hist > 0.0
        nominal_rel_unc = np.zeros_like(nominal_hist, dtype=np.float64)
        nominal_rel_unc[denom_mask] = nominal_err[denom_mask] / nominal_hist[denom_mask]
        band_low = np.full_like(nominal_hist, np.nan, dtype=np.float64)
        band_high = np.full_like(nominal_hist, np.nan, dtype=np.float64)
        band_low[denom_mask] = 1.0 - nominal_rel_unc[denom_mask]
        band_high[denom_mask] = 1.0 + nominal_rel_unc[denom_mask]
        ax_ratio.fill_between(
            centers,
            band_low,
            band_high,
            step="mid",
            color="gray",
            alpha=0.20,
            label=f"{nominal_key} stat. unc.",
            zorder=1,
        )
        for key, label, color, linestyle in ordered[1:]:
            num = hist[key]
            var_num = var_hist[key]
            ratio = np.full_like(num, np.nan, dtype=np.float64)
            ratio_err = np.full_like(num, np.nan, dtype=np.float64)

            valid_ratio = denom_mask.copy()
            ratio[valid_ratio] = num[valid_ratio] / nominal_hist[valid_ratio]

            # Propagation for independent numerator/denominator:
            # Var(A/B) = Var(A)/B^2 + A^2*Var(B)/B^4
            var_ratio = np.full_like(num, np.nan, dtype=np.float64)
            var_ratio[valid_ratio] = (
                var_num[valid_ratio] / np.square(nominal_hist[valid_ratio])
                + (np.square(num[valid_ratio]) * nominal_var[valid_ratio]) / np.power(nominal_hist[valid_ratio], 4)
            )
            ratio_err[valid_ratio] = np.sqrt(np.maximum(var_ratio[valid_ratio], 0.0))

            if np.any(valid_ratio):
                ratio_step = ratio.copy()
                ax_ratio.step(
                    edges,
                    np.r_[ratio_step, ratio_step[-1]],
                    where="post",
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.4,
                    label=f"{key}/{nominal_key}",
                    zorder=2,
                )
                ax_ratio.errorbar(
                    centers[valid_ratio],
                    ratio[valid_ratio],
                    yerr=ratio_err[valid_ratio],
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.9,
                    alpha=0.9,
                    capsize=0,
                    zorder=3,
                )
            else:
                ax_ratio.plot([], [], color=color, linestyle=linestyle, label=f"{key}/{nominal_key}")
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.set_xlabel(feature)
        ax_ratio.legend(fontsize=7, ncol=2)

        out_path = out_dir / f"{_safe_stem(feature)}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

        feature_meta = {
            "plot_file": str(out_path),
            "x_range": [float(xlow), float(xhigh)],
            "counts": {key: int(slices[key].shape[0]) for key in masks},
        }
        summary["features"][feature] = feature_meta
        plotted_features += 1

    summary["plotted_feature_count"] = int(plotted_features)
    summary["skipped_feature_count"] = int(skipped_features)
    summary["output_dir"] = str(out_dir)
    pio.write_json(out_dir / "cut_comparison_summary.json", summary)
    return {
        "plotted_feature_count": float(plotted_features),
        "skipped_feature_count": float(skipped_features),
    }


def build_inference_m123_histogram_cache(
    inference_dataset: str,
    output_hist: str,
    score_cut: float,
    score_column: Optional[str] = None,
    score_column_fallback: Optional[str] = None,
    observable_bins: int = 80,
    observable_min: float = 0.0,
    observable_max: float = 500.0,
    pt_bins: int = 20,
    pt_min: float = 0.0,
    pt_max: float = 1000.0,
    eta_bins: int = 10,
    eta_min: float = -5.0,
    eta_max: float = 5.0,
    weight_column: Optional[str] = None,
    batch_size: int = 100_000,
) -> Dict[str, float]:
    """Single-pass histogram production for m123 in (pt, eta, truth-category) phase space."""

    if observable_bins <= 0 or pt_bins <= 0 or eta_bins <= 0:
        raise RuntimeError("Bin counts must be positive")
    if observable_max <= observable_min:
        raise RuntimeError("observable_max must be greater than observable_min")
    if pt_max <= pt_min:
        raise RuntimeError("pt_max must be greater than pt_min")
    if eta_max <= eta_min:
        raise RuntimeError("eta_max must be greater than eta_min")

    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for histogram cache production") from exc

    preferred_score = score_column if score_column is not None else score_column_fallback
    resolved_score_column = _resolve_score_column(path=inference_dataset, preferred=preferred_score)
    available_columns = set(_parquet_column_names(inference_dataset))

    required_columns = [resolved_score_column, "is_truth", "m123", "triplet_pt", "triplet_eta"]
    if weight_column is not None:
        if weight_column not in available_columns:
            raise RuntimeError(
                f"Requested weight column {weight_column} not found in {inference_dataset}. "
                f"Available columns: {sorted(available_columns)}"
            )
        required_columns.append(weight_column)

    dataset = ds.dataset(inference_dataset, format="parquet")
    scanner = dataset.scanner(columns=required_columns, batch_size=int(batch_size))

    m123_edges = np.linspace(observable_min, observable_max, observable_bins + 1, dtype=np.float64)
    pt_edges = np.linspace(pt_min, pt_max, pt_bins + 1, dtype=np.float64)
    eta_edges = np.linspace(eta_min, eta_max, eta_bins + 1, dtype=np.float64)

    hist_shape = (pt_bins, eta_bins, 2, observable_bins)
    sumw = np.zeros(hist_shape, dtype=np.float64)
    sumw2 = np.zeros(hist_shape, dtype=np.float64)

    total_rows = 0
    selected_rows = 0
    accepted_rows = 0
    total_weight = 0.0

    flat_size = int(np.prod(np.asarray(hist_shape, dtype=np.int64)))
    for batch in scanner.to_batches():
        payload = batch.to_pydict()
        n_rows = len(payload["is_truth"])
        if n_rows == 0:
            continue

        total_rows += n_rows
        y = np.asarray(payload["is_truth"], dtype=np.int64)
        score = np.asarray(payload[resolved_score_column], dtype=np.float64)
        m123 = np.asarray(payload["m123"], dtype=np.float64)
        triplet_pt = np.asarray(payload["triplet_pt"], dtype=np.float64)
        triplet_eta = np.asarray(payload["triplet_eta"], dtype=np.float64)
        if weight_column is None:
            w = np.ones(n_rows, dtype=np.float64)
        else:
            w = np.asarray(payload[weight_column], dtype=np.float64)

        # Score-cut and finite-value selection; class axis is fixed to fake=0, true=1.
        selected_mask = (
            np.isfinite(score)
            & np.isfinite(m123)
            & np.isfinite(triplet_pt)
            & np.isfinite(triplet_eta)
            & np.isfinite(w)
            & (y >= 0)
            & (y <= 1)
            & (score > float(score_cut))
        )
        if not np.any(selected_mask):
            continue
        selected_rows += int(np.sum(selected_mask))

        y_sel = y[selected_mask].astype(np.int64, copy=False)
        m123_sel = m123[selected_mask]
        pt_sel = triplet_pt[selected_mask]
        eta_sel = triplet_eta[selected_mask]
        w_sel = w[selected_mask]

        m123_idx = np.searchsorted(m123_edges, m123_sel, side="right") - 1
        pt_idx = np.searchsorted(pt_edges, pt_sel, side="right") - 1
        eta_idx = np.searchsorted(eta_edges, eta_sel, side="right") - 1

        in_range = (
            (m123_idx >= 0)
            & (m123_idx < observable_bins)
            & (pt_idx >= 0)
            & (pt_idx < pt_bins)
            & (eta_idx >= 0)
            & (eta_idx < eta_bins)
        )
        if not np.any(in_range):
            continue

        y_use = y_sel[in_range]
        m123_idx_use = m123_idx[in_range]
        pt_idx_use = pt_idx[in_range]
        eta_idx_use = eta_idx[in_range]
        w_use = w_sel[in_range]

        accepted_rows += int(y_use.shape[0])
        total_weight += float(np.sum(w_use))

        flat_idx = (
            (((pt_idx_use * eta_bins) + eta_idx_use) * 2 + y_use) * observable_bins
            + m123_idx_use
        )
        binc_sumw = np.bincount(flat_idx, weights=w_use, minlength=flat_size)
        binc_sumw2 = np.bincount(flat_idx, weights=np.square(w_use), minlength=flat_size)
        sumw += binc_sumw.reshape(hist_shape)
        sumw2 += binc_sumw2.reshape(hist_shape)

    out_path = Path(output_hist)
    pio.ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        sumw=sumw,
        sumw2=sumw2,
        m123_edges=m123_edges,
        pt_edges=pt_edges,
        eta_edges=eta_edges,
    )

    metadata_path = out_path.with_suffix(".json")
    metadata = {
        "schema_version": pio.SCHEMA_VERSION,
        "kind": "m123_phase_histogram_cache",
        "inference_dataset": inference_dataset,
        "score_column": resolved_score_column,
        "score_cut": float(score_cut),
        "observable": "m123",
        "phase_axes": {
            "triplet_pt": {"bins": int(pt_bins), "min": float(pt_min), "max": float(pt_max)},
            "triplet_eta": {"bins": int(eta_bins), "min": float(eta_min), "max": float(eta_max)},
            "category": {"labels": ["fake", "true"]},
        },
        "observable_axis": {
            "name": "m123",
            "bins": int(observable_bins),
            "min": float(observable_min),
            "max": float(observable_max),
        },
        "weight_column": weight_column,
        "rows_total": int(total_rows),
        "rows_after_score_and_finite_filter": int(selected_rows),
        "rows_filled_in_hist_range": int(accepted_rows),
        "sumw_total": float(total_weight),
        "output_histogram": str(out_path),
    }
    pio.write_json(metadata_path, metadata)
    return {
        "rows_total": float(total_rows),
        "rows_after_score_and_finite_filter": float(selected_rows),
        "rows_filled_in_hist_range": float(accepted_rows),
        "sumw_total": float(total_weight),
    }


def render_m123_truth_fake_from_histogram_cache(
    histogram_cache: str,
    output_png: str,
    normalize: bool = True,
    title: Optional[str] = None,
) -> Dict[str, float]:
    """Render true-vs-fake m123 distributions from persisted histogram cache only."""

    payload = np.load(histogram_cache)
    required = {"sumw", "sumw2", "m123_edges", "pt_edges", "eta_edges"}
    missing = [name for name in required if name not in payload]
    if missing:
        raise RuntimeError(f"Histogram cache is missing required arrays: {missing}")

    sumw = np.asarray(payload["sumw"], dtype=np.float64)
    sumw2 = np.asarray(payload["sumw2"], dtype=np.float64)
    m123_edges = np.asarray(payload["m123_edges"], dtype=np.float64)

    if sumw.ndim != 4 or sumw.shape[2] != 2:
        raise RuntimeError(
            f"Expected histogram shape (pt_bins, eta_bins, 2, m123_bins), got {tuple(sumw.shape)}"
        )
    if sumw.shape != tuple(np.asarray(sumw2).shape):
        raise RuntimeError("sumw and sumw2 shapes do not match")
    if m123_edges.shape[0] != sumw.shape[3] + 1:
        raise RuntimeError("m123_edges length is inconsistent with histogram observable axis")

    # Projection onto observable axis, summing over phase-space axes.
    projected_sumw = np.sum(sumw, axis=(0, 1))
    projected_sumw2 = np.sum(sumw2, axis=(0, 1))
    fake_counts = projected_sumw[0]
    true_counts = projected_sumw[1]
    fake_var = np.maximum(projected_sumw2[0], 0.0)
    true_var = np.maximum(projected_sumw2[1], 0.0)
    fake_err = np.sqrt(fake_var)
    true_err = np.sqrt(true_var)

    fake_total = float(np.sum(fake_counts))
    true_total = float(np.sum(true_counts))

    if normalize:
        fake_plot = fake_counts / fake_total if fake_total > 0.0 else fake_counts
        true_plot = true_counts / true_total if true_total > 0.0 else true_counts
        fake_var_plot = fake_var / (fake_total * fake_total) if fake_total > 0.0 else fake_var
        true_var_plot = true_var / (true_total * true_total) if true_total > 0.0 else true_var
        ylabel = "Normalized entries"
    else:
        fake_plot = fake_counts
        true_plot = true_counts
        fake_var_plot = fake_var
        true_var_plot = true_var
        ylabel = "Sum of weights"

    def _post_steps(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros(1, dtype=np.float64)
        return np.r_[values, values[-1]]

    out_path = Path(output_png)
    pio.ensure_dir(out_path.parent)

    ratio = np.full(true_plot.shape, np.nan, dtype=np.float64)
    ratio_err = np.full(true_plot.shape, np.nan, dtype=np.float64)
    denom_mask = fake_plot > 0.0
    ratio[denom_mask] = true_plot[denom_mask] / fake_plot[denom_mask]

    var_ratio = np.full(true_plot.shape, np.nan, dtype=np.float64)
    var_ratio[denom_mask] = (
        true_var_plot[denom_mask] / np.square(fake_plot[denom_mask])
        + (np.square(true_plot[denom_mask]) * fake_var_plot[denom_mask]) / np.power(fake_plot[denom_mask], 4)
    )
    ratio_err[denom_mask] = np.sqrt(np.maximum(var_ratio[denom_mask], 0.0))

    centers = 0.5 * (m123_edges[:-1] + m123_edges[1:])
    true_err_plot = np.sqrt(np.maximum(true_var_plot, 0.0))
    fake_err_plot = np.sqrt(np.maximum(fake_var_plot, 0.0))

    fig, (ax_top, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )
    ax_top.step(m123_edges, _post_steps(fake_plot), where="post", linewidth=2.0, label=f"Fake (n={int(fake_total)})")
    ax_top.step(m123_edges, _post_steps(true_plot), where="post", linewidth=2.0, label=f"True (n={int(true_total)})")
    ax_top.errorbar(
        centers,
        fake_plot,
        yerr=fake_err_plot,
        fmt="none",
        ecolor="C0",
        elinewidth=0.9,
        alpha=0.9,
        capsize=0,
    )
    ax_top.errorbar(
        centers,
        true_plot,
        yerr=true_err_plot,
        fmt="none",
        ecolor="C1",
        elinewidth=0.9,
        alpha=0.9,
        capsize=0,
    )
    ax_top.set_ylabel(ylabel)
    ax_top.set_title(title if title is not None else "m123 distribution: true vs fake (score-cut selected)")
    ax_top.legend()

    valid = np.isfinite(ratio) & np.isfinite(ratio_err)
    denom_rel_unc = np.zeros_like(fake_plot, dtype=np.float64)
    denom_rel_unc[denom_mask] = fake_err_plot[denom_mask] / fake_plot[denom_mask]
    band_low = np.full_like(fake_plot, np.nan, dtype=np.float64)
    band_high = np.full_like(fake_plot, np.nan, dtype=np.float64)
    band_low[denom_mask] = 1.0 - denom_rel_unc[denom_mask]
    band_high[denom_mask] = 1.0 + denom_rel_unc[denom_mask]
    ax_ratio.fill_between(
        centers,
        band_low,
        band_high,
        step="mid",
        color="gray",
        alpha=0.20,
        label="fake stat. unc.",
        zorder=1,
    )
    if np.any(valid):
        ratio_step = ratio.copy()
        ax_ratio.step(m123_edges, _post_steps(ratio_step), where="post", linewidth=1.4, color="C2", label="true/fake", zorder=2)
        ax_ratio.errorbar(
            centers[valid],
            ratio[valid],
            yerr=ratio_err[valid],
            fmt="none",
            ecolor="C2",
            elinewidth=0.9,
            alpha=0.9,
            capsize=0,
            zorder=3,
        )
    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax_ratio.set_xlabel("Triplet invariant mass m123 [GeV]")
    ax_ratio.set_ylabel("True/Fake")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.legend(fontsize=8)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    metrics = {
        "histogram_cache": histogram_cache,
        "output_plot": str(out_path),
        "normalize": bool(normalize),
        "fake_sumw": fake_total,
        "true_sumw": true_total,
        "fake_uncertainty_sum_quadrature": float(np.sqrt(np.sum(np.square(fake_err)))),
        "true_uncertainty_sum_quadrature": float(np.sqrt(np.sum(np.square(true_err)))),
        "ratio_bins_finite": int(np.sum(np.isfinite(ratio))),
    }
    pio.write_json(out_path.with_suffix(".json"), metrics)
    return metrics


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

    bins = np.linspace(0.0, 1.0, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig_sb, axes_sb = plt.subplots(
        2,
        2,
        figsize=(12, 7),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05, "wspace": 0.12},
    )
    for col, backend in enumerate(("xgb", "tabpfn")):
        payload = resolved[backend]
        score = np.asarray(payload["score"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.int64)
        sig = score[y == 1]
        bkg = score[y == 0]

        sig_hist, sig_var = _normalized_hist_and_var(sig, bins)
        bkg_hist, bkg_var = _normalized_hist_and_var(bkg, bins)
        sig_err = np.sqrt(np.maximum(sig_var, 0.0))
        bkg_err = np.sqrt(np.maximum(bkg_var, 0.0))

        ax_top = axes_sb[0, col]
        ax_ratio = axes_sb[1, col]

        ax_top.step(bins, np.r_[sig_hist, sig_hist[-1]], where="post", linewidth=1.8, linestyle="-", color="tab:blue", label="Signal")
        ax_top.step(bins, np.r_[bkg_hist, bkg_hist[-1]], where="post", linewidth=1.8, linestyle="--", color="tab:red", label="Background")
        ax_top.errorbar(centers, sig_hist, yerr=sig_err, fmt="none", ecolor="tab:blue", elinewidth=0.9, alpha=0.9, capsize=0)
        ax_top.errorbar(centers, bkg_hist, yerr=bkg_err, fmt="none", ecolor="tab:red", elinewidth=0.9, alpha=0.9, capsize=0)
        ax_top.set_title(f"{_model_label(backend)} score")
        ax_top.set_ylabel("Normalized entries")
        ax_top.legend(fontsize=8)

        denom_mask = sig_hist > 0.0
        sig_rel_unc = np.zeros_like(sig_hist, dtype=np.float64)
        sig_rel_unc[denom_mask] = sig_err[denom_mask] / sig_hist[denom_mask]
        band_low = np.full_like(sig_hist, np.nan, dtype=np.float64)
        band_high = np.full_like(sig_hist, np.nan, dtype=np.float64)
        band_low[denom_mask] = 1.0 - sig_rel_unc[denom_mask]
        band_high[denom_mask] = 1.0 + sig_rel_unc[denom_mask]
        ax_ratio.fill_between(centers, band_low, band_high, step="mid", color="gray", alpha=0.20, label="signal stat. unc.", zorder=1)

        ratio = np.full_like(bkg_hist, np.nan, dtype=np.float64)
        ratio_err = np.full_like(bkg_hist, np.nan, dtype=np.float64)
        ratio[denom_mask] = bkg_hist[denom_mask] / sig_hist[denom_mask]
        var_ratio = np.full_like(bkg_hist, np.nan, dtype=np.float64)
        var_ratio[denom_mask] = (
            bkg_var[denom_mask] / np.square(sig_hist[denom_mask])
            + (np.square(bkg_hist[denom_mask]) * sig_var[denom_mask]) / np.power(sig_hist[denom_mask], 4)
        )
        ratio_err[denom_mask] = np.sqrt(np.maximum(var_ratio[denom_mask], 0.0))
        valid = np.isfinite(ratio) & np.isfinite(ratio_err)
        if np.any(valid):
            ratio_step = ratio.copy()
            ax_ratio.step(bins, np.r_[ratio_step, ratio_step[-1]], where="post", linewidth=1.4, linestyle="--", color="tab:red", label="background/signal", zorder=2)
            ax_ratio.errorbar(centers[valid], ratio[valid], yerr=ratio_err[valid], fmt="none", ecolor="tab:red", elinewidth=0.9, alpha=0.9, capsize=0, zorder=3)
        ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.set_xlabel(f"{_model_label(backend)} score")
        ax_ratio.legend(fontsize=7)

    fig_sb.savefig(out_dir / "score_distribution_signal_background_comparison.png", dpi=160)
    plt.close(fig_sb)

    model_series = []
    for backend in ("xgb", "tabpfn"):
        payload = resolved[backend]
        score = np.asarray(payload["score"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.int64)
        sig = score[y == 1]
        bkg = score[y == 0]
        sig_hist, sig_var = _normalized_hist_and_var(sig, bins)
        bkg_hist, bkg_var = _normalized_hist_and_var(bkg, bins)
        model_series.extend(
            [
                (f"{backend}_signal", f"{_model_label(backend)} signal", "tab:blue" if backend == "xgb" else "tab:green", "-", sig_hist, sig_var),
                (f"{backend}_background", f"{_model_label(backend)} background", "tab:red" if backend == "xgb" else "tab:orange", "--", bkg_hist, bkg_var),
            ]
        )

    nominal_key = model_series[0][0]
    nominal_hist = model_series[0][4]
    nominal_var = model_series[0][5]
    nominal_err = np.sqrt(np.maximum(nominal_var, 0.0))

    fig_m, (ax_top_m, ax_ratio_m) = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )

    for key, label, color, linestyle, hist_vals, var_vals in model_series:
        err_vals = np.sqrt(np.maximum(var_vals, 0.0))
        ax_top_m.step(bins, np.r_[hist_vals, hist_vals[-1]], where="post", linewidth=1.8, linestyle=linestyle, color=color, label=label)
        ax_top_m.errorbar(centers, hist_vals, yerr=err_vals, fmt="none", ecolor=color, elinewidth=0.9, alpha=0.9, capsize=0)
    ax_top_m.set_ylabel("Normalized entries")
    ax_top_m.set_title("Score Distribution Comparison (test)")
    ax_top_m.legend(fontsize=8, ncol=2)

    denom_mask = nominal_hist > 0.0
    nominal_rel_unc = np.zeros_like(nominal_hist, dtype=np.float64)
    nominal_rel_unc[denom_mask] = nominal_err[denom_mask] / nominal_hist[denom_mask]
    band_low = np.full_like(nominal_hist, np.nan, dtype=np.float64)
    band_high = np.full_like(nominal_hist, np.nan, dtype=np.float64)
    band_low[denom_mask] = 1.0 - nominal_rel_unc[denom_mask]
    band_high[denom_mask] = 1.0 + nominal_rel_unc[denom_mask]
    ax_ratio_m.fill_between(centers, band_low, band_high, step="mid", color="gray", alpha=0.20, label=f"{nominal_key} stat. unc.", zorder=1)

    for key, label, color, linestyle, hist_vals, var_vals in model_series[1:]:
        ratio = np.full_like(hist_vals, np.nan, dtype=np.float64)
        ratio_err = np.full_like(hist_vals, np.nan, dtype=np.float64)
        ratio[denom_mask] = hist_vals[denom_mask] / nominal_hist[denom_mask]
        var_ratio = np.full_like(hist_vals, np.nan, dtype=np.float64)
        var_ratio[denom_mask] = (
            var_vals[denom_mask] / np.square(nominal_hist[denom_mask])
            + (np.square(hist_vals[denom_mask]) * nominal_var[denom_mask]) / np.power(nominal_hist[denom_mask], 4)
        )
        ratio_err[denom_mask] = np.sqrt(np.maximum(var_ratio[denom_mask], 0.0))

        valid = np.isfinite(ratio) & np.isfinite(ratio_err)
        if np.any(valid):
            ratio_step = ratio.copy()
            ax_ratio_m.step(bins, np.r_[ratio_step, ratio_step[-1]], where="post", linewidth=1.4, linestyle=linestyle, color=color, label=f"{key}/{nominal_key}", zorder=2)
            ax_ratio_m.errorbar(centers[valid], ratio[valid], yerr=ratio_err[valid], fmt="none", ecolor=color, elinewidth=0.9, alpha=0.9, capsize=0, zorder=3)
        else:
            ax_ratio_m.plot([], [], color=color, linestyle=linestyle, label=f"{key}/{nominal_key}")
    ax_ratio_m.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax_ratio_m.set_ylim(0.5, 1.5)
    ax_ratio_m.set_ylabel("Ratio")
    ax_ratio_m.set_xlabel("Model score")
    ax_ratio_m.legend(fontsize=7, ncol=2)

    fig_m.savefig(out_dir / "score_distribution_model_overlay.png", dpi=160)
    plt.close(fig_m)

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


def run_build_m123_hist_cache(args: argparse.Namespace) -> None:
    build_inference_m123_histogram_cache(
        inference_dataset=args.inference,
        output_hist=args.output_hist,
        score_cut=args.score_cut,
        score_column=args.score_column,
        score_column_fallback=args.score_column_fallback,
        observable_bins=args.observable_bins,
        observable_min=args.observable_min,
        observable_max=args.observable_max,
        pt_bins=args.pt_bins,
        pt_min=args.pt_min,
        pt_max=args.pt_max,
        eta_bins=args.eta_bins,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        weight_column=args.weight_column,
        batch_size=args.batch_size,
    )


def run_plot_m123_hist_cache(args: argparse.Namespace) -> None:
    render_m123_truth_fake_from_histogram_cache(
        histogram_cache=args.histogram_cache,
        output_png=args.output_png,
        normalize=not args.no_normalize,
        title=args.title,
    )


def run_plot_cut_comparison(args: argparse.Namespace) -> None:
    generate_cut_comparison_plots(
        inference_dataset=args.inference,
        output_root=args.output_root,
        score_cut=args.score_cut,
        score_column=args.score_column,
        columns=args.columns,
        bins=args.bins,
        nominal=args.nominal,
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

    build_m123_hist_cache = subparsers.add_parser(
        "build_m123_hist_cache",
        help="Build persistent m123 histogram cache in (pt, eta, truth/fake) phase space from inference parquet",
    )
    build_m123_hist_cache.add_argument("--inference", required=True, help="Inference parquet path")
    build_m123_hist_cache.add_argument(
        "--output-hist",
        required=True,
        help="Output histogram cache .npz path",
    )
    build_m123_hist_cache.add_argument(
        "--score-cut",
        type=float,
        default=0.5,
        help="Score threshold (strictly score > threshold)",
    )
    build_m123_hist_cache.add_argument(
        "--score-column",
        default=None,
        help="Optional explicit score column name (auto-resolved if omitted)",
    )
    build_m123_hist_cache.add_argument(
        "--score-column-fallback",
        default=None,
        help="Optional fallback score column checked before built-in defaults",
    )
    build_m123_hist_cache.add_argument(
        "--observable-bins",
        type=int,
        default=80,
        help="Number of m123 histogram bins",
    )
    build_m123_hist_cache.add_argument(
        "--observable-min",
        type=float,
        default=0.0,
        help="Minimum m123 [GeV] for histogram axis",
    )
    build_m123_hist_cache.add_argument(
        "--observable-max",
        type=float,
        default=500.0,
        help="Maximum m123 [GeV] for histogram axis",
    )
    build_m123_hist_cache.add_argument("--pt-bins", type=int, default=20, help="Number of triplet_pt phase bins")
    build_m123_hist_cache.add_argument("--pt-min", type=float, default=0.0, help="Minimum triplet_pt [GeV]")
    build_m123_hist_cache.add_argument("--pt-max", type=float, default=1000.0, help="Maximum triplet_pt [GeV]")
    build_m123_hist_cache.add_argument("--eta-bins", type=int, default=10, help="Number of triplet_eta phase bins")
    build_m123_hist_cache.add_argument("--eta-min", type=float, default=-5.0, help="Minimum triplet_eta")
    build_m123_hist_cache.add_argument("--eta-max", type=float, default=5.0, help="Maximum triplet_eta")
    build_m123_hist_cache.add_argument(
        "--weight-column",
        default=None,
        help="Optional per-row weight column; defaults to unit weights",
    )
    build_m123_hist_cache.add_argument("--batch-size", type=int, default=100_000, help="Parquet scan batch size")
    build_m123_hist_cache.set_defaults(func=run_build_m123_hist_cache)

    plot_m123_hist_cache = subparsers.add_parser(
        "plot_m123_hist_cache",
        help="Render true-vs-fake m123 plot from cached histogram (no parquet/event loop)",
    )
    plot_m123_hist_cache.add_argument("--histogram-cache", required=True, help="Input histogram cache .npz")
    plot_m123_hist_cache.add_argument("--output-png", required=True, help="Output plot path")
    plot_m123_hist_cache.add_argument("--no-normalize", action="store_true", help="Disable shape normalization")
    plot_m123_hist_cache.add_argument("--title", default=None, help="Optional custom plot title")
    plot_m123_hist_cache.set_defaults(func=run_plot_m123_hist_cache)

    plot_cut_comparison = subparsers.add_parser(
        "plot_cut_comparison",
        help="Invariant-compliant 4-way overlays (true/fake x pass/fail cut) with ratio panels",
    )
    plot_cut_comparison.add_argument("--inference", required=True, help="Inference parquet path")
    plot_cut_comparison.add_argument("--output-root", default="plots", help="Root directory for plot outputs")
    plot_cut_comparison.add_argument("--score-cut", type=float, default=0.5, help="Score threshold for pass/fail split")
    plot_cut_comparison.add_argument(
        "--score-column",
        default=None,
        help="Optional explicit score column name (auto-resolved when omitted)",
    )
    plot_cut_comparison.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Optional list of columns to plot (default: all numeric columns except score/is_truth/event_id)",
    )
    plot_cut_comparison.add_argument("--bins", type=int, default=20, help="Histogram bin count (default: 20)")
    plot_cut_comparison.add_argument(
        "--nominal",
        choices=["true_pass", "true_fail", "fake_pass", "fake_fail"],
        default="true_pass",
        help="Nominal distribution used in ratio denominator",
    )
    plot_cut_comparison.set_defaults(func=run_plot_cut_comparison)
