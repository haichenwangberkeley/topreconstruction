#!/usr/bin/env python3
"""Stage 5: select reconstructed top candidates from scored triplet inference parquet."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import progress as prog
from . import triplet_io as pio


STRATEGIES = ("greedy_disjoint", "top1", "topk", "threshold", "best_pair_avg_disjoint")
EVENT_TOP_SLOTS = 4
PAIR_STRATEGY_MIN_JETS = 6


@dataclass(frozen=True)
class TripletCandidate:
    i: int
    j: int
    k: int
    score: float
    triplet_pt: float
    triplet_eta: float
    triplet_phi: float
    triplet_mass: float


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for select_triplets stage") from exc
    return pa, ds, pq


def _parquet_column_names(path: str) -> Sequence[str]:
    _, _, pq = _require_pyarrow()
    return list(pq.read_schema(path).names)


def _resolve_score_column(path: str, preferred: Optional[str]) -> str:
    names = set(_parquet_column_names(path))
    candidates: List[str] = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(["score_tabpfn", "score_xgb", "score"])
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise RuntimeError(f"Could not find score column in {path}; available columns: {sorted(names)}")


def _fourvec_from_pt_eta_phi_m(pt: float, eta: float, phi: float, mass: float) -> Tuple[float, float, float, float]:
    px = float(pt) * math.cos(float(phi))
    py = float(pt) * math.sin(float(phi))
    pz = float(pt) * math.sinh(float(eta))
    p2 = px * px + py * py + pz * pz
    m2 = float(mass) * float(mass)
    energy = math.sqrt(max(m2 + p2, 0.0))
    return px, py, pz, energy


def _pair_invariant_mass(t1: TripletCandidate, t2: TripletCandidate) -> float:
    px1, py1, pz1, e1 = _fourvec_from_pt_eta_phi_m(t1.triplet_pt, t1.triplet_eta, t1.triplet_phi, t1.triplet_mass)
    px2, py2, pz2, e2 = _fourvec_from_pt_eta_phi_m(t2.triplet_pt, t2.triplet_eta, t2.triplet_phi, t2.triplet_mass)
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    energy = e1 + e2
    m2 = energy * energy - (px * px + py * py + pz * pz)
    return float(math.sqrt(max(m2, 0.0)))


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for select_triplets plotting; use --skip-plots to disable plotting") from exc
    return plt


@dataclass
class _RunningMoments:
    count: int = 0
    sum_values: float = 0.0
    sum_squares: float = 0.0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.count += int(values.size)
        self.sum_values += float(np.sum(values))
        self.sum_squares += float(np.sum(np.square(values)))

    def mean_std(self) -> Optional[Tuple[float, float]]:
        if self.count <= 0:
            return None
        mean = self.sum_values / float(self.count)
        variance = max((self.sum_squares / float(self.count)) - (mean * mean), 0.0)
        return float(mean), float(math.sqrt(variance))


def _clean_values(values: Sequence[Any], drop_dummy: bool, dummy_value: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr)
    if drop_dummy:
        mask &= arr != float(dummy_value)
    return arr[mask]


def _safe_filename_token(value: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in str(value))
    token = token.strip("._")
    return token or "strategy"


def _auto_hist_range(moments: Sequence[_RunningMoments]) -> Tuple[float, float]:
    lowers: List[float] = []
    uppers: List[float] = []
    for moment in moments:
        mean_std = moment.mean_std()
        if mean_std is None:
            continue
        mean, std = mean_std
        width = 3.0 * float(std)
        if not np.isfinite(width) or width <= 0.0:
            width = max(abs(float(mean)) * 0.05, 0.5)
        lowers.append(float(mean - width))
        uppers.append(float(mean + width))

    if not lowers:
        return 0.0, 1.0

    x_min = min(lowers)
    x_max = max(uppers)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        center = 0.5 * (x_min + x_max) if np.isfinite(x_min) and np.isfinite(x_max) else 0.0
        width = max(abs(center) * 0.05, 0.5)
        return float(center - width), float(center + width)
    return float(x_min), float(x_max)


def _post_steps(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(1, dtype=np.float64)
    return np.r_[values, values[-1]]


def _compute_histograms_for_columns(
    dataset,
    columns: Sequence[str],
    drop_dummy_columns: Sequence[str],
    batch_size: int,
    dummy_value: float,
    edges: np.ndarray,
) -> Dict[str, np.ndarray]:
    drop_dummy = set(drop_dummy_columns)
    counts: Dict[str, np.ndarray] = {name: np.zeros(edges.shape[0] - 1, dtype=np.float64) for name in columns}
    scanner = dataset.scanner(columns=list(columns), batch_size=batch_size)
    for batch in scanner.to_batches():
        payload = batch.to_pydict()
        for name in columns:
            clean = _clean_values(payload[name], drop_dummy=name in drop_dummy, dummy_value=dummy_value)
            if clean.size == 0:
                continue
            hist, _ = np.histogram(clean, bins=edges)
            counts[name] += hist.astype(np.float64)
    return counts


def _compute_moments_for_columns(
    dataset,
    columns: Sequence[str],
    drop_dummy_columns: Sequence[str],
    batch_size: int,
    dummy_value: float,
) -> Dict[str, _RunningMoments]:
    drop_dummy = set(drop_dummy_columns)
    moments: Dict[str, _RunningMoments] = {name: _RunningMoments() for name in columns}
    scanner = dataset.scanner(columns=list(columns), batch_size=batch_size)
    for batch in scanner.to_batches():
        payload = batch.to_pydict()
        for name in columns:
            clean = _clean_values(payload[name], drop_dummy=name in drop_dummy, dummy_value=dummy_value)
            moments[name].update(clean)
    return moments


def _plot_single_hist(
    plt,
    out_path: Path,
    edges: np.ndarray,
    counts: np.ndarray,
    variances: np.ndarray,
    *,
    title: str,
    xlabel: str,
    legend_label: str,
) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    errors = np.sqrt(np.maximum(variances, 0.0))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.step(edges, _post_steps(counts), where="post", linewidth=2.0, color="C0", label=legend_label)
    ax.errorbar(
        centers,
        counts,
        yerr=errors,
        fmt="none",
        ecolor="C0",
        elinewidth=0.9,
        alpha=0.9,
        capsize=0,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Entries")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_overlay_with_ratio(
    plt,
    out_path: Path,
    edges: np.ndarray,
    ordered_series: Sequence[Tuple[str, np.ndarray, np.ndarray, str, str]],
    *,
    title: str,
    xlabel: str,
) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, (ax_top, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )

    for label, counts, variances, color, linestyle in ordered_series:
        errors = np.sqrt(np.maximum(variances, 0.0))
        ax_top.step(
            edges,
            _post_steps(counts),
            where="post",
            linewidth=1.9,
            color=color,
            linestyle=linestyle,
            label=label,
        )
        ax_top.errorbar(
            centers,
            counts,
            yerr=errors,
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            alpha=0.9,
            capsize=0,
        )
    ax_top.set_ylabel("Entries")
    ax_top.set_title(title)
    ax_top.legend(fontsize=9)

    nominal_label, nominal_counts, nominal_var, _, _ = ordered_series[0]
    nominal_err = np.sqrt(np.maximum(nominal_var, 0.0))
    denom_mask = nominal_counts > 0.0

    nominal_rel_unc = np.zeros_like(nominal_counts, dtype=np.float64)
    nominal_rel_unc[denom_mask] = nominal_err[denom_mask] / nominal_counts[denom_mask]
    band_low = np.full_like(nominal_counts, np.nan, dtype=np.float64)
    band_high = np.full_like(nominal_counts, np.nan, dtype=np.float64)
    band_low[denom_mask] = 1.0 - nominal_rel_unc[denom_mask]
    band_high[denom_mask] = 1.0 + nominal_rel_unc[denom_mask]
    ax_ratio.fill_between(
        centers,
        band_low,
        band_high,
        step="mid",
        color="gray",
        alpha=0.20,
        label=f"{nominal_label} stat. unc.",
        zorder=1,
    )

    for idx, (label, counts, variances, color, linestyle) in enumerate(ordered_series):
        if idx == 0:
            continue
        ratio = np.full_like(counts, np.nan, dtype=np.float64)
        ratio_err = np.full_like(counts, np.nan, dtype=np.float64)
        ratio[denom_mask] = counts[denom_mask] / nominal_counts[denom_mask]
        ratio_variance = np.full_like(counts, np.nan, dtype=np.float64)
        ratio_variance[denom_mask] = (
            variances[denom_mask] / np.square(nominal_counts[denom_mask])
            + (np.square(counts[denom_mask]) * nominal_var[denom_mask]) / np.power(nominal_counts[denom_mask], 4)
        )
        ratio_err[denom_mask] = np.sqrt(np.maximum(ratio_variance[denom_mask], 0.0))
        valid = np.isfinite(ratio) & np.isfinite(ratio_err)

        ax_ratio.step(
            edges,
            _post_steps(ratio),
            where="post",
            linewidth=1.4,
            color=color,
            linestyle=linestyle,
            label=f"{label}/{nominal_label}",
            zorder=2,
        )
        if np.any(valid):
            ax_ratio.errorbar(
                centers[valid],
                ratio[valid],
                yerr=ratio_err[valid],
                fmt="none",
                ecolor=color,
                elinewidth=0.9,
                alpha=0.9,
                capsize=0,
                zorder=3,
            )

    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.legend(fontsize=8)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _generate_selection_plots(
    event_selection_file: Path,
    output_dir: Path,
    batch_size: int,
    bins: int,
    dummy_value: float,
    strategy: str,
) -> Dict[str, Any]:
    _, ds, _ = _require_pyarrow()
    plt = _require_matplotlib()

    dataset = ds.dataset(str(event_selection_file), format="parquet")
    plot_dir = pio.ensure_dir(output_dir)
    strategy_tag = _safe_filename_token(strategy)
    plot_specs = [
        {
            "stem": "n_top_selected",
            "title": "Selected top candidates per event",
            "xlabel": "n_top_selected",
            "series": [("n_top_selected", "n_top_selected", "C0", "-")],
            "drop_dummy_columns": [],
            "overlay": False,
        },
        {
            "stem": "m_top1_top2",
            "title": "Invariant mass of leading top pair",
            "xlabel": "m(top1, top2) [GeV]",
            "series": [("m(top1,top2)", "m_top1_top2", "C0", "-")],
            "drop_dummy_columns": ["m_top1_top2"],
            "overlay": False,
        },
        {
            "stem": "top_pt_by_rank",
            "title": "Top-candidate pT by rank",
            "xlabel": "pT [GeV]",
            "series": [
                ("top1", "top1_pt", "C0", "-"),
                ("top2", "top2_pt", "C1", "--"),
                ("top3", "top3_pt", "C2", ":"),
                ("top4", "top4_pt", "C3", "-."),
            ],
            "drop_dummy_columns": ["top1_pt", "top2_pt", "top3_pt", "top4_pt"],
            "overlay": True,
        },
        {
            "stem": "top_eta_by_rank",
            "title": "Top-candidate eta by rank",
            "xlabel": "eta",
            "series": [
                ("top1", "top1_eta", "C0", "-"),
                ("top2", "top2_eta", "C1", "--"),
                ("top3", "top3_eta", "C2", ":"),
                ("top4", "top4_eta", "C3", "-."),
            ],
            "drop_dummy_columns": ["top1_eta", "top2_eta", "top3_eta", "top4_eta"],
            "overlay": True,
        },
        {
            "stem": "top_phi_by_rank",
            "title": "Top-candidate phi by rank",
            "xlabel": "phi",
            "series": [
                ("top1", "top1_phi", "C0", "-"),
                ("top2", "top2_phi", "C1", "--"),
                ("top3", "top3_phi", "C2", ":"),
                ("top4", "top4_phi", "C3", "-."),
            ],
            "drop_dummy_columns": ["top1_phi", "top2_phi", "top3_phi", "top4_phi"],
            "overlay": True,
        },
        {
            "stem": "top_mass_by_rank",
            "title": "Top-candidate mass by rank",
            "xlabel": "mass [GeV]",
            "series": [
                ("top1", "top1_mass", "C0", "-"),
                ("top2", "top2_mass", "C1", "--"),
                ("top3", "top3_mass", "C2", ":"),
                ("top4", "top4_mass", "C3", "-."),
            ],
            "drop_dummy_columns": ["top1_mass", "top2_mass", "top3_mass", "top4_mass"],
            "overlay": True,
        },
    ]

    summary: Dict[str, Any] = {
        "schema_version": pio.SCHEMA_VERSION,
        "plots": [],
        "plot_dir": str(plot_dir),
        "bins": int(bins),
        "dummy_value": float(dummy_value),
        "strategy": str(strategy),
    }

    for spec in plot_specs:
        series = spec["series"]
        columns = [item[1] for item in series]
        moments = _compute_moments_for_columns(
            dataset=dataset,
            columns=columns,
            drop_dummy_columns=spec["drop_dummy_columns"],
            batch_size=batch_size,
            dummy_value=dummy_value,
        )
        x_min, x_max = _auto_hist_range([moments[name] for name in columns])
        edges = np.linspace(float(x_min), float(x_max), int(bins) + 1, dtype=np.float64)

        counts_by_column = _compute_histograms_for_columns(
            dataset=dataset,
            columns=columns,
            drop_dummy_columns=spec["drop_dummy_columns"],
            batch_size=batch_size,
            dummy_value=dummy_value,
            edges=edges,
        )

        out_path = plot_dir / f"{spec['stem']}_{strategy_tag}.png"
        title_with_strategy = f"{spec['title']} [strategy: {strategy}]"
        if spec["overlay"]:
            ordered_series = [
                (
                    label,
                    counts_by_column[column],
                    counts_by_column[column].copy(),
                    color,
                    linestyle,
                )
                for (label, column, color, linestyle) in series
            ]
            ordered_series_nonempty = [row for row in ordered_series if float(np.sum(row[1])) > 0.0]
            if len(ordered_series_nonempty) >= 2:
                _plot_overlay_with_ratio(
                    plt=plt,
                    out_path=out_path,
                    edges=edges,
                    ordered_series=ordered_series_nonempty,
                    title=title_with_strategy,
                    xlabel=spec["xlabel"],
                )
            elif len(ordered_series_nonempty) == 1:
                label, counts, variances, _, _ = ordered_series_nonempty[0]
                _plot_single_hist(
                    plt=plt,
                    out_path=out_path,
                    edges=edges,
                    counts=counts,
                    variances=variances,
                    title=title_with_strategy,
                    xlabel=spec["xlabel"],
                    legend_label=f"{label} (n={int(np.sum(counts))})",
                )
            else:
                # Keep a deterministic empty artifact if all series are dummy-filtered.
                label, counts, variances, _, _ = ordered_series[0]
                _plot_single_hist(
                    plt=plt,
                    out_path=out_path,
                    edges=edges,
                    counts=counts,
                    variances=variances,
                    title=title_with_strategy,
                    xlabel=spec["xlabel"],
                    legend_label=f"{label} (n={int(np.sum(counts))})",
                )
        else:
            label, column, _, _ = series[0]
            _plot_single_hist(
                plt=plt,
                out_path=out_path,
                edges=edges,
                counts=counts_by_column[column],
                variances=counts_by_column[column].copy(),
                title=title_with_strategy,
                xlabel=spec["xlabel"],
                legend_label=f"{label} (n={int(np.sum(counts_by_column[column]))})",
            )

        summary["plots"].append(
            {
                "name": spec["stem"],
                "output": str(out_path),
                "overlay": bool(spec["overlay"]),
                "x_min": float(x_min),
                "x_max": float(x_max),
                "series": [
                    {
                        "label": label,
                        "column": column,
                        "entries": int(np.sum(counts_by_column[column])),
                    }
                    for (label, column, _, _) in series
                ],
            }
        )

    metrics_path = plot_dir / f"selection_plot_metrics_{strategy_tag}.json"
    summary["metrics_file"] = str(metrics_path)
    pio.write_json(metrics_path, summary)
    return summary


def _sorted_candidates(triplets: Iterable[TripletCandidate], min_score: float) -> List[TripletCandidate]:
    accepted = [t for t in triplets if np.isfinite(t.score) and float(t.score) >= float(min_score)]
    accepted.sort(key=lambda t: (-float(t.score), int(t.i), int(t.j), int(t.k)))
    return accepted


def _select_top1(candidates: Sequence[TripletCandidate], max_top_per_event: int) -> List[TripletCandidate]:
    if max_top_per_event <= 0 or len(candidates) == 0:
        return []
    return [candidates[0]]


def _select_topk(candidates: Sequence[TripletCandidate], max_top_per_event: int, top_k: int) -> List[TripletCandidate]:
    if max_top_per_event <= 0 or top_k <= 0 or len(candidates) == 0:
        return []
    return list(candidates[: min(max_top_per_event, top_k)])


def _select_threshold(candidates: Sequence[TripletCandidate], max_top_per_event: int) -> List[TripletCandidate]:
    if max_top_per_event <= 0 or len(candidates) == 0:
        return []
    return list(candidates[:max_top_per_event])


def _select_greedy_disjoint(candidates: Sequence[TripletCandidate], max_top_per_event: int) -> List[TripletCandidate]:
    if max_top_per_event <= 0 or len(candidates) == 0:
        return []

    available_jets = set()
    for t in candidates:
        available_jets.update((int(t.i), int(t.j), int(t.k)))

    selected: List[TripletCandidate] = []
    for cand in candidates:
        if len(selected) >= max_top_per_event:
            break
        if len(available_jets) < 3:
            break
        triplet_jets = (int(cand.i), int(cand.j), int(cand.k))
        if all(j in available_jets for j in triplet_jets):
            selected.append(cand)
            for j in triplet_jets:
                available_jets.discard(j)

    return selected


def _event_n_jets(triplets: Sequence[TripletCandidate]) -> int:
    jets = set()
    for t in triplets:
        jets.add(int(t.i))
        jets.add(int(t.j))
        jets.add(int(t.k))
    return int(len(jets))


def _select_best_pair_avg_disjoint(
    candidates: Sequence[TripletCandidate],
    *,
    n_jets_in_event: int,
    max_top_per_event: int,
) -> List[TripletCandidate]:
    if max_top_per_event < 2:
        return []
    if n_jets_in_event < PAIR_STRATEGY_MIN_JETS:
        return []
    if len(candidates) < 2:
        return []

    best_pair: Optional[Tuple[TripletCandidate, TripletCandidate]] = None
    best_key: Optional[Tuple[float, float, float, float, int, int, int, int, int, int]] = None

    for idx_a in range(len(candidates) - 1):
        a = candidates[idx_a]
        jets_a = {int(a.i), int(a.j), int(a.k)}
        for idx_b in range(idx_a + 1, len(candidates)):
            b = candidates[idx_b]
            jets_b = {int(b.i), int(b.j), int(b.k)}
            if not jets_a.isdisjoint(jets_b):
                continue

            avg_score = 0.5 * (float(a.score) + float(b.score))
            key = (
                float(avg_score),
                float(a.score + b.score),
                float(a.score),
                float(b.score),
                -int(a.i),
                -int(a.j),
                -int(a.k),
                -int(b.i),
                -int(b.j),
                -int(b.k),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_pair = (a, b)

    if best_pair is None:
        return []
    return [best_pair[0], best_pair[1]]


def _apply_strategy(
    triplets: Sequence[TripletCandidate],
    strategy: str,
    min_score: float,
    max_top_per_event: int,
    top_k: int,
    n_jets_in_event: int,
) -> List[TripletCandidate]:
    candidates = _sorted_candidates(triplets, min_score=min_score)
    if strategy == "greedy_disjoint":
        return _select_greedy_disjoint(candidates, max_top_per_event=max_top_per_event)
    if strategy == "top1":
        return _select_top1(candidates, max_top_per_event=max_top_per_event)
    if strategy == "topk":
        return _select_topk(candidates, max_top_per_event=max_top_per_event, top_k=top_k)
    if strategy == "threshold":
        return _select_threshold(candidates, max_top_per_event=max_top_per_event)
    if strategy == "best_pair_avg_disjoint":
        return _select_best_pair_avg_disjoint(
            candidates,
            n_jets_in_event=n_jets_in_event,
            max_top_per_event=max_top_per_event,
        )
    raise RuntimeError(f"Unknown strategy: {strategy}")


def run(args: argparse.Namespace) -> None:
    if args.strategy not in STRATEGIES:
        raise ValueError(f"--strategy must be one of: {STRATEGIES}")
    if args.max_top_per_event <= 0:
        raise ValueError("--max-top-per-event must be > 0")
    if args.max_top_per_event > EVENT_TOP_SLOTS:
        raise ValueError(f"--max-top-per-event must be <= {EVENT_TOP_SLOTS}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.strategy == "best_pair_avg_disjoint" and args.max_top_per_event < 2:
        raise ValueError("--max-top-per-event must be >= 2 for --strategy best_pair_avg_disjoint")
    if args.plot_bins <= 0:
        raise ValueError("--plot-bins must be > 0")

    pa, ds, _ = _require_pyarrow()
    show_progress = prog.should_show_progress(args.no_progress)
    score_column = _resolve_score_column(path=args.inference, preferred=args.score_column)

    required_columns = ["event_id", "i", "j", "k", "triplet_pt", "triplet_eta", "triplet_phi", "m123", score_column]
    output_dir = pio.ensure_dir(args.output_dir)
    selected_path = output_dir / "selected_triplets.parquet"
    event_path = output_dir / "event_selection.parquet"

    selected_schema = pa.schema(
        [
            pa.field("event_id", pa.int64()),
            pa.field("selected_rank", pa.int32()),
            pa.field("i", pa.int32()),
            pa.field("j", pa.int32()),
            pa.field("k", pa.int32()),
            pa.field("score", pa.float32()),
            pa.field("triplet_pt", pa.float32()),
            pa.field("triplet_eta", pa.float32()),
            pa.field("triplet_phi", pa.float32()),
            pa.field("triplet_mass", pa.float32()),
            pa.field("strategy", pa.string()),
        ]
    )
    event_fields = [
        pa.field("event_id", pa.int64()),
        pa.field("n_triplets_total", pa.int32()),
        pa.field("n_top_selected", pa.int32()),
    ]
    for rank in range(1, EVENT_TOP_SLOTS + 1):
        event_fields.extend(
            [
                pa.field(f"top{rank}_pt", pa.float32()),
                pa.field(f"top{rank}_eta", pa.float32()),
                pa.field(f"top{rank}_phi", pa.float32()),
                pa.field(f"top{rank}_mass", pa.float32()),
            ]
        )
    event_fields.append(pa.field("m_top1_top2", pa.float32()))
    event_schema = pa.schema(
        event_fields
    )

    selected_writer = pio.StreamingParquetWriter(
        output_path=selected_path,
        schema=selected_schema,
        row_group_size=args.row_group_size,
    )
    event_writer = pio.StreamingParquetWriter(
        output_path=event_path,
        schema=event_schema,
        row_group_size=args.row_group_size,
    )

    selected_buffer = pio.ColumnBuffer([field.name for field in selected_schema])
    event_buffer = pio.ColumnBuffer([field.name for field in event_schema])

    dataset = ds.dataset(args.inference, format="parquet")
    total_rows_estimate = None
    if show_progress:
        try:
            total_rows_estimate = int(dataset.count_rows())
        except Exception:
            total_rows_estimate = None
    progress_bar = prog.ProgressBar(
        desc="select_triplets rows",
        total=total_rows_estimate,
        unit="rows",
        enabled=show_progress,
    )
    scanner = dataset.scanner(columns=required_columns, batch_size=args.batch_size)

    current_event_id: Optional[int] = None
    current_triplets: List[TripletCandidate] = []
    closed_events: set[int] = set()

    rows_input = 0
    events_total = 0
    selected_rows_total = 0
    events_with_selection = 0
    events_with_lt6_jets = 0
    events_with_ge6_jets = 0

    def flush_buffers(force: bool = False) -> None:
        if selected_buffer.size >= args.flush_rows or (force and selected_buffer.size > 0):
            selected_writer.write_rows(selected_buffer.take_all())
        if event_buffer.size >= args.flush_rows or (force and event_buffer.size > 0):
            event_writer.write_rows(event_buffer.take_all())

    def flush_event(event_id: int, triplets: Sequence[TripletCandidate]) -> None:
        nonlocal events_total, selected_rows_total, events_with_selection, events_with_lt6_jets, events_with_ge6_jets

        events_total += 1
        n_jets_in_event = _event_n_jets(triplets)
        if n_jets_in_event < PAIR_STRATEGY_MIN_JETS:
            events_with_lt6_jets += 1
        else:
            events_with_ge6_jets += 1
        selected = _apply_strategy(
            triplets=triplets,
            strategy=args.strategy,
            min_score=args.min_score,
            max_top_per_event=args.max_top_per_event,
            top_k=args.top_k,
            n_jets_in_event=n_jets_in_event,
        )

        event_row: Dict[str, float | int] = {
            "event_id": int(event_id),
            "n_triplets_total": int(len(triplets)),
            "n_top_selected": int(len(selected)),
        }
        for rank in range(1, EVENT_TOP_SLOTS + 1):
            event_row[f"top{rank}_pt"] = float(args.dummy_value)
            event_row[f"top{rank}_eta"] = float(args.dummy_value)
            event_row[f"top{rank}_phi"] = float(args.dummy_value)
            event_row[f"top{rank}_mass"] = float(args.dummy_value)
        event_row["m_top1_top2"] = float(args.dummy_value)

        for idx, triplet in enumerate(selected[:EVENT_TOP_SLOTS], start=1):
            event_row[f"top{idx}_pt"] = float(triplet.triplet_pt)
            event_row[f"top{idx}_eta"] = float(triplet.triplet_eta)
            event_row[f"top{idx}_phi"] = float(triplet.triplet_phi)
            event_row[f"top{idx}_mass"] = float(triplet.triplet_mass)

        if len(selected) >= 2:
            event_row["m_top1_top2"] = float(_pair_invariant_mass(selected[0], selected[1]))

        event_buffer.append_row(event_row)

        if len(selected) > 0:
            events_with_selection += 1
        for rank, triplet in enumerate(selected, start=1):
            selected_buffer.append_row(
                {
                    "event_id": int(event_id),
                    "selected_rank": int(rank),
                    "i": int(triplet.i),
                    "j": int(triplet.j),
                    "k": int(triplet.k),
                    "score": float(triplet.score),
                    "triplet_pt": float(triplet.triplet_pt),
                    "triplet_eta": float(triplet.triplet_eta),
                    "triplet_phi": float(triplet.triplet_phi),
                    "triplet_mass": float(triplet.triplet_mass),
                    "strategy": str(args.strategy),
                }
            )
            selected_rows_total += 1

        flush_buffers(force=False)

    try:
        for batch in scanner.to_batches():
            payload = batch.to_pydict()
            n_rows = len(payload["event_id"])
            if n_rows == 0:
                continue

            rows_input += n_rows
            progress_bar.update(n_rows)

            for idx in range(n_rows):
                event_id = int(payload["event_id"][idx])
                row = TripletCandidate(
                    i=int(payload["i"][idx]),
                    j=int(payload["j"][idx]),
                    k=int(payload["k"][idx]),
                    score=float(payload[score_column][idx]),
                    triplet_pt=float(payload["triplet_pt"][idx]),
                    triplet_eta=float(payload["triplet_eta"][idx]),
                    triplet_phi=float(payload["triplet_phi"][idx]),
                    triplet_mass=float(payload["m123"][idx]),
                )

                if current_event_id is None:
                    current_event_id = event_id
                elif event_id != current_event_id:
                    flush_event(current_event_id, current_triplets)
                    closed_events.add(current_event_id)
                    if event_id in closed_events:
                        raise RuntimeError(
                            "Input inference parquet rows are not grouped by event_id; selection stage expects grouped events."
                        )
                    current_event_id = event_id
                    current_triplets = []

                current_triplets.append(row)

        if current_event_id is not None:
            flush_event(current_event_id, current_triplets)
    finally:
        progress_bar.close()
        flush_buffers(force=True)
        selected_writer.close()
        event_writer.close()

    plot_root = Path(args.plot_root) if args.plot_root else output_dir / "plots"
    plot_metrics: Dict[str, Any] = {}
    if not args.skip_plots:
        plot_metrics = _generate_selection_plots(
            event_selection_file=event_path,
            output_dir=plot_root,
            batch_size=args.batch_size,
            bins=args.plot_bins,
            dummy_value=args.dummy_value,
            strategy=args.strategy,
        )

    report = {
        "schema_version": pio.SCHEMA_VERSION,
        "strategy": args.strategy,
        "score_column": score_column,
        "min_score": float(args.min_score),
        "max_top_per_event": int(args.max_top_per_event),
        "top_k": int(args.top_k),
        "dummy_value": float(args.dummy_value),
        "rows_input": int(rows_input),
        "events_total": int(events_total),
        "selected_rows_total": int(selected_rows_total),
        "events_with_selection": int(events_with_selection),
        "events_without_selection": int(events_total - events_with_selection),
        "avg_selected_per_event": float(selected_rows_total / events_total) if events_total > 0 else 0.0,
        "events_with_lt6_jets_inferred": int(events_with_lt6_jets),
        "events_with_ge6_jets_inferred": int(events_with_ge6_jets),
        "selected_triplets_file": str(selected_path),
        "event_selection_file": str(event_path),
        "plot_metrics": plot_metrics,
    }
    pio.write_json(output_dir / "selection_report.json", report)

    pio.write_config_snapshot(
        output_dir=output_dir,
        stage="select_triplets",
        input_files=[args.inference],
        parameters={
            "strategy": args.strategy,
            "score_column": score_column,
            "min_score": args.min_score,
            "max_top_per_event": args.max_top_per_event,
            "top_k": args.top_k,
            "pair_strategy_min_jets": PAIR_STRATEGY_MIN_JETS,
            "dummy_value": args.dummy_value,
            "row_group_size": args.row_group_size,
            "batch_size": args.batch_size,
            "flush_rows": args.flush_rows,
            "skip_plots": args.skip_plots,
            "plot_root": str(plot_root),
            "plot_bins": args.plot_bins,
            "no_progress": args.no_progress,
        },
        seed=42,
    )


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "select_triplets",
        help="Stage 5: select reconstructed top-candidate triplets from scored inference parquet",
    )
    parser.add_argument("--inference", required=True, help="Inference parquet path")
    parser.add_argument("--output-dir", default="artifacts/select_triplets", help="Output directory")
    parser.add_argument(
        "--strategy",
        default="greedy_disjoint",
        choices=list(STRATEGIES),
        help="Triplet selection strategy",
    )
    parser.add_argument("--score-column", default=None, help="Optional explicit score column name")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum score required for a selected triplet",
    )
    parser.add_argument(
        "--max-top-per-event",
        type=int,
        default=4,
        help="Maximum selected triplets per event (must be <= 4)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Top-k used when --strategy topk",
    )
    parser.add_argument("--batch-size", type=int, default=100_000, help="Parquet read batch size")
    parser.add_argument("--row-group-size", type=int, default=50_000, help="Output parquet row group size")
    parser.add_argument("--flush-rows", type=int, default=50_000, help="In-memory flush threshold")
    parser.add_argument(
        "--dummy-value",
        type=float,
        default=-999.0,
        help="Placeholder value used for missing top-candidate slots in event output",
    )
    parser.add_argument("--plot-root", default=None, help="Output directory for selection plots (default: <output-dir>/plots)")
    parser.add_argument("--plot-bins", type=int, default=20, help="Histogram bin count for selection plots")
    parser.add_argument("--skip-plots", action="store_true", help="Skip automatic selection plotting")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress output")
    parser.set_defaults(func=run)
