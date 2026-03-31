#!/usr/bin/env python3
"""
Validate triplet branch interpretation in ttbar ntuple.

This script builds and compares six kinematic distribution sets:

A: reco_triplet_0 interpreted as indices into reco jets (jet_*)
B: reco_triplet_0 interpreted as indices into gen jets (genjet_*)
C: truth_triplet_0 interpreted as indices into reco jets (jet_*)
D: truth_triplet_0 interpreted as indices into gen jets (genjet_*)
E: top[0] truth-level top kinematics (top_*)
F: top[1] truth-level top kinematics (top_*)

For each observable (pt, eta, phi, mass), the six sets are overlaid.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import awkward as ak
import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OBSERVABLES = ("pt", "eta", "phi", "mass")
SET_ORDER = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
)
NO_TOP_SET_ORDER = ("A", "B", "C", "D")

SET_LABELS = {
    "A": "A: reco_triplet_0 -> jet_*",
    "B": "B: reco_triplet_0 -> genjet_*",
    "C": "C: truth_triplet_0 -> jet_*",
    "D": "D: truth_triplet_0 -> genjet_*",
    "E": "E: top[0]",
    "F": "F: top[1]",
}

SET_STYLES = {
    "A": {"color": "#1f77b4", "linestyle": "-"},
    "B": {"color": "#ff7f0e", "linestyle": "--"},
    "C": {"color": "#2ca02c", "linestyle": "-."},
    "D": {"color": "#d62728", "linestyle": ":"},
    "E": {"color": "#9467bd", "linestyle": (0, (5, 1))},
    "F": {"color": "#8c564b", "linestyle": (0, (3, 1, 1, 1))},
}

DEFAULT_INPUT = "ttbar.root"
DEFAULT_OUTDIR = "triplet_validation_plots"
DEFAULT_MAX_EVENTS = 1000
DEFAULT_STEP_SIZE = 200_000

REQUIRED_BRANCHES = [
    "reco_triplet_0",
    "truth_triplet_0",
    "triplet_0_pt",
    "triplet_0_eta",
    "triplet_0_phi",
    "triplet_0_m",
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_m",
    "genjet_pt",
    "genjet_eta",
    "genjet_phi",
    "genjet_m",
    "top_pt",
    "top_eta",
    "top_phi",
    "top_m",
]


@dataclass
class ValidationStats:
    events_read: int = 0
    valid_A: int = 0
    valid_B: int = 0
    valid_C: int = 0
    valid_D: int = 0
    valid_E: int = 0
    valid_F: int = 0
    skipped_A_bad_indices: int = 0
    skipped_B_bad_indices: int = 0
    skipped_C_bad_indices: int = 0
    skipped_D_bad_indices: int = 0
    skipped_A_malformed: int = 0
    skipped_B_malformed: int = 0
    skipped_C_malformed: int = 0
    skipped_D_malformed: int = 0
    skipped_E_missing: int = 0
    skipped_F_missing: int = 0
    skipped_event_exceptions: int = 0


def parse_max_events(text: str) -> Optional[int]:
    lower = text.strip().lower()
    if lower == "none":
        return None
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("--max-events must be > 0 or 'None'")
    return value


def find_ttree_name(root_file: uproot.ReadOnlyDirectory, preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    for key, cls_name in root_file.classnames().items():
        if "TTree" in cls_name:
            return key.split(";")[0]
    raise RuntimeError("No TTree found in ROOT file.")


def read_branches(
    tree: uproot.behaviors.TTree.TTree,
    entry_start: int,
    entry_stop: int,
) -> ak.Array:
    return tree.arrays(
        REQUIRED_BRANCHES,
        entry_start=entry_start,
        entry_stop=entry_stop,
        library="ak",
    )


def extract_triplet_indices(value: ak.Array) -> Optional[np.ndarray]:
    try:
        indices = np.asarray(ak.to_list(value), dtype=np.int64)
    except Exception:
        return None
    if indices.ndim != 1 or indices.shape[0] != 3:
        return None
    return indices


def ptetaphim_to_components(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0))
    return px, py, pz, energy


def build_sum_fourvector(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    mass: np.ndarray,
    indices: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    if indices is None or indices.shape[0] != 3:
        return None
    if np.any(indices < 0) or np.any(indices >= len(pt)):
        return None

    idx = indices.astype(np.int64)
    pts = pt[idx]
    etas = eta[idx]
    phis = phi[idx]
    masses = mass[idx]

    if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(etas)) or not np.all(np.isfinite(phis)) or not np.all(np.isfinite(masses)):
        return None

    px, py, pz, energy = ptetaphim_to_components(pts, etas, phis, masses)
    sum_px = float(np.sum(px))
    sum_py = float(np.sum(py))
    sum_pz = float(np.sum(pz))
    sum_energy = float(np.sum(energy))

    out_pt = float(np.hypot(sum_px, sum_py))
    out_phi = float(np.arctan2(sum_py, sum_px))
    if out_pt > 0.0:
        out_eta = float(np.arcsinh(sum_pz / out_pt))
    else:
        out_eta = np.nan

    m2 = sum_energy * sum_energy - (sum_px * sum_px + sum_py * sum_py + sum_pz * sum_pz)
    out_mass = float(np.sqrt(max(m2, 0.0)))

    if not np.isfinite(out_pt) or not np.isfinite(out_eta) or not np.isfinite(out_phi) or not np.isfinite(out_mass):
        return None
    return out_pt, out_eta, out_phi, out_mass


def extract_top_fourvector(
    top_pt: np.ndarray,
    top_eta: np.ndarray,
    top_phi: np.ndarray,
    top_mass: np.ndarray,
    index: int,
) -> Optional[Tuple[float, float, float, float]]:
    if len(top_pt) <= index or len(top_eta) <= index or len(top_phi) <= index or len(top_mass) <= index:
        return None

    values = (
        float(top_pt[index]),
        float(top_eta[index]),
        float(top_phi[index]),
        float(top_mass[index]),
    )
    if not np.all(np.isfinite(values)):
        return None
    if values[0] < 0.0:
        return None
    return values


def append_kinematics(store: Dict[str, Dict[str, List[float]]], set_name: str, values: Tuple[float, float, float, float]) -> None:
    pt, eta, phi, mass = values
    store[set_name]["pt"].append(pt)
    store[set_name]["eta"].append(eta)
    store[set_name]["phi"].append(phi)
    store[set_name]["mass"].append(mass)


def build_distributions(
    arrays: ak.Array,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], ValidationStats, Dict[str, Dict[str, np.ndarray]]]:
    stats = ValidationStats(events_read=len(arrays["reco_triplet_0"]))
    data_store: Dict[str, Dict[str, List[float]]] = {
        key: {obs: [] for obs in OBSERVABLES} for key in SET_ORDER
    }

    triplet_branch_store = {obs: [] for obs in OBSERVABLES}
    reco_from_indices_store = {obs: [] for obs in OBSERVABLES}

    for i in range(stats.events_read):
        try:
            reco_indices = extract_triplet_indices(arrays["reco_triplet_0"][i])
            truth_indices = extract_triplet_indices(arrays["truth_triplet_0"][i])

            jet_pt = ak.to_numpy(arrays["jet_pt"][i])
            jet_eta = ak.to_numpy(arrays["jet_eta"][i])
            jet_phi = ak.to_numpy(arrays["jet_phi"][i])
            jet_m = ak.to_numpy(arrays["jet_m"][i])

            genjet_pt = ak.to_numpy(arrays["genjet_pt"][i])
            genjet_eta = ak.to_numpy(arrays["genjet_eta"][i])
            genjet_phi = ak.to_numpy(arrays["genjet_phi"][i])
            genjet_m = ak.to_numpy(arrays["genjet_m"][i])

            top_pt = ak.to_numpy(arrays["top_pt"][i])
            top_eta = ak.to_numpy(arrays["top_eta"][i])
            top_phi = ak.to_numpy(arrays["top_phi"][i])
            top_m = ak.to_numpy(arrays["top_m"][i])

            cand_A = build_sum_fourvector(jet_pt, jet_eta, jet_phi, jet_m, reco_indices)
            if cand_A is None:
                if reco_indices is None:
                    stats.skipped_A_malformed += 1
                else:
                    stats.skipped_A_bad_indices += 1
            else:
                append_kinematics(data_store, "A", cand_A)
                stats.valid_A += 1

                # Triplet branch cross-check uses only events with valid reconstructed A.
                t0_pt = float(arrays["triplet_0_pt"][i])
                t0_eta = float(arrays["triplet_0_eta"][i])
                t0_phi = float(arrays["triplet_0_phi"][i])
                t0_m = float(arrays["triplet_0_m"][i])
                t0_values = (t0_pt, t0_eta, t0_phi, t0_m)
                if np.all(np.isfinite(t0_values)) and t0_pt >= 0.0 and t0_m >= 0.0:
                    reco_from_indices_store["pt"].append(cand_A[0])
                    reco_from_indices_store["eta"].append(cand_A[1])
                    reco_from_indices_store["phi"].append(cand_A[2])
                    reco_from_indices_store["mass"].append(cand_A[3])
                    triplet_branch_store["pt"].append(t0_pt)
                    triplet_branch_store["eta"].append(t0_eta)
                    triplet_branch_store["phi"].append(t0_phi)
                    triplet_branch_store["mass"].append(t0_m)

            cand_B = build_sum_fourvector(genjet_pt, genjet_eta, genjet_phi, genjet_m, reco_indices)
            if cand_B is None:
                if reco_indices is None:
                    stats.skipped_B_malformed += 1
                else:
                    stats.skipped_B_bad_indices += 1
            else:
                append_kinematics(data_store, "B", cand_B)
                stats.valid_B += 1

            cand_C = build_sum_fourvector(jet_pt, jet_eta, jet_phi, jet_m, truth_indices)
            if cand_C is None:
                if truth_indices is None:
                    stats.skipped_C_malformed += 1
                else:
                    stats.skipped_C_bad_indices += 1
            else:
                append_kinematics(data_store, "C", cand_C)
                stats.valid_C += 1

            cand_D = build_sum_fourvector(genjet_pt, genjet_eta, genjet_phi, genjet_m, truth_indices)
            if cand_D is None:
                if truth_indices is None:
                    stats.skipped_D_malformed += 1
                else:
                    stats.skipped_D_bad_indices += 1
            else:
                append_kinematics(data_store, "D", cand_D)
                stats.valid_D += 1

            cand_E = extract_top_fourvector(top_pt, top_eta, top_phi, top_m, index=0)
            if cand_E is None:
                stats.skipped_E_missing += 1
            else:
                append_kinematics(data_store, "E", cand_E)
                stats.valid_E += 1

            cand_F = extract_top_fourvector(top_pt, top_eta, top_phi, top_m, index=1)
            if cand_F is None:
                stats.skipped_F_missing += 1
            else:
                append_kinematics(data_store, "F", cand_F)
                stats.valid_F += 1
        except Exception:
            stats.skipped_event_exceptions += 1
            continue

    output = {
        set_name: {obs: np.asarray(vals[obs], dtype=float) for obs in OBSERVABLES}
        for set_name, vals in data_store.items()
    }
    triplet_check = {
        "reco": {obs: np.asarray(reco_from_indices_store[obs], dtype=float) for obs in OBSERVABLES},
        "triplet": {obs: np.asarray(triplet_branch_store[obs], dtype=float) for obs in OBSERVABLES},
    }
    return output, stats, triplet_check


def init_distribution_chunks() -> Dict[str, Dict[str, List[np.ndarray]]]:
    return {set_name: {obs: [] for obs in OBSERVABLES} for set_name in SET_ORDER}


def finalize_distribution_chunks(chunks: Dict[str, Dict[str, List[np.ndarray]]]) -> Dict[str, Dict[str, np.ndarray]]:
    finalized: Dict[str, Dict[str, np.ndarray]] = {}
    for set_name in SET_ORDER:
        finalized[set_name] = {}
        for obs in OBSERVABLES:
            parts = [arr for arr in chunks[set_name][obs] if arr.size > 0]
            if parts:
                finalized[set_name][obs] = np.concatenate(parts)
            else:
                finalized[set_name][obs] = np.array([], dtype=float)
    return finalized


def merge_stats(total: ValidationStats, chunk: ValidationStats) -> None:
    for field_name in total.__dataclass_fields__:
        setattr(total, field_name, getattr(total, field_name) + getattr(chunk, field_name))


def init_triplet_chunks() -> Dict[str, Dict[str, List[np.ndarray]]]:
    return {
        "reco": {obs: [] for obs in OBSERVABLES},
        "triplet": {obs: [] for obs in OBSERVABLES},
    }


def finalize_triplet_chunks(chunks: Dict[str, Dict[str, List[np.ndarray]]]) -> Dict[str, Dict[str, np.ndarray]]:
    output: Dict[str, Dict[str, np.ndarray]] = {}
    for key in ("reco", "triplet"):
        output[key] = {}
        for obs in OBSERVABLES:
            parts = [arr for arr in chunks[key][obs] if arr.size > 0]
            if parts:
                output[key][obs] = np.concatenate(parts)
            else:
                output[key][obs] = np.array([], dtype=float)
    return output


def _collect_nonempty(arrays: Iterable[np.ndarray]) -> np.ndarray:
    valid = [arr[np.isfinite(arr)] for arr in arrays if arr.size > 0]
    if not valid:
        return np.array([], dtype=float)
    return np.concatenate(valid)


def choose_bins(
    distributions: Dict[str, Dict[str, np.ndarray]],
    observable: str,
    n_bins: int = 60,
    set_order: Optional[Sequence[str]] = None,
) -> np.ndarray:
    order = SET_ORDER if set_order is None else tuple(set_order)
    all_values = _collect_nonempty(distributions[set_name][observable] for set_name in order)

    if observable == "phi":
        return np.linspace(-np.pi, np.pi, n_bins + 1)

    if all_values.size == 0:
        if observable in ("pt", "mass"):
            return np.linspace(0.0, 500.0, n_bins + 1)
        return np.linspace(-5.0, 5.0, n_bins + 1)

    if observable in ("pt", "mass"):
        upper = float(np.percentile(all_values[all_values >= 0.0], 99.5)) if np.any(all_values >= 0.0) else float(np.max(all_values))
        upper = max(1.0, upper)
        return np.linspace(0.0, upper, n_bins + 1)

    low = float(np.percentile(all_values, 0.5))
    high = float(np.percentile(all_values, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = -5.0, 5.0
    return np.linspace(low, high, n_bins + 1)


def fill_histograms(
    distributions: Dict[str, Dict[str, np.ndarray]],
    observable: str,
    bins: np.ndarray,
    set_order: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    order = SET_ORDER if set_order is None else tuple(set_order)
    histograms: Dict[str, np.ndarray] = {}
    bin_widths = np.diff(bins).astype(float)
    for set_name in order:
        values = distributions[set_name][observable]
        values = values[np.isfinite(values)]
        counts, _ = np.histogram(values, bins=bins)
        counts = counts.astype(float)
        area = float(np.sum(counts * bin_widths))
        if area > 0.0:
            histograms[set_name] = counts / area
        else:
            histograms[set_name] = np.zeros_like(counts, dtype=float)
    return histograms


def plot_histogram_overlay(
    histograms: Dict[str, np.ndarray],
    bins: np.ndarray,
    distributions: Dict[str, Dict[str, np.ndarray]],
    observable: str,
    outdir: str,
    image_format: str,
    set_order: Optional[Sequence[str]] = None,
    filename_suffix: str = "",
    title_suffix: str = "",
    xlim: Optional[Tuple[float, float]] = None,
) -> str:
    order = SET_ORDER if set_order is None else tuple(set_order)
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    for set_name in order:
        counts = histograms[set_name]
        style = SET_STYLES[set_name]
        n_entries = len(distributions[set_name][observable])
        label = f"{SET_LABELS[set_name]} (n={n_entries})"
        ax.step(
            bins[:-1],
            counts,
            where="post",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=label,
        )

    xlabel = {
        "pt": r"$p_T$ [GeV]",
        "eta": r"$\eta$",
        "phi": r"$\phi$ [rad]",
        "mass": r"mass [GeV]",
    }[observable]

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized density (area = 1)")
    ax.set_title(f"Triplet/Top Validation: {observable} (area-normalized){title_suffix}")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    outfile = os.path.join(outdir, f"{observable}_overlay{filename_suffix}.{image_format}")
    fig.savefig(outfile, dpi=160)
    plt.close(fig)
    return outfile


def summarize_triplet_branch_consistency(triplet_check: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    reco = triplet_check["reco"]
    triplet = triplet_check["triplet"]
    n = min(len(reco["pt"]), len(triplet["pt"]))
    summary["n_compared"] = float(n)
    if n == 0:
        return summary

    for obs in OBSERVABLES:
        diff = np.abs(reco[obs][:n] - triplet[obs][:n])
        summary[f"{obs}_mean_abs_diff"] = float(np.mean(diff))
        summary[f"{obs}_median_abs_diff"] = float(np.median(diff))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate triplet branch interpretation with kinematic overlays.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input ROOT file path (default: ttbar.root)")
    parser.add_argument("--tree", default=None, help="Optional TTree name. If omitted, first TTree is used.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for plots.")
    parser.add_argument(
        "--max-events",
        type=parse_max_events,
        default=DEFAULT_MAX_EVENTS,
        help="Events to process (default: 1000). Use 'None' to process all events.",
    )
    parser.add_argument("--bins", type=int, default=60, help="Number of bins per histogram.")
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
        help="Output figure format.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=DEFAULT_STEP_SIZE,
        help="Chunk size for reading branches with uproot.iterate.",
    )
    args = parser.parse_args()

    distribution_chunks = init_distribution_chunks()
    triplet_chunks = init_triplet_chunks()
    stats = ValidationStats()

    with uproot.open(args.input) as root_file:
        resolved_tree_name = find_ttree_name(root_file, preferred=args.tree)
        tree = root_file[resolved_tree_name]
        total_entries = int(tree.num_entries)
        entries_to_process = total_entries if args.max_events is None else min(total_entries, args.max_events)

        for start in range(0, entries_to_process, args.step_size):
            stop = min(start + args.step_size, entries_to_process)
            chunk_arrays = read_branches(tree, entry_start=start, entry_stop=stop)
            chunk_distributions, chunk_stats, chunk_triplet = build_distributions(chunk_arrays)
            merge_stats(stats, chunk_stats)
            for set_name in SET_ORDER:
                for obs in OBSERVABLES:
                    distribution_chunks[set_name][obs].append(chunk_distributions[set_name][obs])
            for key in ("reco", "triplet"):
                for obs in OBSERVABLES:
                    triplet_chunks[key][obs].append(chunk_triplet[key][obs])

    distributions = finalize_distribution_chunks(distribution_chunks)
    triplet_check = finalize_triplet_chunks(triplet_chunks)
    triplet_summary = summarize_triplet_branch_consistency(triplet_check)

    os.makedirs(args.outdir, exist_ok=True)
    saved_files: List[str] = []
    saved_files_no_truth_tops: List[str] = []
    saved_mass_zoom_files_no_truth_tops: List[str] = []
    for observable in OBSERVABLES:
        bins = choose_bins(distributions, observable, n_bins=args.bins)
        histograms = fill_histograms(distributions, observable, bins)
        saved_files.append(
            plot_histogram_overlay(histograms, bins, distributions, observable, args.outdir, args.format)
        )
        bins_no_truth_tops = choose_bins(
            distributions,
            observable,
            n_bins=args.bins,
            set_order=NO_TOP_SET_ORDER,
        )
        histograms_no_truth_tops = fill_histograms(
            distributions,
            observable,
            bins_no_truth_tops,
            set_order=NO_TOP_SET_ORDER,
        )
        saved_files_no_truth_tops.append(
            plot_histogram_overlay(
                histograms_no_truth_tops,
                bins_no_truth_tops,
                distributions,
                observable,
                args.outdir,
                args.format,
                set_order=NO_TOP_SET_ORDER,
                filename_suffix="_no_truth_tops",
                title_suffix=" (no top[0]/top[1])",
            )
        )

        if observable == "mass":
            mass_zoom_bins = np.linspace(120.0, 220.0, args.bins + 1)
            mass_zoom_hists_no_truth_tops = fill_histograms(
                distributions,
                observable,
                mass_zoom_bins,
                set_order=NO_TOP_SET_ORDER,
            )
            saved_mass_zoom_files_no_truth_tops.append(
                plot_histogram_overlay(
                    mass_zoom_hists_no_truth_tops,
                    mass_zoom_bins,
                    distributions,
                    observable,
                    args.outdir,
                    args.format,
                    set_order=NO_TOP_SET_ORDER,
                    filename_suffix="_no_truth_tops_zoom120_220",
                    title_suffix=" (no top[0]/top[1], 120-220 GeV)",
                    xlim=(120.0, 220.0),
                )
            )

    print(f"Total entries in tree: {total_entries}")
    print(f"Resolved tree: {resolved_tree_name}")
    print(f"Requested max events: {args.max_events}")
    print(f"Processed events: {stats.events_read} / {entries_to_process}")
    print("")
    print("Valid candidates:")
    print(f"  A (reco_triplet_0 -> jet_*): {stats.valid_A}")
    print(f"  B (reco_triplet_0 -> genjet_*): {stats.valid_B}")
    print(f"  C (truth_triplet_0 -> jet_*): {stats.valid_C}")
    print(f"  D (truth_triplet_0 -> genjet_*): {stats.valid_D}")
    print(f"  E (top[0]): {stats.valid_E}")
    print(f"  F (top[1]): {stats.valid_F}")
    print("")
    print("Skipped candidates:")
    print(
        f"  A skipped: malformed={stats.skipped_A_malformed}, bad_indices={stats.skipped_A_bad_indices}"
    )
    print(
        f"  B skipped: malformed={stats.skipped_B_malformed}, bad_indices={stats.skipped_B_bad_indices}"
    )
    print(
        f"  C skipped: malformed={stats.skipped_C_malformed}, bad_indices={stats.skipped_C_bad_indices}"
    )
    print(
        f"  D skipped: malformed={stats.skipped_D_malformed}, bad_indices={stats.skipped_D_bad_indices}"
    )
    print(f"  E skipped missing/invalid: {stats.skipped_E_missing}")
    print(f"  F skipped missing/invalid: {stats.skipped_F_missing}")
    print(f"  Event-level exceptions: {stats.skipped_event_exceptions}")
    print("")
    print("triplet_0_* cross-check against A (for events where both are valid):")
    if triplet_summary.get("n_compared", 0.0) <= 0:
        print("  No comparable events.")
    else:
        n_compared = int(triplet_summary["n_compared"])
        print(f"  Compared events: {n_compared}")
        for obs in OBSERVABLES:
            mean_diff = triplet_summary[f"{obs}_mean_abs_diff"]
            med_diff = triplet_summary[f"{obs}_median_abs_diff"]
            print(f"  {obs}: mean|diff|={mean_diff:.4g}, median|diff|={med_diff:.4g}")
    print("")
    print("Saved overlay plots:")
    for path in saved_files:
        print(f"  {path}")
    print("")
    print("Saved overlay plots (truth tops removed):")
    for path in saved_files_no_truth_tops:
        print(f"  {path}")
    print("")
    print("Saved invariant-mass zoom plots (truth tops removed, 120-220 GeV):")
    for path in saved_mass_zoom_files_no_truth_tops:
        print(f"  {path}")


if __name__ == "__main__":
    main()
