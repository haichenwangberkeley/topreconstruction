#!/usr/bin/env python3
"""
Deterministic triplet branch interpretation diagnostics (no plotting).

Run:
    python analysis/diagnose_triplet_interpretation.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import awkward as ak
import numpy as np
import uproot


DEFAULT_INPUT = "ttbar.root"
DEFAULT_MAX_EVENTS = 20


@dataclass
class ResidualRecord:
    event: int
    indices: Tuple[int, int, int]
    dpt: float
    deta: float
    dphi: float
    dm: float


@dataclass
class TestResult:
    name: str
    result: str
    reasoning: str
    evidence_lines: List[str]
    tag: Optional[str] = None


def parse_max_events(value: str) -> Optional[int]:
    low = value.strip().lower()
    if low == "none":
        return None
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("--max-events must be > 0 or None")
    return ivalue


def find_tree_name(root_file: uproot.ReadOnlyDirectory, preferred: Optional[str]) -> str:
    if preferred:
        return preferred
    for key, cls in root_file.classnames().items():
        if "TTree" in cls:
            return key.split(";")[0]
    raise RuntimeError("No TTree found in ROOT file.")


def wrap_phi(delta_phi: float) -> float:
    return math.atan2(math.sin(delta_phi), math.cos(delta_phi))


def ptetaphim_to_pxpypze(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0))
    return px, py, pz, e


def build_triplet_sum(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, mass: np.ndarray, indices: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    if indices.shape != (3,):
        return None
    if np.any(indices < 0) or np.any(indices >= len(pt)):
        return None

    idx = indices.astype(np.int64)
    ptx = pt[idx]
    etax = eta[idx]
    phix = phi[idx]
    mx = mass[idx]

    if not (np.all(np.isfinite(ptx)) and np.all(np.isfinite(etax)) and np.all(np.isfinite(phix)) and np.all(np.isfinite(mx))):
        return None

    px, py, pz, ee = ptetaphim_to_pxpypze(ptx, etax, phix, mx)
    sx = float(np.sum(px))
    sy = float(np.sum(py))
    sz = float(np.sum(pz))
    se = float(np.sum(ee))

    out_pt = float(math.hypot(sx, sy))
    out_phi = float(math.atan2(sy, sx))
    out_eta = float(np.arcsinh(sz / out_pt)) if out_pt > 0 else np.nan
    m2 = se * se - (sx * sx + sy * sy + sz * sz)
    out_m = float(np.sqrt(max(m2, 0.0)))

    if not np.all(np.isfinite([out_pt, out_eta, out_phi, out_m])):
        return None
    return out_pt, out_eta, out_phi, out_m


def extract_int_triplet(value: ak.Array) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(ak.to_list(value), dtype=np.int64)
    except Exception:
        return None
    if arr.shape != (3,):
        return None
    return arr


def summarize_abs(values: np.ndarray) -> Tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    av = np.abs(values)
    return float(np.max(av)), float(np.mean(av)), float(np.median(av))


def load_arrays(filename: str, tree_name: Optional[str], max_events: Optional[int]) -> Tuple[ak.Array, int, int, str]:
    branches = [
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
        "N_jet",
        "N_genjet",
    ]
    with uproot.open(filename) as root_file:
        resolved_tree = find_tree_name(root_file, tree_name)
        tree = root_file[resolved_tree]
        total_entries = int(tree.num_entries)
        stop = total_entries if max_events is None else min(total_entries, max_events)
        arrays = tree.arrays(branches, entry_start=0, entry_stop=stop, library="ak")
    return arrays, stop, total_entries, resolved_tree


def run_test1(arrays: ak.Array) -> TestResult:
    reasoning = (
        "Deterministic identity check: if triplet_0_* is built from reco_triplet_0 over jet_*, "
        "event-by-event residuals must be near zero, not just correlated."
    )

    records: List[ResidualRecord] = []
    skipped_bad_indices = 0
    skipped_malformed = 0

    n_events = len(arrays["reco_triplet_0"])
    for i in range(n_events):
        reco_idx = extract_int_triplet(arrays["reco_triplet_0"][i])
        if reco_idx is None:
            skipped_malformed += 1
            continue

        jet_pt = ak.to_numpy(arrays["jet_pt"][i])
        jet_eta = ak.to_numpy(arrays["jet_eta"][i])
        jet_phi = ak.to_numpy(arrays["jet_phi"][i])
        jet_m = ak.to_numpy(arrays["jet_m"][i])

        built = build_triplet_sum(jet_pt, jet_eta, jet_phi, jet_m, reco_idx)
        if built is None:
            skipped_bad_indices += 1
            continue

        ref = (
            float(arrays["triplet_0_pt"][i]),
            float(arrays["triplet_0_eta"][i]),
            float(arrays["triplet_0_phi"][i]),
            float(arrays["triplet_0_m"][i]),
        )
        if not np.all(np.isfinite(ref)):
            skipped_malformed += 1
            continue

        dpt = built[0] - ref[0]
        deta = built[1] - ref[1]
        dphi = wrap_phi(built[2] - ref[2])
        dm = built[3] - ref[3]
        records.append(
            ResidualRecord(
                event=i,
                indices=(int(reco_idx[0]), int(reco_idx[1]), int(reco_idx[2])),
                dpt=float(dpt),
                deta=float(deta),
                dphi=float(dphi),
                dm=float(dm),
            )
        )

    n_valid = len(records)
    if n_valid == 0:
        return TestResult(
            name="TEST 1: triplet_0_* construction test",
            result="INCONCLUSIVE",
            reasoning=reasoning,
            evidence_lines=[
                "No valid events after skipping malformed/invalid triplet indices.",
                f"Skipped malformed: {skipped_malformed}, skipped bad indices: {skipped_bad_indices}",
            ],
            tag="triplet0_from_reco_triplet0_unknown",
        )

    dpt = np.array([r.dpt for r in records], dtype=float)
    deta = np.array([r.deta for r in records], dtype=float)
    dphi = np.array([r.dphi for r in records], dtype=float)
    dm = np.array([r.dm for r in records], dtype=float)

    pt_abs_max, pt_abs_mean, pt_abs_med = summarize_abs(dpt)
    eta_abs_max, eta_abs_mean, eta_abs_med = summarize_abs(deta)
    phi_abs_max, phi_abs_mean, phi_abs_med = summarize_abs(dphi)
    m_abs_max, m_abs_mean, m_abs_med = summarize_abs(dm)

    # Tight deterministic thresholds with slight slack for float precision / branch rounding.
    pass_mask = (
        (np.abs(dpt) < 1e-3)
        & (np.abs(deta) < 1e-4)
        & (np.abs(dphi) < 1e-4)
        & (np.abs(dm) < 5e-2)
    )
    pass_fraction = float(np.mean(pass_mask))

    if n_valid < 5:
        result = "INCONCLUSIVE"
        tag = "triplet0_from_reco_triplet0_unknown"
    elif pass_fraction >= 0.9:
        result = "PASS"
        tag = "triplet0_from_reco_triplet0_yes"
    else:
        result = "FAIL"
        tag = "triplet0_from_reco_triplet0_no"

    worst = sorted(records, key=lambda r: abs(r.dpt) + abs(r.deta) + abs(r.dphi) + abs(r.dm), reverse=True)[:3]
    evidence = [
        f"Valid events tested: {n_valid}",
        f"Skipped malformed: {skipped_malformed}, skipped bad indices: {skipped_bad_indices}",
        (
            "Residual |Δ| summaries: "
            f"pt max/mean/median={pt_abs_max:.3g}/{pt_abs_mean:.3g}/{pt_abs_med:.3g}, "
            f"eta={eta_abs_max:.3g}/{eta_abs_mean:.3g}/{eta_abs_med:.3g}, "
            f"phi={phi_abs_max:.3g}/{phi_abs_mean:.3g}/{phi_abs_med:.3g}, "
            f"m={m_abs_max:.3g}/{m_abs_mean:.3g}/{m_abs_med:.3g}"
        ),
        f"Events passing tight identity tolerances: {int(np.sum(pass_mask))}/{n_valid} ({100.0*pass_fraction:.1f}%)",
    ]
    evidence.append("Example events (indices, residuals):")
    for r in worst:
        evidence.append(
            f"  evt {r.event}: idx={list(r.indices)} | "
            f"Δpt={r.dpt:.3g}, Δη={r.deta:.3g}, Δphi={r.dphi:.3g}, Δm={r.dm:.3g}"
        )

    return TestResult(
        name="TEST 1: triplet_0_* construction test",
        result=result,
        reasoning=reasoning,
        evidence_lines=evidence,
        tag=tag,
    )


def run_test2(arrays: ak.Array) -> TestResult:
    reasoning = (
        "Hard index-space constraints: valid reco interpretation requires 0<=idx<N_jet, "
        "valid gen interpretation requires 0<=idx<N_genjet."
    )

    reco_only = 0
    gen_only = 0
    both = 0
    neither = 0
    skipped_malformed = 0
    skipped_negative_dummy = 0

    reco_only_examples: List[str] = []
    gen_only_examples: List[str] = []
    neither_examples: List[str] = []

    n_events = len(arrays["truth_triplet_0"])
    for i in range(n_events):
        truth_idx = extract_int_triplet(arrays["truth_triplet_0"][i])
        if truth_idx is None:
            skipped_malformed += 1
            continue

        if np.any(truth_idx < 0):
            skipped_negative_dummy += 1
            continue

        n_jet = int(arrays["N_jet"][i])
        n_genjet = int(arrays["N_genjet"][i])
        valid_reco = bool(np.all(truth_idx < n_jet))
        valid_gen = bool(np.all(truth_idx < n_genjet))

        if valid_reco and valid_gen:
            both += 1
        elif valid_reco and (not valid_gen):
            reco_only += 1
            if len(reco_only_examples) < 3:
                reco_only_examples.append(
                    f"evt {i}: idx={truth_idx.tolist()}, N_jet={n_jet}, N_genjet={n_genjet}"
                )
        elif valid_gen and (not valid_reco):
            gen_only += 1
            if len(gen_only_examples) < 3:
                gen_only_examples.append(
                    f"evt {i}: idx={truth_idx.tolist()}, N_jet={n_jet}, N_genjet={n_genjet}"
                )
        else:
            neither += 1
            if len(neither_examples) < 3:
                neither_examples.append(
                    f"evt {i}: idx={truth_idx.tolist()}, N_jet={n_jet}, N_genjet={n_genjet}"
                )

    evaluated = reco_only + gen_only + both + neither
    if evaluated == 0:
        return TestResult(
            name="TEST 2: truth_triplet_0 index space test",
            result="INCONCLUSIVE",
            reasoning=reasoning,
            evidence_lines=[
                "No evaluable events with non-negative truth_triplet_0 indices.",
                f"Skipped malformed: {skipped_malformed}, skipped negative/dummy: {skipped_negative_dummy}",
            ],
            tag="truth_triplet_index_space_unknown",
        )

    reco_valid = reco_only + both
    gen_valid = gen_only + both
    reco_only_frac = reco_only / evaluated
    gen_only_frac = gen_only / evaluated

    if evaluated < 5:
        result = "INCONCLUSIVE"
        tag = "truth_triplet_index_space_unknown"
    elif reco_only_frac >= 0.6 and reco_only_frac > (gen_only_frac + 0.2):
        result = "PASS"
        tag = "truth_triplet_index_space_reco"
    elif gen_only_frac >= 0.6 and gen_only_frac > (reco_only_frac + 0.2):
        result = "PASS"
        tag = "truth_triplet_index_space_gen"
    else:
        result = "INCONCLUSIVE"
        tag = "truth_triplet_index_space_ambiguous"

    evidence = [
        f"Evaluated events (non-negative indices): {evaluated}",
        f"Skipped malformed: {skipped_malformed}, skipped negative/dummy: {skipped_negative_dummy}",
        (
            "Counts: "
            f"reco-valid={reco_valid}, gen-valid={gen_valid}, "
            f"reco-only={reco_only}, gen-only={gen_only}, both={both}, neither={neither}"
        ),
        (
            "Fractions (of evaluated): "
            f"reco-only={100.0*reco_only/evaluated:.1f}%, "
            f"gen-only={100.0*gen_only/evaluated:.1f}%, "
            f"both={100.0*both/evaluated:.1f}%, "
            f"neither={100.0*neither/evaluated:.1f}%"
        ),
    ]
    if reco_only_examples:
        evidence.append("Examples where reco interpretation valid and gen fails:")
        evidence.extend([f"  {x}" for x in reco_only_examples])
    if gen_only_examples:
        evidence.append("Examples where gen interpretation valid and reco fails:")
        evidence.extend([f"  {x}" for x in gen_only_examples])
    if neither_examples:
        evidence.append("Examples where both interpretations fail:")
        evidence.extend([f"  {x}" for x in neither_examples])

    return TestResult(
        name="TEST 2: truth_triplet_0 index space test",
        result=result,
        reasoning=reasoning,
        evidence_lines=evidence,
        tag=tag,
    )


def run_test3(arrays: ak.Array, test2_tag: Optional[str]) -> TestResult:
    reasoning = (
        "Order/alignment check via order-invariant set overlap |set(reco_triplet_0)∩set(truth_triplet_0)| in {0,1,2,3}. "
        "Only meaningful if truth_triplet_0 is in reco-jet index space."
    )

    if test2_tag != "truth_triplet_index_space_reco":
        return TestResult(
            name="TEST 3: alignment test reco_triplet_0 vs truth_triplet_0",
            result="INCONCLUSIVE",
            reasoning=reasoning,
            evidence_lines=[
                "Test 2 does not indicate reco-jet index space with confidence.",
                "Overlap diagnosis is not meaningful under non-reco or ambiguous index-space semantics.",
            ],
            tag="alignment_unknown",
        )

    overlap_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    skipped = 0
    examples: List[str] = []

    n_events = len(arrays["reco_triplet_0"])
    for i in range(n_events):
        reco_idx = extract_int_triplet(arrays["reco_triplet_0"][i])
        truth_idx = extract_int_triplet(arrays["truth_triplet_0"][i])
        if reco_idx is None or truth_idx is None:
            skipped += 1
            continue
        if np.any(reco_idx < 0) or np.any(truth_idx < 0):
            skipped += 1
            continue

        n_jet = int(arrays["N_jet"][i])
        if not (np.all(reco_idx < n_jet) and np.all(truth_idx < n_jet)):
            skipped += 1
            continue

        ov = len(set(reco_idx.tolist()).intersection(set(truth_idx.tolist())))
        overlap_counts[ov] += 1
        if len(examples) < 5:
            examples.append(
                f"evt {i}: reco={reco_idx.tolist()}, truth={truth_idx.tolist()}, overlap={ov}"
            )

    total = sum(overlap_counts.values())
    if total == 0:
        return TestResult(
            name="TEST 3: alignment test reco_triplet_0 vs truth_triplet_0",
            result="INCONCLUSIVE",
            reasoning=reasoning,
            evidence_lines=[f"No valid events for overlap check. Skipped events: {skipped}"],
            tag="alignment_unknown",
        )

    frac3 = overlap_counts[3] / total
    frac01 = (overlap_counts[0] + overlap_counts[1]) / total

    if total < 5:
        result = "INCONCLUSIVE"
        tag = "alignment_unknown"
    elif frac3 >= 0.7:
        result = "PASS"
        tag = "alignment_yes"
    elif frac01 >= 0.7:
        result = "FAIL"
        tag = "alignment_no"
    else:
        result = "INCONCLUSIVE"
        tag = "alignment_mixed"

    evidence = [
        f"Valid overlap events: {total}, skipped: {skipped}",
        (
            "Overlap counts: "
            f"0->{overlap_counts[0]}, 1->{overlap_counts[1]}, 2->{overlap_counts[2]}, 3->{overlap_counts[3]}"
        ),
        (
            "Overlap fractions: "
            f"0->{100.0*overlap_counts[0]/total:.1f}%, "
            f"1->{100.0*overlap_counts[1]/total:.1f}%, "
            f"2->{100.0*overlap_counts[2]/total:.1f}%, "
            f"3->{100.0*overlap_counts[3]/total:.1f}%"
        ),
        "Example events:",
    ]
    evidence.extend([f"  {x}" for x in examples[:3]])

    return TestResult(
        name="TEST 3: alignment test reco_triplet_0 vs truth_triplet_0",
        result=result,
        reasoning=reasoning,
        evidence_lines=evidence,
        tag=tag,
    )


def final_summary(test1: TestResult, test2: TestResult, test3: TestResult) -> List[str]:
    lines: List[str] = []

    if test1.tag == "triplet0_from_reco_triplet0_yes":
        lines.append("- triplet_0_* is very likely built from reco_triplet_0 + jet_* (deterministic identity holds).")
    elif test1.tag == "triplet0_from_reco_triplet0_no":
        lines.append("- triplet_0_* does not appear to be built directly from reco_triplet_0 + jet_*.")
    else:
        lines.append("- triplet_0_* construction is inconclusive with tested events.")

    if test2.tag == "truth_triplet_index_space_reco":
        lines.append("- truth_triplet_0 likely indexes reco jets (jet_* index space).")
    elif test2.tag == "truth_triplet_index_space_gen":
        lines.append("- truth_triplet_0 likely indexes gen jets (genjet_* index space).")
    else:
        lines.append("- truth_triplet_0 index space remains ambiguous on tested events.")

    if test3.tag == "alignment_yes":
        lines.append("- truth_triplet_0 appears aligned with reco_triplet_0 in slot 0 (same jet set).")
    elif test3.tag == "alignment_no":
        lines.append("- truth_triplet_0 is not aligned with reco_triplet_0 in slot 0 (low set overlap).")
    elif test2.tag != "truth_triplet_index_space_reco":
        lines.append("- alignment test not meaningful because test 2 did not establish reco-jet index semantics.")
    else:
        lines.append("- alignment remains inconclusive with tested events.")

    return lines


def print_test(test: TestResult) -> None:
    print(test.name)
    print(f"Result: {test.result}")
    print(f"Reasoning: {test.reasoning}")
    print("Evidence:")
    for line in test.evidence_lines:
        print(line)
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic triplet interpretation diagnostics.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input ROOT file path.")
    parser.add_argument("--tree", default=None, help="Optional tree name.")
    parser.add_argument(
        "--max-events",
        type=parse_max_events,
        default=DEFAULT_MAX_EVENTS,
        help="Maximum events to process (default 20). Use None for all events.",
    )
    args = parser.parse_args()

    arrays, processed_events, total_entries, resolved_tree = load_arrays(
        filename=args.input,
        tree_name=args.tree,
        max_events=args.max_events,
    )

    test1 = run_test1(arrays)
    test2 = run_test2(arrays)
    test3 = run_test3(arrays, test2.tag)

    print("--------------------------------------------------")
    print("Triplet Interpretation Diagnostic Report")
    print(
        f"Processed events: {processed_events} "
        f"(max_events={args.max_events}, total_entries={total_entries}, tree={resolved_tree})"
    )
    print("Valid events used per test are reported in each Evidence block.")
    print("--------------------------------------------------")
    print("")

    print_test(test1)
    print_test(test2)
    print_test(test3)

    print("Final Interpretation Summary:")
    for line in final_summary(test1, test2, test3):
        print(line)
    print("")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
