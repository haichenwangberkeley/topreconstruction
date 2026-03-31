#!/usr/bin/env python3
"""
cutflow_and_store.py

Reads a ROOT file with jet branches, produces a cutflow table and stores selected
event jet information. In addition to the per-jet storage (pT, eta, phi, m) this
script extracts up to four non-overlapping 3-jet "top" triplets per event and
stores their jet indices and kinematic properties for convenient downstream
analysis.

Outputs (written to --outdir):
 - cutflow.csv / cutflow.txt
 - selected_jets.npy            : legacy backward-compatible (Nsel,10,4)
 - selected_events.npz         : compressed archive with named arrays (recommended)

Named arrays inside selected_events.npz:
 - selected_jets           : float array (Nsel,10,4) with last axis (pt,eta,phi,m)
 - truth_triplet_0..3      : int array (Nsel,3) jet indices for up to 4 triplets (-1 padded)
 - triplet_0_pt/eta/phi/m  : float arrays (Nsel,) for the leading triplet (0 if missing)
 - top_pt/eta/phi/m        : float arrays (Nsel,4) per-event up to 4 top candidates (0 if missing)
 - top_PID                 : int array (Nsel,4) placeholder for PID (filled with -1)

Notes:
 - The triplet-finding algorithm selects the highest-pT 3-jet system, removes
   its constituent jets, and repeats up to four times to produce disjoint triplets.
 - Jet indices refer to the per-event ordering of selected jets (0..n_jets-1).
 - Missing values are padded with -1 for integer indices and 0.0 for floats.
"""
import argparse
import os
import sys
import csv
from datetime import datetime
import itertools

import uproot
import awkward as ak
import numpy as np


def ensure_outdir(d):
    os.makedirs(d, exist_ok=True)


def write_cutflow(outdir, cutflow):
    csv_path = os.path.join(outdir, "cutflow.csv")
    txt_path = os.path.join(outdir, "cutflow.txt")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "count"]) 
        for step, count in cutflow:
            w.writerow([step, int(count)])

    # human readable
    with open(txt_path, "w") as f:
        f.write(f"Cutflow generated: {datetime.utcnow().isoformat()} UTC\n")
        maxlen = max(len(s) for s, _ in cutflow)
        for step, count in cutflow:
            f.write(f"{step.ljust(maxlen)} : {int(count)}\n")


def format_cutflow_print(cutflow):
    maxlen = max(len(s) for s, _ in cutflow)
    lines = ["Cutflow:"]
    for step, count in cutflow:
        lines.append(f"{step.ljust(maxlen)} : {int(count)}")
    return "\n".join(lines)


def fourvec_from_pt_eta_phi_m(pt, eta, phi, m):
    # returns px, py, pz, E
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    p = pt * np.cosh(eta)
    E = np.sqrt(np.maximum(m * m + p * p, 0.0))
    return px, py, pz, E


def compute_system_kinematics(jet_pts, jet_etas, jet_phis, jet_ms, indices):
    # indices: iterable of integer indices into the per-event jet arrays
    px_sum = 0.0
    py_sum = 0.0
    pz_sum = 0.0
    E_sum = 0.0
    for i in indices:
        px, py, pz, E = fourvec_from_pt_eta_phi_m(jet_pts[i], jet_etas[i], jet_phis[i], jet_ms[i])
        px_sum += px
        py_sum += py
        pz_sum += pz
        E_sum += E
    pt_sys = np.hypot(px_sum, py_sum)
    p_sys = np.sqrt(max(px_sum * px_sum + py_sum * py_sum + pz_sum * pz_sum, 0.0))
    # safe mass
    m2 = max(E_sum * E_sum - p_sys * p_sys, 0.0)
    mass_sys = np.sqrt(m2)
    # eta: use asinh(pz / pt)
    eta_sys = np.arcsinh(pz_sum / max(pt_sys, 1e-12))
    phi_sys = np.arctan2(py_sum, px_sum)
    return pt_sys, eta_sys, phi_sys, mass_sys


def main():
    p = argparse.ArgumentParser(description="Produce cutflow and store jet info for events with >=6 jets")
    p.add_argument("input", help="input ROOT file")
    p.add_argument("--outdir", "-o", default="cutflow_output", help="output directory")
    p.add_argument("--ptmin", type=float, default=25.0, help="jet pT threshold [GeV]")
    p.add_argument("--etamax", type=float, default=2.5, help="jet |eta| maximum for selection")
    args = p.parse_args()

    try:
        f = uproot.open(args.input)
    except Exception as e:
        print(f"Error opening file: {e}", file=sys.stderr)
        sys.exit(2)

    # find a tree
    tree = None
    try:
        classmap = f.classnames()
        for name, classname in classmap.items():
            if "TTree" in classname:
                tree = f[name]
                break
    except Exception:
        pass
    if tree is None:
        for k in f.keys():
            try:
                obj = f[k]
                if hasattr(obj, "arrays"):
                    tree = obj
                    break
            except Exception:
                continue

    if tree is None:
        print("No TTree found in file.", file=sys.stderr)
        sys.exit(3)

    # read required branches
    needed = ["jet_pt", "jet_eta", "jet_phi", "jet_m"]
    arrays = tree.arrays(needed, library="ak")

    # Number of events
    nevents = len(arrays)

    # build jet record
    jets = ak.zip({"pt": arrays["jet_pt"], "eta": arrays["jet_eta"], "phi": arrays["jet_phi"], "m": arrays["jet_m"]})

    # jet selection (vectorized)
    sel_mask = (jets.pt > args.ptmin) & (np.abs(jets.eta) < args.etamax)
    sel_jets = jets[sel_mask]
    nsel = ak.num(sel_jets)

    # cutflow counts
    cutflow = []
    cutflow.append(("Initial events", nevents))
    cutflow.append((">=1 selected jet", int(ak.sum(nsel >= 1))))
    cutflow.append((">=3 selected jets", int(ak.sum(nsel >= 3))))
    cutflow.append((">=6 selected jets", int(ak.sum(nsel >= 6))))

    # print and save cutflow
    ensure_outdir(args.outdir)
    print(format_cutflow_print(cutflow))
    write_cutflow(args.outdir, cutflow)

    # For events with >=6 selected jets, operate on the jagged selected_events
    sel_events_mask = (nsel >= 6)
    selected_events = sel_jets[sel_events_mask]

    if len(selected_events) == 0:
        print("No events with >=6 selected jets — nothing to store.")
        # still write empty outputs
        np.save(os.path.join(args.outdir, "selected_jets.npy"), np.zeros((0, 10, 4), dtype=float))
        np.savez_compressed(os.path.join(args.outdir, "selected_events.npz"))
        return

    Nsel = len(selected_events)

    # prepare legacy padded arrays (Nsel,10)
    first10 = selected_events[:, :10]
    pad_pt = ak.fill_none(ak.pad_none(first10.pt, 10, axis=1)[:, :10], 0.0)
    pad_eta = ak.fill_none(ak.pad_none(first10.eta, 10, axis=1)[:, :10], 0.0)
    pad_phi = ak.fill_none(ak.pad_none(first10.phi, 10, axis=1)[:, :10], 0.0)
    pad_m = ak.fill_none(ak.pad_none(first10.m, 10, axis=1)[:, :10], 0.0)

    np_pt = ak.to_numpy(pad_pt)
    np_eta = ak.to_numpy(pad_eta)
    np_phi = ak.to_numpy(pad_phi)
    np_m = ak.to_numpy(pad_m)
    stacked = np.stack([np_pt, np_eta, np_phi, np_m], axis=-1)

    # Prepare arrays to hold up to 4 disjoint triplets per event
    truth_triplets = np.full((Nsel, 4, 3), -1, dtype=np.int32)
    top_pt = np.zeros((Nsel, 4), dtype=float)
    top_eta = np.zeros((Nsel, 4), dtype=float)
    top_phi = np.zeros((Nsel, 4), dtype=float)
    top_m = np.zeros((Nsel, 4), dtype=float)
    # top_PID: 0 means missing, 6 is used as a simple label for a reconstructed top
    top_PID = np.zeros((Nsel, 4), dtype=np.int32)

    # iterate events and find up to 4 disjoint highest-pT triplets
    for ievt in range(Nsel):
        ev = selected_events[ievt]
        jet_pts = ak.to_numpy(ev.pt)
        jet_etas = ak.to_numpy(ev.eta)
        jet_phis = ak.to_numpy(ev.phi)
        jet_ms = ak.to_numpy(ev.m)
        nj = len(jet_pts)
        if nj < 3:
            continue

        available = list(range(nj))
        for itrip in range(4):
            if len(available) < 3:
                break
            best_pt = -1.0
            best_comb = None
            # enumerate all combinations from available jets
            for comb in itertools.combinations(available, 3):
                pt_sys, eta_sys, phi_sys, mass_sys = compute_system_kinematics(jet_pts, jet_etas, jet_phis, jet_ms, comb)
                if pt_sys > best_pt:
                    best_pt = pt_sys
                    best_comb = (comb, pt_sys, eta_sys, phi_sys, mass_sys)
            if best_comb is None:
                break
            comb, pt_sys, eta_sys, phi_sys, mass_sys = best_comb
            # store indices (relative to per-event selected jets)
            truth_triplets[ievt, itrip, :len(comb)] = np.array(comb, dtype=np.int32)
            top_pt[ievt, itrip] = pt_sys
            top_eta[ievt, itrip] = eta_sys
            top_phi[ievt, itrip] = phi_sys
            top_m[ievt, itrip] = mass_sys
            # label as a reconstructed top candidate (simple placeholder PID)
            top_PID[ievt, itrip] = 6
            # remove used jets from available
            for j in comb:
                available.remove(j)

    # Also expose triplet_0_* convenience arrays (leading triplet only)
    triplet_0_pt = top_pt[:, 0].copy()
    triplet_0_eta = top_eta[:, 0].copy()
    triplet_0_phi = top_phi[:, 0].copy()
    triplet_0_m = top_m[:, 0].copy()

    # save legacy npy and a compressed npz with named arrays
    out_npy = os.path.join(args.outdir, "selected_jets.npy")
    np.save(out_npy, stacked)

    out_npz = os.path.join(args.outdir, "selected_events.npz")
    np.savez_compressed(
        out_npz,
        selected_jets=stacked,
        truth_triplet_0=truth_triplets[:, 0, :],
        truth_triplet_1=truth_triplets[:, 1, :],
        truth_triplet_2=truth_triplets[:, 2, :],
        truth_triplet_3=truth_triplets[:, 3, :],
        triplet_0_pt=triplet_0_pt,
        triplet_0_eta=triplet_0_eta,
        triplet_0_phi=triplet_0_phi,
        triplet_0_m=triplet_0_m,
        top_pt=top_pt,
        top_eta=top_eta,
        top_phi=top_phi,
        top_m=top_m,
        top_PID=top_PID,
    )

    print(f"Saved selected jets (legacy) {stacked.shape} -> {out_npy}")
    print(f"Saved named arrays to: {out_npz}")


if __name__ == "__main__":
    main()
