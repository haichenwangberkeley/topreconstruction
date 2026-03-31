#!/usr/bin/env python3
"""
triplet_reco.py

Read `selected_jets.npy` (N_events,10,4) with columns (pt,eta,phi,m),
find the jet triplet with the highest pT per event, and produce kinematic
histograms for the selected triplet system (pT, eta, phi, mass).

Usage:
  python analysis/triplet_reco.py cutflow_output/selected_jets.npy --outdir triplet_plots [--max-events N] [--deploy]

Options:
  --max-events : limit number of processed events (for testing)
  --deploy     : copy plots to public CFS dir (/global/cfs/projectdirs/atlas/www/haichen/plots)

Notes:
 - Jets with pt==0 are treated as padding and ignored.
 - The chosen candidate per event is the 3-jet combination with the highest pT.
 - Event-level loops are used; this is correct but may be slow for millions of events.
"""
import argparse
import os
import math
import itertools
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fourvec_from_ptetaphim(pt, eta, phi, m):
    """Return px,py,pz,E for given pt,eta,phi,m (arrays or scalars)."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    p2 = px * px + py * py + pz * pz
    E = np.sqrt(p2 + m * m)
    return px, py, pz, E


def inv_mass(E, px, py, pz):
    m2 = E * E - (px * px + py * py + pz * pz)
    m2 = np.where(m2 < 0, 0, m2)
    return np.sqrt(m2)


def process_event(jets):
    """Given jets array shape (10,4), return best triplet kinematics or (np.nan,...)."""
    # jets: Nx4 where N==10 padded with zeros
    pts = jets[:, 0]
    valid_idx = np.where(pts > 0)[0]
    if valid_idx.size < 3:
        return (np.nan, np.nan, np.nan, np.nan)

    best_pt = -1.0
    best_kin = (np.nan, np.nan, np.nan, np.nan)

    # iterate combinations
    for (i, j, k) in itertools.combinations(valid_idx, 3):
        pt_vals = jets[[i, j, k], 0]
        eta_vals = jets[[i, j, k], 1]
        phi_vals = jets[[i, j, k], 2]
        m_vals = jets[[i, j, k], 3]

        px, py, pz, E = fourvec_from_ptetaphim(pt_vals, eta_vals, phi_vals, m_vals)
        Px = px.sum()
        Py = py.sum()
        Pz = pz.sum()
        Esum = E.sum()

        cand_pt = math.hypot(Px, Py)
        if cand_pt > best_pt:
            best_pt = cand_pt
            cand_eta = 0.5 * math.log((Esum + Pz) / (Esum - Pz)) if (Esum != abs(Pz)) else np.nan
            cand_phi = math.atan2(Py, Px)
            cand_m = math.sqrt(max(Esum * Esum - (Px * Px + Py * Py + Pz * Pz), 0.0))
            best_kin = (cand_pt, cand_eta, cand_phi, cand_m)

    return best_kin


def plot_and_save(kin_array, outdir, prefix="triplet"):
    # kin_array shape (N,4)
    os.makedirs(outdir, exist_ok=True)
    pt = kin_array[:, 0]
    eta = kin_array[:, 1]
    phi = kin_array[:, 2]
    mass = kin_array[:, 3]

    # filters to finite values
    fin_pt = pt[np.isfinite(pt)]
    fin_eta = eta[np.isfinite(eta)]
    fin_phi = phi[np.isfinite(phi)]
    fin_mass = mass[np.isfinite(mass)]

    plt.figure()
    plt.hist(fin_pt, bins=50, histtype="stepfilled", alpha=0.8)
    plt.xlabel("triplet $p_T$ [GeV]")
    plt.ylabel("Events")
    plt.title("Selected triplet $p_T$")
    out_pt = os.path.join(outdir, f"{prefix}_pt.png")
    plt.tight_layout(); plt.savefig(out_pt, dpi=150); plt.close()

    plt.figure()
    plt.hist(fin_eta, bins=50, histtype="stepfilled", alpha=0.8)
    plt.xlabel("triplet $\\eta$")
    plt.ylabel("Events")
    plt.title("Selected triplet $\\eta$")
    out_eta = os.path.join(outdir, f"{prefix}_eta.png")
    plt.tight_layout(); plt.savefig(out_eta, dpi=150); plt.close()

    plt.figure()
    plt.hist(fin_phi, bins=50, histtype="stepfilled", alpha=0.8)
    plt.xlabel("triplet $\\phi$ (rad)")
    plt.ylabel("Events")
    plt.title("Selected triplet $\\phi$")
    out_phi = os.path.join(outdir, f"{prefix}_phi.png")
    plt.tight_layout(); plt.savefig(out_phi, dpi=150); plt.close()

    plt.figure()
    plt.hist(fin_mass, bins=50, histtype="stepfilled", alpha=0.8)
    plt.xlabel("triplet mass [GeV]")
    plt.ylabel("Events")
    plt.title("Selected triplet mass")
    out_mass = os.path.join(outdir, f"{prefix}_mass.png")
    plt.tight_layout(); plt.savefig(out_mass, dpi=150); plt.close()

    return [os.path.basename(out_pt), os.path.basename(out_eta), os.path.basename(out_phi), os.path.basename(out_mass)]


def deploy_plots(outdir, public_dir):
    # rsync and chmod similar to README
    cmd = [
        "rsync", "-av", "--delete", f"{outdir}/", f"{public_dir}/"
    ]
    subprocess.check_call(cmd)
    # set permissions
    subprocess.check_call("find {0} -type d -exec chmod 755 {{}} \\;".format(public_dir), shell=True)
    subprocess.check_call("find {0} -type f -exec chmod 644 {{}} \\;".format(public_dir), shell=True)


def main():
    p = argparse.ArgumentParser(description="Reconstruct highest-pT triplet top candidate per event")
    p.add_argument("input", help="input .npy file (selected_jets.npy)")
    p.add_argument("--outdir", "-o", default="triplet_plots", help="local output dir for plots")
    p.add_argument("--max-events", type=int, default=None, help="limit number of events to process (for testing)")
    p.add_argument("--deploy", action="store_true", help="deploy plots to public CFS dir /global/cfs/projectdirs/atlas/www/haichen/plots/")
    p.add_argument("--task-name", type=str, default=None, help="optional task name to group deployed plots (safe characters only)")
    args = p.parse_args()

    arr = np.load(args.input, mmap_mode='r')
    nevents = arr.shape[0]
    if args.max_events:
        nevents = min(nevents, args.max_events)

    kin = np.full((nevents, 4), np.nan, dtype=float)

    for i in range(nevents):
        jets = arr[i]
        kin[i] = process_event(jets)
        if (i + 1) % 10000 == 0:
            print(f"Processed {i+1}/{nevents} events", file=sys.stderr)

    files = plot_and_save(kin, args.outdir, prefix="triplet")
    # write index.html snippet for these plots
    html = ["<html>", "<body>", "<h1>Triplet plots</h1>", "<div>"]
    for f in files:
        html.append(f"<div style='display:inline-block;margin:8px;'><a href=\"{f}\"><img src=\"{f}\" style=\"width:300px\"/></a><div style='text-align:center'>{f}</div></div>")
    html.append("</div></body></html>")
    with open(os.path.join(args.outdir, "index.html"), "w") as fh:
        fh.write("\n".join(html))

    print(f"Wrote plots to {args.outdir}")

    if args.deploy:
        from datetime import datetime
        public_root = "/global/cfs/projectdirs/atlas/www/haichen/plots"
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        task = args.task_name.replace(" ", "_") if args.task_name else ts
        if args.task_name:
            subdir = f"{task}_{ts}"
        else:
            subdir = ts
        public = os.path.join(public_root, subdir)
        deploy_plots(args.outdir, public)
        print(f"Deployed plots to {public}")


if __name__ == "__main__":
    main()
