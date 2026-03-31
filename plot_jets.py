#!/usr/bin/env python3
"""
Plot jet distributions from a ROOT file using only uproot.

Generates PNG files for:
 - overall jet `pt`, `eta`, `phi`
 - jet multiplicity
 - leading jet `pt`, `eta`, `phi`

Also writes `index.html` in the output directory to view the images.

Usage:
  python plot_jets.py input.root --outdir plots

Dependencies: uproot, awkward, numpy, matplotlib
"""
import argparse
import os
import sys

import uproot

def find_ttree(f):
    try:
        classmap = f.classnames()
        for name, classname in classmap.items():
            if "TTree" in classname:
                return f[name]
    except Exception:
        pass
    for k in f.keys():
        try:
            obj = f[k]
            # heuristic: tree-like objects support `arrays`
            if hasattr(obj, "arrays"):
                return obj
        except Exception:
            continue
    return None


def safe_mkdir(d):
    os.makedirs(d, exist_ok=True)


def make_hist(ax, data, bins=50, xlabel="", ylabel="Events", title=None, range=None):
    import numpy as np
    data = data[~(np.isnan(data))]
    ax.hist(data, bins=bins, range=range, histtype="stepfilled", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def write_html(outdir, sections, title="Jet distributions"):
    html = [
        "<html>",
        f"<head><meta charset=\"utf-8\"><title>{title}</title></head>",
        "<body>",
        f"<h1>{title}</h1>",
    ]

    for section_title, images in sections:
        html.append(f"<h2>{section_title}</h2>")
        html.append("<div style=\"display:flex;flex-wrap:wrap;gap:16px;\">")
        for img, caption in images:
            html.append("<div style=\"width:320px;\">")
            html.append(f"<a href=\"{img}\"><img src=\"{img}\" style=\"width:300px;display:block;\"/></a>")
            html.append(f"<div style=\"text-align:center;\">{caption}</div>")
            html.append("</div>")
        html.append("</div>")

    html.append("</body></html>")
    path = os.path.join(outdir, "index.html")
    with open(path, "w") as f:
        f.write("\n".join(html))
    return path


def main():
    p = argparse.ArgumentParser(description="Plot jet distributions from ROOT file (uproot)")
    p.add_argument("input", help="input ROOT file")
    p.add_argument("--outdir", "-o", default="plots", help="output directory for PNG and HTML")
    p.add_argument("--bins", type=int, default=50, help="number of histogram bins")
    p.add_argument("--deploy", action="store_true", help="deploy plots to public CFS dir /global/cfs/projectdirs/atlas/www/haichen/plots/")
    p.add_argument("--task-name", type=str, default=None, help="optional task name to group deployed plots (safe characters only)")
    args = p.parse_args()

    try:
        f = uproot.open(args.input)
    except Exception as e:
        print(f"Error opening file: {e}", file=sys.stderr)
        sys.exit(2)

    tree = find_ttree(f)
    if tree is None:
        print("No TTree found in file.", file=sys.stderr)
        sys.exit(3)

    # prefer to use explicit branches if available
    branches = [b for b in ("jet_pt", "jet_eta", "jet_phi", "N_jet", "genjet_pt", "genjet_eta", "genjet_phi", "N_genjet")]
    available = []
    for b in branches:
        try:
            if b in tree.keys():
                available.append(b)
        except Exception:
            # tree.keys() might not be present; try accessing
            try:
                _ = tree[b]
                available.append(b)
            except Exception:
                continue

    import awkward as ak
    import numpy as np

    # Preferences / clipping ranges (from agent.md):
    pt_max = 500.0
    jet_eta_range = (-4.4, 4.4)
    photon_eta_range = (-2.5, 2.5)
    electron_eta_range = (-2.5, 2.5)
    muon_eta_range = (-2.5, 2.5)

    # Read data for the available branches
    read_list = available[:]  # copy
    if not read_list:
        print("No jet branches found (jet_pt, jet_eta, jet_phi, N_jet).", file=sys.stderr)
        sys.exit(4)

    arrays = tree.arrays(read_list, library="ak")

    safe_mkdir(args.outdir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jet_images = []
    gen_images = []

    # overall jet pt/eta/phi (flattened) with clipping
    if "jet_pt" in arrays.fields:
        jet_pt = ak.to_numpy(ak.flatten(arrays["jet_pt"]))
        # apply pt_max
        jet_pt = jet_pt[(jet_pt >= 0) & (jet_pt <= pt_max)]
        fig, ax = plt.subplots()
        make_hist(ax, jet_pt, bins=args.bins, xlabel="jet $p_T$ [GeV]", title="Jet $p_T$ (all jets)")
        ax.set_xlim(0, pt_max)
        out = os.path.join(args.outdir, "jet_pt.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        jet_images.append((os.path.basename(out), "Jet $p_T$ (all jets)"))

    if "jet_eta" in arrays.fields:
        jet_eta = ak.to_numpy(ak.flatten(arrays["jet_eta"]))
        # restrict to jet_eta_range
        jet_eta = jet_eta[(jet_eta >= jet_eta_range[0]) & (jet_eta <= jet_eta_range[1])]
        fig, ax = plt.subplots()
        make_hist(ax, jet_eta, bins=args.bins, xlabel="jet $\\eta$", title="Jet $\\eta$ (all jets)")
        ax.set_xlim(jet_eta_range)
        out = os.path.join(args.outdir, "jet_eta.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        jet_images.append((os.path.basename(out), "Jet $\\eta$ (all jets)"))

    if "jet_phi" in arrays.fields:
        jet_phi = ak.to_numpy(ak.flatten(arrays["jet_phi"]))
        fig, ax = plt.subplots()
        make_hist(ax, jet_phi, bins=args.bins, xlabel="jet $\\phi$ (rad)", title="Jet $\\phi$ (all jets)")
        out = os.path.join(args.outdir, "jet_phi.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        jet_images.append((os.path.basename(out), "Jet $\\phi$ (all jets)"))

    # jet multiplicity
    if "N_jet" in arrays.fields:
        njet = ak.to_numpy(arrays["N_jet"])
    else:
        # try derive from jet_pt
        if "jet_pt" in arrays.fields:
            njet = ak.to_numpy(ak.num(arrays["jet_pt"]))
        else:
            njet = None

    if njet is not None:
        fig, ax = plt.subplots()
        make_hist(ax, njet, bins=range(0, int(np.nanmax(njet) + 2)), xlabel="Number of jets", title="Jet multiplicity")
        out = os.path.join(args.outdir, "jet_multiplicity.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        jet_images.append((os.path.basename(out), "Jet multiplicity"))

    # leading jet = first element in jet arrays
    if "jet_pt" in arrays.fields:
        # mask events with at least one jet, apply pt_max
        first_pt = ak.to_numpy(ak.fill_none(ak.firsts(arrays["jet_pt"]), np.nan))
        first_pt = first_pt[(~np.isnan(first_pt)) & (first_pt <= pt_max)]
        fig, ax = plt.subplots()
        make_hist(ax, first_pt, bins=args.bins, xlabel="leading jet $p_T$ [GeV]", title="Leading jet $p_T$")
        ax.set_xlim(0, pt_max)
        out = os.path.join(args.outdir, "leading_jet_pt.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        jet_images.append((os.path.basename(out), "Leading jet $p_T$"))

    if "jet_eta" in arrays.fields:
        first_eta = ak.to_numpy(ak.fill_none(ak.firsts(arrays["jet_eta"]), np.nan))
        first_eta = first_eta[(~np.isnan(first_eta)) & (first_eta >= jet_eta_range[0]) & (first_eta <= jet_eta_range[1])]
        fig, ax = plt.subplots()
        make_hist(ax, first_eta, bins=args.bins, xlabel="leading jet $\\eta$", title="Leading jet $\\eta$")
        ax.set_xlim(jet_eta_range)
        out = os.path.join(args.outdir, "leading_jet_eta.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        jet_images.append((os.path.basename(out), "Leading jet $\\eta$"))

    if "jet_phi" in arrays.fields:
        first_phi = ak.to_numpy(ak.fill_none(ak.firsts(arrays["jet_phi"]), np.nan))
        first_phi = first_phi[~np.isnan(first_phi)]
        fig, ax = plt.subplots()
        make_hist(ax, first_phi, bins=args.bins, xlabel="leading jet $\\phi$ (rad)", title="Leading jet $\\phi$")
        out = os.path.join(args.outdir, "leading_jet_phi.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        jet_images.append((os.path.basename(out), "Leading jet $\\phi$") )

    # === genjet (truth-level) plots: genjet_pt, genjet_eta, genjet_phi, N_genjet ===
    if any(x in arrays.fields for x in ("genjet_pt", "genjet_eta", "genjet_phi", "N_genjet")):
        # overall genjet pt/eta/phi
        if "genjet_pt" in arrays.fields:
            genjet_pt = ak.to_numpy(ak.flatten(arrays["genjet_pt"]))
            genjet_pt = genjet_pt[(genjet_pt >= 0) & (genjet_pt <= pt_max)]
            fig, ax = plt.subplots()
            make_hist(ax, genjet_pt, bins=args.bins, xlabel="genjet $p_T$ [GeV]", title="GenJet $p_T$ (all genjets)")
            ax.set_xlim(0, pt_max)
            out = os.path.join(args.outdir, "genjet_pt.png")
            fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
            gen_images.append((os.path.basename(out), "GenJet $p_T$ (all genjets)"))

        if "genjet_eta" in arrays.fields:
            genjet_eta = ak.to_numpy(ak.flatten(arrays["genjet_eta"]))
            genjet_eta = genjet_eta[(genjet_eta >= jet_eta_range[0]) & (genjet_eta <= jet_eta_range[1])]
            fig, ax = plt.subplots()
            make_hist(ax, genjet_eta, bins=args.bins, xlabel="genjet $\\eta$", title="GenJet $\\eta$ (all genjets)")
            ax.set_xlim(jet_eta_range)
            out = os.path.join(args.outdir, "genjet_eta.png")
            fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
            gen_images.append((os.path.basename(out), "GenJet $\\eta$ (all genjets)"))

        if "genjet_phi" in arrays.fields:
            genjet_phi = ak.to_numpy(ak.flatten(arrays["genjet_phi"]))
            fig, ax = plt.subplots()
            make_hist(ax, genjet_phi, bins=args.bins, xlabel="genjet $\\phi$ (rad)", title="GenJet $\\phi$ (all genjets)")
            out = os.path.join(args.outdir, "genjet_phi.png")
            fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
            gen_images.append((os.path.basename(out), "GenJet $\\phi$ (all genjets)"))

        # genjet multiplicity
        if "N_genjet" in arrays.fields:
            ngenjet = ak.to_numpy(arrays["N_genjet"])
        else:
            if "genjet_pt" in arrays.fields:
                ngenjet = ak.to_numpy(ak.num(arrays["genjet_pt"]))
            else:
                ngenjet = None

        if ngenjet is not None:
            fig, ax = plt.subplots()
            make_hist(ax, ngenjet, bins=range(0, int(np.nanmax(ngenjet) + 2)), xlabel="Number of genjets", title="GenJet multiplicity")
            out = os.path.join(args.outdir, "genjet_multiplicity.png")
            fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
            gen_images.append((os.path.basename(out), "GenJet multiplicity"))

        # leading genjet
        if "genjet_pt" in arrays.fields:
            first_gpt = ak.to_numpy(ak.fill_none(ak.firsts(arrays["genjet_pt"]), np.nan))
            first_gpt = first_gpt[(~np.isnan(first_gpt)) & (first_gpt <= pt_max)]
            fig, ax = plt.subplots()
            make_hist(ax, first_gpt, bins=args.bins, xlabel="leading genjet $p_T$ [GeV]", title="Leading GenJet $p_T$")
            ax.set_xlim(0, pt_max)
            out = os.path.join(args.outdir, "leading_genjet_pt.png")
            fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
            gen_images.append((os.path.basename(out), "Leading GenJet $p_T$"))

        if "genjet_eta" in arrays.fields:
            first_geta = ak.to_numpy(ak.fill_none(ak.firsts(arrays["genjet_eta"]), np.nan))
            first_geta = first_geta[(~np.isnan(first_geta)) & (first_geta >= jet_eta_range[0]) & (first_geta <= jet_eta_range[1])]
            fig, ax = plt.subplots()
            make_hist(ax, first_geta, bins=args.bins, xlabel="leading genjet $\\eta$", title="Leading GenJet $\\eta$")
            ax.set_xlim(jet_eta_range)
            out = os.path.join(args.outdir, "leading_genjet_eta.png")
            fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
            gen_images.append((os.path.basename(out), "Leading GenJet $\\eta$"))

        if "genjet_phi" in arrays.fields:
            first_gphi = ak.to_numpy(ak.fill_none(ak.firsts(arrays["genjet_phi"]), np.nan))
            first_gphi = first_gphi[~np.isnan(first_gphi)]
            fig, ax = plt.subplots()
            make_hist(ax, first_gphi, bins=args.bins, xlabel="leading genjet $\\phi$ (rad)", title="Leading GenJet $\\phi$")
            out = os.path.join(args.outdir, "leading_genjet_phi.png")
            fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
            gen_images.append((os.path.basename(out), "Leading GenJet $\\phi$"))

    # write HTML with separate sections for reconstructed jets and generator jets
    sections = []
    if jet_images:
        sections.append(("Reconstructed jets", jet_images))
    if gen_images:
        sections.append(("Generator jets (truth)", gen_images))
    html_path = write_html(args.outdir, sections, title=f"Jet plots for {os.path.basename(args.input)}")
    print(f"Wrote plots and HTML to: {args.outdir}")
    print(f"Open {html_path} in a web browser to view the plots.")
    # optional deploy
    if getattr(args, "deploy", False):
        from datetime import datetime
        public_root = "/global/cfs/projectdirs/atlas/www/haichen/plots"
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        task = args.task_name.replace(" ", "_") if args.task_name else ts
        if args.task_name:
            subdir = f"{task}_{ts}"
        else:
            subdir = ts
        public_dir = os.path.join(public_root, subdir)
        # rsync and set permissions
        import subprocess
        subprocess.check_call(["rsync", "-av", "--delete", f"{args.outdir}/", f"{public_dir}/"]) 
        subprocess.check_call("find {0} -type d -exec chmod 755 {{}} \\;".format(public_dir), shell=True)
        subprocess.check_call("find {0} -type f -exec chmod 644 {{}} \\;".format(public_dir), shell=True)
        print(f"Deployed plots to {public_dir}")


if __name__ == "__main__":
    main()
