# print_ttree_branches.py

Small helper to list branches (and their types) of the first TTree found (or a named tree)
in a ROOT file using only `uproot`.

Usage:

```bash
# Print branches (name : type) to stdout
python print_ttree_branches.py ttbar.root

# Save branches (name : type) to a text file
python print_ttree_branches.py ttbar.root --output branches.txt

# Specify a tree name
python print_ttree_branches.py ttbar.root --tree MyTree --output MyTree_branches.txt
```

Output is a plain text file with one branch name per line.

Place this script in the same directory as your ROOT files and run
from the command line. The script only depends on `uproot`.

Recommended install (virtualenv):

```bash
python -m pip install --user --upgrade pip
python -m pip install uproot
```

**Plotting**

- **Install:** Install the plotting dependencies (in your virtualenv):

```bash
python -m pip install uproot awkward numpy matplotlib
```

- **Run:** Generate PNG plots and an HTML viewer from a ROOT file:

```bash
# create plots/ and write PNG + index.html
python plot_jets.py ttbar.root --outdir plots
```

- **View:** Open the generated viewer in your desktop browser:

```bash
# from the workspace root
xdg-open plots/index.html
```

- **Output:** The script writes PNG files and `index.html` to the `plots/` directory. Click any image in the HTML to open the full PNG.

**Deploying plots to the public CFS directory**

After you generate plots with `plot_jets.py`, copy them to the public CFS folder so you can open them via the portal. Recommended, safe steps (do not use `777`):

```bash
# copy plots to the public web directory (preserves timestamps, removes deleted files)
rsync -av --delete plots/ /global/cfs/projectdirs/atlas/www/haichen/plots/

# directories: readable and traversable by the webserver
find /global/cfs/projectdirs/atlas/www/haichen/plots -type d -exec chmod 755 {} \;

# files: readable by others
find /global/cfs/projectdirs/atlas/www/haichen/plots -type f -exec chmod 644 {} \;

# ensure parent dirs are traversable
chmod +x /global/cfs/projectdirs/atlas/www/haichen
chmod +x /global/cfs/projectdirs/atlas/www
```

Confirm the `index.html` exists and is readable:

```bash
ls -l /global/cfs/projectdirs/atlas/www/haichen/plots/index.html
```

Example public URL (portal CFS view):

```
https://portal.nersc.gov/cfs/atlas/haichen/plots/index.html
```

Notes:
- Avoid `chmod 777` for security. The `755` (dirs) and `644` (files) pattern is usually sufficient.
- If you still cannot access the files after setting these permissions, the mount or ACLs may restrict access; contact your site admin.
- If you want me to copy and set permissions for you, run the command `please run` and I will execute the rsync+chmod steps here.


---

## Cutflow and event storage script

`cutflow_and_store.py` computes a simple jet-based cutflow and stores selected event
jet information for events with six or more selected jets.

Quick usage:

```bash
# run and save to cutflow_output/
python cutflow_and_store.py ttbar.root --outdir cutflow_output
```

Outputs written to `cutflow_output/`:

- `cutflow.csv` and `cutflow.txt` — the cutflow table
- `selected_jets.npy` — NumPy array shape `(N_selected_events, 10, 4)` containing (pt, eta, phi, m)

Defaults and options:
- Jet selection defaults: `pt > 25 GeV` and `|eta| < 2.5`. Change with `--ptmin` and `--etamax`.
- The script processes all events in the file using `uproot` + `awkward` (vectorized),
  pads each selected event to 10 jets (zeros for missing jets), and saves a single
    NumPy array containing all selected events.

New output archive:

After adding triplet-index storage the script now writes a compressed named archive
`selected_events.npz` in the output directory alongside the legacy `selected_jets.npy`.
This archive contains the following named arrays (shapes refer to `Nsel`, the number
of events that pass the selection):

- `selected_jets` : float array `(Nsel, 10, 4)` with last axis `(pt, eta, phi, m)` (legacy)
- `truth_triplet_0`, `truth_triplet_1`, `truth_triplet_2`, `truth_triplet_3` : int arrays `(Nsel, 3)`
        - Jet indices refer to the per-event ordering of selected jets (0..n_jets-1).
        - Missing triplets are padded with `-1`.
- `triplet_0_pt`, `triplet_0_eta`, `triplet_0_phi`, `triplet_0_m` : float arrays `(Nsel,)`
        - Kinematics of the leading (highest-pT) triplet per event; `0.0` if missing.
- `top_pt`, `top_eta`, `top_phi`, `top_m` : float arrays `(Nsel, 4)`
        - Up to four disjoint triplet candidates per event (columns correspond to triplet 0..3).
        - Missing entries are zero-filled.
- `top_PID` : int array `(Nsel, 4)` placeholder for particle ID.
    - Values: `6` for a reconstructed top candidate, `0` for missing entries.

Behavior:
- The triplet extraction is sequential and disjoint: the code finds the highest-pT
    3-jet system, removes its jets, then repeats up to four times. This creates up to
    four non-overlapping triplet candidates per event and is designed to be convenient
    for downstream per-event analyses.

Example: load the named arrays in Python

```python
import numpy as np
data = np.load('cutflow_output/selected_events.npz')
selected = data['selected_jets']         # (Nsel,10,4)
trip0_idx = data['truth_triplet_0']      # (Nsel,3)
top_pts = data['top_pt']                 # (Nsel,4)
```

See `cutflow_and_store.py` for implementation details.

## Triplet reconstruction (top candidate)

Use `triplet_reco.py` to process a saved `selected_jets.npy` file and reconstruct a
top-quark candidate per event by selecting the jet triplet with the highest triplet pT.

```bash
# generate triplet plots from the stored selected_jets.npy
python triplet_reco.py cutflow_output/selected_jets.npy --outdir triplet_plots
```

The script writes PNG histograms for the triplet-level `pT`, `eta`, `phi`, and `mass`.
Add `--max-events N` to test on a subset, and `--deploy` to copy the results to the public
web directory (`/global/cfs/projectdirs/atlas/www/haichen/plots/`).

Deployment with task names:

Both `plot_jets.py` and `triplet_reco.py` support an optional `--task-name` together with `--deploy`.
When `--deploy` is given the generated plots are copied into a timestamped subdirectory under
`/global/cfs/projectdirs/atlas/www/haichen/plots/`. If `--task-name` is provided the directory name
is `{task_name}_{timestamp}`, otherwise just `{timestamp}` is used. This avoids overwriting prior
results.

Example:

```bash
# deploy and group under task 'runA'
python triplet_reco.py cutflow_output/selected_jets.npy --outdir triplet_plots --max-events 10000 --deploy --task-name runA
```


Details and outputs:

- Input: `selected_jets.npy` with shape `(N_events, 10, 4)` where the last axis is `(pt, eta, phi, m)`.
- The script ignores zero-padded jets (pt == 0) and, per event, evaluates all unordered 3-jet
    combinations to find the triplet with the largest system `pT`.
- Outputs (written to the `--outdir` directory):
    - `triplet_pt.png`, `triplet_eta.png`, `triplet_phi.png`, `triplet_mass.png`
    - `index.html` (viewer for the four plots)

Example runs:

```bash
# run on all events (may take long if file is large)
python triplet_reco.py cutflow_output/selected_jets.npy --outdir triplet_plots

# test on the first 10k events
python triplet_reco.py cutflow_output/selected_jets.npy --outdir triplet_plots --max-events 10000

# run and deploy the plots to the public CFS folder (requires write access)
python triplet_reco.py cutflow_output/selected_jets.npy --outdir triplet_plots --deploy
```

Notes:
- The triplet selection uses event-level loops over combinations — it's correct but may be slow for very large `N_events`.
- The `--deploy` option rsyncs the generated plots to `/global/cfs/projectdirs/atlas/www/haichen/plots/` and sets safe permissions; see the README section on deployment for details.



