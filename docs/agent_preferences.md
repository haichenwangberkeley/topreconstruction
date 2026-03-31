# Agent Preferences for Plotting

These are my preferred default plotting cuts and ranges so Copilot (or other agents)
can apply them automatically when generating plots.

Default numeric cutoffs:

- pt_max: 500  # GeV — all pT distributions should be limited to 0..500 GeV

Eta ranges:

- jet_eta_range: [-4.4, 4.4]  # jets: allow up to |eta| = 4.4
- photon_eta_range: [-2.5, 2.5]  # photons: restrict to tracker/acceptance region
- electron_eta_range: [-2.5, 2.5]
- muon_eta_range: [-2.5, 2.5]

Application notes for scripts (how to use these preferences):

- When producing pT histograms, set the x-axis limits to [0, pt_max]. Use overflow/underflow
  bins if needed but do not display points above `pt_max`.
- When producing eta histograms, apply the appropriate per-object-range before plotting.
  For jets use `jet_eta_range`, for photons/electrons/muons use their respective ranges.
- Leading-object plots (leading jet, leading photon, ...) should apply the same range and
  pt_max as the inclusive distributions.

File output and deployment:

- Produce PNGs and an `index.html` as the default output. Prefer `plots/` as the local
  output directory and then deploy to `/global/cfs/projectdirs/atlas/www/haichen/plots/` via
  `rsync` (see README.md) so I can view them via the portal.

If you want, I can modify `data_processing/plot_jets.py` (and other plotting scripts) to automatically
apply these ranges by default. Say "please apply preferences" and I'll update the scripts.

Cutflow / event storage preferences:

- Default jet selection for cutflow scripts: `pt > 25 GeV`, `|eta| < 2.5` (changeable via script args).
- When storing selected-event jets, store up to 10 jets per event in order of appearance and pad missing jets with zeros.
- Store per-jet quantities in the order `(pt, eta, phi, m)` and save as a single NumPy file `selected_jets.npy` with shape `(N_events_selected, 10, 4)`.

When asked to generate cutflow + event storage, the agent should run `data_processing/cutflow_and_store.py` and save outputs to the specified outdir. Document any deviations in the HTML or README output.

Triplet reconstruction notes for agents:

- Input: `selected_jets.npy` with shape `(N_events, 10, 4)` containing `(pt,eta,phi,m)`.
- For each event, ignore zero-padded jets (pt == 0) and consider all unordered 3-jet combinations.
- Select the triplet with the largest triplet pT and record `(pT, eta, phi, mass)` for the system.
- Output plots and files should be saved to `triplet_plots/` locally and can be deployed to
  `/global/cfs/projectdirs/atlas/www/haichen/plots/` using the same `rsync` + `chmod` steps in `README.md`.

Deployment guidance for agents:

- Use `--deploy` to copy plots to the public CFS folder. To avoid overwriting previous outputs
  provide `--task-name` and the agent will create a subdirectory `{task_name}_{timestamp}` under
  `/global/cfs/projectdirs/atlas/www/haichen/plots/`. If `--task-name` is omitted the agent will
  use `{timestamp}` as the subdirectory name.

Example agent command:

```
python data_processing/plot_jets.py ttbar.root --outdir plots --deploy --task-name myAnalysis
```

The agent should then report the public URL for the deployed folder:

```
https://portal.nersc.gov/cfs/atlas/haichen/plots/{task_or_timestamp}
```

Agent-run procedure (recommended):

1. Verify `cutflow_output/selected_jets.npy` exists and matches the expected shape.

2. Run (test subset):

```
python analysis/triplet_reco.py cutflow_output/selected_jets.npy --outdir triplet_plots --max-events 10000
```

3. Inspect `triplet_plots/index.html` locally (or deploy):

```
python analysis/triplet_reco.py cutflow_output/selected_jets.npy --outdir triplet_plots --deploy
```

Files produced:

- `triplet_pt.png`, `triplet_eta.png`, `triplet_phi.png`, `triplet_mass.png` — histograms of chosen triplet kinematics
- `triplet_plots/index.html` — simple viewer

Data conventions reminder for downstream agents:

- `selected_jets.npy` layout: `array[event, jet_index(0..9), feature(0..3)]`
  - `feature` order: 0=`pt` [GeV], 1=`eta`, 2=`phi` [rad], 3=`m` [GeV]
- Zero-padded jets have `pt==0` and should be ignored.
