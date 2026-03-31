# Data Processing Subproject

This subproject contains input sample processing and ROOT diagnostics scripts.

## Canonical scripts

- `print_ttree_branches.py`: list TTree branch names.
- `inspect_ttbar.py`: inspect key branches in the ttbar sample.
- `cutflow_and_store.py`: cutflow and event storage conversion utility.
- `inspect_selected_events.py`: inspect stored selected-event outputs.

## Notes

Root-level files with the same names are compatibility wrappers and forward to
these canonical script locations.
