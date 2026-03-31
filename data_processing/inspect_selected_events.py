#!/usr/bin/env python3
"""
inspect_selected_events.py

Quick script to report number of entries and variables in a .npz file.
Usage: python data_processing/inspect_selected_events.py [path/to/file.npz]
"""
import argparse
import os
import sys
import numpy as np


def inspect_file(fp: str) -> int:
    if not os.path.exists(fp):
        print(f"File not found: {fp}", file=sys.stderr)
        return 2

    print(f"File: {fp}")
    print(f"Size (bytes): {os.path.getsize(fp):,}")

    try:
        with np.load(fp, allow_pickle=True) as data:
            keys = list(data.keys())
            if not keys:
                print("No variables found in the NPZ.")
                return 0

            print("Variables:")
            entry_counts = {}
            for k in keys:
                arr = data[k]
                shape = getattr(arr, 'shape', None)
                dtype = getattr(arr, 'dtype', None)
                entries = shape[0] if (shape and len(shape) > 0) else None
                entry_counts[k] = entries
                print(f" - {k}: shape={shape}, dtype={dtype}, entries={entries}")

            counts = [v for v in entry_counts.values() if v is not None]
            if counts:
                uniq = set(counts)
                if len(uniq) == 1:
                    print(f"Number of entries (consistent across array-like variables): {counts[0]:,}")
                else:
                    print("Number of entries varies between variables:")
                    for k, v in entry_counts.items():
                        print(f"  {k}: {v}")
            else:
                print("No array-like variables with a leading dimension to report entries for.")

    except Exception as e:
        print(f"Error loading NPZ: {e}", file=sys.stderr)
        return 3

    return 0


def main():
    p = argparse.ArgumentParser(description='Inspect a selected_events .npz file')
    p.add_argument('file', nargs='?', default='cutflow_output/selected_events.npz', help='Path to .npz file')
    args = p.parse_args()
    sys.exit(inspect_file(args.file))


if __name__ == '__main__':
    main()
