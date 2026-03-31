#!/usr/bin/env python3
"""
Print branch names of a TTree in a ROOT file using only uproot.

Usage:
    python data_processing/print_ttree_branches.py input.root [--tree TreeName] [--output out.txt]

The script finds the first TTree in the file (or the named tree) and
prints the branch names, one per line. If `--output` is provided the
branch list is written to the given text file instead of stdout.
"""
import argparse
import sys
import uproot


def find_ttree(f, tree_name=None):
    # If user provided a tree name, prefer that
    if tree_name:
        # try direct lookup
        try:
            if tree_name in f:
                return tree_name, f[tree_name]
        except Exception:
            pass
        # try matching without cycle suffix
        for k in f.keys():
            kname = k.name if hasattr(k, "name") else str(k)
            if kname.split(";")[0] == tree_name:
                return k, f[k]

    # Try classnames mapping (works with uproot >=4)
    try:
        classmap = f.classnames()
        for name, classname in classmap.items():
            if "TTree" in classname:
                return name, f[name]
    except Exception:
        pass

    # Fallback: iterate keys and pick first object that looks like a tree
    for k in f.keys():
        try:
            obj = f[k]
            if hasattr(obj, "keys") or hasattr(obj, "branches") or hasattr(obj, "arrays"):
                return k, obj
        except Exception:
            continue

    return None, None


def get_branches(tree):
    """
    Return a list of (branch_name, type_string) tuples.

    The function prefers to read a single entry (lightweight) to
    determine types. Falls back to reading zero entries or the full
    arrays if needed.
    """
    arrays = None
    for stop in (1, 0, None):
        try:
            arrays = tree.arrays(entry_stop=stop) if stop is not None else tree.arrays()
            break
        except Exception:
            arrays = None

    if arrays is None:
        return []

    # Avoid using truthiness on awkward arrays; check length where possible
    try:
        if hasattr(arrays, "__len__") and len(arrays) == 0:
            return []
    except Exception:
        pass

    # Obtain iterable of (name, array) pairs robustly
    items = None
    if isinstance(arrays, dict):
        items = arrays.items()
    else:
        fields = getattr(arrays, "fields", None)
        if fields:
            items = ((f, arrays[f]) for f in fields)
        else:
            try:
                items = dict(arrays).items()
            except Exception:
                try:
                    items = ((str(i), arrays[i]) for i in range(len(arrays)))
                except Exception:
                    return []

    branches = []
    for name, arr in items:
        type_str = None
        # numpy-style arrays have dtype
        try:
            dtype = getattr(arr, "dtype", None)
            if dtype is not None:
                type_str = str(dtype)
            else:
                try:
                    import awkward as ak
                    if isinstance(arr, ak.Array):
                        try:
                            type_str = str(arr.layout.type)
                        except Exception:
                            type_str = type(arr).__name__
                    else:
                        type_str = type(arr).__name__
                except Exception:
                    type_str = type(arr).__name__
        except Exception:
            try:
                type_str = type(arr).__name__
            except Exception:
                type_str = "unknown"

        branches.append((name, type_str))
    return branches


def main():
    p = argparse.ArgumentParser(description="Print TTree branch names using uproot only")
    p.add_argument("input", help="input ROOT file")
    p.add_argument("--tree", "-t", help="TTree name (optional)")
    p.add_argument("--output", "-o", help="write branches to this text file (optional)")
    args = p.parse_args()

    try:
        f = uproot.open(args.input)
    except Exception as e:
        print(f"Error opening file: {e}", file=sys.stderr)
        sys.exit(2)

    name, tree = find_ttree(f, args.tree)
    if tree is None:
        print("No TTree found in file.", file=sys.stderr)
        sys.exit(3)

    branches = get_branches(tree)
    if not branches:
        print("No branches found.", file=sys.stderr)
        sys.exit(4)

    def _format_line(item):
        n, t = item
        return f"{n} : {t}"

    if args.output:
        try:
            with open(args.output, "w") as f_out:
                for item in branches:
                    f_out.write(_format_line(item) + "\n")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(5)
    else:
        for item in branches:
            print(_format_line(item))


if __name__ == "__main__":
    main()
