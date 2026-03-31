#!/usr/bin/env python3
"""Stage 1: ROOT ntuple -> triplets_raw.parquet using streaming writes."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple

import awkward as ak
import numpy as np
import uproot

import features
import triplet_io as pio


TRUTH_BRANCHES = (
    "truth_triplet_0",
    "truth_triplet_1",
    "truth_triplet_2",
    "truth_triplet_3",
)

BASE_BRANCHES = (
    "N_genjet",
    "genjet_pt",
    "genjet_eta",
    "genjet_phi",
    *TRUTH_BRANCHES,
)


def _parse_triplet(raw: ak.Array) -> Optional[Tuple[int, int, int]]:
    try:
        arr = np.asarray(ak.to_list(raw), dtype=np.int64)
    except Exception:
        return None
    if arr.shape != (3,):
        return None
    return int(arr[0]), int(arr[1]), int(arr[2])


def _event_id_from_branch(raw_value: ak.Array) -> int:
    if np.isscalar(raw_value):
        return int(raw_value)

    try:
        arr = np.asarray(ak.to_list(raw_value), dtype=np.int64)
    except Exception:
        arr = np.asarray(raw_value, dtype=np.int64)

    if arr.ndim == 0:
        return int(arr.item())
    if arr.size == 1:
        return int(arr.reshape(-1)[0])
    raise ValueError(f"Event ID branch value has unexpected shape: {arr.shape}")


def _flush_buffer(writer: pio.StreamingParquetWriter, buffer: pio.ColumnBuffer) -> None:
    payload = buffer.take_all()
    writer.write_rows(payload)


def run(args: argparse.Namespace) -> None:
    if args.max_events is not None and args.max_events <= 0:
        raise ValueError("--max-events must be > 0 when provided")

    np.random.seed(args.seed)

    output_dir = pio.ensure_dir(args.output_dir)
    output_path = Path(args.output_file) if args.output_file else output_dir / "triplets_raw.parquet"

    tree_paths = pio.make_tree_paths(args.inputs, args.tree_name)

    with uproot.open(tree_paths[0]) as tree:
        available_branches = set(tree.keys())

    use_event_branch: Optional[str]
    if args.event_id_branch in available_branches:
        use_event_branch = args.event_id_branch
    else:
        use_event_branch = None

    branches = list(BASE_BRANCHES)
    if use_event_branch is not None:
        branches.append(use_event_branch)

    writer = pio.StreamingParquetWriter(
        output_path=output_path,
        schema=pio.triplet_schema(include_score=False),
        row_group_size=args.row_group_size,
    )
    buffer = pio.ColumnBuffer([field.name for field in pio.triplet_schema(include_score=False)])

    processed_events = 0
    processed_candidates = 0
    truth_triplets_total = 0
    unordered_truth_examples = 0
    fallback_event_id = 0
    stop = False

    for chunk in uproot.iterate(tree_paths, branches, step_size=args.step_size, library="ak"):
        n_chunk_events = len(chunk["N_genjet"])
        for local_idx in range(n_chunk_events):
            if args.max_events is not None and processed_events >= args.max_events:
                stop = True
                break

            n_genjet = int(chunk["N_genjet"][local_idx])
            if use_event_branch is not None:
                event_id = _event_id_from_branch(chunk[use_event_branch][local_idx])
            else:
                event_id = fallback_event_id
                fallback_event_id += 1

            genjet_pt = np.asarray(ak.to_numpy(chunk["genjet_pt"][local_idx]), dtype=np.float64)[:n_genjet]
            genjet_eta = np.asarray(ak.to_numpy(chunk["genjet_eta"][local_idx]), dtype=np.float64)[:n_genjet]
            genjet_phi = np.asarray(ak.to_numpy(chunk["genjet_phi"][local_idx]), dtype=np.float64)[:n_genjet]

            truth_set: Set[Tuple[int, int, int]] = set()
            for truth_branch in TRUTH_BRANCHES:
                parsed = _parse_triplet(chunk[truth_branch][local_idx])
                if parsed is None:
                    continue
                if len(set(parsed)) != 3:
                    continue
                if min(parsed) < 0 or max(parsed) >= n_genjet:
                    continue
                sorted_triplet = tuple(sorted(parsed))
                if parsed != sorted_triplet:
                    unordered_truth_examples += 1
                truth_set.add(sorted_triplet)
                truth_triplets_total += 1

            expected = math.comb(n_genjet, 3) if n_genjet >= 3 else 0
            produced = 0

            for i, j, k in itertools.combinations(range(n_genjet), 3):
                produced += 1
                payload = features.compute_triplet_feature_payload(genjet_pt, genjet_eta, genjet_phi, i, j, k)
                features.assert_feature_values_sane([payload[name] for name in features.FEATURE_COLUMNS])

                row = {
                    "event_id": int(event_id),
                    "i": int(i),
                    "j": int(j),
                    "k": int(k),
                    "dr_ab": float(payload["dr_ab"]),
                    "dr_ac": float(payload["dr_ac"]),
                    "dr_bc": float(payload["dr_bc"]),
                    "mij_over_m123_ab": float(payload["mij_over_m123_ab"]),
                    "mij_over_m123_ac": float(payload["mij_over_m123_ac"]),
                    "mij_over_m123_bc": float(payload["mij_over_m123_bc"]),
                    "m123": float(payload["m123"]),
                    "mij_ab": float(payload["mij_ab"]),
                    "mij_ac": float(payload["mij_ac"]),
                    "mij_bc": float(payload["mij_bc"]),
                    "triplet_pt": float(payload["triplet_pt"]),
                    "triplet_eta": float(payload["triplet_eta"]),
                    "triplet_phi": float(payload["triplet_phi"]),
                    "is_truth": int((i, j, k) in truth_set),
                }
                buffer.append_row(row)
                processed_candidates += 1

                if buffer.size >= args.flush_rows:
                    _flush_buffer(writer, buffer)

            if produced != expected:
                raise RuntimeError(
                    f"Candidate enumeration mismatch for event_id={event_id}: produced={produced}, expected={expected}"
                )

            processed_events += 1

        if stop:
            break

    if buffer.size > 0:
        _flush_buffer(writer, buffer)

    writer.close()

    report = {
        "schema_version": pio.SCHEMA_VERSION,
        "processed_events": processed_events,
        "processed_candidate_triplets": processed_candidates,
        "N_truth_triplet_total": truth_triplets_total,
        "unordered_truth_triplet_examples": unordered_truth_examples,
        "event_id_branch": use_event_branch if use_event_branch is not None else "entry_index_fallback",
        "output_file": str(output_path),
        "rows_written": writer.rows_written,
    }
    pio.write_json(output_dir / "dataset_build_report.json", report)

    pio.write_config_snapshot(
        output_dir=output_dir,
        stage="dataset_build",
        input_files=list(args.inputs),
        parameters={
            "tree_name": args.tree_name,
            "event_id_branch_requested": args.event_id_branch,
            "event_id_branch_used": use_event_branch,
            "max_events": args.max_events,
            "step_size": args.step_size,
            "row_group_size": args.row_group_size,
            "flush_rows": args.flush_rows,
        },
        seed=args.seed,
    )


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("dataset_build", help="Stage 1: build triplet-level dataset from ROOT")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input ROOT ntuple files")
    parser.add_argument("--tree-name", default=None, help="TTree name; auto-detected if omitted")
    parser.add_argument("--event-id-branch", default="Number", help="Branch to use as event_id")
    parser.add_argument("--output-dir", default="artifacts/dataset_build", help="Stage output directory")
    parser.add_argument("--output-file", default=None, help="Output parquet path (default: output-dir/triplets_raw.parquet)")
    parser.add_argument("--max-events", type=int, default=None, help="Limit number of processed events")
    parser.add_argument("--step-size", type=int, default=1000, help="ROOT iteration chunk size")
    parser.add_argument("--row-group-size", type=int, default=50_000, help="Parquet row group size")
    parser.add_argument("--flush-rows", type=int, default=50_000, help="In-memory row buffer flush threshold")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for reproducibility metadata")
    parser.set_defaults(func=run)
