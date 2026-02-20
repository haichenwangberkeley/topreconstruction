#!/usr/bin/env python3
"""Stage 2: triplets_raw.parquet -> train/val/test parquet with event-level split and balancing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import features
from . import plotting
from . import triplet_io as pio


@dataclass
class EventBuffer:
    event_id: int
    rows: Dict[str, List[float | int]]


def _new_event_buffer(event_id: int, columns: Sequence[str]) -> EventBuffer:
    return EventBuffer(event_id=event_id, rows={name: [] for name in columns})


def _append_row(buffer: EventBuffer, row_values: Dict[str, float | int]) -> None:
    for key, value in row_values.items():
        buffer.rows[key].append(value)


def _event_background_score(event_id: int, i: int, j: int, k: int, seed: int) -> float:
    return pio.stable_hash_to_unit(f"{event_id}:{i}:{j}:{k}", seed)


def _select_indices(buffer: EventBuffer, split: str, max_bg_per_event: Optional[int], seed: int) -> List[int]:
    n_rows = len(buffer.rows["event_id"])
    if split == "test" or max_bg_per_event is None:
        return list(range(n_rows))

    is_truth = buffer.rows["is_truth"]
    signal_indices = [idx for idx, val in enumerate(is_truth) if int(val) == 1]
    background_indices = [idx for idx, val in enumerate(is_truth) if int(val) == 0]

    if len(background_indices) <= max_bg_per_event:
        return sorted(signal_indices + background_indices)

    scored = []
    for idx in background_indices:
        score = _event_background_score(
            event_id=buffer.event_id,
            i=int(buffer.rows["i"][idx]),
            j=int(buffer.rows["j"][idx]),
            k=int(buffer.rows["k"][idx]),
            seed=seed,
        )
        scored.append((score, idx))

    scored.sort(key=lambda item: item[0])
    kept_background = [idx for _, idx in scored[:max_bg_per_event]]
    return sorted(signal_indices + kept_background)


def _slice_payload(rows: Dict[str, List[float | int]], indices: Sequence[int]) -> Dict[str, List[float | int]]:
    return {key: [values[i] for i in indices] for key, values in rows.items()}


def _fraction(num: int, den: int) -> float:
    return float(num / den) if den > 0 else 0.0


def _write_dataset_statistics(
    output_dir: Path,
    split_counts: Dict[str, Dict[str, int]],
    split_events: Dict[str, set[int]],
    input_rows: int,
    max_bg_per_event: Optional[int],
) -> None:
    totals = {
        "events": sum(len(split_events[s]) for s in ("train", "val", "test")),
        "triplets": sum(split_counts[s]["rows"] for s in ("train", "val", "test")),
        "truth": sum(split_counts[s]["signal"] for s in ("train", "val", "test")),
        "background": sum(split_counts[s]["background"] for s in ("train", "val", "test")),
        "removed": sum(split_counts[s]["removed"] for s in ("train", "val", "test")),
    }

    statistics = {
        "schema_version": pio.SCHEMA_VERSION,
        "input_rows": int(input_rows),
        "events": {split: len(split_events[split]) for split in ("train", "val", "test")},
        "triplets": {split: split_counts[split]["rows"] for split in ("train", "val", "test")},
        "truth_triplets": {split: split_counts[split]["signal"] for split in ("train", "val", "test")},
        "background_triplets": {split: split_counts[split]["background"] for split in ("train", "val", "test")},
        "average_triplets_per_event": {
            split: _fraction(split_counts[split]["rows"], len(split_events[split])) for split in ("train", "val", "test")
        },
        "class_fractions": {
            split: {
                "signal": _fraction(split_counts[split]["signal"], split_counts[split]["rows"]),
                "background": _fraction(split_counts[split]["background"], split_counts[split]["rows"]),
            }
            for split in ("train", "val", "test")
        },
        "totals": {
            "events": totals["events"],
            "triplets": totals["triplets"],
            "truth_triplets": totals["truth"],
            "background_triplets": totals["background"],
            "average_triplets_per_event": _fraction(totals["triplets"], totals["events"]),
            "signal_fraction": _fraction(totals["truth"], totals["triplets"]),
            "background_fraction": _fraction(totals["background"], totals["triplets"]),
        },
        "balancing": {
            "max_bg_per_event": max_bg_per_event,
            "triplets_removed_total": totals["removed"],
            "triplets_removed_by_split": {split: split_counts[split]["removed"] for split in ("train", "val", "test")},
        },
    }
    pio.write_json(output_dir / "dataset_statistics.json", statistics)

    md_lines = [
        "# Dataset Statistics",
        "",
        f"- Schema version: `{pio.SCHEMA_VERSION}`",
        f"- Input rows: `{input_rows}`",
        f"- Total triplets: `{totals['triplets']}`",
        f"- Truth triplets: `{totals['truth']}`",
        f"- Background triplets: `{totals['background']}`",
        f"- Triplets removed by balancing: `{totals['removed']}`",
        f"- Background cap: `{max_bg_per_event}`",
        "",
        "| Split | Events | Triplets | Truth | Background | Avg Triplets/Event | Signal Fraction | Background Fraction | Removed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        rows = split_counts[split]["rows"]
        sig = split_counts[split]["signal"]
        bkg = split_counts[split]["background"]
        ev = len(split_events[split])
        md_lines.append(
            "| {split} | {ev} | {rows} | {sig} | {bkg} | {avg:.4f} | {sf:.4f} | {bf:.4f} | {removed} |".format(
                split=split,
                ev=ev,
                rows=rows,
                sig=sig,
                bkg=bkg,
                avg=_fraction(rows, ev),
                sf=_fraction(sig, rows),
                bf=_fraction(bkg, rows),
                removed=split_counts[split]["removed"],
            )
        )

    with open(output_dir / "dataset_statistics.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines) + "\n")


def run(args: argparse.Namespace) -> None:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for dataset_prepare stage") from exc

    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("--train-frac must be in (0,1)")
    if not (0.0 < args.val_frac < 1.0):
        raise ValueError("--val-frac must be in (0,1)")
    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")
    if args.max_bg_per_event is not None and args.max_bg_per_event <= 0:
        raise ValueError("--max-bg-per-event must be > 0 when provided")

    output_dir = pio.ensure_dir(args.output_dir)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    schema = pio.triplet_schema(include_score=False)
    ordered_columns = [field.name for field in schema]

    writers = {
        "train": pio.StreamingParquetWriter(train_path, schema=schema, row_group_size=args.row_group_size),
        "val": pio.StreamingParquetWriter(val_path, schema=schema, row_group_size=args.row_group_size),
        "test": pio.StreamingParquetWriter(test_path, schema=schema, row_group_size=args.row_group_size),
    }

    split_events: Dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    split_counts = {
        "train": {"rows": 0, "signal": 0, "background": 0, "removed": 0},
        "val": {"rows": 0, "signal": 0, "background": 0, "removed": 0},
        "test": {"rows": 0, "signal": 0, "background": 0, "removed": 0},
    }

    dataset = ds.dataset(args.input, format="parquet")
    scanner = dataset.scanner(columns=ordered_columns, batch_size=args.batch_size)

    current: Optional[EventBuffer] = None
    closed_events: set[int] = set()
    total_input_rows = 0
    total_output_rows = 0
    deterministic_check_events: List[int] = []

    def flush_event(buffer: EventBuffer) -> None:
        nonlocal total_output_rows

        split = pio.assign_split(
            event_id=buffer.event_id,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )

        selected_indices = _select_indices(buffer, split, args.max_bg_per_event, args.seed)
        payload = _slice_payload(buffer.rows, selected_indices)
        n_input_rows = len(buffer.rows["event_id"])

        if payload["event_id"]:
            features.assert_feature_batch_sane({name: np.asarray(payload[name]) for name in features.FEATURE_COLUMNS})
            features.assert_observable_batch_sane({name: np.asarray(payload[name]) for name in features.OBSERVABLE_COLUMNS})

        writers[split].write_rows(payload)

        n_rows = len(payload["event_id"])
        n_signal = int(np.sum(np.asarray(payload["is_truth"], dtype=np.int64))) if n_rows > 0 else 0
        n_background = n_rows - n_signal

        split_events[split].add(buffer.event_id)
        split_counts[split]["rows"] += n_rows
        split_counts[split]["signal"] += n_signal
        split_counts[split]["background"] += n_background
        split_counts[split]["removed"] += int(n_input_rows - n_rows)
        total_output_rows += n_rows

        if len(deterministic_check_events) < 50:
            deterministic_check_events.append(buffer.event_id)

    for batch in scanner.to_batches():
        batch_dict = batch.to_pydict()
        n_rows = len(batch_dict["event_id"])
        total_input_rows += n_rows

        for row_idx in range(n_rows):
            event_id = int(batch_dict["event_id"][row_idx])

            if current is None:
                current = _new_event_buffer(event_id, ordered_columns)
            elif event_id != current.event_id:
                flush_event(current)
                closed_events.add(current.event_id)
                if event_id in closed_events:
                    raise RuntimeError(
                        "Input parquet rows are not grouped by event_id; event-level balancing requires grouped events."
                    )
                current = _new_event_buffer(event_id, ordered_columns)

            row_values = {name: batch_dict[name][row_idx] for name in ordered_columns}
            _append_row(current, row_values)

    if current is not None:
        flush_event(current)

    for writer in writers.values():
        writer.close()

    if split_events["train"] & split_events["val"]:
        raise RuntimeError("Event overlap detected between train and val splits")
    if split_events["train"] & split_events["test"]:
        raise RuntimeError("Event overlap detected between train and test splits")
    if split_events["val"] & split_events["test"]:
        raise RuntimeError("Event overlap detected between val and test splits")

    # Reproducibility check: same seed must map same event to same split.
    for event_id in deterministic_check_events:
        split1 = pio.assign_split(event_id, args.seed, args.train_frac, args.val_frac)
        split2 = pio.assign_split(event_id, args.seed, args.train_frac, args.val_frac)
        if split1 != split2:
            raise RuntimeError("Deterministic split reproducibility check failed")

    report = {
        "schema_version": pio.SCHEMA_VERSION,
        "input_rows": total_input_rows,
        "output_rows": total_output_rows,
        "train": split_counts["train"],
        "val": split_counts["val"],
        "test": split_counts["test"],
        "event_counts": {
            "train": len(split_events["train"]),
            "val": len(split_events["val"]),
            "test": len(split_events["test"]),
        },
        "max_bg_per_event": args.max_bg_per_event,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
    }
    pio.write_json(output_dir / "dataset_prepare_report.json", report)
    _write_dataset_statistics(output_dir, split_counts, split_events, total_input_rows, args.max_bg_per_event)

    if not args.skip_plots:
        plotting.generate_feature_validation_plots(str(train_path), split_name="train", output_root=args.plot_root)
        plotting.generate_feature_validation_plots(str(test_path), split_name="test", output_root=args.plot_root)

    pio.write_config_snapshot(
        output_dir=output_dir,
        stage="dataset_prepare",
        input_files=[args.input],
        parameters={
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "max_bg_per_event": args.max_bg_per_event,
            "row_group_size": args.row_group_size,
            "batch_size": args.batch_size,
            "plot_root": args.plot_root,
            "skip_plots": args.skip_plots,
        },
        seed=args.seed,
    )


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("dataset_prepare", help="Stage 2: split and balance triplet dataset")
    parser.add_argument("--input", required=True, help="Input raw triplet parquet (file or directory)")
    parser.add_argument("--output-dir", default="artifacts/dataset_prepare", help="Output directory")
    parser.add_argument("--train-frac", type=float, default=0.20, help="Train split fraction")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Validation split fraction")
    parser.add_argument("--max-bg-per-event", type=int, default=200, help="Background cap per event for train/val")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Parquet read batch size")
    parser.add_argument("--row-group-size", type=int, default=50_000, help="Output parquet row group size")
    parser.add_argument("--plot-root", default="plots", help="Root directory for generated validation plots")
    parser.add_argument("--skip-plots", action="store_true", help="Skip automatic feature/observable validation plotting")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split/balancing seed")
    parser.set_defaults(func=run)
