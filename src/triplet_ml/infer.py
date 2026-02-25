#!/usr/bin/env python3
"""Stage 4: model + test.parquet -> inference parquet (streaming)."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Dict

import numpy as np

from . import features
from . import plotting
from . import progress as prog
from . import triplet_io as pio
from .models import (
    MODEL_BACKENDS,
    default_inference_filename,
    inference_score_column,
    load_model,
    normalize_model_backend,
    resolve_backend_and_path,
)


def _discover_comparison_inputs(output_dir: Path, active_model: str, active_path: Path) -> Dict[str, str]:
    payload: Dict[str, str] = {active_model: str(active_path)}
    for backend in MODEL_BACKENDS:
        if backend == active_model:
            continue
        candidate = output_dir / default_inference_filename(backend)
        if candidate.exists():
            payload[backend] = str(candidate)
    return payload


def run(args: argparse.Namespace) -> None:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for inference stage") from exc

    model_backend, model_path = resolve_backend_and_path(
        model_arg=args.model,
        model_path=args.model_path,
        default_train_dir=args.train_output_dir,
        test_dataset=args.test,
    )
    model_backend = normalize_model_backend(model_backend)
    show_progress = prog.should_show_progress(args.no_progress)

    output_dir = pio.ensure_dir(args.output_dir)
    output_path = (
        Path(args.output_file)
        if args.output_file
        else output_dir / default_inference_filename(model_backend)
    )
    report_path = output_dir / f"inference_report_{model_backend}.json"
    score_column = inference_score_column(model_backend)

    model = load_model(
        model_backend=model_backend,
        path=model_path,
        feature_columns=list(features.FEATURE_COLUMNS),
    )

    input_columns = [field.name for field in pio.triplet_schema(include_score=False)]
    output_schema = pio.triplet_schema(include_score=True, score_column=score_column)

    writer = pio.StreamingParquetWriter(
        output_path=output_path,
        schema=output_schema,
        row_group_size=args.row_group_size,
    )

    dataset = ds.dataset(args.test, format="parquet")
    total_rows_estimate = None
    if show_progress:
        try:
            total_rows_estimate = int(dataset.count_rows())
        except Exception:
            total_rows_estimate = None
    progress_bar = prog.ProgressBar(
        desc="inference rows",
        total=total_rows_estimate,
        unit="rows",
        enabled=show_progress,
    )
    scanner = dataset.scanner(columns=input_columns, batch_size=args.batch_size)

    n_input = 0
    n_output = 0
    prediction_time_seconds = 0.0

    try:
        for batch in scanner.to_batches():
            payload = batch.to_pydict()
            n_rows = len(payload["event_id"])
            if n_rows == 0:
                continue

            feature_matrix = np.column_stack(
                [np.asarray(payload[col], dtype=np.float32) for col in features.FEATURE_COLUMNS]
            )
            features.assert_feature_batch_sane(
                {name: feature_matrix[:, i] for i, name in enumerate(features.FEATURE_COLUMNS)}
            )

            pred_start = time.perf_counter()
            scores = model.predict_proba(feature_matrix)[:, 1].astype(np.float32)
            prediction_time_seconds += time.perf_counter() - pred_start

            payload[score_column] = scores.tolist()
            writer.write_rows(payload)

            n_input += n_rows
            n_output += len(scores)
            progress_bar.update(n_rows)
    finally:
        progress_bar.close()

    writer.close()

    if n_input != n_output:
        raise RuntimeError(f"Inference row count mismatch: input={n_input}, output={n_output}")

    plot_metrics: Dict[str, float] = {}
    comparison_plot_metrics: Dict[str, float] = {}
    if not args.skip_plots:
        plot_metrics = plotting.generate_inference_plots(
            inference_dataset=str(output_path),
            model_name=model_backend,
            output_root=args.plot_root,
            score_column=score_column,
        )
        comparison_inputs = _discover_comparison_inputs(
            output_dir=output_dir,
            active_model=model_backend,
            active_path=output_path,
        )
        if "xgb" in comparison_inputs and "tabpfn" in comparison_inputs:
            comparison_plot_metrics = plotting.generate_inference_comparison_plots(
                inference_datasets=comparison_inputs,
                output_root=args.plot_root,
            )

    report = {
        "schema_version": pio.SCHEMA_VERSION,
        "model_backend": model_backend,
        "rows_input": n_input,
        "rows_output": n_output,
        "model": model_path,
        "output_file": str(output_path),
        "report_file": str(report_path),
        "score_column": score_column,
        "prediction_time_seconds": float(prediction_time_seconds),
        "plot_metrics": plot_metrics,
        "comparison_plot_metrics": comparison_plot_metrics,
    }
    pio.write_json(report_path, report)

    snapshot_path = pio.write_config_snapshot(
        output_dir=output_dir,
        stage="inference",
        input_files=[model_path, args.test],
        parameters={
            "model": model_backend,
            "model_path": model_path,
            "batch_size": args.batch_size,
            "row_group_size": args.row_group_size,
            "plot_root": args.plot_root,
            "skip_plots": args.skip_plots,
            "train_output_dir": args.train_output_dir,
            "no_progress": args.no_progress,
        },
        seed=args.seed,
    )
    shutil.copyfile(snapshot_path, output_dir / f"config_snapshot_infer_{model_backend}.json")


def register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("infer", help="Stage 4: run model inference on triplets")
    parser.add_argument(
        "--model",
        default="xgb",
        help="Model backend (`xgb` or `tabpfn`) or legacy model path",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Explicit trained model path (optional; defaults are auto-resolved from backend)",
    )
    parser.add_argument(
        "--train-output-dir",
        default="artifacts/train",
        help="Directory used for default model path resolution",
    )
    parser.add_argument("--test", required=True, help="Test parquet path")
    parser.add_argument("--output-dir", default="artifacts/infer", help="Output directory")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output parquet path (default depends on --model backend)",
    )
    parser.add_argument("--batch-size", type=int, default=100_000, help="Read batch size")
    parser.add_argument("--row-group-size", type=int, default=50_000, help="Parquet row group size")
    parser.add_argument("--plot-root", default="plots", help="Root directory for generated inference plots")
    parser.add_argument("--skip-plots", action="store_true", help="Skip automatic inference plotting")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress output")
    parser.add_argument("--seed", type=int, default=42, help="Seed stored in config snapshot")
    parser.set_defaults(func=run)
