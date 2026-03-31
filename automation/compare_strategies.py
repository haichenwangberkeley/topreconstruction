#!/usr/bin/env python3
"""Run strategy sweep on existing inference artifacts and produce comparison summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from triplet_ml import select_triplets
from triplet_ml import strategy_compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare select_triplets strategies on one scored inference dataset")
    parser.add_argument("--artifact-root", required=True, help="Run artifact root (e.g. top_reco/artifacts/run_40000)")
    parser.add_argument("--inference", default=None, help="Inference parquet path (default: <artifact-root>/infer/inference_test_xgb.parquet)")
    parser.add_argument("--output-dir", default=None, help="Comparison output directory (default: <artifact-root>/strategy_comparison)")
    parser.add_argument("--strategies", nargs="+", default=list(select_triplets.STRATEGIES), choices=list(select_triplets.STRATEGIES), help="Strategies to compare")
    parser.add_argument("--score-column", default=None, help="Optional explicit score column name")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum score for selected triplets")
    parser.add_argument("--max-top-per-event", type=int, default=4, help="Maximum selected triplets per event")
    parser.add_argument("--top-k", type=int, default=4, help="Top-k for strategy topk")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Parquet read batch size")
    parser.add_argument("--row-group-size", type=int, default=50_000, help="Parquet output row group size")
    parser.add_argument("--flush-rows", type=int, default=50_000, help="Flush threshold for parquet writes")
    parser.add_argument("--dummy-value", type=float, default=-999.0, help="Placeholder for missing slots")
    parser.add_argument("--plot-bins", type=int, default=20, help="Bin count if plots are enabled")
    parser.add_argument("--with-plots", action="store_true", help="Generate plots for each strategy")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.strategies:
        raise ValueError("--strategies must include at least one strategy")

    artifact_root = Path(args.artifact_root)
    inference_path = Path(args.inference) if args.inference else artifact_root / "infer" / "inference_test_xgb.parquet"
    if not inference_path.exists():
        raise FileNotFoundError(f"Inference parquet not found: {inference_path}")

    output_dir = Path(args.output_dir) if args.output_dir else artifact_root / "strategy_comparison"

    summary = strategy_compare.run_strategy_sweep(
        inference_path=str(inference_path),
        output_dir=output_dir,
        strategies=args.strategies,
        score_column=args.score_column,
        min_score=args.min_score,
        max_top_per_event=args.max_top_per_event,
        top_k=args.top_k,
        batch_size=args.batch_size,
        row_group_size=args.row_group_size,
        flush_rows=args.flush_rows,
        dummy_value=args.dummy_value,
        skip_plots=not args.with_plots,
        plot_bins=args.plot_bins,
        no_progress=args.no_progress,
    )

    print(f"Wrote {summary['summary_file']}")
    print(f"Wrote {summary['csv_file']}")
    print(f"Wrote {summary['markdown_file']}")
    print(f"Best strategy by efficiency: {summary['best_strategy_by_efficiency']}")


if __name__ == "__main__":
    main()
