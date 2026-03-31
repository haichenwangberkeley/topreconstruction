#!/usr/bin/env python3
"""Utilities for running and summarizing selection-strategy comparisons."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import select_triplets
from . import triplet_io as pio


STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    "greedy_disjoint": "Greedy selection by descending score while forbidding jet reuse across selected triplets.",
    "top1": "Select exactly one triplet per event: the highest-scored candidate.",
    "topk": "Select up to K highest-scored triplets per event.",
    "threshold": "Select all triplets above the score threshold (up to max-top-per-event).",
    "best_pair_avg_disjoint": "Select exactly one disjoint pair with highest average score (requires inferred >=6 jets).",
}


def _efficiency_sort_key(row: Dict[str, Any]) -> tuple[int, float, int, str]:
    eff = row.get("triplet_reconstruction_efficiency")
    if eff is None:
        return (0, -1.0, int(row.get("selected_truth_matched_triplets", 0)), str(row.get("strategy", "")))
    return (1, float(eff), int(row.get("selected_truth_matched_triplets", 0)), str(row.get("strategy", "")))


def build_strategy_comparison_rows(strategy_reports: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for strategy, report in strategy_reports.items():
        rows.append(
            {
                "strategy": str(strategy),
                "strategy_description": STRATEGY_DESCRIPTIONS.get(str(strategy), "No description available."),
                "triplet_reconstruction_efficiency": report.get("triplet_reconstruction_efficiency"),
                "selected_truth_matched_triplets": int(report.get("selected_truth_matched_triplets", 0)),
                "truth_matched_triplets_total": int(report.get("truth_matched_triplets_total", 0)),
                "selected_rows_total": int(report.get("selected_rows_total", 0)),
                "events_total": int(report.get("events_total", 0)),
                "events_with_truth_matched_triplets": int(report.get("events_with_truth_matched_triplets", 0)),
                "events_without_truth_matched_triplets": int(report.get("events_without_truth_matched_triplets", 0)),
            }
        )

    rows.sort(key=_efficiency_sort_key, reverse=True)

    rank = 0
    for row in rows:
        if row["triplet_reconstruction_efficiency"] is None:
            row["rank_by_efficiency"] = None
        else:
            rank += 1
            row["rank_by_efficiency"] = rank

    return rows


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        headers = [
            "strategy",
            "strategy_description",
            "rank_by_efficiency",
            "triplet_reconstruction_efficiency",
            "selected_truth_matched_triplets",
            "truth_matched_triplets_total",
            "selected_rows_total",
            "events_total",
            "events_with_truth_matched_triplets",
            "events_without_truth_matched_triplets",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
        return

    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    headers = [
        "Rank",
        "Strategy",
        "Description",
        "Efficiency",
        "Selected Truth",
        "Truth Total",
        "Selected Total",
    ]

    lines: List[str] = [
        "# Strategy Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        efficiency = row.get("triplet_reconstruction_efficiency")
        efficiency_text = "N/A" if efficiency is None else f"{float(efficiency):.12f}"
        rank = row.get("rank_by_efficiency")
        rank_text = "N/A" if rank is None else str(rank)
        line_cells = [
            rank_text,
            str(row.get("strategy", "")),
            str(row.get("strategy_description", "")),
            efficiency_text,
            str(row.get("selected_truth_matched_triplets", 0)),
            str(row.get("truth_matched_triplets_total", 0)),
            str(row.get("selected_rows_total", 0)),
        ]
        lines.append("| " + " | ".join(line_cells) + " |")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _run_single_strategy(
    inference_path: str,
    output_dir: Path,
    strategy: str,
    score_column: Optional[str],
    min_score: float,
    max_top_per_event: int,
    top_k: int,
    batch_size: int,
    row_group_size: int,
    flush_rows: int,
    dummy_value: float,
    skip_plots: bool,
    plot_bins: int,
    no_progress: bool,
) -> Dict[str, Any]:
    args = argparse.Namespace(
        inference=inference_path,
        output_dir=str(output_dir),
        strategy=strategy,
        score_column=score_column,
        min_score=min_score,
        max_top_per_event=max_top_per_event,
        top_k=top_k,
        batch_size=batch_size,
        row_group_size=row_group_size,
        flush_rows=flush_rows,
        dummy_value=dummy_value,
        plot_root=str(output_dir / "plots"),
        plot_bins=plot_bins,
        skip_plots=skip_plots,
        no_progress=no_progress,
    )
    select_triplets.run(args)
    report_path = output_dir / "selection_report.json"
    return pio.read_json(str(report_path))


def run_strategy_sweep(
    inference_path: str,
    output_dir: Path,
    strategies: Iterable[str],
    score_column: Optional[str],
    min_score: float,
    max_top_per_event: int,
    top_k: int,
    batch_size: int,
    row_group_size: int,
    flush_rows: int,
    dummy_value: float,
    skip_plots: bool,
    plot_bins: int,
    no_progress: bool,
) -> Dict[str, Any]:
    output_dir = pio.ensure_dir(output_dir)

    strategy_list = list(strategies)
    if len(strategy_list) == 0:
        raise ValueError("strategies must contain at least one strategy")

    strategy_reports: Dict[str, Dict[str, Any]] = {}
    for strategy in strategy_list:
        strategy_output = output_dir / f"select_{strategy}"
        strategy_reports[strategy] = _run_single_strategy(
            inference_path=inference_path,
            output_dir=strategy_output,
            strategy=strategy,
            score_column=score_column,
            min_score=min_score,
            max_top_per_event=max_top_per_event,
            top_k=top_k,
            batch_size=batch_size,
            row_group_size=row_group_size,
            flush_rows=flush_rows,
            dummy_value=dummy_value,
            skip_plots=skip_plots,
            plot_bins=plot_bins,
            no_progress=no_progress,
        )

    rows = build_strategy_comparison_rows(strategy_reports)

    csv_path = output_dir / "strategy_comparison.csv"
    _write_rows_csv(csv_path, rows)
    markdown_path = output_dir / "strategy_comparison.md"
    _write_rows_markdown(markdown_path, rows)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inference_path": str(inference_path),
        "strategies": strategy_list,
        "min_score": float(min_score),
        "max_top_per_event": int(max_top_per_event),
        "top_k": int(top_k),
        "skip_plots": bool(skip_plots),
        "rows": rows,
        "best_strategy_by_efficiency": rows[0]["strategy"] if rows and rows[0]["rank_by_efficiency"] == 1 else None,
        "csv_file": str(csv_path),
        "markdown_file": str(markdown_path),
        "strategy_reports": {
            strategy: str(output_dir / f"select_{strategy}" / "selection_report.json")
            for strategy in strategy_reports.keys()
        },
    }

    summary_path = output_dir / "strategy_comparison_summary.json"
    pio.write_json(summary_path, summary)
    summary["summary_file"] = str(summary_path)
    return summary
