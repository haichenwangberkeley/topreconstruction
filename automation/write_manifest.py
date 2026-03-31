#!/usr/bin/env python3
"""Write a structured run manifest for monorepo pipeline executions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_ID = "top-monorepo/run-manifest/v1"


@dataclass(frozen=True)
class ManifestArgs:
    root_input: str
    max_events: int
    seed: int
    artifact_root: Path
    pipeline: str
    select_strategy: str
    select_min_score: float
    select_max_top: int


def _parse_args() -> ManifestArgs:
    """Parse and validate CLI arguments for manifest generation."""
    parser = argparse.ArgumentParser(description="Write top-monorepo run manifest")
    parser.add_argument("--root-input", required=True, help="Path to ROOT input file used")
    parser.add_argument("--max-events", type=int, required=True, help="Max events used in run")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for run")
    parser.add_argument("--artifact-root", required=True, help="Artifact root directory for this run")
    parser.add_argument("--pipeline", default="top_reco", help="Pipeline name")
    parser.add_argument("--select-strategy", default="greedy_disjoint", help="Triplet selection strategy")
    parser.add_argument("--select-min-score", type=float, default=0.5, help="Minimum score for selection")
    parser.add_argument("--select-max-top", type=int, default=4, help="Max selected top candidates per event")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root).resolve()
    workspace_root = Path.cwd().resolve()
    if args.max_events <= 0:
        raise ValueError("--max-events must be > 0")
    if args.select_max_top <= 0:
        raise ValueError("--select-max-top must be > 0")
    if not args.root_input:
        raise ValueError("--root-input must be non-empty")

    root_input_path = Path(args.root_input).resolve()
    if not root_input_path.exists() or not root_input_path.is_file():
        raise ValueError(f"--root-input must be an existing file: {root_input_path}")

    if not artifact_root.is_relative_to(workspace_root):
        raise ValueError(
            "--artifact-root must be inside the current workspace: "
            f"{artifact_root} not under {workspace_root}"
        )

    return ManifestArgs(
        root_input=str(root_input_path),
        max_events=args.max_events,
        seed=args.seed,
        artifact_root=artifact_root,
        pipeline=args.pipeline,
        select_strategy=args.select_strategy,
        select_min_score=args.select_min_score,
        select_max_top=args.select_max_top,
    )


def _status_from_artifacts(artifact_root: Path) -> str:
    """Infer pipeline run status from expected stage output files."""
    if (artifact_root / "error.log").exists() or (artifact_root / "failed.marker").exists():
        return "failed"

    infer_reports = [
        artifact_root / "infer" / "inference_report_xgb.json",
        artifact_root / "infer" / "inference_report_tabpfn.json",
        artifact_root / "infer" / "inference_report.json",
    ]
    return "complete" if any(path.exists() for path in infer_reports) else "incomplete"


def _build_manifest(data: ManifestArgs) -> dict[str, Any]:
    """Construct a run manifest dictionary matching v1 schema."""
    stages = {
        "dataset_build": str(data.artifact_root / "dataset_build"),
        "dataset_prepare": str(data.artifact_root / "dataset_prepare"),
        "train": str(data.artifact_root / "train"),
        "infer": str(data.artifact_root / "infer"),
    }
    if (data.artifact_root / "select_triplets" / "selection_report.json").exists():
        stages["analysis"] = str(data.artifact_root / "select_triplets")

    return {
        "schema": MANIFEST_SCHEMA_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": data.pipeline,
        "parameters": {
            "root_input": data.root_input,
            "max_events": data.max_events,
            "seed": data.seed,
            "selection": {
                "strategy": data.select_strategy,
                "min_score": data.select_min_score,
                "max_top_per_event": data.select_max_top,
            },
        },
        "stages": stages,
        "status": _status_from_artifacts(data.artifact_root),
    }


def _write_manifest(manifest: dict[str, Any], artifact_root: Path) -> Path:
    """Write manifest JSON and return output path."""
    output = artifact_root / "run_manifest.json"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write manifest to {output}: {exc}") from exc
    return output


def main() -> None:
    config = _parse_args()
    manifest = _build_manifest(config)
    output = _write_manifest(manifest, config.artifact_root)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
