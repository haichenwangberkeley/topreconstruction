#!/usr/bin/env python3
"""Shared I/O and deterministic helpers for the triplet ML pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import uproot

SCHEMA_VERSION = "triplet-ml/v1.1"


@dataclass(frozen=True)
class StageArtifactPaths:
    output_dir: Path
    primary_output: Path
    config_snapshot: Path


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_tree_name(input_file: str, preferred_tree: Optional[str]) -> str:
    if preferred_tree:
        return preferred_tree
    with uproot.open(input_file) as root_file:
        for key, cls_name in root_file.classnames().items():
            if "TTree" in cls_name:
                return key.split(";")[0]
    raise RuntimeError(f"No TTree found in ROOT file: {input_file}")


def make_tree_paths(input_files: Sequence[str], preferred_tree: Optional[str]) -> List[str]:
    tree_paths: List[str] = []
    for input_file in input_files:
        tree_name = resolve_tree_name(input_file, preferred_tree)
        tree_paths.append(f"{input_file}:{tree_name}")
    return tree_paths


def write_json(path: os.PathLike[str] | str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_config_snapshot(
    output_dir: os.PathLike[str] | str,
    stage: str,
    input_files: Sequence[str],
    parameters: Dict[str, Any],
    seed: int,
    schema_version: str = SCHEMA_VERSION,
) -> Path:
    out_dir = ensure_dir(output_dir)
    path = out_dir / "config_snapshot.json"
    payload = {
        "schema_version": schema_version,
        "stage": stage,
        "input_files": list(input_files),
        "parameters": parameters,
        "seed": int(seed),
    }
    write_json(path, payload)
    return path


def stable_hash_to_unit(value: str, seed: int) -> float:
    payload = f"{seed}|{value}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="little", signed=False)
    return integer / float(2**64)


def assign_split(event_id: int, seed: int, train_frac: float, val_frac: float) -> str:
    u = stable_hash_to_unit(str(event_id), seed)
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


def triplet_schema(include_score: bool = False, score_column: str = "score") -> pa.Schema:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for parquet schema creation") from exc

    fields: List[pa.Field] = [
        pa.field("event_id", pa.int64()),
        pa.field("i", pa.int32()),
        pa.field("j", pa.int32()),
        pa.field("k", pa.int32()),
        pa.field("dr_ab", pa.float32()),
        pa.field("dr_ac", pa.float32()),
        pa.field("dr_bc", pa.float32()),
        pa.field("mij_over_m123_ab", pa.float32()),
        pa.field("mij_over_m123_ac", pa.float32()),
        pa.field("mij_over_m123_bc", pa.float32()),
        pa.field("m123", pa.float32()),
        pa.field("mij_ab", pa.float32()),
        pa.field("mij_ac", pa.float32()),
        pa.field("mij_bc", pa.float32()),
        pa.field("triplet_pt", pa.float32()),
        pa.field("triplet_eta", pa.float32()),
        pa.field("triplet_phi", pa.float32()),
        pa.field("is_truth", pa.int8()),
    ]
    if include_score:
        fields.append(pa.field(str(score_column), pa.float32()))
    return pa.schema(fields)


class StreamingParquetWriter:
    """Append-only Parquet writer with explicit row-group control."""

    def __init__(
        self,
        output_path: os.PathLike[str] | str,
        schema: pa.Schema,
        compression: str = "zstd",
        row_group_size: int = 50_000,
    ) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for parquet writing") from exc

        self.output_path = Path(output_path)
        ensure_dir(self.output_path.parent)
        self.schema = schema
        self.row_group_size = int(row_group_size)
        self._writer = pq.ParquetWriter(self.output_path, schema=self.schema, compression=compression)
        self.rows_written = 0

    def write_rows(self, columns: Dict[str, Sequence[Any]]) -> None:
        try:
            import pyarrow as pa
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for parquet writing") from exc

        if not columns:
            return
        any_column = next(iter(columns.values()))
        n_rows = len(any_column)
        if n_rows == 0:
            return
        table = pa.Table.from_pydict(columns, schema=self.schema)
        self._writer.write_table(table, row_group_size=self.row_group_size)
        self.rows_written += n_rows

    def close(self) -> None:
        self._writer.close()


class ColumnBuffer:
    """Lightweight columnar append buffer for streaming output."""

    def __init__(self, columns: Iterable[str]) -> None:
        self._columns = list(columns)
        self._store: Dict[str, List[Any]] = {name: [] for name in self._columns}
        self.size = 0

    def append_row(self, row: Dict[str, Any]) -> None:
        for key in self._columns:
            self._store[key].append(row[key])
        self.size += 1

    def take_all(self) -> Dict[str, List[Any]]:
        payload = self._store
        self._store = {name: [] for name in self._columns}
        self.size = 0
        return payload

    @property
    def columns(self) -> List[str]:
        return self._columns
