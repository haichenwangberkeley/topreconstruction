#!/usr/bin/env python3
"""Compatibility wrapper. Canonical script moved to analysis/triplet_reco.py."""

import os
import subprocess
import sys
from pathlib import Path


def _resolve_python() -> str:
    workspace_root = Path(__file__).resolve().parent
    venv_python = workspace_root / ".venv" / "bin" / "python"
    if venv_python.exists() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    return sys.executable


def main() -> None:
    script = Path(__file__).resolve().parent / "analysis" / "triplet_reco.py"
    python_exec = _resolve_python()
    cmd = [python_exec, str(script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
