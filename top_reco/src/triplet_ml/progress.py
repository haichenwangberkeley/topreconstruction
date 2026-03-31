#!/usr/bin/env python3
"""Lightweight terminal progress helpers with graceful no-TTY fallback."""

from __future__ import annotations

import shutil
import sys
import time
from typing import Optional, TextIO


def should_show_progress(no_progress: bool, stream: Optional[TextIO] = None) -> bool:
    """Return True when interactive progress output should be enabled."""
    if no_progress:
        return False
    target = stream if stream is not None else sys.stderr
    try:
        return bool(target.isatty())
    except Exception:
        return False


class ProgressBar:
    """Minimal terminal progress bar for row/event/step counters."""

    def __init__(
        self,
        *,
        desc: str,
        total: Optional[int],
        unit: str = "items",
        enabled: bool = True,
        stream: Optional[TextIO] = None,
        min_interval_seconds: float = 0.2,
    ) -> None:
        self.desc = desc
        self.total = int(total) if total is not None else None
        self.unit = unit
        self.enabled = bool(enabled)
        self.stream = stream if stream is not None else sys.stderr
        self.min_interval_seconds = float(min_interval_seconds)

        self.current = 0
        self._closed = False
        self._last_render = 0.0

        if self.enabled:
            self._render(force=True)

    def update(self, n: int = 1) -> None:
        self.set_current(self.current + int(n))

    def set_current(self, value: int) -> None:
        if self._closed:
            return
        self.current = max(int(value), 0)
        if not self.enabled:
            return

        now = time.monotonic()
        if now - self._last_render >= self.min_interval_seconds:
            self._render(force=False)

    def close(self) -> None:
        if self._closed:
            return
        if self.enabled:
            self._render(force=True)
            self.stream.write("\n")
            self.stream.flush()
        self._closed = True

    def _render(self, force: bool) -> None:
        if not self.enabled:
            return

        now = time.monotonic()
        if not force and now - self._last_render < self.min_interval_seconds:
            return
        self._last_render = now

        if self.total is None or self.total <= 0:
            line = f"{self.desc}: {self.current} {self.unit}"
        else:
            width = shutil.get_terminal_size(fallback=(120, 20)).columns
            bar_width = max(10, min(40, width - 55))
            ratio = min(max(float(self.current) / float(self.total), 0.0), 1.0)
            filled = int(round(ratio * bar_width))
            bar = "#" * filled + "-" * (bar_width - filled)
            pct = 100.0 * ratio
            line = f"{self.desc}: [{bar}] {self.current}/{self.total} ({pct:5.1f}%)"

        self.stream.write("\r" + line)
        self.stream.flush()
