"""Lightweight in-process metrics, Prometheus text format.

No external dependency: counters are plain floats with a lock.  Enough
for a production-style ``/metrics`` endpoint without pulling in the
prometheus client library.
"""

from __future__ import annotations

import threading
import time
from typing import Self


class Metrics:
    """Simple counter/histogram registry exposing Prometheus text format."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}  # name -> raw samples

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def inc(self, name: str, amount: float = 1.0) -> None:
        """Increment a counter (created on first use)."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + amount

    def observe(self, name: str, value: float) -> None:
        """Record a sample in a histogram (bounded to 1000 samples)."""
        with self._lock:
            samples = self._histograms.setdefault(name, [])
            samples.append(value)
            if len(samples) > 1000:
                del samples[: len(samples) - 1000]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, prefix: str = "smart_contract_rag") -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                lines.append(f"# TYPE {prefix}_{name} counter")
                lines.append(f"{prefix}_{name} {value:.0f}")
            for name, samples in sorted(self._histograms.items()):
                if not samples:
                    continue
                count = len(samples)
                total = sum(samples)
                sorted_s = sorted(samples)
                lines.append(f"# TYPE {prefix}_{name}_count counter")
                lines.append(f"# TYPE {prefix}_{name}_sum counter")
                lines.append(f"{prefix}_{name}_count {count}")
                lines.append(f"{prefix}_{name}_sum {total:.6f}")
                for q in (0.50, 0.90, 0.95, 0.99):
                    idx = min(int(q * count), count - 1)
                    lines.append(f"{prefix}_{name}_quantile{{quantile=\"{q}\"}} {sorted_s[idx]:.6f}")
        return "\n".join(lines) + "\n"

    # Context manager for timing a block
    def time(self, name: str) -> _Timer:
        return _Timer(self, name)


class _Timer:
    def __init__(self, metrics: Metrics, name: str) -> None:
        self._metrics = metrics
        self._name = name

    def __enter__(self) -> Self:
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self._metrics.observe(self._name, time.monotonic() - self._start)
