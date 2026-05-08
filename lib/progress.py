"""Shared ProgressTracker for the calibration scripts.

A small utility — extracted from `scripts/08_calibrate_decay.py` so
both 07 (appraisal) and 08 (decay) can use the same logging shape.
The cadence is one line per "block" (`step()` call), where a block
is the unit of work the caller defines (typically `n_samples`
generations).

Output line shape:

    [ 12/126   9.5%]  <label>  (1m18s elapsed, ETA 5m32s, 0.15 gen/s)

Specifying `sample_n` in `step()` lets a single block report multiple
generations — useful when one logical step contains an internal
sample loop (e.g., N=3 samples per event×persona×payload).
"""

from __future__ import annotations

import time
from typing import Optional


class ProgressTracker:
    """Counts generations, prints a one-line status per `step()`."""

    def __init__(self, total: int) -> None:
        self.total = max(total, 1)
        self.i = 0
        self.t0 = time.time()

    @staticmethod
    def _format_dt(sec: float) -> str:
        m, s = divmod(int(round(sec)), 60)
        return f"{m:d}m{s:02d}s" if m else f"{s:d}s"

    def adjust_total(self, new_total: int) -> None:
        """Resume hook: caller bumps the denominator after loading a checkpoint."""
        self.total = max(new_total, max(self.i, 1))

    def step(self, label: str, *, sample_n: int = 1) -> None:
        self.i += sample_n
        elapsed = time.time() - self.t0
        rate = self.i / max(elapsed, 1e-3)
        remaining = max(self.total - self.i, 0)
        eta = remaining / rate if rate > 0 else 0.0
        pct = 100.0 * self.i / max(self.total, 1)
        print(
            f"  [{self.i:>3d}/{self.total}  {pct:5.1f}%]  {label}  "
            f"({self._format_dt(elapsed)} elapsed, "
            f"ETA {self._format_dt(eta)}, "
            f"{rate:4.2f} gen/s)",
            flush=True,
        )

    def credit_skipped(self, n: int, label: Optional[str] = None) -> None:
        """Resume hook: bump the counter for already-completed work, print one line."""
        self.i += n
        if label:
            elapsed = time.time() - self.t0
            pct = 100.0 * self.i / max(self.total, 1)
            print(f"  [{self.i:>3d}/{self.total}  {pct:5.1f}%]  {label}  (resumed)",
                  flush=True)
