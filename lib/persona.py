"""Driver persona — the stable trait vector.

Five scalars chosen to be (a) interpretable to a domain reader without
psych training and (b) sufficient to drive *qualitatively distinct*
behavior under the same event stream. Not a claim that real driver
personality reduces to five axes — it's the minimum that makes the
demo land.

Axes:

  temperament      [-1, +1]  calm ↔ volatile. Multiplies the raw
                             magnitude of arousal/valence shifts during
                             appraisal.
  patience         [0, 1]    impatient ↔ patient. Drives the time
                             constant for frustration decay.
  risk_preference  [-1, +1]  cautious ↔ aggressive. Sets the baseline
                             gap-acceptance / follow-distance / speed
                             modulation.
  reactivity       [0, 1]    flat ↔ expressive. Overall amplitude of
                             V/A/D response to events.
  baseline_va      (V, A)    homeostatic target the state decays toward
                             when no events are firing.

`PRESETS` provides the three demo personas referenced in the
scenario sweep: calm commuter, aggressive-late, anxious-new.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DriverPersona:
    """Immutable per-driver trait vector. See module docstring for axes."""

    name: str
    temperament: float
    patience: float
    risk_preference: float
    reactivity: float
    baseline_va: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        # Validate ranges. Out-of-range values usually indicate a bad
        # JSON file rather than a deliberate choice — fail loud.
        _check_range("temperament", self.temperament, -1.0, 1.0)
        _check_range("patience", self.patience, 0.0, 1.0)
        _check_range("risk_preference", self.risk_preference, -1.0, 1.0)
        _check_range("reactivity", self.reactivity, 0.0, 1.0)
        v, a = self.baseline_va
        _check_range("baseline_va.V", v, -1.0, 1.0)
        _check_range("baseline_va.A", a, -1.0, 1.0)

    # -- I/O ---------------------------------------------------------------

    @classmethod
    def from_json(cls, path: Path | str) -> "DriverPersona":
        data = json.loads(Path(path).read_text())
        if "baseline_va" in data:
            data["baseline_va"] = tuple(data["baseline_va"])
        return cls(**data)

    def to_json(self, path: Path | str) -> None:
        d = asdict(self)
        d["baseline_va"] = list(d["baseline_va"])
        Path(path).write_text(json.dumps(d, indent=2))


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    if not lo <= value <= hi:
        raise ValueError(
            f"DriverPersona.{name}={value} outside expected range [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# Demo presets — referenced by 04_persona_sweep.py
# ---------------------------------------------------------------------------

PRESETS: dict[str, DriverPersona] = {
    "calm_commuter": DriverPersona(
        name="calm_commuter",
        temperament=-0.2,
        patience=0.7,
        risk_preference=-0.3,
        reactivity=0.4,
        baseline_va=(0.1, -0.1),
    ),
    "aggressive_late": DriverPersona(
        name="aggressive_late",
        temperament=+0.6,
        patience=0.2,
        risk_preference=+0.7,
        reactivity=0.8,
        baseline_va=(-0.1, +0.2),
    ),
    "anxious_new_driver": DriverPersona(
        name="anxious_new_driver",
        temperament=+0.3,
        patience=0.5,
        risk_preference=-0.6,
        reactivity=0.9,
        baseline_va=(-0.2, +0.3),
    ),
}
