"""Decision coupling: state × persona → driving parameters.

Three transparent functions that turn the symbolic affect state
into the parameters a driving policy actually consumes. Each is a
linear modulation of a persona baseline by current state — the
*interface* is the deliverable, not the specific functional forms.

Coefficients are picked so the demo personas span a realistic range:
- calm_commuter (risk=-0.3, neutral state):  gap≈3.0s, follow≈34m
- aggressive_late (risk=+0.7, neutral):       gap≈1.5s, follow≈20m
- High frustration on top:                    gap drops further, follow shortens
"""

from __future__ import annotations

from .persona import DriverPersona
from .state import DriverState


# ---------------------------------------------------------------------------
# Driving-parameter outputs
# ---------------------------------------------------------------------------

def gap_acceptance_threshold(s: DriverState, p: DriverPersona) -> float:
    """Minimum gap (seconds) the driver will accept for a lane change/merge.

    Calm baseline ≈ 3.0s; aggressive baseline ≈ 1.0s. State pushes both
    downward — frustration shortens, arousal shortens further.
    """
    base = 2.5 - 1.5 * p.risk_preference
    state_modulator = 1.0 - 0.4 * s.frustration - 0.2 * max(0.0, s.arousal)
    return max(0.4, base * state_modulator)            # floor at 0.4s


def following_distance(s: DriverState, p: DriverPersona) -> float:
    """Preferred following distance (meters) at cruising speed.

    Calm baseline ≈ 33m; aggressive baseline ≈ 18m. Frustration
    compresses; fatigue extends slightly (slower reaction → more space).
    """
    base = 30.0 - 15.0 * p.risk_preference
    state_modulator = (1.0 - 0.3 * s.frustration) * (1.0 + 0.1 * s.fatigue)
    return max(5.0, base * state_modulator)            # floor at 5m


def speed_overage(s: DriverState, p: DriverPersona) -> float:
    """Fractional overage above the speed limit (e.g. +0.07 = 7% over).

    Aggressive risk preference and high frustration both add to it.
    Negative values mean the driver *underdrives* (cautious calm
    drivers do this).
    """
    return 0.10 * p.risk_preference + 0.15 * s.frustration


def merge_decision(
    gap_seconds: float, s: DriverState, p: DriverPersona
) -> bool:
    """Convenience: take the merge if the available gap exceeds threshold."""
    return gap_seconds >= gap_acceptance_threshold(s, p)


# ---------------------------------------------------------------------------
# Aggregate snapshot for trace logging
# ---------------------------------------------------------------------------

def snapshot(s: DriverState, p: DriverPersona) -> dict:
    """All three params at once. Useful for trace logging at each tick."""
    return {
        "gap_threshold_s": gap_acceptance_threshold(s, p),
        "follow_distance_m": following_distance(s, p),
        "speed_overage_frac": speed_overage(s, p),
    }
