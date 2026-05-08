"""Driver state and dynamics.

Six scalars: continuous V/A/D (Mehrabian PAD) plus three accumulators
(stress, frustration, fatigue) that capture phenomena PAD alone
collapses. Frustration in particular is a fast event-driven scalar
that decouples from arousal: a patient driver in heavy traffic can
have low frustration but elevated arousal; an impatient driver
without traffic can have high frustration without elevated arousal.

Dynamics: every step pulls each scalar toward a target (mostly the
persona's `baseline_va` for V/A, zero for everything else) on an
exponential time constant.

V/A/D τ are loaded from `data/empirical_decay.json` when present
(produced by `scripts/08_calibrate_decay.py` — measures the LLM's
implicit prior on emotional recovery times). Hand-authored fallbacks
remain when that file is absent. Frustration/stress/fatigue τ stay
hand-authored because they're symbolic accumulators with no direct
activation-space readout.

The interface is deliberately small:

  state.step(dt_s, persona)   advance dynamics one tick (mutates).
  state.apply(delta)          add an appraisal-emitted ΔState (mutates).
  state.snapshot()            return a dict for trace logging.

`DeltaState` is what the appraisal layer returns; it composes naturally
under addition so multiple events firing in the same step accumulate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Empirical decay loading (lazy, at import time)
# ---------------------------------------------------------------------------

_EMPIRICAL_DECAY_PATH = Path(__file__).resolve().parent.parent / "data" / "empirical_decay.json"


_R2_QUALITY_FLOOR = 0.3   # below this we treat the empirical fit as uninformative


def _load_empirical_taus() -> dict:
    """Return `{axis: tau_seconds}` from calibration, or empty dict if absent.

    Quality filter: only adopt an empirical τ when the per-axis fit has
    R² ≥ 0.3 *and* a positive numeric τ. The decay calibration is
    informative only when the LLM's implicit prior on elapsed time
    actually behaves like exponential decay — which (as the
    May 2026 calibration revealed) it often doesn't. Negative R²
    means the model fits worse than predicting the mean, and adopting
    a τ from such a fit would be cargo-culting. In that case we fall
    through to the hand-authored constants — see `notes/design.md`
    §3.3 for the methodological note.
    """
    if not _EMPIRICAL_DECAY_PATH.exists():
        return {}
    try:
        blob = json.loads(_EMPIRICAL_DECAY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    taus = blob.get("tau_per_axis_s") or {}
    fits = blob.get("fits") or {}
    accepted: dict = {}
    for axis, tau in taus.items():
        if not isinstance(tau, (int, float)) or tau <= 0:
            continue
        r2 = (fits.get(axis) or {}).get("r2")
        if not isinstance(r2, (int, float)) or r2 < _R2_QUALITY_FLOOR:
            continue
        accepted[axis] = float(tau)
    return accepted


_EMPIRICAL_TAUS = _load_empirical_taus()


# ---------------------------------------------------------------------------
# Time constants for state decay (seconds)
# ---------------------------------------------------------------------------
# V / A / D fall back to hand-authored constants when no empirical
# calibration is present. With empirical present, these become
# distinct per axis (the LLM-prior calibration measures V, A, D
# independently rather than collapsing V and A onto one τ).
TAU_V_S = _EMPIRICAL_TAUS.get("V", 8.0)
TAU_A_S = _EMPIRICAL_TAUS.get("A", 8.0)
TAU_D_S = _EMPIRICAL_TAUS.get("D", 6.0)
# Backward-compat alias kept for any older code that references TAU_VA_S
# directly. New code should use TAU_V_S / TAU_A_S.
TAU_VA_S = (TAU_V_S + TAU_A_S) / 2.0

# These are not measurable via V/A/D probes; they stay hand-authored.
TAU_STRESS_S = 120.0
TAU_FRUSTRATION_PATIENT_S = 10.0     # patience=1.0
TAU_FRUSTRATION_IMPATIENT_S = 30.0   # patience=0.0
FATIGUE_RATE_PER_S = 0.0008


@dataclass
class DeltaState:
    """Increment emitted by appraisal. Add to a `DriverState` via `apply`."""

    dv: float = 0.0
    da: float = 0.0
    dd: float = 0.0
    d_stress: float = 0.0
    d_frustration: float = 0.0
    d_fatigue: float = 0.0

    def __add__(self, other: "DeltaState") -> "DeltaState":
        return DeltaState(
            dv=self.dv + other.dv,
            da=self.da + other.da,
            dd=self.dd + other.dd,
            d_stress=self.d_stress + other.d_stress,
            d_frustration=self.d_frustration + other.d_frustration,
            d_fatigue=self.d_fatigue + other.d_fatigue,
        )


@dataclass
class DriverState:
    """Mutable state vector. See module docstring for axis semantics."""

    valence: float = 0.0       # [-1, +1] pleasant ↔ unpleasant
    arousal: float = 0.0       # [-1, +1] calm ↔ activated
    dominance: float = 0.0     # [-1, +1] submissive ↔ in-control
    stress: float = 0.0        # [0, 1]   slow accumulator
    frustration: float = 0.0   # [0, 1]   event spikes, persona decay
    fatigue: float = 0.0       # [0, 1]   monotonic during session

    @classmethod
    def from_persona(cls, persona) -> "DriverState":
        """Initialise at the persona's homeostatic baseline."""
        v, a = persona.baseline_va
        return cls(valence=v, arousal=a)

    # -- Dynamics ----------------------------------------------------------

    def step(self, dt_s: float, persona) -> "DriverState":
        """Advance one tick. Mutates and returns self for chaining."""
        v_target, a_target = persona.baseline_va

        self.valence = _decay_toward(self.valence, v_target, TAU_V_S, dt_s)
        self.arousal = _decay_toward(self.arousal, a_target, TAU_A_S, dt_s)
        self.dominance = _decay_toward(self.dominance, 0.0, TAU_D_S, dt_s)

        # Frustration decay rate depends on persona.patience.
        tau_f = (
            TAU_FRUSTRATION_PATIENT_S * persona.patience
            + TAU_FRUSTRATION_IMPATIENT_S * (1.0 - persona.patience)
        )
        self.frustration = _decay_toward(self.frustration, 0.0, tau_f, dt_s)
        self.stress = _decay_toward(self.stress, 0.0, TAU_STRESS_S, dt_s)

        # Fatigue accumulates monotonically during the session.
        self.fatigue = min(1.0, self.fatigue + FATIGUE_RATE_PER_S * dt_s)

        self._clip()
        return self

    def apply(self, delta: DeltaState) -> "DriverState":
        """Add an appraisal ΔState. Mutates and returns self."""
        self.valence += delta.dv
        self.arousal += delta.da
        self.dominance += delta.dd
        self.stress += delta.d_stress
        self.frustration += delta.d_frustration
        self.fatigue += delta.d_fatigue
        self._clip()
        return self

    def _clip(self) -> None:
        self.valence = _clamp(self.valence, -1.0, 1.0)
        self.arousal = _clamp(self.arousal, -1.0, 1.0)
        self.dominance = _clamp(self.dominance, -1.0, 1.0)
        self.stress = _clamp(self.stress, 0.0, 1.0)
        self.frustration = _clamp(self.frustration, 0.0, 1.0)
        self.fatigue = _clamp(self.fatigue, 0.0, 1.0)

    # -- I/O ---------------------------------------------------------------

    def snapshot(self) -> dict:
        """Frozen dict for trace logging. Independent of mutations."""
        return asdict(self)

    def copy(self) -> "DriverState":
        return replace(self)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# State → natural-language descriptor
# ---------------------------------------------------------------------------
# Threshold-binned mapping from state scalars to a short English phrase
# the LLM can ground on. Bins beat raw numbers because "frustration: 0.6"
# isn't interpretable to a chat model trained on natural language, but
# "noticeably frustrated" is. The wording also stays *value-neutral* so
# the persona card determines tone — the descriptor just states facts.

def state_to_description(state: "DriverState") -> str:
    """Translate the symbolic state vector to a one-sentence English summary.

    Returns a semicolon-joined list of phrases capturing whichever axes
    are currently above (or below) interesting thresholds. Returns
    `"composed and present"` if no axis crosses a threshold.
    """
    parts: list[str] = []

    # Frustration first — fastest, most-felt of the accumulators.
    if state.frustration > 0.7:    parts.append("seething with frustration")
    elif state.frustration > 0.4:  parts.append("noticeably frustrated")
    elif state.frustration > 0.2:  parts.append("a little annoyed")

    # Arousal: physiological activation level.
    if state.arousal > 0.6:        parts.append("heart rate elevated")
    elif state.arousal > 0.3:      parts.append("on edge")
    elif state.arousal < -0.4:     parts.append("relaxed")

    # Valence: hedonic tone of the current moment.
    if state.valence < -0.5:       parts.append("things are going badly")
    elif state.valence < -0.2:     parts.append("in a bit of a mood")
    elif state.valence > 0.4:      parts.append("in a decent mood")

    # Dominance: feeling of agency vs. being acted upon.
    if state.dominance < -0.4:     parts.append("feeling out of control")
    elif state.dominance > 0.4:    parts.append("feeling in command")

    # Stress: slow-decay accumulator for time pressure / sustained load.
    if state.stress > 0.6:         parts.append("under heavy time pressure")
    elif state.stress > 0.3:       parts.append("aware of time slipping")

    # Fatigue rarely changes monologue unless severe.
    if state.fatigue > 0.7:        parts.append("tired")

    return "; ".join(parts) if parts else "composed and present"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _decay_toward(x: float, target: float, tau_s: float, dt_s: float) -> float:
    """Exponential pull toward `target` on time-constant `tau_s`.

    Closed-form one-step update: x' = target + (x - target) * exp(-dt/tau).
    Matches a continuous-time first-order LTI for arbitrary dt.
    """
    import math
    return target + (x - target) * math.exp(-dt_s / tau_s)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
