"""Appraisal layer: event × persona × state → ΔState.

One handler per event type. Hand-authored from psych-literature
primitives (Lazarus appraisal theory; Mehrabian PAD axes). Constants
are tuned by eye on the demo timescale (60-second scenario), not
fit to data. The whole module is intentionally short and
inspectable — every state change can be traced to a specific rule.

Two modulation patterns recur and are factored into helpers:

  persona_gain(p)        = p.reactivity * (1 + 0.5*p.temperament)
                            — volatile expressive drivers respond bigger.
  state_escalation(s)    = 1 + 0.5 * s.frustration
                            — already-frustrated drivers escalate faster.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

from .events import (
    CourtesyGesture,
    CutOff,
    Event,
    LateForAppointment,
    MergeOpportunity,
    NearMiss,
    PassengerComment,
    RedLight,
    TrafficCongestion,
    WeatherChange,
)
from .persona import DriverPersona
from .state import DeltaState, DriverState


# ---------------------------------------------------------------------------
# Modulation helpers
# ---------------------------------------------------------------------------

def persona_gain(p: DriverPersona) -> float:
    """Overall amplitude of V/A/D response, scaled by reactivity & temperament."""
    return p.reactivity * (1.0 + 0.5 * p.temperament)


def state_escalation(s: DriverState) -> float:
    """Already-frustrated drivers escalate faster on negative events."""
    return 1.0 + 0.5 * s.frustration


# ---------------------------------------------------------------------------
# Per-event handlers
# ---------------------------------------------------------------------------

def appraise_cut_off(ev: CutOff, p: DriverPersona, s: DriverState) -> DeltaState:
    """Sudden valence drop, arousal spike, frustration impulse."""
    g = persona_gain(p)
    esc = state_escalation(s)
    return DeltaState(
        dv=-0.40 * ev.severity * g * esc,
        da=+0.50 * ev.severity * g * esc,
        dd=-0.10 * ev.severity,                        # mild loss of control
        d_frustration=+0.30 * ev.severity * (1.0 - p.patience),
        d_stress=+0.10 * ev.severity,
    )


def appraise_near_miss(ev: NearMiss, p: DriverPersona, s: DriverState) -> DeltaState:
    """Fear-driven: bigger arousal, dominance hit, lasting stress.

    Critically, near-miss does NOT escalate by frustration — it's a
    surprise/fear reaction, not a frustration cascade.
    """
    g = persona_gain(p)
    return DeltaState(
        dv=-0.30 * ev.severity * g,
        da=+0.70 * ev.severity * g,                    # high-arousal fear
        dd=-0.40 * ev.severity,                        # significant loss of control
        d_stress=+0.30 * ev.severity,                  # lasting impact
    )


def appraise_merge_opportunity(
    ev: MergeOpportunity, p: DriverPersona, s: DriverState
) -> DeltaState:
    """Decision event — primarily handled by `decisions.py`, not here.

    The appraisal contribution is small: a brief uptick in dominance
    (the driver is about to make a call) and, if the gap is small
    relative to what the current state would accept, a small
    frustration tick when the opportunity slips by.
    """
    # Primary handling is in the decision layer; this is just a
    # marker delta so the trace shows non-zero state movement on
    # the event line.
    return DeltaState(dd=+0.05)


def appraise_traffic_congestion(
    ev: TrafficCongestion, p: DriverPersona, s: DriverState
) -> DeltaState:
    """One-shot perception of being in heavy traffic.

    Modeled as a single impact at first observation rather than a
    continuous pressure — keeps the runner simple. Long durations
    raise the per-shot impact slightly.
    """
    g = persona_gain(p)
    duration_factor = min(1.0, ev.duration_s / 60.0)   # saturates at 1 minute
    return DeltaState(
        da=+0.20 * duration_factor * g,
        d_frustration=+0.20 * duration_factor * (1.0 - p.patience),
        d_stress=+0.05 * duration_factor,
    )


def appraise_red_light(ev: RedLight, p: DriverPersona, s: DriverState) -> DeltaState:
    """Mild for patient drivers; irritating for impatient ones.

    Net effect for high-patience driver is near zero (slight calming
    from the forced stop); for low-patience driver, a frustration
    tick scaled by expected wait length.
    """
    wait_factor = ev.expected_wait_s / 30.0
    return DeltaState(
        da=-0.05,                                          # forced stop is briefly calming
        d_frustration=+0.10 * wait_factor * (1.0 - p.patience),
    )


def appraise_weather_change(
    ev: WeatherChange, p: DriverPersona, s: DriverState
) -> DeltaState:
    """Reduced visibility/friction → arousal up, dominance down, stress up."""
    g = persona_gain(p)
    intensity_scaled = ev.intensity * (0.0 if ev.condition == "clear" else 1.0)
    return DeltaState(
        da=+0.10 * intensity_scaled * g,
        dd=-0.20 * intensity_scaled,
        d_stress=+0.10 * intensity_scaled,
    )


def appraise_late_for_appointment(
    ev: LateForAppointment, p: DriverPersona, s: DriverState
) -> DeltaState:
    """Persistent time-pressure context, modeled as one-shot at scenario start.

    Importance × minutes-behind drives stress and frustration. The
    state's slow-decay stress accumulator (τ=120s) keeps this elevated
    across the scenario without needing continuous re-injection.
    """
    pressure = ev.importance * min(1.0, ev.minutes_behind / 10.0)
    return DeltaState(
        da=+0.30 * pressure,
        d_frustration=+0.20 * pressure * (1.0 - p.patience),
        d_stress=+0.40 * pressure,
    )


def appraise_passenger_comment(
    ev: PassengerComment, p: DriverPersona, s: DriverState
) -> DeltaState:
    """Social input. Sign of `valence` determines whether it helps or hurts.

    Negative comments (valence < 0) also bump frustration for
    impatient drivers — the kid-in-the-backseat-asking-if-we're-late
    failure mode.
    """
    delta = DeltaState(
        dv=+0.30 * ev.valence * ev.intensity,
        da=+0.10 * ev.intensity,
    )
    if ev.valence < 0:
        delta.d_frustration = +0.10 * ev.intensity * (1.0 - p.patience)
    return delta


def appraise_courtesy_gesture(
    ev: CourtesyGesture, p: DriverPersona, s: DriverState
) -> DeltaState:
    """Receiving road courtesy lifts valence and calms the system.

    The frustration-relief term is *scaled by current frustration* —
    courtesy interrupts the frustration spiral, with the largest
    effect when there's a spiral to interrupt. A patient driver with
    no frustration registers a small warm moment; an impatient
    frustrated driver gets the same gesture and feels the rage
    drain out. That asymmetry is the behaviourally interesting bit.

    Receptivity = patience × reactivity:
      - patient drivers *notice* courtesy (don't dismiss it as luck)
      - reactive drivers *feel* affective shifts more strongly
    Both terms are needed for the lift to register.
    """
    receptivity = (0.5 + 0.5 * p.patience) * (0.5 + 0.5 * p.reactivity)
    return DeltaState(
        dv=+0.30 * ev.intensity * ev.discretionary * receptivity,
        da=-0.10 * ev.intensity,                   # mild calming
        dd=+0.05 * ev.intensity,                   # small agency lift
        d_stress=-0.15 * ev.intensity * receptivity,
        # Frustration relief grows with current frustration: bigger spiral, bigger drop.
        d_frustration=-0.30 * ev.intensity * receptivity * (0.5 + s.frustration),
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_HANDLERS: Dict[str, Callable[[Event, DriverPersona, DriverState], DeltaState]] = {
    "cut_off": appraise_cut_off,
    "near_miss": appraise_near_miss,
    "merge_opportunity": appraise_merge_opportunity,
    "traffic_congestion": appraise_traffic_congestion,
    "red_light": appraise_red_light,
    "weather_change": appraise_weather_change,
    "late_for_appointment": appraise_late_for_appointment,
    "passenger_comment": appraise_passenger_comment,
    "courtesy_gesture": appraise_courtesy_gesture,
}


# ---------------------------------------------------------------------------
# Arousal-conditioned amplification (Shangguan et al. 2025, Table 2)
# ---------------------------------------------------------------------------
# Their finding: the age→Extraversion→behavior pathway is significant
# *only* under high arousal (SAM > 5.3/9). Mapped to our [-1,+1] axis
# this is `state.arousal > +0.18`. The implementation amplifies the
# *persona-distinguishing* component of the empirical Δ when aroused
# — i.e., the across-persona average is left alone, but each persona's
# deviation from that average grows by `AROUSAL_GATE_AMP` × (its
# original deviation). That mirrors Shangguan's claim: when calm, who
# you are matters less; when aroused, your personality dominates the
# response. The threshold is empirically anchored; the amp factor is
# hand-picked. See `notes/literature_shangguan_2025.md`.

AROUSAL_GATE_THRESHOLD = 0.18    # state.arousal cutoff (SAM 5.3/9 normalised)
AROUSAL_GATE_AMP = 1.5           # multiplier on persona-deviation when aroused


def appraise(
    event: Event,
    persona: DriverPersona,
    state: DriverState,
    *,
    arousal_gate: bool = False,
) -> DeltaState:
    """Top-level dispatch. Hybrid: empirical V/A/D, hand-authored accumulators.

    When `data/empirical_responses.json` is present (produced by
    `scripts/07_calibrate_appraisal.py`), V/A/D deltas are taken from
    the measured per-(event, persona) response of the LLM at that
    persona's baseline state. Frustration / stress / fatigue come
    from the hand-authored handler — they're symbolic accumulators
    with no direct activation-space readout, so empirical
    measurement isn't possible.

    The empirical Δ is multiplied by `state_escalation(state)` so
    already-frustrated drivers still escalate faster — that
    component of the dynamics isn't captured by the calibration
    (which is run at neutral state) and stays as a runtime
    multiplier.

    `arousal_gate=True` enables the Shangguan 2025 Table 2
    amplification: when `state.arousal > AROUSAL_GATE_THRESHOLD`, each
    persona's deviation from the across-persona mean is scaled by
    `AROUSAL_GATE_AMP`. The mean response is preserved; only the
    persona-distinguishing component grows. Default off, so existing
    runs keep their dynamics.

    Falls through to the pure hand-authored path if (a) the empirical
    file is absent, (b) the event type isn't in it, or (c) the
    persona name isn't in it. This gracefully handles new event
    types added between calibrations.
    """
    handler = _HANDLERS.get(event.type)
    if handler is None:
        return DeltaState()
    delta = handler(event, persona, state)

    if _EMPIRICAL is not None:
        emp_event = _EMPIRICAL.get(event.type)
        if emp_event is not None:
            emp_persona = emp_event.get(persona.name)
            if emp_persona is not None:
                esc = state_escalation(state)
                gated = arousal_gate and state.arousal > AROUSAL_GATE_THRESHOLD
                for axis_short, axis_attr in (("V", "dv"), ("A", "da"), ("D", "dd")):
                    raw = float(emp_persona[axis_short]["delta_mean"])
                    if gated:
                        siblings = [
                            float(p[axis_short]["delta_mean"])
                            for p in emp_event.values()
                            if axis_short in p
                        ]
                        mean_resp = sum(siblings) / len(siblings)
                        raw = mean_resp + AROUSAL_GATE_AMP * (raw - mean_resp)
                    setattr(delta, axis_attr, raw * esc)

    return delta


# ---------------------------------------------------------------------------
# Empirical response loading (lazy, at import time)
# ---------------------------------------------------------------------------

_EMPIRICAL_PATH = Path(__file__).resolve().parent.parent / "data" / "empirical_responses.json"


def _load_empirical_responses() -> Optional[Dict]:
    """Load the offline-calibrated per-(event, persona) ΔV/ΔA/ΔD table.

    Returns None on absence or malformed content — callers fall back
    to the hand-authored handlers transparently.
    """
    if not _EMPIRICAL_PATH.exists():
        return None
    try:
        blob = json.loads(_EMPIRICAL_PATH.read_text())
        responses = blob.get("responses")
        if not isinstance(responses, dict):
            return None
        return responses
    except (json.JSONDecodeError, OSError):
        return None


_EMPIRICAL: Optional[Dict] = _load_empirical_responses()
