"""Driver affect simulator core library.

Persona-driven affective dynamics for simulated drivers. Designed as a
generative model of plausible-looking driver state for AV scenario
generation, not a predictive model of any specific human.

Layered architecture (each layer ~80-150 LOC, transparent on purpose):

  events    — typed event taxonomy (cut_off, merge_opportunity, ...).
  persona   — stable trait vector (temperament, patience, ...).
  state     — dynamic state (V/A/D + stress/frustration/fatigue) with decay.
  appraisal — event × persona × state → ΔState rules. (next phase)
  decisions — state × persona → driving params (gap thresh, follow dist).
              (next phase)
"""

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
    event_to_text,
    parse_events,
)
from .persona import DriverPersona, PRESETS
from .state import DeltaState, DriverState, state_to_description

__all__ = [
    "CourtesyGesture",
    "CutOff",
    "DeltaState",
    "DriverPersona",
    "DriverState",
    "Event",
    "LateForAppointment",
    "MergeOpportunity",
    "NearMiss",
    "PassengerComment",
    "PRESETS",
    "RedLight",
    "TrafficCongestion",
    "WeatherChange",
    "event_to_text",
    "parse_events",
    "state_to_description",
]
