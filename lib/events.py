"""Event taxonomy for the driver affect simulator.

Each event is a typed Pydantic model with a `t` timestamp (seconds
since scenario start) and a `type` discriminator. The `Event` union
type is what scenarios serialize and what the appraisal layer
consumes.

Adding an event type:
  1. Subclass `_EventBase` with a `Literal[...]` `type` field and any
     event-specific payload fields.
  2. Add it to the `Event` union below (the discriminator picks it up
     automatically).
  3. Add an appraisal rule in `lib/appraisal.py` (next phase).

The taxonomy intentionally stays small (~8 types). Real-world driving
has far more, but the demo is about persona × event interaction, not
event coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


class _EventBase(BaseModel):
    """Common fields. Subclasses set the `type` discriminator."""

    t: float = Field(..., description="Seconds since scenario start.")


# ---------------------------------------------------------------------------
# External-perception events (things the world does to the driver)
# ---------------------------------------------------------------------------

class CutOff(_EventBase):
    """Another vehicle merges aggressively into the ego lane."""

    type: Literal["cut_off"] = "cut_off"
    severity: float = Field(..., ge=0.0, le=1.0)
    relative_speed: float = Field(
        0.0, description="m/s of cutting vehicle relative to ego (negative = slower)."
    )


class NearMiss(_EventBase):
    """Close call without contact (pedestrian, cyclist, oncoming swerve)."""

    type: Literal["near_miss"] = "near_miss"
    severity: float = Field(..., ge=0.0, le=1.0)
    ttc_s: float = Field(..., description="Estimated time-to-collision at min distance.")


class MergeOpportunity(_EventBase):
    """A gap appears in adjacent traffic. Decision-coupling input."""

    type: Literal["merge_opportunity"] = "merge_opportunity"
    gap_seconds: float = Field(..., ge=0.0)


class TrafficCongestion(_EventBase):
    """Sustained slow-speed condition. Persistent low-grade stressor."""

    type: Literal["traffic_congestion"] = "traffic_congestion"
    duration_s: float = Field(..., ge=0.0)


class RedLight(_EventBase):
    """Stopped at a signal. Whether this perturbs state depends on persona."""

    type: Literal["red_light"] = "red_light"
    expected_wait_s: float = Field(30.0, ge=0.0)


class WeatherChange(_EventBase):
    """Visibility / surface friction degradation."""

    type: Literal["weather_change"] = "weather_change"
    condition: Literal["clear", "rain", "snow", "fog"] = "rain"
    intensity: float = Field(0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Internal / contextual events (driver's situation, not the world)
# ---------------------------------------------------------------------------

class LateForAppointment(_EventBase):
    """Time pressure. Fired once at scenario start; persistent context."""

    type: Literal["late_for_appointment"] = "late_for_appointment"
    minutes_behind: float = Field(..., ge=0.0)
    importance: float = Field(..., ge=0.0, le=1.0)


class PassengerComment(_EventBase):
    """In-cabin social pressure (kid asks 'are we late?', etc.)."""

    type: Literal["passenger_comment"] = "passenger_comment"
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    intensity: float = Field(0.5, ge=0.0, le=1.0)


class CourtesyGesture(_EventBase):
    """A favourable interaction with another road user.

    Counter-balances the negative-skewed core taxonomy. Driver
    behaviour datasets oversample friction; real driving has plenty
    of small positive moments, and the closed loop should be able
    to register them. The `discretionary` parameter scales the
    affective lift: a wave from someone who didn't have to give it
    is more moving than a yield in a situation where they had no
    choice.
    """

    type: Literal["courtesy_gesture"] = "courtesy_gesture"
    gesture: Literal[
        "let_merge",            # they made room and waved you in
        "wave_thanks",          # they thanked you for letting them in
        "yield_unnecessarily",  # gestured "go ahead" when they had right of way
        "polite_pass",          # smooth pass without crowding
    ] = "let_merge"
    intensity: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Magnitude of the gesture. 0.3=minor, 0.7=notable, 1.0=memorable.",
    )
    discretionary: float = Field(
        0.7, ge=0.0, le=1.0,
        description="How much they went out of their way. 0=just being decent, 1=actively kind.",
    )


# ---------------------------------------------------------------------------
# Discriminated union + parser
# ---------------------------------------------------------------------------

Event = Annotated[
    Union[
        CutOff,
        NearMiss,
        MergeOpportunity,
        TrafficCongestion,
        RedLight,
        WeatherChange,
        LateForAppointment,
        PassengerComment,
        CourtesyGesture,
    ],
    Field(discriminator="type"),
]

_EVENT_LIST_ADAPTER = TypeAdapter(List[Event])


# ---------------------------------------------------------------------------
# Event → natural-language descriptor
# ---------------------------------------------------------------------------
# Per-event-type templates that turn a structured `Event` into a short
# English sentence describing what just happened. Used by `SteeredAgent`
# to slot the event into the user-turn of the chat prompt. Severity- and
# magnitude-conditional phrasing keeps the descriptor concrete.

def event_to_text(ev) -> str:
    """One-sentence English description of an event payload."""
    if isinstance(ev, CutOff):
        if ev.severity > 0.7:
            severity = "very aggressively"
        elif ev.severity > 0.3:
            severity = "with little warning"
        else:
            severity = "but with room to spare"
        if ev.relative_speed < -5:
            speed = "much slower than you"
        elif ev.relative_speed < -1:
            speed = "a bit slower than you"
        else:
            speed = "at roughly your speed"
        return f"A vehicle just merged in front of you {severity}, {speed}."

    if isinstance(ev, NearMiss):
        return (
            f"You came within {ev.ttc_s:.1f} seconds of a collision. "
            f"You had to swerve or brake hard."
        )

    if isinstance(ev, MergeOpportunity):
        return f"A {ev.gap_seconds:.1f}-second gap just opened in the next lane."

    if isinstance(ev, TrafficCongestion):
        return f"Traffic has been crawling for {ev.duration_s:.0f} seconds with no end in sight."

    if isinstance(ev, RedLight):
        return f"You've stopped at a red light. The wait looks like about {ev.expected_wait_s:.0f} seconds."

    if isinstance(ev, WeatherChange):
        if ev.condition == "clear":
            return "The weather just cleared up."
        return (
            f"The weather just shifted to {ev.condition} "
            f"({'heavy' if ev.intensity > 0.6 else 'moderate' if ev.intensity > 0.3 else 'light'})."
        )

    if isinstance(ev, LateForAppointment):
        if ev.importance > 0.7:
            tag = "and the appointment matters"
        elif ev.importance > 0.3:
            tag = "and it's somewhat important"
        else:
            tag = "but it's a low-stakes appointment"
        return f"You're {ev.minutes_behind:.0f} minutes behind schedule, {tag}."

    if isinstance(ev, PassengerComment):
        if ev.valence < -0.4:
            return "Your passenger just said something pointed and unhappy."
        if ev.valence < -0.1:
            return "Your passenger just made a small, annoyed remark."
        if ev.valence > 0.4:
            return "Your passenger just said something genuinely nice."
        if ev.valence > 0.1:
            return "Your passenger just made a friendly comment."
        return "Your passenger just said something neutral."

    if isinstance(ev, CourtesyGesture):
        warm = ev.discretionary > 0.6
        if ev.gesture == "let_merge":
            return (
                "Another driver let you merge in front of them, "
                "and even smiled and waved — they didn't have to."
                if warm else
                "Another driver let you merge in front of them."
            )
        if ev.gesture == "wave_thanks":
            return (
                "The driver you just let in waved a clear, genuine thanks."
                if warm else
                "The driver you just let in flicked a quick thanks."
            )
        if ev.gesture == "yield_unnecessarily":
            return (
                "A driver slowed and gestured for you to go first — "
                "they had right of way and gave it up anyway."
                if warm else
                "A driver waved you through ahead of them."
            )
        if ev.gesture == "polite_pass":
            return "A driver overtook you smoothly and patiently, no crowding."
        return "Another road user just did something kind."

    return f"An event of type {ev.type!r} occurred."


def parse_events(raw: list[dict] | str | Path) -> list[Event]:
    """Accept a list of dicts, a JSON string, or a path to a JSON file.

    JSON-file shape: either a top-level list of event dicts, or an
    object with an `events` key.
    """
    if isinstance(raw, (str, Path)):
        path = Path(raw)
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = json.loads(str(raw))
        if isinstance(data, dict) and "events" in data:
            data = data["events"]
        raw = data
    return _EVENT_LIST_ADAPTER.validate_python(raw)
