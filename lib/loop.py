"""Closed-loop scenario runner.

Combines the symbolic discrete-event simulator (Phase A) with the
steered LLM agent (Phase C). Per event:

  1. Symbolic appraisal:  ΔState_appraisal = appraise(event, persona, state)
                          state.apply(ΔState_appraisal)
  2. LLM utterance:       response = agent.respond(state, persona, event)
  3. Re-appraisal:        state.apply(response.delta_state)

Between events, only state.step(dt, persona) runs — pure dynamics,
no LLM calls. The LLM cost is paid per-event, not per-tick.

`agent` may be None to skip the LLM path entirely (then this is just
the Phase A pure-symbolic runner). Useful for fast iteration on the
symbolic side.

Output is JSONL via `TraceLogger`. The figure script reads these
traces back and renders the persona-sweep comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .agent import SteeredAgent
from .appraisal import appraise
from .decisions import snapshot as decisions_snapshot
from .events import parse_events
from .persona import DriverPersona
from .state import DriverState
from .trace import TraceLogger


def run_scenario(
    scenario: dict,
    persona: DriverPersona,
    output_path: Path | str,
    *,
    agent: Optional[SteeredAgent] = None,
    dt: float = 0.1,
    state_log_every_n_ticks: int = 5,
    arousal_gate: bool = False,
) -> None:
    """Run a scenario end-to-end and write a JSONL trace.

    `scenario` is a parsed dict (from `json.loads(scenario_path.read_text())`).
    `agent=None` skips the LLM branch — useful for quick symbolic-only runs.
    `arousal_gate=True` activates the Shangguan 2025 high-arousal
    amplifier in the appraisal layer.
    """
    events = sorted(parse_events(scenario["events"]), key=lambda e: e.t)
    duration = float(scenario.get("duration_s", 60.0))

    state = DriverState.from_persona(persona)

    with TraceLogger(
        output_path,
        scenario=scenario.get("name"),
        persona=persona,
    ) as trace:
        trace.state(0.0, state)
        trace.decision(0.0, decisions_snapshot(state, persona))

        next_event_idx = 0
        t = 0.0
        while t < duration + 0.5 * dt:
            # Fire any events whose time has come.
            while (
                next_event_idx < len(events)
                and events[next_event_idx].t <= t + 1e-9
            ):
                ev = events[next_event_idx]

                # 1. Symbolic appraisal.
                delta_appraisal = appraise(ev, persona, state, arousal_gate=arousal_gate)
                state.apply(delta_appraisal)
                trace.event(ev.t, ev, delta_appraisal)
                trace.state(ev.t, state)

                # 2-3. Closed loop (only if agent provided).
                if agent is not None:
                    response = agent.respond(state, persona, ev)
                    trace.utterance(ev.t, response.text, felt_vad=response.felt_VAD)
                    state.apply(response.delta_state)
                    # Re-log state post-re-appraisal so the trace shows
                    # the symbolic correction the LLM induced.
                    trace.state(ev.t, state)

                trace.decision(ev.t, decisions_snapshot(state, persona))
                next_event_idx += 1

            # Advance dynamics one tick.
            state.step(dt, persona)
            t += dt

            # Sparser logging between events to keep the trace readable.
            if int(round(t / dt)) % state_log_every_n_ticks == 0:
                trace.state(t, state)
                trace.decision(t, decisions_snapshot(state, persona))
