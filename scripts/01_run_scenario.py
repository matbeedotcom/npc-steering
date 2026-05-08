#!/usr/bin/env python3
"""Pure-Python scenario replay — no LLM, no driving sim, no GPU.

Loads a scenario JSON and a persona (preset name or JSON path), runs
the symbolic dynamics forward, and writes a JSONL trace. This is
Phase A's terminal artifact: the symbolic affect simulator validated
in isolation.

Usage:
  python scripts/01_run_scenario.py late_school_run.json calm_commuter
  python scripts/01_run_scenario.py scenarios/late_school_run.json personas/custom.json
  python scripts/01_run_scenario.py late_school_run aggressive_late --output runs/foo.jsonl

Output: runs/<scenario>__<persona>.jsonl by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the parent `lib` importable when running this file directly.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib import (
    DriverPersona,
    DriverState,
    PRESETS,
    parse_events,
)
from lib.appraisal import appraise
from lib.decisions import snapshot as decisions_snapshot
from lib.trace import TraceLogger


SCENARIOS_DIR = _ROOT / "scenarios"
PERSONAS_DIR = _ROOT / "personas"
RUNS_DIR = _ROOT / "runs"


def _resolve_scenario(arg: str) -> Path:
    """Accept a bare name, a name with .json, or an explicit path."""
    p = Path(arg)
    if p.exists():
        return p
    if not arg.endswith(".json"):
        arg = arg + ".json"
    candidate = SCENARIOS_DIR / arg
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Scenario not found: {arg}")


def _resolve_persona(arg: str) -> DriverPersona:
    """Accept a preset name, a name with .json, or an explicit path."""
    if arg in PRESETS:
        return PRESETS[arg]
    p = Path(arg)
    if p.exists():
        return DriverPersona.from_json(p)
    if not arg.endswith(".json"):
        # Try presets again with normalised name (e.g. case-insensitive).
        if arg.lower() in PRESETS:
            return PRESETS[arg.lower()]
        candidate = PERSONAS_DIR / (arg + ".json")
        if candidate.exists():
            return DriverPersona.from_json(candidate)
    raise FileNotFoundError(
        f"Persona not found: {arg}. "
        f"Available presets: {sorted(PRESETS.keys())}"
    )


def run(
    scenario_path: Path,
    persona: DriverPersona,
    output_path: Path,
    dt: float = 0.1,
) -> None:
    scenario = json.loads(scenario_path.read_text())
    events = sorted(parse_events(scenario["events"]), key=lambda e: e.t)
    duration = float(scenario.get("duration_s", 60.0))

    state = DriverState.from_persona(persona)

    with TraceLogger(
        output_path,
        scenario=scenario.get("name", scenario_path.stem),
        persona=persona,
    ) as trace:
        trace.state(0.0, state)
        trace.decision(0.0, decisions_snapshot(state, persona))

        next_event_idx = 0
        t = 0.0
        # Slight epsilon avoids skipping the last tick due to FP drift.
        while t < duration + 0.5 * dt:
            # Fire any events whose time has come.
            while (
                next_event_idx < len(events)
                and events[next_event_idx].t <= t + 1e-9
            ):
                ev = events[next_event_idx]
                delta = appraise(ev, persona, state)
                state.apply(delta)
                trace.event(ev.t, ev, delta)
                trace.state(ev.t, state)
                trace.decision(ev.t, decisions_snapshot(state, persona))
                next_event_idx += 1

            # Step dynamics forward.
            state.step(dt, persona)
            t += dt

            # Log state + decisions at coarser cadence to keep trace small.
            # (every 5 ticks ≈ 0.5s)
            if int(round(t / dt)) % 5 == 0:
                trace.state(t, state)
                trace.decision(t, decisions_snapshot(state, persona))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", help="Scenario name (in scenarios/) or path to JSON.")
    ap.add_argument(
        "persona",
        help=f"Persona preset name {sorted(PRESETS.keys())} or path to JSON.",
    )
    ap.add_argument(
        "--output", "-o", default=None,
        help="Output trace JSONL path (default: runs/<scenario>__<persona>.jsonl).",
    )
    ap.add_argument("--dt", type=float, default=0.1, help="Simulation step (s).")
    args = ap.parse_args()

    scenario_path = _resolve_scenario(args.scenario)
    persona = _resolve_persona(args.persona)

    if args.output is None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RUNS_DIR / f"{scenario_path.stem}__{persona.name}.jsonl"
    else:
        output_path = Path(args.output)

    run(scenario_path, persona, output_path, dt=args.dt)
    print(f"trace -> {output_path}")


if __name__ == "__main__":
    main()
