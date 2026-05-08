#!/usr/bin/env python3
"""Closed-loop scenario runner: symbolic dynamics + steered LLM utterances.

Phase D's single-persona artifact. Loads the model + probes + persona
cards, builds a `SteeredAgent`, and runs `lib.loop.run_scenario` for
one (scenario, persona) pair. Output: JSONL trace under `runs/`.

For the 3-persona comparison figure, see `scripts/06_persona_sweep.py`
which reuses one loaded model across all three runs.

Usage:
  python scripts/05_drive_agent.py late_school_run calm_commuter
  python scripts/05_drive_agent.py late_school_run aggressive_late --output runs/foo.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mlx_lm import load                                       # noqa: E402

from lib import PRESETS                                       # noqa: E402
from lib.agent import SteeredAgent, load_persona_cards        # noqa: E402
from lib.loop import run_scenario                             # noqa: E402
from lib.persona import DriverPersona                         # noqa: E402
from lib.probes import ProbeBundle                            # noqa: E402


SCENARIOS_DIR = _ROOT / "scenarios"
PERSONAS_DIR = _ROOT / "personas"
RUNS_DIR = _ROOT / "runs"
DEFAULT_BUNDLE = _ROOT / "artifacts" / "probes_Qwen3.5-9B-MLX-4bit.pkl"
DEFAULT_CARDS = _ROOT / "data" / "persona_cards.json"


def _resolve_scenario(arg: str) -> Path:
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
    if arg in PRESETS:
        return PRESETS[arg]
    p = Path(arg)
    if p.exists():
        return DriverPersona.from_json(p)
    candidate = PERSONAS_DIR / (arg + ".json")
    if candidate.exists():
        return DriverPersona.from_json(candidate)
    raise FileNotFoundError(
        f"Persona not found: {arg}. Available presets: {sorted(PRESETS.keys())}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", help="Scenario name (in scenarios/) or path to JSON.")
    ap.add_argument("persona",  help=f"Persona preset {sorted(PRESETS.keys())} or path to JSON.")
    ap.add_argument("--output", "-o", default=None,
                    help="Output trace JSONL (default: runs/<scenario>__<persona>.agent.jsonl).")
    ap.add_argument("--probes", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--cards",  default=str(DEFAULT_CARDS))
    ap.add_argument("--dt", type=float, default=0.1)
    args = ap.parse_args()

    scenario_path = _resolve_scenario(args.scenario)
    persona = _resolve_persona(args.persona)
    scenario = json.loads(scenario_path.read_text())

    if args.output is None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RUNS_DIR / f"{scenario_path.stem}__{persona.name}.agent.jsonl"
    else:
        output_path = Path(args.output)

    bundle = ProbeBundle.load(args.probes)
    cards = load_persona_cards(args.cards)
    print(f"[probes]  {args.probes}  selected_layers={bundle.selected_layers}")
    print(f"[model]   loading {bundle.model_id}...")
    model, tokenizer = load(bundle.model_id)
    agent = SteeredAgent(model, tokenizer, bundle, cards)

    print(f"[run]     scenario={scenario.get('name')}  persona={persona.name}  events={len(scenario.get('events', []))}")
    run_scenario(scenario, persona, output_path, agent=agent, dt=args.dt)
    print(f"[trace]   {output_path}")


if __name__ == "__main__":
    main()
