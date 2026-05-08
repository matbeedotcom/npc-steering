#!/usr/bin/env python3
"""Smoke test for `SteeredAgent`: one turn per persona × one event.

Verifies end-to-end Phase C wiring: persona card → chat prompt →
steered generation → felt-VAD readout → re-appraisal Δ. Run before
moving on to Phase D's full closed-loop scenario replay.

Usage:
  python scripts/04_test_agent.py
  python scripts/04_test_agent.py --probes artifacts/probes_test.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mlx_lm import load                                           # noqa: E402

from lib import DriverState, PRESETS                              # noqa: E402
from lib.agent import SteeredAgent, load_persona_cards            # noqa: E402
from lib.events import CutOff, LateForAppointment                 # noqa: E402
from lib.probes import ProbeBundle                                # noqa: E402


DEFAULT_BUNDLE = _ROOT / "artifacts" / "probes_Qwen3.5-9B-MLX-4bit.pkl"
DEFAULT_CARDS = _ROOT / "data" / "persona_cards.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--cards", default=str(DEFAULT_CARDS))
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    bundle = ProbeBundle.load(args.probes)
    cards = load_persona_cards(args.cards)
    print(f"[probes]  {args.probes}")
    print(f"          model={bundle.model_id}")
    print(f"          selected_layers={bundle.selected_layers}")
    print(f"[cards]   {sorted(cards.keys())}")
    print(f"[model]   loading {bundle.model_id}...")

    model, tokenizer = load(bundle.model_id)
    agent = SteeredAgent(model, tokenizer, bundle, cards)

    # Construct a charged moment: late + just got cut off.
    # Build state from each persona's baseline and apply a couple of
    # synthetic appraisals so V/A/D are nontrivial when steering is
    # computed.
    from lib.appraisal import appraise

    cut_off = CutOff(t=12.0, severity=0.7, relative_speed=-8.5)
    late = LateForAppointment(t=0.0, minutes_behind=8.0, importance=0.9)

    print("\n=== three personas, identical scenario "
          "(8 min late + just cut off, severity 0.7) ===\n")
    for persona_name in ["calm_commuter", "aggressive_late", "anxious_new_driver"]:
        persona = PRESETS[persona_name]
        state = DriverState.from_persona(persona)
        # Synthetic event history to charge the state.
        state.apply(appraise(late, persona, state))
        state.apply(appraise(cut_off, persona, state))

        print(f"--- {persona_name} ---")
        print(f"  state:    V={state.valence:+.2f}  A={state.arousal:+.2f}  "
              f"D={state.dominance:+.2f}  F={state.frustration:.2f}  "
              f"S={state.stress:.2f}")

        response = agent.respond(state, persona, cut_off, max_tokens=args.max_tokens)
        print(f"  intended: V={response.intended_VAD['V']:+.2f}  "
              f"A={response.intended_VAD['A']:+.2f}  "
              f"D={response.intended_VAD['D']:+.2f}")
        print(f"  felt:     V={response.felt_VAD['V']:+.2f}  "
              f"A={response.felt_VAD['A']:+.2f}  "
              f"D={response.felt_VAD['D']:+.2f}")
        print(f"  Δstate:   dV={response.delta_state.dv:+.3f}  "
              f"dA={response.delta_state.da:+.3f}  "
              f"dD={response.delta_state.dd:+.3f}")
        print(f"  → {response.text!r}")
        print()


if __name__ == "__main__":
    main()
