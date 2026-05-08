#!/usr/bin/env python3
"""Persona-sweep figure: same scenario × N personas → side-by-side comparison.

This is the money-shot artifact for the project writeup. Loads the model
once, runs `lib.loop.run_scenario` for each persona, then renders a
single figure with N rows (one per persona). Each row shows V/A/D +
frustration trajectory with event markers and time-stamped utterances.

Output: PNG saved under `artifacts/`. Trace JSONLs are written under
`runs/` and reused on subsequent --no-rerun invocations (so iterating
on the figure layout doesn't require re-running the LLM).

Usage:
  python scripts/06_persona_sweep.py
  python scripts/06_persona_sweep.py --scenario highway_merge_storm
  python scripts/06_persona_sweep.py --no-rerun                  # re-render only
  python scripts/06_persona_sweep.py --personas calm_commuter aggressive_late
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mlx_lm import load                                       # noqa: E402

from lib import PRESETS                                       # noqa: E402
from lib.agent import SteeredAgent, load_persona_cards        # noqa: E402
from lib.loop import run_scenario                             # noqa: E402
from lib.probes import ProbeBundle                            # noqa: E402


SCENARIOS_DIR = _ROOT / "scenarios"
RUNS_DIR = _ROOT / "runs"
ARTIFACTS_DIR = _ROOT / "artifacts"
DEFAULT_BUNDLE = ARTIFACTS_DIR / "probes_Qwen3.5-9B-MLX-4bit.pkl"
DEFAULT_CARDS = _ROOT / "data" / "persona_cards.json"
DEFAULT_PERSONAS = ["calm_commuter", "aggressive_late", "anxious_new_driver"]


# ---------------------------------------------------------------------------
# Trace reading
# ---------------------------------------------------------------------------

def load_trace(path: Path) -> dict:
    """Load a JSONL trace file and bucket records by kind."""
    out = {"meta": {}, "state": [], "event": [], "decision": [], "utterance": []}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.pop("kind")
            if kind == "meta":
                out["meta"] = rec
            else:
                out[kind].append(rec)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_LINE_COLORS = {
    "valence":     "#1f77b4",   # blue
    "arousal":     "#d62728",   # red
    "dominance":   "#2ca02c",   # green
    "frustration": "#ff7f0e",   # orange
}


def plot_sweep(
    traces_by_persona: dict[str, dict],
    scenario_name: str,
    output_path: Path,
) -> None:
    """N rows × 2 columns: left = state trajectory, right = utterance log.

    Splitting the visual (state lines) from the textual (utterances)
    avoids the previous layout's overlap problem — long utterances
    stomping on each other below the x-axis. Now they live in a
    text panel sized to fit them.
    """
    n = len(traces_by_persona)
    fig = plt.figure(figsize=(17, 3.6 * n + 0.8))
    gs = fig.add_gridspec(
        nrows=n, ncols=2,
        width_ratios=[2.4, 1.0],
        hspace=0.45, wspace=0.06,
    )

    state_axes = []
    for row, (persona_name, trace) in enumerate(traces_by_persona.items()):
        ax = fig.add_subplot(gs[row, 0])
        state_axes.append(ax)

        # ─── State trajectory ─────────────────────────────────────────
        s = trace["state"]
        ts = np.array([r["t"] for r in s])
        for axis_name in ("valence", "arousal", "dominance", "frustration"):
            ys = np.array([r[axis_name] for r in s])
            ax.plot(
                ts, ys, color=_LINE_COLORS[axis_name], linewidth=1.7,
                label=axis_name.capitalize() if row == 0 else None,
            )

        # Event markers — short label below the line, no top label
        for ev in trace["event"]:
            ax.axvline(ev["t"], color="grey", linestyle=":", linewidth=0.9, alpha=0.6)
            ax.text(
                ev["t"], 1.04, ev["type"].replace("_", " "),
                ha="center", va="bottom", fontsize=7.5, color="dimgrey",
                rotation=20, transform=ax.get_xaxis_transform(),
            )

        # ─── Persona heading ──────────────────────────────────────────
        traits = trace["meta"].get("persona_traits", {})
        trait_str = (
            f"temp={traits.get('temperament', 0):+.1f}  "
            f"pat={traits.get('patience', 0):.1f}  "
            f"risk={traits.get('risk_preference', 0):+.1f}  "
            f"react={traits.get('reactivity', 0):.1f}"
        )
        ax.set_title(
            f"{persona_name}    ({trait_str})",
            loc="left", fontsize=11.5, fontweight="bold", pad=14,
        )

        # ─── Aesthetics ───────────────────────────────────────────────
        ax.set_ylabel("state value", fontsize=9)
        ax.set_ylim(-1.15, 1.15)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
        ax.grid(True, axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if row != n - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time (s)", fontsize=10)

        # ─── Utterance text panel (right column) ──────────────────────
        tax = fig.add_subplot(gs[row, 1])
        tax.set_xlim(0, 1)
        tax.set_ylim(0, 1)
        tax.axis("off")
        tax.set_title("Internal monologue", loc="left",
                      fontsize=10, fontweight="bold", pad=10)

        utts = trace["utterance"]
        events = {e["t"]: e["type"] for e in trace["event"]}
        n_u = max(len(utts), 1)
        # Stack utterances top-down with even spacing
        for i, u in enumerate(utts):
            y = 0.96 - (i + 0.5) * (0.96 / n_u)
            ev_type = events.get(u["t"], "")
            line = u["text"].replace("\n", " ").strip()
            if len(line) > 110:
                line = line[:107] + "..."
            tax.text(
                0.0, y,
                f"t={u['t']:>4.1f}s  ·  {ev_type}",
                fontsize=8.0, color="dimgrey",
                ha="left", va="top",
            )
            tax.text(
                0.0, y - 0.04,
                f'"{line}"',
                fontsize=8.5, color="black",
                ha="left", va="top",
                wrap=True,
            )

    # ─── Legend on the first state axis ──────────────────────────────
    state_axes[0].legend(
        loc="upper right", ncol=4, fontsize=8, frameon=False,
        bbox_to_anchor=(1.0, 1.16),
    )

    fig.suptitle(
        f"Persona sweep: scenario = '{scenario_name}'  "
        f"(same events, three personas, three internal-monologue trajectories)",
        fontsize=13, y=0.995,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {output_path}")


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="late_school_run",
                    help="Scenario name (in scenarios/) or path to JSON.")
    ap.add_argument("--personas", nargs="+", default=DEFAULT_PERSONAS,
                    help="Persona preset names to compare.")
    ap.add_argument("--probes", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--cards",  default=str(DEFAULT_CARDS))
    ap.add_argument("--no-rerun", action="store_true",
                    help="Skip the LLM runs, re-render the figure from existing traces.")
    ap.add_argument("--output", default=None,
                    help="Output PNG (default: artifacts/persona_sweep_<scenario>.png).")
    args = ap.parse_args()

    # Resolve scenario path.
    scen_arg = args.scenario
    p = Path(scen_arg)
    if not p.exists():
        if not scen_arg.endswith(".json"):
            scen_arg = scen_arg + ".json"
        p = SCENARIOS_DIR / scen_arg
    if not p.exists():
        raise FileNotFoundError(f"Scenario not found: {args.scenario}")
    scenario = json.loads(p.read_text())
    scenario_stem = p.stem

    # Default output path.
    if args.output is None:
        output_png = ARTIFACTS_DIR / f"persona_sweep_{scenario_stem}.png"
    else:
        output_png = Path(args.output)

    # Run each persona unless --no-rerun.
    trace_paths: List[Path] = []
    if not args.no_rerun:
        bundle = ProbeBundle.load(args.probes)
        cards = load_persona_cards(args.cards)
        print(f"[probes]  {args.probes}  selected_layers={bundle.selected_layers}")
        print(f"[model]   loading {bundle.model_id}...")
        model, tokenizer = load(bundle.model_id)
        agent = SteeredAgent(model, tokenizer, bundle, cards)

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        for persona_name in args.personas:
            if persona_name not in PRESETS:
                raise KeyError(f"Unknown persona preset {persona_name!r}")
            persona = PRESETS[persona_name]
            output_jsonl = RUNS_DIR / f"{scenario_stem}__{persona_name}.agent.jsonl"
            print(f"[run]     {persona_name} → {output_jsonl}")
            run_scenario(scenario, persona, output_jsonl, agent=agent)
            trace_paths.append(output_jsonl)
    else:
        for persona_name in args.personas:
            trace_paths.append(RUNS_DIR / f"{scenario_stem}__{persona_name}.agent.jsonl")

    # Load all traces, render figure.
    traces_by_persona = {}
    for persona_name, tpath in zip(args.personas, trace_paths):
        if not tpath.exists():
            raise FileNotFoundError(
                f"Trace missing for {persona_name}: {tpath}. Drop --no-rerun."
            )
        traces_by_persona[persona_name] = load_trace(tpath)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_sweep(traces_by_persona, scenario.get("name", scenario_stem), output_png)


if __name__ == "__main__":
    main()
