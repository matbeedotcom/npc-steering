#!/usr/bin/env python3
"""Card-vs-steering ablation: a 2×2 attribution of behavioural distinctness.

Decomposes where the closed loop's persona differentiation comes from:

  | condition          | persona card | activation steering |
  |--------------------|:------------:|:-------------------:|
  | card+steer         |      ✓       |          ✓          |
  | card_only          |      ✓       |          —          |
  | steer_only         |      —       |          ✓          |
  | neither            |      —       |          —          |

Card off → minimal generic system prompt (task framing only, no
character/voice/reference lines). Steering off → no activation
offsets applied (the wrappers see an empty offset dict).

Three metrics reported:

  1. Persona-distinctness — mean pairwise L2 distance across personas
     on (V, A, D) trajectories. Larger = personas behave more
     differently from one another.
  2. State-tracking correlation — Pearson r between intended V (the
     symbolic state at utterance time) and felt V (read back from the
     LLM's hidden state via probe projection). Larger = the closed
     loop keeps intended-and-expressed in sync.
  3. Per-condition mean utterance V/A/D - to surface whether either
     channel pushes the read-back projection.

Output: 2×2 figure of persona-distinctness over time, plus a
markdown report at `notes/card_vs_steering_ablation.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mlx_lm import load                                   # noqa: E402

from lib import PRESETS                                   # noqa: E402
from lib.agent import SteeredAgent, load_persona_cards    # noqa: E402
from lib.loop import run_scenario                         # noqa: E402
from lib.probes import ProbeBundle                        # noqa: E402


SCENARIOS_DIR = _ROOT / "scenarios"
RUNS_DIR = _ROOT / "runs"
ARTIFACTS_DIR = _ROOT / "artifacts"
NOTES_DIR = _ROOT / "notes"
DEFAULT_BUNDLE = ARTIFACTS_DIR / "probes_Qwen3.5-9B-MLX-4bit.pkl"
DEFAULT_CARDS = _ROOT / "data" / "persona_cards.json"
DEFAULT_PERSONAS = ["calm_commuter", "aggressive_late", "anxious_new_driver"]
DEFAULT_SCENARIO = "mixed_emotions"

CONDITIONS: Dict[str, Tuple[bool, bool]] = {
    # tag → (use_card, use_steering)
    "card+steer": (True, True),
    "card_only":  (True, False),
    "steer_only": (False, True),
    "neither":    (False, False),
}


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------

def load_trace(path: Path) -> dict:
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


def trace_state_array(trace: dict) -> Tuple[np.ndarray, np.ndarray]:
    s = trace["state"]
    ts = np.array([r["t"] for r in s])
    cols = np.column_stack([
        [r["valence"] for r in s],
        [r["arousal"] for r in s],
        [r["dominance"] for r in s],
        [r["frustration"] for r in s],
    ])
    return ts, cols


def resample_to_grid(
    ts: np.ndarray, cols: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    out = np.zeros((len(grid), cols.shape[1]))
    idx = np.searchsorted(ts, grid, side="right") - 1
    idx = np.clip(idx, 0, len(ts) - 1)
    for j in range(cols.shape[1]):
        out[:, j] = cols[idx, j]
    return out


def pairwise_distance_over_time(
    persona_traces: Dict[str, np.ndarray]
) -> np.ndarray:
    """Mean pairwise L2 across personas at each tick on (V, A, D)."""
    keys = sorted(persona_traces.keys())
    n = len(keys)
    if n < 2:
        return np.zeros(next(iter(persona_traces.values())).shape[0])
    series = np.stack([persona_traces[k][:, :3] for k in keys])
    out = np.zeros(series.shape[1])
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            out += np.linalg.norm(series[i] - series[j], axis=-1)
            pairs += 1
    return out / max(pairs, 1)


# ---------------------------------------------------------------------------
# State-tracking: pair each utterance with the immediately-preceding state.
# ---------------------------------------------------------------------------

def utterance_intended_vs_felt(trace: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Pair each utterance's felt VAD with the state record just before it.

    Returns (intended_VAD, felt_VAD) as (N, 3) arrays.

    Order in the JSONL is:
      ... event  state(post-appraisal)  utterance(felt)  state(post-re-appraisal) ...
    so the state record immediately preceding each utterance is the
    intended one. We rebuild the chronological stream and join.
    """
    # Read the file fresh in chronological order (load_trace bucketed by kind).
    pass  # See `chronological_intended_felt` below for the actual implementation.


def chronological_intended_felt(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk the JSONL in order; for each utterance, take the previous state.

    Returns `(t, intended_VAD, felt_VAD)` as `(N,)`, `(N, 3)`, `(N, 3)`.
    The trace logger persists the readback under the key `"felt"`
    (not `"felt_vad"`); we read both for forward compat.
    """
    intended: List[Tuple[float, float, float]] = []
    felt: List[Tuple[float, float, float]] = []
    times: List[float] = []
    last_state: Dict[str, float] | None = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "state":
                last_state = rec
            elif kind == "utterance":
                felt_vad = rec.get("felt") or rec.get("felt_vad") or {}
                if last_state is None:
                    continue
                times.append(float(rec.get("t", 0.0)))
                intended.append((
                    last_state.get("valence", 0.0),
                    last_state.get("arousal", 0.0),
                    last_state.get("dominance", 0.0),
                ))
                felt.append((
                    float(felt_vad.get("V", 0.0)),
                    float(felt_vad.get("A", 0.0)),
                    float(felt_vad.get("D", 0.0)),
                ))
    if not intended:
        return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
    return np.array(times), np.array(intended), np.array(felt)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sx = np.std(x)
    sy = np.std(y)
    if sx < 1e-9 or sy < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PERSONA_COLORS = {
    "calm_commuter":      "#2ca02c",
    "aggressive_late":    "#d62728",
    "anxious_new_driver": "#9467bd",
}
_CONDITION_ORDER = ["card+steer", "card_only", "steer_only", "neither"]
_CONDITION_TITLES = {
    "card+steer": "card ON  +  steering ON   (default)",
    "card_only":  "card ON  +  steering OFF  (prompt-only)",
    "steer_only": "card OFF +  steering ON   (steering-only)",
    "neither":    "card OFF +  steering OFF  (floor)",
}


def render_felt_grid(
    felt_per: Dict[str, Dict[str, np.ndarray]],
    times_per: Dict[str, Dict[str, np.ndarray]],
    events: List[dict],
    output_path: Path,
) -> None:
    """2×2 grid: per-condition felt-V trajectories, one line per persona.

    This is the LLM-channel-specific view. The symbolic-state version
    is dominated by persona-specific dynamics that evolve identically
    regardless of which LLM channels are on; felt-V (the projection
    readout) is what each LLM utterance actually produced, so this
    figure is sensitive to card / steering toggles.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True)

    for ax, cond in zip(axes.flat, _CONDITION_ORDER):
        per_p = felt_per.get(cond, {})
        for persona_name, felt in sorted(per_p.items()):
            t_arr = times_per[cond].get(persona_name, np.arange(felt.shape[0]))
            color = _PERSONA_COLORS.get(persona_name, "#444")
            ax.plot(t_arr, felt[:, 0], color=color, linewidth=1.8,
                    marker="o", markersize=4, label=persona_name)

        for ev in events:
            ax.axvline(ev["t"], color="grey", linestyle=":", linewidth=0.7, alpha=0.4)

        ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
        ax.set_title(_CONDITION_TITLES[cond],
                     loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("felt valence  (probe readback, [-1,+1])")
        ax.set_ylim(-0.6, 0.6)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    fig.suptitle(
        "Card-vs-steering ablation - felt valence per utterance, by condition",
        fontsize=12.5, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {output_path}")


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def trace_path(scen_stem: str, cond: str, persona_name: str) -> Path:
    return RUNS_DIR / f"{scen_stem}__{cond}__{persona_name}.agent.jsonl"


def run_all(
    scenario_path: Path,
    personas: List[str],
    bundle_path: str,
    cards_path: str,
    rerun: bool,
) -> Dict[Tuple[str, str], Path]:
    scenario = json.loads(scenario_path.read_text())
    scen_stem = scenario_path.stem
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    out_paths: Dict[Tuple[str, str], Path] = {
        (cond, persona_name): trace_path(scen_stem, cond, persona_name)
        for cond in CONDITIONS
        for persona_name in personas
    }

    needs_run = any(
        rerun or not p.exists() for p in out_paths.values()
    )
    if not needs_run:
        return out_paths

    bundle = ProbeBundle.load(bundle_path)
    cards = load_persona_cards(cards_path)
    print(f"[probes]  {bundle_path}  selected_layers={bundle.selected_layers}")
    print(f"[model]   loading {bundle.model_id}...")
    model, tokenizer = load(bundle.model_id)
    agent = SteeredAgent(model, tokenizer, bundle, cards)

    for (cond, persona_name), path in out_paths.items():
        if not rerun and path.exists():
            print(f"[skip]    {path.name} present (--rerun to overwrite)")
            continue
        use_card, use_steering = CONDITIONS[cond]
        print(f"[run]     {cond:<12}  {persona_name:<22}  card={int(use_card)} steer={int(use_steering)}  → {path.name}")
        run_scenario(
            scenario, PRESETS[persona_name], path,
            agent=agent,
            use_card=use_card,
            use_steering=use_steering,
        )

    return out_paths


def compute_metrics(
    out_paths: Dict[Tuple[str, str], Path], duration_s: float
) -> dict:
    grid_t = np.arange(0.0, duration_s + 0.05, 0.1)
    grids: Dict[Tuple[str, str], np.ndarray] = {}
    events_per: List[dict] = []
    state_track: Dict[str, Dict[str, float]] = {c: {} for c in CONDITIONS}
    felt_per: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in CONDITIONS}
    times_per: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in CONDITIONS}

    for (cond, persona_name), path in out_paths.items():
        trace = load_trace(path)
        ts, cols = trace_state_array(trace)
        grids[(cond, persona_name)] = resample_to_grid(ts, cols, grid_t)
        if not events_per:
            events_per = trace["event"]

        t_arr, intended, felt = chronological_intended_felt(path)
        felt_per[cond][persona_name] = felt
        times_per[cond][persona_name] = t_arr
        if intended.shape[0] >= 2:
            state_track[cond][persona_name] = pearson_r(intended[:, 0], felt[:, 0])
        else:
            state_track[cond][persona_name] = float("nan")

    # Symbolic state distinctness (per-tick mean pairwise L2 in V,A,D).
    distinctness: Dict[str, np.ndarray] = {}
    distinctness_means: Dict[str, float] = {}
    personas = sorted({k[1] for k in out_paths})
    for cond in CONDITIONS:
        per_persona = {p: grids[(cond, p)] for p in personas}
        d = pairwise_distance_over_time(per_persona)
        distinctness[cond] = d
        distinctness_means[cond] = float(d.mean())

    # Felt-VAD distinctness — the LLM-channel-specific signal. Uses
    # per-utterance felt readings (one per event), assumes the same
    # event sequence across personas (true for shared scenarios).
    felt_distinctness: Dict[str, np.ndarray] = {}
    felt_distinctness_means: Dict[str, float] = {}
    for cond in CONDITIONS:
        per_p = felt_per[cond]
        if len(per_p) < 2:
            continue
        keys = sorted(per_p.keys())
        try:
            stacked = np.stack([per_p[k] for k in keys])  # (n_p, n_utts, 3)
        except ValueError:
            # Different utterance counts across personas — skip this condition.
            continue
        n_p, n_utts, _ = stacked.shape
        if n_utts == 0:
            continue
        d = np.zeros(n_utts)
        pairs = 0
        for i in range(n_p):
            for j in range(i + 1, n_p):
                d += np.linalg.norm(stacked[i] - stacked[j], axis=-1)
                pairs += 1
        d /= max(pairs, 1)
        felt_distinctness[cond] = d
        felt_distinctness_means[cond] = float(d.mean())

    return {
        "grid_t": grid_t,
        "events": events_per,
        "distinctness": distinctness,
        "distinctness_means": distinctness_means,
        "felt_distinctness": felt_distinctness,
        "felt_distinctness_means": felt_distinctness_means,
        "felt_per": felt_per,
        "times_per": times_per,
        "state_track": state_track,
        "personas": personas,
    }


def write_report(scenario_name: str, metrics: dict, report_path: Path) -> None:
    dm = metrics["distinctness_means"]
    fdm = metrics.get("felt_distinctness_means", {})
    st = metrics["state_track"]
    personas = metrics["personas"]

    def _row(d: Dict[str, float], cond: str) -> str:
        v = d.get(cond)
        return f"{v:.3f}" if v is not None else "  -  "

    def _track_row(cond: str) -> str:
        rs = [v for v in st[cond].values() if not (isinstance(v, float) and np.isnan(v))]
        return f"{np.mean(rs):+.3f}" if rs else "  nan "

    md = f"""# Card-vs-steering ablation — `{scenario_name}`

A 2×2 attribution of where the closed loop's persona differentiation
comes from. Same scenario × 3 personas × 4 conditions, with the LLM
agent in the loop for all four. Two metrics:

1. **Felt-VAD distinctness** - mean pairwise L2 distance across
   personas on the *probe-readback* V/A/D per utterance. This is the
   LLM-channel-specific signal: it measures whether the LLM's
   produced text actually projects onto different V/A/D coordinates
   per persona, in each condition.
2. **Symbolic-state distinctness** - the same metric but on the
   internal symbolic state. This is dominated by persona-specific
   dynamics (`baseline_va`, `patience`, `reactivity`) that evolve
   identically regardless of LLM channels, so it should be roughly
   flat across conditions. Reported here for comparison.

## Headline metric: felt-VAD distinctness (the LLM-relevant one)

| condition  | mean pairwise L2 on felt VAD |
|---|---:|
| card+steer (default) | {_row(fdm, 'card+steer')} |
| card_only            | {_row(fdm, 'card_only')} |
| steer_only           | {_row(fdm, 'steer_only')} |
| neither              | {_row(fdm, 'neither')} |

The single most informative comparison is **`card+steer` vs
`card_only`**: the gap is the distinctness contribution attributable
to activation steering *on top of* an already-strong prompt anchor.
A second informative comparison is **`steer_only` vs `neither`**:
the gap measures what activation steering produces when the LLM has
*no* character anchor at all.

## Symbolic-state distinctness (sanity check; expected near-flat)

| condition  | mean L2 (V,A,D) |
|---|---:|
| card+steer  | {_row(dm, 'card+steer')} |
| card_only   | {_row(dm, 'card_only')} |
| steer_only  | {_row(dm, 'steer_only')} |
| neither     | {_row(dm, 'neither')} |

If these are tightly clustered, the result confirms the metric was
the wrong primary one for this ablation: symbolic dynamics dominate
and the LLM's contribution gets washed out at κ=0.3 re-appraisal
coupling.

## State-tracking — Pearson r(intended V, felt V), averaged across personas

| condition  | mean r(V) |
|---|---:|
| card+steer  | {_track_row('card+steer')} |
| card_only   | {_track_row('card_only')} |
| steer_only  | {_track_row('steer_only')} |
| neither     | {_track_row('neither')} |

How well each condition keeps the symbolic state and the LLM's
expressed projection in sync, *at the moment of utterance*.
Activation steering should help here because steering is the only
mechanism that puts the LLM's hidden state into the same coordinate
system as the symbolic state.

### Per-persona r(V) by condition

| persona | card+steer | card_only | steer_only | neither |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {p} | {st['card+steer'].get(p, float('nan')):+.3f} | {st['card_only'].get(p, float('nan')):+.3f} | {st['steer_only'].get(p, float('nan')):+.3f} | {st['neither'].get(p, float('nan')):+.3f} |"
        for p in personas
    ) + f"""

## Reading guide

- Strongest reading of the architectural claim: `card+steer` >
  `card_only` on felt-distinctness *and* on r(V). That'd say
  steering does work the prompt alone doesn't.
- A weaker reading: `card+steer` ≈ `card_only` on felt-distinctness,
  but `card+steer` > `card_only` on r(V). Steering doesn't add
  *visible* persona differentiation, but it *does* keep the loop
  closer to closed.
- A null reading: all four conditions look similar on felt-distinctness
  AND r(V). The symbolic dynamics are doing all the work; the LLM
  channel is paint, not load-bearing.

The figure
`artifacts/card_vs_steering_felt_{scenario_name}.png` shows
felt-V trajectories per persona × condition for visual inspection.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md)
    print(f"[report]  {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--personas", nargs="+", default=DEFAULT_PERSONAS)
    ap.add_argument("--probes", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--cards", default=str(DEFAULT_CARDS))
    ap.add_argument("--rerun", action="store_true",
                    help="Re-run all conditions even if traces already exist.")
    args = ap.parse_args()

    p = Path(args.scenario)
    if not p.exists():
        if not args.scenario.endswith(".json"):
            p = SCENARIOS_DIR / f"{args.scenario}.json"
        else:
            p = SCENARIOS_DIR / args.scenario
    if not p.exists():
        raise FileNotFoundError(f"Scenario not found: {args.scenario}")
    scenario_blob = json.loads(p.read_text())
    duration_s = float(scenario_blob.get("duration_s", 60.0))

    out_paths = run_all(p, args.personas, args.probes, args.cards, args.rerun)
    metrics = compute_metrics(out_paths, duration_s)

    fig_path = ARTIFACTS_DIR / f"card_vs_steering_felt_{p.stem}.png"
    report_path = NOTES_DIR / f"card_vs_steering_ablation.md"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    render_felt_grid(
        metrics["felt_per"], metrics["times_per"],
        metrics["events"], fig_path,
    )
    write_report(scenario_blob.get("name", p.stem), metrics, report_path)

    print()
    print("=== SUMMARY ===")
    fdm = metrics["felt_distinctness_means"]
    dm = metrics["distinctness_means"]
    st = metrics["state_track"]

    print("felt-VAD distinctness (LLM-channel-specific signal):")
    for cond in ["card+steer", "card_only", "steer_only", "neither"]:
        v = fdm.get(cond)
        if v is not None:
            print(f"  {cond:<12}  {v:.3f}")
        else:
            print(f"  {cond:<12}  -")
    print()
    print("symbolic-state distinctness (sanity, expected near-flat):")
    for cond in ["card+steer", "card_only", "steer_only", "neither"]:
        print(f"  {cond:<12}  {dm[cond]:.3f}")
    print()
    print("state-tracking r(V), averaged across personas:")
    for cond in ["card+steer", "card_only", "steer_only", "neither"]:
        rs = [v for v in st[cond].values() if not (isinstance(v, float) and np.isnan(v))]
        if rs:
            print(f"  {cond:<12}  {np.mean(rs):+.3f}")
        else:
            print(f"  {cond:<12}  (no valid data)")


if __name__ == "__main__":
    main()
