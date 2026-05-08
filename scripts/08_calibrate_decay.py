#!/usr/bin/env python3
"""LLM-prior decay calibration: estimate τ_V, τ_A, τ_D from the
LLM's implicit model of how driver affect fades over time.

Why: Phase A's `lib/state.py` uses hand-authored time constants
(τ_VA = 8s, τ_D = 6s, etc.). Same methodological gap as the
hand-authored appraisal: the constants are picked by feel from
psych literature heuristics, not measured. This script uses the
*same instrument* as the appraisal calibration (steered LLM +
probe projection) to ground the decay rates in something
measurable from inside the project.

Method (offline):

  for each event type (small representative set):
      for each persona:
          for each delay ∈ {0, 5, 15, 30, 60} seconds:
              build prompt with explicit "X seconds ago: <event>"
                  framing — the LLM's task is to express what it
                  feels NOW given that time has passed
              for sample in 1..N:
                  felt_VAD ← agent.respond(...)
              average felt_VAD across samples
          → curve: felt_V(t), felt_A(t), felt_D(t) for this group
  Per axis (pooled across event × persona groups):
      normalise each group's curve by its t=0 deviation
      fit log-linear exponential decay → single τ_axis

**Caveat (state honestly).** This measures the LLM's *prior over
human emotional recovery times*, not actual human regulation. The
LLM is a stand-in for "what someone with broad text exposure
believes recovery looks like," which is more grounded than a
guessed constant but less than biometric data. Replace with
naturalistic-driving traces (e.g., SHRP2 HRV) when available; this
calibration is the second-best methodology under that ceiling.

Output: `data/empirical_decay.json` with `{axis → tau_s}` plus
diagnostics (per-(event, persona) τ estimates, fit quality, R²).
`lib/state.py` reads it at import and falls back to hand-authored
constants if missing.

Usage:
  python scripts/08_calibrate_decay.py
  python scripts/08_calibrate_decay.py --n-samples 3 --delays 0 5 15 30 60
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mlx_lm import load                                     # noqa: E402

from lib import (                                           # noqa: E402
    PRESETS, DriverState,
    CourtesyGesture, CutOff, NearMiss, PassengerComment,
)
from lib.agent import SteeredAgent, load_persona_cards      # noqa: E402
from lib.events import event_to_text                        # noqa: E402
from lib.probes import ProbeBundle                          # noqa: E402


DEFAULT_BUNDLE = _ROOT / "artifacts" / "probes_Qwen3.5-9B-MLX-4bit.pkl"
DEFAULT_CARDS = _ROOT / "data" / "persona_cards.json"
DEFAULT_OUTPUT = _ROOT / "data" / "empirical_decay.json"

# Smaller event set than appraisal calibration — we want events that
# produce strong, *measurable* affective spikes whose decay is worth
# fitting. Neutral events (red_light, merge_opportunity) wouldn't
# give us a clean decay curve.
EVENTS_FOR_DECAY = {
    "cut_off":            CutOff(t=0, severity=0.8, relative_speed=-9.0),
    "near_miss":          NearMiss(t=0, severity=0.8, ttc_s=0.9),
    "courtesy_let_merge": CourtesyGesture(t=0, gesture="let_merge",
                                          intensity=0.8, discretionary=0.9),
    "passenger_negative": PassengerComment(t=0, valence=-0.6, intensity=0.7),
}

# Sample timepoints. 0s is "right now"; the upper bound matches the
# 60s scenario timescale.
DEFAULT_DELAYS_S = [0.0, 5.0, 15.0, 30.0, 60.0]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def temporal_event_text(event, delay_s: float) -> str:
    """Render an event with explicit elapsed-time framing."""
    base = event_to_text(event).rstrip(".") + "."
    if delay_s < 1.0:
        return f"Just now: {base}"
    if delay_s < 10:
        return (
            f"{int(delay_s)} seconds ago: {base} "
            f"You've been driving along since."
        )
    return (
        f"About {int(delay_s)} seconds ago: {base} "
        f"You've kept driving since then; it's behind you now."
    )


from lib.progress import ProgressTracker                  # noqa: E402


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _measure(
    agent,
    persona,
    prompt_text: str,
    *,
    n_samples: int,
    progress: Optional[ProgressTracker] = None,
    label: str = "",
) -> Dict[str, float]:
    """Mean felt VAD over N samples at the persona's baseline state."""
    accum = {"V": [], "A": [], "D": []}
    t_block = time.time()
    for _ in range(n_samples):
        state = DriverState.from_persona(persona)
        r = agent.respond(state, persona, prompt_text, max_tokens=30)
        for axis in accum:
            accum[axis].append(r.felt_VAD[axis])
    means = {axis: float(np.mean(vs)) for axis, vs in accum.items()}
    if progress is not None:
        block_dt = time.time() - t_block
        readout = (
            f"V={means['V']:+.2f} A={means['A']:+.2f} D={means['D']:+.2f} "
            f"({block_dt:.1f}s)"
        )
        progress.step(f"{label} | {readout}", sample_n=n_samples)
    return means


# ---------------------------------------------------------------------------
# Per-axis pooled exponential fit
# ---------------------------------------------------------------------------

def _fit_axis(curves: List[Dict[float, float]], *, axis: str) -> dict:
    """Pooled log-linear exponential fit: dev(t) = dev(0) * exp(-t/τ).

    `curves` is a list of dicts {delay_s: deviation_from_baseline}, one
    per (event, persona) group. We normalise each group's curve by its
    t=0 deviation, pool the points, and fit a single τ. Groups whose
    initial deviation is too small to fit (|dev(0)| < 0.05) are
    excluded — they carry no signal.

    Returns dict with `tau_s`, `n_groups_used`, `n_points`, `r2`.
    """
    pooled_t: List[float] = []
    pooled_y: List[float] = []
    n_groups_used = 0

    for curve in curves:
        # Need delay=0 to be present.
        if 0.0 not in curve:
            continue
        dev0 = curve[0.0]
        if abs(dev0) < 0.05:
            continue   # too noisy to fit
        for delay, dev in curve.items():
            if delay == 0:
                continue
            normalized = dev / dev0      # ∈ [0, 1] if same-sign decay
            # Only keep points with same sign (real decay) and not too small
            if normalized <= 0 or normalized > 1.5:
                continue
            pooled_t.append(delay)
            pooled_y.append(normalized)
        n_groups_used += 1

    if len(pooled_t) < 4:
        return {"tau_s": None, "n_groups_used": n_groups_used,
                "n_points": len(pooled_t), "r2": None,
                "note": "insufficient signal for fit"}

    t = np.asarray(pooled_t, dtype=np.float64)
    y = np.asarray(pooled_y, dtype=np.float64)

    # Force the curve through (0, 1) by fitting log y = -t/τ (no intercept).
    log_y = np.log(np.clip(y, 1e-3, None))
    # Closed-form least-squares for slope-only model: slope = Σ(t·log y) / Σ t²
    slope = float(np.sum(t * log_y) / np.sum(t**2))
    tau = -1.0 / slope if slope < 0 else float("inf")

    # R² against the no-intercept fit.
    pred = np.exp(-t / tau) if np.isfinite(tau) else np.full_like(t, 1.0)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)

    return {
        "tau_s": float(tau) if np.isfinite(tau) else None,
        "n_groups_used": n_groups_used,
        "n_points": int(len(t)),
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--cards",  default=str(DEFAULT_CARDS))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--n-samples", type=int, default=2,
                    help="Samples per (event, persona, delay). Default 2 (≈10 min).")
    ap.add_argument("--delays", type=float, nargs="+", default=DEFAULT_DELAYS_S)
    args = ap.parse_args()

    bundle = ProbeBundle.load(args.probes)
    cards = load_persona_cards(args.cards)
    print(f"[probes] {args.probes}  selected_layers={bundle.selected_layers}")
    print(f"[model]  loading {bundle.model_id}...")
    model, tokenizer = load(bundle.model_id)
    agent = SteeredAgent(model, tokenizer, bundle, cards)

    persona_names = sorted(PRESETS.keys())
    delays = sorted(args.delays)
    n_baseline = len(persona_names) * args.n_samples
    n_event = len(EVENTS_FOR_DECAY) * len(persona_names) * len(delays) * args.n_samples
    n_calls = n_baseline + n_event
    print(f"\n[plan] baseline: {len(persona_names)} personas × N={args.n_samples} "
          f"= {n_baseline} gens")
    print(f"[plan] event:    {len(EVENTS_FOR_DECAY)} events × {len(persona_names)} personas "
          f"× {len(delays)} delays × N={args.n_samples} = {n_event} gens")
    print(f"[plan] total:    {n_calls} generations")

    progress = ProgressTracker(total=n_calls)

    # ── 1. Per-persona null-event baseline (so we can isolate the event's
    #      contribution at each delay).
    print(f"\n[baseline] null-event felt VAD per persona (N={args.n_samples}):")
    null_text = "Nothing in particular is happening right now — you're just driving along."
    baselines: Dict[str, Dict[str, float]] = {}
    for pname in persona_names:
        baselines[pname] = _measure(
            agent, PRESETS[pname], null_text,
            n_samples=args.n_samples,
            progress=progress,
            label=f"baseline persona={pname:22s}",
        )

    # ── 2. Per-(event, persona, delay) measurements.
    raw: Dict[str, Dict[str, Dict[float, Dict[str, float]]]] = {}
    for ev_label, ev in EVENTS_FOR_DECAY.items():
        raw[ev_label] = {}
        print(f"\n[event] {ev_label}")
        for pname in persona_names:
            persona = PRESETS[pname]
            curve_for_persona: Dict[float, Dict[str, float]] = {}
            for delay in delays:
                prompt = temporal_event_text(ev, delay)
                means = _measure(
                    agent, persona, prompt,
                    n_samples=args.n_samples,
                    progress=progress,
                    label=(
                        f"event={ev_label:20s} persona={pname:18s} "
                        f"delay={delay:>4.0f}s"
                    ),
                )
                curve_for_persona[delay] = means
            raw[ev_label][pname] = curve_for_persona

    # ── 3. Pool into per-axis curves and fit exponential decay.
    fits: Dict[str, dict] = {}
    print("\n[fit] per-axis pooled exponential decay")
    for axis in ["V", "A", "D"]:
        curves: List[Dict[float, float]] = []
        for ev_label in raw:
            for pname in raw[ev_label]:
                base = baselines[pname][axis]
                deviations = {
                    delay: raw[ev_label][pname][delay][axis] - base
                    for delay in delays
                }
                curves.append(deviations)
        fits[axis] = _fit_axis(curves, axis=axis)
        f = fits[axis]
        print(f"  τ_{axis}: "
              + (f"{f['tau_s']:6.2f}s" if f['tau_s'] is not None else "  N/A   ")
              + f"   (groups={f['n_groups_used']}, points={f['n_points']}"
              + (f", R²={f['r2']:+.3f})" if f['r2'] is not None else ")"))

    # ── 4. Save artifact.
    out = {
        "model_id": bundle.model_id,
        "calibration_date": datetime.date.today().isoformat(),
        "n_samples_per_delay": args.n_samples,
        "delays_s": delays,
        "personas": persona_names,
        "baselines": baselines,
        "raw_curves": raw,
        "fits": fits,
        "tau_per_axis_s": {axis: fits[axis]["tau_s"] for axis in fits},
        "method_note": (
            "Decay τ measured via the LLM's implicit prior on emotional "
            "recovery times: prompts are framed as 'X seconds ago: <event>' "
            "with X in {0, 5, 15, 30, 60}, felt V/A/D is read via probe "
            "projection, and a single τ is fit per axis pooled across "
            "(event, persona) groups. This grounds the dynamics in "
            "something measurable from inside the project, but is *not* "
            "biometric ground truth — replace with naturalistic-driving "
            "HRV traces (SHRP2/UDRIVE) when available."
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
