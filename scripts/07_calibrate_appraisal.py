#!/usr/bin/env python3
"""Empirical appraisal calibration: measure the LLM's felt V/A/D
response to each event type at each persona's baseline state, and
save it for use by `lib/appraisal.py`.

Why: previously, appraisal coefficients (e.g., `dv=-0.40*severity`
for cut_off) were hand-authored from psych literature. They're
plausible but ungrounded — they don't reflect how the actual
steered LLM responds to event descriptions. This script replaces
those constants with measurements from the same model that
generates utterances at runtime.

Method (offline):

  for each event type (in `EVENT_PAYLOADS` below):
      for each persona (calm/aggressive/anxious):
          state ← persona.baseline (V/A/D from baseline_va; F/S/Fa = 0)
          for each parameterised payload (3 per event type):
              for sample in 1..N:
                  utterance, felt_VAD ← agent.respond(state, persona, event)
              average felt_VAD across samples
          average across payloads → mean felt_VAD for (event, persona)
      ΔVAD = mean felt_VAD − baseline felt_VAD

  baseline felt_VAD per persona is measured separately, with a
  null-event prompt. The ΔVAD is what the EVENT itself contributes,
  net of whatever V/A/D the persona expresses at idle.

Output: `data/empirical_responses.json` keyed by event_type → persona.
`lib/appraisal.py` loads this at import time and uses the empirical
ΔV/ΔA/ΔD as the V/A/D portion of `appraise(event, persona, state)`.
Frustration / stress / fatigue stay rule-based — the V/A/D probes
can't directly measure those higher-level accumulators.

**Resume:** if the output file already exists from a prior partial
run, the script picks up where it left off — already-measured
(persona baseline) and (event, persona) pairs are skipped, and the
ProgressTracker accounts for the resumed work in its ETA. Use
`--restart` to discard prior progress and start clean.

Usage:
  python scripts/07_calibrate_appraisal.py
  python scripts/07_calibrate_appraisal.py --n-samples 5
  python scripts/07_calibrate_appraisal.py --events cut_off courtesy_gesture
  python scripts/07_calibrate_appraisal.py --restart      # discard prior progress
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mlx_lm import load                                    # noqa: E402

from lib import (                                          # noqa: E402
    PRESETS, DriverState,
    CourtesyGesture, CutOff, LateForAppointment,
    MergeOpportunity, NearMiss, PassengerComment,
    RedLight, TrafficCongestion, WeatherChange,
)
from lib.agent import SteeredAgent, load_persona_cards     # noqa: E402
from lib.probes import ProbeBundle                         # noqa: E402
from lib.progress import ProgressTracker                   # noqa: E402


DEFAULT_BUNDLE = _ROOT / "artifacts" / "probes_Qwen3.5-9B-MLX-4bit.pkl"
DEFAULT_CARDS = _ROOT / "data" / "persona_cards.json"
DEFAULT_OUTPUT = _ROOT / "data" / "empirical_responses.json"


# Three parameterised payloads per event type — span the realistic
# range so the average isn't dominated by a single severity.
EVENT_PAYLOADS = {
    "cut_off": [
        CutOff(t=0, severity=0.4, relative_speed=-3.0),
        CutOff(t=0, severity=0.7, relative_speed=-8.5),
        CutOff(t=0, severity=0.9, relative_speed=-12.0),
    ],
    "near_miss": [
        NearMiss(t=0, severity=0.5, ttc_s=1.5),
        NearMiss(t=0, severity=0.8, ttc_s=0.9),
    ],
    "merge_opportunity": [
        MergeOpportunity(t=0, gap_seconds=1.5),
        MergeOpportunity(t=0, gap_seconds=3.0),
    ],
    "traffic_congestion": [
        TrafficCongestion(t=0, duration_s=15.0),
        TrafficCongestion(t=0, duration_s=45.0),
    ],
    "red_light": [
        RedLight(t=0, expected_wait_s=15.0),
        RedLight(t=0, expected_wait_s=40.0),
    ],
    "weather_change": [
        WeatherChange(t=0, condition="rain", intensity=0.4),
        WeatherChange(t=0, condition="rain", intensity=0.8),
    ],
    "late_for_appointment": [
        LateForAppointment(t=0, minutes_behind=5.0, importance=0.5),
        LateForAppointment(t=0, minutes_behind=10.0, importance=0.9),
    ],
    "passenger_comment": [
        PassengerComment(t=0, valence=-0.5, intensity=0.6),
        PassengerComment(t=0, valence=+0.5, intensity=0.6),
    ],
    "courtesy_gesture": [
        CourtesyGesture(t=0, gesture="let_merge", intensity=0.7, discretionary=0.8),
        CourtesyGesture(t=0, gesture="polite_pass", intensity=0.5, discretionary=0.5),
    ],
}

# Null-event prompt for baseline measurement.
NULL_EVENT_TEXT = "Nothing in particular is happening right now — you're just driving along."


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _measure(
    agent,
    persona,
    event_or_text,
    *,
    n_samples: int,
) -> Dict[str, np.ndarray]:
    """Sample N utterances at the persona's baseline state, return per-axis arrays."""
    samples = []
    for _ in range(n_samples):
        # Each sample re-creates state to avoid drift from re-appraisal Δ.
        s = DriverState.from_persona(persona)
        r = agent.respond(s, persona, event_or_text, max_tokens=35)
        samples.append(r.felt_VAD)
    return {
        axis: np.array([s[axis] for s in samples], dtype=np.float64)
        for axis in samples[0]
    }


def _stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "n":    int(arr.size),
    }


# ---------------------------------------------------------------------------
# Resume / checkpoint
# ---------------------------------------------------------------------------

def _load_partial(path: Path) -> dict:
    """Read an existing output JSON; return empty skeleton if absent/malformed."""
    if not path.exists():
        return {"baselines": {}, "responses": {}}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"baselines": {}, "responses": {}}
    return {
        "baselines": blob.get("baselines") or {},
        "responses": blob.get("responses") or {},
    }


def _save_checkpoint(
    path: Path,
    *,
    model_id: str,
    n_samples_per_payload: int,
    personas: List[str],
    baselines: Dict,
    responses: Dict,
) -> None:
    """Atomic-ish save (write to .tmp, rename) so a crash mid-write
    doesn't corrupt the resume state."""
    out = {
        "model_id": model_id,
        "calibration_date": datetime.date.today().isoformat(),
        "n_samples_per_payload": n_samples_per_payload,
        "personas": personas,
        "baselines": baselines,
        "responses": responses,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def _count_pending(
    event_types: List[str],
    persona_names: List[str],
    *,
    payloads: Dict[str, List],
    n_samples: int,
    existing_baselines: Dict,
    existing_responses: Dict,
) -> Tuple[int, int]:
    """Return (already_done_gens, pending_gens) so ProgressTracker can budget correctly."""
    total_baseline = len(persona_names) * n_samples
    done_baseline = sum(n_samples for p in persona_names if p in existing_baselines)

    total_event = 0
    done_event = 0
    for ev in event_types:
        if ev not in payloads:
            continue
        per_pair = len(payloads[ev]) * n_samples
        for p in persona_names:
            total_event += per_pair
            if ev in existing_responses and p in existing_responses[ev]:
                done_event += per_pair

    return (done_baseline + done_event), (total_baseline + total_event - done_baseline - done_event)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--cards",  default=str(DEFAULT_CARDS))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--n-samples", type=int, default=3,
                    help="Samples per (event, persona, payload). 3 → ~5 min run, "
                         "5 → ~9 min, more is more reliable.")
    ap.add_argument("--events", nargs="+", default=None,
                    help="Subset of event types to calibrate (default: all 9).")
    ap.add_argument("--personas", nargs="+", default=None,
                    help="Subset of personas to calibrate (default: all PRESETS).")
    ap.add_argument("--restart", action="store_true",
                    help="Discard any prior partial progress and start clean.")
    args = ap.parse_args()

    bundle = ProbeBundle.load(args.probes)
    cards = load_persona_cards(args.cards)
    print(f"[probes] {args.probes}  selected_layers={bundle.selected_layers}")
    print(f"[model]  loading {bundle.model_id}...")
    model, tokenizer = load(bundle.model_id)
    agent = SteeredAgent(model, tokenizer, bundle, cards)

    event_types = list(EVENT_PAYLOADS.keys()) if args.events is None else args.events
    persona_names = sorted(PRESETS.keys()) if args.personas is None else args.personas
    out_path = Path(args.output)

    # ── Resume scaffold ───────────────────────────────────────────────────
    if args.restart:
        baselines: Dict[str, Dict[str, dict]] = {}
        responses: Dict[str, Dict[str, dict]] = {}
        print(f"[resume] --restart: discarding any prior progress at {out_path}")
    else:
        partial = _load_partial(out_path)
        baselines = partial["baselines"]
        responses = partial["responses"]
        if baselines or responses:
            print(f"[resume] loaded {len(baselines)} baselines and "
                  f"{sum(len(v) for v in responses.values())} (event,persona) pairs "
                  f"from {out_path}")

    # ── Plan + progress tracker ───────────────────────────────────────────
    done_gens, pending_gens = _count_pending(
        event_types, persona_names,
        payloads=EVENT_PAYLOADS, n_samples=args.n_samples,
        existing_baselines=baselines, existing_responses=responses,
    )
    total_gens = done_gens + pending_gens
    print(f"\n[plan] baseline + event pairs to run: {total_gens} total generations "
          f"({done_gens} already done, {pending_gens} pending)")

    progress = ProgressTracker(total=total_gens)
    if done_gens > 0:
        progress.credit_skipped(done_gens, label=f"resumed prior work")

    # ── 1. Per-persona null-event baseline ────────────────────────────────
    print(f"\n[baseline] null-event response per persona (N={args.n_samples})")
    for pname in persona_names:
        if pname in baselines:
            continue
        persona = PRESETS[pname]
        arrs = _measure(agent, persona, NULL_EVENT_TEXT, n_samples=args.n_samples)
        baselines[pname] = {axis: _stats(arr) for axis, arr in arrs.items()}
        b = baselines[pname]
        progress.step(
            f"baseline persona={pname:22s} | "
            f"V={b['V']['mean']:+.2f} A={b['A']['mean']:+.2f} D={b['D']['mean']:+.2f}",
            sample_n=args.n_samples,
        )
        _save_checkpoint(
            out_path, model_id=bundle.model_id,
            n_samples_per_payload=args.n_samples, personas=persona_names,
            baselines=baselines, responses=responses,
        )

    # ── 2. Per-(event, persona) measurements ──────────────────────────────
    for ev_type in event_types:
        if ev_type not in EVENT_PAYLOADS:
            print(f"  [skip] no payloads defined for {ev_type!r}")
            continue
        responses.setdefault(ev_type, {})
        payloads = EVENT_PAYLOADS[ev_type]
        print(f"\n[event] {ev_type}  ({len(payloads)} payloads × "
              f"{len(persona_names)} personas × N={args.n_samples})")

        for pname in persona_names:
            if pname in responses[ev_type]:
                continue   # resume: already done
            persona = PRESETS[pname]
            # Pool samples across payloads for a per-(event, persona) estimate.
            pooled: Dict[str, List[float]] = {"V": [], "A": [], "D": []}
            for payload in payloads:
                arrs = _measure(agent, persona, payload, n_samples=args.n_samples)
                for axis in pooled:
                    pooled[axis].extend(arrs[axis].tolist())
            # Compute Δ relative to that persona's baseline.
            deltas: Dict[str, dict] = {}
            for axis, samples in pooled.items():
                arr = np.array(samples, dtype=np.float64)
                base_mean = baselines[pname][axis]["mean"]
                deltas[axis] = {
                    "delta_mean": float(arr.mean() - base_mean),
                    "raw_mean":   float(arr.mean()),
                    "std":        float(arr.std()),
                    "n":          int(arr.size),
                }
            responses[ev_type][pname] = deltas
            n_for_this_pair = len(payloads) * args.n_samples
            progress.step(
                f"event={ev_type:22s} persona={pname:18s} | "
                f"ΔV={deltas['V']['delta_mean']:+.2f} "
                f"ΔA={deltas['A']['delta_mean']:+.2f} "
                f"ΔD={deltas['D']['delta_mean']:+.2f}",
                sample_n=n_for_this_pair,
            )
            _save_checkpoint(
                out_path, model_id=bundle.model_id,
                n_samples_per_payload=args.n_samples, personas=persona_names,
                baselines=baselines, responses=responses,
            )

    print(f"\n[done] all measurements complete.")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
