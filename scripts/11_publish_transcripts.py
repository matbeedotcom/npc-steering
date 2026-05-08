#!/usr/bin/env python3
"""Render per-persona JSONL traces as readable markdown transcripts.

Walks a closed-loop trace produced by `lib.loop.run_scenario`,
interleaves events / state / utterances in chronological order, and
writes a markdown document per (scenario, persona). Output goes to
`docs/transcripts/<scenario>__<persona>.md`.

Used to generate human-readable artifacts that can be linked from
the README and shared without the reader having to spelunk JSONL.

Usage:
  python scripts/11_publish_transcripts.py
  python scripts/11_publish_transcripts.py --scenario mixed_emotions
  python scripts/11_publish_transcripts.py --personas calm_commuter aggressive_late
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

RUNS_DIR = _ROOT / "runs"
TRANSCRIPTS_DIR = _ROOT / "docs" / "transcripts"
DEFAULT_SCENARIO = "mixed_emotions"
DEFAULT_PERSONAS = ["calm_commuter", "aggressive_late", "anxious_new_driver"]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _format_event_payload(rec: Dict[str, Any]) -> str:
    """Compact one-line description of an event's payload fields."""
    payload = rec.get("payload") or {}
    if not payload:
        return ""
    parts = []
    for k, v in payload.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _state_summary(state: Dict[str, Any]) -> str:
    """One-line state summary in the four most informative scalars."""
    return (
        f"V={state.get('valence', 0):+.2f}  "
        f"A={state.get('arousal', 0):+.2f}  "
        f"D={state.get('dominance', 0):+.2f}  "
        f"frust={state.get('frustration', 0):.2f}"
    )


def _felt_summary(felt: Dict[str, Any]) -> str:
    return (
        f"V={felt.get('V', 0):+.2f}  "
        f"A={felt.get('A', 0):+.2f}  "
        f"D={felt.get('D', 0):+.2f}"
    )


def _trait_block(traits: Dict[str, Any]) -> str:
    bv = traits.get("baseline_va", [0, 0])
    return (
        f"- temperament: **{traits.get('temperament', 0):+.2f}** (-1 calm ↔ +1 volatile)\n"
        f"- patience: **{traits.get('patience', 0):.2f}** (0 impatient ↔ 1 patient)\n"
        f"- risk_preference: **{traits.get('risk_preference', 0):+.2f}** (-1 cautious ↔ +1 aggressive)\n"
        f"- reactivity: **{traits.get('reactivity', 0):.2f}** (0 flat ↔ 1 expressive)\n"
        f"- baseline (V, A): **({bv[0]:+.2f}, {bv[1]:+.2f})**"
    )


def render_transcript(records: List[Dict[str, Any]], scenario: str, persona: str) -> str:
    """Produce the markdown transcript for one persona's run.

    The trace logger emits, per event time:
      event → state(post-appraisal) → utterance(felt) → state(post-re-appraisal)
    so we walk linearly and pair the most recent event/state with each
    utterance. State records between events are dynamics-only and get
    suppressed for readability.
    """
    meta = next((r for r in records if r.get("kind") == "meta"), {})
    traits = meta.get("persona_traits") or {}

    out = []
    out.append(f"# Persona transcript: `{persona}` on `{scenario}`")
    out.append("")
    out.append(
        "Closed-loop run with the steered LLM agent "
        "(Qwen3.5-9B-MLX-4bit, layer 15, g=25) - same scenario as the "
        "`persona_sweep_mixed_emotions_v3.png` figure. Each event "
        "block shows the symbolic state after the symbolic appraisal "
        "ran, the LLM's first-person monologue, and the felt V/A/D "
        "read back from the LLM's own hidden state via the same probe "
        "directions used for steering."
    )
    out.append("")
    out.append("## Persona traits")
    out.append("")
    out.append(_trait_block(traits))
    out.append("")
    out.append("## Trace")
    out.append("")

    # Walk the records in order, emitting one block per (event, utterance) pair.
    pending_event: Dict[str, Any] | None = None
    pending_state: Dict[str, Any] | None = None
    initial_state_emitted = False

    for rec in records:
        kind = rec.get("kind")
        if kind == "meta":
            continue
        if kind == "state":
            if not initial_state_emitted:
                out.append(f"**Initial state** ({_state_summary(rec)})")
                out.append("")
                initial_state_emitted = True
            pending_state = rec
            continue
        if kind == "event":
            pending_event = rec
            continue
        if kind == "utterance" and pending_event is not None:
            ev_type = (pending_event.get("type") or "").replace("_", " ")
            payload_desc = _format_event_payload(pending_event)
            head = f"### t = {rec.get('t', 0.0):>4.1f}s  -  {ev_type}"
            out.append(head)
            if payload_desc:
                out.append(f"*{payload_desc}*")
            if pending_state is not None:
                out.append(f"**State after appraisal:** {_state_summary(pending_state)}")
            text = (rec.get("text") or "").strip().replace("\n", " ")
            out.append("")
            out.append(f"> *\"{text}\"*")
            out.append("")
            felt = rec.get("felt") or rec.get("felt_vad") or {}
            if felt:
                out.append(f"**Felt readback:** {_felt_summary(felt)}")
            out.append("")
            out.append("---")
            out.append("")
            pending_event = None
            # Don't reset pending_state; the post-utterance state record
            # will overwrite it on the next iteration.

    out.append("")
    out.append(
        "## How to read this\n\n"
        "- **State after appraisal** = the symbolic V/A/D after the "
        "rule-based appraisal layer fired, before the LLM gets to "
        "weigh in. This is what's being *steered into* the LLM via "
        "activation injection.\n"
        "- **The line in italics** is the LLM's first-person monologue, "
        "produced under steering at the layer-15 residual stream "
        "(α scaled from V/A/D with g=25).\n"
        "- **Felt readback** = the LLM's final hidden state at the same "
        "layer, projected onto the same V/A/D probe directions used "
        "for steering. The closed-loop story's load-bearing claim is "
        "that intended (state) and felt (readback) live in the same "
        "coordinate system, so their difference can drive a "
        "re-appraisal Δ that pulls the symbolic state toward what "
        "was actually expressed."
    )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--personas", nargs="+", default=DEFAULT_PERSONAS)
    ap.add_argument(
        "--runs-dir", default=str(RUNS_DIR),
        help="Where the source JSONL traces live.",
    )
    ap.add_argument(
        "--output-dir", default=str(TRANSCRIPTS_DIR),
        help="Where to write the rendered markdown transcripts.",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for persona in args.personas:
        # Prefer an explicit-suffix path; fall back to the default suffix.
        candidates = [
            runs_dir / f"{args.scenario}__{persona}.agent.jsonl",
            runs_dir / f"{args.scenario}__{persona}.jsonl",
        ]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            print(f"[skip] {persona} - no trace found at {candidates[0]}")
            continue
        records = _read_jsonl(path)
        md = render_transcript(records, args.scenario, persona)
        out_path = out_dir / f"{args.scenario}__{persona}.md"
        out_path.write_text(md)
        print(f"[wrote] {out_path}  ({path.name})")


if __name__ == "__main__":
    main()
