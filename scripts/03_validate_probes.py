#!/usr/bin/env python3
"""Validate extracted probes by injecting α·v_axis at each candidate layer.

For each (axis, layer) we generate completions of a fixed driving-relevant
prompt at α ∈ {-50, -25, 0, +25, +50} (calibrated to Qwen3.5-9B's
residual norms; smaller models use smaller α). Output dumps to a
markdown report so layer selection is by visual inspection.

Pass criterion: at high |α| the output should be qualitatively shifted
along the axis (valence-positive: cheerful; valence-negative: glum)
without collapsing into incoherence. The report also prints the
per-layer extraction diagnostics from `02_extract_probes.py` for
context — high `cosine_consistency` is necessary but not sufficient.

After visual inspection, edit `selected_layers` in the saved bundle
(or call `bundle.selected_layers[axis] = layer; bundle.save(...)`)
so Phase C's closed loop knows which probe to use.

Usage:
  python scripts/03_validate_probes.py                                  # default bundle
  python scripts/03_validate_probes.py --probes artifacts/foo.pkl
  python scripts/03_validate_probes.py --alphas -50 -25 0 25 50
  python scripts/03_validate_probes.py --layers 15 19 23 --axes V A
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import mlx.core as mx                                              # noqa: E402
from mlx_lm import generate, load                                  # noqa: E402
from mlx_lm.sample_utils import make_logits_processors             # noqa: E402

from lib.probes import ProbeBundle                                 # noqa: E402
from lib.mlx_steering import wrap_steering, unwrap                 # noqa: E402


DEFAULT_BUNDLE_PAT = "artifacts/probes_*.pkl"
DEFAULT_REPORT = _ROOT / "artifacts" / "probes_validation.md"

# Driving-relevant neutral prompt — gives the model latitude to express
# however the steering pushes it without forcing affect through the
# context.
DEFAULT_PROMPT = (
    "You are driving to work. In one sentence, describe how you are "
    "feeling and what is on your mind right now."
)
# Calibrated for Qwen3.5-9B residual norms (mid-stack ~30-100). For
# Hermes 3 / Llama 3.1 8B residuals are smaller — bring these down to
# {-10, -5, 0, +5, +10} on retrials.
DEFAULT_ALPHAS = [-50.0, -25.0, 0.0, 25.0, 50.0]


def _resolve_default_bundle() -> Path:
    """Find the most recently saved probe bundle in artifacts/."""
    matches = sorted(
        _ROOT.glob(DEFAULT_BUNDLE_PAT),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No bundle found at {_ROOT}/{DEFAULT_BUNDLE_PAT}. "
            f"Run scripts/02_extract_probes.py first."
        )
    return matches[0]


def steered_generate(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    direction_np: np.ndarray,
    alpha: float,
    *,
    max_tokens: int = 60,
    enable_thinking: bool = False,
) -> str:
    """Generate `prompt` continuation with `+alpha·direction` at `layer_idx`.

    `enable_thinking=False` (default) tells the Qwen3 chat template to
    prepend a closed `<think></think>` block, suppressing the model's
    "Thinking Process" preamble. This is critical for steering
    validation: with thinking enabled, ~all outputs collapse into
    `Thinking Process: 1. **Analyze...`-style meta-commentary that
    drowns the affective shift we are trying to read.
    """
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )

    # Repetition penalty matters at high |α|: without it, strong
    # steering collapses output into n-gram loops that look like
    # steering failure but are really decoding failure.
    processors = make_logits_processors(
        repetition_penalty=1.2, repetition_context_size=24,
    )

    if abs(alpha) < 1e-9:
        return generate(
            model, tokenizer, prompt=formatted,
            max_tokens=max_tokens, verbose=False,
            logits_processors=processors,
        )

    direction = mx.array(direction_np, dtype=mx.bfloat16)
    offset = alpha * direction

    originals = wrap_steering(model, {layer_idx: offset})
    try:
        text = generate(
            model, tokenizer, prompt=formatted,
            max_tokens=max_tokens, verbose=False,
            logits_processors=processors,
        )
    finally:
        unwrap(model, originals)
    return text.strip()


def _short_model_name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=None,
                    help="ProbeBundle path (default: most recent artifacts/probes_*.pkl).")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Subset of candidate_layers (default: all).")
    ap.add_argument("--axes", nargs="+", default=None,
                    help="Subset of axes to validate (default: all).")
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    bundle_path = Path(args.probes) if args.probes else _resolve_default_bundle()
    bundle = ProbeBundle.load(bundle_path)
    print(f"[probes] loaded {bundle_path}")
    print(f"[probes] model={bundle.model_id}  extracted={bundle.extraction_date}")
    layers = args.layers or bundle.candidate_layers
    axes = args.axes or list(bundle.axes.keys())

    print(f"[mlx]    metal_available={mx.metal.is_available()}")
    print(f"[model]  loading {bundle.model_id}...")
    model, tokenizer = load(bundle.model_id)

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        f.write(f"# Probe validation: `{_short_model_name(bundle.model_id)}`\n\n")
        f.write(f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Probes extracted: {bundle.extraction_date}\n")
        f.write(f"- Pairs per axis: {bundle.n_pairs_per_axis}\n")
        f.write(f"- Candidate layers: {bundle.candidate_layers}\n")
        f.write(f"- α values: {args.alphas}\n")
        f.write(f"- Validation prompt:\n\n  > {args.prompt}\n\n")
        f.write(
            "Layer selection criterion: pick the layer where high-|α| outputs are "
            "clearly shifted along the axis but still grammatical and on-topic. "
            "Diagnostics (`‖mean_diff‖`, `cos_consistency`) shown for context — "
            "high values are necessary but not sufficient for good steering.\n\n"
        )

        for axis in axes:
            axis_diag = bundle.diagnostics.get(axis, {})
            f.write(f"## Axis: {axis}\n\n")
            for layer in layers:
                if layer not in bundle.axes[axis]:
                    continue
                d = axis_diag.get(layer, {})
                f.write(
                    f"### Layer {layer}  "
                    f"(‖diff‖={d.get('norm_unnormalised', float('nan')):.2f}, "
                    f"cos_consistency={d.get('cosine_consistency', float('nan')):+.3f})\n\n"
                )
                f.write("| α | output |\n|---|---|\n")
                direction = bundle.axes[axis][layer]
                for alpha in args.alphas:
                    print(f"[gen]    axis={axis} layer={layer:2d} α={alpha:+.1f}", flush=True)
                    text = steered_generate(
                        model, tokenizer, args.prompt,
                        layer_idx=layer, direction_np=direction, alpha=alpha,
                        max_tokens=args.max_tokens,
                    )
                    one_line = " ".join(text.replace("|", "\\|").split())
                    f.write(f"| {alpha:+.1f} | {one_line} |\n")
                f.write("\n")
                f.flush()

    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
