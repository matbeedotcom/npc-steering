#!/usr/bin/env python3
"""Extract V/A/D probe directions from an MLX-quantised LLM.

Approach: contrastive activation differencing (CAA-style; Rimsky 2023,
Zou 2023). For each axis we forward-pass 12 paired sentences differing
along that axis and take the mean difference of the last-token residual
stream at each candidate layer. Result: one direction vector per
(axis, layer); layer selection happens in `03_validate_probes.py`.

Output: `artifacts/probes_<model_short>.pt` — pickled `ProbeBundle`
loaded back via `ProbeBundle.load`.

Default model is `mlx-community/Qwen3.5-9B-MLX-4bit`: a 32-layer
hybrid linear/full-attention model where full-attention layers live at
indices [3, 7, 11, 15, 19, 23, 27, 31]. Default candidate set sticks
to mid-stack full-attention layers, where high-level concept directions
typically live.

Usage:
  python scripts/02_extract_probes.py
  python scripts/02_extract_probes.py --model mlx-community/Hermes-3-Llama-3.1-8B-4bit
  python scripts/02_extract_probes.py --layers 11 15 19 23 27 --output artifacts/probes_test.pt
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

import mlx.core as mx                                        # noqa: E402
from mlx_lm import load                                       # noqa: E402

from lib.probes import ProbeBundle                            # noqa: E402
from lib.mlx_steering import extract_last_token_hiddens       # noqa: E402


DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
# Mid-stack full-attention layers for Qwen3.5-9B (32 layers, full-attn at
# [3, 7, 11, 15, 19, 23, 27, 31]). Linear-attention layers are valid
# candidates too but tend to mix tokens differently — start with the
# attention layers and expand if probe quality is poor.
DEFAULT_LAYERS = [11, 15, 19, 23, 27]
DEFAULT_PAIRS_PATH = _ROOT / "data" / "contrastive_pairs.json"


def _short_model_name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _default_output_path(model_id: str) -> Path:
    return _ROOT / "artifacts" / f"probes_{_short_model_name(model_id)}.pkl"


# ---------------------------------------------------------------------------
# Per-axis probe computation
# ---------------------------------------------------------------------------

def compute_axis_probes(
    pairs: List[dict],
    model,
    tokenizer,
    layers: List[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """Run all positive/negative pairs, return (per-layer vectors, diagnostics).

    Vectors are L2-normalised float32. Diagnostics include the mean
    pairwise cosine similarity of per-pair differences (a proxy for how
    *consistent* the pairs are — high means they all encode the same
    direction; low means they sprawl).
    """
    per_pair_diffs: List[Dict[int, np.ndarray]] = []
    for pair in pairs:
        h_pos = extract_last_token_hiddens(model, tokenizer, pair["pos"], layers)
        h_neg = extract_last_token_hiddens(model, tokenizer, pair["neg"], layers)
        per_pair_diffs.append({L: h_pos[L] - h_neg[L] for L in layers})

    vectors: Dict[int, np.ndarray] = {}
    diagnostics: Dict[int, dict] = {}
    for L in layers:
        layer_diffs = np.stack([d[L] for d in per_pair_diffs])  # (N, D)
        mean_diff = layer_diffs.mean(axis=0)
        norm = float(np.linalg.norm(mean_diff))
        unit = mean_diff / (norm + 1e-8)

        # Cosine consistency across pairs: do they all point the same way?
        diff_norms = np.linalg.norm(layer_diffs, axis=1, keepdims=True) + 1e-8
        normed = layer_diffs / diff_norms
        cos = normed @ normed.T   # (N, N)
        n = cos.shape[0]
        off_diag = cos[~np.eye(n, dtype=bool)]
        mean_cos = float(off_diag.mean())

        vectors[L] = unit.astype(np.float32)
        diagnostics[L] = {
            "norm_unnormalised": norm,
            "cosine_consistency": mean_cos,
            "n_pairs": int(n),
            "residual_norm_avg": float(np.linalg.norm(layer_diffs, axis=1).mean()),
        }
    return vectors, diagnostics


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"MLX model id (default: {DEFAULT_MODEL}).")
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS_PATH),
                    help="Path to contrastive_pairs.json.")
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                    help=f"Candidate layer indices (default: {DEFAULT_LAYERS}).")
    ap.add_argument("--output", default=None,
                    help="Output bundle path (default: artifacts/probes_<model>.pkl).")
    args = ap.parse_args()

    output_path = Path(args.output) if args.output else _default_output_path(args.model)

    print(f"[mlx]    metal_available={mx.metal.is_available()}")
    print(f"[model]  loading {args.model}...")
    model, tokenizer = load(args.model)
    n_layers = len(model.layers)
    bad = [L for L in args.layers if not 0 <= L < n_layers]
    if bad:
        raise ValueError(
            f"Layer indices {bad} out of range; model has {n_layers} layers (0..{n_layers-1})."
        )
    print(f"[model]  n_layers={n_layers}  candidate_layers={args.layers}")

    pairs_data = json.loads(Path(args.pairs).read_text())
    axes = pairs_data["axes"]
    n_pairs = {ax: len(axes[ax]["pairs"]) for ax in axes}
    print(f"[pairs]  {n_pairs}")

    all_vectors: Dict[str, Dict[int, np.ndarray]] = {}
    all_diagnostics: Dict[str, Dict[int, dict]] = {}
    for axis, axis_data in axes.items():
        print(f"\n[axis]   {axis} ({axis_data['label']}): "
              f"{axis_data['negative_pole']} ↔ {axis_data['positive_pole']}")
        vectors, diagnostics = compute_axis_probes(
            axis_data["pairs"], model, tokenizer, args.layers,
        )
        all_vectors[axis] = vectors
        all_diagnostics[axis] = diagnostics
        for L in args.layers:
            d = diagnostics[L]
            print(
                f"  layer {L:2d}:  ‖mean_diff‖={d['norm_unnormalised']:7.3f}"
                f"   cos_consistency={d['cosine_consistency']:+.3f}"
                f"   ‖h‖_avg={d['residual_norm_avg']:7.2f}"
                f"   (n={d['n_pairs']})"
            )

    bundle = ProbeBundle(
        model_id=args.model,
        extraction_date=datetime.date.today().isoformat(),
        n_pairs_per_axis=int(min(n_pairs.values())),
        candidate_layers=list(args.layers),
        axes=all_vectors,
        diagnostics=all_diagnostics,
        selected_layers={},
    )
    bundle.save(output_path)
    print(f"\n[save]   {output_path}")


if __name__ == "__main__":
    main()
