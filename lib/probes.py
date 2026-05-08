"""Probe storage container.

`ProbeBundle` is a pure data class — numpy arrays inside, no MLX or
torch dependency. Loaders/savers use `pickle` (via `torch.save` if
available, falling back to standard pickle) so the bundle is
portable across runtime backends.

Steering and projection helpers live in `lib/mlx_steering.py` because
they touch `mx.array` directly. Phase C will share both via
`lib/agent.py`.

Why split this way: the *artifact* (extracted probes) is tied to the
model that produced it but otherwise platform-agnostic. Keeping the
storage layer free of MLX imports means a future torch / numpy-only
consumer can read the bundle without needing the whole stack.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class ProbeBundle:
    """All probes for one model. Layer-indexed so validation can pick
    the best layer per axis after extraction.

    `axes[axis_name][layer_idx]` is a unit-normalised np.ndarray of
    shape `(d_model,)` and dtype float32. Extraction may have run on a
    bfloat16 model — we always upcast at save time so downstream
    consumers don't need MLX or torch.
    """

    model_id: str
    extraction_date: str
    n_pairs_per_axis: int
    candidate_layers: List[int]
    axes: Dict[str, Dict[int, np.ndarray]]
    # diagnostics[axis][layer] = {"cosine_consistency": float, "norm": float, ...}
    diagnostics: Dict[str, Dict[int, dict]] = field(default_factory=dict)
    selected_layers: Dict[str, int] = field(default_factory=dict)

    # -- I/O ---------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> "ProbeBundle":
        with open(Path(path), "rb") as f:
            blob = pickle.load(f)
        return cls(**blob)

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(Path(path), "wb") as f:
            pickle.dump(self.__dict__, f)

    # -- Lookup ------------------------------------------------------------

    def vec(self, axis: str, layer: int | None = None) -> np.ndarray:
        """Get the probe direction for an axis. Defaults to the selected layer."""
        if layer is None:
            if axis not in self.selected_layers:
                raise KeyError(
                    f"No selected layer for axis {axis!r}. Either pass `layer` or "
                    f"set `bundle.selected_layers[{axis!r}] = ...` first."
                )
            layer = self.selected_layers[axis]
        return self.axes[axis][layer]


# ---------------------------------------------------------------------------
# State → α mapping (Phase C will tune the gains)
# ---------------------------------------------------------------------------

def state_to_alphas(
    state,
    *,
    g_V: float = 25.0,
    g_A: float = 25.0,
    g_D: float = 25.0,
) -> Dict[str, float]:
    """Map symbolic `DriverState` scalars to per-axis steering coefficients.

    Defaults calibrated against Qwen3.5-9B-MLX-4bit at layer 15
    (`probes_validation.md`): α=±25 produced visible affective shifts
    while staying grammatically coherent; α=±50 hit a coherence
    cliff (output collapsed into repetition or syntax breaks).
    Per-axis equal gains because the V/A/D probe magnitudes after
    L2-normalisation are equal by construction; differences in axis
    *quality* (cos_consistency: V=0.38, A=0.25, D=0.18) translate
    into differences in steering precision, not magnitude.

    Different model / layer / quantisation combinations will need
    re-calibration via `03_validate_probes.py`.

    `state` exposes `valence`, `arousal`, `dominance` floats in
    `[-1, +1]` (see `lib/state.py`).
    """
    return {
        "V": g_V * state.valence,
        "A": g_A * state.arousal,
        "D": g_D * state.dominance,
    }
