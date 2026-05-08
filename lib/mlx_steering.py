"""MLX-side: layer-wrapping primitives for capture and steering injection.

MLX has no PyTorch-style `register_forward_hook`, so we wrap individual
`model.layers[i]` instances with thin `nn.Module` subclasses that either
**capture** the layer's output (for probe extraction) or **add an
offset** to it (for steering injection).

The wrappers explicitly carry through the `is_linear` attribute that
the Qwen3.5 hybrid-attention parent reads to decide which mask to
apply (linear vs full attention layer). If you swap the model to a
homogeneous-attention transformer (Llama, Mistral), this attribute
just won't be present on the base and the wrapper falls back cleanly.

Pattern for use:

    originals = wrap_capture(model, [11, 15, 19])
    try:
        _ = model(input_ids)              # forward; captures populated
        for i, layer in originals.items():
            h = model.layers[i].captured  # mx.array (B, T, D)
    finally:
        unwrap(model, originals)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------

class CapturingLayer(nn.Module):
    """Wraps a transformer layer; saves the layer's output for inspection."""

    def __init__(self, base) -> None:
        super().__init__()
        self.base = base
        if hasattr(base, "is_linear"):
            self.is_linear = base.is_linear
        self.captured: Optional[mx.array] = None

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        out = self.base(x, mask, cache)
        self.captured = out
        return out


class SteeringLayer(nn.Module):
    """Wraps a transformer layer; adds a precomputed offset to its output.

    `offset` should be a 1-D `mx.array` of shape `(d_model,)` matching
    the layer's hidden width and dtype. Broadcast across the time axis
    happens automatically.
    """

    def __init__(self, base, offset: mx.array) -> None:
        super().__init__()
        self.base = base
        if hasattr(base, "is_linear"):
            self.is_linear = base.is_linear
        self.offset = offset

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        out = self.base(x, mask, cache)
        return out + self.offset


# ---------------------------------------------------------------------------
# Wrap / unwrap helpers
# ---------------------------------------------------------------------------

def wrap_capture(model, layer_indices: List[int]) -> Dict[int, "nn.Module"]:
    """Replace `model.layers[i]` with `CapturingLayer(original)` for each i.

    Returns the originals dict so the caller can `unwrap` afterwards.
    """
    originals: Dict[int, "nn.Module"] = {}
    for i in layer_indices:
        originals[i] = model.layers[i]
        model.layers[i] = CapturingLayer(originals[i])
    return originals


def wrap_steering(
    model,
    offsets_by_layer: Dict[int, mx.array],
) -> Dict[int, "nn.Module"]:
    """Replace `model.layers[i]` with `SteeringLayer(original, offset)`.

    `offsets_by_layer` is `{layer_idx: combined_offset_mx_array}`. The
    caller is responsible for combining multi-axis steering into one
    offset per layer before calling.
    """
    originals: Dict[int, "nn.Module"] = {}
    for i, offset in offsets_by_layer.items():
        originals[i] = model.layers[i]
        model.layers[i] = SteeringLayer(originals[i], offset)
    return originals


def unwrap(model, originals: Dict[int, "nn.Module"]) -> None:
    """Restore each `model.layers[i]` from `originals[i]`."""
    for i, base in originals.items():
        model.layers[i] = base


# ---------------------------------------------------------------------------
# High-level extraction
# ---------------------------------------------------------------------------

def extract_last_token_hiddens(
    model,
    tokenizer,
    text: str,
    layer_indices: List[int],
) -> Dict[int, np.ndarray]:
    """Forward `text`, return last-token residual at each requested layer.

    Wraps & unwraps within the call. Returned arrays are numpy float32
    so they can be averaged / pickled without further conversion.
    """
    ids = tokenizer.encode(text)
    inputs = mx.array(ids)[None, :]

    originals = wrap_capture(model, layer_indices)
    try:
        _ = model(inputs)
        last_idx = inputs.shape[1] - 1
        results: Dict[int, np.ndarray] = {}
        for i in layer_indices:
            captured = model.layers[i].captured
            assert captured is not None, f"Layer {i} did not capture (forward path?)"
            h = captured[0, last_idx, :]
            mx.eval(h)
            # Cast to float32 for numerical stability on the CPU side;
            # MLX bf16 → numpy float32 via direct conversion.
            results[i] = np.array(h.astype(mx.float32), copy=True)
    finally:
        unwrap(model, originals)

    return results


# ---------------------------------------------------------------------------
# Projection readout (read side; mirrors steering)
# ---------------------------------------------------------------------------

def project_VAD(
    hidden: mx.array | np.ndarray,
    directions_np: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Project a hidden-state vector onto unit-norm V/A/D directions.

    `hidden` is `(d_model,)` — caller selects the token (e.g. last
    non-pad of generation). `directions_np` maps axis name to numpy
    unit vector.
    """
    if isinstance(hidden, mx.array):
        hidden_np = np.array(hidden.astype(mx.float32), copy=False)
    else:
        hidden_np = hidden.astype(np.float32)
    out: Dict[str, float] = {}
    for axis, vec in directions_np.items():
        out[axis] = float(np.dot(hidden_np, vec.astype(np.float32)))
    return out
