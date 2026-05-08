# Getting started: running the project from scratch with a new LLM

This walks through every step from a fresh clone to a closed-loop
persona-sweep figure, with explicit notes at each step on what changes
when you swap the LLM. The bundled probes were extracted from
`mlx-community/Qwen3.5-9B-MLX-4bit` and won't transfer to other
models - use this doc when you want to re-extract for a different
backbone.

The total wall-clock for a fresh end-to-end run (clone → figure) is
**roughly 30-45 minutes** on an M-series Mac, dominated by the model
download and probe extraction.

---

## Prerequisites

- **Apple Silicon Mac** (M1/M2/M3/M4). The default path is MLX-native;
  CUDA / PyTorch is not used.
- **Python 3.11+**.
- **~12 GB free disk** for one 4-bit 8-9B model. Disk-budget more if
  you plan to compare backbones.
- **HuggingFace access**. `mlx_lm.load()` downloads from the Hub on
  first run; gated models need `huggingface-cli login` first.

---

## Step 1: clone and install

```bash
git clone git@github.com:matbeedotcom/npc-steering.git
cd npc-steering

python -m venv .venv && source .venv/bin/activate
pip install -e '.[mlx,viz]'      # adds mlx, mlx-lm, matplotlib, numpy
```

Verify the import works:

```bash
python -c "from lib import PRESETS; print(sorted(PRESETS.keys()))"
# → ['aggressive_late', 'anxious_new_driver', 'calm_commuter']
```

---

## Step 2: pick an LLM

Anything `mlx_lm.load()` can load *and* exposes `model.layers[i]` as
an iterable of decoder blocks should work. Practical guidance:

- **Size**: 7-9B parameters is the sweet spot. Below ~3B the V/A/D
  directions get noisy; above ~13B the wall-clock per probe-extraction
  pass starts to bite.
- **Quantization**: 4-bit MLX quants are fine. Steering operates on
  the residual stream, which dequantizes cleanly during inference.
- **Tuning**: an instruction-tuned chat model is required. The
  contrastive pairs are first-person sentences - completion-only base
  models won't continue them in voice.
- **Architecture quirks**: Qwen3.5's hybrid linear/full-attention
  layers carry an `is_linear` flag through the steering wrapper; other
  architectures simply don't have it and the wrapper's `hasattr` check
  handles them transparently.

Three known-working alternatives:

| model id | layers | notes |
|---|---:|---|
| `mlx-community/Qwen3.5-9B-MLX-4bit` | 32 | the default; needs `enable_thinking=False` in chat template |
| `mlx-community/Hermes-3-Llama-3.1-8B-4bit` | 32 | same layer-count, no thinking-mode toggle needed |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | 32 | similar; well-behaved residual stream |

**Layer count matters.** [`scripts/02_extract_probes.py`](../scripts/02_extract_probes.py)
defaults to candidate layers `[11, 15, 19, 23, 27]` (mid-stack of 32).
For a model with a different layer count, pass `--layers` explicitly
- e.g. for a 28-layer model, try `--layers 9 13 17 21 25`.

For the rest of this guide, set:

```bash
export MODEL=mlx-community/Hermes-3-Llama-3.1-8B-4bit  # or your choice
```

---

## Step 3: extract V/A/D probes

This forwards 12 contrastive sentence pairs per axis through the
model, computes the mean-difference vector at each candidate layer,
and saves a `ProbeBundle`.

```bash
python scripts/02_extract_probes.py --model "$MODEL"
# → artifacts/probes_<model_short_name>.pkl  (~250 KB)
```

If your model has a non-32 layer count, override candidates:

```bash
python scripts/02_extract_probes.py --model "$MODEL" --layers 9 13 17 21 25
```

Time: **2-5 min** including the model download (one-shot per model).
Subsequent runs reuse the HuggingFace cache.

What this produces: a pickle containing per-axis (V, A, D) per-layer
direction vectors plus diagnostics (mean pairwise cosine consistency
across the pairs - a proxy for "is this axis well-defined?"). Higher
cosine = better.

---

## Step 4: validate probes and pick a layer

The validation script does an α-sweep at each candidate layer and each
axis, generating sample completions so you can read which layer
produces the cleanest affective shift without breaking coherence.

```bash
python scripts/03_validate_probes.py \
  --probes "artifacts/probes_$(basename $MODEL).pkl"
# → artifacts/probes_validation.md  (markdown table of α-sweep samples)
```

Time: **3-7 min**.

**Read the report.** For each axis, you're looking for a layer where:

- α=0 produces neutral, in-character output
- α=+25 (positive direction) clearly shifts the felt valence/arousal/
  dominance up - the shift should be *semantic*, not just lexical
- α=-25 clearly shifts it down
- coherence holds at |α|=25; outputs only break at |α|≥50

Layers that produce meta-commentary ("Thinking Process:"), n-gram
loops, or syntax breaks at moderate α should be skipped. Earlier
layers (≤9) tend to encode low-level features; later layers (≥27)
tend to be locked into the model's response template. The mid-stack
sweet spot is usually layer 13-19 in a 32-layer model.

For Qwen3.5-9B the picked layer was **15** for all three axes. For a
different backbone, expect a similar ballpark but verify per-axis -
sometimes V and A want the same layer and D wants a different one.

---

## Step 5: write the selection back to the bundle

Once you know the layer per axis, save it onto the bundle so
downstream code can find it. There's no separate script - it's a
two-line snippet:

```bash
python -c "
from lib.probes import ProbeBundle
b = ProbeBundle.load('artifacts/probes_$(basename $MODEL).pkl')
b.selected_layers = {'V': 15, 'A': 15, 'D': 15}   # ← whatever you picked
b.save('artifacts/probes_$(basename $MODEL).pkl')
print('selected:', b.selected_layers)
"
```

Verify:

```bash
python -c "
from lib.probes import ProbeBundle
b = ProbeBundle.load('artifacts/probes_$(basename $MODEL).pkl')
print('model_id      :', b.model_id)
print('selected_layers:', b.selected_layers)
print('axes          :', list(b.directions.keys()))
"
```

---

## Step 6: smoke-test the steered agent

Single charged event, three personas, prints the utterances directly
to the terminal. Fastest way to confirm the closed loop works
end-to-end.

```bash
python scripts/04_test_agent.py \
  --probes "artifacts/probes_$(basename $MODEL).pkl"
```

Time: **30-90 s** (loads the model once).

You should see three first-person utterances, each in distinct voice
- the calm commuter understated, the aggressive driver clipped and
imperative, the new driver fragmented and panicky. If all three
sound the same, the persona cards aren't loading; if they're
incoherent or refuse, your steering gain may be too high (see
Troubleshooting).

---

## Step 7: closed-loop single run

One scenario, one persona, the full discrete-event loop with
re-appraisal.

```bash
python scripts/05_drive_agent.py \
  late_school_run aggressive_late \
  --probes "artifacts/probes_$(basename $MODEL).pkl"
# → runs/late_school_run__aggressive_late.agent.jsonl
```

Time: **30-90 s** depending on event count.

Inspect the trace:

```bash
head -5 runs/late_school_run__aggressive_late.agent.jsonl
```

Each line is a JSON record (`kind` ∈ `{meta, state, event, decision,
utterance}`).

---

## Step 8: persona-sweep figure (the money shot)

Three personas through the same scenario, one comparison figure.

```bash
python scripts/06_persona_sweep.py \
  --scenario mixed_emotions \
  --probes "artifacts/probes_$(basename $MODEL).pkl"
# → artifacts/persona_sweep_mixed_emotions.png
```

Time: **2-4 min**.

Open the PNG. You should see three rows (one per persona), each with
V/A/D + frustration trajectories and the per-event utterance log.
Different personas should react visibly differently to the same
event stream, especially around the cut-off (t≈12s) and the
courtesy gestures (t≈14s, t≈42s).

If the three rows look near-identical, either the persona cards are
not differentiating or the steering gain is too low for the new
backbone. Try increasing `g_V`/`g_A`/`g_D` in
[`lib/probes.py`](../lib/probes.py) by 10-20%.

---

## Step 9 (optional): re-run empirical calibrations for the new backbone

The bundled [`data/empirical_responses.json`](../data/empirical_responses.json)
contains per-(event, persona) ΔV/ΔA/ΔD measured on **Qwen3.5-9B**. The
appraisal layer is happy to reuse those numbers across backbones, but
the values were specifically the response of Qwen3.5 at neutral
state, so they aren't strictly representative of a new model's felt
response.

To re-measure for your backbone:

```bash
python scripts/07_calibrate_appraisal.py \
  --probes "artifacts/probes_$(basename $MODEL).pkl"
# → data/empirical_responses.json  (overwritten; old version not preserved)
```

Time: **15-25 min** (~180 LLM generations × persona × event × axis).

Resume is supported (writes a checkpoint after each (event, persona)
measurement) - if it crashes or you Ctrl-C, just re-run.

If you'd rather keep the Qwen3.5 calibration as a baseline:

```bash
cp data/empirical_responses.json data/empirical_responses.qwen35.json
python scripts/07_calibrate_appraisal.py --probes "..."
```

---

## Step 10 (optional): arousal-gate ablation

This is the Shangguan 2025 reproduction. Re-runs `mixed_emotions` for
3 personas × {gate off, gate on} and produces the persona-distinctness
metric.

```bash
python scripts/09_arousal_gate_ablation.py \
  --probes "artifacts/probes_$(basename $MODEL).pkl"
# → artifacts/arousal_gate_distinctness_mixed_emotions.png
# → notes/arousal_gate_ablation.md
```

Time: **6-10 min**.

The expected qualitative result: `gain_above > gain_below` (persona
distinctness grows more during high-arousal regimes than calm ones).
If a different backbone gives the opposite result, that's an
interesting finding - the closed-loop coupling differs by model.

---

## Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| all outputs start `"Thinking Process: 1. **Analyze the Request:**"` | Qwen3 thinking-mode template is on | `enable_thinking=False` is already set in [`lib/agent.py`](../lib/agent.py); confirm tokenizer applies it |
| outputs collapse into n-gram repetition at high \|α\| | "coherence cliff" past steering capacity | reduce `g_V`/`g_A`/`g_D` in [`lib/probes.py`](../lib/probes.py) (default 25.0) |
| three personas all sound the same | persona cards not loading, or steering gain too low | check `data/persona_cards.json` has entries for each persona name; bump gains 10-20% |
| persistent refusals from one persona | safety guardrails triggering on aggressive persona | already handled by retry-at-higher-temperature in [`lib/agent.py`](../lib/agent.py); if still failing, soften the persona card's character description |
| probe extraction OOMs | model doesn't fit in unified memory | use a smaller backbone or lower-bit quant; an M-series 16 GB Mac comfortably handles 4-bit 7-9B |
| validation report shows ~0 cosine consistency | contrastive pairs aren't landing on this model | the model is likely not chat-tuned, or the prompt template is wrong; try a different backbone |

---

## What's regenerable vs what's pinned

When swapping LLMs, here's what changes:

**Regenerated** (run the corresponding step):
- `artifacts/probes_<model>.pkl` - per-model direction vectors
- `selected_layers` on the bundle - per-model layer choice
- `data/empirical_responses.json` - per-model appraisal Δ table (optional re-cal)
- All `runs/*.jsonl` traces - per-model utterances

**Stays the same** (model-agnostic):
- `lib/persona.py` - the trait-vector definitions
- `data/persona_cards.json` - the LLM voice anchors
- `data/contrastive_pairs.json` - the V/A/D probe-extraction prompts
- `scenarios/*.json` - the timed event streams
- `lib/state.py`, `lib/events.py`, `lib/appraisal.py` (rules) - the symbolic dynamics

---

For deeper context on what each piece is doing, see
[`notes/writeup.md`](../notes/writeup.md) (the 1-page narrative) and
[`notes/design.md`](../notes/design.md) (the engineering spine).
