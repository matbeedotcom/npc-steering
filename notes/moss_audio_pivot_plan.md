# MOSS-Audio Pivot Plan — Audio-Native V/A/D Probes

*Plan doc. Companion to [`writeup.md`](writeup.md) and
[`design.md`](design.md). Closes the "calibrated probes from a paired
audio→text emotional speech dataset" item in writeup §What's Next.*

> **Status:** Pre-spike. Compute moved to dual RTX 4090 (CUDA),
> which lifts the MLX-only constraint that pinned the original
> implementation to `mlx-community/Qwen3.5-9B-MLX-4bit`. This
> document plans the audio-native pivot end-to-end: Phase 0 spike
> to de-risk three architectural unknowns, then either Path A
> (full audio-native probes on MOSS-Audio) or Path B (MOSS-Audio
> as labeling oracle, Qwen3.5 stays as the affect engine).

## 1. Goal

Replace the current contrastive-text probes (12 sentence pairs per
axis on Qwen3.5-9B residual stream at L15, cos consistency
V=0.377 / A=0.246 / D=0.183) with **regression probes fit on
labeled emotional audio**. Two improvements compound:

1. **Better labels.** Human-rated dimensional V/A/D from emotional
   speech corpora carry prosody-grounded affect that text-only
   contrastive pairs can't reach — especially for the dominance
   axis, which currently conflates control / agency / will.
2. **Continuous supervision at scale.** 12 binary pairs →
   thousands of dimensional ratings; `mean(h_pos) − mean(h_neg)` →
   multivariate regression of hidden states on V/A/D scalars,
   which jointly orthogonalises the three axes.

The deeper question this pivot answers: **does the architecture's
"same vectors write and read state" claim transfer to an
audio-native model?** Path A tests it directly; Path B keeps the
text architecture but upgrades its labels.

## 2. Hardware & runtime

- 2 × RTX 4090 (24 GB VRAM each, 48 GB total).
- MOSS-Audio-8B-Thinking weights ≈ 16 GB at bf16; serving with
  KV-cache + audio tokenizer + reasonable context fits a single
  4090 with 4-bit quantization, or both 4090s with bf16 +
  tensor-parallelism. Probe extraction (no autoregressive cache)
  is the easy case: bf16 forward passes batch comfortably on
  one 4090.
- GPU 1 stays free during Phase 0 → Phase 1 for parallel batch
  extraction or to keep Qwen3.5-9B loaded for Path B fallback.
- Existing repo clones cleanly to the 4090 box;
  [`pyproject.toml`](../pyproject.toml) needs a `[cuda]` extra
  alongside the existing `[mlx]` extra (torch, transformers,
  accelerate, flash-attn).

## 3. Architectural unknowns

Three things must work for Path A to be worth the larger refactor.
All three resolve in Phase 0.

**3.1 — Thinking-mode escape hatch.** The `-Thinking` suffix
mirrors Qwen3's thinking mode, which broke probe extraction
during V1 calibration (writeup §Calibration findings: "all outputs
collapse into meta-commentary that drowns the affective shift").
Need either an `enable_thinking=False` equivalent in the chat
template, a non-thinking sibling model, or a probe layer that
fires before `<think>`-block tokens dominate the residual stream.

**3.2 — Generation capability.** MOSS-Audio's listed capabilities
are heavy on *understanding* (transcribe, summarize, analyze).
The architecture's central claim depends on the model being a
*speaker*, not just an analyst — it must produce in-character
first-person continuations (or affectively-modulated speech),
not third-person reports. Rejection criterion at spike time:
"The driver appears to be experiencing frustration" instead of
"Eight goddamn minutes."

**3.3 — Audio contrastive-pair sourcing.** Steering currently
uses 12 hand-written sentence pairs per axis. Audio probes need
audio pairs — either sourced from labeled corpora (MSP-Podcast /
IEMOCAP), drawn at the high/low tail of dimensional ratings, or
synthesised via emotional TTS (with the synthetic-data
circularity caveats from writeup §Limitations: TTS prosody
encodes the TTS engine's stereotype of an emotion, not human
affect).

## 4. Phase 0 — De-risk spike (1–2 days, GPU 0)

Single decision gate: do all three unknowns resolve favourably?

**0.1 — Smoke test.** Pull MOSS-Audio-8B-Thinking via
transformers; confirm bf16 load on a single 4090.

```bash
huggingface-cli download OpenMOSS-Team/MOSS-Audio-8B-Thinking
python tools/driver_affect_sim/scripts/00_moss_smoke.py
```

`00_moss_smoke.py` (new): loads model + tokenizer, runs one audio
input forward pass, prints layer count / d_model / param count,
verifies activation hooks attach cleanly. Mirrors what
[`02_extract_probes.py`](../scripts/02_extract_probes.py) does at
its first `extract_last_token_hiddens` call.

**0.2 — Thinking-mode inspection.** Read the chat template;
locate the `<think>` token boundary; compare token streams of
identical input with thinking-on vs thinking-off (or
post-`<think>`-scrub). If thinking is non-defeatable, candidate
probe layers must sit *before* the thinking tokens take over
the residual stream — the Qwen3 finding (writeup
§Calibration findings: "Layers ≥23 absorbed all steering into a
fixed Thinking Process preamble") will translate.

**0.3 — Generation test.** Feed the writeup's `mixed_emotions`
scenario as text + persona card; ask for a first-person
continuation in each of the three personas. Compare
output style / voice / register to the current Qwen3.5
baseline (transcripts in [`docs/transcripts/`](../docs/transcripts/)).
Pass = clearly persona-distinct first-person utterances. Fail =
generic analytical responses.

**0.4 — Single-pair probe sanity check.** Two emotional audio
clips from a freely-licensed corpus (CREMA-D works for this
spike — even though categorical, you only need one
clearly-positive-V and one clearly-negative-V clip), forward
through MOSS, extract last-audio-token hiddens across candidate
layers, look for a layer with non-trivial cosine separation on
a candidate V direction. CAA-style minimal version of the
existing extractor.

**0.5 — Decision.** Path A if 3.1, 3.2, and 3.4 all pass.
Path B if 3.1 or 3.2 fails but the model is still usable as an
audio classifier (3.4 doesn't matter for B). Hard stop if the
model is unusable for our purposes (treat this as a real
possibility — write the spike to fail fast).

## 5. Phase 1A — Path A: full audio-native probes (1–2 weeks)

**5.1 — Corpus acquisition.** MSP-Podcast (UT Dallas license
form, 3–5 day turnaround) as primary; IEMOCAP (USC license,
also form-gated) as cross-corpus validation. Both are
research-licence-only — file the forms on day 0 of Phase 1A
so they're in hand by the time the extraction pipeline is ready.

**5.2 — Hidden-state extraction pipeline.**
[`scripts/02b_extract_probes_regression.py`](../scripts/02b_extract_probes_regression.py)
(new): for each clip, audio → MOSS-Audio token sequence → forward
pass → save per-layer (last-audio-token, mean-pooled-audio-tokens)
hidden state. Output: `(N_clips, N_layers, 2, D_model)` tensor +
aligned `(N_clips, 3)` V/A/D label tensor. ~250 hr MSP-Podcast at
moderate batch size on one 4090 ≈ wall time of hours, not days.

**5.3 — Fit regression probes.** `MultiTaskElasticNet(alpha,
l1_ratio)` from sklearn on `(hidden, V/A/D)`. The three columns
of the regression weight matrix become the three probe
directions, jointly orthogonalised by the regression. L2-normalise
for compatibility with the existing
[`ProbeBundle`](../lib/probes.py) format. Save into the same
pickle layout — only `axes` content changes; downstream consumers
(steering, projection) stay the same.

**5.4 — Layer selection.** Empirical, not heuristic. Pick the
per-axis layer that maximises held-out Pearson r between probe
projection and label. Replaces the current "L15 because we tried
it and it worked" with a principled pick. Expectation: V and A
land mid-stack as before; D may move because the regression has
access to dominance signal that contrastive pairs couldn't isolate.

**5.5 — Validation.** Adapt
[`03_validate_probes.py`](../scripts/03_validate_probes.py) to
emit α=±25 continuations on a fixed set of driving prompts and
read them back via projection. Same coherence-cliff check
(syntax breaks, n-gram repetition above |α|=50). If
MOSS-Audio generates speech, add a perceptual check — listen to
3 clips per axis per pole.

**5.6 — Re-run downstream ablations.** All of
[`06_persona_sweep.py`](../scripts/06_persona_sweep.py) through
[`10_card_vs_steering_ablation.py`](../scripts/10_card_vs_steering_ablation.py)
with new probes. Things to watch:

- D-axis cosine consistency should climb from 0.18 — that's the
  axis with the most headroom and the one most likely to benefit
  from prosody-grounded labels.
- The `card_only` vs `steer_only` r(V) tradeoff
  (writeup §card vs steering ablation: 0.382 / 0.398
  felt-distinctness, −0.109 vs +0.625 r(V)) may rebalance. If
  audio-trained probes carry voice-register information they
  didn't before, the gap could narrow — meaning steering-alone
  might no longer sound generic.

**5.7 — Writeup update.** New section in
[`writeup.md`](writeup.md): "Empirical anchoring: audio-calibrated
probes". Direct comparison table: contrastive vs regression
probes on the same axis. Removes the corresponding bullet from
§Limitations.

## 6. Phase 1B — Path B: MOSS-Audio as labeling oracle (3–5 days)

Fallback if 3.1 or 3.2 fails — keeps the existing text
architecture, upgrades the labels.

**6.1 — Inference-only labeling.** Use MOSS-Audio as a V/A/D
labeler on MSP-Podcast clips. Prompt-engineer for dimensional
output ("rate this clip's valence/arousal/dominance from −1 to
+1, no commentary"). Cross-validate against MSP-Podcast's
human-rated dimensional labels on a held-out subset; if MOSS
labels track human ones with r > 0.6 per axis, trust them at
scale. If not, fall back to human labels alone.

**6.2 — Fit regression probes on Qwen3.5.** Run transcripts
through `Qwen3.5-9B-MLX-4bit` (or its CUDA-friendly sibling on
4090, since we now have the option) and fit
`MultiTaskElasticNet` on `(hidden, MOSS-V/A/D-labels)`. Same
[`ProbeBundle`](../lib/probes.py) output format.

**6.3 — Validation + ablations.** Same sequence as 5.5–5.6.

**6.4 — Writeup update.** Smaller — same architecture, better
probes. One paragraph in §Empirical anchoring + a note in §What's
Next that the audio-native version is staged behind 3.1/3.2
unblocks.

## 7. Phase 2 — Optional: synthetic augmentation

Only after Path A or B baseline is in. Generate driver-domain
utterances via emotional TTS with V/A/D conditioning labels,
covering the appraisal-event taxonomy from
[`lib/events.py`](../lib/events.py) (cut-off, near-miss, merge
gap, congestion, red light, weather, late-for-appointment,
passenger comment) at densities the natural corpora don't cover.
Mix into the calibration set, refit probes, ablate.

Synthetic-data circularity (writeup §Limitations reasoning) still
applies: TTS prosody encodes the TTS engine's stereotype of an
emotion, not human affect. Use only as augmentation on top of
naturalistic data — never as the sole calibration source. Drop
if the augmented probes don't beat the naturalistic-only ones on
held-out human-rated clips.

## 8. Deliverables

- New: [`scripts/00_moss_smoke.py`](../scripts/00_moss_smoke.py) — Phase 0 spike harness
- New: [`scripts/02b_extract_probes_regression.py`](../scripts/02b_extract_probes_regression.py) — regression-flavour extractor
- New: [`scripts/11_contrastive_vs_regression_ablation.py`](../scripts/11_contrastive_vs_regression_ablation.py) — direct probe-method comparison
- New: `data/probe_calibration_set.json` — clip ids, splits, V/A/D labels
- New: `artifacts/probes_MOSS-Audio-8B.pkl` (Path A) **or** `artifacts/probes_Qwen3.5-9B_audio_calibrated.pkl` (Path B)
- New: `[cuda]` extra in [`pyproject.toml`](../pyproject.toml) — torch, transformers, accelerate, flash-attn
- Update: [`writeup.md`](writeup.md) — new §Empirical anchoring subsection, drop the corresponding §Limitations bullet
- Update: [`docs/getting_started.md`](../docs/getting_started.md) — CUDA path alongside the existing MLX path

## 9. Out of scope

- Replacing the symbolic-state layer. The "six floats outside the
  model are authoritative" claim (writeup §What's Novel point 2)
  is the part of the architecture this pivot is supposed to make
  *more* defensible, not undermine.
- Replacing hand-authored decay τ. The negative result in writeup
  §"A productive negative result" stands — biometric calibration
  (SHRP2 / UDRIVE HRV) is the right answer for τ, not audio
  probes. The R² ≥ 0.3 floor in [`lib/state.py`](../lib/state.py)
  stays as the gate.
- Co-training the persona vector. Separate workstream. The pivot
  here is about probes, not personas.
- Coupling to a real driving sim (highway-env, CARLA). Listed in
  writeup §What's Next as a separate follow-up.

## 10. Open questions for the spike

These get answered by Phase 0, not by this doc:

- Does MOSS-Audio-8B-Thinking emit audio output, or only text?
  (Affects whether steering operates on speech generation or text
  generation.)
- What's the residual-stream shape — single trunk, or separate
  audio / text encoders that merge late? (Affects which layers
  are probe candidates.)
- Is the `MOSS-Audio-Tokenizer` independent enough to use with a
  different LM trunk if MOSS itself doesn't pass the spike?
  (Hedge: keeps audio-token approach alive even if the
  Thinking model is unusable.)
