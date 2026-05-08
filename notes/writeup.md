# Driver Affect Simulator — Closed-Loop Activation Steering

*One-page application piece. Engineering notes in `design.md`,
implementation in `tools/driver_affect_sim/`.*

## Problem

NPC drivers in autonomous-vehicle simulation are mostly cardboard:
either deterministic IDM controllers, or LLM-prompted agents whose
behaviour is whatever the prompt happens to elicit on a given turn.
Neither produces the *causally coherent, individually distinct,
persona-driven* trajectories an AV planner needs to learn against.
Cardboard NPCs make synthetic miles cheap but uninformative.

## Approach

A closed-loop affective agent in which **symbolic V/A/D state
dynamics control activation-space steering of a frozen LLM, and the
LLM's outputs are re-projected onto the same probe directions to
update the symbolic state**. Architecture in three layers:

1. **Symbolic affect dynamics** — six-scalar state (V/A/D + stress,
   frustration, fatigue) with persona-modulated decay equations and
   hand-authored appraisal rules over an 8-event taxonomy
   (cut-off, near-miss, merge gap, congestion, red light, weather,
   late-for-appointment, passenger comment). Transparent, legible,
   evolves on event timescales.
2. **Activation steering** — three direction vectors `(v_V, v_A, v_D)`
   in the residual stream of `mlx-community/Qwen3.5-9B-MLX-4bit`,
   extracted from 12 contrastive prompt pairs per axis. The same
   vectors are used both *to write* state into the model (additive
   injection at layer 15) and *to read* expressed state out of the
   model (projection of the post-output hidden state).
3. **Closed loop** — the felt V/A/D from the readout is fed back
   into the symbolic layer as a re-appraisal event, so the model's
   own utterances modulate its subsequent state. Affect closes in
   representation space, not in context tokens.

A separate decision-coupling layer turns the same symbolic state
into driving parameters (gap acceptance, following distance, speed
overage), giving the AV simulator behavioural diversity *causally
tied* to the NPC's emotional trajectory.

## Demo

![Persona sweep on `mixed_emotions` (V3 calibrated)](../artifacts/persona_sweep_mixed_emotions_v3.png)

**Figure caption.** Three personas — `calm_commuter`,
`aggressive_late`, `anxious_new_driver` — driven through the same
60-second `mixed_emotions` scenario: 8 minutes late dropping a kid
at school; stuck in traffic; cut off at *t*=12s; another driver
*lets ego merge with a friendly wave* at *t*=14s; passenger remark
at *t*=25s; *polite-pass courtesy* at *t*=42s; tight merge gap at
*t*=55s. Each row shows one persona's V/A/D and frustration under
closed-loop steering with **empirically-calibrated appraisal
deltas (V3)**, paired with the LLM's time-stamped internal
monologues. The persona vector is the only knob; everything
downstream — symbolic dynamics, steering offsets, monologue voice
— follows.

**The two courtesy events are the methodologically-novel test.**
The empirical calibration revealed that the LLM's representation
of courtesy is **per-persona sign-flipped**: `calm_commuter`
registers ΔV ≈ +0.08 (genuine appreciation), `aggressive_late`
registers ΔV ≈ −0.16 (the persona reads kindness as suspicion or
inconvenience even with no prior frustration), and
`anxious_new_driver` registers ΔV ≈ +0.04 (mild ambivalence — the
"burden of obligation" reading). Hand-authored deltas would have
forced all three to lift positively. The empirical layer encodes
each persona's actual lens, and the figure shows the symbolic
state and the LLM utterance now *agreeing*: at *t*=14
(let-merge), `calm_commuter` says *"They didn't have to do that,
but I won't let them down today either"* (reciprocal kindness),
`aggressive_late` produces a confused fragment that doesn't
register the courtesy positively, and `anxious_new_driver` says
*"Okay, they waved. If I just breathe and thank them?"* (anxiety-
flavored uncertainty).

**Same input, three qualitatively distinct trajectories** in
state-space, voice-space, AND in *how each persona perceives the
same kind moment*. That last layer is where the closed-loop
empirical calibration earns its keep.

## Calibration iteration: prompt construction matters as much as method

The empirical-appraisal calibration ran three times before settling.
The full record is on disk; the short version:

- **V1** (initial, "you are speaking your internal monologue"
  prompt): produced strong empirical signals but Qwen3.5's safety
  alignment refused some `aggressive_late` outputs verbatim
  (*"I can't generate content that encourages dangerous driving..."*).
- **V2** (added "fictional behavioural-research simulation" framing
  to suppress refusals): refusals stopped, but the
  research-distancing language *dampened the LLM's affective
  embodiment* — Δ magnitudes shrank 4–10×, and an aggressive driver
  near-collision reading produced ΔV ≈ 0.00. The empirical layer
  became uninformative.
- **V3** (creative-writing exercise framing): kept the structural
  protections (`<think>`-block scrub, refusal retry at higher temp)
  while restoring the emotional commitment of v1. **25 of 27
  (event × persona) Δ-cells preserve sign across v1 → v3** — the
  per-persona asymmetry findings (aggressive reads courtesy
  negatively; anxious reads scary events as relief) are robust
  to the prompt revision. Magnitudes are tightened in places where
  v1 was over-extreme (e.g. aggressive on red light: −0.25 → −0.06).

**Methodological takeaway:** the LLM's measured affect is sensitive
to *how the model is invited into the persona*, not just the
persona card content. "Behavioural research subject" and "creative-
writing character" produce measurably different Δ tables for the
same persona. We document this as part of the calibration recipe
because it's the kind of thing that would be invisible without
multiple runs and side-by-side comparison.

## Calibration findings (Phase B)

Probe extraction on Qwen3.5-9B-MLX-4bit yielded three actionable
constants for the closed loop:

- **Layer 15** is the cleanest steering layer for all three V/A/D
  axes — late enough to encode high-level concept directions, early
  enough that the model's reasoning template hasn't asserted
  itself. Layers ≥23 absorbed all steering into a fixed
  *"Thinking Process: 1. **Analyze the Request:**..."* preamble
  regardless of α.
- **α ≈ ±25** is the steering sweet spot. Below ±15 the affective
  shift is hard to read; above ±50 outputs hit a *coherence cliff* —
  collapse into n-gram repetition or syntax breaks. The portfolio-
  relevant range is ±25; the symbolic state's `[-1, +1]` axes are
  scaled by `g=25.0` into this range.
- **`enable_thinking=False`** is a hard requirement. With Qwen3's
  default thinking-mode chat template, ~all outputs collapse into
  meta-commentary that drowns the affective shift. Disabling it
  via `apply_chat_template(..., enable_thinking=False)` produces
  clean first-person continuations the steering can act on.

Probe diagnostics:

| axis | cosine consistency across 12 pairs | ‖mean diff‖ at L15 |
|---|---|---|
| V (valence)   | +0.377 | 10.6 |
| A (arousal)   | +0.246 |  9.8 |
| D (dominance) | +0.183 | 10.4 |

V is cleanest, D weakest — the dominance contrastive pairs
conflated multiple sub-concepts (control / agency / will), giving
a noisier axis. All three nonetheless produce qualitatively
distinct outputs at α=±25 (validated end-to-end in
`artifacts/probes_validation.md`).

## What's Novel

Three claims, in increasing order of strength:

1. **The same probe directions are used for steering and for
   reading.** Standard activation-steering work writes one direction;
   standard probing reads another. Reusing one set of vectors for
   both means "intended" and "expressed" state live in the same
   coordinate system. The difference is itself a meaningful signal.

2. **Symbolic state is the source of truth, not LLM
   self-consistency.** Existing affective LLM agents either
   re-inject state in the prompt each turn (drift, context bloat)
   or fine-tune for one persona (fragile, expensive). Here, six
   floats outside the model are authoritative; the LLM is a
   controllable expressive head.

3. **Closed-loop re-appraisal keeps state and expression coupled
   without RL or fine-tuning.** The persona-modulated dynamics
   determine *what should be felt*; the projection readout
   determines *what was expressed*; their difference updates state.
   No reward model, no preference data. Empirically the loop is
   stable: re-appraisal Δ-magnitudes stay under |0.2| per turn
   over the demo scenario, with state clipped to `[-1, +1]` per
   axis providing a hard contractive bound.

## Empirical anchoring: arousal-gate ablation (Shangguan et al. 2025)

The hand-authored modulators in `lib/appraisal.py` were initially
picked from psych-literature primitives "by feel". Shangguan et al.
2025 (*Frontiers in Psychology* 16:1487493) provides a load-bearing
empirical hook: in their moderated-mediation analysis, the
personality→behavior pathway is statistically significant **only
under high arousal** (their Table 2: indirect-effect β=0.002 on
ACC95 and β=0.0017 on SpeedSD with high-arousal CIs strictly above
zero, vs. CIs straddling zero in the low-arousal condition).
Their high-arousal threshold of SAM 5.3/9 maps to our axis as
`state.arousal > +0.18`.

The implementation is one block inside `appraise()`. When the gate
is on and `state.arousal > +0.18`, each persona's deviation from
the across-persona mean response (taken from the same empirical
table) is scaled by 1.5; the mean response itself is preserved.
That mirrors Shangguan's claim — when calm, who you are matters
less; when aroused, your personality dominates. The threshold is
empirically anchored; the 1.5× amp is hand-picked.

### Ablation result (`scripts/09_arousal_gate_ablation.py`)

`mixed_emotions` × 3 personas × {gate off, gate on}, with the LLM
agent in the closed loop. Metric: mean pairwise L2 distance across
personas on (V, A, D), bucketed by whether each tick is above the
empirical threshold under the gate-off baseline.

| regime | gate off | gate on | gain |
|---|---:|---:|---:|
| A ≤ +0.18 (calm)   | 0.298 | 0.326 | **+0.028** |
| A >  +0.18 (aroused) | 0.377 | 0.418 | **+0.042** |
| overall            | 0.306 | 0.335 | +0.029 |

The above-threshold gain is **1.5× the below-threshold gain** —
directionally consistent with Shangguan's prediction that arousal
moderates the personality→behavior link. The persona-distinctness
figure
(`artifacts/arousal_gate_distinctness_mixed_emotions.png`) shows
the gap widening visibly inside the two shaded high-arousal regions
(post-cut-off, post-passenger-comment).

`gain_below` is not zero — and shouldn't be expected to be in a
closed-loop system. A pure-symbolic open-loop ablation would have
`gain_below = 0` by construction (the gate is a no-op below
threshold). With the LLM in the loop, a high-arousal event amplifies
persona-deviation while aroused, then leaves residual state the LLM
perceives during the subsequent calm decay — so the amplification
carries forward. This is closed-loop coupling, not a flaw in the
gate. The empirical signature we'd expect — and what we observe — is
the *interaction*: gain rises when arousal rises.

Per-persona drift (mean V/A/D L2 between gate-off and gate-on)
correlates with baseline arousal: `aggressive_late` and
`anxious_new_driver` (both elevated baseline_va arousal) drift ~0.04;
`calm_commuter` drifts 0.026. The gate fires more often for
personas whose homeostatic state sits closer to the threshold —
structurally consistent with Shangguan's high-trait-arousal
amplification finding.

What this transforms in the writeup: a single hand-picked modulator
becomes a parameter with one empirically-anchored part (the +0.18
threshold from SAM 5.3/9) and one acknowledged hand-tune (the 1.5×
amp). The architecture's central claim — that personality-modulated
appraisal of driving events shapes affective state and downstream
decisions — now carries a behavioural signature consistent with
the moderation literature, not just a citation.

## A productive negative result: LLM-prior decay calibration fails

Same methodology as the appraisal calibration was applied to the
**decay time constants** (`scripts/08_calibrate_decay.py`): for each
event × persona × elapsed-time delay ∈ {0, 5, 15, 30, 60}s, prompt the
LLM with *"About X seconds ago: <event>; you've kept driving since"*,
read felt V/A/D via the same probe projection, and fit an exponential
decay per axis.

**The fit produced R² < 0 across all three axes** (V: −0.275, A:
−0.284, D: −0.127) — the exponential decay model fits *worse than
predicting the mean*. Per-group inspection shows why: the LLM's
representation of *"X seconds ago"* is non-monotonic. Several
persona/event combinations show deviation from baseline *growing*
with elapsed-time framing (calm_commuter ruminating on a cut-off,
aggressive_late on a courtesy gesture). The model depicts deepening
reflection, not fading response — a different cognitive-recall
semantics than physiological decay.

**This is methodologically informative.** It says the LLM-prior
calibration approach works for *immediate-event affective response*
(the appraisal Δs) but not for *post-event temporal regulation* (the
decay τ). We adopt empirical appraisal, retain hand-authored τ, and
recommend biometric calibration (SHRP2 / UDRIVE HRV traces) as the
principled answer for τ. `lib/state.py` enforces an R² ≥ 0.3 floor
on adopted τ, so this finding is encoded in code: the decay file is
present, but the loader correctly rejects it and falls back to
hand-authored constants.

## Limitations (stated honestly)

- **Probes are extracted from contrastive prompts**, not calibrated
  on labeled emotional speech. Probe quality is sufficient for a
  working demo but is an obvious avenue for improvement.
- **Appraisal rules are hand-authored** from psych-literature
  primitives (Lazarus, Mehrabian PAD), not learned from naturalistic
  driving data (SHRP2, UDRIVE).
- **The agent is not validated against human driver behaviour.**
  This is a *generative* model of plausible-looking diversity, not a
  *predictive* model of any specific human. Alignment to behavioural
  data is the obvious follow-up.
- **Persona-card voice can lose to steering at extreme states.**
  When a persona's reference-line voice and the steering direction
  conflict on emotional surface form (e.g. an `aggressive_late`
  driver at very high arousal), the steering vocabulary's
  panic/anxiety register can override the persona's clipped-
  imperative voice. Fixable with stronger high-state reference
  lines or per-persona steering gain modulation.

## What's Next

If extended for a real AV simulation deployment:

- Replace contrastive probes with calibrated probes from a paired
  audio→text emotional speech dataset.
- Replace hand-authored appraisal weights with weights fit to
  naturalistic driving traces (event severity → physiological
  response → behaviour change).
- Co-train the persona vector with a population model so persona
  is *sampled* from a distribution rather than authored, supporting
  scenario generation at scale.
- Plug the decision-coupling layer (`gap_threshold`,
  `following_distance`, `speed_overage`) into a real driving sim
  (highway-env, CARLA) so the closed loop drives both narration
  *and* observed behaviour.

## Why this is the right pitch

The core artifact — closed-loop V/A/D-conditioned activation
steering — is application-agnostic. The driver demo is one
downstream coupling. The same V/A/D state representation also
drives a separate audio→blendshape diffusion model in a parallel
project, which shows the architecture transfers cleanly across
modalities. The most concrete near-term application is
**behavioural diversity for AV scenario generation**: a tunable knob
(the persona vector) for stress-testing planners against
statistically rare but causally coherent NPC behaviour.

## Reproducing

```bash
cd tools/driver_affect_sim
pip install -e .[mlx,viz]                          # mlx-lm + matplotlib

python scripts/02_extract_probes.py                # ~2 min on M-series MPS
python scripts/03_validate_probes.py               # ~3 min, writes probes_validation.md
# Manually inspect probes_validation.md, then:
python -c "from lib.probes import ProbeBundle; b = ProbeBundle.load('artifacts/probes_Qwen3.5-9B-MLX-4bit.pkl'); b.selected_layers = {'V': 15, 'A': 15, 'D': 15}; b.save('artifacts/probes_Qwen3.5-9B-MLX-4bit.pkl')"

python scripts/06_persona_sweep.py                 # ~3 min, writes the figure

# Optional ablations:
python scripts/07_calibrate_appraisal.py           # ~17 min, empirical Δ table
python scripts/08_calibrate_decay.py               # ~12 min, decay τ (R²<0 negative result)
python scripts/09_arousal_gate_ablation.py         # ~8 min, Shangguan 2025 arousal-gate result
```
