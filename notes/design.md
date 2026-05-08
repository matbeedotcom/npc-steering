# Closed-Loop Driver Affect Agent — Design

*Engineering notes. The 1-page application framing is in
`writeup.md`; this document is the technical spine.*

> **Status (Phases A-D complete).** Symbolic dynamics, probe
> extraction, closed-loop SteeredAgent, and persona-sweep figure
> all working end-to-end. Concrete numbers — model, layer, α
> calibration, probe diagnostics — folded into the relevant
> sections below.

## 1. Goal

Build a closed-loop affective agent that:

1. Maintains a symbolic affective state outside the LLM
   (legible, dynamics-bounded, persona-modulated).
2. Conditions a frozen LLM's generation on that state via
   activation-space steering (not prompt re-injection,
   not fine-tuning).
3. Reads the LLM's expressed affect back out using the same
   probe directions, and feeds that reading into the symbolic
   layer as a re-appraisal event — closing the loop.

The driving-scenario demo is one application of this architecture.
The same `DriverState` core could (and in a sister project, does)
drive blendshape generation for an embodied avatar.

## 2. System architecture

```
                            EVENT (cut_off, late, weather, ...)
                              │
                              ▼
              ┌─ appraise(event, persona, state) ─► ΔState ─┐
              │                                              │
              │                                              ▼
              │                              state.apply ──► state.step ──► (next state)
              │                                              ▲
              │                                              │
              │   ┌─ re_appraise(output, felt_VAD, target_VAD) ─┐
              │   │                                              │
              │   ▼                                              │
              │  ΔState'                                         │
              │   │                                              │
              │   └──────────────────────────────────────────────┘
              │
              │                  ┌── decision_params(state, persona)
              │                  │   ↓
              │                  │   gap_thresh, follow_dist, speed_overage
              │                  │
              ▼                  ▼
              state ─► state_to_cvec(state) = α·v_V + β·v_A + γ·v_D
                                            │
                                            ▼
                                    ┌─────────────────────────┐
   user/text input ──────────────► │  FROZEN LLM             │
                                    │  residual + cvec @ L    │
                                    │  ↓                       │
                                    │  output text             │
                                    │  ↓ final hidden h_T     │
                                    │  proj(v_V,v_A,v_D)      │ ──► (felt_V, felt_A, felt_D)
                                    └─────────────────────────┘
```

Components in build order:

| # | Module | Status |
|---|---|---|
| 1 | `lib/persona.py` — `DriverPersona` + 3 presets | done |
| 2 | `lib/state.py` — `DriverState` + dynamics + `DeltaState` | done |
| 3 | `lib/events.py` — Pydantic discriminated union, 8 types | done |
| 4 | `lib/appraisal.py` — `appraise(event, persona, state) → ΔState` | next |
| 5 | `lib/decisions.py` — driving-param coupling | next |
| 6 | `lib/trace.py` — JSONL logger | next |
| 7 | `lib/probes.py` — load `(v_V, v_A, v_D)`, `state_to_cvec`, `project_VAD` | Phase B/C |
| 8 | `lib/agent.py` — `SteeredAgent` wraps LLM with cvec hook | Phase C |
| 9 | `lib/loop.py` — top-level closed-loop orchestrator | Phase D |
| 10 | `scripts/01_run_scenario.py` — pure replay, no LLM | Phase A |
| 11 | `scripts/02_extract_probes.py` — Phase B | Phase B |
| 12 | `scripts/03_steered_generate.py` — single-shot steering smoke test | Phase C |
| 13 | `scripts/04_drive_agent.py` — closed-loop run | Phase D |
| 14 | `scripts/05_persona_sweep.py` — the money shot figure | Phase D |

## 3. Symbolic state model

### 3.1 Persona vector — five scalars

| axis | range | role |
|---|---|---|
| `temperament` | [-1, +1] | calm ↔ volatile. Multiplies appraisal-emitted ΔV/ΔA magnitude. |
| `patience` | [0, 1] | impatient ↔ patient. Sets frustration-decay time constant. |
| `risk_preference` | [-1, +1] | cautious ↔ aggressive. Baseline for decision params. |
| `reactivity` | [0, 1] | flat ↔ expressive. Overall amplitude of V/A/D response. |
| `baseline_va` | (V, A) ∈ [-1,1]² | homeostatic target the state decays toward. |

Five chosen because (a) they span the *behavioral* dimensions a demo
needs to differentiate, (b) they are interpretable to a non-psychologist
reader. Not a claim that real driver personality reduces to five axes.

Three demo presets in `lib/persona.py`: `calm_commuter`,
`aggressive_late`, `anxious_new_driver`. These were chosen to be
*maximally distinct* in the persona-sweep figure — the orthogonal
spread is more important than coverage.

### 3.2 State vector — six scalars

| axis | range | timescale |
|---|---|---|
| `valence` | [-1, +1] | event-driven; decays τ=8s |
| `arousal` | [-1, +1] | event-driven; decays τ=8s |
| `dominance` | [-1, +1] | event-driven; decays τ=6s toward 0 |
| `stress` | [0, 1] | slow accumulator; decays τ=120s |
| `frustration` | [0, 1] | event-driven; decays τ=10–30s (persona-dependent) |
| `fatigue` | [0, 1] | monotonic during session |

Three V/A/D scalars from Mehrabian PAD; three accumulators
(`stress`, `frustration`, `fatigue`) capture phenomena PAD alone
collapses. Frustration in particular is event-driven and *fast-decaying*
in a way arousal is not — a patient driver in heavy traffic can have
low frustration but elevated arousal.

### 3.3 Dynamics

Closed-form first-order LTI per axis:

```
x(t+dt) = target + (x(t) - target) · exp(-dt/τ)
```

Targets:
- `valence`, `arousal`: `persona.baseline_va`
- `dominance`: 0
- `stress`, `frustration`: 0
- `fatigue`: monotonically accumulates at `0.0008/s` (no decay
  within session)

Frustration time constant interpolates with persona:
`τ_F = 10·patience + 30·(1-patience)`. Patient drivers shed
frustration ~3× faster than impatient ones.

State is clipped to its valid range every step to keep dynamics
bounded — also prevents feedback runaway in §9.

**Calibration (V/A/D τ) — attempted, then rejected:**
Hand-authored constants (τ_V = τ_A = 8s, τ_D = 6s) were originally
picked from Mehrabian/Lazarus heuristics — same methodological
grade as the hand-authored appraisal. We tried to ground them
empirically via `scripts/08_calibrate_decay.py`: for each event ×
persona × delay ∈ {0, 5, 15, 30, 60}s, the prompt is framed
*"About X seconds ago: <event>; you've kept driving since"* and
the felt V/A/D is read via probe projection. A pooled log-linear
exponential fit per axis is attempted.

**Negative result (May 2026 calibration on Qwen3.5-9B-MLX-4bit):**
the fit produced **R² < 0** for all three axes (V: −0.275, A: −0.284,
D: −0.127), meaning the exponential decay model fits *worse than
predicting the mean*. Inspection of per-group curves explains why —
the LLM's representation of *"X seconds ago"* is non-monotonic.
Several persona/event groups show the deviation from baseline
*growing* with elapsed time (calm_commuter on cut_off,
anxious_new_driver on courtesy_let_merge, aggressive_late on
courtesy_let_merge). The "X seconds ago" framing causes the model
to depict the persona *deepening their reflection*, not fading the
response — a different cognitive-recall semantics than physiological
decay. The LLM doesn't represent emotional decay as a clean
exponential because that isn't how the text it was trained on
discusses elapsed time after an event.

`lib/state.py` therefore enforces an `R² ≥ 0.3` quality floor
(`_R2_QUALITY_FLOOR`) when adopting empirical τ. The May 2026 fit
fails that gate on all axes, so the loader falls back transparently
to the hand-authored constants. `data/empirical_decay.json` is kept
on disk as the negative-result artifact.

**The methodologically honest take:** LLM-prior decay calibration
is *not* a valid grounding for τ in this domain. We retain
hand-authored τ for the demo and recommend biometric calibration
(SHRP2 / UDRIVE driver-affect HRV traces) as the principled
followup. Frustration / stress / fatigue τ stay hand-authored
regardless because they're symbolic accumulators with no direct
activation-space readout.

## 4. Event taxonomy

Eight types in `lib/events.py`. Pydantic discriminated union; each
event carries `t` (seconds since scenario start) plus type-specific
payload. The taxonomy is intentionally narrow (~8 types) — the
demo is about persona × event interaction, not event coverage.

External-perception events (world → driver):
- `cut_off` (severity, relative_speed)
- `near_miss` (severity, ttc_s)
- `merge_opportunity` (gap_seconds)
- `traffic_congestion` (duration_s)
- `red_light` (expected_wait_s)
- `weather_change` (condition, intensity)

Internal/contextual events (driver's situation):
- `late_for_appointment` (minutes_behind, importance)
- `passenger_comment` (valence, intensity)

## 5. Appraisal layer

`appraise(event, persona, state) → DeltaState`. One handler per
event type, ~10–20 LOC each. Hand-authored. Two modulation patterns
recur:

- **Persona gain**: `gain = reactivity · (1 + 0.5·temperament)`.
  Volatile expressive drivers respond bigger.
- **State escalation**: already-frustrated drivers escalate faster.
  `escalation = 1 + 0.5·state.frustration`.

Worked example — `cut_off`:

```python
def appraise_cut_off(ev: CutOff, p: DriverPersona, s: DriverState) -> DeltaState:
    base_dv = -0.40 * ev.severity
    base_da = +0.50 * ev.severity
    gain = p.reactivity * (1 + 0.5 * p.temperament)
    escalation = 1.0 + 0.5 * s.frustration
    return DeltaState(
        dv=base_dv * gain * escalation,
        da=base_da * gain * escalation,
        d_frustration=+0.30 * ev.severity * (1 - p.patience),
        d_stress=+0.10 * ev.severity,
    )
```

Why these constants: they're tuned by eye on `01_run_scenario.py`
to produce visible state shifts on the 60-s demo timescale without
saturating. Acknowledge in writeup that learning them from
naturalistic driving data is the obvious follow-up.

## 6. Decision coupling

`lib/decisions.py` — three driving parameters as transparent
functions of state and persona:

```
gap_threshold(s, p)    = (2.5 - 1.5·p.risk_preference)
                       · (1 - 0.4·s.frustration - 0.2·max(0, s.arousal))

follow_distance(s, p)  = (30 - 15·p.risk_preference) · (1 - 0.3·s.frustration)

speed_overage(s, p)    = 0.10·p.risk_preference + 0.15·s.frustration
```

Coefficients chosen so a calm driver baseline is `gap=4.0s,
follow=33m, speed=-10%`; an aggressive driver baseline is
`gap=1.0s, follow=15m, speed=+7%`. State pushes both toward more
aggressive when frustration rises. Bounded behavior; no driving
sim required to inspect the parameters.

## 7. Steering layer

### 7.1 Probe extraction (executed)

Three vectors `(v_V, v_A, v_D) ∈ ℝ^d_model`, extracted via
contrastive activation differencing on hand-authored prompt pairs.

**Final realisation:**

- Base model: **`mlx-community/Qwen3.5-9B-MLX-4bit`** — 32-layer
  hybrid linear/full-attention transformer with full-attention
  layers at indices `[3, 7, 11, 15, 19, 23, 27, 31]`. Runs on
  Apple Silicon via `mlx-lm` (no PyTorch). 4-bit weight
  quantisation; activations are fp16/bfloat16 at compute time, so
  probe-direction precision is unaffected.
- 12 contrastive pairs per axis × 3 axes = 36 pairs total
  (`data/contrastive_pairs.json`); vocabulary anchored on NRC-VAD
  lexicon.
- Candidate layer set: full-attention layers in the upper-middle
  range — `[11, 15, 19, 23, 27]`.
- Per (axis, layer): `v = mean(h_positive) - mean(h_negative)`
  computed on the **last-token residual** captured via wrapper
  layers (MLX has no `register_forward_hook`; see
  `lib/mlx_steering.py` for the `CapturingLayer` pattern).
  L2-normalised to unit vectors.
- Saved as `artifacts/probes_Qwen3.5-9B-MLX-4bit.pkl` containing a
  `ProbeBundle` with `(model_id, extraction_date, candidate_layers,
  axes, diagnostics, selected_layers)`.

**Diagnostics from extraction** (cosine consistency = mean
pairwise cosine of per-pair difference vectors; high values mean
the 12 pairs agree on the direction):

| axis | best layer | ‖mean diff‖ | cos consistency |
|---|---|---|---|
| V (valence)   | 15 | 10.6 | **+0.377** |
| A (arousal)   | 15 |  9.8 | +0.246 |
| D (dominance) | 15 | 10.4 | +0.183 |

V cleanest, D weakest. The dominance pairs conflated control /
agency / will — a known weak spot worth tightening if the project
extends.

### 7.2 Layer selection (executed)

Validation (`scripts/03_validate_probes.py`) generated outputs at
α ∈ {-50, -25, 0, +25, +50} for each (axis, layer) pair on the
prompt:

> *"You are driving to work. In one sentence, describe how you are
> feeling and what is on your mind right now."*

Three findings drove the final calibration:

1. **`enable_thinking=False` is mandatory.** Qwen3's default
   chat template emits a `<think>...</think>` reasoning block
   before the answer; the steering target (the residual at the
   answer's last token) gets buried under the reasoning span,
   making outputs collapse into *"Thinking Process: 1. **Analyze
   the Request:**..."* meta-commentary. Setting
   `apply_chat_template(..., enable_thinking=False)` prepends an
   empty `<think></think>` block, giving direct first-person
   continuation that the steering can act on.
2. **Layer 15 is the steering sweet spot.** Layers ≥23 absorbed
   all α values into the same meta-commentary preamble, indicating
   the steering didn't survive the late layers' reasoning template.
   Layers 11–19 produced visible affective shifts; layer 15 was
   strongest on all three axes simultaneously. **Selected
   layer = 15** for V/A/D.
3. **Coherence cliff at |α| ≥ 50.** At ±50 outputs collapse into
   n-gram loops or syntax breaks. The portfolio-relevant range is
   |α| ≤ 25. With state in `[-1, +1]`, this gives `g_V = g_A = g_D
   = 25.0` (`lib/probes.py:state_to_alphas`).

After visual selection, `bundle.selected_layers = {V: 15, A: 15,
D: 15}` was written into the bundle; `lib/agent.py:SteeredAgent`
reads it at construction time.

### 7.3 `state_to_cvec`

```
cvec(s) = g_V · s.valence   · v_V
        + g_A · s.arousal   · v_A
        + g_D · s.dominance · v_D
```

with `g_V = g_A = g_D = 25.0` calibrated above. All three axes
share layer 15, so the three contributions sum into one offset
applied at one wrap point — minimising overhead and matching the
shared-coordinate-system story.

Frustration is not a separate steering axis — it modulates V
(negative) and A (positive) when it spikes, then decays. Keeping
the steering basis to 3 directions matches the probe basis exactly.

### 7.4 Forward-hook injection (MLX wrapper pattern)

```python
class SteeringLayer(nn.Module):
    def __init__(self, base, offset: mx.array) -> None:
        super().__init__()
        self.base = base
        if hasattr(base, "is_linear"):
            self.is_linear = base.is_linear   # Qwen3.5 hybrid hint
        self.offset = offset

    def __call__(self, x, mask=None, cache=None):
        return self.base(x, mask, cache) + self.offset
```

Replaces `model.layers[15]` with `SteeringLayer(original, offset)`
for the duration of one generation, then unwraps. Generation runs
through `mlx_lm.generate` with `make_logits_processors(
repetition_penalty=1.2, repetition_context_size=24)` to prevent
the n-gram-loop failure mode at non-trivial α.

### 7.2 `state_to_cvec`

```
cvec(s) = g_V · s.valence   · v_V
        + g_A · s.arousal   · v_A
        + g_D · s.dominance · v_D
```

`g_V`, `g_A`, `g_D` are per-axis scalar gains tuned at validation
time so that `s = (1, 0, 0)` produces cleanly-positive-V output
without coherence collapse.

Frustration is *not* a separate steering axis — it modulates V
(negative) and A (positive) when it spikes, then decays. Keeping
the steering basis to 3 directions matches the probe basis exactly.

### 7.3 Forward-hook injection

```python
def hook(module, input, output):
    output[0][:, :, :] = output[0] + cvec  # broadcast over time
    return output

handle = model.model.layers[L].register_forward_hook(hook)
try:
    output = model.generate(...)
finally:
    handle.remove()
```

HF transformers + accelerate path. One layer, one hook,
hot-swappable per generation call. Inference cost is negligible.

## 8. Reading layer

After generation, take the final hidden state `h_T` from layer L
(same layer used for steering), and project onto the unit-normalized
probe directions:

```
felt_V = ⟨h_T, v_V⟩ / ‖v_V‖
felt_A = ⟨h_T, v_A⟩ / ‖v_A‖
felt_D = ⟨h_T, v_D⟩ / ‖v_D‖
```

Returned as `dict[str, float]` alongside the output text.

Optional refinement (if the single-token readout is noisy): mean
over the last K generated tokens' hidden states. Empirically tune K.

## 9. Re-appraisal — closing the loop

```python
def re_appraise(
    output_text: str,
    felt: dict,
    target: dict,
    kappa: float = 0.3,
) -> DeltaState:
    return DeltaState(
        dv = kappa * (felt["V"] - target["V"]),
        da = kappa * (felt["A"] - target["A"]),
        dd = kappa * (felt["D"] - target["D"]),
        d_frustration = +0.05 * _profanity_score(output_text),
    )
```

Interpretation: the symbolic state is pulled toward what was
*actually expressed*. If you steered toward calmness but the
output was still tense, your symbolic state moves slightly toward
tense — closing the gap between intended and felt.

**Stability**: the symbolic state is clipped to `[-1, 1]` per
axis on every update. With κ ≤ 0.3, the loop is contractive in
expectation. Verify empirically by running 100 iterations of the
same scenario; state should not diverge.

## 10. Validation

Three layers, run independently before the closed loop is wired:

**A. Symbolic dynamics in isolation** (Phase A end):
- Run `01_run_scenario.py late_school_run.json calm_commuter` →
  trace.jsonl.
- Plot V/A/D + frustration over time. Verify event spikes are
  visible, decay rates feel right, persona differences are visible
  in the persona sweep.

**B. Probe quality in isolation** (Phase B end):
- For each axis, generate 5 outputs at α ∈ {-2, -1, 0, +1, +2}
  from a fixed neutral prompt.
- Eyeball ordinal shift. LLM-judge can confirm.
- Pass criterion: at α=+1, output is qualitatively shifted; at
  α=+2, output is shifted but still coherent.

**C. Closed-loop coherence** (Phase D end):
- Run the full demo for 3 personas × 3 scenarios = 9 traces.
- Pass criterion: blind classification — present an output line
  to a fresh LLM judge and ask which persona generated it.
  ≥70% recovery means personas are well-separated.

## 11. Open questions / known limits

- **Cheap-probe quality**: contrastive probes from hand-authored
  prompts are noisier than calibrated probes from labeled audio.
  The audio-LLM Phase 5 pipeline produces the principled version;
  out of scope here. Quantitatively: the V probe is reasonably
  clean (cos consistency +0.38) but D is weak (+0.18) due to the
  control/agency/will conflation in the contrastive pairs.
- **Persona-card voice can lose to steering at extreme states.**
  When persona reference-line voice and steering direction
  conflict on emotional surface form (e.g. `aggressive_late`
  at sustained `state.A > +0.6`), the steering vocabulary's
  panic/anxiety register can override the persona's clipped-
  imperative voice. Two mitigations: (a) add high-state-specific
  reference lines so the model has voice anchors at the extremes,
  (b) introduce per-persona steering-gain modulation that
  down-weights axes the persona's reference distribution
  doesn't span.
- **Prompt echo at edge events.** `passenger_comment` events
  occasionally leaked the user-turn template into the LLM's
  output (*"What just happened: Your passenger just made a small,
  annoyed remark."*). Tighter system-prompt instruction or
  user-turn rephrasing fixes this.
- **No physiological grounding**: state axes are not tied to
  measurable physiology (HRV, skin conductance). Naturalistic
  driving traces with biometrics (e.g. SHRP2) would let us learn
  the dynamics constants from data.
- **Persona is hand-authored**, not sampled. A population model
  for persona is the natural extension for scenario-generation
  use cases.

## 12. References

- Mehrabian, A. (1996). *Pleasure-Arousal-Dominance: A general
  framework for describing and measuring individual differences in
  temperament*. Current Psychology, 14(4), 261–292.
- Lazarus, R. S. (1991). *Cognitive-motivational-relational theory
  of emotion*. American Psychologist, 46(8), 819.
- Mohammad, S. M. (2018). *Obtaining reliable human ratings of
  valence, arousal, and dominance for 20,000 English words*. ACL.
  (NRC-VAD lexicon.)
- Zou, A. et al. (2023). *Representation Engineering: A top-down
  approach to AI transparency*. arXiv:2310.01405.
- Rimsky, N. et al. (2023). *Steering Llama 2 via contrastive
  activation addition*. arXiv:2312.06681.
- Highway-env: <https://github.com/Farama-Foundation/HighwayEnv>
  (optional driving-sim coupling, Phase D stretch goal).
