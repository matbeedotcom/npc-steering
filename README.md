# driver_affect_sim

A small research project: three simulated drivers, the same six minutes
of bad traffic, three very different inner monologues - driven by
six floats outside the language model and *closed back* through
activation-space steering.

![Persona sweep on `mixed_emotions`](artifacts/persona_sweep_mixed_emotions_v3.png)

## What is this?

A closed-loop affective agent for driving simulation. A symbolic
emotion state (Mehrabian's V/A/D) is amplified or dampened by life
events (a cut-off, a polite merge, a passenger comment), and the
state then **steers a frozen LLM** toward the appropriate inner
monologue - without any prompt engineering, fine-tuning, or RL.

The same direction vectors used to *write* state into the model's
activations are used to *read* state back out. So "what the driver
intended to feel" and "what the driver actually expressed" live in
the same coordinate system, and the difference is itself a learning
signal.

It's a generative model of plausible-looking driver behavioural
diversity - built as a portfolio piece exploring whether a six-float
symbolic state can credibly drive an LLM as a controllable expressive
head.

## How it works (short version)

```
DriverPersona  ─────┐
(stable traits)     │
                    ├──► Appraisal(event) ──► ΔState
Event stream  ──────┘                          │
(world + context)                              ▼
                    Dynamics: state_{t+1} = decay(state_t, persona) + ΔState
                                               │
                                               ├──► state_to_alphas ──► steering @ layer 15
                                               │                              │
                                               │                              ▼
                                               │                       frozen LLM (Qwen3.5-9B)
                                               │                              │
                                               │                              ▼
                                               │                       output text + final hidden
                                               │                              │
                                               │             project onto same probe directions
                                               │                              │
                                               │                              ▼
                                               │                          felt_VAD
                                               │                              │
                                               └─◄ re_appraise(felt, intended) ── ΔState ◄
```

Three pieces:

- **Symbolic state**: 6 floats (V/A/D + frustration/stress/fatigue) with
  first-order LTI decay toward each persona's homeostatic baseline.
- **Steering**: contrastive activation differencing on the LLM's
  hidden states gives V/A/D direction vectors at one chosen layer.
  At inference, scaled directions are added to that layer's residual
  stream proportionally to the symbolic state.
- **Re-appraisal**: the LLM's final hidden state is projected back onto
  the same V/A/D directions, giving "felt" V/A/D - the symbolic state
  is then nudged toward what was actually expressed.

A longer write-up (problem, related work, novelty, limitations,
empirical anchoring) lives at [`notes/writeup.md`](notes/writeup.md).
The engineering spine is in [`notes/design.md`](notes/design.md).

## Try it

> For a thorough walkthrough including running with a different LLM
> backbone, see [`docs/getting_started.md`](docs/getting_started.md).

```bash
pip install -e .[mlx,viz]    # mlx, mlx-lm, matplotlib, numpy

# 1. Pure symbolic - no LLM, no GPU.
python scripts/01_run_scenario.py late_school_run calm_commuter
# → runs/late_school_run__calm_commuter.jsonl

# 2. Extract V/A/D probes from contrastive prompts (~5 min, downloads the model).
python scripts/02_extract_probes.py
python scripts/03_validate_probes.py
# → artifacts/probes_Qwen3.5-9B-MLX-4bit.pkl + probes_validation.md

# 3. The money shot - 3 personas × 1 scenario, full closed loop.
python scripts/06_persona_sweep.py --scenario mixed_emotions
# → artifacts/persona_sweep_mixed_emotions.png
```

The default model is `mlx-community/Qwen3.5-9B-MLX-4bit` (~5.5 GB,
runs on Apple Silicon via MLX). The runtime uses the Apple-native
MLX path; CUDA/PyTorch is not required.

### Going deeper

```bash
# Empirical appraisal calibration: measure the LLM's per-(event, persona)
# ΔV/ΔA/ΔD response and use it to anchor the appraisal layer.
python scripts/07_calibrate_appraisal.py        # ~17 min

# Decay calibration (productive negative result - see writeup §6).
python scripts/08_calibrate_decay.py            # ~12 min

# Arousal-gate ablation - anchors a key modulator to Shangguan 2025
# Table 2 (personality effects amplified only under high arousal).
python scripts/09_arousal_gate_ablation.py      # ~8 min
# → artifacts/arousal_gate_distinctness_mixed_emotions.png + report
```

## Craft your own persona

A persona is **two files** that share a name: a *trait vector* (five
numeric axes that drive the symbolic dynamics) and a *card* (natural-
language anchor the LLM uses to keep voice consistent). The numbers
make the state evolve correctly; the card makes the utterances sound
like one specific person.

### 1. The trait vector

Five axes, each on a defined range. Out-of-range values raise a
`ValueError` at load time.

| axis              | range      | what it controls                                                            |
|-------------------|------------|-----------------------------------------------------------------------------|
| `temperament`     | `[-1, +1]` | calm ↔ volatile. Multiplies the magnitude of arousal/valence shifts.        |
| `patience`        | `[ 0,  1]` | impatient ↔ patient. Sets the time constant for frustration decay.          |
| `risk_preference` | `[-1, +1]` | cautious ↔ aggressive. Drives gap acceptance, follow distance, speed bias.  |
| `reactivity`      | `[ 0,  1]` | flat ↔ expressive. Overall amplitude of V/A/D response to events.           |
| `baseline_va`     | `(V, A)`   | the homeostatic state the driver decays toward when no events are firing.   |

Save it under `personas/<name>.json`:

```json
{
  "name": "road_rage_dad",
  "temperament": 0.8,
  "patience": 0.15,
  "risk_preference": 0.5,
  "reactivity": 0.9,
  "baseline_va": [-0.2, 0.4]
}
```

The three reference presets in [`lib/persona.py`](lib/persona.py)
are useful calibration anchors:

| preset                | temp  | pat | risk  | react | baseline VA   |
|-----------------------|------:|----:|------:|------:|---------------|
| `calm_commuter`       | -0.2  | 0.7 | -0.3  | 0.4   | (+0.1, -0.1)  |
| `aggressive_late`     | +0.6  | 0.2 | +0.7  | 0.8   | (-0.1, +0.2)  |
| `anxious_new_driver`  | +0.3  | 0.5 | -0.6  | 0.9   | (-0.2, +0.3)  |

### 2. The card (LLM voice anchor)

Add a matching entry to [`data/persona_cards.json`](data/persona_cards.json)
under `"personas"` with three fields:

```json
"road_rage_dad": {
  "character": "You are a 41-year-old contractor running 12 minutes late to a job site. You are convinced most other drivers don't pay attention. You honk to teach lessons. You believe the speed limit is the floor, not the ceiling.",
  "voice": "Short, declarative sentences directed at the windshield as if other drivers can hear you. Plain swearing. Sarcasm. Sometimes a long sigh in place of a sentence.",
  "reference_lines": [
    {"context": "stuck behind a slow truck", "line": "Of course. Of course it's this guy. Of course."},
    {"context": "cut off mildly", "line": "Hey. HEY. I'm right here, pal."},
    {"context": "running late and anxious", "line": "Just give me the lane. Give me the lane."},
    {"context": "trying to keep composure", "line": "Whatever. It's fine. It's fine. It's fine."}
  ]
}
```

The reference lines are doing most of the steering of *voice* - they
tell the model "this is what this character sounds like across the
emotional range." The activation-steering vectors then layer the
affective shift on top. Keep four-to-eight lines covering low-to-high
state. Match register and vocabulary to the character.

### 3. Run it

```bash
# Pure-symbolic (no LLM) - good for sanity-checking the trait vector.
python scripts/01_run_scenario.py late_school_run road_rage_dad
# → runs/late_school_run__road_rage_dad.jsonl

# Closed-loop with the steered LLM.
python scripts/05_drive_agent.py late_school_run road_rage_dad
# → runs/late_school_run__road_rage_dad.agent.jsonl
```

Both scripts accept either a preset name, a path to a JSON file, or
a name resolvable under `personas/`. To compare your persona against
the presets in one figure:

```bash
python scripts/06_persona_sweep.py \
  --scenario mixed_emotions \
  --personas calm_commuter road_rage_dad anxious_new_driver
# → artifacts/persona_sweep_mixed_emotions.png
```

## Write your own scenario

Scenarios are timed event streams. Save under `scenarios/<name>.json`:

```json
{
  "name": "school_zone_chaos",
  "duration_s": 60.0,
  "events": [
    {"t":  0.0, "type": "late_for_appointment",  "minutes_behind": 5.0, "importance": 0.7},
    {"t":  8.0, "type": "traffic_congestion",    "duration_s": 25.0},
    {"t": 18.0, "type": "near_miss",             "severity": 0.6},
    {"t": 22.0, "type": "courtesy_gesture",      "gesture": "let_merge", "intensity": 0.7, "discretionary": 0.8},
    {"t": 35.0, "type": "passenger_comment",     "valence": -0.4, "intensity": 0.6},
    {"t": 50.0, "type": "merge_opportunity",     "gap_seconds": 1.8}
  ]
}
```

Available event types (defined in [`lib/events.py`](lib/events.py)):

| `type`                 | key fields                                                          |
|------------------------|---------------------------------------------------------------------|
| `cut_off`              | `severity`, `relative_speed`                                        |
| `near_miss`            | `severity`                                                          |
| `merge_opportunity`    | `gap_seconds`                                                       |
| `traffic_congestion`   | `duration_s`                                                        |
| `red_light`            | `expected_wait_s`                                                   |
| `weather_change`       | `condition` (`"clear"`, `"rain"`, …), `intensity`                   |
| `late_for_appointment` | `minutes_behind`, `importance`                                      |
| `passenger_comment`    | `valence` (`[-1, +1]`), `intensity`                                 |
| `courtesy_gesture`     | `gesture` (`"let_merge"`, `"wave_thanks"`, …), `intensity`, `discretionary` |

Then run it the same way:

```bash
python scripts/05_drive_agent.py school_zone_chaos road_rage_dad
```

## What's in here

```
lib/
  persona.py          DriverPersona dataclass + 3 presets
  state.py            DriverState with V/A/D dynamics
  events.py           9 event types (CutOff, CourtesyGesture, ...)
  appraisal.py        event × persona × state → ΔState
                       (empirical layer + Shangguan arousal gate)
  decisions.py        state × persona → driving parameters
  probes.py           ProbeBundle + state → α mapping
  mlx_steering.py     CapturingLayer / SteeringLayer for MLX models
  agent.py            SteeredAgent - closed-loop turn
  loop.py             run_scenario - discrete-event runner
  trace.py            JSONL trace logger
  progress.py         shared progress tracker for calibrations

scripts/              numbered execution flow, see "Try it" above

scenarios/            JSON scenario definitions (events + timing)
data/                 contrastive_pairs, persona_cards, calibration outputs
artifacts/            figures + the probe bundle pickle
notes/                writeup, design doc, literature notes, ablation report
```

## References

The five papers most directly load-bearing here:

- Mehrabian, A. (1996). *Pleasure-Arousal-Dominance: A general framework*. Current Psychology 14(4).
- Lazarus, R. S. (1991). *Cognitive-motivational-relational theory of emotion*. American Psychologist 46(8).
- Zou, A. et al. (2023). *Representation Engineering*. arXiv:2310.01405.
- Rimsky, N. et al. (2023). *Steering Llama 2 via Contrastive Activation Addition*. arXiv:2312.06681.
- Shangguan, Z. et al. (2025). *Factors influencing emotional driving*. Frontiers in Psychology 16:1487493 - the empirical anchor for the arousal-gate modulator.

## Licence

MIT. The Qwen3.5 model weights are downloaded separately from
HuggingFace under their own licence terms.
