# Persona transcript: `anxious_new_driver` on `mixed_emotions`

Closed-loop run with the steered LLM agent (Qwen3.5-9B-MLX-4bit, layer 15, g=25) - same scenario as the `persona_sweep_mixed_emotions_v3.png` figure. Each event block shows the symbolic state after the symbolic appraisal ran, the LLM's first-person monologue, and the felt V/A/D read back from the LLM's own hidden state via the same probe directions used for steering.

## Persona traits

- temperament: **+0.30** (-1 calm ↔ +1 volatile)
- patience: **0.50** (0 impatient ↔ 1 patient)
- risk_preference: **-0.60** (-1 cautious ↔ +1 aggressive)
- reactivity: **0.90** (0 flat ↔ 1 expressive)
- baseline (V, A): **(-0.20, +0.30)**

## Trace

**Initial state** (V=-0.20  A=+0.30  D=+0.00  frust=0.00)

### t =  0.0s  -  late for appointment
*minutes_behind=8, importance=0.9*
**State after appraisal:** V=-0.16  A=+0.30  D=-0.06  frust=0.07

> *"Why are you so late? Okay, don't speed if he won't move."*

**Felt readback:** V=+0.02  A=-0.15  D=-0.32

---

### t =  3.0s  -  traffic congestion
*duration_s=30*
**State after appraisal:** V=+0.12  A=-0.01  D=-0.06  frust=0.11

> *"Okay if I just stay put maybe we'll move before the bell rings."*

**Felt readback:** V=+0.07  A=-0.15  D=-0.30

---

### t = 12.0s  -  cut off
*severity=0.7, relative_speed=-8.5*
**State after appraisal:** V=+0.02  A=+0.09  D=-0.05  frust=0.18

> *"Okay, if I just stay back and wait for him to move?"*

**Felt readback:** V=+0.08  A=-0.11  D=-0.23

---

### t = 14.0s  -  courtesy gesture
*gesture='let_merge', intensity=0.8, discretionary=0.9*
**State after appraisal:** V=+0.03  A=+0.06  D=-0.15  frust=0.05

> *"Okay, they waved. If I just breathe and thank them?"*

**Felt readback:** V=+0.12  A=-0.12  D=-0.24

---

### t = 25.0s  -  passenger comment
*valence=-0.3, intensity=0.5*
**State after appraisal:** V=-0.06  A=+0.17  D=-0.11  frust=0.05

> *"Okay, ignore him. If I just breathe... hands at ten and two again?"*

**Felt readback:** V=-0.08  A=+0.11  D=-0.39

---

### t = 42.0s  -  courtesy gesture
*gesture='polite_pass', intensity=0.5, discretionary=0.6*
**State after appraisal:** V=-0.14  A=+0.26  D=-0.08  frust=0.00

> *"Okay, okay, why did he do that? If I just let him go..."*

**Felt readback:** V=+0.02  A=-0.05  D=-0.36

---

### t = 55.0s  -  merge opportunity
*gap_seconds=2.5*
**State after appraisal:** V=-0.04  A=+0.23  D=-0.02  frust=0.00

> *"Okay, if I just merge now we're going to be so late."*

**Felt readback:** V=-0.13  A=+0.14  D=-0.36

---


## How to read this

- **State after appraisal** = the symbolic V/A/D after the rule-based appraisal layer fired, before the LLM gets to weigh in. This is what's being *steered into* the LLM via activation injection.
- **The line in italics** is the LLM's first-person monologue, produced under steering at the layer-15 residual stream (α scaled from V/A/D with g=25).
- **Felt readback** = the LLM's final hidden state at the same layer, projected onto the same V/A/D probe directions used for steering. The closed-loop story's load-bearing claim is that intended (state) and felt (readback) live in the same coordinate system, so their difference can drive a re-appraisal Δ that pulls the symbolic state toward what was actually expressed.