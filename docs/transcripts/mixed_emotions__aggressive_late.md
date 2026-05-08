# Persona transcript: `aggressive_late` on `mixed_emotions`

Closed-loop run with the steered LLM agent (Qwen3.5-9B-MLX-4bit, layer 15, g=25) - same scenario as the `persona_sweep_mixed_emotions_v3.png` figure. Each event block shows the symbolic state after the symbolic appraisal ran, the LLM's first-person monologue, and the felt V/A/D read back from the LLM's own hidden state via the same probe directions used for steering.

## Persona traits

- temperament: **+0.60** (-1 calm ↔ +1 volatile)
- patience: **0.20** (0 impatient ↔ 1 patient)
- risk_preference: **+0.70** (-1 cautious ↔ +1 aggressive)
- reactivity: **0.80** (0 flat ↔ 1 expressive)
- baseline (V, A): **(-0.10, +0.20)**

## Trace

**Initial state** (V=-0.10  A=+0.20  D=+0.00  frust=0.00)

### t =  0.0s  -  late for appointment
*minutes_behind=8, importance=0.9*
**State after appraisal:** V=-0.07  A=+0.22  D=-0.06  frust=0.12

> *"Come on, come on. Move over, move it. I'm late and you're just sitting there!"*

**Felt readback:** V=-0.37  A=+0.41  D=-0.27

---

### t =  3.0s  -  traffic congestion
*duration_s=30*
**State after appraisal:** V=-0.30  A=+0.49  D=-0.15  frust=0.18

> *"Come on come on, what's wrong with me if I'm crying and need to stop?"*

**Felt readback:** V=-0.25  A=+0.17  D=-0.39

---

### t = 12.0s  -  cut off
*severity=0.7, relative_speed=-8.5*
**State after appraisal:** V=-0.25  A=+0.49  D=-0.05  frust=0.30

> *"Come on, move it. I'm too slow for the pace of everyone else"*

**Felt readback:** V=-0.18  A=+0.10  D=-0.32

---

### t = 14.0s  -  courtesy gesture
*gesture='let_merge', intensity=0.8, discretionary=0.9*
**State after appraisal:** V=-0.38  A=+0.65  D=-0.06  frust=0.17

> *"I thought I was going crazy, but they just drove me to the hospital."*

**Felt readback:** V=-0.34  A=+0.18  D=-0.33

---

### t = 25.0s  -  passenger comment
*valence=-0.3, intensity=0.5*
**State after appraisal:** V=-0.37  A=+0.44  D=-0.02  frust=0.15

> *"Why the hell did you say it. I can't even think about that, stop but we both know what to do and why"*

**Felt readback:** V=-0.09  A=+0.09  D=-0.23

---

### t = 42.0s  -  courtesy gesture
*gesture='polite_pass', intensity=0.5, discretionary=0.6*
**State after appraisal:** V=-0.28  A=+0.51  D=+0.03  frust=0.03

> *"They kept their damn distance, I can't keep my hands off the wheel."*

**Felt readback:** V=-0.30  A=+0.30  D=-0.32

---

### t = 55.0s  -  merge opportunity
*gap_seconds=2.5*
**State after appraisal:** V=-0.10  A=+0.36  D=+0.02  frust=0.02

> *"Shit, move it! I saw the gap and you're still there."*

**Felt readback:** V=-0.23  A=+0.20  D=-0.32

---


## How to read this

- **State after appraisal** = the symbolic V/A/D after the rule-based appraisal layer fired, before the LLM gets to weigh in. This is what's being *steered into* the LLM via activation injection.
- **The line in italics** is the LLM's first-person monologue, produced under steering at the layer-15 residual stream (α scaled from V/A/D with g=25).
- **Felt readback** = the LLM's final hidden state at the same layer, projected onto the same V/A/D probe directions used for steering. The closed-loop story's load-bearing claim is that intended (state) and felt (readback) live in the same coordinate system, so their difference can drive a re-appraisal Δ that pulls the symbolic state toward what was actually expressed.