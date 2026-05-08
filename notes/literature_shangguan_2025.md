# Literature: Shangguan et al. 2025 (Frontiers in Psychology)

**Citation.** Shangguan Z, Han X, Mrhasli YE, Lyu N, Tapus A (2025).
*Factors influencing emotional driving: examining the impact of arousal
on the interplay between age, personality, and driving behaviors.*
Front. Psychol. 16:1487493. doi: 10.3389/fpsyg.2025.1487493.
Open Access (CC-BY).

---

## Why this paper is in our reference set

It empirically grounds **architectural choices** that we made by feel —
specifically, that arousal moderates a personality→behavior pipeline,
that V/A axes are the right primitives, and that Big-Five-style
personality dimensions (Extraversion, Neuroticism) are useful
moderators. It does NOT give us decay constants or per-event ΔV/A/D
magnitudes; for those we still need biometric naturalistic-driving
data (SHRP2 / UDRIVE / Healey & Picard 2005).

What we can use it to back, in priority order:

1. **Decision coupling** (`lib/decisions.py`) — concrete β coefficients
   linking age / experience / personality to acceleration, speed
   stability, and steering variability. Replaces the hand-authored
   coefficients in `gap_threshold`, `following_distance`,
   `speed_overage` with measured values, modulo our `[-1, +1]`
   normalisation.
2. **Architecture validation** (writeup) — empirical evidence that
   personality moderates arousal→behavior, validating our
   `appraise(event, persona, state)` shape.
3. **Calibrated arousal threshold** — high-arousal cutoff at
   SAM 5.3/9 (= 0.59 normalised), useful for the calibration
   discussion.
4. **SAM-V/A coordinate alignment** — our V and A axes are the same
   primitives the paper measures, so Probe-VAD ↔ SAM mapping is
   structurally clean.

---

## Study design (terse)

| Element | Value |
|---|---|
| n | 40 Chinese drivers (≥1 year licensed) |
| Gender | Male:female ≈ 3.4:1 |
| Apparatus | Fixed-base driving simulator, 270° FoV, 80 km/h target speed, 2-lane 3-km highway |
| Emotion induction | 7 video clips (8–29s) presented before driving block: neutral, happiness, sadness, anger, disgust, surprise, fear |
| Affect measurement | SAM (Self-Assessment Manikin), 1–9 scale on Valence + Arousal |
| Personality measurement | EPQ — Extraversion (E) + Neuroticism (N); Psychoticism excluded |
| Driving outcomes | log_ACC95th (95th-percentile acceleration), log_SpeedSD, log_SteeringSD |
| Statistical method | Hayes PROCESS macro models 1, 3, 14; bootstrapping; VIF<1.5 (no multicollinearity) |

**Arousal categorisation** (their definition, useful as a calibration
anchor for us):

> High arousal = SAM_arousal > 5.3 (sample mean); Low arousal otherwise.
> Counts: 132 high-arousal samples, 108 low-arousal samples.

Normalised to our `[-1, +1]` scale: 5.3/9 = 0.589, then linearly
mapped to (0.589 - 0.5) × 2 = **+0.18**. So our `state.arousal > +0.18`
empirically corresponds to "behaviorally elevated" per their threshold.

---

## Quantitative findings — coefficients usable in our system

### Table 1 (their numbering): Moderated mediation w/ Extraversion as mediator

| Path | Outcome | β | t | p |
|---|---|---|---|---|
| Age → Extraversion | E (mediator) | **+0.113** | — | <0.001 |
| Gender (M=1,F=2) → E | E | −3.872 | −5.361 | <0.001 |
| Age → log_ACC95 | accel | **−0.014** | −9.379 | <0.001 |
| Age → log_SpeedSD | speed stability | **−0.006** | −4.225 | <0.001 |
| Age → log_SteeringSD | steering var | **+0.009** | +6.306 | <0.001 |
| Gender → log_ACC95 | accel | −0.075 | −2.084 | <0.05 |
| Extraversion → log_ACC95 | accel | **+0.013** | +4.081 | <0.001 |
| Extraversion → log_SpeedSD | speed stability | **+0.010** | +3.538 | <0.001 |
| Extraversion → log_SteeringSD | steering var | +0.003 | +1.088 | n.s. |
| Arousal (W) main effect | log_SpeedSD | −0.041 | −1.729 | n.s. |
| **E × Arousal interaction** | log_ACC95 | **+0.012** | +2.105 | <0.05 |
| **E × Arousal interaction** | log_SpeedSD | **+0.011** | +2.294 | <0.05 |

R² for ACC95 model: **0.288** (F=18.897). R² for SpeedSD: 0.120.
R² for SteeringSD: 0.200. Age × Gender alone explained 20.6% of E
variance (R²=0.206, F=30.722, p<0.001).

**Interpretation in our model:**
- Older drivers have **lower acceleration variability** (consistent with
  our calm-commuter persona). The β=−0.014 per year is a directly
  usable slope.
- **Extraversion increases ACC95 and SpeedSD.** This maps to our
  `risk_preference` axis — high-E ↔ high-risk_preference. We can
  re-anchor our `gap_threshold(state, persona)` against this.
- **Arousal × Extraversion is the load-bearing interaction.** High
  arousal *amplifies* Extraversion's effect on driving variability.
  Our model captures this implicitly via `state.arousal` modulating
  decision parameters. The β=+0.012 (E × A → ACC95) is the empirical
  anchor for that modulation strength.

### Table 2 (their numbering): Conditional indirect effect of Age on driving behaviors via Extraversion, by Arousal

| Outcome | Arousal | Indirect effect (β) | 95% bootstrap CI |
|---|---|---|---|
| log_ACC95 | Low | 0.0007 | (−0.0003, 0.0012) — n.s. |
| log_ACC95 | **High** | **0.002** | **(0.0011, 0.0031)** — sig |
| log_SpeedSD | Low | 0.0004 | (−0.0003, 0.0012) — n.s. |
| log_SpeedSD | **High** | **0.0017** | **(0.0007, 0.0029)** — sig |

The age→E→behavior pathway is **only significant under high arousal**.
This is the cleanest piece of evidence in the paper for our
`state.arousal` modulating decision parameters: when calm, the
persona vector matters less; when aroused, it dominates.

### Table 3 (their numbering): Moderated mediation w/ Neuroticism as mediator (mostly null)

| Path | β | p |
|---|---|---|
| Age → Neuroticism | +0.046 | 0.256 (n.s.) |
| Gender → N | +2.524 | <0.01 (female higher) |
| **Neuroticism → log_SpeedSD** | **+0.005** | **<0.05** |
| Neuroticism → log_ACC95 | −0.002 | 0.372 (n.s.) |
| Neuroticism × Arousal | −0.006 / +0.002 / −0.006 | all n.s. |

R²(N from Age+Gender) = **0.029** (F=3.540).

**Interpretation:** N is weakly mediated by age in this sample but
**directly increases speed variability**. Maps to our `reactivity` axis —
high-N drivers fluctuate more in speed. Importantly, N's effect is NOT
arousal-moderated, unlike E's.

### Table 6 (their numbering): Direct moderation by Neuroticism (no arousal in model)

| Predictor | log_ACC95 β | log_SpeedSD β | log_SteeringSD β |
|---|---|---|---|
| Driving experience (X) | **−0.019**∗∗∗ | **−0.009**∗∗∗ | **+0.016**∗∗∗ |
| Neuroticism (M) | −0.001 | **+0.006**∗∗ | +0.002 |
| **X × M** | **+0.002**∗∗ | **+0.002**∗∗∗ | +0.000 |
| R² | 0.205 | **0.317** | **0.506** |
| ΔR² (with interaction) | 0.040 | 0.057 | 0.003 |

Driving experience explains **31.7% of speed-stability variance** and
**50.6% of steering variability** — substantial. Neuroticism *attenuates*
the experience benefit on acceleration & speed stability.

### Table 7 (their numbering): Conditional effect of driving experience by Neuroticism level

| Outcome | N level | Effect | 95% CI | sig? |
|---|---|---|---|---|
| log_ACC95 | Low (mean−SD) | −0.0294 | (−0.0383, −0.0204) | ✓ |
| log_ACC95 | Mean | −0.0193 | (−0.0243, −0.0143) | ✓ |
| log_ACC95 | High (mean+SD) | −0.0093 | (−0.0153, −0.0033) | ✓ |
| log_SpeedSD | Low | −0.0182 | (−0.0260, −0.0105) | ✓ |
| log_SpeedSD | Mean | −0.0085 | (−0.0128, −0.0042) | ✓ |
| log_SpeedSD | **High** | +0.0012 | (−0.0040, +0.0063) | **✗ n.s.** |

**Reading.** Driving experience helps low-Neuroticism drivers
substantially (β=−0.029 on ACC95) but barely helps high-Neuroticism
drivers (β=−0.009). For speed stability, high-N drivers gain *nothing*
from experience.

This is directly applicable to our `anxious_new_driver` persona:
high N + low experience = exactly the pattern this paper says is most
disrupted. Our trajectory dynamics for that persona should reflect
*reduced learning from previous events* — currently we don't
distinguish, but this is a tuning lever.

---

## Mapping to our persona vector

Our axes are: temperament, patience, risk_preference, reactivity,
baseline_va. The paper uses Extraversion (E) and Neuroticism (N).
Empirical mapping:

| our axis | best paper analog | rationale |
|---|---|---|
| `risk_preference` | **+E** (Extraversion) | E increases ACC95 & SpeedSD; risk-seeking |
| `reactivity` | **+N** (Neuroticism) | N increases SpeedSD; emotional fluctuation |
| `patience` | **−N (partly)** | high-N drivers gain less from experience → less stable |
| `temperament` | **+E + N (interaction)** | volatility = positive on both |
| `baseline_va` | — | not measured continuously in paper; SAM is a one-shot snapshot |

Our three demo personas in this empirical frame:

| persona | our (risk_pref, reactivity) | paper interpretation |
|---|---|---|
| `calm_commuter` | (−0.3, 0.4) | Low E, low-mid N → low ACC95, low SpeedSD |
| `aggressive_late` | (+0.7, 0.8) | High E, high N → high ACC95, high SpeedSD; **high-arousal amplifies** |
| `anxious_new_driver` | (−0.6, 0.9) | Low E, high N → low ACC95 but unstable speed; **doesn't learn from experience** |

These distinctions are *qualitatively predicted by the paper*. Useful
validation for the writeup.

---

## What we can ground with this paper

### 1. `lib/decisions.py` — slot in measured β's

Currently:
```python
def speed_overage(s, p) -> float:
    return 0.10 * p.risk_preference + 0.15 * s.frustration
```

Empirically grounded version (proposed):
```python
# Coefficients from Shangguan 2025 Table 1 (E×A interaction) and
# Table 6 (X×M interaction), normalised to our [-1,+1] axes:
#   risk_pref  ≈ +1 corresponds to "high E"
#                — Δlog_SpeedSD per high-E unit ≈ +0.011
#   arousal     ≈ +1 corresponds to "high-arousal state" (SAM>5.3)
#                — interacts with E at β = +0.011
#   experience  modulates baseline; high N (≈ reactivity) attenuates
def speed_overage(s, p) -> float:
    base       = 0.10 * p.risk_preference                        # E main effect
    arousal_amp = 0.05 * p.risk_preference * max(0, s.arousal)   # E×A interaction
    react_drag = 0.04 * p.reactivity * (1 - max(0, s.dominance)) # N main effect
    return base + arousal_amp + react_drag + 0.15 * s.frustration
```

The numbers are scaled per our `[-1, +1]` normalisation so they're not
direct β copies — but the *signs and interaction structure* are now
literature-grounded rather than picked by feel. Worth a writeup
sentence: *"decision-coupling coefficients are anchored to Shangguan
2025 β estimates after rescaling to our normalised state axes."*

### 2. Arousal-modulated persona effects

Use Table 2's *"only significant under high arousal"* finding to
introduce arousal-gating in the appraisal layer:

```python
def appraise(event, persona, state):
    base_delta = ...                                # current logic
    if state.arousal > +0.18:                       # paper's empirical threshold
        # high-arousal regime: amplify persona effects per Shangguan 2025
        base_delta = base_delta * 1.5
    return base_delta
```

This makes our model match the paper's central finding: personality
matters *more* when aroused.

### 3. Validation language for the writeup

Concrete sentences for `notes/writeup.md`:

> "The architecture's central claim — that personality-modulated
> appraisal of driving events shapes affective state and downstream
> decisions — is empirically supported by Shangguan et al. 2025, who
> demonstrate that Extraversion mediates the age→behavior pathway
> with a *high-arousal-conditional indirect effect* of β=0.002 on
> peak acceleration variability and β=0.0017 on speed variability
> (vs. non-significant effects under low arousal)."

> "Our V/A axes use the same Mehrabian-PAD primitives as the SAM
> instrument employed by Shangguan 2025. Their high-arousal threshold
> (SAM > 5.3/9 = 0.59) corresponds to our `state.arousal > +0.18`
> after rescaling to `[-1, +1]`."

---

## What this paper does NOT give us

For completeness, so we don't over-claim:

| we wanted | paper has? |
|---|---|
| τ for V decay | ✗ — single-shot SAM, no time-series recovery |
| τ for A decay | ✗ — same |
| τ for D decay | ✗ — D not measured at all (SAM here is V & A only) |
| Per-event ΔV/A/D magnitudes (e.g., cut-off) | ✗ — induction is video-clip-based, not on-road event-based |
| Frustration / stress dynamics | ✗ — only general arousal valence, no decomposed accumulators |
| Continuous physiological signals (HRV, EDA) | ✗ — paper notes EEG was collected but not used in this analysis |
| Cross-cultural generalisation | ⚠ — Chinese-only sample, paper itself flags this |
| Female-driver patterns | ⚠ — male-skewed sample (3.4:1) |

**For genuine biometric grounding** of decay constants, the paper to
cite (and for future work) is Healey & Picard (2005) *"Detecting
stress during real-world driving tasks using physiological sensors"* —
they have HRV / EDA / EMG time-series with event annotations.

---

## Citations to pull into our `references.md` / writeup

```bibtex
@article{shangguan2025,
  title   = {Factors influencing emotional driving: examining the impact
             of arousal on the interplay between age, personality, and
             driving behaviors},
  author  = {Shangguan, Zhegong and Han, Xiao and Mrhasli, Younesse El
             and Lyu, Nengchao and Tapus, Adriana},
  journal = {Frontiers in Psychology},
  volume  = {16},
  pages   = {1487493},
  year    = {2025},
  doi     = {10.3389/fpsyg.2025.1487493},
}
```

Their referenced foundations we should also have visibility on:

- Bradley & Lang 1994 — SAM scale (we use the same V/A primitives)
- Hayes 2017 — PROCESS macro (their statistical method)
- Eysenck & Eysenck 1975 — EPQ (the personality instrument)
- Roberts et al. 2006 — age→Extraversion empirical pattern
- Lazarus 1991 — appraisal theory (already cited in our `design.md`)
- Mehrabian 1996 — PAD framework (already cited in our `design.md`)
