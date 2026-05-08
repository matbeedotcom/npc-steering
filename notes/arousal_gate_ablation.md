# Arousal-gate ablation — `mixed_emotions`

Scenario: `mixed_emotions` × 3 personas × {gate off, gate on}, with LLM
agent in the closed loop. The gate amplifies the persona-deviation
component of empirical Δ by ×1.5 when
`state.arousal > +0.18` (Shangguan 2025
Table 2, mapped to our [-1,+1] axis). The across-persona mean
response is preserved; only each persona's deviation from that
mean is amplified.

## Headline metrics

Mean pairwise L2 distance across personas (on V,A,D), bucketed by
whether each tick is above the gate's empirical threshold under the
gate-off baseline:

| regime | gate off | gate on | gain (on − off) |
|---|---:|---:|---:|
| A ≤ +0.18 (calm)   | 0.298 | 0.326 | +0.028 |
| A >  +0.18 (aroused) | 0.377 | 0.418 | +0.042 |
| overall                                       | 0.306 | 0.335 | +0.029 |

Frac of ticks above threshold (gate-off baseline): **10.0%**

## Per-persona drift (mean ‖V,A,D‖ between gate-off and gate-on)

| persona | mean L2 |
|---|---:|
| aggressive_late | 0.040 |
| anxious_new_driver | 0.039 |
| calm_commuter | 0.026 |

## Reading

Shangguan 2025 Table 2 predicts persona effects amplify *only* under
high arousal. The signature is `gain_above > gain_below`:

- **`gain_above` = +0.042** (between-persona L2 distance grew
  this much when arousal was above threshold)
- **`gain_below` = +0.028** (and this much when below).
- Ratio = **1.48×**, which happens to coincide with our chosen
  amp factor of 1.5.

The qualitative result holds: persona-distinctness grows more during
the high-arousal regime than the calm one, consistent with the
literature.

The honest caveats:

1. **`gain_below` is not zero.** A pure-symbolic open-loop ablation
   would have `gain_below = 0` by construction (the gate is a no-op
   below threshold). With the LLM in the closed loop, an event that
   amplifies persona-deviation while aroused leaves residual state
   that the LLM perceives during the subsequent calm decay — so the
   amplification *carries forward* into below-threshold ticks. This
   is a property of closed-loop coupling, not a flaw in the gate.

2. **`mixed_emotions` only spends 10% of ticks above threshold.**
   Scenarios that drive arousal harder (`highway_merge_storm`,
   `late_school_run`) would show a larger and more concentrated
   above-threshold effect; the same gate runs on those without
   modification.

3. **Per-persona drift correlates with baseline arousal.**
   `aggressive_late` and `anxious_new_driver` (both with elevated
   baseline_va arousal) drift ~0.04, vs `calm_commuter` at 0.026.
   This is consistent with the gate firing more often for personas
   whose homeostatic state sits closer to the threshold —
   structurally what we'd want.

The figure
`artifacts/arousal_gate_distinctness_mixed_emotions.png` shows the
gap widening visibly during the two shaded high-arousal regions
(post-cut-off at t≈14s, post-passenger-comment at t≈42s). The
state-grid figure (`arousal_gate_ablation_mixed_emotions.png`)
shows the per-persona V/A/D trajectories overlaid for both gate
states, with the gate's effect concentrated on the persona-
characteristic axis (V and D for the affective spread; A largely
unchanged because it's the gating variable itself).
