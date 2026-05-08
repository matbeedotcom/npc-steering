# Card-vs-steering ablation — `mixed_emotions`

A 2×2 attribution of where the closed loop's persona differentiation
comes from. Same scenario × 3 personas × 4 conditions, with the LLM
agent in the loop for all four. Two metrics:

1. **Felt-VAD distinctness** - mean pairwise L2 distance across
   personas on the *probe-readback* V/A/D per utterance. This is the
   LLM-channel-specific signal: it measures whether the LLM's
   produced text actually projects onto different V/A/D coordinates
   per persona, in each condition.
2. **Symbolic-state distinctness** - the same metric but on the
   internal symbolic state. This is dominated by persona-specific
   dynamics (`baseline_va`, `patience`, `reactivity`) that evolve
   identically regardless of LLM channels, so it should be roughly
   flat across conditions. Reported here for comparison.

## Headline metric: felt-VAD distinctness (the LLM-relevant one)

| condition  | mean pairwise L2 on felt VAD |
|---|---:|
| card+steer (default) | 0.399 |
| card_only            | 0.382 |
| steer_only           | 0.398 |
| neither              | 0.240 |

The single most informative comparison is **`card+steer` vs
`card_only`**: the gap is the distinctness contribution attributable
to activation steering *on top of* an already-strong prompt anchor.
A second informative comparison is **`steer_only` vs `neither`**:
the gap measures what activation steering produces when the LLM has
*no* character anchor at all.

## Symbolic-state distinctness (sanity check; expected near-flat)

| condition  | mean L2 (V,A,D) |
|---|---:|
| card+steer  | 0.283 |
| card_only   | 0.303 |
| steer_only  | 0.298 |
| neither     | 0.272 |

If these are tightly clustered, the result confirms the metric was
the wrong primary one for this ablation: symbolic dynamics dominate
and the LLM's contribution gets washed out at κ=0.3 re-appraisal
coupling.

## State-tracking — Pearson r(intended V, felt V), averaged across personas

| condition  | mean r(V) |
|---|---:|
| card+steer  | +0.150 |
| card_only   | -0.109 |
| steer_only  | +0.625 |
| neither     | +0.118 |

How well each condition keeps the symbolic state and the LLM's
expressed projection in sync, *at the moment of utterance*.
Activation steering should help here because steering is the only
mechanism that puts the LLM's hidden state into the same coordinate
system as the symbolic state.

### Per-persona r(V) by condition

| persona | card+steer | card_only | steer_only | neither |
|---|---:|---:|---:|---:|
| aggressive_late | +0.615 | +0.122 | +0.581 | +0.348 |
| anxious_new_driver | -0.429 | -0.347 | +0.376 | -0.482 |
| calm_commuter | +0.265 | -0.103 | +0.917 | +0.489 |

## Reading guide

- Strongest reading of the architectural claim: `card+steer` >
  `card_only` on felt-distinctness *and* on r(V). That'd say
  steering does work the prompt alone doesn't.
- A weaker reading: `card+steer` ≈ `card_only` on felt-distinctness,
  but `card+steer` > `card_only` on r(V). Steering doesn't add
  *visible* persona differentiation, but it *does* keep the loop
  closer to closed.
- A null reading: all four conditions look similar on felt-distinctness
  AND r(V). The symbolic dynamics are doing all the work; the LLM
  channel is paint, not load-bearing.

The figure
`artifacts/card_vs_steering_felt_mixed_emotions.png` shows
felt-V trajectories per persona × condition for visual inspection.
