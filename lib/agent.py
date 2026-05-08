"""SteeredAgent — closed-loop affect-conditioned LLM driver narrator.

Wires three things together:

  symbolic state ──► state_to_alphas ──► steering offset at layer 15 ──┐
                                                                       │
  persona card + state descriptor + event description ──► chat prompt ─┤
                                                                       │
                                                 frozen LLM (Qwen3.5)  │
                                                          │            │
                                                          ▼            │
                                                       output text     │
                                                          │            │
                                                          ▼            │
                                                  forward(prompt+output)
                                                          │
                                                          ▼
                                              final hidden @ layer 15
                                                          │
                                              project onto v_V, v_A, v_D
                                                          │
                                                          ▼
                                                       felt_VAD
                                                          │
                              re_appraise(felt, intended) ─► ΔState (closes loop)

The single most important design choice is the symmetry: the *same*
probe directions are used to write state into the model (steering)
and to read state out (projection). The symbolic state is pulled
toward what was actually expressed by `re_appraise`.

For Phase D, the closed loop runs:
  while scenario_active:
      events = poll_events()
      for ev in events:
          state.apply(appraise(ev, persona, state))
          response = agent.respond(state, persona, event_to_text(ev))
          state.apply(response.delta_state)         # closed-loop re-appraisal
      state.step(dt, persona)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from .events import Event, event_to_text
from .persona import DriverPersona
from .probes import ProbeBundle, state_to_alphas
from .state import DeltaState, DriverState, state_to_description

if TYPE_CHECKING:
    import mlx.core as mx


# ---------------------------------------------------------------------------
# Output cleanup — Qwen3 leaks two failure modes we have to scrub
# ---------------------------------------------------------------------------

# Even with `enable_thinking=False`, ~5% of turns at temperature=0.7
# emit a fresh `<think>...</think>` block mid-response. We strip them.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)

# Qwen3's safety-alignment training occasionally refuses persona-card
# voices that include cursing or aggression — even under the explicit
# fictional-roleplay framing. Detected refusals are retried with
# temperature bumped up; that breaks the safety-mode attractor most
# of the time without needing a stronger jailbreak.
_REFUSAL_LEADS = (
    "i can't generate",
    "i cannot generate",
    "i'm sorry, but i",
    "i won't generate",
    "i refuse to",
    "i shouldn't",
    "i don't think i should",
    "as an ai",
    "i can't roleplay",
)


def _scrub_output(text: str) -> str:
    """Strip leaked `<think>` blocks and trim whitespace."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _OPEN_THINK_RE.sub("", text)   # un-closed open <think> stragglers
    return text.strip()


def _looks_like_refusal(text: str) -> bool:
    """True if the output's first ~80 chars match a known refusal lead."""
    head = text.lower().lstrip()[:80]
    return any(head.startswith(lead) for lead in _REFUSAL_LEADS)


# ---------------------------------------------------------------------------
# Persona card assembly
# ---------------------------------------------------------------------------

def assemble_card(card_data: dict) -> str:
    """Render a structured persona card as the [CHARACTER]/[VOICE]/[REFERENCE LINES] block."""
    refs = "\n".join(
        f"- When {r['context']}: \"{r['line']}\""
        for r in card_data["reference_lines"]
    )
    return (
        f"[CHARACTER]\n{card_data['character']}\n\n"
        f"[VOICE]\n{card_data['voice']}\n\n"
        f"[REFERENCE LINES]\n{refs}"
    )


def load_persona_cards(path: str | Path) -> Dict[str, str]:
    """Load `data/persona_cards.json` and pre-render each card."""
    blob = json.loads(Path(path).read_text())
    return {name: assemble_card(card) for name, card in blob["personas"].items()}


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """Output of one closed-loop turn.

    `intended_VAD` is what we asked for (state.{V/A/D}, in [-1,+1]).
    `felt_VAD` is what we read back, normalised to roughly the same
    range. `delta_state` is the symbolic update that closes the loop —
    add it to the symbolic state to pull toward what was expressed.
    """

    text: str
    intended_VAD: Dict[str, float]
    felt_VAD: Dict[str, float]
    delta_state: DeltaState
    raw_felt_projection: Dict[str, float]   # unnormalised <h, v_axis>


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

_AXIS_TO_STATE_ATTR = {"V": "valence", "A": "arousal", "D": "dominance"}


class SteeredAgent:
    """Frozen LLM + V/A/D probes + persona cards. One method: `respond`."""

    def __init__(
        self,
        model,
        tokenizer,
        probes: ProbeBundle,
        persona_cards: Dict[str, str],
    ) -> None:
        if not probes.selected_layers:
            raise ValueError(
                "ProbeBundle has no selected_layers — run 03_validate_probes.py "
                "and set bundle.selected_layers before constructing SteeredAgent."
            )
        self.model = model
        self.tokenizer = tokenizer
        self.probes = probes
        self.persona_cards = persona_cards

    # -- Prompt construction ----------------------------------------------

    def _build_messages(
        self,
        state: DriverState,
        persona: DriverPersona,
        event_description: str,
    ) -> list[dict]:
        if persona.name not in self.persona_cards:
            raise KeyError(
                f"No persona card for {persona.name!r}. "
                f"Available: {sorted(self.persona_cards.keys())}"
            )
        card = self.persona_cards[persona.name]
        system = (
            "You are voicing a fictional driver's internal thoughts for "
            "a creative-writing exercise. Stay fully in character. "
            "Match their voice, register, and vocabulary exactly — "
            "including unfiltered language and strong emotion where the "
            "character's reference lines call for it. Don't sanitise.\n\n"
            f"{card}\n\n"
            "You ARE this character right now, in this moment, and you "
            "speak their internal monologue. Output ONLY a single "
            "first-person line — what they'd think to themselves, "
            "not say out loud. Maximum 18 words. No analysis, no "
            "explanation, no quoting, no narration, no thinking process, "
            "no <think> tags. Stay in the voice and reference style "
            "described above."
        )
        user = (
            f"Your current internal state: {state_to_description(state)}.\n"
            f"What just happened: {event_description}\n"
            "\nWhat goes through your head, in one short sentence:"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # -- Steering offset construction -------------------------------------

    def _build_offsets(self, state: DriverState) -> Dict[int, "mx.array"]:
        """Compute combined per-layer steering offsets from current state.

        If multiple axes share a layer (the canonical case — V/A/D all
        on layer 15), their contributions sum into one offset.
        """
        import mlx.core as mx

        alphas = state_to_alphas(state)
        offsets: Dict[int, "mx.array"] = {}
        for axis, layer in self.probes.selected_layers.items():
            v_np = self.probes.vec(axis, layer)
            v_mx = mx.array(v_np, dtype=mx.bfloat16)
            contrib = float(alphas[axis]) * v_mx
            offsets[layer] = offsets.get(layer, mx.zeros_like(v_mx)) + contrib
        return offsets

    # -- Readout / projection ---------------------------------------------

    def _read_felt_VAD(self, prompt_str: str, output_text: str) -> tuple[Dict[str, float], Dict[str, float]]:
        """Forward (prompt + output), capture last-token hidden, project onto probes.

        Returns `(felt_normalised, raw_projection)`. Normalisation uses
        `‖mean_diff‖/2` per axis (from extraction diagnostics) as the
        half-range calibration anchor — so a "fully positive" prompt
        reads close to +1 and "fully negative" close to -1.
        """
        from .mlx_steering import extract_last_token_hiddens

        full = prompt_str + output_text
        layers_set = sorted(set(self.probes.selected_layers.values()))
        hiddens = extract_last_token_hiddens(self.model, self.tokenizer, full, layers_set)

        raw: Dict[str, float] = {}
        normalised: Dict[str, float] = {}
        for axis, layer in self.probes.selected_layers.items():
            v = self.probes.vec(axis, layer).astype(np.float32)
            h = hiddens[layer].astype(np.float32)
            proj = float(np.dot(h, v))
            raw[axis] = proj
            calib = max(self.probes.diagnostics[axis][layer]["norm_unnormalised"] / 2.0, 1.0)
            n = proj / calib
            normalised[axis] = float(np.clip(n, -1.5, 1.5))
        return normalised, raw

    # -- Public API --------------------------------------------------------

    def respond(
        self,
        state: DriverState,
        persona: DriverPersona,
        event: "Event | str",
        *,
        max_tokens: int = 50,
        kappa: float = 0.3,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        repetition_context_size: int = 24,
    ) -> AgentResponse:
        """Run one closed-loop turn.

        `event` may be a typed `Event` (auto-converted via `event_to_text`)
        or a pre-built English string for ad-hoc use.

        `kappa` is the re-appraisal coupling: how strongly the symbolic
        state is pulled toward what was actually expressed. With state
        clipped to `[-1, +1]` per axis, κ ≤ 0.3 keeps the loop stable.

        Generation defaults — `temperature=0.7`, `repetition_penalty=1.2`,
        `repetition_context_size=24` — were picked to break the looping
        failure mode that strong steering (state.|axis| > 0.5) induces
        under greedy decoding. With these on, outputs stay coherent
        across the full state range; without them, high-arousal /
        high-frustration turns collapse into n-gram loops.
        """
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        from .mlx_steering import unwrap, wrap_steering

        if not isinstance(event, str):
            event_description = event_to_text(event)
        else:
            event_description = event

        # 1. Build chat prompt with thinking disabled.
        messages = self._build_messages(state, persona, event_description)
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        # 2. Wrap layers with steering offsets.
        offsets = self._build_offsets(state)

        # 3. Generate with sampler + repetition penalty. Retry once on
        #    refusal at higher temperature (breaks the safety-mode
        #    attractor without needing a real jailbreak).
        processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )
        originals = wrap_steering(self.model, offsets)
        try:
            for attempt in range(2):
                temp_attempt = temperature if attempt == 0 else min(1.1, temperature + 0.3)
                sampler = make_sampler(temp=temp_attempt, top_p=0.9)
                raw = generate(
                    self.model, self.tokenizer, prompt=prompt_str,
                    max_tokens=max_tokens, verbose=False,
                    sampler=sampler, logits_processors=processors,
                )
                text = _scrub_output(raw)
                if not _looks_like_refusal(text) and text:
                    break
        finally:
            unwrap(self.model, originals)

        # 4. Read felt V/A/D out (separate forward pass on prompt + output,
        #    no steering active). The probe directions used here are the
        #    *same* directions used for steering — coordinate-system
        #    symmetry is the closed-loop story's load-bearing piece.
        felt_normalised, raw_projection = self._read_felt_VAD(prompt_str, text)

        # 5. Intended V/A/D = symbolic state scalars (already in [-1,+1]).
        intended = {
            axis: float(getattr(state, _AXIS_TO_STATE_ATTR[axis]))
            for axis in self.probes.selected_layers
        }

        # 6. Re-appraisal: pull symbolic state toward expressed.
        delta = DeltaState(
            dv=kappa * (felt_normalised.get("V", 0.0) - intended.get("V", 0.0)),
            da=kappa * (felt_normalised.get("A", 0.0) - intended.get("A", 0.0)),
            dd=kappa * (felt_normalised.get("D", 0.0) - intended.get("D", 0.0)),
        )

        return AgentResponse(
            text=text,
            intended_VAD=intended,
            felt_VAD=felt_normalised,
            delta_state=delta,
            raw_felt_projection=raw_projection,
        )
