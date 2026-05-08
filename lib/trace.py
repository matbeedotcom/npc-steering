"""JSONL trace logger.

One line per logged record. Three record kinds:

  {"kind": "state",    "t": float, "valence": ..., ...}
  {"kind": "event",    "t": float, "type": str, "payload": {...}, "delta": {...}}
  {"kind": "decision", "t": float, "gap_threshold_s": float, ...}

JSONL is the right shape because it's append-only, partial-failure
tolerant, and trivially loaded by `pandas.read_json(..., lines=True)`
or `jq -s` for the visualization scripts.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import IO, Optional

from .events import Event
from .persona import DriverPersona
from .state import DeltaState, DriverState


class TraceLogger:
    """Context-manager around a JSONL file.

    Use as:

        with TraceLogger("runs/foo.jsonl", scenario="late_school_run",
                         persona=p) as trace:
            trace.state(t, state)
            trace.event(t, event, delta)
            trace.decision(t, decisions_dict)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        scenario: Optional[str] = None,
        persona: Optional[DriverPersona] = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: Optional[IO[str]] = None
        self._scenario = scenario
        self._persona = persona

    # -- Context-manager plumbing -----------------------------------------

    def __enter__(self) -> "TraceLogger":
        self._fp = open(self.path, "w")
        # First line is a meta record for plot scripts.
        meta = {"kind": "meta"}
        if self._scenario is not None:
            meta["scenario"] = self._scenario
        if self._persona is not None:
            meta["persona"] = self._persona.name
            meta["persona_traits"] = {
                "temperament": self._persona.temperament,
                "patience": self._persona.patience,
                "risk_preference": self._persona.risk_preference,
                "reactivity": self._persona.reactivity,
                "baseline_va": list(self._persona.baseline_va),
            }
        self._write(meta)
        return self

    def __exit__(self, *args) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    # -- Record kinds ------------------------------------------------------

    def state(self, t: float, state: DriverState) -> None:
        self._write({"kind": "state", "t": round(t, 4), **state.snapshot()})

    def event(self, t: float, event: Event, delta: DeltaState) -> None:
        self._write({
            "kind": "event",
            "t": round(t, 4),
            "type": event.type,
            "payload": event.model_dump(exclude={"type", "t"}),
            "delta": asdict(delta),
        })

    def decision(self, t: float, decisions: dict) -> None:
        self._write({"kind": "decision", "t": round(t, 4), **decisions})

    def utterance(self, t: float, text: str, felt_vad: Optional[dict] = None) -> None:
        """LLM-generated monologue line. Used in Phase D once steering is wired."""
        rec = {"kind": "utterance", "t": round(t, 4), "text": text}
        if felt_vad is not None:
            rec["felt"] = felt_vad
        self._write(rec)

    # -- Internals ---------------------------------------------------------

    def _write(self, record: dict) -> None:
        if self._fp is None:
            raise RuntimeError("TraceLogger not opened (use as context manager).")
        self._fp.write(json.dumps(record) + "\n")
