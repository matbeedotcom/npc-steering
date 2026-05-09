#!/usr/bin/env python3
"""Cross-modal audit: probe-based felt_VAD vs an independent audio readout.

The closed loop's central architectural claim is that the same probe
directions are used to *write* state into the LLM (steering) and to
*read* state out (projection). That symmetry is load-bearing - but
also self-confirming. To break the circularity we use an external,
audio-domain V/A/D regressor as an independent auditor of the LLM's
expressed affect:

  LLM utterance  →  TTS  →  16 kHz WAV
                              │
                              ▼
                       Whisper-large-v3-turbo encoder (layer -2)
                              │
                              ▼
                       mean-pool over time → (1, 1280)
                              │
                              ▼
                       matbee/whisper-to-vad (ONNX)
                              │
                              ▼
                       audio_VAD ∈ [-1, +1]^3
                              │
                       compare with felt_VAD from the trace

Methodological caveat: the regressor was trained on natural human
speech (CREMA-D / EmoVoice-DB / JL-Corpus). Synthetic TTS audio sits
in a different distribution, so disagreement may reflect
TTS-vs-human distribution shift as much as it reflects probe-vs-
audio modality disagreement. Re-running with human-recorded audio
of each utterance would be the cleanest version; this is the
pragmatic v1.

Output:
- `notes/audio_audit.md` - aggregate metrics + per-utterance table
- `artifacts/audio_audit_scatter.png` - probe vs audio scatter, V/A/D
- `runs/audio_cache/<hash>.wav` - cached TTS audio (gitignored)

Usage:
  pip install -e '.[audit]'        # one-time
  python scripts/12_audio_audit.py
  python scripts/12_audio_audit.py --traces runs/mixed_emotions__calm_commuter.agent.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

RUNS_DIR = _ROOT / "runs"
ARTIFACTS_DIR = _ROOT / "artifacts"
NOTES_DIR = _ROOT / "notes"
AUDIO_CACHE = RUNS_DIR / "audio_cache"

DEFAULT_TRACES = [
    RUNS_DIR / f"mixed_emotions__{p}.agent.jsonl"
    for p in ("calm_commuter", "aggressive_late", "anxious_new_driver")
]

WHISPER_ID = "openai/whisper-large-v3-turbo"
ONNX_REPO = "matbee/whisper-to-vad"
WHISPER_LAYER = -2
TARGET_SR = 16_000


# ---------------------------------------------------------------------------
# TTS - macOS `say` + afconvert. Swap this function on other platforms.
# ---------------------------------------------------------------------------

def synthesize_speech(text: str, out_wav: Path) -> None:
    """Render `text` to a 16 kHz mono WAV via macOS native TTS.

    Two-stage because `say -o` defaults to AIFF and `afconvert` is the
    cleanest macOS-native path to a 16 kHz WAV. If you're not on macOS,
    replace this function with `coqui-tts`, `pyttsx3`, or an OpenAI
    TTS call.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    aiff_path = out_wav.with_suffix(".aiff")
    subprocess.run(
        ["say", "-o", str(aiff_path), text],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", f"LEF32@{TARGET_SR}",
         str(aiff_path), str(out_wav)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    aiff_path.unlink(missing_ok=True)


def cache_path_for(text: str) -> Path:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return AUDIO_CACHE / f"{h}.wav"


def get_or_synthesize(text: str) -> Path:
    out = cache_path_for(text)
    if not out.exists():
        synthesize_speech(text, out)
    return out


# ---------------------------------------------------------------------------
# Whisper encoder + ONNX regressor
# ---------------------------------------------------------------------------

def load_whisper():
    """Lazy import - keeps the script importable without the audit deps."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(WHISPER_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_ID, torch_dtype=torch.float32,
    )
    model.eval()
    return model, processor, torch


def load_onnx_session():
    from huggingface_hub import hf_hub_download, list_repo_files
    import onnxruntime as ort
    files = list_repo_files(ONNX_REPO)
    onnx_files = [f for f in files if f.endswith(".onnx")]
    if not onnx_files:
        raise FileNotFoundError(
            f"No .onnx file found in HF repo {ONNX_REPO}. "
            f"Files seen: {files}"
        )
    # Prefer canonical names; fall back to first .onnx.
    preferred = next(
        (f for f in onnx_files if f in ("model.onnx", "whisper_to_vad.onnx")),
        onnx_files[0],
    )
    local_path = hf_hub_download(ONNX_REPO, preferred)
    sess = ort.InferenceSession(local_path, providers=["CPUExecutionProvider"])
    return sess


def whisper_embed(model, processor, torch_mod, wav_path: Path) -> np.ndarray:
    """Forward a WAV through the Whisper encoder, mean-pool layer -2."""
    import soundfile as sf
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        raise ValueError(
            f"Audio at {wav_path} is {sr} Hz; expected {TARGET_SR}. "
            f"Re-synthesize with `synthesize_speech`."
        )
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch_mod.no_grad():
        encoded = model.model.encoder(
            inputs.input_features, output_hidden_states=True,
        )
    hidden = encoded.hidden_states[WHISPER_LAYER]   # (1, T, 1280)
    pooled = hidden.mean(dim=1).numpy().astype(np.float32)  # (1, 1280)
    return pooled


def regress_audio_vad(sess, embed: np.ndarray) -> np.ndarray:
    """Run the ONNX regressor; returns (V, A, D) ∈ [-1, +1]^3."""
    out = sess.run(["vad"], {"whisper_embed": embed})[0]
    return out[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------

def load_trace_records(path: Path) -> Tuple[Dict, List[Dict]]:
    meta: Dict = {}
    utterances: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "meta":
                meta = rec
            elif kind == "utterance":
                utterances.append(rec)
    return meta, utterances


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------

_PERSONA_COLORS = {
    "calm_commuter":      "#2ca02c",
    "aggressive_late":    "#d62728",
    "anxious_new_driver": "#9467bd",
}


def render_scatter(rows: List[Dict], output_path: Path) -> None:
    """3-panel scatter: probe-V vs audio-V, probe-A vs audio-A, etc.

    Each point coloured by persona. Diagonal y=x reference for visual
    "are they agreeing" check.
    """
    import matplotlib.pyplot as plt

    axes_meta = [
        ("V", "valence",   "probe felt V",   "audio felt V"),
        ("A", "arousal",   "probe felt A",   "audio felt A"),
        ("D", "dominance", "probe felt D",   "audio felt D"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
    for ax, (key, name, xlab, ylab) in zip(axes, axes_meta):
        for persona, color in _PERSONA_COLORS.items():
            xs = [r[f"probe_{key}"] for r in rows if r["persona"] == persona]
            ys = [r[f"audio_{key}"] for r in rows if r["persona"] == persona]
            if xs:
                ax.scatter(xs, ys, color=color, s=60, alpha=0.85,
                           edgecolors="white", linewidth=0.5, label=persona)

        ax.plot([-1, 1], [-1, 1], color="grey", linewidth=0.8,
                linestyle=":", alpha=0.6, label="y = x")
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
        ax.axvline(0, color="black", linewidth=0.4, alpha=0.3)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"{name.capitalize()}", loc="left",
                     fontsize=11.5, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if key == "V":
            ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    fig.suptitle(
        "Cross-modal audit: probe-based felt_VAD vs whisper-to-vad audio readout",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {output_path}")


def pearson(a: List[float], b: List[float]) -> float:
    if len(a) < 2:
        return float("nan")
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < 1e-9 or sb < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def write_report(rows: List[Dict], report_path: Path) -> Dict[str, float]:
    """Aggregate metrics + per-utterance table; return summary dict."""
    summary: Dict[str, float] = {}
    for key in ("V", "A", "D"):
        probe = [r[f"probe_{key}"] for r in rows]
        audio = [r[f"audio_{key}"] for r in rows]
        diff = [a - p for p, a in zip(probe, audio)]
        summary[f"r_{key}"] = pearson(probe, audio)
        summary[f"mae_{key}"] = float(np.mean(np.abs(diff)))
        summary[f"bias_{key}"] = float(np.mean(diff))

    md = [
        "# Cross-modal audit - probe felt_VAD vs whisper-to-vad audio readout",
        "",
        "Each utterance from the canonical `mixed_emotions` traces was",
        "rendered with macOS TTS to a 16 kHz WAV, encoded with",
        "`openai/whisper-large-v3-turbo` (layer -2 mean-pooled), then",
        f"regressed to V/A/D via `{ONNX_REPO}` (ONNX). The audio readout",
        "is compared against the trace's `felt_VAD` produced by the",
        "self-projection of the LLM's hidden state onto the same probe",
        "directions used for steering.",
        "",
        "## Aggregate agreement",
        "",
        "| axis | Pearson r | mean abs diff | mean bias (audio - probe) |",
        "|---|---:|---:|---:|",
        f"| V | {summary['r_V']:+.3f} | {summary['mae_V']:.3f} | {summary['bias_V']:+.3f} |",
        f"| A | {summary['r_A']:+.3f} | {summary['mae_A']:.3f} | {summary['bias_A']:+.3f} |",
        f"| D | {summary['r_D']:+.3f} | {summary['mae_D']:.3f} | {summary['bias_D']:+.3f} |",
        "",
        "Read: `r` is *do they move together?*; `mae` is *how far apart in absolute terms?*;",
        "`bias` is *does one consistently read higher than the other?*.",
        "",
        "## Reading the result",
        "",
        "- A high `r` (≥ 0.5) on V/A would be strong cross-modal validation",
        "  of the closed-loop's coordinate-system claim - the LLM's",
        "  hidden-state projection and an independent audio model agree on",
        "  what the utterance felt like.",
        "- A near-zero `r` would be the harder reading: either the probes",
        "  overclaim, the audio model misreads synthetic TTS, or the two",
        "  modalities encode different aspects of affect.",
        "- A consistent `bias` would point to a systematic offset between",
        "  the two readouts - rescalable but worth flagging.",
        "",
        "**Methodological caveat**: the regressor was trained on natural",
        "human speech (CREMA-D / EmoVoice-DB / JL-Corpus). Synthetic TTS",
        "sits in a different distribution. A v2 with human-recorded",
        "utterances would settle this; this v1 audit is suggestive, not",
        "definitive.",
        "",
        "## Per-utterance disagreement",
        "",
        "| persona | text | probe (V/A/D) | audio (V/A/D) | ‖diff‖ |",
        "|---|---|---|---|---:|",
    ]
    for r in rows:
        text = r["text"].replace("|", "\\|")
        if len(text) > 80:
            text = text[:77] + "..."
        probe = f"({r['probe_V']:+.2f}, {r['probe_A']:+.2f}, {r['probe_D']:+.2f})"
        audio = f"({r['audio_V']:+.2f}, {r['audio_A']:+.2f}, {r['audio_D']:+.2f})"
        diff_norm = float(np.linalg.norm([
            r["probe_V"] - r["audio_V"],
            r["probe_A"] - r["audio_A"],
            r["probe_D"] - r["audio_D"],
        ]))
        md.append(f"| {r['persona']} | {text} | {probe} | {audio} | {diff_norm:.2f} |")

    md.append("")
    md.append(
        f"Scatter figure: "
        f"[`artifacts/audio_audit_scatter.png`](../artifacts/audio_audit_scatter.png)"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md))
    print(f"[report]  {report_path}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--traces", nargs="+", default=[str(p) for p in DEFAULT_TRACES],
        help="JSONL trace files to audit.",
    )
    ap.add_argument(
        "--output-figure",
        default=str(ARTIFACTS_DIR / "audio_audit_scatter.png"),
    )
    ap.add_argument(
        "--output-report", default=str(NOTES_DIR / "audio_audit.md"),
    )
    args = ap.parse_args()

    trace_paths = [Path(p) for p in args.traces]
    missing = [p for p in trace_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trace files: {missing}")

    print(f"[whisper] loading {WHISPER_ID}...")
    model, processor, torch_mod = load_whisper()
    print(f"[onnx]    fetching {ONNX_REPO}...")
    sess = load_onnx_session()

    AUDIO_CACHE.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for path in trace_paths:
        meta, utterances = load_trace_records(path)
        persona = meta.get("persona", path.stem)
        print(f"[trace]   {path.name}  ({len(utterances)} utterances)")
        for i, u in enumerate(utterances):
            text = (u.get("text") or "").strip()
            if not text:
                continue
            felt = u.get("felt") or u.get("felt_vad") or {}
            wav = get_or_synthesize(text)
            embed = whisper_embed(model, processor, torch_mod, wav)
            audio_vad = regress_audio_vad(sess, embed)
            rows.append({
                "persona": persona,
                "t": float(u.get("t", 0.0)),
                "text": text,
                "probe_V": float(felt.get("V", 0.0)),
                "probe_A": float(felt.get("A", 0.0)),
                "probe_D": float(felt.get("D", 0.0)),
                "audio_V": float(audio_vad[0]),
                "audio_A": float(audio_vad[1]),
                "audio_D": float(audio_vad[2]),
            })
            print(
                f"  [{i+1}/{len(utterances)}]  "
                f"probe=({felt.get('V', 0):+.2f},{felt.get('A', 0):+.2f},{felt.get('D', 0):+.2f})  "
                f"audio=({audio_vad[0]:+.2f},{audio_vad[1]:+.2f},{audio_vad[2]:+.2f})  "
                f"text={text[:40]!r}{'...' if len(text) > 40 else ''}"
            )

    Path(args.output_figure).parent.mkdir(parents=True, exist_ok=True)
    render_scatter(rows, Path(args.output_figure))
    summary = write_report(rows, Path(args.output_report))

    print()
    print("=== SUMMARY ===")
    for axis in ("V", "A", "D"):
        print(
            f"  {axis}:  r={summary[f'r_{axis}']:+.3f}  "
            f"mae={summary[f'mae_{axis}']:.3f}  "
            f"bias={summary[f'bias_{axis}']:+.3f}"
        )


if __name__ == "__main__":
    main()
