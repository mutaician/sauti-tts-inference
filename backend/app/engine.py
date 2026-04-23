from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .f5_infer_safe import safe_infer_process
from .utils import normalize_swahili_text, waveform_to_audio_bytes

logger = logging.getLogger(__name__)


class SpeakerNotFoundError(ValueError):
    """Raised when a speaker outside the packaged set is requested."""


class SynthesisError(RuntimeError):
    """Raised when synthesis fails or produces unusable audio."""


@dataclass(frozen=True)
class SpeakerSpec:
    speaker: int
    label: str
    ref_audio_path: str
    ref_text: str


@dataclass(frozen=True)
class InferencePlan:
    """Metadata for responses (e.g. X-TTS-Mode). Matches the sequential F5 path used locally."""

    mode: str = "f5-safe-sequential"
    infer_backend: str = "safe"
    nfe_steps: int = 32
    cfg_strength: float = 2.0


def _patch_torchaudio_load_with_soundfile() -> None:
    if getattr(torchaudio.load, "_sauti_soundfile_patch", False):
        return

    def _load_with_soundfile(path: str, *args, **kwargs):
        audio_np, sample_rate = sf.read(path, dtype="float32")
        if audio_np.ndim == 1:
            audio_np = audio_np[:, None]
        waveform = torch.from_numpy(audio_np.T.copy())
        return waveform, sample_rate

    _load_with_soundfile._sauti_soundfile_patch = True
    torchaudio.load = _load_with_soundfile


def _load_vocoder_compat(device: str):
    from f5_tts.infer.utils_infer import load_vocoder as f5_load_vocoder

    load_vocoder_kwargs = {"vocoder_name": "vocos", "device": device, "is_local": False}
    signature = inspect.signature(f5_load_vocoder)
    supported = {k: v for k, v in load_vocoder_kwargs.items() if k in signature.parameters}
    return f5_load_vocoder(**supported)


class SautiTTSService:
    """
    Loads the F5 DiT checkpoint once, resolves speakers from speakers.json + app/references/,
    and runs synthesis via safe_infer_process (sequential chunks — safe on GPU).
    """

    def __init__(self, checkpoint_path: str, speakers_json_path: str) -> None:
        self.speakers = self._load_speakers(Path(speakers_json_path))
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.vocoder = None
        self._load_model()

    def _load_model(self) -> None:
        from f5_tts.infer.utils_infer import load_model
        from f5_tts.model import DiT

        _patch_torchaudio_load_with_soundfile()
        logger.info("Loading inference model from %s", self.checkpoint_path)
        self.vocoder = _load_vocoder_compat(self.device)

        kwargs = {
            "model_cls": DiT,
            "model_cfg": dict(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                conv_layers=4,
            ),
            "ckpt_path": self.checkpoint_path,
            "mel_spec_type": "vocos",
            "vocab_file": "",
            "is_local": True,
            "use_ema": True,
            "device": self.device,
        }
        signature = inspect.signature(load_model)
        supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
        self.model = load_model(**supported)

    @staticmethod
    def _load_speakers(path: Path) -> dict[int, SpeakerSpec]:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)

        speakers: dict[int, SpeakerSpec] = {}
        for row in payload.get("speakers", []):
            sid = int(row["waxal_speaker_id"])
            ref_path = (path.parent / "references" / row["reference_wav"]).resolve()
            if not ref_path.is_file():
                raise RuntimeError(f"Missing reference audio for speaker {sid}: {ref_path}")
            speakers[sid] = SpeakerSpec(
                speaker=sid,
                label=str(row["label"]),
                ref_audio_path=str(ref_path),
                ref_text=str(row["ref_text"]),
            )

        declared = int(payload.get("speaker_count", len(speakers)))
        if len(speakers) != declared:
            raise RuntimeError(
                f"speakers.json speaker_count={declared} but loaded {len(speakers)} entries"
            )
        if not speakers:
            raise RuntimeError("No speakers in speakers.json")

        logger.info("Loaded %d speakers: %s", len(speakers), sorted(speakers.keys()))
        return speakers

    def list_speakers(self) -> list[SpeakerSpec]:
        return [self.speakers[k] for k in sorted(self.speakers)]

    def synthesize(
        self,
        text: str,
        speaker: int,
        *,
        audio_format: str = "wav",
    ) -> tuple[bytes, InferencePlan]:
        spec = self.speakers.get(speaker)
        if spec is None:
            raise SpeakerNotFoundError(f"Unsupported speaker: {speaker}")

        normalized = normalize_swahili_text(text)
        _patch_torchaudio_load_with_soundfile()

        plan = InferencePlan()
        try:
            audio, sample_rate, _ = safe_infer_process(
                spec.ref_audio_path,
                spec.ref_text,
                normalized,
                self.model,
                self.vocoder,
                nfe_step=plan.nfe_steps,
                cfg_strength=plan.cfg_strength,
                sway_sampling_coef=-1.0,
                speed=1.0,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001
            raise SynthesisError(str(exc)) from exc

        if audio is None:
            raise SynthesisError("Inference produced no audio.")

        out = waveform_to_audio_bytes(
            np.asarray(audio, dtype=np.float32),
            int(sample_rate),
            audio_format=audio_format,
        )
        return out, plan
