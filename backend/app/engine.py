from __future__ import annotations

import inspect
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .f5_infer_safe import safe_infer_process
from .utils import normalize_swahili_text, seed_everything, waveform_to_audio_bytes

logger = logging.getLogger(__name__)
TARGET_SAMPLE_RATE = 24000


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
    mode: str
    infer_backend: str
    use_long_text: bool
    max_chars_per_chunk: int
    long_text_cross_fade_sec: float
    chunk_cross_fade_duration: float | None
    f5_short_text_stretch: bool
    min_gen_audio_sec: float
    nfe_steps: int = 32
    cfg_strength: float = 2.0
    speed: float = 1.0


def _patch_torchaudio_load_with_soundfile() -> None:
    if getattr(torchaudio.load, "_soundfile_patch", False):
        return

    def _load_with_soundfile(path: str, *args, **kwargs):
        audio_np, sample_rate = sf.read(path, dtype="float32")
        if audio_np.ndim == 1:
            audio_np = audio_np[:, None]
        waveform = torch.from_numpy(audio_np.T.copy())
        return waveform, sample_rate

    _load_with_soundfile._soundfile_patch = True
    torchaudio.load = _load_with_soundfile


def _load_vocoder_compat(device: str):
    from f5_tts.infer.utils_infer import load_vocoder as f5_load_vocoder

    load_vocoder_kwargs = {"vocoder_name": "vocos", "device": device}
    signature = inspect.signature(f5_load_vocoder)
    supported = {
        key: value
        for key, value in load_vocoder_kwargs.items()
        if key in signature.parameters
    }
    return f5_load_vocoder(**supported)


def _apply_edge_fades(
    audio: np.ndarray,
    sample_rate: int,
    fade_in_sec: float = 0.005,
    fade_out_sec: float = 0.03,
) -> np.ndarray:
    if audio.ndim != 1 or len(audio) == 0:
        return audio

    result = audio.astype(np.float32, copy=True)
    fade_in_samples = min(len(result), max(0, int(sample_rate * fade_in_sec)))
    fade_out_samples = min(len(result), max(0, int(sample_rate * fade_out_sec)))

    if fade_in_samples > 1:
        result[:fade_in_samples] *= np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
    if fade_out_samples > 1:
        result[-fade_out_samples:] *= np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
    return result


def _smooth_quiet_pauses(
    audio: np.ndarray,
    sample_rate: int,
    rms_threshold: float = 0.01,
    min_silence_sec: float = 0.025,
    merge_gap_sec: float = 0.012,
    fade_sec: float = 0.01,
    zero_cross_search_sec: float = 0.008,
) -> np.ndarray:
    if audio.ndim != 1 or len(audio) == 0:
        return audio

    source = audio.astype(np.float32, copy=False)
    result = source.copy()
    rms_window = max(1, int(sample_rate * 0.01))
    min_silence_samples = max(1, int(sample_rate * min_silence_sec))
    merge_gap_samples = max(0, int(sample_rate * merge_gap_sec))
    fade_samples = max(1, int(sample_rate * fade_sec))
    zero_cross_search = max(1, int(sample_rate * zero_cross_search_sec))

    rms = np.sqrt(
        np.convolve(source**2, np.ones(rms_window, dtype=np.float32) / rms_window, mode="same")
    )
    quiet = rms < rms_threshold

    def nearest_zero_crossing(center: int) -> int:
        if len(source) < 2:
            return center

        lo = max(1, center - zero_cross_search)
        hi = min(len(source) - 1, center + zero_cross_search)
        window = source[lo - 1 : hi + 1]
        zero_crossings = np.where(np.signbit(window[:-1]) != np.signbit(window[1:]))[0]
        if zero_crossings.size:
            candidates = []
            for rel_idx in zero_crossings:
                left_idx = lo - 1 + int(rel_idx)
                right_idx = min(left_idx + 1, len(source) - 1)
                if abs(source[left_idx]) <= abs(source[right_idx]):
                    candidates.append(left_idx)
                else:
                    candidates.append(right_idx)
            return min(candidates, key=lambda idx: (abs(idx - center), abs(source[idx])))
        return lo + int(np.argmin(np.abs(source[lo : hi + 1])))

    segments: list[list[int]] = []
    start = None
    for idx, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = idx
        elif not is_quiet and start is not None:
            if idx - start >= min_silence_samples:
                segments.append([start, idx])
            start = None
    if start is not None and len(result) - start >= min_silence_samples:
        segments.append([start, len(result)])

    if not segments:
        return result

    merged: list[tuple[int, int]] = []
    current_start, current_end = segments[0]
    for next_start, next_end in segments[1:]:
        if next_start - current_end <= merge_gap_samples:
            current_end = next_end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))

    for start, end in merged:
        snapped_start = nearest_zero_crossing(start)
        snapped_end = nearest_zero_crossing(end)
        if snapped_end <= snapped_start:
            snapped_start, snapped_end = start, end

        fade_in_start = max(0, snapped_start - fade_samples)
        fade_out_end = min(len(result), snapped_end + fade_samples)

        if snapped_start > fade_in_start:
            result[fade_in_start:snapped_start] *= np.linspace(
                1.0, 0.0, snapped_start - fade_in_start, dtype=np.float32
            )
        result[snapped_start:snapped_end] = 0.0
        if fade_out_end > snapped_end:
            result[snapped_end:fade_out_end] *= np.linspace(
                0.0, 1.0, fade_out_end - snapped_end, dtype=np.float32
            )

    return result


def _repair_quiet_micro_clicks(
    audio: np.ndarray,
    sample_rate: int,
    rms_threshold: float = 0.015,
    spike_threshold: float = 0.018,
    max_click_sec: float = 0.0002,
    search_window_sec: float = 0.004,
) -> np.ndarray:
    if audio.ndim != 1 or len(audio) < 5:
        return audio

    result = audio.astype(np.float32, copy=True)
    rms_window = max(1, int(sample_rate * 0.005))
    max_click_samples = max(1, int(sample_rate * max_click_sec))
    search_window = max(2, int(sample_rate * search_window_sec))

    rms = np.sqrt(
        np.convolve(result**2, np.ones(rms_window, dtype=np.float32) / rms_window, mode="same")
    )
    quiet = rms < rms_threshold
    diff_prev = np.abs(result[1:-1] - result[:-2])
    diff_next = np.abs(result[1:-1] - result[2:])
    interp_dev = np.abs(result[1:-1] - 0.5 * (result[:-2] + result[2:]))
    candidate_core = (
        quiet[1:-1]
        & ((diff_prev > spike_threshold) | (diff_next > spike_threshold))
        & (interp_dev > spike_threshold * 0.6)
    )
    candidate_idx = np.where(candidate_core)[0] + 1
    if candidate_idx.size == 0:
        return result

    runs: list[tuple[int, int]] = []
    start = int(candidate_idx[0])
    prev = start
    for idx in candidate_idx[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev + 1))
        start = idx
        prev = idx
    runs.append((start, prev + 1))

    for start, end in runs:
        if end - start > max_click_samples:
            continue
        left = max(0, start - search_window)
        right = min(len(result), end + search_window)

        left_candidates = np.where(~quiet[left:start])[0]
        right_candidates = np.where(~quiet[end:right])[0]
        anchor_left = left + int(left_candidates[-1]) if left_candidates.size else max(0, start - 1)
        anchor_right = end + int(right_candidates[0]) if right_candidates.size else min(len(result) - 1, end)
        if anchor_right <= anchor_left:
            continue

        repair_start = max(anchor_left + 1, start)
        repair_end = min(anchor_right, end)
        if repair_end <= repair_start:
            continue

        interp = np.linspace(
            result[anchor_left],
            result[anchor_right],
            anchor_right - anchor_left + 1,
            dtype=np.float32,
        )
        result[repair_start:repair_end] = interp[
            repair_start - anchor_left : repair_end - anchor_left
        ]

    return result


def _trim_chunk_edges(
    audio: np.ndarray,
    sample_rate: int,
    rms_threshold: float = 0.01,
    window_sec: float = 0.01,
    pad_sec: float = 0.035,
    min_trim_sec: float = 0.02,
) -> np.ndarray:
    if audio.ndim != 1 or len(audio) == 0:
        return audio

    source = audio.astype(np.float32, copy=False)
    window = max(1, int(sample_rate * window_sec))
    pad = max(0, int(sample_rate * pad_sec))
    min_trim = max(0, int(sample_rate * min_trim_sec))

    rms = np.sqrt(
        np.convolve(source**2, np.ones(window, dtype=np.float32) / window, mode="same")
    )
    active = rms > rms_threshold
    if not np.any(active):
        return source.copy()

    start = int(np.argmax(active))
    end = len(source) - int(np.argmax(active[::-1]))
    start = max(0, start - pad)
    end = min(len(source), end + pad)
    if start < min_trim:
        start = 0
    if len(source) - end < min_trim:
        end = len(source)
    if end <= start:
        return source.copy()
    return source[start:end].copy()


def _gate_quiet_regions(
    audio: np.ndarray,
    sample_rate: int,
    rms_threshold: float = 0.017,
    min_silence_sec: float = 0.02,
    merge_gap_sec: float = 0.015,
    fade_sec: float = 0.012,
) -> np.ndarray:
    if audio.ndim != 1 or len(audio) == 0:
        return audio

    result = audio.astype(np.float32, copy=True)
    rms_window = max(1, int(sample_rate * 0.01))
    min_silence_samples = max(1, int(sample_rate * min_silence_sec))
    merge_gap_samples = max(0, int(sample_rate * merge_gap_sec))
    fade_samples = max(1, int(sample_rate * fade_sec))

    rms = np.sqrt(
        np.convolve(result**2, np.ones(rms_window, dtype=np.float32) / rms_window, mode="same")
    )
    quiet = rms < rms_threshold

    segments: list[list[int]] = []
    start = None
    for idx, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = idx
        elif not is_quiet and start is not None:
            if idx - start >= min_silence_samples:
                segments.append([start, idx])
            start = None
    if start is not None and len(result) - start >= min_silence_samples:
        segments.append([start, len(result)])
    if not segments:
        return result

    merged: list[tuple[int, int]] = []
    current_start, current_end = segments[0]
    for next_start, next_end in segments[1:]:
        if next_start - current_end <= merge_gap_samples:
            current_end = next_end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))

    for start, end in merged:
        fade_in_start = max(0, start - fade_samples)
        fade_out_end = min(len(result), end + fade_samples)
        if start > fade_in_start:
            result[fade_in_start:start] *= np.linspace(
                1.0, 0.0, start - fade_in_start, dtype=np.float32
            )
        result[start:end] = 0.0
        if fade_out_end > end:
            result[end:fade_out_end] *= np.linspace(
                0.0, 1.0, fade_out_end - end, dtype=np.float32
            )

    return result


class SautiInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str | None = None,
        use_ema: bool = True,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.vocab_path = vocab_path
        self.use_ema = use_ema
        self.model = None
        self.vocoder = None
        self._load_model()

    def _load_model(self) -> None:
        from f5_tts.infer.utils_infer import load_model
        from f5_tts.model import DiT

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
            "vocab_file": self.vocab_path or "",
            "is_local": True,
            "use_ema": self.use_ema,
            "device": self.device,
        }
        signature = inspect.signature(load_model)
        supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
        self.model = load_model(**supported)

    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        *,
        nfe_steps: int,
        cfg_strength: float,
        speed: float,
        seed: int | None,
        infer_backend: str,
        f5_short_text_stretch: bool,
        cross_fade_duration: float | None,
        min_gen_audio_sec: float,
    ) -> np.ndarray:
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        _patch_torchaudio_load_with_soundfile()
        normalized = normalize_swahili_text(text)
        if seed is not None:
            seed_everything(seed)

        ref_audio, normalized_ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
        start_time = time.time()

        if infer_backend == "safe":
            generated_audio, sample_rate, _ = safe_infer_process(
                ref_audio=ref_audio,
                ref_text=normalized_ref_text,
                gen_text=normalized,
                model_obj=self.model,
                vocoder=self.vocoder,
                nfe_step=nfe_steps,
                cfg_strength=cfg_strength,
                speed=speed,
                device=self.device,
                f5_short_text_stretch=f5_short_text_stretch,
                cross_fade_duration=cross_fade_duration,
                min_gen_audio_sec=min_gen_audio_sec,
            )
        elif infer_backend == "upstream":
            infer_kwargs = {
                "ref_audio": ref_audio,
                "ref_text": normalized_ref_text,
                "gen_text": normalized,
                "model_obj": self.model,
                "vocoder": self.vocoder,
                "nfe_step": nfe_steps,
                "cfg_strength": cfg_strength,
                "speed": speed,
            }
            if cross_fade_duration is not None:
                infer_kwargs["cross_fade_duration"] = cross_fade_duration
            generated_audio, sample_rate, _ = infer_process(**infer_kwargs)
        else:
            raise ValueError(f"Unknown inference backend: {infer_backend}")

        if generated_audio is None:
            raise SynthesisError("Inference produced no audio.")

        generated_audio = _smooth_quiet_pauses(generated_audio, sample_rate)
        generated_audio = _repair_quiet_micro_clicks(generated_audio, sample_rate)
        generated_audio = _gate_quiet_regions(generated_audio, sample_rate)
        generated_audio = _apply_edge_fades(generated_audio, sample_rate)

        duration = len(generated_audio) / sample_rate if sample_rate else 0
        elapsed = time.time() - start_time
        logger.info("Generated %.2fs audio in %.2fs", duration, elapsed)
        return generated_audio.astype(np.float32)

    def generate_long(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        *,
        max_chars_per_chunk: int,
        cross_fade_sec: float,
        **kwargs,
    ) -> np.ndarray:
        chunks = self._split_text(text, max_chars_per_chunk)
        if not chunks:
            raise SynthesisError("No text chunks were produced for synthesis.")

        audio_chunks = []
        base_seed = kwargs.get("seed")
        for idx, chunk in enumerate(chunks):
            chunk_kwargs = dict(kwargs)
            if base_seed is not None:
                chunk_kwargs["seed"] = base_seed + idx
            audio = self.generate(
                text=chunk,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                **chunk_kwargs,
            )
            audio_chunks.append(_trim_chunk_edges(audio, TARGET_SAMPLE_RATE))

        if len(audio_chunks) == 1:
            result = audio_chunks[0]
        else:
            result = self._stitch_with_pause(
                audio_chunks,
                sample_rate=TARGET_SAMPLE_RATE,
                pause_sec=min(max(cross_fade_sec * 0.45, 0.06), 0.12),
                fade_sec=min(cross_fade_sec * 0.2, 0.03),
            )

        result = _smooth_quiet_pauses(result, TARGET_SAMPLE_RATE)
        result = _repair_quiet_micro_clicks(result, TARGET_SAMPLE_RATE)
        result = _gate_quiet_regions(result, TARGET_SAMPLE_RATE)
        result = _apply_edge_fades(result, TARGET_SAMPLE_RATE)
        return result.astype(np.float32)

    @staticmethod
    def _split_text(text: str, max_chars: int) -> list[str]:
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in ".!?;" and len(current) >= 20:
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())

        chunks = []
        current_chunk = ""
        for sentence in sentences or [text]:
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    @staticmethod
    def _stitch_with_pause(
        chunks: list[np.ndarray],
        *,
        sample_rate: int,
        pause_sec: float,
        fade_sec: float,
    ) -> np.ndarray:
        if len(chunks) == 1:
            return chunks[0]

        pause_samples = max(0, int(sample_rate * pause_sec))
        fade_samples = max(1, int(sample_rate * min(fade_sec, 0.04)))
        silence = np.zeros(pause_samples, dtype=np.float32)

        result = chunks[0].astype(np.float32, copy=True)
        for chunk in chunks[1:]:
            next_chunk = chunk.astype(np.float32, copy=True)
            if len(result) >= fade_samples:
                result[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            if len(next_chunk) >= fade_samples:
                next_chunk[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            pieces = [result]
            if pause_samples:
                pieces.append(silence)
            pieces.append(next_chunk)
            result = np.concatenate(pieces)
        return result


class SautiTTSService:
    def __init__(self, checkpoint_path: str, speakers_json_path: str) -> None:
        self.speakers = self._load_speakers(Path(speakers_json_path))
        self.engine = SautiInferenceEngine(checkpoint_path=checkpoint_path)

    @staticmethod
    def _load_speakers(path: Path) -> dict[int, SpeakerSpec]:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)

        speakers: dict[int, SpeakerSpec] = {}
        for row in payload.get("speakers", []):
            speaker = int(row["waxal_speaker_id"])
            speakers[speaker] = SpeakerSpec(
                speaker=speaker,
                label=row["label"],
                ref_audio_path=str((path.parent / "references" / row["reference_wav"]).resolve()),
                ref_text=row["ref_text"],
            )
        expected = {1, 4, 6}
        if set(speakers) != expected:
            raise RuntimeError(f"Expected speaker set {expected}, found {set(speakers)}")
        return speakers

    def list_speakers(self) -> list[SpeakerSpec]:
        return [self.speakers[key] for key in sorted(self.speakers)]

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

        plan = self._choose_plan(text)
        kwargs = {
            "nfe_steps": plan.nfe_steps,
            "cfg_strength": plan.cfg_strength,
            "speed": plan.speed,
            "seed": None,
            "infer_backend": plan.infer_backend,
            "f5_short_text_stretch": plan.f5_short_text_stretch,
            "cross_fade_duration": plan.chunk_cross_fade_duration,
            "min_gen_audio_sec": plan.min_gen_audio_sec,
        }

        try:
            if plan.use_long_text:
                audio = self.engine.generate_long(
                    text=text,
                    ref_audio_path=spec.ref_audio_path,
                    ref_text=spec.ref_text,
                    max_chars_per_chunk=plan.max_chars_per_chunk,
                    cross_fade_sec=plan.long_text_cross_fade_sec,
                    **kwargs,
                )
            else:
                audio = self.engine.generate(
                    text=text,
                    ref_audio_path=spec.ref_audio_path,
                    ref_text=spec.ref_text,
                    **kwargs,
                )
        except Exception as exc:  # noqa: BLE001
            raise SynthesisError(str(exc)) from exc

        return waveform_to_audio_bytes(audio, TARGET_SAMPLE_RATE, audio_format=audio_format), plan

    @staticmethod
    def _choose_plan(text: str) -> InferencePlan:
        normalized = normalize_swahili_text(text)
        byte_length = len(normalized.encode("utf-8"))
        sentence_count = len([chunk for chunk in re.split(r"[.!?;]+", normalized) if chunk.strip()])

        if byte_length < 12:
            return InferencePlan(
                mode="short",
                infer_backend="safe",
                use_long_text=False,
                max_chars_per_chunk=140,
                long_text_cross_fade_sec=0.2,
                chunk_cross_fade_duration=0.1,
                f5_short_text_stretch=True,
                min_gen_audio_sec=0.35,
            )

        if byte_length > 160 or sentence_count > 1:
            max_chars = 120 if byte_length > 280 else 140
            return InferencePlan(
                mode="long",
                infer_backend="safe",
                use_long_text=True,
                max_chars_per_chunk=max_chars,
                long_text_cross_fade_sec=0.2,
                chunk_cross_fade_duration=0.1,
                f5_short_text_stretch=True,
                min_gen_audio_sec=0.35,
            )

        return InferencePlan(
            mode="normal",
            infer_backend="upstream",
            use_long_text=False,
            max_chars_per_chunk=140,
            long_text_cross_fade_sec=0.2,
            chunk_cross_fade_duration=0.15,
            f5_short_text_stretch=True,
            min_gen_audio_sec=0.35,
        )
