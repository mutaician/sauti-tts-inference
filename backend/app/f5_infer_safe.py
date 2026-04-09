from __future__ import annotations

import numpy as np
import torch
import torchaudio
import tqdm

import f5_tts.infer.utils_infer as ui
from f5_tts.model.utils import convert_char_to_pinyin


def safe_infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
    f5_short_text_stretch: bool = True,
    min_gen_audio_sec: float = 0.35,
):
    tsr = ui.target_sample_rate
    hop_length = ui.hop_length

    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != tsr:
        resampler = torchaudio.transforms.Resample(sr, tsr)
        audio = resampler(audio)

    if device is None:
        device = next(model_obj.parameters()).device
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def _infer_basic(gen_text: str):
        local_speed = speed
        if f5_short_text_stretch and len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * tsr / hop_length)
        else:
            ref_text_len = max(1, len(ref_text.encode("utf-8")))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(
                ref_audio_len / ref_text_len * gen_text_len / local_speed
            )
            min_gen_mel = int(max(0.0, min_gen_audio_sec) * tsr / hop_length)
            gen_mel = duration - ref_audio_len
            if gen_mel < min_gen_mel:
                duration = ref_audio_len + min_gen_mel

        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            else:
                raise ValueError(f"Unknown mel_spec_type: {mel_spec_type}")
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms
            generated_wave = generated_wave.squeeze().cpu().numpy()

        return generated_wave, generated

    if streaming:
        iterator = progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches
        for gen_text in iterator:
            generated_wave, _ = _infer_basic(gen_text)
            for start in range(0, len(generated_wave), chunk_size):
                yield generated_wave[start : start + chunk_size], tsr
        return

    iterator = progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches
    for gen_text in iterator:
        generated_wave, generated = _infer_basic(gen_text)
        generated_waves.append(generated_wave)
        spectrograms.append(generated[0].cpu().numpy())

    if not generated_waves:
        yield None, tsr, None
        return

    if cross_fade_duration <= 0:
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for next_wave in generated_waves[1:]:
            prev_wave = final_wave
            cross_fade_samples = int(cross_fade_duration * tsr)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
            final_wave = np.concatenate(
                [
                    prev_wave[:-cross_fade_samples],
                    cross_faded_overlap,
                    next_wave[cross_fade_samples:],
                ]
            )

    combined_spectrogram = np.concatenate(spectrograms, axis=1)
    yield final_wave, tsr, combined_spectrogram


def safe_infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=None,
    show_info=print,
    progress=tqdm,
    target_rms=None,
    cross_fade_duration=None,
    nfe_step=None,
    cfg_strength=None,
    sway_sampling_coef=None,
    speed=None,
    fix_duration=None,
    device=None,
    f5_short_text_stretch: bool = True,
    min_gen_audio_sec: float = 0.35,
):
    mel_spec_type = mel_spec_type if mel_spec_type is not None else ui.mel_spec_type
    target_rms = target_rms if target_rms is not None else ui.target_rms
    cross_fade_duration = (
        cross_fade_duration if cross_fade_duration is not None else ui.cross_fade_duration
    )
    nfe_step = nfe_step if nfe_step is not None else ui.nfe_step
    cfg_strength = cfg_strength if cfg_strength is not None else ui.cfg_strength
    sway_sampling_coef = (
        sway_sampling_coef if sway_sampling_coef is not None else ui.sway_sampling_coef
    )
    speed = speed if speed is not None else ui.speed
    fix_duration = fix_duration if fix_duration is not None else ui.fix_duration

    audio, sr = torchaudio.load(ref_audio)
    seconds = float(audio.shape[-1]) / float(sr)
    if seconds <= 0:
        seconds = 0.01

    headroom = 22.0 - seconds
    if headroom <= 0:
        max_chars = 4096
    else:
        max_chars = int(len(ref_text.encode("utf-8")) / seconds * headroom * speed)
    max_chars = max(32, max_chars)
    gen_text_batches = ui.chunk_text(gen_text, max_chars=max_chars)

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    if not gen_text_batches:
        return None, ui.target_sample_rate, None

    return next(
        safe_infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
            f5_short_text_stretch=f5_short_text_stretch,
            min_gen_audio_sec=min_gen_audio_sec,
        )
    )
