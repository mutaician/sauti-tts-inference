from __future__ import annotations

import io
import logging
import random
import re
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

SWAHILI_ONES = {
    0: "sifuri",
    1: "moja",
    2: "mbili",
    3: "tatu",
    4: "nne",
    5: "tano",
    6: "sita",
    7: "saba",
    8: "nane",
    9: "tisa",
}
SWAHILI_TENS = {
    10: "kumi",
    20: "ishirini",
    30: "thelathini",
    40: "arobaini",
    50: "hamsini",
    60: "sitini",
    70: "sabini",
    80: "themanini",
    90: "tisini",
}


def number_to_swahili(value: int) -> str:
    if value < 0:
        return "hasi " + number_to_swahili(-value)
    if value in SWAHILI_ONES:
        return SWAHILI_ONES[value]
    if value < 100:
        tens = (value // 10) * 10
        ones = value % 10
        if ones == 0:
            return SWAHILI_TENS[tens]
        return f"{SWAHILI_TENS[tens]} na {SWAHILI_ONES[ones]}"
    if value < 1000:
        hundreds = value // 100
        remainder = value % 100
        prefix = f"mia {SWAHILI_ONES[hundreds]}" if hundreds > 1 else "mia moja"
        if remainder == 0:
            return prefix
        return f"{prefix} na {number_to_swahili(remainder)}"
    if value < 1_000_000:
        thousands = value // 1000
        remainder = value % 1000
        prefix = f"elfu {number_to_swahili(thousands)}"
        if remainder == 0:
            return prefix
        return f"{prefix} na {number_to_swahili(remainder)}"
    return str(value)


def normalize_swahili_text(text: str) -> str:
    abbreviations = {
        r"\bDkt\.\b": "Daktari",
        r"\bBw\.\b": "Bwana",
        r"\bBi\.\b": "Bibi",
        r"\bProf\.\b": "Profesa",
        r"\bMh\.\b": "Mheshimiwa",
        r"\bn\.k\.": "na kadhalika",
        r"\bk\.m\.": "kwa mfano",
        r"\bKsh\.?\s?": "shilingi ",
        r"\bTsh\.?\s?": "shilingi ",
        r"\bUSD\s?": "dola za Kimarekani ",
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    def replace_number(match: re.Match[str]) -> str:
        num_str = match.group(0).replace(",", "")
        try:
            return number_to_swahili(int(num_str))
        except ValueError:
            try:
                parts = num_str.split(".")
                integer_part = number_to_swahili(int(parts[0]))
                decimal_part = " ".join(SWAHILI_ONES.get(int(d), d) for d in parts[1])
                return f"{integer_part} nukta {decimal_part}"
            except (IndexError, ValueError):
                return num_str

    text = re.sub(r"\d[\d,]*\.?\d*", replace_number, text)
    text = re.sub(r"[–—]", ", ", text)
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def waveform_to_audio_bytes(
    audio: np.ndarray | torch.Tensor,
    sample_rate: int,
    audio_format: str = "wav",
) -> bytes:
    if isinstance(audio, torch.Tensor):
        array = audio.detach().cpu().numpy()
    else:
        array = np.asarray(audio)

    if array.ndim == 2:
        if array.shape[0] == 1:
            array = array[0]
        else:
            array = np.mean(array, axis=0)

    audio_format = audio_format.lower()
    if audio_format == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, array.astype(np.float32), sample_rate, format="WAV")
        return buffer.getvalue()

    if audio_format == "mp3":
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file, tempfile.NamedTemporaryFile(
            suffix=".mp3"
        ) as mp3_file:
            sf.write(wav_file.name, array.astype(np.float32), sample_rate, format="WAV")
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    wav_file.name,
                    "-codec:a",
                    "libmp3lame",
                    "-q:a",
                    "2",
                    mp3_file.name,
                ],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="ignore").strip()
                raise RuntimeError(f"ffmpeg mp3 conversion failed: {stderr or 'unknown error'}")
            mp3_file.seek(0)
            return mp3_file.read()

    raise ValueError(f"Unsupported audio format: {audio_format}")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
