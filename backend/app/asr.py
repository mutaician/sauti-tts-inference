from __future__ import annotations

import logging
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Raised when ASR inference fails or produces no transcript."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    model: str
    language: str = "sw"


class SautiASRService:
    """Loads the Sauti ASR transformers pipeline once per warm container."""

    def __init__(
        self,
        *,
        model_path: str,
        model_id: str,
        chunk_length_s: int = 25,
        preload: bool = False,
        load_lock: threading.Lock | None = None,
    ) -> None:
        self.model_path = model_path
        self.model_id = model_id
        self.chunk_length_s = chunk_length_s
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = None
        self._load_lock = load_lock or threading.Lock()
        if preload:
            self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self.pipe is not None:
            return
        with self._load_lock:
            if self.pipe is not None:
                return
            self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise TranscriptionError(
                f"Missing ASR model files: {self.model_path}. "
                "Run the Modal model preparation command before deploying."
            )

        logger.info("Loading ASR model from %s", self.model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            chunk_length_s=self.chunk_length_s,
        )

    def transcribe(self, audio_bytes: bytes, *, suffix: str = ".webm") -> TranscriptionResult:
        if not audio_bytes:
            raise TranscriptionError("Audio file is empty.")

        self.ensure_loaded()
        if self.pipe is None:
            raise TranscriptionError("ASR model failed to initialize.")

        normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        with tempfile.NamedTemporaryFile(suffix=normalized_suffix) as audio_file:
            audio_file.write(audio_bytes)
            audio_file.flush()
            try:
                output = self.pipe(audio_file.name)
            except Exception as exc:  # noqa: BLE001
                raise TranscriptionError(str(exc)) from exc

        text = str(output.get("text", "")).strip() if isinstance(output, dict) else ""
        if not text:
            raise TranscriptionError("Transcription produced no text.")
        return TranscriptionResult(text=text, model=self.model_id)
