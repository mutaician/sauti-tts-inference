from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SpeakerResponse(BaseModel):
    speaker: int
    label: str


class TranscribeResponse(BaseModel):
    text: str
    model: str
    language: str


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    speaker: int

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text must not be empty.")
        return cleaned


class ErrorResponse(BaseModel):
    code: str
    detail: str


class StsStageResponse(BaseModel):
    text: str
    duration_ms: int


class StsWarmupResponse(BaseModel):
    ready: bool
    asr_loaded: bool
    tts_loaded: bool
    elapsed_ms: int


class StsTurnResponse(BaseModel):
    session_id: str
    turn_index: int
    input_mode: Literal["audio", "text"]
    user_text: str
    assistant_text: str
    asr: StsStageResponse | None = None
    llm: StsStageResponse
    tts: StsStageResponse
    audio_base64: str
    audio_format: str = "mp3"


class StsEndSessionRequest(BaseModel):
    session_id: str = Field(min_length=1)
