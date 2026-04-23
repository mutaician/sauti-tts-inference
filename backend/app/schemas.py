from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SpeakerResponse(BaseModel):
    speaker: int
    label: str


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
