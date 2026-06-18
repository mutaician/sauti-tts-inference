from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI

from .asr import SautiASRService, TranscriptionError
from .engine import SautiTTSService, SynthesisError
from .sts_prompt import STS_SPEAKER_ID, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MAX_TOKENS = 512
MAX_HISTORY_MESSAGES = 40


class StsError(RuntimeError):
    """Raised when STS orchestration fails."""


@dataclass(frozen=True)
class StageResult:
    text: str
    duration_ms: int


@dataclass(frozen=True)
class WarmupResult:
    ready: bool
    asr_loaded: bool
    tts_loaded: bool
    elapsed_ms: int


@dataclass(frozen=True)
class TurnResult:
    session_id: str
    turn_index: int
    input_mode: Literal["audio", "text"]
    user_text: str
    assistant_text: str
    asr: StageResult | None
    llm: StageResult
    tts: StageResult
    audio_base64: str
    audio_format: str


def strip_for_tts(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", None) or ""
    return str(content).strip()


def format_sse(event: str, data: dict[str, Any]) -> str:
    # Padding helps some proxies flush each stage as it completes.
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n: .\n\n"


class StsOrchestrator:
    """Chains ASR → DeepSeek → TTS using existing inference services."""

    def __init__(
        self,
        *,
        asr_service: SautiASRService,
        tts_service: SautiTTSService,
    ) -> None:
        self.asr_service = asr_service
        self.tts_service = tts_service
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise StsError("DEEPSEEK_API_KEY is not configured.")
            self._client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        return self._client

    def warmup(self) -> WarmupResult:
        start = time.perf_counter()
        self.asr_service.ensure_loaded()
        self.tts_service.ensure_loaded()
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return WarmupResult(
            ready=True,
            asr_loaded=self.asr_service.pipe is not None,
            tts_loaded=self.tts_service.model is not None,
            elapsed_ms=elapsed_ms,
        )

    def resolve_session(self, session_id: str | None) -> str:
        sid = (session_id or "").strip() or str(uuid.uuid4())
        if sid not in self._sessions:
            self._sessions[sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
        return sid

    def end_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _trim_history(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if len(messages) <= MAX_HISTORY_MESSAGES:
            return messages
        system = messages[0]
        tail = messages[-(MAX_HISTORY_MESSAGES - 1) :]
        return [system, *tail]

    def _complete_llm(self, messages: list[dict[str, str]]) -> str:
        client = self._get_client()
        last_error = "LLM returned an empty reply."

        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=MAX_TOKENS,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except Exception as exc:  # noqa: BLE001
                raise StsError(f"LLM request failed: {exc}") from exc

            choice = response.choices[0]
            content = _extract_message_text(choice.message)
            if content:
                return content

            finish_reason = getattr(choice, "finish_reason", None)
            logger.warning(
                "Empty LLM reply (attempt %s/%s, finish_reason=%s)",
                attempt + 1,
                2,
                finish_reason,
            )
            last_error = (
                f"LLM returned an empty reply (finish_reason={finish_reason})."
                if finish_reason
                else last_error
            )

        raise StsError(last_error)

    def iter_turn_events(
        self,
        *,
        session_id: str | None,
        text: str | None = None,
        audio_bytes: bytes | None = None,
        audio_suffix: str = ".webm",
    ) -> Iterator[dict[str, Any]]:
        has_text = bool(text and text.strip())
        has_audio = bool(audio_bytes)
        if has_text and has_audio:
            yield {"event": "error", "data": {"detail": "Provide either text or audio, not both."}}
            return
        if not has_text and not has_audio:
            yield {"event": "error", "data": {"detail": "Provide text or an audio recording."}}
            return

        sid = self.resolve_session(session_id)
        messages = self._sessions[sid]

        asr_stage: StageResult | None = None
        try:
            if has_audio:
                asr_start = time.perf_counter()
                asr_result = self.asr_service.transcribe(audio_bytes, suffix=audio_suffix)
                user_text = asr_result.text.strip()
                asr_stage = StageResult(
                    text=user_text,
                    duration_ms=int((time.perf_counter() - asr_start) * 1000),
                )
                input_mode: Literal["audio", "text"] = "audio"
                yield {
                    "event": "asr_done",
                    "data": {
                        "session_id": sid,
                        "text": user_text,
                        "duration_ms": asr_stage.duration_ms,
                    },
                }
            else:
                user_text = text.strip()  # type: ignore[union-attr]
                input_mode = "text"

            if not user_text:
                yield {"event": "error", "data": {"detail": "No user text to process."}}
                return

            messages.append({"role": "user", "content": user_text})

            llm_start = time.perf_counter()
            try:
                assistant_text = self._complete_llm(messages)
            except StsError as exc:
                messages.pop()
                yield {"event": "error", "data": {"detail": str(exc)}}
                return

            llm_stage = StageResult(
                text=assistant_text,
                duration_ms=int((time.perf_counter() - llm_start) * 1000),
            )
            yield {
                "event": "llm_done",
                "data": {
                    "session_id": sid,
                    "text": assistant_text,
                    "duration_ms": llm_stage.duration_ms,
                },
            }

            messages.append({"role": "assistant", "content": assistant_text})
            self._sessions[sid] = self._trim_history(messages)

            tts_text = strip_for_tts(assistant_text)
            if not tts_text:
                yield {"event": "error", "data": {"detail": "Reply is empty after TTS normalization."}}
                return

            tts_start = time.perf_counter()
            audio_out, _plan = self.tts_service.synthesize(
                tts_text,
                STS_SPEAKER_ID,
                audio_format="mp3",
            )
            tts_stage = StageResult(
                text=tts_text,
                duration_ms=int((time.perf_counter() - tts_start) * 1000),
            )
            turn_index = sum(1 for item in messages if item["role"] == "assistant")

            result = TurnResult(
                session_id=sid,
                turn_index=turn_index,
                input_mode=input_mode,
                user_text=user_text,
                assistant_text=assistant_text,
                asr=asr_stage,
                llm=llm_stage,
                tts=tts_stage,
                audio_base64=base64.b64encode(audio_out).decode("ascii"),
                audio_format="mp3",
            )

            yield {
                "event": "tts_done",
                "data": {
                    "duration_ms": tts_stage.duration_ms,
                    "audio_base64": result.audio_base64,
                    "audio_format": result.audio_format,
                },
            }
            yield {
                "event": "done",
                "data": {
                    "session_id": result.session_id,
                    "turn_index": result.turn_index,
                    "input_mode": result.input_mode,
                    "user_text": result.user_text,
                    "assistant_text": result.assistant_text,
                    "asr": (
                        {"text": asr_stage.text, "duration_ms": asr_stage.duration_ms}
                        if asr_stage
                        else None
                    ),
                    "llm": {"text": llm_stage.text, "duration_ms": llm_stage.duration_ms},
                    "tts": {"text": tts_stage.text, "duration_ms": tts_stage.duration_ms},
                    "audio_base64": result.audio_base64,
                    "audio_format": result.audio_format,
                },
            }
        except TranscriptionError as exc:
            yield {"event": "error", "data": {"detail": str(exc)}}
        except SynthesisError as exc:
            yield {"event": "error", "data": {"detail": str(exc)}}
        except Exception as exc:  # noqa: BLE001
            logger.exception("STS turn failed")
            yield {"event": "error", "data": {"detail": str(exc)}}
