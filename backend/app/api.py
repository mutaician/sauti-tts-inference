from __future__ import annotations

from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .asr import SautiASRService, TranscriptionError
from .engine import SpeakerNotFoundError, SynthesisError, SautiTTSService
from .schemas import (
    ErrorResponse,
    SpeakerResponse,
    StsEndSessionRequest,
    StsWarmupResponse,
    SynthesizeRequest,
    TranscribeResponse,
)
from .security import RateLimitExceeded, SharedRateLimiter, client_ip
from .sts import StsError, StsOrchestrator, format_sse

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".flac"}


def create_app(
    *,
    tts_service: SautiTTSService,
    asr_service: SautiASRService,
    sts_orchestrator: StsOrchestrator,
    rate_limiter: SharedRateLimiter,
    allowed_origins: list[str],
    max_upload_bytes: int = 25 * 1024 * 1024,
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        max_age=86400,
    )

    @app.exception_handler(StsError)
    async def handle_sts_error(request: Request, exc: StsError) -> JSONResponse:
        payload = ErrorResponse(code="sts_failed", detail=str(exc))
        return JSONResponse(status_code=502, content=payload.model_dump())

    @app.exception_handler(SpeakerNotFoundError)
    async def handle_unknown_speaker(request: Request, exc: SpeakerNotFoundError) -> JSONResponse:
        payload = ErrorResponse(code="unsupported_speaker", detail=str(exc))
        return JSONResponse(status_code=400, content=payload.model_dump())

    @app.exception_handler(SynthesisError)
    async def handle_synthesis_error(request: Request, exc: SynthesisError) -> JSONResponse:
        payload = ErrorResponse(code="synthesis_failed", detail=str(exc))
        return JSONResponse(status_code=502, content=payload.model_dump())

    @app.exception_handler(TranscriptionError)
    async def handle_transcription_error(request: Request, exc: TranscriptionError) -> JSONResponse:
        payload = ErrorResponse(code="transcription_failed", detail=str(exc))
        return JSONResponse(status_code=502, content=payload.model_dump())

    @app.exception_handler(RateLimitExceeded)
    async def handle_rate_limit(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        payload = ErrorResponse(code="rate_limited", detail="Too many requests. Please try again soon.")
        return JSONResponse(
            status_code=429,
            content=payload.model_dump(),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        payload = ErrorResponse(code="invalid_request", detail="Request validation failed.")
        return JSONResponse(status_code=422, content={**payload.model_dump(), "errors": exc.errors()})

    @app.get("/v1/speakers", response_model=list[SpeakerResponse])
    async def list_speakers() -> list[SpeakerResponse]:
        return [
            SpeakerResponse(speaker=item.speaker, label=item.label)
            for item in tts_service.list_speakers()
        ]

    @app.post(
        "/v1/synthesize",
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
        },
    )
    async def synthesize(
        payload: SynthesizeRequest,
        request: Request,
        format: Literal["wav", "mp3"] = "wav",
    ) -> Response:
        rate_limiter.check(client_ip(request))
        audio_bytes, plan = tts_service.synthesize(payload.text, payload.speaker, audio_format=format)
        media_type = "audio/mpeg" if format == "mp3" else "audio/wav"
        filename = f"sauti-tts-{payload.speaker}.{format}"
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-TTS-Speaker": str(payload.speaker),
                "X-TTS-Mode": plan.mode,
            },
        )

    @app.post(
        "/v1/transcribe",
        response_model=TranscribeResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
        },
    )
    async def transcribe(request: Request, file: UploadFile = File(...)) -> TranscribeResponse:
        rate_limiter.check(client_ip(request))
        filename = file.filename or "audio.webm"
        suffix = f".{filename.rsplit('.', 1)[-1].lower()}" if "." in filename else ".webm"
        if suffix not in ALLOWED_AUDIO_EXTENSIONS:
            payload = ErrorResponse(
                code="unsupported_audio",
                detail="Upload an audio file in mp3, mp4, m4a, wav, webm, ogg, or flac format.",
            )
            return JSONResponse(status_code=400, content=payload.model_dump())

        audio_bytes = await file.read(max_upload_bytes + 1)
        if len(audio_bytes) > max_upload_bytes:
            payload = ErrorResponse(
                code="audio_too_large",
                detail=f"Audio upload is too large. Limit is {max_upload_bytes // (1024 * 1024)} MB.",
            )
            return JSONResponse(status_code=413, content=payload.model_dump())

        result = asr_service.transcribe(audio_bytes, suffix=suffix)
        return TranscribeResponse(text=result.text, model=result.model, language=result.language)

    @app.post(
        "/v1/sts/warmup",
        response_model=StsWarmupResponse,
        responses={
            429: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
        },
    )
    async def sts_warmup(request: Request) -> StsWarmupResponse:
        rate_limiter.check(client_ip(request))
        result = sts_orchestrator.warmup()
        return StsWarmupResponse(
            ready=result.ready,
            asr_loaded=result.asr_loaded,
            tts_loaded=result.tts_loaded,
            elapsed_ms=result.elapsed_ms,
        )

    @app.post("/v1/sts/turn")
    async def sts_turn(
        request: Request,
        session_id: Annotated[str | None, Form()] = None,
        text: Annotated[str | None, Form()] = None,
        file: UploadFile | None = File(None),
    ) -> StreamingResponse:
        rate_limiter.check(client_ip(request))

        audio_bytes: bytes | None = None
        audio_suffix = ".webm"
        if file is not None and file.filename:
            filename = file.filename
            audio_suffix = f".{filename.rsplit('.', 1)[-1].lower()}" if "." in filename else ".webm"
            if audio_suffix not in ALLOWED_AUDIO_EXTENSIONS:
                payload = ErrorResponse(
                    code="unsupported_audio",
                    detail="Upload an audio file in mp3, mp4, m4a, wav, webm, ogg, or flac format.",
                )
                return JSONResponse(status_code=400, content=payload.model_dump())

            audio_bytes = await file.read(max_upload_bytes + 1)
            if len(audio_bytes) > max_upload_bytes:
                payload = ErrorResponse(
                    code="audio_too_large",
                    detail=f"Audio upload is too large. Limit is {max_upload_bytes // (1024 * 1024)} MB.",
                )
                return JSONResponse(status_code=413, content=payload.model_dump())

        cleaned_text = text.strip() if text else None

        def event_stream():
            for item in sts_orchestrator.iter_turn_events(
                session_id=session_id,
                text=cleaned_text,
                audio_bytes=audio_bytes,
                audio_suffix=audio_suffix,
            ):
                yield format_sse(item["event"], item["data"])

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post(
        "/v1/sts/end",
        responses={
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
        },
    )
    async def sts_end_session(payload: StsEndSessionRequest, request: Request) -> dict[str, bool]:
        rate_limiter.check(client_ip(request))
        sts_orchestrator.end_session(payload.session_id.strip())
        return {"ended": True}

    return app
