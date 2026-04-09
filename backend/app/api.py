from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .engine import SpeakerNotFoundError, SynthesisError, SautiTTSService
from .schemas import ErrorResponse, SpeakerResponse, SynthesizeRequest
from .security import RateLimitExceeded, SharedRateLimiter, client_ip


def create_app(
    *,
    service: SautiTTSService,
    rate_limiter: SharedRateLimiter,
    allowed_origins: list[str],
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

    @app.exception_handler(SpeakerNotFoundError)
    async def handle_unknown_speaker(request: Request, exc: SpeakerNotFoundError) -> JSONResponse:
        payload = ErrorResponse(code="unsupported_speaker", detail=str(exc))
        return JSONResponse(status_code=400, content=payload.model_dump())

    @app.exception_handler(SynthesisError)
    async def handle_synthesis_error(request: Request, exc: SynthesisError) -> JSONResponse:
        payload = ErrorResponse(code="synthesis_failed", detail=str(exc))
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
            for item in service.list_speakers()
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
        audio_bytes, plan = service.synthesize(payload.text, payload.speaker, audio_format=format)
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

    return app
