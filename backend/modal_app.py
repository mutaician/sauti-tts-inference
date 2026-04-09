from __future__ import annotations

import os
import sys

import modal

APP_NAME = "sauti-tts-inference"
PYTHON_VERSION = "3.11"
GPU_TYPE = "T4"
F5_TTS_REPO = "https://github.com/SWivid/F5-TTS.git"
F5_TTS_REF = "main"
F5_TTS_DIR = "/opt/F5-TTS"
CHECKPOINT_VOLUME_NAME = "sauti-tts-ckpts"
RATE_LIMIT_DICT_NAME = "sauti-tts-inference-rate-limit"
REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_CHECKPOINT_DIR = "/vol/ckpts"
REMOTE_MODEL_PATH = f"{REMOTE_CHECKPOINT_DIR}/sauti_tts_multi/model_last.pt"
REMOTE_SPEAKERS_JSON = f"{REMOTE_PROJECT_ROOT}/app/speakers.json"

app = modal.App(APP_NAME)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=False)
rate_limit_store = modal.Dict.from_name(RATE_LIMIT_DICT_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "fastapi",
        "modal",
        "numpy",
        "rich",
        "soundfile",
        "torch",
        "torchaudio",
        "tqdm",
    )
    .run_commands(
        f"git clone --depth 1 --branch {F5_TTS_REF} {F5_TTS_REPO} {F5_TTS_DIR}",
        f"python -m pip install -e {F5_TTS_DIR}",
    )
    .add_local_dir(
        ".",
        remote_path=REMOTE_PROJECT_ROOT,
        ignore=[".venv", "__pycache__", ".pytest_cache", "node_modules", "dist"],
    )
)


def _allowed_origins() -> list[str]:
    raw = os.environ.get("FRONTEND_ORIGINS", "http://localhost:5173")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _ensure_project_on_path() -> None:
    if REMOTE_PROJECT_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_ROOT)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 10,
    volumes={REMOTE_CHECKPOINT_DIR: checkpoint_volume},
    secrets=[modal.Secret.from_name("sauti-tts-inference-env")],
)
class InferenceApi:
    @modal.enter()
    def load(self) -> None:
        _ensure_project_on_path()
        from app.engine import SautiTTSService
        from app.security import SharedRateLimiter
        from app.utils import setup_logging

        os.chdir(REMOTE_PROJECT_ROOT)
        setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
        limit = int(os.environ.get("RATE_LIMIT_REQUESTS", "10"))
        window_seconds = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
        self.service = SautiTTSService(
            checkpoint_path=REMOTE_MODEL_PATH,
            speakers_json_path=REMOTE_SPEAKERS_JSON,
        )
        self.rate_limiter = SharedRateLimiter(
            store=rate_limit_store,
            limit=limit,
            window_seconds=window_seconds,
        )

    @modal.asgi_app()
    def asgi_app(self):
        _ensure_project_on_path()
        from app.api import create_app

        return create_app(
            service=self.service,
            rate_limiter=self.rate_limiter,
            allowed_origins=_allowed_origins(),
        )


@app.local_entrypoint()
def main() -> None:
    print(
        "Deploy with: modal deploy backend/modal_app.py",
        flush=True,
    )
