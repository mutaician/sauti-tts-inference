from __future__ import annotations

import os
import sys
import threading

import modal

APP_NAME = "sauti-inference"
PYTHON_VERSION = "3.11"
GPU_TYPE = "T4"
F5_TTS_REPO = "https://github.com/SWivid/F5-TTS.git"
F5_TTS_REF = "main"
F5_TTS_DIR = "/opt/F5-TTS"
INFERENCE_VOLUME_NAME = "sauti-inference-models"
RATE_LIMIT_DICT_NAME = "sauti-tts-inference-rate-limit"
REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_MODEL_STORE_DIR = "/vol/models"
TTS_MODEL_ID = "msingiai/sauti-tts"
ASR_MODEL_ID = "msingiai/sauti-asr"
REMOTE_TTS_MODEL_DIR = f"{REMOTE_MODEL_STORE_DIR}/tts/sauti_tts_multi"
REMOTE_TTS_MODEL_PATH = f"{REMOTE_TTS_MODEL_DIR}/model_last.pt"
REMOTE_ASR_MODEL_DIR = f"{REMOTE_MODEL_STORE_DIR}/asr/msingiai-sauti-asr"
REMOTE_SPEAKERS_JSON = f"{REMOTE_PROJECT_ROOT}/app/speakers.json"

app = modal.App(APP_NAME)
inference_volume = modal.Volume.from_name(INFERENCE_VOLUME_NAME, create_if_missing=True)
rate_limit_store = modal.Dict.from_name(RATE_LIMIT_DICT_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "ffmpeg", "libsndfile1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .pip_install(
        "accelerate",
        "fastapi",
        "modal",
        "numpy",
        "openai",
        "python-multipart",
        "rich",
        "safetensors",
        "soundfile",
        "torch",
        "torchaudio",
        "tqdm",
        "transformers",
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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_project_on_path() -> None:
    if REMOTE_PROJECT_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_ROOT)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 10,
    scaledown_window=20 * 60,
    volumes={REMOTE_MODEL_STORE_DIR: inference_volume},
    secrets=[modal.Secret.from_name("sauti-tts-inference-env")],
)
class InferenceApi:
    @modal.enter()
    def load(self) -> None:
        _ensure_project_on_path()
        from app.asr import SautiASRService
        from app.engine import SautiTTSService
        from app.security import SharedRateLimiter
        from app.sts import StsOrchestrator
        from app.utils import setup_logging

        os.chdir(REMOTE_PROJECT_ROOT)
        setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
        limit = int(os.environ.get("RATE_LIMIT_REQUESTS", "10"))
        window_seconds = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
        self.max_upload_bytes = int(os.environ.get("ASR_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
        model_load_lock = threading.Lock()
        self.tts_service = SautiTTSService(
            checkpoint_path=REMOTE_TTS_MODEL_PATH,
            speakers_json_path=REMOTE_SPEAKERS_JSON,
            preload=_env_flag("PRELOAD_TTS_MODEL", False),
            load_lock=model_load_lock,
        )
        self.asr_service = SautiASRService(
            model_path=REMOTE_ASR_MODEL_DIR,
            model_id=ASR_MODEL_ID,
            chunk_length_s=int(os.environ.get("ASR_CHUNK_LENGTH_SECONDS", "25")),
            preload=_env_flag("PRELOAD_ASR_MODEL", False),
            load_lock=model_load_lock,
        )
        self.rate_limiter = SharedRateLimiter(
            store=rate_limit_store,
            limit=limit,
            window_seconds=window_seconds,
        )
        self.sts_orchestrator = StsOrchestrator(
            asr_service=self.asr_service,
            tts_service=self.tts_service,
        )

    @modal.asgi_app()
    def asgi_app(self):
        _ensure_project_on_path()
        from app.api import create_app

        return create_app(
            tts_service=self.tts_service,
            asr_service=self.asr_service,
            sts_orchestrator=self.sts_orchestrator,
            rate_limiter=self.rate_limiter,
            allowed_origins=_allowed_origins(),
            max_upload_bytes=self.max_upload_bytes,
        )


@app.local_entrypoint()
def main() -> None:
    print("Deploy with: modal deploy modal_app.py", flush=True)
