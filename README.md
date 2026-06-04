# Sauti TTS Inference

Minimal inference-only package for deploying Swahili TTS on Modal.

## What Is Included

- Public Modal backend with:
  - `GET /v1/speakers`
  - `POST /v1/synthesize`
- Only speakers `1`, `4`, and `6`
- Existing checkpoint from Modal volume:
  - `sauti-tts-ckpts/sauti_tts_multi/model_last.pt`
- Vite frontend that sends `text` + `speaker` and plays returned WAV audio

## Layout

- `backend/`
  - `modal_app.py`
  - `app/`
- `frontend/`

## Backend Behavior

- Uses `T4` GPU on demand.
- Scales to zero when idle by default on Modal.
- Loads the model once per warm container with `@modal.enter()`.
- Accepts only:
  - `text`
  - `speaker`
- Internally routes requests:
  - very short text: safe single-pass mode
  - normal single sentence: upstream mode
  - longer or multi-sentence text: safe chunked mode

## Security Controls

- Only speakers `1`, `4`, `6` are allowed.
- No user-provided paths or inference flags are accepted.
- Shared IP-based rate limiting uses a Modal Dict.
- CORS is restricted by `FRONTEND_ORIGINS`.
- FastAPI docs and OpenAPI are disabled in production.

## Modal Secret

Create the runtime secret before deploy:

```bash
modal secret create sauti-tts-inference-env \
  FRONTEND_ORIGINS=https://your-frontend.example \
  RATE_LIMIT_REQUESTS=60 \
  RATE_LIMIT_WINDOW_SECONDS=60 \
  LOG_LEVEL=INFO
```

For local frontend development, you can start with:

```bash
modal secret create sauti-tts-inference-env \
  FRONTEND_ORIGINS=http://localhost:5173 \
  RATE_LIMIT_REQUESTS=60 \
  RATE_LIMIT_WINDOW_SECONDS=60 \
  LOG_LEVEL=INFO
```

## Deploy Backend

From `sauti-tts-inference/backend`:

```bash
modal deploy modal_app.py
```

This deploys the public backend and mounts:

- `sauti-tts-ckpts`

The deploy entrypoint is lazy-imported, so local deploy only needs the Modal
CLI/runtime. FastAPI, Torch, F5-TTS, and the rest are installed inside the
Modal image and do not need to exist in your local Python environment.

## Frontend Setup

From `sauti-tts-inference/frontend`:

```bash
pnpm install
```

Create `.env.local`:

```bash
VITE_API_BASE_URL=https://your-modal-url.modal.run
```

Run locally:

```bash
pnpm dev
```

Build for static hosting:

```bash
pnpm build
```

Host `dist/` on your preferred static host.

## API Contract

### `GET /v1/speakers`

Returns:

```json
[
  { "speaker": 1, "label": "Female 1" },
  { "speaker": 4, "label": "Male" },
  { "speaker": 6, "label": "Female 2" }
]
```

### `POST /v1/synthesize`

Request:

```json
{
  "text": "Mama alienda shuleni.",
  "speaker": 1
}
```

Optional query:

- `format=wav`
- `format=mp3`

Response:

- `200 OK`
- `Content-Type: audio/wav` or `audio/mpeg`

Error responses are JSON with `code` and `detail`.

## Operational Notes

- First request after idle will have a cold start because the GPU container scales down.
- T4 is cheaper than larger GPUs but will have slower generation and higher cold-start impact.
- Backend timeout is set to 10 minutes.
- The sync API is suitable for moderate-length text. If you later want much longer passages, move to a queued async job flow.

## Cloudflare Pages

This frontend is a static Vite build and can be deployed directly to Cloudflare Pages.

Suggested settings:

- Framework preset: `Vite`
- Build command: `pnpm build`
- Build output directory: `dist`
- Environment variable:
  - `VITE_API_BASE_URL=https://your-modal-url.modal.run`
