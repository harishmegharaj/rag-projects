# Project C — Video upload and transcription API

Upload videos over HTTP, bucket them under a **logical index** (namespace), and **transcribe speech to text** with either the **OpenAI Whisper** HTTP API or **local faster-whisper** (no cloud account). The service extracts audio with **ffmpeg**, splits long audio when needed, and persists **metadata + transcripts** in **SQLite** (swappable for Postgres via `DATABASE_URL`).

---

## What you get

| Concern | Approach |
|--------|----------|
| **API** | FastAPI, OpenAPI at `/docs` |
| **Auth** | Optional `API_KEY` → require `X-API-Key` or `Authorization: Bearer …` |
| **Indexes** | Each video belongs to one `index_id`. Omit on upload → new UUID index (one index per video). Reuse the same `index_id` to group many videos. |
| **Transcription** | Background worker polls DB; `pending` → `processing` → `completed` / `failed` |
| **Speaker + time** | Whisper/OpenAI: timed segments only, no speaker IDs. **Deepgram** backend: diarized turns with `speaker`, `start`/`end`, plus `conversation_text` on `GET …/transcript` |
| **Storage** | Files under `VIDEO_STORAGE_DIR` / `data/videos/{index_id}/` |
| **Production hooks** | Docker + volume for DB and files; readiness checks; env-based limits |

For **multiple API replicas**, replace the in-process worker with a **queue + workers** (Celery, RQ, Bull, SQS, etc.) using the same `videos` table as the job source so only one worker claims a row.

---

## Requirements

- **Python 3.11+**
- **ffmpeg** and **ffprobe** on `PATH` (included in the Docker image)
- **Either** an **OpenAI API key** (for `TRANSCRIPTION_BACKEND=openai` or `auto` with the key set) **or** **faster-whisper** for fully local transcription (see below)

---

## Local transcription (no OpenAI)

1. Install the extra dependency (downloads CTranslate2 + model weights on first run):

   ```bash
   pip install -r requirements-local.txt
   ```

2. In `.env`, force local mode or rely on **auto** without a key:

   ```bash
   TRANSCRIPTION_BACKEND=local
   # or leave TRANSCRIPTION_BACKEND=auto and omit OPENAI_API_KEY
   ```

3. Optional tuning:

   | Variable | Typical values |
   |----------|----------------|
   | `LOCAL_WHISPER_MODEL` | `tiny`, `base`, `small`, `medium`, `large-v3` (speed vs accuracy) |
   | `LOCAL_WHISPER_DEVICE` | `cpu` or `cuda` |
   | `LOCAL_WHISPER_COMPUTE_TYPE` | `int8` on CPU; `float16` on CUDA GPU |

Transcripts use the same JSON shape as the cloud path: `text`, `segments` with `start` / `end` / `text`, `language`, and `model` stored as `local:<size>` (for example `local:base`).

---

## Quick start (local)

```bash
cd project-c-video-transcription
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt
cp .env.example .env
# For local-only: TRANSCRIPTION_BACKEND=local (or auto with no OPENAI_API_KEY)
# Optional: API_KEY=...
python scripts/serve.py
```

- **Health:** `GET http://localhost:8000/health`
- **Ready (ffmpeg):** `GET http://localhost:8000/ready`
- **Docs:** `http://localhost:8000/docs`

---

## Configuration

| Variable | Meaning |
|----------|---------|
| `TRANSCRIPTION_BACKEND` | `auto` (default), `openai`, `local`, or `deepgram` |
| `OPENAI_API_KEY` | Required when backend is `openai` or `auto` with key-based OpenAI |
| `DEEPGRAM_API_KEY` | Required when `TRANSCRIPTION_BACKEND=deepgram` |
| `DEEPGRAM_MODEL` | e.g. `nova-2` (default) |
| `LOCAL_WHISPER_MODEL` | faster-whisper model id (default `base`) |
| `LOCAL_WHISPER_DEVICE` / `LOCAL_WHISPER_COMPUTE_TYPE` | Inference device and precision |
| `API_KEY` | If set, required on all `/v1/*` routes |
| `VIDEO_STORAGE_DIR` | Override video file root (default: `data/videos/`) |
| `DATABASE_URL` | SQLAlchemy URL (default: SQLite under `data/app.sqlite3`) |
| `WHISPER_MODEL` | OpenAI audio model (default `whisper-1`) |
| `TRANSCRIBE_SEGMENT_SECONDS` | Max chunk length in seconds for long files (default `480`) |
| `MAX_UPLOAD_BYTES` | Max upload size (default 500 MiB) |

---

## HTTP API

All **`/v1/*`** routes accept optional auth when `API_KEY` is set:

```http
X-API-Key: your-secret
```

or

```http
Authorization: Bearer your-secret
```

### Create an empty index (optional)

`POST /v1/indexes` JSON body:

```json
{ "label": "Customer onboarding" }
```

Response: `{ "index_id": "...", "label": "..." }`.

### Upload a video

`POST /v1/videos` (`multipart/form-data`):

| Field | Required | Description |
|-------|----------|-------------|
| `file` | Yes | Video file |
| `index_id` | No | Existing or new index ID (see below) |
| `index_label` | No | Sets or updates the index label when the index is created |

**Index behavior**

- **No `index_id`:** a **new** UUID index is created; this upload is the first video in that index (typical “one index per video” setup).
- **`index_id` set:** video is attached to that index. If the index does not exist, it is **created**. Optional `index_label` sets the label on create or update.

**Example (curl):**

```bash
curl -sS -X POST "http://localhost:8000/v1/videos" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/talk.mp4" \
  -F "index_id=my-demo-index" \
  -F "index_label=Demo"
```

Response includes `video_id`, `index_id`, `status` (`pending` first).

### Poll video status

`GET /v1/videos/{video_id}` — includes `status`, `error_message` if failed, and a short `transcript_preview` when done.

### Get full transcript

`GET /v1/videos/{video_id}/transcript` — returns `text`, `segments`, `language`, `model`, and:

- **`has_speaker_labels`** — `true` when segments include Deepgram diarization (`speaker` is an integer id per turn).
- **`conversation_text`** — present only when speakers are known: human-readable lines like `[00:12] Speaker 1: …` using each segment’s `start` time.

Each segment (when diarized) includes **`start`**, **`end`** (seconds), **`text`**, **`speaker`** (0-based id), **`speaker_label`** (`Speaker 1`, `Speaker 2`, …), and optional **`confidence`**. Whisper/OpenAI/local runs set `speaker` / `speaker_label` to `null` but still expose timestamps on segments when available.

**Two-person (or more) conversations:** set `TRANSCRIPTION_BACKEND=deepgram`, add `DEEPGRAM_API_KEY` in `.env`, restart the server, then upload and transcribe as usual.

- **`409`** while status is `pending` or `processing`.
- **`502`** if status is `failed` (body includes `error_message`).

### Index introspection

- `GET /v1/indexes/{index_id}` — label, `video_count`, `created_at`
- `GET /v1/indexes/{index_id}/videos` — list videos and statuses

---

## Docker

```bash
export OPENAI_API_KEY=sk-...
# optional: export API_KEY=...
docker compose up --build
```

SQLite and uploaded files persist in the **`vtx_data`** volume. For **Postgres**, set `DATABASE_URL` to a `postgresql+psycopg://…` URL and add the driver to `requirements.txt`.

---

## Operations notes

1. **Stuck `processing`:** if the process dies mid-job, rows can stay `processing`. A production job runner should use **leases**, **heartbeats**, or **timeout reclaim**.
2. **Horizontal scale:** run **one worker pool** (or partitioned workers) so each video is transcribed once; use **object storage (S3)** for `storage_path` in a fork.
3. **Secrets:** inject `OPENAI_API_KEY` and `API_KEY` via your platform’s secret manager, not the image.
4. **Cost / limits:** long files generate **multiple Whisper calls** when chunked; tune `TRANSCRIBE_SEGMENT_SECONDS` and monitor usage.

---

## Layout

| Path | Role |
|------|------|
| `src/config.py` | Env and paths |
| `src/models.py` | `MediaIndex`, `Video`, statuses |
| `src/db.py` | SQLAlchemy engine + `init_db` |
| `src/transcribe.py` | ffmpeg + Whisper API |
| `src/worker.py` | DB-backed polling worker |
| `src/api.py` | FastAPI routes |
| `scripts/serve.py` | uvicorn entrypoint |

---

## License

Same as your learning monorepo; used for education and production-style patterns.
