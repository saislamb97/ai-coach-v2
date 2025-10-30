# AI Coach API & Realtime Chat — README

**Version:** 1.0.0
**Date:** 2025-10-30
**Base URL (prod):** `https://ai-coach-pmkn.onrender.com`

---

## Overview

This service exposes a focused REST API (DRF + Spectacular) and a WebSocket endpoint to:

* Manage **Voices**, **Agents**, **Sessions**, **Chats**, **Slides**
* Stream real-time chat via **WebSocket** (token/sentence streaming, optional TTS + visemes, slide updates)
* Browse OpenAPI docs (Swagger UI / ReDoc / raw schema)
* Use an in-browser **Test Console** to exercise endpoints

> 🔑 You will need:
>
> * `Api-Key` (tenant-scoped)
> * `bot_id` (UUID of the Agent you’ll use, for WS)
> * A `thread_id` matching `^user_[0-9a-f]{16}$` (example: `user_ab12cd34ef56gh78`)

---

## Quick Links

| Name                             | URL                                                         | Notes            |
| -------------------------------- | ----------------------------------------------------------- | ---------------- |
| **Main site**                    | `https://ai-coach-pmkn.onrender.com/`                       | Landing          |
| **Swagger UI**                   | `https://ai-coach-pmkn.onrender.com/api/schema/swagger-ui/` | Interactive docs |
| **ReDoc**                        | `https://ai-coach-pmkn.onrender.com/api/schema/redoc/`      | Reference docs   |
| **Raw OpenAPI Schema**           | `https://ai-coach-pmkn.onrender.com/api/schema/`            | JSON schema      |
| **Test Endpoint (HTML Console)** | `https://ai-coach-pmkn.onrender.com/api/test/`              | Try WS & REST    |

---

## Authentication & Allow-list

### REST

Include on **all REST** requests:

```
Authorization: Api-Key <YOUR_API_KEY>
```

### WebSocket

Provide the API key via **one** of:

1. Header: `Authorization: Api-Key <YOUR_API_KEY>`
2. Query param: `?api_key=<YOUR_API_KEY>`
3. Cookie: `api_key=<YOUR_API_KEY>`

### Origin / Host / IP Allow-list

Requests must also pass at least one allow-list check (configured per API key):

* Origin/Referer host is allowed, **or**
* Host header is allowed, **or**
* Client IP/CIDR is allowed

Errors are returned with meaningful messages; permission failures include both API-key and origin/host/IP checks.

---

## Response & Pagination

List endpoints use DRF **page-number** pagination:

```json
{
  "count": 42,
  "next": "https://.../api/agents/?page=3&page_size=10",
  "previous": "https://.../api/agents/?page=1&page_size=10",
  "results": [ /* items */ ]
}
```

Use `?page=` and `?page_size=` (1–100). Field sets are defined by the API serializers (see Swagger UI).

---

## REST Endpoints (ViewSets)

> Base prefix for all routes below is `/api/`.
> Standard actions: `list`, `retrieve`, `create`, `update`, `partial_update`, `destroy`.

### 1) Voices

```
GET    /api/voices/
POST   /api/voices/
GET    /api/voices/{id}/
PATCH  /api/voices/{id}/
PUT    /api/voices/{id}/
DELETE /api/voices/{id}/
```

**Example — List:**

```bash
curl -H "Authorization: Api-Key $API_KEY" \
  https://ai-coach-pmkn.onrender.com/api/voices/
```

---

### 2) Agents

```
GET    /api/agents/
POST   /api/agents/
GET    /api/agents/{id}/
PATCH  /api/agents/{id}/
PUT    /api/agents/{id}/
DELETE /api/agents/{id}/
```

**Example — Retrieve one (by numeric id):**

```bash
curl -H "Authorization: Api-Key $API_KEY" \
  https://ai-coach-pmkn.onrender.com/api/agents/1/
```

> ℹ️ **SmartMediaMiddleware** absolutizes media URLs (thumbnails, avatar assets) in API/WS JSON responses.

---

### 3) Sessions

```
GET    /api/sessions/
POST   /api/sessions/
GET    /api/sessions/{id}/
PATCH  /api/sessions/{id}/
PUT    /api/sessions/{id}/
DELETE /api/sessions/{id}/
```

Sessions link an end-user `thread_id` (shape `user_16hex`) to an `agent`.
Exact create fields depend on your serializer; see Swagger UI for required inputs.

---

### 4) Chats

```
GET    /api/chats/
POST   /api/chats/
GET    /api/chats/{id}/
PATCH  /api/chats/{id}/
PUT    /api/chats/{id}/
DELETE /api/chats/{id}/
```

Chat entries are typically created automatically by the realtime pipeline after each completed response.
`POST` exists mainly for admin/import use cases.

---

### 5) Slides

```
GET    /api/slides/
POST   /api/slides/
GET    /api/slides/{id}/
PATCH  /api/slides/{id}/
PUT    /api/slides/{id}/
DELETE /api/slides/{id}/
```

Slides store an **Editor.js** document (`editorjs`) plus optional `title`/`summary`.
The engine can generate/update slides synchronously and persist them to the session.

---

## WebSocket — `/ws/chat/`

**Endpoint**

```
wss://ai-coach-pmkn.onrender.com/ws/chat/?bot_id=<uuid>&thread_id=<user_16hex>&website_language=en
```

**Query parameters**

| Param              | Type   | Notes                                       |
| ------------------ | ------ | ------------------------------------------- |
| `bot_id`           | UUID   | Agent identifier (WS uses `bot_id`, not id) |
| `thread_id`        | string | Must match `^user_[0-9a-f]{16}$`            |
| `website_language` | string | UI hint (e.g., `en`)                        |
| `api_key`          | string | Optional if sent in header/cookie           |

**Limits & Behavior**

* **Inbound message rate:** up to **20 Hz** (excess dropped)
* **Audio upload size:** up to **25 MiB**
* **Audio formats:** inferred from data URL or `format` hint (`webm`, `wav`, `m4a`, `ogg`, `mp3`)
* **STT model:** `OPENAI_STT_MODEL` env (default `whisper-1`)
* **Concurrency:** one active run at a time; new queries cancel the previous run

### Client → Server

```json
{ "type": "text_query",  "text": "Hello", "local_time": "2025-10-30 10:00:00", "muteAudio": false }
{ "type": "audio_query", "audio": "<base64|dataurl>", "format": "webm", "muteAudio": true }
{ "type": "mute_audio" }
{ "type": "unmute_audio" }
{ "type": "stop_audio" }
{ "type": "ping" }
```

> **Audio tips:** Prefer `data:audio/<mime>;base64,<...>`; if you send raw base64, include a `format` hint.

### Server → Client

* `connected` — handshake info: `{ bot_id, thread_id }`
* `text_query` — echo of user text (with `local_time`)
* `response_start` → `text_token`* → `text_sentence`* → `response_done` → `response_ended`
* `audio_muted` — `{ muted: boolean }`
* `audio_response` — `{ audio: <base64>, viseme: number[][], local_time }`
* `slides_response` — `{ slides: { title, summary, editorjs }, local_time }`
* `slides_done` — end-of-slides marker
* `stop_audio` — confirmation
* `error` — `{ message }`
* `pong` — ping reply

* **Streaming:**
`text_token` are raw tokens; `text_sentence` contains finalized sentences and optional emotion:

```json
{
  "type": "text_sentence",
  "text": "Let's start with a simple plan.",
  "emotion": {
    "items": [
      {"name":"Joy","intensity":2},
      {"name":"Anger","intensity":1},
      {"name":"Sadness","intensity":1}
    ]
  },
  "local_time": "2025-10-30 10:00:02"
}
```

**Timings payload**

```json
{
  "type": "response_done",
  "timings": {
    "prepare_ms": 41,
    "run_ms": 1850,
    "total_ms": 2104,
    "ws_total_sec": 2.12
  }
}
```

**Close codes (common)**

* `4403` — auth failed (bad/missing API key or allow-list)
* `4000` — missing required query params
* `4004` — agent not found/inactive
* `4005` — invalid `thread_id` pattern or session not found

---

## Data Model (examples)

> Exact fields are defined by your serializers; below are representative shapes.

**Agent**

```json
{
  "id": 1,
  "bot_id": "6a97a3c4-5d25-47a8-bb31-b6f35c98e6b7",
  "name": "AI Coach",
  "thumbnail": "https://.../media/agents/1/thumb.png",
  "voice": { "service": "elevenlabs", "voice_id": "..." }
}
```

**Session**

```json
{
  "id": 10,
  "thread_id": "user_ab12cd34ef56gh78",
  "agent": 1,
  "created_at": "2025-10-30T02:11:22Z"
}
```

**Chat**

```json
{
  "id": 123,
  "session": 10,
  "query": "How do I plan my week?",
  "response": "Here’s a simple 3-step plan...",
  "emotion": { "items": [{"name":"Joy","intensity":2}, {"name":"Anger","intensity":1}, {"name":"Sadness","intensity":1}], "stats": { "count": 3, "avg": {"Joy":2,"Anger":1,"Sadness":1}, "max": {"Joy":3,"Anger":1,"Sadness":1}, "last": {"Joy":2,"Anger":1,"Sadness":1} } },
  "viseme": { "segments": 3, "frames_total": 480, "last_frames": [[...]] },
  "meta": {
    "schema": "engine.v3",
    "bot_id": "6a97a3c4-5d25-47a8-bb31-b6f35c98e6b7",
    "thread_id": "user_ab12cd34ef56gh78",
    "model": "gpt-4o-mini",
    "timings_ms": { "prepare_ms": 42, "run_ms": 1876, "total_ms": 2120 },
    "response_len": 312,
    "tools": ["search_wikipedia", "generate_or_update_slides"]
  },
  "created_at": "2025-10-30T02:12:05Z"
}
```

**Slides (Editor.js)**

```json
{
  "id": 7,
  "session": 10,
  "title": "Weekly Planning",
  "summary": "A compact guide to plan your week.",
  "editorjs": { "time": 1698650000, "version": "2.x", "blocks": [ /* ... */ ] },
  "updated_at": "2025-10-30T02:12:10Z"
}
```

---

## Curl Quickstart

**List Agents**

```bash
curl -H "Authorization: Api-Key $API_KEY" \
  https://ai-coach-pmkn.onrender.com/api/agents/
```

**Create a Session** *(fields depend on serializer; see Swagger UI)*

```bash
curl -X POST -H "Authorization: Api-Key $API_KEY" -H "Content-Type: application/json" \
  -d '{"thread_id":"user_ab12cd34ef56gh78","agent":1}' \
  https://ai-coach-pmkn.onrender.com/api/sessions/
```

**Connect WebSocket** (JS)

```js
const url = new URL("wss://ai-coach-pmkn.onrender.com/ws/chat/");
url.searchParams.set("bot_id", "6a97a3c4-5d25-47a8-bb31-b6f35c98e6b7");
url.searchParams.set("thread_id", "user_ab12cd34ef56gh78");
url.searchParams.set("website_language", "en");
url.searchParams.set("api_key", "<YOUR_API_KEY>");

const ws = new WebSocket(url);
ws.onmessage = (e) => console.log("WS:", JSON.parse(e.data));
ws.onopen = () => ws.send(JSON.stringify({type:"text_query", text:"Hello!", muteAudio:false}));
```

---

## Project Structure (high-level)

```
aicoach/
├─ api/                 # DRF endpoints (Voices, Agents, Sessions, Chats, Slides)
│  ├─ views.py          # ModelViewSets with ApiKeyAuthentication + permission
│  ├─ urls.py           # Routers + /api/test/ console
│  └─ permissions.py    # HasValidAPIKeyAndAllowedOrigin (key + allow-list)
├─ engine/
│  ├─ rag.py            # prepare → run (stream text, emotion, TTS, slides) → finalize/persist
│  ├─ nodes.py          # sentence-level emotion; slide tool router; streaming logic
│  ├─ tools.py          # search_wikipedia, generate_or_update_slides
│  └─ tts_stt.py        # TTS/STT integration
├─ stream/
│  ├─ consumers.py      # WebSocket ChatConsumer (protocol above)
│  └─ ws_auth.py        # WS auth: header/query/cookie + origin/host/IP checks
├─ agent/, memory/      # Models for Agent/Voice and Session/Chat/Slides
├─ templates/test.html  # Raw WS debug console
└─ aicoach/settings.py  # DRF/Channels/Redis/CORS/etc.
```

---

## Environment & Settings

### `.env` (sample)

```dotenv
# Core / Django
DEBUG=true
DJANGO_SECRET_KEY="<set-me>"
BASE_URL=http://127.0.0.1:8000
CACHE_TTL=600

# OpenAI / ElevenLabs
OPENAI_API_KEY="<set-me>"
ELEVENLABS_API_KEY="<set-me>"
OPENAI_STT_MODEL=whisper-1
ELEVENLABS_TTS_MODEL=eleven_multilingual_v2

# Postgres
DATABASE_NAME=aicoach
DATABASE_USER=postgres
DATABASE_PASSWORD=postgres
DATABASE_HOST=localhost
DATABASE_PORT=5432
```

### Redis

* `REDIS_URL_CACHE` (default `redis://localhost:6379/1`)
* `REDIS_URL_CHANNELS` (default `redis://localhost:6379/2`)

> Channels uses `CHANNEL_LAYERS` for WS groups.

### CORS/CSRF

* `CORS_ALLOW_ALL_ORIGINS` defaults to `true` for dev
* `CSRF_TRUSTED_ORIGINS` includes `BASE_URL` and local hosts by default

---

## Running Locally

> Requires: Python 3.10+, PostgreSQL, Redis

```bash
# 1) Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Environment
cp .env.example .env  # or create as above and set your keys

# 3) Database
python manage.py migrate
python manage.py createsuperuser  # optional

# 4) Static (if needed)
python manage.py collectstatic --noinput

# 5) Start Redis (if not already)
redis-server  # or: docker run -p 6379:6379 redis:7

# 6) Start server (dev)
python manage.py runserver 0.0.0.0:8000
# or ASGI (recommended for WS):
daphne -b 0.0.0.0 -p 8000 aicoach.asgi:application
```

Open:

* Swagger: `http://127.0.0.1:8000/api/schema/swagger-ui/`
* ReDoc: `http://127.0.0.1:8000/api/schema/redoc/`
* Test console: `http://127.0.0.1:8000/api/test/`

---

## Notes & Tips

* **WebSocket auth:** For quick demos, append `?api_key=...` to the WS URL; in production, prefer the `Authorization: Api-Key ...` header if your client supports it.
* **Thread IDs:** Must match `^user_[0-9a-f]{16}$`. Generate one per end-user session.
* **Slides:** Slide generation/update is **synchronous** in this build—expect one or more `slides_response` followed by `slides_done`. Always use the **latest** payload received for a run.
* **SmartMediaMiddleware:** Media fields in API/WS JSON are automatically converted to absolute URLs based on deployment.
* **Security:** The DRF permission requires a **valid API key** *and* an allow-listed **Origin/Host/IP**—both are enforced.
