# HTTP API Reference

`olmlx` exposes three parallel API surfaces. They all share the same model
registry, model manager, and inference engine — pick whichever shape your
client expects.

| Surface | Prefix | Doc |
| --- | --- | --- |
| Ollama-compatible | `/api/*` | [ollama.md](ollama.md) |
| OpenAI-compatible | `/v1/*` | [openai.md](openai.md) |
| Anthropic-compatible | `/v1/messages` | [anthropic.md](anthropic.md) |

## Conventions

- All requests are JSON, all responses are JSON (unless streaming).
- Streaming responses use **NDJSON** (one JSON object per line) for Ollama
  routes. The OpenAI and Anthropic surfaces currently respond
  non-streaming regardless of the `stream` flag — they decode the field but
  return a single JSON body.
- Every response includes an `X-Request-ID` header (a random UUID per
  request) — useful for correlating with server logs.

## CORS

`CORSMiddleware` is mounted by default. The default allowed origins are
`http://localhost:*` and `http://127.0.0.1:*`. Override with
`OLMLX_CORS_ORIGINS` (comma-separated list). Allowed methods:
`GET, POST, PUT, OPTIONS, DELETE, PATCH, HEAD`.

## Authentication

There is none. `olmlx` is intended to run on `localhost` or inside a trusted
network. If you need auth, put it in front of `olmlx` (e.g. an nginx reverse
proxy with HTTP Basic auth, or a Tailscale ACL).

## Error shape

`Abort` errors from the server use Vapor's default JSON error format:

```json
{ "error": true, "reason": "model not found" }
```

The HTTP status code carries the actual signal:

| Status | When |
| --- | --- |
| `200 OK` | Success |
| `400 Bad Request` | Body fails schema validation (e.g. invalid HF path) |
| `404 Not Found` | Unknown model |
| `500 Internal Server Error` | Download or inference failure |
| `501 Not Implemented` | Endpoint exists but is not wired up — see [models.md](../models.md#whats-not-implemented) |

## Endpoint inventory

### Ollama (`/api/*`)

| Method | Path | Status |
| --- | --- | --- |
| GET | `/` | implemented (health probe) |
| GET | `/api/version` | implemented |
| GET | `/api/tags` | implemented |
| POST | `/api/show` | implemented |
| POST | `/api/chat` | implemented (streaming + non-streaming) |
| POST | `/api/generate` | implemented (streaming + non-streaming) |
| POST | `/api/pull` | implemented (synchronous, ignores `stream`) |
| DELETE | `/api/delete` | implemented |
| POST | `/api/warmup` | implemented |
| POST | `/api/unload` | implemented |
| GET | `/api/ps` | implemented |
| HEAD | `/api/blobs/:digest` | always returns 200 |
| POST | `/api/blobs/:digest` | implemented (stores raw bytes) |
| POST | `/api/embed`, `/api/embeddings` | **501** |
| POST | `/api/copy` | **501** |
| POST | `/api/create` | **501** |
| POST | `/api/abort` | **501** |

### OpenAI (`/v1/*`)

| Method | Path | Status |
| --- | --- | --- |
| POST | `/v1/chat/completions` | implemented (non-streaming only) |
| GET | `/v1/models` | implemented |
| POST | `/v1/completions` | **501** |
| POST | `/v1/embeddings` | **501** |

### Anthropic (`/v1/messages`)

| Method | Path | Status |
| --- | --- | --- |
| POST | `/v1/messages` | implemented (non-streaming only) |
| POST | `/v1/messages/count_tokens` | **501** |
