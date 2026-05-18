# Ollama-compatible API

All routes live under `/api`. Schemas mirror the Ollama HTTP API so existing
Ollama clients (`ollama` CLI, LangChain `Ollama` provider, LM Studio, etc.)
work without modification.

Source: `Sources/OLMLX/Routes/OllamaRoutes.swift`.

---

## GET `/`

Health probe. Returns the plain-text string `Ollama is running`.

---

## GET `/api/version`

```json
{ "version": "0.1.0" }
```

---

## GET `/api/tags`

List every registered model.

```json
{
  "models": [
    {
      "name": "qwen3:8b",
      "model": "mlx-community/Qwen3-8B-4bit",
      "modified_at": "",
      "size": 0,
      "digest": "",
      "details": { "format": "mlx", "family": "", "parameter_size": "", "quantization_level": "" }
    }
  ]
}
```

`size`, `digest`, and `details` are populated from `manifest.json` if the
snapshot is on disk; otherwise they're left empty.

---

## POST `/api/show`

```json
// request
{ "model": "qwen3:8b", "verbose": false }
```

```json
// response (abbreviated)
{
  "modelfile": "",
  "parameters": "",
  "template": "",
  "details": { "format": "mlx", "family": "qwen2", "parameter_size": "8B", "quantization_level": "q4" },
  "model_info": { "general.architecture": "qwen2", "general.parameter_count": "8B" },
  "modified_at": "2026-05-18T15:30:00Z"
}
```

`404` if the model isn't in the registry.

---

## POST `/api/chat`

Chat completion. Body schema:

```jsonc
{
  "model": "qwen3:8b",
  "messages": [
    { "role": "user", "content": "Hello!" }
  ],
  "tools": [...],               // optional, see Tools below
  "format": "json",             // optional
  "stream": true,               // default true
  "options": { ... },           // see Options below
  "keep_alive": "5m"            // override default keep-alive for this load
}
```

Validation: `messages` must be non-empty (the decoder throws
`SchemaValidationError` otherwise).

### Streaming response (`stream: true`)

NDJSON, one object per line. Content-type: `application/x-ndjson`.

```jsonc
// repeated per token chunk
{"model":"qwen3:8b","created_at":"...","message":{"role":"assistant","content":"Hel"},"done":false}
{"model":"qwen3:8b","created_at":"...","message":{"role":"assistant","content":"lo"},"done":false}
// final message
{"model":"qwen3:8b","created_at":"...","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":12,"eval_count":42,"total_duration":1234000000}
```

### Non-streaming response (`stream: false`)

Single JSON object:

```json
{
  "model": "qwen3:8b",
  "created_at": "2026-05-18T15:30:00Z",
  "message": { "role": "assistant", "content": "Hello back!" },
  "done": true,
  "done_reason": "stop",
  "prompt_eval_count": 12,
  "eval_count": 42,
  "total_duration": 1234000000
}
```

### Options

`options` is the same flat bag of fields the upstream Ollama API uses. Only
the ones below are wired through to MLX `GenerateParameters`:

| Field | Type | Maps to |
| --- | --- | --- |
| `temperature` | double | `params.temperature` |
| `top_p` | double | `params.topP` |
| `top_k` | int | `params.topK` |
| `min_p` | double | `params.minP` |
| `num_predict` | int | `params.maxTokens` |
| `repeat_penalty` | double | `params.repetitionPenalty` |
| `repeat_last_n` | int | `params.repetitionContextSize` |
| `presence_penalty` | double | `params.presencePenalty` |
| `frequency_penalty` | double | `params.frequencyPenalty` |
| `stop` | string[] | (passed through to options, no MLX hookup yet) |
| `seed` | int | currently read but not applied (MLX RNG seeding TBD) |
| `typical_p` | double | **silently dropped** (logs a warning) |

Unknown fields are preserved on decode/encode but otherwise ignored. The rest
of the Ollama options (`num_keep`, `num_ctx`, `num_gpu`, `num_thread`,
`mirostat*`, `use_mmap`, `use_mlock`, etc.) are accepted but have no effect —
MLX manages those internally.

### Tools

`tools` follows the OpenAI function-calling shape:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
          "type": "object",
          "properties": { "location": { "type": "string" } },
          "required": ["location"]
        }
      }
    }
  ]
}
```

The server parses tool calls from the model's raw output using
`ToolParser` (`Sources/OLMLX/Engine/ToolParser.swift`), which understands the
formats emitted by Qwen, Mistral, Llama (`<|python_tag|>`), DeepSeek, MiniMax,
Gemma 4, gpt-oss channels, standalone XML (`<function=...>`), and bare JSON.
Parsed calls surface as `message.tool_calls` on the response.

### Fallback path

If a model is configured but the inference engine couldn't load it (e.g. you
haven't compiled the metallib yet), the server returns a placeholder
`"Hello from <name>"` body instead of crashing. This lets you smoke-test the
routing layer without a working GPU stack — but it does mean a 200 response
is **not** by itself proof that the model worked. Check the `eval_count`
field; a stubbed response has none.

---

## POST `/api/generate`

Single-turn completion (no chat history). Schema:

```jsonc
{
  "model": "qwen3:8b",
  "prompt": "Write a haiku about coffee",      // required, non-empty
  "system": "You are concise.",
  "stream": true,
  "options": { ... },
  "keep_alive": "5m"
}
```

Internally this just wraps `prompt` in a single user message and runs the
same generation path as `/api/chat`. Responses use the `GenerateResponse`
schema instead of `ChatResponse`:

```json
{
  "model": "qwen3:8b",
  "created_at": "...",
  "response": "Bitter morning brew / ...",
  "done": true,
  "done_reason": "stop",
  "prompt_eval_count": 8,
  "eval_count": 31
}
```

---

## POST `/api/pull`

Trigger a HuggingFace snapshot download.

```json
{ "model": "qwen3:8b", "insecure": false, "stream": true }
```

> `insecure` and `stream` are **required** by the Swift decoder even though
> the Ollama spec treats them as optional. If you send `{"model":"qwen3:8b"}`
> alone you'll get a 400. Pass `"insecure":false,"stream":true` explicitly.

The current implementation does **not** stream progress — it blocks until the
download completes and then returns a single object:

```json
{ "status": "success" }
```

Errors:

- `400 Bad Request` if the resolved HF path isn't `namespace/name` shape.
- `500 Internal Server Error` if the HuggingFace download fails (the
  partially-downloaded directory is cleaned up).

If the model alias is not in the registry, the request `model` field is used
verbatim as a HuggingFace path — i.e. you can pull arbitrary repos without
pre-registering them.

---

## DELETE `/api/delete`

```json
{ "model": "qwen3:8b" }
```

Removes the snapshot from disk (under `OLMLX_MODELS_DIR`). Returns `200 OK`
on success, `404` if the model isn't known to the registry. The registry
entry is **not** removed by this endpoint — use `olmlx models delete` from
the CLI for that.

---

## POST `/api/warmup`

Loads (and downloads if needed) a model so the next chat request doesn't pay
the cold-start cost.

```json
{ "model": "qwen3:8b", "keep_alive": "30m" }
```

Returns `200 OK`. The `keep_alive` field overrides the default for this load.

---

## POST `/api/unload`

```json
{ "model": "qwen3:8b" }
```

Drops the model from `ModelManager`'s in-memory map. Returns `200 OK`. A
subsequent request will reload it from disk.

---

## GET `/api/ps`

List currently loaded models.

```json
{
  "models": [
    {
      "name": "qwen3:8b",
      "model": "mlx-community/Qwen3-8B-4bit",
      "size": 0,
      "digest": "",
      "details": { "format": "mlx", "family": "", "parameter_size": "", "quantization_level": "" },
      "expires_at": "",
      "size_vram": 0,
      "active_refs": 0
    }
  ]
}
```

`size_vram`, `active_refs`, and `expires_at` are placeholders today — MLX
doesn't surface per-model VRAM accounting through the Swift bindings, and the
manager doesn't refcount in-flight requests yet.

---

## Blob endpoints

These exist for Ollama-CLI compatibility.

- `HEAD /api/blobs/:digest` always returns `200 OK`.
- `POST /api/blobs/:digest` writes the request body to
  `<modelsDir>/blobs/<digest>`.

The Ollama CLI uploads blobs before calling `/api/create` to build a model
from a Modelfile. `olmlx` doesn't implement `/api/create`, so these blobs
aren't actually consumed today — they're stored to keep the protocol
contract.

---

## Endpoints that return 501

`/api/copy`, `/api/create`, `/api/abort`, `/api/embed`, `/api/embeddings`.
See [Model Management](../models.md#whats-not-implemented).
