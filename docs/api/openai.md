# OpenAI-compatible API

Routes under `/v1`. Source: `Sources/OLMLX/Routes/OpenAIRoutes.swift`.

Point any OpenAI SDK at `http://localhost:11434/v1`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-used")
resp = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Hi"}],
)
print(resp.choices[0].message.content)
```

Auth headers are accepted and ignored — `olmlx` has no auth.

---

## POST `/v1/chat/completions`

### Request

```jsonc
{
  "model": "qwen3:8b",                          // required, matches a registry alias
  "messages": [
    { "role": "user", "content": "Hello!" }
  ],
  "temperature": 0.7,                           // optional
  "top_p": 0.95,                                // optional
  "max_tokens": 512,                            // optional
  "max_completion_tokens": 512,                 // accepted, not separately used
  "n": 1,                                       // accepted; only n=1 is supported
  "stream": false,                              // see streaming note below
  "stop": "###" | ["###", "STOP"],              // accepted, no MLX hookup yet
  "presence_penalty": 0.0,                      // accepted
  "frequency_penalty": 0.0,                     // accepted
  "tools": [...],                               // accepted, decoded; routing TBD
  "tool_choice": "auto" | { "type": "...", ... },
  "seed": 42,                                   // accepted
  "response_format": { "type": "json_schema", "json_schema": { ... } }
}
```

Field shape is decoded by `OpenAIChatRequest`
(`Sources/OLMLX/Schemas/OpenAI.swift`). `response_format` of type
`json_schema` requires a non-empty `name` and a `schema` object —
violations raise a 400.

### Response

```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1747588800,
  "model": "qwen3:8b",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Hi there!" },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 4,
    "total_tokens": 16
  }
}
```

### Streaming caveat

The `stream` field is accepted but **not honored** — the server always
returns a single JSON body. If you need token-by-token streaming today, hit
`/api/chat` instead. The OpenAI streaming response format (`data: {...}\n\n`
SSE) is on the to-do list.

### Tool calls

The schema accepts `tools` and `tool_choice` and decodes them, but the OpenAI
route does not currently re-parse the model output back into a `tool_calls`
array on the response. Use the Ollama `/api/chat` route if you need server-
side tool-call parsing.

### Fallback

Same fallback as the Ollama route: if the model has no inference container
loaded (e.g. the metallib is missing), the server returns a placeholder
`Hello!` body. Confirm the `usage.prompt_tokens` field is non-zero before
trusting the output.

---

## GET `/v1/models`

```json
{
  "object": "list",
  "data": [
    { "id": "qwen3:8b", "object": "model", "created": 0, "owned_by": "olmlx" },
    { "id": "llama3:8b", "object": "model", "created": 0, "owned_by": "olmlx" }
  ]
}
```

Driven by the registry, not by what's currently in memory.

---

## Endpoints that return 501

- `POST /v1/completions` — the legacy text completion endpoint. Use chat
  completions instead.
- `POST /v1/embeddings` — embeddings are not implemented.
