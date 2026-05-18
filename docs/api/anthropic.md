# Anthropic-compatible API

Routes under `/v1/messages`. Source:
`Sources/OLMLX/Routes/AnthropicRoutes.swift`.

Lets you point Anthropic SDK clients at `http://localhost:11434` instead of
`https://api.anthropic.com`. Auth headers are accepted and ignored.

---

## POST `/v1/messages`

### Request

```jsonc
{
  "model": "qwen3:8b",                  // required
  "messages": [
    { "role": "user", "content": "Hello!" }
    // content may also be an array of content blocks
  ],
  "max_tokens": 1024,                   // defaults to 4096
  "stream": false,                      // see streaming note below
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stop_sequences": ["</done>"],        // accepted, no MLX hookup yet
  "system": "You are concise.",         // string or content-block array
  "tools": [...],                       // accepted; routing TBD
  "tool_choice": { "type": "auto" },
  "thinking": { "type": "enabled", "budget_tokens": 2048 },
  "metadata": { "user_id": "..." }
}
```

### Message content

`content` accepts either a plain string or an array of content blocks
(`AnthropicContentBlock`). The route currently flattens content-block arrays
to an empty string — only string content is forwarded to the model today.
If you rely on multimodal or tool-result blocks, use the Ollama route until
this is fully wired up.

Validation: a single message's string content is capped at 1,000,000 chars
and a block-array message at 1,000 blocks (raises `SchemaValidationError`).

### Response

```json
{
  "id": "msg_<uuid>",
  "type": "message",
  "role": "assistant",
  "content": [
    { "type": "text", "text": "Hi there!" }
  ],
  "model": "qwen3:8b",
  "stop_reason": null,
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 4,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

`stop_reason` and `stop_sequence` are always null today — the route doesn't
yet plumb MLX's stop-condition signal back through.

### Streaming caveat

`stream: true` is decoded but the server still returns a single JSON body.
SSE-style `event: ...\n\ndata: {...}\n\n` streaming is not implemented.

### `thinking`

The `thinking` parameter is accepted but not honored — `olmlx` doesn't have
extended-thinking semantics. To get thinking output today you need a model
that natively emits `<think>...</think>` traces (like Qwen3) and parse them
client-side.

---

## Anthropic-style model aliasing

You can publish "Anthropic-shaped" model names without touching the registry
by setting `OLMLX_ANTHROPIC_MODELS`:

```sh
export OLMLX_ANTHROPIC_MODELS='{"haiku":"qwen3:8b","sonnet":"llama3:8b"}'
```

Then a request with `"model":"haiku"` is mapped to the local alias
`qwen3:8b`. Aliases may not contain `-` or `:` (the filter strips those
keys silently on parse) — keep them short and simple. The default value is
an empty map; the model field must be a registry alias.

---

## Endpoints that return 501

- `POST /v1/messages/count_tokens` — token counting is not implemented.
