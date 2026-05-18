# Quickstart

This walks through getting from a freshly built `olmlx` to a working chat
completion in under five minutes.

> Make sure you've followed [Installation & Build](installation.md), including
> the metallib step. Without it, the very first request will crash the server.

## 1. Stop the real Ollama daemon (if installed)

`olmlx` binds to port `11434` by default — the same port the real `ollama`
daemon uses. If both are running you'll see `bind: Address already in use`.
Stop Ollama first:

```sh
launchctl stop com.ollama.ollama  # if installed via the official package
# or
killall ollama
```

Alternatively, run `olmlx` on a different port:

```sh
olmlx serve --port 11435
```

## 2. Register a model

Models are addressed by HuggingFace repo path (e.g.
`mlx-community/Qwen3-8B-4bit`) but are surfaced through the API by an
Ollama-style alias (e.g. `qwen3:8b`). The mapping lives in
`~/.olmlx/models.json`.

Create the file if it does not exist:

```sh
mkdir -p ~/.olmlx
cat > ~/.olmlx/models.json <<'EOF'
{
  "qwen3:8b": {
    "hf_path": "mlx-community/Qwen3-8B-4bit"
  },
  "llama3:8b": {
    "hf_path": "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
  }
}
EOF
```

The shorthand `"qwen3:8b": "mlx-community/Qwen3-8B-4bit"` works too — it's
expanded to the full form on load. See [Model Management](models.md) for the
complete schema (per-model `keep_alive`, KV-cache quant, speculative draft
model, etc.).

## 3. Start the server

```sh
olmlx serve
# olmlx v0.1.0 starting on http://0.0.0.0:11434
```

The server logs every request and is ready as soon as the banner appears.
Models are loaded lazily — the first call that references one will trigger a
HuggingFace download (multi-GB; takes a while the first time).

## 4. Pre-download a model (optional)

Use `/api/pull` to download a model up front. With `stream: false` the call
blocks until the snapshot is on disk:

```sh
curl -s http://localhost:11434/api/pull \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3:8b","insecure":false,"stream":false}'
# {"status":"success"}
```

> `insecure` and `stream` are required fields on `PullRequest` — `curl` calls
> that omit them will fail to decode. See [Ollama API](api/ollama.md).

## 5. Send a chat request

### Ollama-style

```sh
curl -s http://localhost:11434/api/chat \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3:8b",
    "messages": [
      {"role": "user", "content": "Say hello in three languages."}
    ],
    "stream": false
  }'
```

### OpenAI-style

```sh
curl -s http://localhost:11434/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3:8b",
    "messages": [{"role":"user","content":"Say hi"}]
  }'
```

### Anthropic-style

```sh
curl -s http://localhost:11434/v1/messages \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3:8b",
    "max_tokens": 256,
    "messages": [{"role":"user","content":"Say hi"}]
  }'
```

## 6. Quirks worth knowing about

- **Qwen3 emits `<think>...</think>` traces** inline in `message.content`.
  Budget `num_predict` (Ollama) or `max_tokens` (OpenAI / Anthropic)
  generously, or strip the trace client-side.
- **One model at a time by default.** `OLMLX_MAX_LOADED_MODELS=1` evicts the
  oldest model whenever a new one is requested. Bump it if you have RAM to
  spare.
- **Embeddings and `create`/`copy`/`abort` are not implemented.** Calls return
  `501 Not Implemented`. See the [API reference](api/README.md) for the full
  status of each endpoint.
- **CORS is permissive for `http://localhost:*` and `http://127.0.0.1:*`**
  by default. Override with `OLMLX_CORS_ORIGINS`.

You're up. Continue with the [CLI Reference](cli.md), the
[Configuration](configuration.md) page, or jump straight to the
[HTTP API Reference](api/README.md).
