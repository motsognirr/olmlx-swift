# Configuration

Configuration is read **once at startup** from the process environment into an
immutable `Settings` struct. There is no live-reload — restart the server to
pick up changes. Source: `Sources/OLMLX/Config/Settings.swift`.

Every variable is prefixed `OLMLX_`. Invalid values fall back to the default
silently rather than erroring out at boot.

## Server

| Variable | Default | Description |
| --- | --- | --- |
| `OLMLX_HOST` | `0.0.0.0` | Address to bind the HTTP server to |
| `OLMLX_PORT` | `11434` | TCP port (1–65535) |
| `OLMLX_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARN`, `ERROR`) |
| `OLMLX_CORS_ORIGINS` | `http://localhost:*,http://127.0.0.1:*` | **Parsed but not applied today** — the CORS middleware is hard-wired to `allowedOrigin: .originBased`, which echoes whatever `Origin` the client sent. Setting this variable has no effect at runtime. |

## Storage

| Variable | Default | Description |
| --- | --- | --- |
| `OLMLX_MODELS_DIR` | `~/.olmlx/models` | Where HuggingFace snapshots are unpacked |
| `OLMLX_MODELS_CONFIG` | `~/.olmlx/models.json` | Registry mapping aliases → HF paths |
| `OLMLX_PROMPT_CACHE_DISK_PATH` | `~/.olmlx/cache/kv` | Disk-backed KV cache directory |
| `OLMLX_PROMPT_CACHE_DISK_MAX_GB` | `10.0` | Disk KV cache size cap, in GB |

Tilde paths (`~/...`) are expanded against `$HOME`.

## Model lifecycle

| Variable | Default | Description |
| --- | --- | --- |
| `OLMLX_DEFAULT_KEEP_ALIVE` | `5m` | How long a model stays in memory after the last request. Accepts `Ns` / `Nm` / `Nh` (e.g. `30s`, `2h`). `0` pins the model in memory indefinitely (only LRU eviction at `OLMLX_MAX_LOADED_MODELS` capacity can remove it). |
| `OLMLX_MAX_LOADED_MODELS` | `1` | Concurrent in-memory models; oldest is evicted on overflow |
| `OLMLX_MEMORY_LIMIT_FRACTION` | `0.75` | Fraction of system memory MLX is allowed to use (0 < f ≤ 1) |
| `OLMLX_MODEL_LOAD_TIMEOUT` | none | Optional seconds; nothing enforces this today |

## Inference

| Variable | Default | Description |
| --- | --- | --- |
| `OLMLX_INFERENCE_QUEUE_TIMEOUT` | `300` | Seconds a request can wait in the queue before failing |
| `OLMLX_INFERENCE_TIMEOUT` | none | Per-generation hard cap (seconds) |
| `OLMLX_MAX_TOKENS_LIMIT` | `131072` | Upper bound on `max_tokens` / `num_predict` |
| `OLMLX_SYNC_MODE` | `full` | One of `full`, `minimal`, `none` — controls MLX stream sync frequency |

## Prompt cache

The prompt cache memoizes the token sequence + KV state of recent prompts so
that a follow-up request that shares a prefix can skip prompt evaluation.

| Variable | Default | Description |
| --- | --- | --- |
| `OLMLX_PROMPT_CACHE` | `true` | Enable in-memory prompt cache |
| `OLMLX_PROMPT_CACHE_MAX_TOKENS` | `32768` | Max tokens stored per slot |
| `OLMLX_PROMPT_CACHE_MAX_SLOTS` | `4` | LRU slot count |
| `OLMLX_PROMPT_CACHE_DISK` | `false` | Persist KV cache to disk between runs |

## Anthropic-API model aliasing

The Anthropic surface accepts whatever model name is configured in the
registry. There is a planned mechanism for publishing "stable" Claude-style
names that map to local aliases, but it is **not yet wired up**.

| Variable | Default | Description |
| --- | --- | --- |
| `OLMLX_ANTHROPIC_MODELS` | `{}` | **Parsed but not applied today.** JSON map of `<short_name>` → `<local_alias>`. Keys may not contain `-` or `:`. The Anthropic route handler does not consult this map — every request must use a name that resolves against the registry directly. |

Example of the intended (future) shape:

```sh
export OLMLX_ANTHROPIC_MODELS='{"haiku":"qwen3:8b","sonnet":"llama3:8b"}'
```

Once implemented, `POST /v1/messages` with `"model":"haiku"` would resolve
to the `qwen3:8b` local entry. For now, use the registry alias directly.

## KV-cache quantization

| Variable | Default | Description |
| --- | --- | --- |
| `OLMLX_KV_CACHE_QUANT` | none | Quantize the KV cache to save memory |

Format is `method:bits`:

- `method` ∈ {`turboquant`, `spectral`}
- `bits` ∈ {`2`, `4`}

Examples: `turboquant:4`, `spectral:2`. Invalid values are silently ignored
(the validator returns `nil`, the variable becomes `none`).

The global value is applied to all models that do not set their own
`kv_cache_quant` in `models.json`.

## Speculative decoding

Speculative decoding generates several tokens with a small "draft" model and
verifies them against the main model — fewer forward passes on the big model
for the same output.

| Variable | Default | Description |
| --- | --- | --- |
| `OLMLX_SPECULATIVE` | `false` | Enable speculative decoding |
| `OLMLX_SPECULATIVE_STRATEGY` | `classic` | One of `classic`, `dflash`, `eagle` |
| `OLMLX_SPECULATIVE_DRAFT_MODEL` | none | HF path of the draft model |
| `OLMLX_SPECULATIVE_TOKENS` | none | Lookahead window size (tokens) |

These are global defaults; each model in `models.json` may override them on a
per-model basis. See [Model Management](models.md#per-model-config).

## Where defaults come from

All of the above defaults are in `Settings.init(env:)` in
`Sources/OLMLX/Config/Settings.swift`. The struct is `Sendable` and
immutable — pass it explicitly to whatever needs it rather than holding a
global. Tests construct isolated `Settings(env: [...])` instances to verify
parsing behavior.
