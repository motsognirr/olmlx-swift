# Model Management

`olmlx` keeps a registry of model aliases on disk and downloads model weights
on demand from HuggingFace.

## On-disk layout

```
~/.olmlx/
  models.json                                     # the registry
  models/
    mlx-community--Qwen3-8B-4bit/                 # a single snapshot
      config.json
      tokenizer.json
      model.safetensors
      manifest.json                               # written by olmlx
    mlx-community--Meta-Llama-3-8B-Instruct-4bit/
      ...
  cache/
    kv/                                           # disk-backed prompt cache
```

The HuggingFace path `<namespace>/<name>` is mapped to the directory name
`<namespace>--<name>` (forward slashes become double dashes).

## The registry: `models.json`

`~/.olmlx/models.json` (overridable with `OLMLX_MODELS_CONFIG`) is a single
JSON object whose keys are Ollama-style aliases and whose values describe the
mapped HuggingFace repository plus optional per-model config.

The decoder accepts three value shapes for backwards compatibility:

```jsonc
{
  // 1. Plain string shorthand — equivalent to {"hf_path": "..."}
  "qwen3:latest": "mlx-community/Qwen3-8B-4bit",

  // 2. Object form (the common case)
  "qwen3:8b": {
    "hf_path": "mlx-community/Qwen3-8B-4bit",
    "keep_alive": "10m",
    "kv_cache_quant": "turboquant:4"
  },

  // 3. Full form with speculative decoding overrides
  "llama3:70b": {
    "hf_path": "mlx-community/Meta-Llama-3-70B-Instruct-4bit",
    "keep_alive": "1h",
    "speculative": true,
    "speculative_draft_model": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    "speculative_tokens": 4,
    "speculative_strategy": "classic",
    "kv_cache_quant": "spectral:4",
    "sync_mode": "full",
    "options": {
      "temperature": 0.7,
      "top_p": 0.9
    }
  }
}
```

### Alias normalization

Lookups normalize names by appending `:latest` when no tag is present, so
`qwen3` and `qwen3:latest` resolve to the same entry. See
`ModelRegistry.normalizeName` in `Sources/OLMLX/Models/Registry.swift`.

### Per-model config

| Field | Type | Notes |
| --- | --- | --- |
| `hf_path` | string | **Required.** Must be `namespace/name` |
| `keep_alive` | string | Same format as `OLMLX_DEFAULT_KEEP_ALIVE` (`30s`, `5m`, `2h`). `0` pins the model indefinitely (only LRU eviction can remove it). |
| `options` | object | Default `ModelOptions` for this model (see [Common schemas](api/ollama.md#options)) |
| `sync_mode` | enum | `full` \| `minimal` \| `none` |
| `speculative` | bool | Enable speculative decoding for this model |
| `speculative_draft_model` | string | HF path of the draft model |
| `speculative_tokens` | int | Lookahead window |
| `speculative_strategy` | enum | `classic` \| `dflash` \| `eagle` |
| `kv_cache_quant` | string | `turboquant:2`/`turboquant:4`/`spectral:2`/`spectral:4` |
| `experimental` | object | Free-form pass-through bag, not currently consumed |

Per-model values override the global `OLMLX_*` defaults.

## Downloading models

There are three ways to get model weights onto disk:

1. **Lazy pull on first use.** Any chat / generate request that references a
   registered alias triggers a snapshot download if the directory is missing.
2. **Explicit pull via the API.**
   ```sh
   curl -s http://localhost:11434/api/pull \
     -H 'content-type: application/json' \
     -d '{"model":"qwen3:8b","insecure":false,"stream":false}'
   ```
   Both `insecure` and `stream` are required fields on the request — the
   server's decoder will reject the body without them.
3. **Manually.** Drop a HuggingFace snapshot into
   `~/.olmlx/models/<namespace--name>/`. Make sure the directory contains the
   tokenizer + safetensors + config that the MLX loader expects.

## Listing what's registered

```sh
olmlx models list
# llama3:8b
# qwen3:8b
```

Or over HTTP:

```sh
curl -s http://localhost:11434/api/tags | jq .
curl -s http://localhost:11434/v1/models | jq .   # OpenAI shape
```

## Showing model details

```sh
olmlx models show qwen3:8b
```

Prints registry config + on-disk manifest (size, family, parameter size,
quantization level) if the snapshot exists.

## Deleting a model

```sh
olmlx models delete qwen3:8b           # removes from registry + disk
# or
curl -X DELETE http://localhost:11434/api/delete \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3:8b"}'           # removes from disk only
```

The HTTP `DELETE /api/delete` only touches the on-disk snapshot; it leaves
the registry entry in place. The CLI removes both.

## What's not implemented

The following endpoints respond `501 Not Implemented`:

- `POST /api/create` — building models from a Modelfile
- `POST /api/copy` — duplicating models
- `POST /api/abort` — cancelling an in-flight request
- `POST /api/embed` / `POST /api/embeddings` / `POST /v1/embeddings` — embeddings
- `POST /v1/completions` — legacy OpenAI text-completion
- `POST /v1/messages/count_tokens` — Anthropic token count

If you need any of these for a tool you're integrating, file an issue or
contribute. The schemas are already wired up — only the inference path is
missing.

## Manifest format

When `olmlx` knows a model's metadata it writes `manifest.json` alongside the
weights. Format:

```json
{
  "name": "qwen3:8b",
  "hf_path": "mlx-community/Qwen3-8B-4bit",
  "size": 5147483648,
  "modified_at": "2026-05-18T15:30:00Z",
  "digest": "sha256:abcdef012345",
  "format": "mlx",
  "family": "qwen2",
  "parameter_size": "8B",
  "quantization_level": "q4"
}
```

The digest is a SHA-256 prefix of the alias name, not of the actual weights.
It is used to populate the `digest` field on the Ollama API responses for
client compatibility — do not rely on it for integrity checking.
