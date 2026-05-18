# olmlx-swift User Manual

`olmlx` is a drop-in replacement for the Ollama API powered by Apple's
[MLX](https://github.com/ml-explore/mlx-swift) framework. It runs locally on
Apple Silicon, downloads models directly from HuggingFace, and exposes three
parallel HTTP surfaces — Ollama, OpenAI, and Anthropic — so that existing
clients work unchanged.

This manual covers how to build, configure, and operate the server, plus the
full HTTP API and CLI surface.

## Contents

1. [Installation & Build](installation.md) — clone, build, and the metallib
   step required for GPU inference.
2. [Quickstart](quickstart.md) — start the server, register a model, send your
   first request.
3. [CLI Reference](cli.md) — every `olmlx` subcommand and option.
4. [Configuration](configuration.md) — `OLMLX_*` environment variables and
   their defaults.
5. [Model Management](models.md) — `~/.olmlx/models.json`, the on-disk layout,
   pulling, deleting, and per-model settings.
6. [HTTP API Reference](api/README.md)
   - [Ollama-compatible API (`/api/*`)](api/ollama.md)
   - [OpenAI-compatible API (`/v1/*`)](api/openai.md)
   - [Anthropic-compatible API (`/v1/messages`)](api/anthropic.md)
7. [Troubleshooting](troubleshooting.md) — common failure modes and fixes.
8. [Architecture Overview](architecture.md) — internals, request flow, and
   where to look in the source tree.

## At a glance

| Item | Default |
| --- | --- |
| Bind address | `0.0.0.0:11434` |
| Model registry | `~/.olmlx/models.json` |
| Model snapshots | `~/.olmlx/models/<namespace--name>/` |
| Disk KV cache | `~/.olmlx/cache/kv/` |
| Keep-alive | 5 minutes |
| Concurrently loaded models | 1 |
| Memory limit fraction | 0.75 |

> **Heads up:** port 11434 is the same port the real Ollama daemon uses. Stop
> Ollama before running `olmlx`, or pick a different port via `OLMLX_PORT` or
> `--port`.

## Project layout

```
Sources/
  olmlx-cli/         # argument-parser CLI entry point (`olmlx` binary)
  OLMLX/
    App.swift        # Vapor application factory
    Config/          # Settings loaded from OLMLX_* env vars
    Engine/          # ModelManager, inference, tool-call parsing
    Models/          # Registry, on-disk store, manifests
    Routes/          # Ollama / OpenAI / Anthropic HTTP routes
    Schemas/         # Request/response types per API surface
    Utils/           # Streaming, timing, memory helpers
Tests/OLMLXTests/    # Unit tests
```
