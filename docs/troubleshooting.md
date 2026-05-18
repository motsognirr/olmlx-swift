# Troubleshooting

Things that bite people when running `olmlx`, and how to unstick them.

## `MLX error: Failed to load the default metallib`

The very first GPU op crashes the process with this. Cause: `mlx-swift`
doesn't ship the metallib through SwiftPM, so a vanilla `swift build` links
against MLX but produces a binary that can't actually run GPU kernels.

**Fix:** compile the metallib once via CMake and drop it next to the
binary. The full incantation lives in
[Installation §3](installation.md#3-build-the-metal-kernel-library-required).
TL;DR:

```sh
REPO=$(pwd)
cd /tmp/mlx-build && cmake "$REPO/.build/checkouts/mlx-swift/Source/Cmlx/mlx" \
  -DMLX_BUILD_METAL=ON -DMLX_BUILD_TESTS=OFF ...
cmake --build . --target mlx-metallib
cp mlx/backend/metal/kernels/mlx.metallib "$REPO/.build/release/mlx.metallib"
```

The metallib must travel with the binary — `cp olmlx /usr/local/bin/`
without the sibling `mlx.metallib` will fail on the next request.

---

## Every subcommand prints help

If `olmlx serve`, `olmlx models list`, and even `olmlx config show` all
print the help text and exit `0` instead of doing anything, the `@main`
attribute on `OLMXCLI` has been removed (or a trailing `OLMXCLI.main()` call
has been re-added at the bottom of `Sources/olmlx-cli/main.swift`).

Why: `OLMXCLI` is an `AsyncParsableCommand`. A bare `OLMXCLI.main()` at top
level (no `await`) resolves to the synchronous `ParsableCommand.main()`
overload, which for async subcommands like `Serve` throws
`CleanExit.helpRequest` — and exits 0 after printing help.

**Fix:** keep `@main` on the `OLMXCLI` struct and remove any trailing
`.main()` call. The argument parser's `@main` macro wires up the async entry
point correctly.

---

## `bind: Address already in use`

Port 11434 is the same port the real `ollama` daemon uses. Stop Ollama or
use a different port.

```sh
# stop the official Ollama daemon if installed
launchctl stop com.ollama.ollama
killall ollama

# or pick a different port
olmlx serve --port 11435
# or
OLMLX_PORT=11435 olmlx serve
```

---

## `/api/pull` rejects my JSON body

The Swift decoder for `PullRequest` treats both `insecure` and `stream` as
required even though they have Swift default values:

```sh
# fails
curl -d '{"model":"qwen3:8b"}' http://localhost:11434/api/pull

# works
curl -d '{"model":"qwen3:8b","insecure":false,"stream":true}' \
     http://localhost:11434/api/pull
```

Same applies to a couple of other Ollama-shaped requests. If you get a 400
on a body that "should" work for upstream Ollama, fill in every field
mentioned in [the API doc](api/ollama.md).

---

## Model output contains `<think>...</think>` traces

Qwen3 (and a few other "reasoning" models) emit chain-of-thought traces
inline in `message.content`. Two ways to deal with it:

1. **Budget for it.** Bump `num_predict` (Ollama) or `max_tokens`
   (OpenAI/Anthropic) so the model finishes both the trace and the answer.
2. **Strip it client-side.** Regex out `<think>[\s\S]*?</think>` before
   showing to the user.

There is no server-side post-processing of think traces today.

---

## Streaming requests come back as a single JSON body

Only the Ollama `/api/chat` and `/api/generate` routes actually stream.
The OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` routes
accept `stream: true` in the schema but currently return a single complete
JSON object regardless.

If you need real streaming, route through `/api/chat` or wait for the
OpenAI/Anthropic streaming work.

---

## Model downloads are slow / partial

`/api/pull` is currently synchronous — it returns one `{"status":"success"}`
after the full HuggingFace snapshot is on disk. If the download is
interrupted, the half-written directory is removed automatically; you can
just call `pull` again.

Speed depends on the HuggingFace CDN and your link; 5–10 GB models take
several minutes on a typical home connection. For repeated installs, pre-
populate `~/.olmlx/models/<namespace--name>/` from a manual `huggingface-cli
download` or an `rsync` from another machine.

---

## `model not found` on a name that's in `models.json`

The registry normalizes lookups: a name without a `:` tag gets `:latest`
appended. So `qwen3` is equivalent to `qwen3:latest`, but `qwen3:8b` is its
own entry — make sure you're requesting whatever tag you registered.

```sh
olmlx models list      # see what's actually registered
olmlx models search q  # substring match
```

If the list is empty, `~/.olmlx/models.json` is missing or unreadable. Run
`olmlx config show` to confirm the resolved path.

---

## Memory pressure / OOM

MLX is allowed to use up to `OLMLX_MEMORY_LIMIT_FRACTION` (default `0.75`) of
system memory. On an 8 GB Mac that's 6 GB — enough for a 4-bit 7B/8B model
but tight. Levers:

- **Lower the fraction.** `OLMLX_MEMORY_LIMIT_FRACTION=0.6` to give the OS
  more headroom.
- **One model at a time.** `OLMLX_MAX_LOADED_MODELS=1` (the default) keeps
  only the newest model in RAM.
- **Quantize the KV cache.** `OLMLX_KV_CACHE_QUANT=turboquant:4` or
  `spectral:2` shrinks the per-token KV memory at some quality cost.
- **Lower the prompt-cache budget.** `OLMLX_PROMPT_CACHE_MAX_SLOTS=1`
  and/or `OLMLX_PROMPT_CACHE_MAX_TOKENS=8192`.

---

## CORS error from a browser app

Default allowed origins are `http://localhost:*` and `http://127.0.0.1:*`.
A page served from another host or scheme (e.g. an https:// production URL)
will be blocked.

Override:

```sh
OLMLX_CORS_ORIGINS="https://myapp.example.com,http://localhost:*" olmlx serve
```

The middleware uses Vapor's `CORSMiddleware` with `allowedOrigin: .originBased`,
which echoes the request's Origin header when it matches the list.

---

## How do I see what the server actually decided?

```sh
olmlx config show           # resolved Settings
curl /api/tags              # registered models
curl /api/ps                # loaded (in-memory) models
curl /api/version           # version sanity check
```

And the server logs every request with a `X-Request-ID` you can grep for.

---

If you hit something not on this list, capture the request, response, and
the server log lines for the matching `X-Request-ID`, then file an issue.
