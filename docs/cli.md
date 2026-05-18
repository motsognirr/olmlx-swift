# CLI Reference

`olmlx` is a single binary with several subcommands. With no subcommand it
falls through to `serve`. Source: `Sources/olmlx-cli/main.swift`.

```
olmlx [SUBCOMMAND] [OPTIONS]
```

| Subcommand | Purpose |
| --- | --- |
| `serve` (default) | Start the HTTP API server |
| `models list` | List registered models |
| `models show <name>` | Show model details and on-disk manifest |
| `models delete <name>` | Remove model from registry and disk |
| `models search <query>` | Substring-match model names |
| `chat <model>` | Interactive terminal chat *(stub)* |
| `bench run` | Run a benchmark *(stub)* |
| `bench list` | List benchmark results *(stub)* |
| `config show` | Print the resolved configuration |
| `service install` | Install a launchd LaunchAgent *(stub)* |
| `service status` | Check launchd service status *(stub)* |

`--version` prints `0.1.0`. `--help` works at every level (e.g.
`olmlx models --help`).

> **Stubs** listed above are present in the binary but currently print a
> placeholder message; do not rely on them. Track issues on the project tracker.

---

## `serve`

Starts the Vapor HTTP server on the configured host and port. This is the
default subcommand, so `olmlx` and `olmlx serve` are equivalent.

```
olmlx serve [--host HOST] [--port PORT]
            [--speculative]
            [--kv-cache-quant turboquant:4 | spectral:4 | turboquant:2 | spectral:2]
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--host` | string | `0.0.0.0` (or `OLMLX_HOST`) | Address to bind |
| `--port` | int | `11434` (or `OLMLX_PORT`) | Port to listen on |
| `--speculative` | flag | off | Enable speculative decoding (requires a draft model — see [Configuration](configuration.md)) |
| `--kv-cache-quant` | string | none | KV-cache quantization in `method:bits` form (`turboquant`/`spectral` × `2`/`4`) |

CLI flags override the corresponding `OLMLX_*` environment variables for that
process only.

### Behavior

1. Resolves `Settings` from the environment.
2. Loads `~/.olmlx/models.json` into a `ModelRegistry`.
3. Constructs the `ModelStore` (HuggingFace downloads) and `ModelManager`.
4. Starts the keep-alive expiry checker (runs every 30s).
5. Builds the Vapor app, registers Ollama / OpenAI / Anthropic routes, and
   blocks on `app.execute()`.

### Stopping the server

`Ctrl-C` (SIGINT) or `SIGTERM` shuts the server down cleanly. There is no
graceful drain — in-flight requests are interrupted.

---

## `models list`

Prints every name from `~/.olmlx/models.json`, one per line. Names are
sorted lexicographically.

```sh
olmlx models list
# llama3:8b
# qwen3:8b
```

If no models are registered, prints `No models registered`.

---

## `models show <name>`

Dumps registry config and on-disk manifest for a single model.

```sh
olmlx models show qwen3:8b
# Model: qwen3:8b
# HF Path: mlx-community/Qwen3-8B-4bit
# Draft: mlx-community/Qwen3-0.5B-4bit            # if speculative_draft_model set
# KV Cache Quant: turboquant:4                    # if configured
# Size: 5147483648 bytes                          # only if manifest.json exists
# Family: qwen2
# Parameters: 8B
# Quantization: q4
```

If the model is registered but hasn't been downloaded yet, only the
configuration is printed (no manifest).

---

## `models delete <name>`

Removes the registry entry, persists the updated `models.json`, and deletes
the on-disk snapshot directory.

```sh
olmlx models delete qwen3:8b
# Deleted model 'qwen3:8b'
```

This is destructive. There is no `--dry-run`. The HTTP equivalent is
`DELETE /api/delete`.

---

## `models search <query>`

Case-insensitive substring match over registered model names.

```sh
olmlx models search qwen
# qwen3:8b
```

---

## `chat <model>` *(stub)*

Interactive REPL. Currently does **not** call the inference engine — it just
echoes a placeholder. Real implementation is pending.

```
olmlx chat qwen3:8b
> hello
Assistant: [response from qwen3:8b]
> /exit
```

Supported slash commands: `/exit`, `/help`, (planned: `/clear`, `/system`).

---

## `bench run` / `bench list` *(stubs)*

Both print placeholder messages. Benchmarking is not implemented.

---

## `config show`

Prints the resolved `Settings` struct — useful for verifying that your
environment variables are being picked up.

```sh
olmlx config show
# Host: 0.0.0.0:11434
# Models Dir: /Users/<you>/.olmlx/models
# Models Config: /Users/<you>/.olmlx/models.json
# Keep Alive: 5m
# Max Loaded: 1
# Memory Limit: 0.75
# Prompt Cache: true
# Speculative: false
# KV Cache Quant: none
# Sync Mode: full
# Log Level: INFO
```

See [Configuration](configuration.md) for the full list of `OLMLX_*` variables.

---

## `service install` / `service status` *(stubs)*

Print placeholders. Wiring up a real launchd LaunchAgent is left to the
operator for now — drop a `.plist` into `~/Library/LaunchAgents/` that
invokes `/usr/local/bin/olmlx serve` with whatever `OLMLX_*` environment you
want. Example skeleton:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.olmlx</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/olmlx</string>
    <string>serve</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>OLMLX_PORT</key><string>11434</string>
    <key>OLMLX_MAX_LOADED_MODELS</key><string>2</string>
  </dict>
  <key>KeepAlive</key><true/>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>/tmp/olmlx.out.log</string>
  <key>StandardErrorPath</key><string>/tmp/olmlx.err.log</string>
</dict>
</plist>
```

Load with `launchctl load ~/Library/LaunchAgents/com.olmlx.plist`.
