# Architecture Overview

A whirlwind tour for people who want to read the code or extend it.

## Components

```
┌──────────────────────────┐
│   olmlx-cli/main.swift   │     argument-parser entry point
│   (AsyncParsableCommand) │
└────────────┬─────────────┘
             │ creates
             ▼
┌─────────────────────────────────────────────┐
│              Settings (struct)               │   immutable snapshot of OLMLX_* env
├─────────────────────────────────────────────┤
│  ModelRegistry (actor)                      │   maps alias → ModelConfig (HF path + opts)
│  └─ persisted at ~/.olmlx/models.json       │
├─────────────────────────────────────────────┤
│  ModelStore                                 │   downloads + on-disk layout
│  └─ HuggingFace snapshot at                 │
│     ~/.olmlx/models/<ns--name>/             │
├─────────────────────────────────────────────┤
│  ModelManager (actor)                       │   in-memory model map, keep-alive,
│  ├─ DefaultInferenceEngine                  │   LRU eviction, prompt cache
│  └─ PromptCacheStore (actor)                │
└─────────────────────────────────────────────┘
             │ injected into app.storage
             ▼
┌─────────────────────────────────────────────┐
│              Vapor Application               │
│  ├─ CORSMiddleware                          │
│  ├─ RequestIDMiddleware (X-Request-ID)      │
│  ├─ /api/*  → OllamaRoutes                  │
│  ├─ /v1/*   → OpenAIRoutes                  │
│  └─ /v1/messages → AnthropicRoutes          │
└─────────────────────────────────────────────┘
```

## Source map

- `Sources/olmlx-cli/main.swift` — CLI entry point. `@main` on
  `OLMXCLI: AsyncParsableCommand`. Subcommands wire each `serve` / `models`
  / etc. handler.
- `Sources/OLMLX/App.swift` — `createApp(registry:store:manager:)` factory.
  Bootstraps logging, stashes registry/store/manager in `Application.storage`,
  mounts middleware and routes.
- `Sources/OLMLX/Config/Settings.swift` — env-var parsing. Single
  `Settings.init(env:)` that handles every `OLMLX_*` var with safe fallbacks.
- `Sources/OLMLX/Models/Registry.swift` — `ModelRegistry` actor with
  load/save against `models.json`, lookup with `:latest` normalization, and
  the `ModelConfig` decoder that tolerates string-shorthand and full-form
  entries.
- `Sources/OLMLX/Models/Store.swift` — `ModelStore` (pure `Sendable`, not an
  actor). Translates HF paths to local dirs, calls
  `swift-huggingface`'s `HubClient.downloadSnapshot`, cleans up partial
  downloads on failure.
- `Sources/OLMLX/Models/Manifest.swift` — `manifest.json` schema and
  SHA-256-prefix digest helper.
- `Sources/OLMLX/Engine/ModelManager.swift` — central in-memory state.
  `ensureLoaded(name:keepAlive:)` is the hot path: registry lookup →
  on-disk ensure → engine load → keep-alive bookkeeping → LRU eviction →
  cache. `startExpiryChecker()` runs every 30s to drop expired entries.
- `Sources/OLMLX/Engine/InferenceEngine.swift` —
  `InferenceEngineProtocol` + `DefaultInferenceEngine` (wraps
  `MLXLMCommon.loadModelContainer`) + `MockInferenceEngine` (tests).
  `mapToGenerateParameters` converts Ollama-style options to
  `MLXLMCommon.GenerateParameters`. `runGeneration` / `runStreamingGeneration`
  are the two generation entry points; they apply the tokenizer chat
  template, build an `LMInput`, and consume the MLX async stream.
- `Sources/OLMLX/Extensions/ExtensionEntry.swift` — value types for the
  extension layer: `ExtensionKind`, `RemovalCondition` (+ its
  `isRemovable(pinnedVersion:)` gate), and `ExtensionEntry`.
- `Sources/OLMLX/Extensions/OLMLXExtensions.swift` — the `manifest` of
  temporary architectures, `register(_:onto:)`, the once-per-process
  `registerAll()`, and the `listSummary` / `removableEntries` reporting
  helpers. See "Extension layer" below.
- `Sources/olmlx-cli/ExtensionsCommand.swift` — `olmlx ext list` /
  `olmlx ext check` subcommands (thin wrappers over the helpers above).
- `Sources/OLMLX/Engine/ToolParser.swift` — best-effort parser for the
  six-ish tool-call formats popular models emit. Reaches into the model's
  raw text output and returns `ToolUse` structs.
- `Sources/OLMLX/Engine/TemplateCaps.swift` — feature flags inferred from
  the tokenizer's chat template (does it support tools? thinking? channel
  format?). Currently called with `nil` so all caps default to `false`.
- `Sources/OLMLX/Engine/Inference.swift` — `InferenceOptions` (a smaller
  parameter struct than `ModelOptions`) and a stub `countChatTokens`.
- `Sources/OLMLX/Routes/Middleware.swift` —
  `RequestIDMiddleware`, the `StorageKey`s used to stash actors in
  `Application.storage`, and the CORS configuration.
- `Sources/OLMLX/Routes/OllamaRoutes.swift` — every `/api/*` route.
- `Sources/OLMLX/Routes/OpenAIRoutes.swift` — `/v1/chat/completions`,
  `/v1/models`.
- `Sources/OLMLX/Routes/AnthropicRoutes.swift` — `/v1/messages`.
- `Sources/OLMLX/Routes/ResponseHelpers.swift` — the streaming /
  non-streaming response builders shared by `/api/chat` and
  `/api/generate`, plus the "model not loaded" fallback paths.
- `Sources/OLMLX/Schemas/` — Codable structs for every request and
  response shape (split per surface).
- `Sources/OLMLX/Utils/` — small helpers (`Streaming`, `Timing`, `Memory`).
- `Sources/OLMLX/VaporContentExtensions.swift` — conformances so plain
  `Codable` structs work as Vapor `Content`.

## Request lifecycle (chat)

1. Client POSTs JSON to `/api/chat`.
2. `CORSMiddleware` + `RequestIDMiddleware` add headers; the request gets a
   fresh `X-Request-ID`.
3. Route handler decodes `ChatRequest`. Validation throws
   `SchemaValidationError` ⇒ Vapor 400.
4. `ModelManager.ensureLoaded(name:, keepAlive:)` runs:
   - Normalize the name (`qwen3` → `qwen3:latest`).
   - Hit cache; if loaded, refresh expiry and return.
   - Resolve to a `ModelConfig` via `ModelRegistry`.
   - `ModelStore.ensureDownloaded(hfPath:)` — pulls from HF if missing.
   - `DefaultInferenceEngine.loadModel(from:)` — builds an MLX
     `ModelContainer`. This is where the metallib gets exercised.
   - Compute expiry from `keep_alive`, evict oldest if at max-loaded,
     insert.
5. Build `GenerateParameters` from `options` via
   `mapToGenerateParameters`.
6. If `stream: true`, call `streamingChatResponse` — wraps a Vapor body
   stream, drives `runStreamingGeneration`, and writes one NDJSON object
   per chunk.
7. If `stream: false`, call `fullChatResponse` — drives `runGeneration`,
   collects full text, returns a single `ChatResponse`.

The OpenAI and Anthropic surfaces follow the same path but use their own
request/response schemas and currently always take the non-streaming
branch.

## Concurrency model

- `ModelRegistry`, `ModelManager`, and `PromptCacheStore` are Swift actors —
  serialized access for free.
- `ModelStore` is a plain `Sendable` class because it has no mutable
  state (every method derives its result from `modelsDir` + the registry).
- `Settings` is an immutable `Sendable` struct; pass it by value.
- The default route handlers use `async throws` and `await` actor calls
  directly. There is no separate executor or queue.

## Extension layer

Model-architecture and inference-feature support normally comes from the
upstream `mlx-swift-lm` package. When upstream lags — a new `model_type`
isn't registered yet, or a known architecture has a bug — the extension
layer lets us ship *temporary, tracked* support in-tree instead of waiting.
Background and policy: `docs/superpowers/specs/2026-05-19-architecture-feature-extension-layer-design.md`.

**How registration works.** `mlx-swift-lm` exposes `LLMTypeRegistry.shared`,
a mutable `actor ModelTypeRegistry` keyed by `model_type`.
`OLMLXExtensions.registerAll()` layers every entry in
`OLMLXExtensions.manifest` onto that shared registry. It runs once per
process — `DefaultInferenceEngine.loadModel` awaits it before every container
load, so the existing `MLXLMCommon.loadModelContainer` path (including its
VLM-factory fallback) is unchanged. Registering our own `model_type` simply
adds a creator; registering an existing one overrides upstream's.

**Adding an entry.** Implement the architecture as a type conforming to
`LanguageModel` (one file per architecture under `Sources/OLMLX/Extensions/`),
then append an `ExtensionEntry` to `manifest` with:
- `creator: OLMLXExtensions.creator(YourConfig.self, YourModel.init)`,
- a mandatory `upstreamTracking` URL (the issue/PR you're working around — no
  entry without one; open one first if it doesn't exist), and
- a `removeWhen` condition: `.upstreamReleased(version:)` for "delete once we
  pin this version" or `.upstreamMerged(pr:)` for "delete after this PR lands"
  (the latter is never auto-gated and needs a manual check).

**Removing an entry.** Run `olmlx ext check --pinned-version <current-mlx-swift-lm>`.
If it names an entry, that version-gated workaround is now redundant: delete
its `ExtensionEntry` and architecture file, then bump the pinned upstream
version. `olmlx ext list` prints the full manifest with tracking links. The
`olmlx_canary` entry is a permanent self-test (maps an unused `model_type`
onto a trivial weightless built-in `CanaryModel` so the registration path stays
exercised in CI without needing the MLX metal library) and is never removed.

**Discipline.** Keep ≤5 active architecture overrides; beyond that, push the
work upstream and wait. One file per architecture so removal is a clean
delete. Every entry carries an upstream tracking URL.

## Tests

Lightweight unit tests live in `Tests/OLMLXTests/`:

- `ConfigTests` — `Settings` env-var parsing.
- `ModelsTests` — registry serialization, lookup, alias normalization.
- `EngineTests` / `InferenceEngineTests` — option mapping, mock engine
  wiring.
- `SchemaTests` — Codable round-trips and validation for the request/
  response types.
- `UtilsTests` — streaming/timing/memory helpers.
- `ExtensionsTests` — removal-condition logic, registry registration
  (against the real `ModelTypeRegistry`), idempotent `registerAll`, and the
  `ext` reporting helpers.

No integration tests against a real model — those would require the
metallib and a downloaded snapshot. Add them locally if you're changing the
inference path.
