# Architecture & Feature Extension Layer — Strategy Design

**Date:** 2026-05-19
**Status:** Draft for review
**Context for:** issues #57, #58, #59, #61 (and the broader pattern they expose)

## 1. Context

`olmlx-swift` currently delegates all model-architecture and inference-feature support to upstream `mlx-swift-lm` (pinned at 3.31.3). Recent issues have exposed that this strategy is fragile:

- **4 of the last 5 open issues** (#57, #58, #59, #61) are blocked on upstream PRs or missing upstream support.
- The Gemma quantization fix ([mlx-swift-lm#244](https://github.com/ml-explore/mlx-swift-lm/pull/244)) has been stalled for 3 weeks awaiting an author response.
- New popular architectures (`minimax_m2`, `step3p5`, Gemma-4 MoE) reach disk but fail at construction because no entry exists in upstream `LLMTypeRegistry.shared`.
- The Python sibling project `../olmlx` (via `mlx-lm`) supports **~119 architectures** plus richer features (DFlash/Eagle speculative decoding, SpectralQuant/TurboQuant KV cache, Flash-MoE sparse loading, distributed inference). `mlx-swift-examples` supports closer to ~10.

Today, `olmlx-swift` has no in-tree extension surface: the integration is essentially one call to `MLXLMCommon.loadModelContainer()` and one to `MLXLMCommon.generate()`. When upstream lacks a model or feature, we wait.

**Goal:** introduce a unified extension layer that lets us implement *temporary* support for new architectures and features in olmlx-swift itself, with a disciplined path back to upstream once support lands.

## 2. Key technical finding: upstream is more extensible than we've been using

A targeted audit of mlx-swift-lm's public API surface shows we can extend it **without forking**:

| Capability | Public extension point | Verdict |
|---|---|---|
| Register new `model_type` strings | `ModelTypeRegistry.registerModelType(_:creator:)` (actor, public) | ✅ Yes |
| Implement custom architectures | `LanguageModel` / `LLMModel` / `VLMModel` protocols (public) | ✅ Yes |
| Use our own registry instance | `LLMModelFactory.init(typeRegistry:...)` (public) | ✅ Yes |
| Custom KV cache (incl. quant) | `KVCache` / `QuantizedKVCacheProtocol` (public protocols) | ✅ Yes |
| Custom samplers/processors | `LogitSampler` / `LogitProcessor` (public) | ✅ Yes |
| Custom generation loop | `TokenIterator` / `TokenIteratorProtocol` (public) | ✅ Yes |
| Speculative decoding variants | `SpeculativeTokenIterator` public; no high-level hook | ⚠️ Partial — usable but underdeveloped |
| Pluggable weight loader | Only `model.sanitize(weights:metadata:)` is exposed | ⚠️ Partial — works for remapping, not format changes |

This rules out the "fork mlx-swift-lm" option as a default strategy. We can build the extension layer as a normal Swift module that consumes the existing package via its public API.

## 3. Strategy

### 3.1 Recommended approach: an in-tree `Extensions` module, fronted by our own factory

Introduce a new submodule `Sources/OLMLX/Extensions/` that:

1. Owns an **`OLMLXTypeRegistry`** — a project-controlled `LLMTypeRegistry` seeded from `LLMTypeRegistry.shared` and then augmented with our temporary architectures.
2. Exposes an **`OLMLXModelFactory`** wrapping `LLMModelFactory(typeRegistry: ours)`, used by `DefaultInferenceEngine.loadModel()` instead of the package default.
3. Holds **architecture implementations** (`Extensions/Architectures/<ModelType>.swift`) that conform to `LLMModel` and are temporary by design.
4. Holds **feature extensions** (`Extensions/Caches/`, `Extensions/Sampling/`, `Extensions/Generation/`) that plug into the public hooks.

### 3.2 Why not fork mlx-swift-lm

- High maintenance cost: every upstream release means resolving conflicts.
- Coupling: our `mlx-swift-lm` version is tightly bound to `mlx-swift` ABI; a fork pins us out of upstream tag streams.
- The audit showed forking isn't necessary — public APIs cover the high-value 90%.

### 3.3 Why not vendor specific files

- Tempting for surgical bug fixes (e.g., the Gemma `ScaledLinear` quant fix), but it produces hidden divergence that's easy to forget about.
- We can achieve the same result by *re-registering* our patched architecture under the same `model_type` in `OLMLXTypeRegistry` and letting it override the inherited shared entry.

## 4. Architecture

```
Sources/OLMLX/Extensions/
├── OLMLXTypeRegistry.swift        # Builds the augmented LLMTypeRegistry
├── OLMLXModelFactory.swift        # Wraps LLMModelFactory(typeRegistry:)
├── ExtensionManifest.swift        # Lists temp entries + upstream tracking metadata
├── Architectures/
│   ├── MinimaxM2.swift            # implements LLMModel
│   ├── Step3p5.swift
│   ├── Gemma4MoE.swift            # MoE variant missing upstream (#58)
│   └── GemmaScaledLinearPatched.swift  # workaround for #57 (override)
├── Caches/
│   └── (future: SpectralQuantKVCache.swift, TurboQuantKVCache.swift)
├── Sampling/
│   └── (future: custom LogitProcessors)
└── Generation/
    └── (future: DFlashIterator.swift, EagleIterator.swift)
```

**Integration point** — single change in the engine:

- `Sources/OLMLX/Engine/InferenceEngine.swift:73-76` — `DefaultInferenceEngine.loadModel()` currently calls `MLXLMCommon.loadModelContainer(from:using:tokenizerLoader)`. Replace with a path that uses `OLMLXModelFactory` so registry overrides take effect.
- `Sources/OLMLX/Engine/ModelManager.swift:90-101` — `wrapMLXLoadError()` keeps current `unsupportedArchitecture` mapping; behaviour only changes for the architectures we've registered.

**Telemetry / tracking** — `ExtensionManifest` is the source of truth for what is "temporary":

```swift
struct ExtensionEntry {
    let modelType: String           // e.g. "minimax_m2"
    let kind: ExtensionKind         // .architecture | .feature
    let upstreamTracking: URL?      // PR or issue URL
    let addedOn: Date
    let removeWhen: RemovalCondition // .upstreamMerged | .upstreamReleased(version: "3.32.0")
    let notes: String
}
```

The CLI exposes `olmlx ext list` to print the manifest, so it's easy to audit "what are we still patching?"

## 5. Lifecycle / migration discipline

A temporary fix only stays cheap if removal is cheap. Rules:

1. **Every entry has an upstream tracking URL.** No exceptions. If no upstream issue/PR exists, we open one first.
2. **CI guard** — a small script (`Scripts/check-extensions.swift`) fails CI if an entry's `removeWhen` condition has been satisfied (e.g., a release containing the upstream fix is now the pinned version).
3. **Override semantics are explicit.** Architecture overrides re-register under the same `model_type`; on upstream uptake we delete our file, the registry returns to the inherited shared entry, and the test suite catches regressions.
4. **One file per architecture / feature.** No shared "extensions kitchen sink" file — easy to delete cleanly.
5. **No coupling to internal `mlx-swift-lm` symbols.** If a fix would require touching `internal` upstream code, file an upstream PR and accept the wait rather than reaching for a fork.

## 6. Phasing

| Phase | Scope | Unblocks |
|---|---|---|
| **1. Foundation** | `OLMLXTypeRegistry` + `OLMLXModelFactory` + manifest plumbing + `loadModel()` rewrite | Mechanism only — no new model yet |
| **2. Architecture overrides** | Add `Gemma4MoE` (#58), `MinimaxM2` + `Step3p5` (#61), patched Gemma quant (#57) | 3-4 of 5 current issues |
| **3. Attention bug bisect** | #59 (Gemma-4-31B broadcast) — likely fixed by reimplementing `Gemma4Attention` as override; also adds a public `recover` path so we don't `fatalError` at the MLX boundary | #59 |
| **4. Feature hooks (KV cache)** | Implement one `KVCache` conformer from `../olmlx` (start with SpectralQuant) to prove the feature-extension path | Feature-gap precedent |
| **5. Speculative decoding variants** | Port DFlash and/or Eagle from `../olmlx` using `TokenIteratorProtocol` | Feature parity |

Phases 1-3 are the immediate value. Phases 4-5 validate the layer for non-architecture extensions and should only start once Phase 1-3 has shipped and proven stable.

## 7. Critical files

- **New:** `Sources/OLMLX/Extensions/OLMLXTypeRegistry.swift`
- **New:** `Sources/OLMLX/Extensions/OLMLXModelFactory.swift`
- **New:** `Sources/OLMLX/Extensions/ExtensionManifest.swift`
- **New:** `Sources/OLMLX/Extensions/Architectures/*.swift`
- **Modify:** `Sources/OLMLX/Engine/InferenceEngine.swift:73-76` (use new factory)
- **Modify:** `Sources/OLMLX/Engine/ModelManager.swift:90-101` (refine error reporting to distinguish "no upstream + no override" from "override exists but failed")
- **New:** `Sources/olmlx-cli/Commands/ExtensionsCommand.swift` (`olmlx ext list`)
- **New:** `Scripts/check-extensions.swift` (CI guard)
- **New:** `Tests/OLMLXTests/Extensions/RegistryTests.swift`

Reuse where possible: tokenizer plumbing (`TokenizerAdapter`, `MLXTokenizerLoader` in `InferenceEngine.swift:9-52`) is already correct and stays unchanged.

## 8. Risks & trade-offs

- **Maintenance creep.** Architectures are non-trivial to port from Python. Mitigation: a hard upper bound (e.g., ≤5 active overrides at any time) — beyond that, we push back to upstream and wait.
- **Subtle correctness drift.** A model we registered may produce slightly different outputs than the eventual upstream version. Mitigation: golden-output tests pinned to a small prompt set, re-checked against upstream when the override is removed.
- **`SpeculativeTokenIterator` is thin.** Phase 5 may require deeper work than expected, or an upstream contribution. That's acceptable — speculative variants are bonus, not baseline.
- **Quantization-format support.** Some `../olmlx` features (SpectralQuant) require weight-format hooks we don't fully control. We can do per-architecture `sanitize(weights:)` transforms but not arbitrary file-format changes. Scope Phase 4 accordingly.

## 9. Verification plan

1. **Registry mechanism** — add a synthetic dummy `model_type` (`olmlx_test`), verify `loadModelContainer` resolves it through `OLMLXModelFactory`, fails through stock `MLXLMCommon.loadModelContainer`.
2. **Issue #61** — pull a `minimax_m2` model, run `olmlx pull` + `olmlx run` → first token streams successfully.
3. **Issue #58** — same for Gemma-4-26B-A4B MoE.
4. **Issue #57** — load a Gemma-4 e2b OptiQ 4-bit model end to end.
5. **Issue #59** — Gemma-4-31B first inference does not hit `fatalError`; on bad shape, surfaces a typed error.
6. **`olmlx ext list`** — prints all entries with tracking URLs and removal conditions.
7. **Removal drill** — manually delete the `MinimaxM2.swift` override after upstream merges, run tests; failure mode should be a clean `unsupportedArchitecture` error if (and only if) we forget to also bump the pinned upstream version.

## 10. Out of scope (for now)

- Forking mlx-swift-lm.
- Distributed inference (no usable lower-level scaffolding in mlx-swift today).
- Flash-MoE sparse SSD loading — too coupled to `../olmlx`'s loader, deferred indefinitely.
- A code-generation pipeline that converts Python mlx-lm architectures to Swift. Interesting but speculative; revisit after we have ≥3 manual ports and understand the recurring patterns.
