# Gemma4 Architecture Override — Design

**Date:** 2026-05-20
**Status:** Draft for review
**Closes:** #57, #58, #59 (Gemma 4 e2b / 26B-A4B MoE / 31B attention crash)
**Builds on:** the extension layer foundation (`docs/superpowers/specs/2026-05-19-architecture-feature-extension-layer-design.md`, merged in #65)

## 1. Context

Three Gemma 4 checkpoints fail in olmlx-swift, all delegating to upstream `mlx-swift-lm`
(pinned at `1c05248` / v3.31.3, `Libraries/MLXLLM/Models/Gemma4Text.swift`):

| Issue | Checkpoint (on disk) | Symptom |
|---|---|---|
| #57 | `mlx-community/gemma-4-e2b-it-OptiQ-4bit` | `mismatchedSize` loading `per_layer_model_projection.weight` — expected `[8960,1536]`, got `[8960,192]` |
| #58 | `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` | `unhandledKeys`: `experts`, `router`, `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_{1,2}` |
| #59 | `mlx-community/gemma-4-31B-it-OptiQ-4bit` | `fatalError` mid-inference: `broadcast_shapes Shapes (1,21,4,512) and (1,4,21,512) cannot be broadcast` |

All three report `model_type: gemma4`. Ground truth for the correct behavior is `mlx-lm`'s
Python reference at `mlx_lm/models/gemma4_text.py` (and `gemma4.py`), which loads and runs
all three checkpoints. The Swift port diverged from it in three places.

### 1.1 Root causes (confirmed against the Python reference)

**#57 — quantization, not PLE shape.** Python (`gemma4_text.py:409`) builds
`per_layer_model_projection` as a plain `nn.Linear`, applying the `hidden_size**-0.5`
scale separately (`:481`). The Swift port (`Gemma4Text.swift:163-176, 512`) instead baked
the scale into a custom `ScaledLinear` module that holds a *raw* `MLXArray` weight. The MLX
quantizer only rewrites `Linear`/`Embedding`-shaped modules, so `ScaledLinear` stays
unquantized. The e2b OptiQ-4bit checkpoint ships a *quantized* projection (packed weight
`[8960, 192]` = `1536 * 4 / 32`), which cannot load into the unquantized `[8960, 1536]`
slot. This is the same class of defect as the stalled upstream
[mlx-swift-lm#244](https://github.com/ml-explore/mlx-swift-lm/pull/244). The e2b
`config.json` carries a per-layer `quantization` map (incl. the projection), so once the
module is a plain `Linear`, the loader quantizes it automatically.

**#58 — MoE path absent.** Python models a dual-FFN MoE block when `enable_moe_block`
(`gemma4_text.py:289-302, 344-354`): a `Router` (rms-norm → scale → `proj` →
top-k → softmax → per-expert scale), an `Experts` block over `SwitchGLU`, three extra
layernorms (`pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`,
`post_feedforward_layernorm_2`), and the combination `h = h1 + h2`. The Swift
`Gemma4DecoderLayer` is dense-only, so those weight keys are unhandled. The 26B config
confirms `enable_moe_block=true`, `num_experts=128`, `top_k_experts=8`,
`moe_intermediate_size=704`. Python also splits fused expert weights in `sanitize`
(`gate_up_proj` → `switch_glu.{gate,up}_proj`, `down_proj` → `switch_glu.down_proj`,
`:616-625`).

**#59 — `v = k` taken at the wrong point.** In the `attention_k_eq_v` path Python sets
`values = keys` from the *pre-transpose, pre-norm* reshaped tensor `[B,L,kv,d]`, then norms
and transposes `values` independently and never applies rope to it
(`gemma4_text.py:242-254`). The Swift port (`Gemma4Text.swift:302-314`) reshapes and
transposes/ropes `k` first, then does `v = k` on the *already transposed+roped* tensor and
transposes it *again* — producing `values=[1,21,4,512]` while `keys=[1,4,21,512]`. The two
meet in `scaledDotProductAttention` and abort. (As an MLX C++ abort this surfaces as a
`fatalError`, not a Swift throw — see §6.)

## 2. Goals / Non-goals

**Goals**
- Load and run all three checkpoints correctly through the existing extension layer.
- Match the `mlx-lm` reference numerically (within quantization noise).
- Keep the override disciplined: tracked, auto-flagged for removal, one file.

**Non-goals**
- General MLX-boundary hardening so shape mismatches become typed errors instead of
  `fatalError` (#59 recommendation #3). See §6 — not tractable here, filed separately.
- Touching `minimax_m2` / `step3p5` (#61) — deferred.
- Vision/audio Gemma 4 modalities — these checkpoints are text-only.

## 3. Approach

Register a corrected, in-tree Gemma 4 implementation through `OLMLXExtensions`, overriding
upstream for `model_type` `gemma4` and `gemma4_text`. This is the override path the
extension spec already describes (§3.3): re-register under the same `model_type`; on upstream
uptake, delete our file and the registry falls back to the inherited shared entry.

**Why a full-file vendored copy, not a subclass or surgical patch.** The defective types
(`ScaledLinear`, `Gemma4Attention`, `Gemma4DecoderLayer`, `Gemma4TextModelInner`) are all
`private` in upstream. They cannot be subclassed, extended, or partially replaced from our
module. The extension mechanism overrides by `model_type` → a creator that returns a
complete `LanguageModel`. So we vendor a corrected copy. This is the explicit, accepted
trade-off in the extension spec (§3.3, §8: "subtle correctness drift"), mitigated by upstream
tracking, an auto-checked removal condition, and golden-output tests.

### 3.1 Components

**New file — `Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift`** — a copy of
upstream `Gemma4Text.swift` plus the thin `Gemma4.swift` `gemma4` wrapper, with three
corrections and the MoE additions. To avoid symbol collisions with the (still-registered-by-
default) upstream types, all vendored types are namespaced (e.g. `Gemma4PatchedTextModel`,
`Gemma4PatchedAttention`, …) within this file.

Corrections:
1. **#57** — `perLayerModelProjection` is a plain `Linear(hiddenSize, numHiddenLayers *
   hiddenSizePerLayerInput, bias: false)`; the `hidden_size**-0.5` scale is applied to its
   output in `_projectPerLayerInputs`, matching `gemma4_text.py:480-481`.
2. **#59** — in the attention `useKeqV` branch, capture `vSource` = the reshaped
   `[B,L,kv,d]` keys *before* norm/transpose/rope; build `values` from `vSource` with
   `vNorm` + `transpose(0,2,1,3)` and no rope; build `keys` with `kNorm` + transpose + rope.
   Matches `gemma4_text.py:242-254`.
3. **#58** — add:
   - `Gemma4PatchedRouter`: `rmsNorm(x, scale * hidden_size**-0.5)` → `proj` (Linear,
     `hidden_size → num_experts`) → top-k via `argPartition` → `softmax` →
     `* perExpertScale[idx]`. Returns `(indices, weights)`.
   - `Gemma4PatchedExperts`: wraps `SwitchGLU(inputDims: hidden_size,
     hiddenDims: moe_intermediate_size, numExperts: num_experts, activation: gelu-gated,
     bias: false)`; returns `(weights.expandDims(-1) * switchGLU(x, indices)).sum(axis: -2)`.
   - decoder layer: when `enableMoeBlock`, run the dense MLP path
     (`pre_feedforward_layernorm` → `mlp` → `post_feedforward_layernorm_1`) as `h1` and the
     expert path (`pre_feedforward_layernorm_2` → `router`/`experts` →
     `post_feedforward_layernorm_2`) as `h2`, then `h = h1 + h2`, then the shared
     `post_feedforward_layernorm` and residual. Matches `gemma4_text.py:344-360`.
   - `sanitize`: split `*.experts.gate_up_proj` into `*.experts.switch_glu.{gate,up}_proj.weight`
     and remap `*.experts.down_proj` → `*.experts.switch_glu.down_proj.weight`
     (`gemma4_text.py:616-625`). The existing rotary/min-max skips are preserved.

Config additions to the vendored `Gemma4TextConfiguration`: `enableMoeBlock`
(`enable_moe_block`, default false), `numExperts` (`num_experts`), `topKExperts`
(`top_k_experts`), `moeIntermediateSize` (`moe_intermediate_size`).

**Manifest — `Sources/OLMLX/Extensions/OLMLXExtensions.swift`** — two `ExtensionEntry`
values (`gemma4`, `gemma4_text`) whose `creator` decodes the wrapper / text config and
builds the patched model. Each carries:
- `upstreamTracking`: the relevant upstream issue/PR (PR #244 for the quant fix; new
  upstream issues to be filed for the MoE gap and the attention `v=k` bug — opened before
  merge, per extension discipline rule 1).
- `removeWhen: .upstreamReleased(version: <first mlx-swift-lm release carrying the fixes>)`,
  so `olmlx ext check` flags removal once the pin advances past it. (Until a fixed release
  exists, use a high sentinel version and note the tracking issues; revisit when upstream
  merges.)

### 3.2 Quantization

The OptiQ checkpoints embed a per-layer `quantization` map in `config.json`, which the MLX
loader applies by module path. Because the corrected projection and the MoE `proj`/expert
linears are now standard quantizable modules, no custom `quant_predicate` is required —
the checkpoint's own map drives quantization (incl. the 8-bit router that `mlx-lm`'s
`quant_predicate` produces at conversion time). If a checkpoint is found that omits the
per-layer map for the router, that is handled as an implementation follow-up, not a design
change.

## 4. Data flow (unchanged from upstream except the three fixes)

`gemma4` wrapper → `Gemma4PatchedTextModel` → embed (×`sqrt(hidden)`) → optional PLE inputs
(`_get` + `_project`, #57 fix in the projection) → per-layer decoder loop with KV-sharing
(dense or MoE branch per layer, #58) → final norm → tied/`lm_head` logits → softcap.
Attention per layer applies the #59 fix in the `useKeqV` branch.

## 5. Error handling

- Config decode: missing MoE fields default to the dense path (`enableMoeBlock=false`), so
  non-MoE Gemma 4 checkpoints are unaffected.
- `sanitize` only remaps keys it recognizes; unknown keys pass through unchanged, preserving
  upstream's rotary/min-max filtering.
- Load failures continue to surface through the existing `ModelManager.wrapMLXLoadError`
  path as typed errors.

## 6. The `fatalError` boundary (deliberately out of scope)

#59 recommendation #3 asks that MLX shape failures become recoverable `InferenceError`s so a
bad model can't take the daemon down. MLX surfaces these as C++ `abort()`/`fatalError` from
`ops.cpp`, which Swift cannot intercept with `do/try/catch`. A general guard would require an
upstream change (validating shapes and `throw`ing before the C++ call) or a separate process
boundary for inference — both larger than this work and not Gemma-specific. The concrete
attention fix removes the actual crash for these checkpoints. Recommendation: file the
boundary hardening as its own issue and track it independently.

## 7. Critical files

- **New:** `Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift`
- **Modify:** `Sources/OLMLX/Extensions/OLMLXExtensions.swift` (add two manifest entries)
- **New tests:** `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift`
- **Reference (read-only):** `.build/checkouts/mlx-swift-lm/.../Gemma4Text.swift`,
  `Gemma4.swift`, `Qwen35MoE.swift` (SwitchGLU usage);
  `../olmlx/.venv/.../mlx_lm/models/gemma4_text.py` (ground truth)

## 8. Verification plan

1. **Unit (fast, no weights):** config decode sets MoE fields for the 26B config and leaves
   them defaulted for e2b/31B; `sanitize` produces the expected `switch_glu.*` keys from
   synthetic `gate_up_proj`/`down_proj` arrays; the PLE projection module is a `Linear`.
2. **Registration:** `gemma4` / `gemma4_text` resolve to the patched model through
   `OLMLXExtensions.registerAll()`.
3. **End-to-end (gated on checkpoint presence on disk):** load + generate ≥1 token for each
   of e2b, 26B-A4B, 31B without error and without `fatalError`. 26B/31B 4-bit are large;
   these tests are slow and opt-in (skipped when the checkpoint is absent).
4. **Golden output (optional, recommended):** a fixed short prompt → first-token logits or
   decoded text compared to the `mlx-lm` Python output for the same checkpoint, to catch
   numerical drift; re-checked when the override is removed.
5. **`olmlx ext list` / `ext check`:** the two new entries appear with tracking URLs and
   removal conditions.

## 9. Risks & trade-offs

- **Drift from upstream.** A vendored copy can diverge from the eventual upstream fix.
  Mitigation: tracking URLs + auto-checked `removeWhen` + golden tests; delete the file when
  upstream ships.
- **`SwitchGLU` API specifics.** Exact initializer/activation signature must match the
  pinned MLXLLM; resolved during implementation against `Qwen3MoE`/`Qwen35MoE` usage.
- **Verification cost.** Loading 26B/31B 4-bit models is memory- and time-heavy; e2b is the
  cheap smoke test, the larger two are opt-in.

## 10. Out of scope

- #61 (`minimax_m2`, `step3p5`).
- General MLX `fatalError` boundary hardening (§6) — separate issue.
- Gemma 4 vision/audio modalities.
