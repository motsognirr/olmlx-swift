# Gemma4 Architecture Override Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a corrected, in-tree Gemma 4 implementation through the existing extension layer so the e2b (#57), 26B-A4B MoE (#58), and 31B (#59) checkpoints load and run correctly.

**Architecture:** Vendor a complete corrected copy of upstream `Gemma4Text.swift` + the `Gemma4.swift` wrapper into one new file under `Sources/OLMLX/Extensions/Architectures/`, with all types namespaced `Gemma4Patched*` to avoid colliding with the still-default-registered upstream types. Apply three fixes (quantizable per-layer projection, attention `v=k` ordering, MoE dual-FFN path) grounded in the `mlx-lm` Python reference, then register `gemma4`/`gemma4_text` onto `LLMTypeRegistry.shared` via `OLMLXExtensions`, overriding upstream.

**Tech Stack:** Swift, swift-testing (`import Testing`, `@Suite`/`@Test`), MLX / MLXNN / MLXLLM / MLXLMCommon, swift-argument-parser.

**Reference files (read-only ground truth):**
- Upstream Swift to copy: `.build/checkouts/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift`, `Gemma4.swift`
- Python source of truth: `../olmlx/.venv/lib/python3.11/site-packages/mlx_lm/models/gemma4_text.py`
- MoE primitives: `.build/checkouts/mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift` (`SwitchGLU`, `SwitchLinear`), `Models/Qwen35MoE.swift` (sanitize split pattern)
- Foundation to extend: `Sources/OLMLX/Extensions/OLMLXExtensions.swift`, `Sources/OLMLX/Extensions/ExtensionEntry.swift`

**On-disk test checkpoints (HF cache):**
- `mlx-community/gemma-4-e2b-it-OptiQ-4bit` (PLE, dense, cheap smoke test)
- `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` (MoE, large)
- `mlx-community/gemma-4-31B-it-OptiQ-4bit` (k_eq_v dense, large)

**Note on MLX in tests:** unit tests that allocate/run MLX arrays require Metal and run on this Apple-Silicon dev machine. The large-model end-to-end tests are opt-in (skipped when the checkpoint directory is absent).

---

## File Structure

- **Create** `Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift` — the entire vendored, corrected Gemma 4 port: configuration, helper modules, attention (#59 fix), MLP, Router + Experts (#58), decoder layer (dense + MoE branches), text model (#57 fix), the `gemma4` wrapper model, `sanitize`, and `newCache`. One file, one responsibility: "our temporary Gemma 4 architecture."
- **Modify** `Sources/OLMLX/Extensions/OLMLXExtensions.swift` — add two `ExtensionEntry` values (`gemma4`, `gemma4_text`) to `manifest`.
- **Create** `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift` — config decode, `sanitize` key-mapping, projection-is-Linear, attention-shape, registration, and gated end-to-end tests.

---

## Task 1: Vendor the dense Gemma 4 port (behavior-identical copy, namespaced)

Goal: get a compiling, namespaced copy in place with **no behavior changes yet**. Fixes land in Tasks 3–5.

**Files:**
- Create: `Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift`

- [ ] **Step 1: Copy upstream into the new file**

Copy the full contents of `.build/checkouts/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift` into a new file `Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift`. Then append the contents of `Gemma4.swift` *from its `// MARK: - Configuration` line onward* (skip the duplicate import block) into the same file.

Keep the import block at the top:
```swift
import Foundation
import MLX
import MLXLMCommon
import MLXNN
```

- [ ] **Step 2: Rename every Gemma4 type to a `Gemma4Patched*` namespace**

Apply these exact whole-word renames throughout the file (both declarations and uses) so nothing collides with upstream's still-registered types:

| Upstream name | Renamed to |
|---|---|
| `Gemma4TextConfiguration` | `Gemma4PatchedTextConfiguration` |
| `Gemma4Configuration` | `Gemma4PatchedConfiguration` |
| `Gemma4TextModel` | `Gemma4PatchedTextModel` |
| `Gemma4TextModelInner` | `Gemma4PatchedTextModelInner` |
| `Gemma4Model` | `Gemma4PatchedModel` |
| `Gemma4Attention` | `Gemma4PatchedAttention` |
| `Gemma4MLP` | `Gemma4PatchedMLP` |
| `Gemma4DecoderLayer` | `Gemma4PatchedDecoderLayer` |
| `RMSNormNoScale` | `Gemma4PatchedRMSNormNoScale` |
| `ScaledLinear` | `Gemma4PatchedScaledLinear` |
| `Gemma4PositionOffset` | `Gemma4PatchedPositionOffset` |
| `gemma4CapturePositionOffset` | `gemma4PatchedCapturePositionOffset` |
| `gemma4ApplyRotaryPosition` | `gemma4PatchedApplyRotaryPosition` |

Do **not** change any `@ModuleInfo(key:)` string, any `CodingKeys` raw value, or the `modelType` default string values — those map to on-disk weights/config and must stay byte-identical.

- [ ] **Step 3: Verify it builds**

Run: `swift build`
Expected: builds with no errors (a behavior-identical, renamed copy).

- [ ] **Step 4: Commit**

```bash
git add Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift
git commit -m "feat: vendor Gemma4 port into extensions (namespaced, no behavior change)"
```

---

## Task 2: Add MoE configuration fields

**Files:**
- Modify: `Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift`
- Test: `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift`

- [ ] **Step 1: Write the failing test**

Create `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift`:

```swift
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import OLMLX

@Suite("Extensions/Gemma4Patched/Config")
struct Gemma4PatchedConfigTests {

    private func decodeText(_ json: String) throws -> Gemma4PatchedTextConfiguration {
        try JSONDecoder().decode(
            Gemma4PatchedTextConfiguration.self, from: Data(json.utf8))
    }

    @Test func moeFieldsDecodeWhenPresent() throws {
        let cfg = try decodeText(#"""
        {"model_type":"gemma4_text","enable_moe_block":true,
         "num_experts":128,"top_k_experts":8,"moe_intermediate_size":704}
        """#)
        #expect(cfg.enableMoeBlock == true)
        #expect(cfg.numExperts == 128)
        #expect(cfg.topKExperts == 8)
        #expect(cfg.moeIntermediateSize == 704)
    }

    @Test func moeFieldsDefaultToDenseWhenAbsent() throws {
        let cfg = try decodeText(#"{"model_type":"gemma4_text"}"#)
        #expect(cfg.enableMoeBlock == false)
        #expect(cfg.numExperts == nil)
        #expect(cfg.topKExperts == nil)
        #expect(cfg.moeIntermediateSize == nil)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter Gemma4PatchedConfigTests`
Expected: FAIL — `value of type 'Gemma4PatchedTextConfiguration' has no member 'enableMoeBlock'`.

- [ ] **Step 3: Add the fields and decoding**

In `Gemma4PatchedTextConfiguration`, add the stored properties (next to the other `var`s):

```swift
    var enableMoeBlock: Bool = false
    var numExperts: Int? = nil
    var topKExperts: Int? = nil
    var moeIntermediateSize: Int? = nil
```

Add to `CodingKeys`:

```swift
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
```

Add to `init(from:)` (after the existing decode statements, before the RoPE extraction block):

```swift
        self.enableMoeBlock =
            try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        self.topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter Gemma4PatchedConfigTests`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift
git commit -m "feat: add Gemma4 MoE config fields"
```

---

## Task 3: Fix #57 — quantizable per-layer model projection

The custom `Gemma4PatchedScaledLinear` holds a raw `MLXArray` weight the quantizer skips. Replace it with a plain `Linear` (quantizable) and apply the scale separately, matching `gemma4_text.py:409,480-481`.

**Files:**
- Modify: `Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift`
- Test: `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift` (append)

- [ ] **Step 1: Write the failing test**

Append to `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift`:

```swift
import MLXNN

@Suite("Extensions/Gemma4Patched/Projection")
struct Gemma4PatchedProjectionTests {

    @Test func perLayerModelProjectionIsAQuantizableLinear() throws {
        var cfg = try JSONDecoder().decode(
            Gemma4PatchedTextConfiguration.self,
            from: Data(#"{"model_type":"gemma4_text"}"#.utf8))
        // e2b-style: PLE enabled, small model to keep allocation cheap.
        cfg.hiddenSize = 64
        cfg.numHiddenLayers = 2
        cfg.hiddenSizePerLayerInput = 8
        cfg.vocabSize = 32
        cfg.vocabSizePerLayerInput = 32
        cfg.numKvSharedLayers = 0
        cfg.layerTypes = ["sliding_attention", "full_attention"]

        let model = Gemma4PatchedTextModelInner(cfg)
        let proj = try #require(model.perLayerModelProjection)
        // A plain Linear exposes `weight` shaped [out, in] = [layers*ple, hidden].
        #expect(proj.weight.dim(0) == cfg.numHiddenLayers * cfg.hiddenSizePerLayerInput)
        #expect(proj.weight.dim(1) == cfg.hiddenSize)
    }
}
```

(`perLayerModelProjection` must be accessible to `@testable import`. If upstream marked `Gemma4PatchedTextModelInner` `private`, change it to `class Gemma4PatchedTextModelInner` — internal — and likewise drop `private`/`fileprivate` on the `model` property of `Gemma4PatchedTextModel` if the test needs it. Internal visibility is fine; these types are only used within OLMLX.)

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter Gemma4PatchedProjectionTests`
Expected: FAIL — `proj` is a `Gemma4PatchedScaledLinear` which has no `weight: MLXArray` accessor of that shape / type mismatch, or `perLayerModelProjection` is not a `Linear`.

- [ ] **Step 3: Replace ScaledLinear usage with Linear + separate scale**

In `Gemma4PatchedTextModelInner`:

Change the property declaration:
```swift
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
```

Add a stored scale near the other `let`s in the class:
```swift
    let perLayerProjectionScale: Float
```

In `init`, set the scale (always, even when PLE is off it is harmless):
```swift
        self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)
```

Inside the `if config.hiddenSizePerLayerInput > 0 {` block, replace the `ScaledLinear` construction with a plain `Linear`:
```swift
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.numHiddenLayers * config.hiddenSizePerLayerInput,
                bias: false)
```

In `callAsFunction`, where the projection is applied, multiply by the scale explicitly. Replace:
```swift
            let modelPLE = modelProj(h).reshaped(...)
```
with:
```swift
            let modelPLE = (modelProj(h) * perLayerProjectionScale).reshaped(
                h.dim(0), h.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)
```
(Keep the surrounding `reshaped` arguments exactly as they were.)

Finally, delete the now-unused `Gemma4PatchedScaledLinear` class definition.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter Gemma4PatchedProjectionTests`
Expected: PASS.

- [ ] **Step 5: Verify full build**

Run: `swift build`
Expected: builds (no remaining references to `Gemma4PatchedScaledLinear`).

- [ ] **Step 6: Commit**

```bash
git add Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift
git commit -m "fix: make Gemma4 per-layer projection a quantizable Linear (#57)"
```

---

## Task 4: Fix #59 — attention `v = k` taken before transpose/rope

In the `useKeqV` branch, `values` must be built from the reshaped `[B,L,kv,d]` keys *before* norm/transpose/rope, then independently normed and transposed (no rope), matching `gemma4_text.py:242-254`.

**Files:**
- Modify: `Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift`
- Test: `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift` (append)

- [ ] **Step 1: Write the failing test**

Append to `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift`:

```swift
@Suite("Extensions/Gemma4Patched/Attention")
struct Gemma4PatchedAttentionTests {

    /// Reproduces the #59 broadcast crash: a full-attention layer with
    /// attention_k_eq_v and global KV heads. Pre-fix this aborts in SDPA.
    @Test func kEqVFullAttentionForwardProducesCorrectShape() throws {
        var cfg = try JSONDecoder().decode(
            Gemma4PatchedTextConfiguration.self,
            from: Data(#"{"model_type":"gemma4_text"}"#.utf8))
        cfg.hiddenSize = 64
        cfg.numAttentionHeads = 8
        cfg.numKeyValueHeads = 2
        cfg.numGlobalKeyValueHeads = 2
        cfg.headDim = 16
        cfg.globalHeadDim = 32
        cfg.attentionKeqV = true
        cfg.layerTypes = ["full_attention"]

        let attn = Gemma4PatchedAttention(cfg, layerIdx: 0)
        let x = MLXArray.ones([1, 5, cfg.hiddenSize]).asType(.float32)
        let (out, _, _) = attn(x, mask: .none, cache: nil)
        out.eval()
        #expect(out.dim(0) == 1)
        #expect(out.dim(1) == 5)
        #expect(out.dim(2) == cfg.hiddenSize)
    }
}
```

(`Gemma4PatchedAttention` must be internal, not `private`, for the test. Drop `private` on that class.)

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter Gemma4PatchedAttentionTests`
Expected: FAIL — broadcast/shape abort (the `(1,5,2,32)` vs `(1,2,5,32)` mismatch) or process abort during `out.eval()`.

- [ ] **Step 3: Fix the `v = k` ordering**

In `Gemma4PatchedAttention.callAsFunction`, locate the non-shared branch (the `else` after `if let (sharedK, sharedV) = sharedKV`). Replace its body so `values` is derived from the pre-transpose reshaped keys:

```swift
        } else {
            let kReshaped = kProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)

            var k = kNorm(kReshaped)
            k = k.transposed(0, 2, 1, 3)
            k = gemma4PatchedApplyRotaryPosition(rope, to: k, offset: activePositionOffset)

            let vReshaped: MLXArray
            if let vProj {
                vReshaped = vProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
            } else {
                vReshaped = kReshaped
            }
            var v = vNorm(vReshaped)
            v = v.transposed(0, 2, 1, 3)

            if let cache {
                let (updatedK, updatedV) = cache.update(keys: k, values: v)
                keys = updatedK
                values = updatedV
            } else {
                keys = k
                values = v
            }
        }
```

The key change vs upstream: `vReshaped = kReshaped` (the `[B,L,kv,d]` tensor) instead of `v = k` (the already transposed+roped tensor). `values` never gets rope.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter Gemma4PatchedAttentionTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift
git commit -m "fix: correct Gemma4 k_eq_v value tensor layout (#59)"
```

---

## Task 5: Add #58 — MoE Router, Experts, dual-FFN decoder branch, and sanitize split

**Files:**
- Modify: `Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift`
- Test: `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift` (append)

- [ ] **Step 1: Add Router and Experts modules**

In `Gemma4Patched.swift`, after the `Gemma4PatchedMLP` class, add:

```swift
// MARK: - MoE (issue #58)

private class Gemma4PatchedRouter: Module {
    let eps: Float
    let topK: Int
    let rootSize: Float

    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "scale") var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(_ config: Gemma4PatchedTextConfiguration) {
        let numExperts = config.numExperts ?? 0
        self.eps = config.rmsNormEps
        self.topK = config.topKExperts ?? 1
        self.rootSize = pow(Float(config.hiddenSize), -0.5)
        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let normed = MLXFast.rmsNorm(x, weight: scale * rootSize, eps: eps)
        let scores = proj(normed)

        var topKIndices = argPartition(scores, kth: -topK, axis: -1)
        topKIndices = topKIndices[.ellipsis, (-topK)...]

        var topKWeights = takeAlong(scores, topKIndices, axis: -1)
        topKWeights = softmax(topKWeights, axis: -1)
        topKWeights = topKWeights * perExpertScale[topKIndices]

        return (topKIndices, topKWeights)
    }
}

private class Gemma4PatchedExperts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(_ config: Gemma4PatchedTextConfiguration) {
        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize ?? config.intermediateSize,
            numExperts: config.numExperts ?? 0,
            activation: geluApproximate,
            bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray
    ) -> MLXArray {
        let w = topKWeights.expandedDimensions(axis: -1)
        let y = switchGLU(x, topKIndices)
        return (w * y).sum(axis: -2)
    }
}
```

(`argPartition`, `takeAlong`, `softmax`, `geluApproximate`, `MLXFast.rmsNorm` are MLX/MLXNN free functions already used elsewhere in the package. If `argPartition`/`takeAlong` resolve under different names in the pinned MLX, check `Qwen3MoE.swift` / `MiniMax.swift` for the exact spelling and match it.)

- [ ] **Step 2: Add MoE members and the dual-FFN branch to the decoder layer**

In `Gemma4PatchedDecoderLayer`, add the MoE members (after the existing layernorm `@ModuleInfo`s):

```swift
    let enableMoe: Bool
    @ModuleInfo(key: "router") var router: Gemma4PatchedRouter?
    @ModuleInfo(key: "experts") var experts: Gemma4PatchedExperts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: RMSNorm?
```

In the layer's `init`, after the existing four layernorms are set up, add:

```swift
        self.enableMoe = config.enableMoeBlock
        if config.enableMoeBlock {
            self._router.wrappedValue = Gemma4PatchedRouter(config)
            self._experts.wrappedValue = Gemma4PatchedExperts(config)
            self._postFeedforwardLayernorm1.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
```

(Add `self.enableMoe = config.enableMoeBlock` before `super.init()`. Because Swift requires all stored `let`s set before `super.init()`, place the `enableMoe` assignment unconditionally and guard only the optional module assignments.)

In `callAsFunction`, replace the feed-forward section (the block currently doing `preFeedforwardLayernorm` → `mlp` → `postFeedforwardLayernorm` → `residual2 + out`) with the dual-path version matching `gemma4_text.py:342-360`:

```swift
        let residual2 = out
        if enableMoe, let router, let experts,
            let pffn1 = postFeedforwardLayernorm1,
            let pffn2 = postFeedforwardLayernorm2,
            let preffn2 = preFeedforwardLayernorm2
        {
            var h1 = preFeedforwardLayernorm(out)
            h1 = mlp(h1)
            h1 = pffn1(h1)

            let (topKIndices, topKWeights) = router(out)
            var h2 = preffn2(out)
            h2 = experts(h2, topKIndices: topKIndices, topKWeights: topKWeights)
            h2 = pffn2(h2)

            out = h1 + h2
        } else {
            out = preFeedforwardLayernorm(out)
            out = mlp(out)
        }
        out = postFeedforwardLayernorm(out)
        out = residual2 + out
```

(Keep the PLE-gating block and the trailing `out = out * layerScalar` exactly as they are, after this section.)

- [ ] **Step 3: Add the expert-weight split to `sanitize`**

In `Gemma4PatchedTextModel.sanitize`, before the final `return sanitized`, add the split for fused expert weights (mirroring `gemma4_text.py:616-625` and `Qwen35MoE.swift`):

```swift
        var split = [String: MLXArray]()
        for (k, v) in sanitized {
            if k.hasSuffix(".experts.gate_up_proj") {
                let base = String(k.dropLast(".gate_up_proj".count))
                let mid = v.dim(-2) / 2
                split["\(base).switch_glu.gate_proj.weight"] = v[.ellipsis, ..<mid, 0...]
                split["\(base).switch_glu.up_proj.weight"] = v[.ellipsis, mid..., 0...]
            } else if k.hasSuffix(".experts.down_proj") {
                let base = String(k.dropLast(".down_proj".count))
                split["\(base).switch_glu.down_proj.weight"] = v
            } else {
                split[k] = v
            }
        }
        sanitized = split
```

(Confirm the wrapper-level `Gemma4PatchedModel.sanitize` forwards through to this method — upstream's wrapper calls `languageModel.sanitize(weights:)`, so the split runs for `language_model.*`-prefixed keys.)

- [ ] **Step 4: Write the failing test**

Append to `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift`:

```swift
@Suite("Extensions/Gemma4Patched/Sanitize")
struct Gemma4PatchedSanitizeTests {

    private func moeConfig() throws -> Gemma4PatchedConfiguration {
        try JSONDecoder().decode(
            Gemma4PatchedConfiguration.self,
            from: Data(#"""
            {"model_type":"gemma4","text_config":{"model_type":"gemma4_text",
             "hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,
             "num_key_value_heads":1,"hidden_size_per_layer_input":0,
             "enable_moe_block":true,"num_experts":4,"top_k_experts":2,
             "moe_intermediate_size":6,"layer_types":["full_attention"]}}
            """#.utf8))
    }

    @Test func splitsFusedExpertWeights() throws {
        let model = Gemma4PatchedModel(try moeConfig())
        let base = "language_model.model.layers.0.experts"
        // gate_up_proj: [experts, 2*hidden_dims, input]
        let gateUp = MLXArray.ones([4, 12, 8])
        let down = MLXArray.ones([4, 8, 6])
        let out = model.sanitize(weights: [
            "\(base).gate_up_proj": gateUp,
            "\(base).down_proj": down,
        ])
        #expect(out["\(base).gate_up_proj"] == nil)
        #expect(out["\(base).down_proj"] == nil)
        #expect(out["\(base).switch_glu.gate_proj.weight"]?.dim(-2) == 6)
        #expect(out["\(base).switch_glu.up_proj.weight"]?.dim(-2) == 6)
        #expect(out["\(base).switch_glu.down_proj.weight"]?.dim(-1) == 6)
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `swift test --filter Gemma4PatchedSanitizeTests`
Expected: PASS. (If it fails to build first, fix references per the notes in Steps 1–3, then re-run.)

- [ ] **Step 6: Verify full build and existing tests**

Run: `swift build && swift test --filter Gemma4Patched`
Expected: all Gemma4Patched suites PASS.

- [ ] **Step 7: Commit**

```bash
git add Sources/OLMLX/Extensions/Architectures/Gemma4Patched.swift Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift
git commit -m "feat: add Gemma4 MoE router/experts and dual-FFN path (#58)"
```

---

## Task 6: Register `gemma4` / `gemma4_text` overrides in the manifest

**Files:**
- Modify: `Sources/OLMLX/Extensions/OLMLXExtensions.swift`
- Test: `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift` (append)

- [ ] **Step 1: Write the failing test**

Append to `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift`:

```swift
@Suite("Extensions/Gemma4Patched/Registration")
struct Gemma4PatchedRegistrationTests {

    @Test func manifestRegistersGemma4Overrides() {
        let types = Set(OLMLXExtensions.manifest.map { $0.modelType })
        #expect(types.contains("gemma4"))
        #expect(types.contains("gemma4_text"))
    }

    @Test func gemma4EntryBuildsAModelFromConfig() throws {
        let entry = try #require(
            OLMLXExtensions.manifest.first { $0.modelType == "gemma4" })
        let cfg = #"""
        {"model_type":"gemma4","text_config":{"model_type":"gemma4_text",
         "hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,
         "num_key_value_heads":1,"hidden_size_per_layer_input":0,
         "layer_types":["full_attention"]}}
        """#
        let model = try entry.creator(Data(cfg.utf8))
        #expect(model is Gemma4PatchedModel)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter Gemma4PatchedRegistrationTests`
Expected: FAIL — manifest contains only `olmlx_canary`.

- [ ] **Step 3: Add the manifest entries**

In `Sources/OLMLX/Extensions/OLMLXExtensions.swift`, add two entries to the `manifest` array (after the `olmlx_canary` entry). Use the existing `creator(_:_:)` helper:

```swift
        ExtensionEntry(
            modelType: "gemma4",
            kind: .architecture,
            upstreamTracking: URL(
                string: "https://github.com/ml-explore/mlx-swift-lm/pull/244")!,
            addedOn: "2026-05-20",
            removeWhen: .upstreamReleased(version: "99.0.0"),
            notes: "Gemma4 override: quantizable per-layer projection (#57), "
                + "k_eq_v value layout (#59), MoE dual-FFN path (#58). "
                + "Bump removeWhen to the first mlx-swift-lm release carrying all three.",
            creator: creator(Gemma4PatchedConfiguration.self, Gemma4PatchedModel.init)
        ),
        ExtensionEntry(
            modelType: "gemma4_text",
            kind: .architecture,
            upstreamTracking: URL(
                string: "https://github.com/ml-explore/mlx-swift-lm/pull/244")!,
            addedOn: "2026-05-20",
            removeWhen: .upstreamReleased(version: "99.0.0"),
            notes: "Text-only Gemma4 override; see the gemma4 entry.",
            creator: creator(
                Gemma4PatchedTextConfiguration.self, Gemma4PatchedTextModel.init)
        ),
```

(`.upstreamReleased(version: "99.0.0")` is a deliberate high sentinel: no fixed upstream release exists yet. When the upstream PRs land in a tagged release, change it to that version and `olmlx ext check` will flag the entries for removal. The MoE and attention bugs need their own upstream issues filed before merge, per extension discipline; reference them in the notes once opened.)

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter Gemma4PatchedRegistrationTests`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify `olmlx ext list` shows the entries**

Run: `swift run olmlx ext list`
Expected: output lists `[architecture] gemma4` and `[architecture] gemma4_text` with the tracking URL and `remove when upstream >= 99.0.0`.

- [ ] **Step 6: Commit**

```bash
git add Sources/OLMLX/Extensions/OLMLXExtensions.swift Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift
git commit -m "feat: register gemma4/gemma4_text overrides (#57/#58/#59)"
```

---

## Task 7: End-to-end verification against on-disk checkpoints

These tests are opt-in: they locate the checkpoint in the HF cache and skip if absent. They run on this Apple-Silicon machine (Metal + enough RAM). e2b is the cheap gate; 26B/31B are heavy.

**Files:**
- Test: `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift` (append)

- [ ] **Step 1: Write the gated end-to-end test**

Append to `Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift`:

```swift
@Suite("Extensions/Gemma4Patched/EndToEnd")
struct Gemma4PatchedEndToEndTests {

    /// Resolves a checkpoint snapshot dir in the HF cache, or nil if absent.
    private func snapshotDir(_ repo: String) -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let base = home
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots")
        guard let kids = try? FileManager.default.contentsOfDirectory(
            at: base, includingPropertiesForKeys: nil), let first = kids.first
        else { return nil }
        return first
    }

    private func generatesOneToken(_ repo: String) async throws {
        guard let dir = snapshotDir(repo) else {
            // Checkpoint not present on this machine — skip.
            return
        }
        let engine = DefaultInferenceEngine()
        let container = try await engine.loadModel(from: dir)
        let (text, _) = try await runGeneration(
            container: container,
            messages: [["role": "user", "content": "Hi"]],
            tools: nil,
            parameters: { var p = GenerateParameters(); p.maxTokens = 1; return p }()
        )
        #expect(text.isEmpty == false)
    }

    @Test func e2bLoadsAndGenerates() async throws {
        try await generatesOneToken("mlx-community/gemma-4-e2b-it-OptiQ-4bit")
    }

    @Test func gemma26BMoELoadsAndGenerates() async throws {
        try await generatesOneToken("mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit")
    }

    @Test func gemma31BLoadsAndGenerates() async throws {
        try await generatesOneToken("mlx-community/gemma-4-31B-it-OptiQ-4bit")
    }
}
```

(Confirm the exact signature of `runGeneration` / `GenerateParameters` against `Sources/OLMLX/Engine/InferenceEngine.swift`; adjust the call if the project exposes a more convenient generation entry point.)

- [ ] **Step 2: Run the cheap e2b case**

Run: `swift test --filter "e2bLoadsAndGenerates"`
Expected: PASS — model loads (no `mismatchedSize`) and emits a token. If the checkpoint is absent the test is a no-op pass.

- [ ] **Step 3: Run the heavy cases**

Run: `swift test --filter gemma26BMoELoadsAndGenerates` then `swift test --filter gemma31BLoadsAndGenerates`
Expected: each loads (no `unhandledKeys` for 26B; no `fatalError`/broadcast abort for 31B) and emits a token. These are slow and memory-heavy.

- [ ] **Step 4: Manual CLI sanity check**

Run: `swift run olmlx bench bench-run --model mlx-community/gemma-4-e2b-it-OptiQ-4bit:latest`
Expected: completes a benchmark run (this was the original repro command in #57).

- [ ] **Step 5: Commit**

```bash
git add Tests/OLMLXTests/Extensions/Gemma4PatchedTests.swift
git commit -m "test: end-to-end Gemma4 checkpoint load/generate (#57/#58/#59)"
```

---

## Task 8: Documentation and issue cross-references

**Files:**
- Modify: `docs/architecture.md` (the extension-mechanism subsection added by the foundation)

- [ ] **Step 1: Document the override**

In `docs/architecture.md`, in the extension-layer section, add a short note: the `gemma4`/`gemma4_text` entries are temporary overrides fixing #57/#58/#59; ground truth is `mlx-lm`'s `gemma4_text.py`; remove when a fixed `mlx-swift-lm` release is pinned and bump `removeWhen` accordingly.

- [ ] **Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: note the temporary Gemma4 architecture override"
```

- [ ] **Step 3: Final full test run**

Run: `swift build && swift test`
Expected: full suite PASS (heavy e2e tests skip if checkpoints absent on the runner).

---

## Self-Review notes (coverage map)

- Spec §1.1 #57 (quant) → Task 3. #58 (MoE) → Task 5. #59 (`v=k`) → Task 4.
- Spec §3.1 vendored namespaced copy → Task 1; config fields → Task 2; manifest entries → Task 6.
- Spec §3.2 quantization driven by checkpoint config → no code (verified at load in Task 7).
- Spec §6 boundary hardening → intentionally out of scope; not a task.
- Spec §8 verification: unit (Tasks 2–5), registration (Task 6), end-to-end gated (Task 7), `ext list` (Task 6 Step 5).
- Open implementation risks flagged inline: exact MLX free-function spellings (`argPartition`/`takeAlong`) — verify against `Qwen3MoE.swift`/`MiniMax.swift`; `runGeneration` signature — verify against `InferenceEngine.swift`; type visibility (`private` → internal) needed for `@testable` access.
