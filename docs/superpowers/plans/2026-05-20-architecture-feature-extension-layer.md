# Architecture & Feature Extension Layer — Phase 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the in-tree mechanism that lets olmlx-swift register *temporary* model architectures onto mlx-swift-lm at startup, with a tracked manifest and CLI tooling so each temporary entry has a disciplined removal path.

**Architecture:** mlx-swift-lm exposes `LLMTypeRegistry.shared` — a public, mutable `actor ModelTypeRegistry<LanguageModel>` with `registerModelType(_:creator:)`. We register our architectures onto that shared registry once at startup (idempotently, from `DefaultInferenceEngine.loadModel`). This keeps the existing `loadModelContainer(from:using:)` load path — including its VLM-factory fallback — completely intact. A typed `ExtensionEntry` manifest records each temporary registration plus its upstream-tracking metadata and removal condition; `olmlx ext list` / `olmlx ext check` surface and gate that manifest.

> **Note — this refines the spec.** The spec (`docs/superpowers/specs/2026-05-19-architecture-feature-extension-layer-design.md`) proposed wrapping `LLMModelFactory` with our own registry instance. During API verification we found that (a) `ModelTypeRegistry`'s `creators` dict is `private`, so we cannot seed a new registry from `shared`, and (b) wrapping only `LLMModelFactory` would drop the VLM factory from the load path. Registering directly onto `LLMTypeRegistry.shared` is simpler and preserves VLM. Mechanism only — scope and discipline are unchanged.

**Tech Stack:** Swift, swift-testing (`import Testing`, `@Suite`/`@Test`), MLX / MLXNN / MLXLLM / MLXLMCommon, swift-argument-parser (CLI).

**Scope of THIS plan:** the foundation + tooling only. It ships exactly one harmless **canary** architecture entry (`olmlx_canary` → Llama) so the wiring is provably exercised in CI. Real architecture ports for issues #57/#58/#59/#61 are each substantial, research-heavy ports and are deliberately **out of scope** here — they become follow-up plans that simply add `ExtensionEntry` values to the manifest once this foundation is merged.

---

## File Structure

- `Sources/OLMLX/Extensions/ExtensionEntry.swift` — value types: `ExtensionKind`, `RemovalCondition`, `ExtensionEntry`, plus the pure `RemovalCondition.isRemovable(pinnedVersion:)` and a small semantic-version comparator.
- `Sources/OLMLX/Extensions/OLMLXExtensions.swift` — the `OLMLXExtensions` namespace: `creator(_:_:)` helper, the `manifest`, `register(_:onto:)`, idempotent `registerAll()`.
- `Sources/OLMLX/Engine/InferenceEngine.swift` — modify `DefaultInferenceEngine.loadModel` (lines 72-77) to `await OLMLXExtensions.registerAll()` before loading.
- `Sources/olmlx-cli/ExtensionsCommand.swift` — `Ext` parent command with `ExtList` and `ExtCheck` subcommands; pure formatting/gating helpers live here too.
- `Sources/olmlx-cli/OLMXCLI.swift` — register `Ext.self` as a subcommand (line 13-22).
- `Tests/OLMLXTests/ExtensionsTests.swift` — all unit tests for the above.
- `docs/architecture.md` — short subsection documenting the extension mechanism and how to add/remove an entry.

---

## Task 1: Extension value types + pure removal logic

**Files:**
- Create: `Sources/OLMLX/Extensions/ExtensionEntry.swift`
- Test: `Tests/OLMLXTests/ExtensionsTests.swift`

- [ ] **Step 1: Write the failing test**

Create `Tests/OLMLXTests/ExtensionsTests.swift`:

```swift
import Foundation
import MLX
import MLXNN
import MLXLMCommon
import Testing

@testable import OLMLX

@Suite("Extensions/RemovalCondition")
struct RemovalConditionTests {

    @Test func upstreamReleasedIsRemovableWhenPinnedAtOrAboveTarget() {
        let cond = RemovalCondition.upstreamReleased(version: "3.32.0")
        #expect(cond.isRemovable(pinnedVersion: "3.32.0") == true)
        #expect(cond.isRemovable(pinnedVersion: "3.32.1") == true)
        #expect(cond.isRemovable(pinnedVersion: "3.33.0") == true)
        #expect(cond.isRemovable(pinnedVersion: "3.31.3") == false)
        #expect(cond.isRemovable(pinnedVersion: "3.4.0") == false)
    }

    @Test func upstreamMergedIsNeverAutoRemovable() {
        let cond = RemovalCondition.upstreamMerged(
            pr: URL(string: "https://github.com/ml-explore/mlx-swift-lm/pull/244")!)
        #expect(cond.isRemovable(pinnedVersion: "9.9.9") == false)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter RemovalConditionTests`
Expected: FAIL — `cannot find 'RemovalCondition' in scope`.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/OLMLX/Extensions/ExtensionEntry.swift`:

```swift
import Foundation
import MLXLMCommon

/// What kind of upstream gap a temporary extension fills.
public enum ExtensionKind: String, Sendable {
    case architecture
    case feature
}

/// Condition under which a temporary extension can be deleted.
public enum RemovalCondition: Sendable {
    /// Removable once the pinned mlx-swift-lm version reaches `version`.
    case upstreamReleased(version: String)
    /// Tracked by an upstream PR; removal requires a manual check (never auto-gated).
    case upstreamMerged(pr: URL)

    /// True only for `.upstreamReleased` when `pinnedVersion` >= the target version.
    public func isRemovable(pinnedVersion: String) -> Bool {
        switch self {
        case .upstreamMerged:
            return false
        case .upstreamReleased(let target):
            return compareVersions(pinnedVersion, target) >= 0
        }
    }
}

/// Compares dotted numeric versions. Returns -1, 0, or 1.
/// Non-numeric / missing components are treated as 0.
func compareVersions(_ lhs: String, _ rhs: String) -> Int {
    let l = lhs.split(separator: ".").map { Int($0) ?? 0 }
    let r = rhs.split(separator: ".").map { Int($0) ?? 0 }
    for i in 0..<max(l.count, r.count) {
        let a = i < l.count ? l[i] : 0
        let b = i < r.count ? r[i] : 0
        if a != b { return a < b ? -1 : 1 }
    }
    return 0
}

/// A single temporary extension registered onto mlx-swift-lm.
public struct ExtensionEntry: Sendable {
    /// The `model_type` string from `config.json` this entry handles.
    public let modelType: String
    public let kind: ExtensionKind
    /// Upstream issue/PR this is tracking. Required — no entry without a tracking link.
    public let upstreamTracking: URL
    public let addedOn: String  // ISO date, e.g. "2026-05-20"
    public let removeWhen: RemovalCondition
    public let notes: String
    /// Builds a `LanguageModel` from raw `config.json` data.
    public let creator: @Sendable (Data) throws -> LanguageModel

    public init(
        modelType: String,
        kind: ExtensionKind,
        upstreamTracking: URL,
        addedOn: String,
        removeWhen: RemovalCondition,
        notes: String,
        creator: @escaping @Sendable (Data) throws -> LanguageModel
    ) {
        self.modelType = modelType
        self.kind = kind
        self.upstreamTracking = upstreamTracking
        self.addedOn = addedOn
        self.removeWhen = removeWhen
        self.notes = notes
        self.creator = creator
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter RemovalConditionTests`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add Sources/OLMLX/Extensions/ExtensionEntry.swift Tests/OLMLXTests/ExtensionsTests.swift
git commit -m "feat: add ExtensionEntry value types and removal-condition logic"
```

---

## Task 2: OLMLXExtensions namespace, creator helper, and manifest

**Files:**
- Create: `Sources/OLMLX/Extensions/OLMLXExtensions.swift`
- Test: `Tests/OLMLXTests/ExtensionsTests.swift` (append)

- [ ] **Step 1: Write the failing test**

Append to `Tests/OLMLXTests/ExtensionsTests.swift`:

```swift
/// Minimal LanguageModel used to prove registration/resolution without real weights.
private final class StubLanguageModel: Module, LanguageModel {
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { inputs }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

@Suite("Extensions/Registration")
struct RegistrationTests {

    private func testEntry() -> ExtensionEntry {
        ExtensionEntry(
            modelType: "olmlx_unit_stub",
            kind: .architecture,
            upstreamTracking: URL(string: "https://example.com/issue/1")!,
            addedOn: "2026-05-20",
            removeWhen: .upstreamReleased(version: "99.0.0"),
            notes: "unit test stub",
            creator: { _ in StubLanguageModel() }
        )
    }

    @Test func registersEntriesOntoAFreshRegistry() async throws {
        let registry = ModelTypeRegistry<LanguageModel>()
        await OLMLXExtensions.register([testEntry()], onto: registry)

        let model = try await registry.createModel(
            configuration: Data("{}".utf8), modelType: "olmlx_unit_stub")
        #expect(model is StubLanguageModel)
    }

    @Test func unregisteredTypeStillThrows() async {
        let registry = ModelTypeRegistry<LanguageModel>()
        await OLMLXExtensions.register([testEntry()], onto: registry)

        await #expect(throws: (any Error).self) {
            _ = try await registry.createModel(
                configuration: Data("{}".utf8), modelType: "does_not_exist")
        }
    }

    @Test func everyManifestEntryHasTrackingAndIsArchitectureOrFeature() {
        for entry in OLMLXExtensions.manifest {
            #expect(!entry.upstreamTracking.absoluteString.isEmpty)
            #expect(entry.kind == .architecture || entry.kind == .feature)
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter RegistrationTests`
Expected: FAIL — `cannot find 'OLMLXExtensions' in scope`.

- [ ] **Step 3: Write minimal implementation**

Create `Sources/OLMLX/Extensions/OLMLXExtensions.swift`:

```swift
import Foundation
import MLXLLM
import MLXLMCommon

/// Registry of temporary, in-tree architecture/feature extensions layered onto mlx-swift-lm.
///
/// Each entry is a deliberate, tracked deviation from upstream. See
/// `docs/architecture.md` ("Extension layer") and the design spec for the policy:
/// every entry must carry an upstream tracking URL and a removal condition.
public enum OLMLXExtensions {

    /// Builds a `creator` closure that decodes `config.json` into `C` and instantiates `M`.
    /// Uses json5 decoding to match mlx-swift-lm's own configuration parsing.
    public static func creator<C: Decodable, M: LanguageModel>(
        _ configType: C.Type, _ make: @escaping @Sendable (C) -> M
    ) -> @Sendable (Data) throws -> LanguageModel {
        { data in
            let config = try JSONDecoder.json5().decode(C.self, from: data)
            return make(config)
        }
    }

    /// All active temporary extensions.
    ///
    /// `olmlx_canary` is a permanent self-test entry: it maps an unused `model_type`
    /// onto the upstream Llama implementation so the registration path is exercised
    /// by tests/CI without depending on a real third-party model. It is intentionally
    /// never "removable" and documents the pattern new entries should follow.
    public static let manifest: [ExtensionEntry] = [
        ExtensionEntry(
            modelType: "olmlx_canary",
            kind: .architecture,
            upstreamTracking: URL(
                string: "https://github.com/DanielPalmqvist/olmlx-swift/blob/main/docs/architecture.md")!,
            addedOn: "2026-05-20",
            removeWhen: .upstreamMerged(
                pr: URL(string: "https://example.com/never")!),
            notes: "Self-test canary mapping olmlx_canary -> Llama. Not a real model.",
            creator: creator(LlamaConfiguration.self, LlamaModel.init)
        )
    ]

    /// Registers the given entries onto a registry (overwriting any existing entry for
    /// the same `model_type`). Used by tests with a fresh registry and by `registerAll`.
    public static func register(
        _ entries: [ExtensionEntry],
        onto registry: ModelTypeRegistry<LanguageModel>
    ) async {
        for entry in entries {
            await registry.registerModelType(entry.modelType, creator: entry.creator)
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter RegistrationTests`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add Sources/OLMLX/Extensions/OLMLXExtensions.swift Tests/OLMLXTests/ExtensionsTests.swift
git commit -m "feat: add OLMLXExtensions manifest and registry registration"
```

---

## Task 3: Idempotent `registerAll()` wired into the load path

**Files:**
- Modify: `Sources/OLMLX/Extensions/OLMLXExtensions.swift`
- Modify: `Sources/OLMLX/Engine/InferenceEngine.swift:72-77`
- Test: `Tests/OLMLXTests/ExtensionsTests.swift` (append)

- [ ] **Step 1: Write the failing test**

Append to `Tests/OLMLXTests/ExtensionsTests.swift`:

```swift
@Suite("Extensions/RegisterAll")
struct RegisterAllTests {

    @Test func registerAllIsIdempotentAndRegistersTheCanary() async throws {
        // Safe to call repeatedly.
        await OLMLXExtensions.registerAll()
        await OLMLXExtensions.registerAll()

        // The canary resolves on the shared registry after registration.
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(#"{"hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 8, "num_attention_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 32, "num_key_value_heads": 2, "rope_theta": 10000.0}"#.utf8),
            modelType: "olmlx_canary")
        #expect(model is LlamaModel)
    }
}
```

> The JSON above is a minimal valid `LlamaConfiguration`. If `LlamaConfiguration`'s required keys differ, the test failure will name the missing key — add it to the JSON. Do not change the production code to accommodate the test.

Add the needed imports at the top of the test file if not already present: `import MLXLLM`.

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter RegisterAllTests`
Expected: FAIL — `type 'OLMLXExtensions' has no member 'registerAll'`.

- [ ] **Step 3: Write minimal implementation**

Add to `Sources/OLMLX/Extensions/OLMLXExtensions.swift` inside the `OLMLXExtensions` enum (and a private actor below it):

```swift
    /// Registers every manifest entry onto `LLMTypeRegistry.shared` exactly once per process.
    /// Safe to call from any load path; concurrent and repeat calls are coalesced.
    public static func registerAll() async {
        guard await registrationGuard.beginIfNeeded() else { return }
        await register(manifest, onto: LLMTypeRegistry.shared)
    }
```

Add at file scope (below the enum):

```swift
/// Ensures `registerAll` runs its body at most once per process.
private actor RegistrationGuard {
    private var done = false
    /// Returns true exactly once; false on every subsequent call.
    func beginIfNeeded() -> Bool {
        if done { return false }
        done = true
        return true
    }
}

private let registrationGuard = RegistrationGuard()
```

- [ ] **Step 4: Wire into the load path**

In `Sources/OLMLX/Engine/InferenceEngine.swift`, change `loadModel` (currently lines 72-77):

```swift
    public func loadModel(from directory: URL) async throws -> ModelContainer {
        await OLMLXExtensions.registerAll()
        return try await MLXLMCommon.loadModelContainer(
            from: directory,
            using: tokenizerLoader
        )
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `swift test --filter RegisterAllTests`
Expected: PASS (1 test).
Run: `swift build`
Expected: builds clean.

- [ ] **Step 6: Commit**

```bash
git add Sources/OLMLX/Extensions/OLMLXExtensions.swift Sources/OLMLX/Engine/InferenceEngine.swift Tests/OLMLXTests/ExtensionsTests.swift
git commit -m "feat: register extensions idempotently from the model load path"
```

---

## Task 4: `olmlx ext list` and `olmlx ext check` CLI

**Files:**
- Create: `Sources/olmlx-cli/ExtensionsCommand.swift`
- Modify: `Sources/olmlx-cli/OLMXCLI.swift:13-22`
- Test: `Tests/OLMLXTests/ExtensionsTests.swift` (append)

The display and gating logic are pure functions on the OLMLX library (testable); the CLI commands are thin wrappers. Put the pure helpers in the library file `OLMLXExtensions.swift`.

- [ ] **Step 1: Write the failing test**

Append to `Tests/OLMLXTests/ExtensionsTests.swift`:

```swift
@Suite("Extensions/Reporting")
struct ReportingTests {

    private func entries() -> [ExtensionEntry] {
        [
            ExtensionEntry(
                modelType: "a_model",
                kind: .architecture,
                upstreamTracking: URL(string: "https://example.com/pr/1")!,
                addedOn: "2026-05-20",
                removeWhen: .upstreamReleased(version: "3.32.0"),
                notes: "n1",
                creator: { _ in fatalError() }),
            ExtensionEntry(
                modelType: "b_feature",
                kind: .feature,
                upstreamTracking: URL(string: "https://example.com/pr/2")!,
                addedOn: "2026-05-20",
                removeWhen: .upstreamMerged(pr: URL(string: "https://example.com/pr/2")!),
                notes: "n2",
                creator: { _ in fatalError() }),
        ]
    }

    @Test func listSummaryNamesEveryEntry() {
        let text = OLMLXExtensions.listSummary(entries())
        #expect(text.contains("a_model"))
        #expect(text.contains("b_feature"))
        #expect(text.contains("3.32.0"))
    }

    @Test func removableEntriesAreFlaggedAtPinnedVersion() {
        let removable = OLMLXExtensions.removableEntries(entries(), pinnedVersion: "3.32.0")
        #expect(removable.map(\.modelType) == ["a_model"])

        let none = OLMLXExtensions.removableEntries(entries(), pinnedVersion: "3.31.3")
        #expect(none.isEmpty)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ReportingTests`
Expected: FAIL — `type 'OLMLXExtensions' has no member 'listSummary'`.

- [ ] **Step 3: Write minimal implementation (library helpers)**

Add to `Sources/OLMLX/Extensions/OLMLXExtensions.swift` inside the enum:

```swift
    /// Human-readable, one-line-per-entry summary for `olmlx ext list`.
    public static func listSummary(_ entries: [ExtensionEntry]) -> String {
        if entries.isEmpty { return "No active extensions." }
        return entries.map { entry in
            let removal: String
            switch entry.removeWhen {
            case .upstreamReleased(let v): removal = "remove when upstream >= \(v)"
            case .upstreamMerged(let pr): removal = "remove when merged: \(pr.absoluteString)"
            }
            return "[\(entry.kind.rawValue)] \(entry.modelType) — \(removal) — tracking \(entry.upstreamTracking.absoluteString) (added \(entry.addedOn))"
        }.joined(separator: "\n")
    }

    /// Entries whose removal condition is satisfied at the given pinned upstream version.
    public static func removableEntries(
        _ entries: [ExtensionEntry], pinnedVersion: String
    ) -> [ExtensionEntry] {
        entries.filter { $0.removeWhen.isRemovable(pinnedVersion: pinnedVersion) }
    }
```

- [ ] **Step 4: Run library tests to verify they pass**

Run: `swift test --filter ReportingTests`
Expected: PASS (2 tests).

- [ ] **Step 5: Add the CLI command**

Create `Sources/olmlx-cli/ExtensionsCommand.swift`:

```swift
import ArgumentParser
import Foundation
import OLMLX

struct Ext: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ext",
        abstract: "Inspect temporary architecture/feature extensions",
        subcommands: [ExtList.self, ExtCheck.self]
    )
}

struct ExtList: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list", abstract: "List active temporary extensions")

    func run() throws {
        print(OLMLXExtensions.listSummary(OLMLXExtensions.manifest))
    }
}

struct ExtCheck: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "check",
        abstract: "Fail if any extension can be removed at the given upstream version")

    @Option(name: .long, help: "Pinned mlx-swift-lm version to check against, e.g. 3.31.3")
    var pinnedVersion: String

    func run() throws {
        let removable = OLMLXExtensions.removableEntries(
            OLMLXExtensions.manifest, pinnedVersion: pinnedVersion)
        if removable.isEmpty {
            print("No removable extensions at upstream \(pinnedVersion).")
            return
        }
        let names = removable.map(\.modelType).joined(separator: ", ")
        throw ValidationError(
            "These extensions can now be removed (upstream \(pinnedVersion) covers them): \(names)")
    }
}
```

- [ ] **Step 6: Register the command**

In `Sources/olmlx-cli/OLMXCLI.swift`, add `Ext.self` to the `subcommands` array (after `Service.self`, line ~19):

```swift
        subcommands: [
            Serve.self,
            Models.self,
            Chat.self,
            Bench.self,
            Config.self,
            Service.self,
            Ext.self,
        ],
```

- [ ] **Step 7: Build and smoke-test the CLI**

Run: `swift build`
Expected: builds clean.
Run: `swift run olmlx ext list`
Expected: prints a line containing `olmlx_canary`.
Run: `swift run olmlx ext check --pinned-version 3.31.3`
Expected: prints `No removable extensions at upstream 3.31.3.` and exits 0.

- [ ] **Step 8: Commit**

```bash
git add Sources/olmlx-cli/ExtensionsCommand.swift Sources/olmlx-cli/OLMXCLI.swift Tests/OLMLXTests/ExtensionsTests.swift
git commit -m "feat: add 'olmlx ext list' and 'olmlx ext check' commands"
```

---

## Task 5: Document the extension mechanism

**Files:**
- Modify: `docs/architecture.md`

- [ ] **Step 1: Add a section**

Append a `## Extension layer` section to `docs/architecture.md` covering:
- Why it exists (upstream adoption lag — link the spec at `docs/superpowers/specs/2026-05-19-architecture-feature-extension-layer-design.md`).
- How registration works: `OLMLXExtensions.registerAll()` runs once from `DefaultInferenceEngine.loadModel`, layering `manifest` entries onto `LLMTypeRegistry.shared`.
- How to **add** an entry: append an `ExtensionEntry` to `manifest` with a mandatory `upstreamTracking` URL and a `removeWhen` condition; implement the architecture as a `LanguageModel` and pass it via `creator(_:_:)`.
- How to **remove** an entry: run `olmlx ext check --pinned-version <current>`; if it names an entry, delete that `ExtensionEntry` (and its architecture file) and bump the pinned upstream version. The canary entry is permanent.
- The discipline rules from the spec (≤5 active architecture overrides; one file per architecture; tracking URL mandatory).

- [ ] **Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: document the temporary extension layer"
```

---

## Self-Review notes

- **Spec coverage:** This plan implements the spec's "Phase 1 (Foundation)" — registry mechanism (§3.1, §4), `ExtensionManifest` with tracking metadata (§4), lifecycle gating via `ext check` (§5 rule 2), and docs. Spec Phases 2-5 (real architecture ports, KV-cache/sampling/speculative features) are explicitly deferred to follow-up plans; each is a separate spec→plan cycle because each is a non-trivial port.
- **Mechanism deviation from spec** is called out at the top (register-onto-shared vs. wrap-factory) with the reason (private `creators`; preserve VLM fallback).
- **Type consistency:** `creator` is `@Sendable (Data) throws -> LanguageModel` everywhere; `register(_:onto:)`, `registerAll()`, `listSummary(_:)`, `removableEntries(_:pinnedVersion:)`, `RemovalCondition.isRemovable(pinnedVersion:)` names are used identically across tasks and tests.
- **Verification:** unit tests cover removal logic, registration onto a fresh registry, idempotent `registerAll` against the real shared registry via the canary, and reporting helpers. CLI smoke-tested via `swift run olmlx ext list` / `ext check`.

## End-to-end verification

1. `swift test --filter Extensions` → all extension suites pass.
2. `swift build && swift run olmlx ext list` → shows `olmlx_canary`.
3. `swift run olmlx ext check --pinned-version 3.31.3` → exits 0; `--pinned-version 99.0.0` would still exit 0 (canary uses `.upstreamMerged`, never auto-removable) — confirms the canary is sticky.
4. Confirm no regression in existing load path: `swift test` (full suite) stays green.
