import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Registry of temporary, in-tree architecture/feature extensions layered onto mlx-swift-lm.
///
/// Each entry is a deliberate, tracked deviation from upstream. Policy: every entry must
/// carry an upstream tracking URL and a removal condition. See docs/architecture.md.
public enum OLMLXExtensions {

    /// Builds a `creator` closure that decodes `config.json` into `C` and instantiates `M`.
    /// Uses json5 decoding to match mlx-swift-lm's own configuration parsing.
    public static func creator<C: Decodable & Sendable, M: LanguageModel>(
        _ configType: C.Type, _ make: @escaping @Sendable (C) -> M
    ) -> @Sendable (Data) throws -> any LanguageModel {
        { data in
            let config = try JSONDecoder.json5().decode(C.self, from: data)
            return make(config)
        }
    }

    /// All active temporary extensions.
    ///
    /// `olmlx_canary` is a permanent self-test entry: it maps an unused `model_type`
    /// onto a trivial built-in ``CanaryModel`` so the registration path is exercised by
    /// tests/CI without depending on a real model or the MLX metal library. It is
    /// intentionally never auto-removable and documents the pattern new entries follow.
    public static let manifest: [ExtensionEntry] = [
        ExtensionEntry(
            modelType: "olmlx_canary",
            kind: .architecture,
            upstreamTracking: URL(
                string:
                    "https://github.com/DanielPalmqvist/olmlx-swift/blob/main/docs/architecture.md"
            )!,
            addedOn: "2026-05-20",
            // sentinel URL: .upstreamMerged is never auto-gated, so the canary is never removed.
            removeWhen: .upstreamMerged(pr: URL(string: "https://example.com/never")!),
            notes: "Self-test canary: a trivial built-in model proving the registration path.",
            creator: { _ in CanaryModel() }
        ),
        ExtensionEntry(
            modelType: "gemma4",
            kind: .architecture,
            // Gemma 4 family umbrella issue upstream (covers the MoE + 31B gaps).
            upstreamTracking: URL(
                string: "https://github.com/ml-explore/mlx-swift-lm/issues/282")!,
            addedOn: "2026-05-20",
            // placeholder version: no upstream mlx-swift-lm release carries these fixes yet
            removeWhen: .upstreamReleased(version: "99.0.0"),
            notes: "Gemma4 override fixing three upstream gaps. Tracking: "
                + "quantizable per-layer projection (#57) -> mlx-swift-lm PR #244; "
                + "MoE experts/router dual-FFN (#58) -> mlx-swift-lm issue #282 (Gap 2); "
                + "k_eq_v value layout crash (#59) -> mlx-swift-lm issue #231. "
                + "Remove when a release carrying all three is pinned "
                + "(removeWhen 99.0.0 is a placeholder until then).",
            creator: creator(Gemma4PatchedConfiguration.self, Gemma4PatchedModel.init)
        ),
        ExtensionEntry(
            modelType: "gemma4_text",
            kind: .architecture,
            // Gemma 4 family umbrella issue upstream (see the gemma4 entry for per-fix links).
            upstreamTracking: URL(
                string: "https://github.com/ml-explore/mlx-swift-lm/issues/282")!,
            addedOn: "2026-05-20",
            // placeholder version: no upstream mlx-swift-lm release carries these fixes yet
            removeWhen: .upstreamReleased(version: "99.0.0"),
            notes: "Text-only Gemma4 override; see the gemma4 entry for upstream tracking "
                + "(PR #244 / issues #282, #231). "
                + "(removeWhen 99.0.0 is a placeholder until a fixed release exists.)",
            creator: creator(
                Gemma4PatchedTextConfiguration.self, Gemma4PatchedTextModel.init)
        ),
    ]

    /// Registers the given entries onto a registry (overwriting any existing entry for
    /// the same `model_type`). Used by tests with a fresh registry and by `registerAll`.
    public static func register(
        _ entries: [ExtensionEntry],
        onto registry: ModelTypeRegistry<any LanguageModel>
    ) async {
        for entry in entries {
            await registry.registerModelType(entry.modelType, creator: entry.creator)
        }
    }

    /// Human-readable, one-line-per-entry summary for `olmlx ext list`.
    public static func listSummary(_ entries: [ExtensionEntry]) -> String {
        if entries.isEmpty { return "No active extensions." }
        return entries.map { entry in
            let removal: String
            switch entry.removeWhen {
            case .upstreamReleased(let v): removal = "remove when upstream >= \(v)"
            case .upstreamMerged(let pr): removal = "remove when merged: \(pr.absoluteString)"
            }
            return "[\(entry.kind.rawValue)] \(entry.modelType) — \(removal) "
                + "— tracking \(entry.upstreamTracking.absoluteString) (added \(entry.addedOn))"
                + " — \(entry.notes)"
        }.joined(separator: "\n")
    }

    /// Entries whose removal condition is satisfied at the given pinned upstream version.
    public static func removableEntries(
        _ entries: [ExtensionEntry], pinnedVersion: String
    ) -> [ExtensionEntry] {
        entries.filter { $0.removeWhen.isRemovable(pinnedVersion: pinnedVersion) }
    }

    /// Registers every manifest entry onto `LLMTypeRegistry.shared` exactly once per process.
    /// Safe to call from any load path; concurrent and repeat calls all await the same
    /// one-time registration.
    public static func registerAll() async {
        await registrationGuard.run {
            await register(manifest, onto: LLMTypeRegistry.shared)
        }
    }
}

/// Trivial, weightless `LanguageModel` backing the `olmlx_canary` self-test entry.
/// It allocates no parameters, so it can be constructed in unit tests without the MLX
/// metal library. It is never used to serve a real model.
final class CanaryModel: Module, LanguageModel {
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { inputs }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

/// Ensures the registration body runs at most once per process and that every caller
/// — including concurrent ones — awaits its completion.
private actor RegistrationGuard {
    private var task: Task<Void, Never>?
    func run(_ body: @escaping @Sendable () async -> Void) async {
        if task == nil {
            task = Task { await body() }
        }
        await task!.value
    }
}

private let registrationGuard = RegistrationGuard()
