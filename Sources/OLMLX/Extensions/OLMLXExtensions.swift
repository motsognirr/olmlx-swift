import Foundation
import MLXLLM
import MLXLMCommon

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
    /// onto the upstream Llama implementation so the registration path is exercised
    /// by tests/CI without depending on a real third-party model. It is intentionally
    /// never auto-removable and documents the pattern new entries should follow.
    public static let manifest: [ExtensionEntry] = [
        ExtensionEntry(
            modelType: "olmlx_canary",
            kind: .architecture,
            upstreamTracking: URL(
                string: "https://github.com/DanielPalmqvist/olmlx-swift/blob/main/docs/architecture.md")!,
            addedOn: "2026-05-20",
            removeWhen: .upstreamMerged(pr: URL(string: "https://example.com/never")!),  // sentinel: canary is never removed
            notes: "Self-test canary mapping olmlx_canary -> Llama. Not a real model.",
            creator: creator(LlamaConfiguration.self, { LlamaModel($0) })
        )
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

    /// Registers every manifest entry onto `LLMTypeRegistry.shared` exactly once per process.
    /// Safe to call from any load path; concurrent and repeat calls are coalesced.
    public static func registerAll() async {
        guard await registrationGuard.beginIfNeeded() else { return }
        await register(manifest, onto: LLMTypeRegistry.shared)
    }
}

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
