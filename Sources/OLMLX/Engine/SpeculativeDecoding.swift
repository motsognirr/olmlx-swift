import Foundation
import MLX
import MLXLMCommon

// MARK: - Runtime

/// Per-request binding to a draft model for speculative decoding.
///
/// Built by ``makeSpeculativeRuntime(manager:config:)`` after resolving the
/// per-model and global settings; passed into the generation runners so they
/// can hand the draft model to MLX's speculative `generate` overload.
public struct SpeculativeRuntime: Sendable {
    public let draftContainer: ModelContainer
    public let numDraftTokens: Int

    public init(draftContainer: ModelContainer, numDraftTokens: Int) {
        self.draftContainer = draftContainer
        self.numDraftTokens = numDraftTokens
    }
}

/// Wraps a non-Sendable value so it can cross `@Sendable` closure boundaries.
///
/// Used internally by the speculative-decoding runners to carry the draft
/// model into the main container's `perform` block — the draft container's
/// `perform` already serializes access to the underlying model, so passing
/// the reference through is safe for the lifetime of the nested call.
final class UncheckedSendableRef<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

// MARK: - Cache Compatibility

/// Returns `true` when every slot in `cache` reports `isTrimmable`. MLX's
/// `SpeculativeTokenIterator.init` requires this before it will start, so the
/// generation runners use it to detect main models whose architecture cannot
/// participate in speculative decoding (hybrid SSM/transformer families like
/// Qwen3.5 / FalconH1 / BaichuanM1 / Qwen3-Next emit `MambaCache` slots that
/// are not trimmable). Thin wrapper around MLX's `canTrimPromptCache` so call
/// sites stay decoupled from the underlying name.
public func cacheSupportsSpeculative(_ cache: [KVCache]) -> Bool {
    canTrimPromptCache(cache)
}

// MARK: - Runtime Construction

/// Builds a ``SpeculativeRuntime`` from the resolved ``SpeculativeConfig``.
///
/// Returns `nil` when speculative decoding is disabled. Throws
/// ``InferenceError/unsupportedSpeculativeStrategy(_:)`` for `.dflash` / `.eagle`
/// (only `.classic` is implemented) and
/// ``InferenceError/speculativeDraftModelMissing`` when no draft model is
/// configured. Loads the draft container via ``ModelManager/ensureDraftLoaded(hfPath:)``.
public func makeSpeculativeRuntime(
    manager: ModelManager,
    config: SpeculativeConfig
) async throws -> SpeculativeRuntime? {
    guard config.enabled else { return nil }
    switch config.strategy {
    case .classic:
        break
    case .dflash, .eagle:
        throw InferenceError.unsupportedSpeculativeStrategy(config.strategy.rawValue)
    }
    guard let draftPath = config.draftModel, !draftPath.isEmpty else {
        throw InferenceError.speculativeDraftModelMissing
    }
    let container = try await manager.ensureDraftLoaded(hfPath: draftPath)
    // MLX's default in `generate(..., numDraftTokens:)` is 2; honor a positive
    // override from config but never let it drop below 1.
    let numTokens = max(1, config.numTokens ?? 2)
    return SpeculativeRuntime(draftContainer: container, numDraftTokens: numTokens)
}

/// Probes the main model and clears `speculative` when its cache architecture
/// can't participate in speculative decoding.
///
/// MLX's `SpeculativeTokenIterator` rejects non-trimmable caches at init time,
/// which fatally fails the request for hybrid SSM/transformer models (Qwen3.5,
/// FalconH1, BaichuanM1, Qwen3-Next, …) whose `newCache(parameters:)` returns
/// `MambaCache` slots. Rather than surface the cryptic error per request, we
/// build a fresh cache once, run it through ``cacheSupportsSpeculative(_:)``,
/// and on miss log a single warning and fall back to non-speculative decoding
/// so the model still answers. Allocating a fresh cache is cheap — no MLX
/// compute happens until first update.
func resolveSpeculativeForRequest(
    container: ModelContainer,
    parameters: GenerateParameters,
    speculative: SpeculativeRuntime?
) async -> SpeculativeRuntime? {
    guard let speculative else { return nil }
    let supported = await container.perform { context in
        cacheSupportsSpeculative(context.model.newCache(parameters: parameters))
    }
    if supported { return speculative }
    FileHandle.standardError.write(
        Data(
            "warning: main model produced a non-trimmable KV cache (hybrid SSM/transformer architecture is incompatible with speculative decoding); falling back to standard generation for this request\n"
                .utf8))
    return nil
}

// MARK: - Generation Runner

/// Shared body for ``runGeneration`` and ``runStreamingGeneration`` when
/// speculative decoding is enabled.
///
/// Nests `mainContainer.perform` inside `draftContainer.perform` so both KV
/// caches are touched under their respective serial-access locks for the
/// duration of the call. The draft model is carried into the inner closure via
/// an ``UncheckedSendableRef`` — `LanguageModel` is not `Sendable` but the draft
/// container's lock guarantees no concurrent access while we hold it.
@discardableResult
func runSpeculativeGeneration(
    container mainContainer: ModelContainer,
    messages: [[String: any Sendable]],
    tools: [[String: any Sendable]]?,
    parameters: GenerateParameters,
    speculative: SpeculativeRuntime,
    collect: Bool,
    onChunk: (@Sendable (String) -> Void)? = nil,
    onComplete: (@Sendable (GenerateCompletionInfo?) -> Void)? = nil
) async throws -> (text: String, info: GenerateCompletionInfo?) {
    let numDraftTokens = speculative.numDraftTokens
    return try await speculative.draftContainer.perform { draftContext -> (String, GenerateCompletionInfo?) in
        let draftModelRef = UncheckedSendableRef<any LanguageModel>(draftContext.model)
        return try await mainContainer.perform { mainContext -> (String, GenerateCompletionInfo?) in
            let tokenIds = try mainContext.tokenizer.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: nil)

            let promptArray = MLXArray(tokenIds).asType(.int32)
            let mask = MLXArray.ones([promptArray.size]).asType(.bool)
            let input = LMInput(
                text: LMInput.Text(tokens: promptArray, mask: mask),
                image: nil, video: nil)

            var fullText = ""
            var completionInfo: GenerateCompletionInfo?

            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: parameters,
                context: mainContext,
                draftModel: draftModelRef.value,
                numDraftTokens: numDraftTokens
            )
            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    if collect { fullText += text }
                    onChunk?(text)
                case .info(let info):
                    completionInfo = info
                    onComplete?(info)
                case .toolCall:
                    break
                }
            }
            return (fullText, completionInfo)
        }
    }
}
