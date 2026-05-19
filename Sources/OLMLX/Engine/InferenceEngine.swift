import Foundation
import MLX
import MLXLMCommon
import Tokenizers

// MARK: - Tokenizer Adapter

/// Bridges ``Tokenizers.Tokenizer`` (swift-transformers) to ``MLXLMCommon.Tokenizer``.
struct TokenizerAdapter: MLXLMCommon.Tokenizer {
    let wrapped: any Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        wrapped.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        wrapped.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        wrapped.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        wrapped.convertIdToToken(id)
    }

    var bosToken: String? { wrapped.bosToken }
    var eosToken: String? { wrapped.eosToken }
    var unknownToken: String? { wrapped.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try wrapped.applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: additionalContext
        )
    }
}

/// ``TokenizerLoader`` implementation using swift-transformers' ``AutoTokenizer``.
public struct MLXTokenizerLoader: MLXLMCommon.TokenizerLoader, Sendable {
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let tok = try await AutoTokenizer.from(modelFolder: directory)
        return TokenizerAdapter(wrapped: tok)
    }
}

// MARK: - Inference Engine Protocol

/// Protocol wrapping model inference operations for testability.
public protocol InferenceEngineProtocol: AnyObject, Sendable {
    func loadModel(from directory: URL) async throws -> ModelContainer
}

// MARK: - Default Engine

/// Default inference engine that loads models using mlx-swift-lm.
public final class DefaultInferenceEngine: InferenceEngineProtocol {
    private let tokenizerLoader: any TokenizerLoader

    public init(tokenizerLoader: any TokenizerLoader = MLXTokenizerLoader()) {
        self.tokenizerLoader = tokenizerLoader
    }

    public func loadModel(from directory: URL) async throws -> ModelContainer {
        try await MLXLMCommon.loadModelContainer(
            from: directory,
            using: tokenizerLoader
        )
    }
}

// MARK: - Generation Parameter Mapping

/// Converts ``ModelOptions`` (Ollama API options) to ``GenerateParameters``.
public func mapToGenerateParameters(from options: ModelOptions?) -> GenerateParameters {
    guard let opts = options else {
        return GenerateParameters()
    }

    var params = GenerateParameters()

    if let t = opts.temperature {
        params.temperature = Float(t)
    }
    if let p = opts.topP {
        params.topP = Float(p)
    }
    if let k = opts.topK {
        params.topK = k
    }
    if let n = opts.numPredict {
        params.maxTokens = n
    }
    if let r = opts.repeatPenalty {
        params.repetitionPenalty = Float(r)
    }
    if let rn = opts.repeatLastN {
        params.repetitionContextSize = rn
    }
    if let mp = opts.minP {
        params.minP = Float(mp)
    }
    if let pp = opts.presencePenalty {
        params.presencePenalty = Float(pp)
    }
    if let fp = opts.frequencyPenalty {
        params.frequencyPenalty = Float(fp)
    }
    if opts.typicalP != nil {
        // MLX `GenerateParameters` does not currently expose a typical-p sampler;
        // drop the value rather than silently overwriting topP.
        FileHandle.standardError.write(
            Data("warning: typical_p is not supported by the MLX sampler; ignoring\n".utf8))
    }
    if let seed = opts.seed {
        _ = seed
    }

    return params
}

/// Applies a KV cache quantization spec (e.g. `affine:4`) to ``GenerateParameters``.
///
/// Sets ``GenerateParameters/kvBits`` so MLX's `maybeQuantizeKVCache` converts
/// per-layer simple caches into ``QuantizedKVCache`` once the prefill threshold
/// passes. The spec format is validated by ``parseKVCacheQuant``.
public func applyKVCacheQuant(_ params: inout GenerateParameters, spec: String?) {
    guard let spec, let parsed = parseKVCacheQuant(spec) else { return }
    params.kvBits = parsed.bits
    // MLX default group size; not currently configurable per request.
    params.kvGroupSize = 64
}

// MARK: - Chat Message Helpers

/// Converts OLMLX ``Message`` array to raw dict format for chat template application.
public func messagesToRaw(_ messages: [Message]) -> [[String: any Sendable]] {
    messages.map { msg in
        var dict: [String: any Sendable] = [
            "role": msg.role,
            "content": msg.content,
        ]
        if let thinking = msg.thinking {
            dict["thinking"] = thinking
        }
        return dict
    }
}

/// Converts OLMLX ``Tool`` array to raw dict format for chat template application.
public func toolsToRaw(_ tools: [Tool]?) -> [[String: any Sendable]]? {
    guard let tools else { return nil }
    return tools.map { tool in
        var dict: [String: any Sendable] = ["type": tool.type]
        var funcDict: [String: any Sendable] = [:]
        for (key, value) in tool.function {
            funcDict[key] = anyCodableToSendable(value)
        }
        dict["function"] = funcDict
        return dict
    }
}

public func anyCodableToSendable(_ value: AnyCodable) -> any Sendable {
    switch value {
    case .string(let s): return s
    case .int(let i): return i
    case .double(let d): return d
    case .bool(let b): return b
    case .dictionary(let d):
        var result: [String: any Sendable] = [:]
        for (k, v) in d { result[k] = anyCodableToSendable(v) }
        return result
    case .array(let a):
        return a.map { anyCodableToSendable($0) }
    case .null:
        return NSNull()
    }
}

// MARK: - Mock Engine (for tests)

public enum MockInferenceError: Error, Sendable, Equatable {
    /// `loadModel` was called on a `MockInferenceEngine` that has no `stubContainer`
    /// configured. Set `stubContainer` to a real `ModelContainer` if the test needs
    /// one, or assert via `loadedModels` that the call was made.
    case noStubContainerConfigured(String)
}

public final class MockInferenceEngine: InferenceEngineProtocol, @unchecked Sendable {
    public var loadedModels: [String] = []
    public var shouldFail: Bool = false

    /// Optional container to return from `loadModel`. Tests that need a real
    /// container (rare) can supply one here; most tests should leave it nil
    /// and assert on `loadedModels` instead.
    public var stubContainer: ModelContainer?

    public init() {}

    public func loadModel(from directory: URL) async throws -> ModelContainer {
        if shouldFail {
            throw InferenceError.modelNotLoaded(directory.path)
        }
        loadedModels.append(directory.lastPathComponent)
        if let stub = stubContainer {
            return stub
        }
        throw MockInferenceError.noStubContainerConfigured(directory.lastPathComponent)
    }
}

// MARK: - Prompt Cache Binding

/// Per-request binding to a ``ModelManager``'s prompt cache.
///
/// Passed into ``runGeneration(container:messages:tools:parameters:promptCache:)`` and
/// ``runStreamingGeneration(container:messages:tools:parameters:onChunk:onComplete:promptCache:)``
/// so they can reuse / update the per-model KV cache without each route having to
/// reach into the manager. `nil` disables prompt caching for that call.
public struct PromptCacheBinding: Sendable {
    public let manager: ModelManager
    public let key: String
    public let maxTokens: Int?

    public init(manager: ModelManager, key: String, maxTokens: Int?) {
        self.manager = manager
        self.key = key
        self.maxTokens = maxTokens
    }
}

/// Builds a ``PromptCacheBinding`` honoring ``Settings/promptCache`` and
/// ``Settings/promptCacheMaxTokens``. Returns `nil` when caching is disabled.
public func makePromptCacheBinding(
    manager: ModelManager, modelName: String
) -> PromptCacheBinding? {
    guard manager.settings.promptCache else { return nil }
    return PromptCacheBinding(
        manager: manager,
        key: ModelRegistry.normalizeName(modelName),
        maxTokens: manager.settings.promptCacheMaxTokens
    )
}

/// Outcome of matching a new prompt against an existing cache slot.
///
/// `cache` is what to pass to MLX (`nil` => fresh cache will be allocated).
/// `prefillTokens` is the slice of `tokenIds` that still needs to be fed in.
struct PromptCacheReuse {
    let cache: [KVCache]?
    let prefillTokens: [Int]
    let reusedTokens: Int
}

/// Decide how much of `cached` can be reused for `newTokens`.
///
/// On a hit, trims the cache in-place to the longest common prefix and returns
/// only the remaining suffix as `prefillTokens`. On a miss, returns `nil` cache
/// and the full prompt so the caller allocates a fresh KV cache.
///
/// The MLX prompt-prefill path requires at least one token to feed forward, so
/// when `newTokens` is a strict prefix of `cached.tokens` we leave one token
/// out of the trim to preserve a non-empty prefill.
func planPromptCacheReuse(
    cached: CachedPromptState?, newTokens: [Int]
) -> PromptCacheReuse {
    guard let cached, let cache = cached.cache, !cache.isEmpty,
        canTrimPromptCache(cache)
    else {
        return PromptCacheReuse(cache: nil, prefillTokens: newTokens, reusedTokens: 0)
    }

    let common = longestCommonPrefix(cached.tokens, newTokens)
    // We need to prefill at least one token, so cap reuse at newTokens.count - 1.
    let reusable = min(common, max(0, newTokens.count - 1))
    guard reusable > 0 else {
        return PromptCacheReuse(cache: nil, prefillTokens: newTokens, reusedTokens: 0)
    }

    let currentOffset = cache[0].offset
    let trim = currentOffset - reusable
    guard trim >= 0 else {
        // Cache offset is behind the prefix match — shouldn't happen, but bail safely.
        return PromptCacheReuse(cache: nil, prefillTokens: newTokens, reusedTokens: 0)
    }
    if trim > 0 {
        trimPromptCache(cache, numTokens: trim)
    }
    return PromptCacheReuse(
        cache: cache,
        prefillTokens: Array(newTokens[reusable...]),
        reusedTokens: reusable
    )
}

// MARK: - Speculative Decoding

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

// MARK: - Generation Runners

/// Runs generation via a ``ModelContainer`` and collects the full response.
///
/// When `speculative` is non-nil the draft model from `speculative.draftContainer`
/// is paired with `container` via MLX's speculative-decoding `generate` overload.
/// In that mode prompt-cache reuse is bypassed — the speculative iterator manages
/// its own main + draft KV caches.
public func runGeneration(
    container: ModelContainer,
    messages: [[String: any Sendable]],
    tools: [[String: any Sendable]]?,
    parameters: GenerateParameters,
    promptCache: PromptCacheBinding? = nil,
    speculative: SpeculativeRuntime? = nil
) async throws -> (text: String, info: GenerateCompletionInfo?) {
    if let speculative {
        return try await runSpeculativeGeneration(
            container: container,
            messages: messages, tools: tools,
            parameters: parameters, speculative: speculative,
            collect: true
        )
    }

    return try await container.perform { context -> (String, GenerateCompletionInfo?) in
        let tokenIds = try context.tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil)

        let cached: CachedPromptState?
        if let binding = promptCache {
            cached = await binding.manager.acquirePromptCache(key: binding.key)
        } else {
            cached = nil
        }
        let reuse = planPromptCacheReuse(cached: cached, newTokens: tokenIds)
        let kvCache: [KVCache] =
            reuse.cache ?? context.model.newCache(parameters: parameters)

        let promptArray = MLXArray(reuse.prefillTokens).asType(.int32)
        let mask = MLXArray.ones([promptArray.size]).asType(.bool)
        let input = LMInput(
            text: LMInput.Text(tokens: promptArray, mask: mask),
            image: nil, video: nil)

        var fullText = ""
        var completionInfo: GenerateCompletionInfo?

        let stream = try MLXLMCommon.generate(
            input: input, cache: kvCache, parameters: parameters, context: context)
        for await generation in stream {
            switch generation {
            case .chunk(let text):
                fullText += text
            case .info(let info):
                completionInfo = info
            case .toolCall:
                break
            }
        }

        if let binding = promptCache {
            await storePromptCacheSlot(
                binding: binding, tokens: tokenIds, cache: kvCache)
        }

        return (fullText, completionInfo)
    }
}

/// Runs streaming generation via a ``ModelContainer``, calling back for each chunk.
///
/// When `speculative` is non-nil the draft model from `speculative.draftContainer`
/// is paired with `container` via MLX's speculative-decoding `generate` overload.
/// In that mode prompt-cache reuse is bypassed — the speculative iterator manages
/// its own main + draft KV caches.
public func runStreamingGeneration(
    container: ModelContainer,
    messages: [[String: any Sendable]],
    tools: [[String: any Sendable]]?,
    parameters: GenerateParameters,
    onChunk: @Sendable @escaping (String) -> Void,
    onComplete: @Sendable @escaping (GenerateCompletionInfo?) -> Void,
    promptCache: PromptCacheBinding? = nil,
    speculative: SpeculativeRuntime? = nil
) async throws {
    if let speculative {
        _ = try await runSpeculativeGeneration(
            container: container,
            messages: messages, tools: tools,
            parameters: parameters, speculative: speculative,
            collect: false,
            onChunk: onChunk, onComplete: onComplete
        )
        return
    }

    try await container.perform { context in
        let tokenIds = try context.tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil)

        let cached: CachedPromptState?
        if let binding = promptCache {
            cached = await binding.manager.acquirePromptCache(key: binding.key)
        } else {
            cached = nil
        }
        let reuse = planPromptCacheReuse(cached: cached, newTokens: tokenIds)
        let kvCache: [KVCache] =
            reuse.cache ?? context.model.newCache(parameters: parameters)

        let promptArray = MLXArray(reuse.prefillTokens).asType(.int32)
        let mask = MLXArray.ones([promptArray.size]).asType(.bool)
        let input = LMInput(
            text: LMInput.Text(tokens: promptArray, mask: mask),
            image: nil, video: nil)

        let stream = try MLXLMCommon.generate(
            input: input, cache: kvCache, parameters: parameters, context: context)

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                onChunk(text)
            case .info(let info):
                onComplete(info)
            case .toolCall:
                break
            }
        }

        if let binding = promptCache {
            await storePromptCacheSlot(
                binding: binding, tokens: tokenIds, cache: kvCache)
        }
    }
}

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
    return try await speculative.draftContainer.perform {
        draftContext -> (String, GenerateCompletionInfo?) in
        let draftModelRef = UncheckedSendableRef<any LanguageModel>(draftContext.model)
        return try await mainContainer.perform {
            mainContext -> (String, GenerateCompletionInfo?) in
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

/// Either stores or evicts the post-generation cache, respecting the configured
/// `promptCacheMaxTokens` ceiling.
private func storePromptCacheSlot(
    binding: PromptCacheBinding, tokens: [Int], cache: [KVCache]
) async {
    let offset = cache.first?.offset ?? tokens.count
    if let cap = binding.maxTokens, offset > cap {
        await binding.manager.releasePromptCache(key: binding.key, state: nil)
        return
    }
    let state = CachedPromptState(tokens: tokens, cache: cache)
    await binding.manager.releasePromptCache(key: binding.key, state: state)
}
