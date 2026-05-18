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

// MARK: - Prompt Cache Plumbing

/// Configuration handed to `runGeneration`/`runStreamingGeneration` to enable
/// KV-cache reuse across requests.
public struct PromptCacheContext: Sendable {
    public let store: PromptCacheStore
    public let key: String
    public let maxTokens: Int?

    public init(store: PromptCacheStore, key: String, maxTokens: Int?) {
        self.store = store
        self.key = key
        self.maxTokens = maxTokens
    }
}

/// Result of consulting the cache and preparing tokens for inference.
private struct PreparedPrompt {
    let lmInput: LMInput
    let kvCaches: [KVCache]
    let fullPromptTokens: [Int]
    let reusedFromCache: Bool
}

/// Bundled outcome of `runGeneration`'s perform block. Wrapped in a struct
/// rather than a tuple so it sails past SwiftLint's `large_tuple` rule.
private struct GenerationOutcome {
    let text: String
    let info: GenerateCompletionInfo?
    let postEntry: PromptCacheEntry
}

private func prepareForGeneration(
    context: ModelContext,
    messages: [[String: any Sendable]],
    tools: [[String: any Sendable]]?,
    parameters: GenerateParameters,
    cachedEntry: PromptCacheEntry?
) throws -> PreparedPrompt {
    let tokenIds = try context.tokenizer.applyChatTemplate(
        messages: messages, tools: tools, additionalContext: nil)

    let kvCaches: [KVCache]
    let inputTokens: [Int]
    let reused: Bool
    if let entry = cachedEntry,
        entry.tokens.count > 0,
        tokenIds.count > entry.tokens.count,
        entry.tokens == Array(tokenIds.prefix(entry.tokens.count))
    {
        kvCaches = entry.caches
        inputTokens = Array(tokenIds.dropFirst(entry.tokens.count))
        reused = true
    } else {
        kvCaches = makePromptCache(model: context.model, parameters: parameters)
        inputTokens = tokenIds
        reused = false
    }

    let promptArray = MLXArray(inputTokens).asType(.int32)
    let mask = MLXArray.ones([promptArray.size]).asType(.bool)
    let input = LMInput(
        text: LMInput.Text(tokens: promptArray, mask: mask),
        image: nil, video: nil)

    return PreparedPrompt(
        lmInput: input, kvCaches: kvCaches,
        fullPromptTokens: tokenIds, reusedFromCache: reused)
}

/// Patch up the engine's completion info so callers see the *full* prompt token
/// count even when most of it came from a cache hit. Without this the
/// `prompt_eval_count` reported to clients only reflects the suffix that was
/// actually prefilled this call.
private func adjustInfo(
    _ info: GenerateCompletionInfo?, fullPromptTokens: Int
) -> GenerateCompletionInfo? {
    guard let info else { return nil }
    if info.promptTokenCount == fullPromptTokens { return info }
    return GenerateCompletionInfo(
        promptTokenCount: fullPromptTokens,
        generationTokenCount: info.generationTokenCount,
        promptTime: info.promptTime,
        generationTime: info.generateTime,
        stopReason: info.stopReason
    )
}

/// Trim a freshly used cache back to prompt-only state and return it to the store.
///
/// We don't keep the generated tokens in the cache because we can't observe the
/// raw token ids that came out of `MLXLMCommon.generate` — only decoded text.
/// Storing only the prompt prefix keeps `tokens.count == caches.first.offset`,
/// which is the invariant the prefix-match code relies on.
private func persistCache(
    context: PromptCacheContext?,
    entry: PromptCacheEntry
) async {
    guard let context else { return }
    if let limit = context.maxTokens, entry.tokens.count > limit { return }
    if entry.caches.isEmpty { return }
    let currentOffset = entry.caches.first?.offset ?? 0
    let extra = currentOffset - entry.tokens.count
    if extra > 0, canTrimPromptCache(entry.caches) {
        trimPromptCache(entry.caches, numTokens: extra)
    } else if extra != 0 {
        // Offset doesn't line up with prompt tokens (likely a rotating cache
        // overflowed past the prompt). Don't persist a cache we can't reason
        // about.
        return
    }
    await context.store.set(key: context.key, entry: entry)
}

// MARK: - Generation Runners

/// Runs generation via a ``ModelContainer`` and collects the full response.
public func runGeneration(
    container: ModelContainer,
    messages: [[String: any Sendable]],
    tools: [[String: any Sendable]]?,
    parameters: GenerateParameters,
    cacheContext: PromptCacheContext? = nil
) async throws -> (text: String, info: GenerateCompletionInfo?) {
    // Take the entry out of the store before entering `container.perform` so a
    // concurrent request for the same key sees a miss instead of stomping on a
    // mutable `[KVCache]` that another generation is already using. The prefix
    // match is re-verified inside the perform block once we have the tokens.
    let cachedEntry: PromptCacheEntry?
    if let cacheContext {
        cachedEntry = await cacheContext.store.take(key: cacheContext.key)
    } else {
        cachedEntry = nil
    }

    let outcome = try await container.perform { context -> GenerationOutcome in
        let prepared = try prepareForGeneration(
            context: context, messages: messages, tools: tools,
            parameters: parameters, cachedEntry: cachedEntry)

        var fullText = ""
        var completionInfo: GenerateCompletionInfo?

        let stream = try MLXLMCommon.generate(
            input: prepared.lmInput, cache: prepared.kvCaches,
            parameters: parameters, context: context)
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

        let postEntry = PromptCacheEntry(
            tokens: prepared.fullPromptTokens, caches: prepared.kvCaches)
        return GenerationOutcome(text: fullText, info: completionInfo, postEntry: postEntry)
    }

    await persistCache(context: cacheContext, entry: outcome.postEntry)
    return (
        outcome.text,
        adjustInfo(outcome.info, fullPromptTokens: outcome.postEntry.tokens.count)
    )
}

/// Runs streaming generation via a ``ModelContainer``, calling back for each chunk.
public func runStreamingGeneration(
    container: ModelContainer,
    messages: [[String: any Sendable]],
    tools: [[String: any Sendable]]?,
    parameters: GenerateParameters,
    cacheContext: PromptCacheContext? = nil,
    onChunk: @Sendable @escaping (String) -> Void,
    onComplete: @Sendable @escaping (GenerateCompletionInfo?) -> Void
) async throws {
    let cachedEntry: PromptCacheEntry?
    if let cacheContext {
        cachedEntry = await cacheContext.store.take(key: cacheContext.key)
    } else {
        cachedEntry = nil
    }

    let postEntry = try await container.perform { context -> PromptCacheEntry in
        let prepared = try prepareForGeneration(
            context: context, messages: messages, tools: tools,
            parameters: parameters, cachedEntry: cachedEntry)

        let stream = try MLXLMCommon.generate(
            input: prepared.lmInput, cache: prepared.kvCaches,
            parameters: parameters, context: context)

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                onChunk(text)
            case .info(let info):
                onComplete(adjustInfo(info, fullPromptTokens: prepared.fullPromptTokens.count))
            case .toolCall:
                break
            }
        }

        return PromptCacheEntry(
            tokens: prepared.fullPromptTokens, caches: prepared.kvCaches)
    }

    await persistCache(context: cacheContext, entry: postEntry)
}
