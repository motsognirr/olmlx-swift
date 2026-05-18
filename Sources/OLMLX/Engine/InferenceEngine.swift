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

// MARK: - Generation Runners

/// Runs generation via a ``ModelContainer`` and collects the full response.
public func runGeneration(
    container: ModelContainer,
    messages: [[String: any Sendable]],
    tools: [[String: any Sendable]]?,
    parameters: GenerateParameters
) async throws -> (text: String, info: GenerateCompletionInfo?) {
    return try await container.perform { context -> (String, GenerateCompletionInfo?) in
        let tokenIds = try context.tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil)

        let promptArray = MLXArray(tokenIds).asType(.int32)
        let mask = MLXArray.ones([promptArray.size]).asType(.bool)
        let input = LMInput(
            text: LMInput.Text(tokens: promptArray, mask: mask),
            image: nil, video: nil)

        var fullText = ""
        var completionInfo: GenerateCompletionInfo?

        let stream = try MLXLMCommon.generate(
            input: input, parameters: parameters, context: context)
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

        return (fullText, completionInfo)
    }
}

/// Runs streaming generation via a ``ModelContainer``, calling back for each chunk.
public func runStreamingGeneration(
    container: ModelContainer,
    messages: [[String: any Sendable]],
    tools: [[String: any Sendable]]?,
    parameters: GenerateParameters,
    onChunk: @Sendable @escaping (String) -> Void,
    onComplete: @Sendable @escaping (GenerateCompletionInfo?) -> Void
) async throws {
    try await container.perform { context in
        let tokenIds = try context.tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil)

        let promptArray = MLXArray(tokenIds).asType(.int32)
        let mask = MLXArray.ones([promptArray.size]).asType(.bool)
        let input = LMInput(
            text: LMInput.Text(tokens: promptArray, mask: mask),
            image: nil, video: nil)

        let stream = try MLXLMCommon.generate(
            input: input, parameters: parameters, context: context)

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
    }
}
