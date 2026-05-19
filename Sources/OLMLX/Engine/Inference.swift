import Foundation

public struct InferenceOptions: Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var maxTokens: Int?
    public var stop: [String]?
    public var seed: Int?
    public var repetitionPenalty: Double?

    public init(
        temperature: Double? = nil, topP: Double? = nil, topK: Int? = nil,
        maxTokens: Int? = nil, stop: [String]? = nil, seed: Int? = nil,
        repetitionPenalty: Double? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.maxTokens = maxTokens
        self.stop = stop
        self.seed = seed
        self.repetitionPenalty = repetitionPenalty
    }

    public static func from(_ options: ModelOptions?) -> InferenceOptions {
        guard let opts = options else { return InferenceOptions() }
        return InferenceOptions(
            temperature: opts.temperature,
            topP: opts.topP,
            topK: opts.topK,
            maxTokens: opts.numPredict,
            stop: opts.stop,
            seed: opts.seed,
            repetitionPenalty: opts.repeatPenalty
        )
    }
}

public func countChatTokens(
    messages: [Message],
    tools: [Tool]?,
    caps: TemplateCaps
) -> Int {
    let contentText = messages.map { $0.content }.joined(separator: " ")
    let toolText =
        tools?.map {
            "\($0.function["name"].flatMap { if case .string(let n) = $0 { return n } else { return nil } } ?? "")"
        }.joined(separator: " ") ?? ""
    let allText = contentText + " " + toolText
    return max(1, allText.count / 4)
}

public func buildInferenceOptions(from requestOptions: ModelOptions?) -> InferenceOptions {
    return InferenceOptions.from(requestOptions)
}

public enum InferenceError: Error, Sendable, Equatable {
    case modelNotLoaded(String)
    case generationFailed(String)
    case embeddingFailed(String)
    case tokenizationFailed(String)
    /// Speculative decoding was requested with a strategy the engine cannot run
    /// (currently anything other than `.classic`). Carries the strategy name for diagnostics.
    case unsupportedSpeculativeStrategy(String)
    /// Speculative decoding was enabled but no draft model HF path was configured.
    case speculativeDraftModelMissing
}

public struct TokenizeResult: Sendable {
    public var tokens: [Int]

    public init(tokens: [Int]) {
        self.tokens = tokens
    }
}
