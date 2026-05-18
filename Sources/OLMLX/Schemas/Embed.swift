import Foundation

public struct EmbedRequest: Codable, Sendable {
    public var model: String
    public var input: EmbedInput
    public var truncate: Bool = true
    public var options: [String: AnyCodable]?
    public var keepAlive: String?

    private enum CodingKeys: String, CodingKey {
        case model, input, truncate, options
        case keepAlive = "keep_alive"
    }

    public init(model: String, input: EmbedInput, truncate: Bool = true,
                options: [String: AnyCodable]? = nil, keepAlive: String? = nil) throws {
        switch input {
        case .string(let s):
            try validateNonEmptyTextInput(s, fieldName: "input")
        case .array(let arr):
            if arr.isEmpty { throw SchemaValidationError("input cannot be an empty list") }
            for s in arr where s.isEmpty { throw SchemaValidationError("input cannot contain empty strings") }
        }
        self.model = model
        self.input = input
        self.truncate = truncate
        self.options = options
        self.keepAlive = keepAlive
    }
}

public enum EmbedInput: Codable, Sendable {
    case string(String)
    case array([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let a = try? container.decode([String].self) {
            self = .array(a)
        } else {
            throw DecodingError.typeMismatch(EmbedInput.self, DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "Expected String or [String]"
            ))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .array(let a): try container.encode(a)
        }
    }
}

public struct EmbedResponse: Codable, Sendable {
    public var model: String
    public var embeddings: [[Double]]
    public var totalDuration: Int?
    public var loadDuration: Int?
    public var promptEvalCount: Int?

    private enum CodingKeys: String, CodingKey {
        case model, embeddings
        case totalDuration = "total_duration"
        case loadDuration = "load_duration"
        case promptEvalCount = "prompt_eval_count"
    }
}

public struct EmbeddingsRequest: Codable, Sendable {
    public var model: String
    public var prompt: String
    public var options: [String: AnyCodable]?
    public var keepAlive: String?

    private enum CodingKeys: String, CodingKey {
        case model, prompt, options
        case keepAlive = "keep_alive"
    }

    public init(model: String, prompt: String, options: [String: AnyCodable]? = nil,
                keepAlive: String? = nil) throws {
        try validateNonEmptyTextInput(prompt, fieldName: "prompt")
        self.model = model
        self.prompt = prompt
        self.options = options
        self.keepAlive = keepAlive
    }
}

public struct EmbeddingsResponse: Codable, Sendable {
    public var embedding: [Double]
}
