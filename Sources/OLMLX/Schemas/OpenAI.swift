import Foundation

public struct OpenAIChatMessage: Codable, Sendable {
    public var role: String
    public var content: String?
    public var name: String?
    public var toolCalls: [[String: AnyCodable]]?
    public var toolCallId: String?

    private enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
    }
}

public struct ResponseFormat: Codable, Sendable {
    public var type: String = "text"
    public var jsonSchema: [String: AnyCodable]?

    private enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
    }

    public init(type: String = "text", jsonSchema: [String: AnyCodable]? = nil) throws {
        if type == "json_schema" {
            guard let schema = jsonSchema else {
                throw SchemaValidationError("json_schema is required when type is 'json_schema'")
            }
            guard let name = schema["name"], case .string(let n) = name, !n.isEmpty else {
                throw SchemaValidationError("json_schema.name is required and must be non-empty")
            }
            guard schema["schema"] != nil else {
                throw SchemaValidationError("json_schema.schema is required")
            }
        }
        self.type = type
        self.jsonSchema = jsonSchema
    }
}

public struct OpenAIChatRequest: Codable, Sendable {
    public var model: String
    public var messages: [OpenAIChatMessage]
    public var temperature: Double?
    public var topP: Double?
    public var n: Int = 1
    public var stream: Bool = false
    public var stop: OpenAIStop?
    public var maxTokens: Int?
    public var maxCompletionTokens: Int?
    public var presencePenalty: Double?
    public var frequencyPenalty: Double?
    public var tools: [[String: AnyCodable]]?
    public var toolChoice: ToolChoiceValue?
    public var seed: Int?
    public var responseFormat: ResponseFormat?

    private enum CodingKeys: String, CodingKey {
        case model, messages, temperature, n, stream, stop, tools, seed
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case toolChoice = "tool_choice"
        case responseFormat = "response_format"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        messages = try container.decode([OpenAIChatMessage].self, forKey: .messages)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        n = try container.decodeIfPresent(Int.self, forKey: .n) ?? 1
        stream = try container.decodeIfPresent(Bool.self, forKey: .stream) ?? false
        stop = try container.decodeIfPresent(OpenAIStop.self, forKey: .stop)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        maxCompletionTokens = try container.decodeIfPresent(Int.self, forKey: .maxCompletionTokens)
        presencePenalty = try container.decodeIfPresent(Double.self, forKey: .presencePenalty)
        frequencyPenalty = try container.decodeIfPresent(Double.self, forKey: .frequencyPenalty)
        tools = try container.decodeIfPresent([[String: AnyCodable]].self, forKey: .tools)
        toolChoice = try container.decodeIfPresent(ToolChoiceValue.self, forKey: .toolChoice)
        seed = try container.decodeIfPresent(Int.self, forKey: .seed)
        responseFormat = try container.decodeIfPresent(ResponseFormat.self, forKey: .responseFormat)
    }
}

public enum OpenAIStop: Codable, Sendable {
    case string(String)
    case array([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let a = try? container.decode([String].self) {
            self = .array(a)
        } else {
            throw DecodingError.typeMismatch(OpenAIStop.self, DecodingError.Context(
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

public enum ToolChoiceValue: Codable, Sendable {
    case string(String)
    case dictionary([String: AnyCodable])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let d = try? container.decode([String: AnyCodable].self) {
            self = .dictionary(d)
        } else {
            throw DecodingError.typeMismatch(ToolChoiceValue.self, DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "Expected String or object"
            ))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .dictionary(let d): try container.encode(d)
        }
    }
}

public struct OpenAIUsage: Codable, Sendable {
    public var promptTokens: Int = 0
    public var completionTokens: Int = 0
    public var totalTokens: Int = 0

    private enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

public struct OpenAIChoice: Codable, Sendable {
    public var index: Int = 0
    public var message: OpenAIChatMessage?
    public var delta: OpenAIChatMessage?
    public var finishReason: String?

    private enum CodingKeys: String, CodingKey {
        case index, message, delta
        case finishReason = "finish_reason"
    }
}

public struct OpenAIChatResponse: Codable, Sendable {
    public var id: String
    public var object: String = "chat.completion"
    public var created: Int
    public var model: String
    public var choices: [OpenAIChoice]
    public var usage: OpenAIUsage?
}

public struct OpenAICompletionRequest: Codable, Sendable {
    public var model: String
    public var prompt: OpenAICompletionPrompt
    public var temperature: Double?
    public var topP: Double?
    public var n: Int = 1
    public var stream: Bool = false
    public var stop: OpenAIStop?
    public var maxTokens: Int?
    public var presencePenalty: Double?
    public var frequencyPenalty: Double?
    public var seed: Int?

    private enum CodingKeys: String, CodingKey {
        case model, prompt, temperature, n, stream, stop, seed
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
    }
}

public enum OpenAICompletionPrompt: Codable, Sendable {
    case string(String)
    case array([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let a = try? container.decode([String].self) {
            self = .array(a)
        } else {
            throw DecodingError.typeMismatch(OpenAICompletionPrompt.self, DecodingError.Context(
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

public struct OpenAICompletionChoice: Codable, Sendable {
    public var index: Int = 0
    public var text: String = ""
    public var finishReason: String?

    private enum CodingKeys: String, CodingKey {
        case index, text
        case finishReason = "finish_reason"
    }
}

public struct OpenAICompletionResponse: Codable, Sendable {
    public var id: String
    public var object: String = "text_completion"
    public var created: Int
    public var model: String
    public var choices: [OpenAICompletionChoice]
    public var usage: OpenAIUsage?
}

public struct OpenAIModel: Codable, Sendable {
    public var id: String
    public var object: String = "model"
    public var created: Int = 0
    public var ownedBy: String = "olmlx"

    private enum CodingKeys: String, CodingKey {
        case id, object, created
        case ownedBy = "owned_by"
    }
}

public struct OpenAIModelList: Codable, Sendable {
    public var object: String = "list"
    public var data: [OpenAIModel]
}

public struct OpenAIEmbeddingRequest: Codable, Sendable {
    public var model: String
    public var input: EmbedInput
    public var encodingFormat: String = "float"

    private enum CodingKeys: String, CodingKey {
        case model, input
        case encodingFormat = "encoding_format"
    }
}

public struct OpenAIEmbeddingData: Codable, Sendable {
    public var object: String = "embedding"
    public var index: Int = 0
    public var embedding: [Double]
}

public struct OpenAIEmbeddingResponse: Codable, Sendable {
    public var object: String = "list"
    public var data: [OpenAIEmbeddingData]
    public var model: String
    public var usage: OpenAIUsage = OpenAIUsage()
}
