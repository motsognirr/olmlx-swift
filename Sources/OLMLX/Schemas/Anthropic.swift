import Foundation

public struct AnthropicToolInputSchema: Codable, Sendable {
    public var type: String = "object"
    public var properties: [String: AnyCodable]?
    public var required: [String]?
}

public struct AnthropicTool: Codable, Sendable {
    public var name: String
    public var description: String?
    public var inputSchema: AnthropicToolInputSchema

    private enum CodingKeys: String, CodingKey {
        case name, description
        case inputSchema = "input_schema"
    }
}

public struct AnthropicContentBlock: Codable, Sendable {
    public var type: String = "text"
    public var text: String?
    public var thinking: String?
    public var signature: String?
    public var id: String?
    public var name: String?
    public var input: [String: AnyCodable]?
    public var toolUseId: String?
    public var content: AnthropicContentValue?
    public var isError: Bool?

    private enum CodingKeys: String, CodingKey {
        case type, text, thinking, signature, id, name, input
        case toolUseId = "tool_use_id"
        case content
        case isError = "is_error"
    }
}

public enum AnthropicContentValue: Codable, Sendable {
    case string(String)
    case array([AnyCodable])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let a = try? container.decode([AnyCodable].self) {
            self = .array(a)
        } else {
            throw DecodingError.typeMismatch(
                AnthropicContentValue.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected String or array"
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

public struct AnthropicMessage: Codable, Sendable {
    public var role: String
    public var content: AnthropicMessageContent

    public init(role: String, content: AnthropicMessageContent) throws {
        switch content {
        case .string(let s):
            guard s.count <= 1_000_000 else {
                throw SchemaValidationError("content length exceeds limit")
            }
        case .blocks(let b):
            guard b.count <= 1_000 else {
                throw SchemaValidationError("content block count exceeds limit")
            }
        }
        self.role = role
        self.content = content
    }
}

public enum AnthropicMessageContent: Codable, Sendable {
    case string(String)
    case blocks([AnthropicContentBlock])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let b = try? container.decode([AnthropicContentBlock].self) {
            self = .blocks(b)
        } else {
            throw DecodingError.typeMismatch(
                AnthropicMessageContent.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected String or [ContentBlock]"
                ))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .blocks(let b): try container.encode(b)
        }
    }
}

public struct AnthropicThinkingParam: Codable, Sendable {
    public var type: String
    public var budgetTokens: Int?

    private enum CodingKeys: String, CodingKey {
        case type
        case budgetTokens = "budget_tokens"
    }
}

public struct AnthropicMessagesRequest: Codable, Sendable {
    public var model: String
    public var messages: [AnthropicMessage]
    public var maxTokens: Int = 4096
    public var stream: Bool = false
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var stopSequences: [String]?
    public var system: AnthropicSystemValue?
    public var tools: [AnthropicTool]?
    public var toolChoice: [String: AnyCodable]?
    public var thinking: AnthropicThinkingParam?
    public var metadata: [String: AnyCodable]?

    private enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, tools, thinking, metadata
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case stopSequences = "stop_sequences"
        case system
        case toolChoice = "tool_choice"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        messages = try container.decode([AnthropicMessage].self, forKey: .messages)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 4096
        stream = try container.decodeIfPresent(Bool.self, forKey: .stream) ?? false
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        stopSequences = try container.decodeIfPresent([String].self, forKey: .stopSequences)
        system = try container.decodeIfPresent(AnthropicSystemValue.self, forKey: .system)
        tools = try container.decodeIfPresent([AnthropicTool].self, forKey: .tools)
        toolChoice = try container.decodeIfPresent([String: AnyCodable].self, forKey: .toolChoice)
        thinking = try container.decodeIfPresent(AnthropicThinkingParam.self, forKey: .thinking)
        metadata = try container.decodeIfPresent([String: AnyCodable].self, forKey: .metadata)
    }
}

public enum AnthropicSystemValue: Codable, Sendable {
    case string(String)
    case blocks([AnthropicContentBlock])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let b = try? container.decode([AnthropicContentBlock].self) {
            self = .blocks(b)
        } else {
            throw DecodingError.typeMismatch(
                AnthropicSystemValue.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected String or [ContentBlock]"
                ))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .blocks(let b): try container.encode(b)
        }
    }
}

public struct AnthropicUsage: Codable, Sendable {
    public var inputTokens: Int = 0
    public var outputTokens: Int = 0
    public var cacheCreationInputTokens: Int = 0
    public var cacheReadInputTokens: Int = 0

    private enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case cacheCreationInputTokens = "cache_creation_input_tokens"
        case cacheReadInputTokens = "cache_read_input_tokens"
    }
}

public struct AnthropicTokenCountResponse: Codable, Sendable {
    public var inputTokens: Int

    private enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
    }
}

public struct AnthropicMessagesResponse: Codable, Sendable {
    public var id: String
    public var type: String = "message"
    public var role: String = "assistant"
    public var content: [AnthropicContentBlock]
    public var model: String
    public var stopReason: String?
    public var stopSequence: String?
    public var usage: AnthropicUsage

    private enum CodingKeys: String, CodingKey {
        case id, type, role, content, model, usage
        case stopReason = "stop_reason"
        case stopSequence = "stop_sequence"
    }
}
