import Foundation

public struct ToolCallFunction: Codable, Sendable {
    public var name: String
    public var arguments: [String: AnyCodable]
}

public struct ToolCall: Codable, Sendable {
    public var function: ToolCallFunction
}

public struct Message: Codable, Sendable {
    public var role: String
    public var content: String = ""
    public var thinking: String?
    public var images: [String]?
    public var toolCalls: [ToolCall]?

    private enum CodingKeys: String, CodingKey {
        case role, content, thinking, images
        case toolCalls = "tool_calls"
    }

    public init(
        role: String, content: String = "", thinking: String? = nil,
        images: [String]? = nil, toolCalls: [ToolCall]? = nil
    ) {
        self.role = role
        self.content = content
        self.thinking = thinking
        self.images = images
        self.toolCalls = toolCalls
    }
}

public struct Tool: Codable, Sendable {
    public var type: String = "function"
    public var function: [String: AnyCodable]

    public init(type: String = "function", function: [String: AnyCodable]) {
        self.type = type
        self.function = function
    }
}

public struct ChatRequest: Codable, Sendable {
    public var model: String
    public var messages: [Message]
    public var tools: [Tool]?
    public var format: String?
    public var stream: Bool = true
    public var options: ModelOptions?
    public var keepAlive: String?

    private enum CodingKeys: String, CodingKey {
        case model, messages, tools, format, stream, options
        case keepAlive = "keep_alive"
    }

    public init(
        model: String, messages: [Message], tools: [Tool]? = nil,
        format: String? = nil, stream: Bool = true,
        options: ModelOptions? = nil, keepAlive: String? = nil
    ) throws {
        guard !messages.isEmpty else {
            throw SchemaValidationError("messages cannot be empty")
        }
        self.model = model
        self.messages = messages
        self.tools = tools
        self.format = format
        self.stream = stream
        self.options = options
        self.keepAlive = keepAlive
    }
}

public struct ChatResponse: Codable, Sendable {
    public var model: String
    public var createdAt: String
    public var message: Message
    public var done: Bool
    public var doneReason: String?
    public var totalDuration: Int?
    public var loadDuration: Int?
    public var promptEvalCount: Int?
    public var promptEvalDuration: Int?
    public var evalCount: Int?
    public var evalDuration: Int?

    private enum CodingKeys: String, CodingKey {
        case model, message, done
        case createdAt = "created_at"
        case doneReason = "done_reason"
        case totalDuration = "total_duration"
        case loadDuration = "load_duration"
        case promptEvalCount = "prompt_eval_count"
        case promptEvalDuration = "prompt_eval_duration"
        case evalCount = "eval_count"
        case evalDuration = "eval_duration"
    }
}
