import Foundation

public struct GenerateRequest: Codable, Sendable {
    public var model: String
    public var prompt: String
    public var suffix: String?
    public var images: [String]?
    public var system: String?
    public var template: String?
    public var context: [Int]?
    public var stream: Bool = true
    public var raw: Bool = false
    public var format: String?
    public var options: ModelOptions?
    public var keepAlive: String?

    private enum CodingKeys: String, CodingKey {
        case model, prompt, suffix, images, system, template, context
        case stream, raw, format, options
        case keepAlive = "keep_alive"
    }

    public init(
        model: String, prompt: String, suffix: String? = nil,
        images: [String]? = nil, system: String? = nil,
        template: String? = nil, context: [Int]? = nil,
        stream: Bool = true, raw: Bool = false,
        format: String? = nil, options: ModelOptions? = nil,
        keepAlive: String? = nil
    ) throws {
        try validateNonEmptyTextInput(prompt, fieldName: "prompt")
        self.model = model
        self.prompt = prompt
        self.suffix = suffix
        self.images = images
        self.system = system
        self.template = template
        self.context = context
        self.stream = stream
        self.raw = raw
        self.format = format
        self.options = options
        self.keepAlive = keepAlive
    }
}

public struct GenerateResponse: Codable, Sendable {
    public var model: String
    public var createdAt: String
    public var response: String
    public var done: Bool
    public var doneReason: String?
    public var context: [Int]?
    public var totalDuration: Int?
    public var loadDuration: Int?
    public var promptEvalCount: Int?
    public var promptEvalDuration: Int?
    public var evalCount: Int?
    public var evalDuration: Int?

    private enum CodingKeys: String, CodingKey {
        case model, response, done, context
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
