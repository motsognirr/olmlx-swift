import Foundation

public struct TemplateCaps: Sendable {
    public var supportsTools: Bool = false
    public var supportsEnableThinking: Bool = false
    public var hasThinkingTags: Bool = false
    public var hasChannelFormat: Bool = false
    public var usesToolResponses: Bool = false

    public init() {}
}

public func detectCaps(tokenizerChatTemplate: String?) -> TemplateCaps {
    var caps = TemplateCaps()

    guard let template = tokenizerChatTemplate else { return caps }

    caps.supportsTools = template.contains("tools") || template.contains("tool")
    caps.supportsEnableThinking = template.contains("enable_thinking") || template.contains("enableThinking")
    caps.hasThinkingTags = template.contains("think") || template.contains("thinking")
    caps.hasChannelFormat = template.contains("channel")
    caps.usesToolResponses = template.contains("tool_responses") || template.contains("tool_calls")

    return caps
}

public protocol TokenizerProtocol: Sendable {
    var chatTemplate: String? { get }
    func applyChatTemplate(messages: [[String: String]], tools: [Tool]?, addGenerationPrompt: Bool) throws -> String
    func encode(text: String) throws -> [Int]
    func decode(tokens: [Int]) throws -> String
}
