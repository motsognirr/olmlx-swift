import Foundation

public struct ToolUse: Sendable {
    public var id: String
    public var name: String
    public var arguments: [String: AnyCodable]

    public init(id: String, name: String, arguments: [String: AnyCodable]) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    /// Convenience initializer for callers holding `[String: Any]` from
    /// `JSONSerialization`.
    public init(id: String, name: String, jsonArguments: [String: Any]) {
        self.id = id
        self.name = name
        self.arguments = jsonArguments.mapValues(AnyCodable.from)
    }
}

public struct ParseResult: Sendable {
    public var thinking: String?
    public var visibleText: String?
    public var toolUses: [ToolUse]

    public init(thinking: String? = nil, visibleText: String? = nil, toolUses: [ToolUse] = []) {
        self.thinking = thinking
        self.visibleText = visibleText
        self.toolUses = toolUses
    }
}

public final class ToolParser: Sendable {

    public static let shared = ToolParser()

    public func parseModelOutput(
        text: String,
        hasTools: Bool = false,
        thinkingExpected: Bool = false
    ) -> ParseResult {
        if !hasTools {
            return ParseResult(visibleText: text)
        }

        // 1. gpt-oss channels: <|start|>analysis...<|call|>{"name":...}
        if let gptResult = parseGPTOSSChannels(text: text) {
            return gptResult
        }

        // 2. Qwen format: <tool_call>...</tool_call>
        if let qwenResult = parseQwenFormat(text: text) {
            return qwenResult
        }

        // 3. Gemma 4 channel format
        if let gemmaResult = parseGemma4Channel(text: text) {
            return gemmaResult
        }

        // 4. Mistral format: [TOOL_CALLS] [...]
        if let mistralResult = parseMistralFormat(text: text) {
            return mistralResult
        }

        // 5. Llama format: <|python_tag|>{...}
        if let llamaResult = parseLlamaPythonTag(text: text) {
            return llamaResult
        }

        // 6. DeepSeek format: <|tool_calls_begin|>...<|tool_calls_end|>
        if let deepseekResult = parseDeepSeekFormat(text: text) {
            return deepseekResult
        }

        // 7. MiniMax format
        if let minimaxResult = parseMiniMaxFormat(text: text) {
            return minimaxResult
        }

        // 8. Standalone XML: <function=Name>...</function>
        if let xmlResult = parseStandaloneXML(text: text) {
            return xmlResult
        }

        // 9. Bare JSON: {"name": "...", "arguments": {...}}
        if let jsonResult = parseBareJSON(text: text) {
            return jsonResult
        }

        return ParseResult(visibleText: text)
    }

    public func resolveToolNames(_ toolUses: [ToolUse], declaredTools: [Tool]) -> [ToolUse] {
        let toolLookup = Dictionary(
            uniqueKeysWithValues: declaredTools.map {
                (
                    $0.function["name"].flatMap {
                        if case .string(let n) = $0 { return n } else { return nil }
                    } ?? "", $0
                )
            })
        // Lookup is currently unused; left in place so callers can later validate
        // that each parsed tool name matches one of the declared tools.
        _ = toolLookup
        return toolUses
    }

    public func fillMissingRequiredArgs(_ toolUses: [ToolUse], tools: [Tool]) -> [ToolUse] {
        return toolUses.map { use in
            var args = use.arguments
            if let tool = tools.first(where: {
                if case .string(let n) = $0.function["name"] ?? .string("") { return n == use.name }
                return false
            }) {
                if case .dictionary(let params) = tool.function["parameters"] ?? .dictionary([:]),
                    case .array(let required) = params["required"] ?? .array([])
                {
                    for req in required {
                        if case .string(let key) = req, args[key] == nil {
                            args[key] = .null
                        }
                    }
                }
            }
            return ToolUse(id: use.id, name: use.name, arguments: args)
        }
    }

    // MARK: - Format Parsers

    private func parseGPTOSSChannels(text: String) -> ParseResult? {
        guard text.contains("<|call|>") else { return nil }
        let callPattern = try? NSRegularExpression(
            pattern: "<\\|call\\|>\\s*(\\{[^}]+\\})",
            options: [.dotMatchesLineSeparators]
        )
        let nsText = text as NSString
        if let match = callPattern?.firstMatch(in: text, range: NSRange(location: 0, length: nsText.length)) {
            let jsonStr = nsText.substring(with: match.range(at: 1))
            if let toolUse = parseToolCallJSON(jsonStr) {
                return ParseResult(toolUses: [toolUse])
            }
        }
        return nil
    }

    private func parseQwenFormat(text: String) -> ParseResult? {
        guard text.contains("<tool_call>") else { return nil }
        let pattern = try? NSRegularExpression(
            pattern: "<tool_call>\\s*(.+?)\\s*</tool_call>",
            options: [.dotMatchesLineSeparators]
        )
        let nsText = text as NSString
        if let match = pattern?.firstMatch(in: text, range: NSRange(location: 0, length: nsText.length)) {
            let content = nsText.substring(with: match.range(at: 1))
            if let toolUse = parseToolCallJSON(content) {
                let before = nsText.substring(to: match.range.location)
                return ParseResult(
                    visibleText: before.trimmingCharacters(in: .whitespacesAndNewlines), toolUses: [toolUse])
            }
        }
        return nil
    }

    private func parseGemma4Channel(text: String) -> ParseResult? {
        guard text.contains("<|channel|>") else { return nil }
        let pattern = try? NSRegularExpression(
            pattern: "<\\|call\\|>\\s*(\\{[^}]+\\})",
            options: [.dotMatchesLineSeparators]
        )
        let nsText = text as NSString
        if let match = pattern?.firstMatch(in: text, range: NSRange(location: 0, length: nsText.length)) {
            let jsonStr = nsText.substring(with: match.range(at: 1))
            if let toolUse = parseToolCallJSON(jsonStr) {
                return ParseResult(toolUses: [toolUse])
            }
        }
        return nil
    }

    private func parseMistralFormat(text: String) -> ParseResult? {
        guard text.contains("[TOOL_CALLS]") else { return nil }
        let pattern = try? NSRegularExpression(
            pattern: "\\[TOOL_CALLS\\]\\s*(\\[[\\s\\S]*?\\])",
            options: [.dotMatchesLineSeparators]
        )
        let nsText = text as NSString
        if let match = pattern?.firstMatch(in: text, range: NSRange(location: 0, length: nsText.length)) {
            let jsonStr = nsText.substring(with: match.range(at: 1))
            if let data = jsonStr.data(using: .utf8),
                let calls = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
            {
                let toolUses = calls.compactMap { call -> ToolUse? in
                    guard let name = call["name"] as? String,
                        let args = call["arguments"] as? [String: Any]
                    else { return nil }
                    return ToolUse(id: generateToolUseID(), name: name, jsonArguments: args)
                }
                if !toolUses.isEmpty {
                    return ParseResult(toolUses: toolUses)
                }
            }
        }
        return nil
    }

    private func parseLlamaPythonTag(text: String) -> ParseResult? {
        guard text.contains("<|python_tag|>") else { return nil }
        let pattern = try? NSRegularExpression(
            pattern: "<\\|python_tag\\|>\\s*(\\{[\\s\\S]*?\\})",
            options: [.dotMatchesLineSeparators]
        )
        let nsText = text as NSString
        if let match = pattern?.firstMatch(in: text, range: NSRange(location: 0, length: nsText.length)) {
            let jsonStr = nsText.substring(with: match.range(at: 1))
            if let data = jsonStr.data(using: .utf8),
                let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let name = dict["name"] as? String
            {
                let args = dict["arguments"] as? [String: Any] ?? [:]
                let toolUse = ToolUse(id: generateToolUseID(), name: name, jsonArguments: args)
                return ParseResult(toolUses: [toolUse])
            }
        }
        return nil
    }

    private func parseDeepSeekFormat(text: String) -> ParseResult? {
        guard text.contains("<|tool_calls_begin|>") else { return nil }
        let pattern = try? NSRegularExpression(
            pattern: "<\\|tool_calls_begin\\|>([\\s\\S]*?)<\\|tool_calls_end\\|>",
            options: [.dotMatchesLineSeparators]
        )
        let nsText = text as NSString
        if let match = pattern?.firstMatch(in: text, range: NSRange(location: 0, length: nsText.length)) {
            let content = nsText.substring(with: match.range(at: 1))
            if let data = content.data(using: .utf8),
                let calls = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
            {
                let toolUses = calls.compactMap { call -> ToolUse? in
                    guard let name = call["name"] as? String,
                        let args = call["arguments"] as? [String: Any]
                    else { return nil }
                    return ToolUse(id: generateToolUseID(), name: name, jsonArguments: args)
                }
                if !toolUses.isEmpty {
                    return ParseResult(toolUses: toolUses)
                }
            }
        }
        return nil
    }

    private func parseMiniMaxFormat(text: String) -> ParseResult? {
        guard text.contains("<minimax:tool_call>") else { return nil }
        let pattern = try? NSRegularExpression(
            pattern: "<minimax:tool_call>\\s*(.+?)\\s*</minimax:tool_call>",
            options: [.dotMatchesLineSeparators]
        )
        let nsText = text as NSString
        if let match = pattern?.firstMatch(in: text, range: NSRange(location: 0, length: nsText.length)) {
            let content = nsText.substring(with: match.range(at: 1))
            if let toolUse = parseToolCallJSON(content) {
                return ParseResult(toolUses: [toolUse])
            }
        }
        return nil
    }

    private func parseStandaloneXML(text: String) -> ParseResult? {
        let pattern = try? NSRegularExpression(
            pattern: "<function=(\\w+)>([\\s\\S]*?)</function>",
            options: [.dotMatchesLineSeparators]
        )
        let nsText = text as NSString
        if let match = pattern?.firstMatch(in: text, range: NSRange(location: 0, length: nsText.length)) {
            let name = nsText.substring(with: match.range(at: 1))
            let argsStr = nsText.substring(with: match.range(at: 2)).trimmingCharacters(in: .whitespaces)
            var args: [String: Any] = [:]
            if let data = argsStr.data(using: .utf8),
                let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            {
                args = dict
            }
            return ParseResult(toolUses: [ToolUse(id: generateToolUseID(), name: name, jsonArguments: args)])
        }
        return nil
    }

    private func parseBareJSON(text: String) -> ParseResult? {
        guard text.contains(#""name""#) && (text.contains(#""arguments""#) || text.contains(#""parameters""#)) else {
            return nil
        }
        let pattern = try? NSRegularExpression(
            pattern:
                "\\{[^{}]*\"name\"\\s*:\\s*\"(\\w+)\"[^{}]*\"(?:arguments|parameters)\"\\s*:\\s*(\\{[^}]+\\})[^{}]*\\}",
            options: [.dotMatchesLineSeparators]
        )
        let nsText = text as NSString
        if let match = pattern?.firstMatch(in: text, range: NSRange(location: 0, length: nsText.length)) {
            let name = nsText.substring(with: match.range(at: 1))
            let argsStr = nsText.substring(with: match.range(at: 2))
            var args: [String: Any] = [:]
            if let data = argsStr.data(using: .utf8),
                let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            {
                args = dict
            }
            return ParseResult(toolUses: [ToolUse(id: generateToolUseID(), name: name, jsonArguments: args)])
        }
        return nil
    }

    // MARK: - Helpers

    public func generateToolUseID() -> String {
        return "toolu_" + UUID().uuidString.prefix(8).lowercased()
    }

    private func parseToolCallJSON(_ jsonStr: String) -> ToolUse? {
        guard let data = jsonStr.data(using: .utf8),
            let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let name = dict["name"] as? String
        else { return nil }
        let args = dict["arguments"] as? [String: Any] ?? [:]
        return ToolUse(id: generateToolUseID(), name: name, jsonArguments: args)
    }
}
