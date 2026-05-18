import Foundation
import Testing

@testable import OLMLX

@Suite("Tool Parser")
struct ToolParserTests {
    @Test func bareJSON() {
        let text = #"{"name": "get_weather", "arguments": {"city": "Paris"}}"#
        let result = ToolParser.shared.parseModelOutput(text: text, hasTools: true)
        #expect(result.toolUses.count > 0)
    }

    @Test func qwenFormat() {
        let text = #"Some text <tool_call>{"name":"search","arguments":{"query":"test"}}</tool_call>"#
        let result = ToolParser.shared.parseModelOutput(text: text, hasTools: true)
        #expect(result.toolUses.count > 0)
    }

    @Test func noTools() {
        let result = ToolParser.shared.parseModelOutput(text: "Hello, how are you?", hasTools: false)
        #expect(result.toolUses.count == 0)
        #expect(result.visibleText == "Hello, how are you?")
    }

    @Test func mistralFormat() {
        let text = #"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Berlin"}}]"#
        let result = ToolParser.shared.parseModelOutput(text: text, hasTools: true)
        #expect(result.toolUses.count > 0)
    }

    @Test func deepSeekFormat() {
        let text = #"<|tool_calls_begin|>[{"name": "search", "arguments": {"query": "hello"}}]<|tool_calls_end|>"#
        let result = ToolParser.shared.parseModelOutput(text: text, hasTools: true)
        #expect(result.toolUses.count > 0)
    }

    @Test func llamaPythonTag() {
        let text = #"<|python_tag|>{"name": "calculator", "arguments": {"expr": "2+2"}}"#
        let result = ToolParser.shared.parseModelOutput(text: text, hasTools: true)
        #expect(result.toolUses.count > 0)
    }

    @Test func xmlFormat() {
        let text = #"<function=get_weather>{"city": "London"}</function>"#
        let result = ToolParser.shared.parseModelOutput(text: text, hasTools: true)
        #expect(result.toolUses.count > 0)
    }

    @Test func miniMaxFormat() {
        let text = #"<minimax:tool_call>{"name":"search","arguments":{"query":"test"}}</minimax:tool_call>"#
        let result = ToolParser.shared.parseModelOutput(text: text, hasTools: true)
        #expect(result.toolUses.count > 0)
    }

    @Test func noMatchReturnsVisible() {
        let text = "Just some regular text without any tool calls"
        let result = ToolParser.shared.parseModelOutput(text: text, hasTools: true)
        #expect(result.toolUses.count == 0)
        #expect(result.visibleText == text)
    }

    @Test func generateID() {
        let id = ToolParser.shared.generateToolUseID()
        #expect(id.hasPrefix("toolu_"))
    }
}

@Suite("Template Caps")
struct TemplateCapsTests {
    @Test func detectTools() {
        let template = "{% for message in messages %}{% if message.tools %}..."
        let caps = detectCaps(tokenizerChatTemplate: template)
        #expect(caps.supportsTools)
    }

    @Test func detectThinking() {
        let template = "{% if enable_thinking %}...<think>...</think>{% endif %}"
        let caps = detectCaps(tokenizerChatTemplate: template)
        #expect(caps.supportsEnableThinking)
        #expect(caps.hasThinkingTags)
    }

    @Test func nilTemplate() {
        let caps = detectCaps(tokenizerChatTemplate: nil)
        #expect(caps.supportsTools == false)
        #expect(caps.hasThinkingTags == false)
    }
}

@Suite("Inference Options")
struct InferenceOptionsTests {
    @Test func fromEmptyModelOptions() {
        let infOpts = InferenceOptions.from(ModelOptions())
        #expect(infOpts.temperature == nil)
        #expect(infOpts.maxTokens == nil)
    }

    @Test func withValues() {
        var opts = ModelOptions()
        opts.temperature = 0.7
        opts.topP = 0.9
        opts.seed = 42
        let infOpts = InferenceOptions.from(opts)
        #expect(infOpts.temperature == 0.7)
        #expect(infOpts.topP == 0.9)
        #expect(infOpts.seed == 42)
    }
}

@Suite("Chat Token Counting")
struct ChatTokenTests {
    @Test func countChatTokensEmpty() {
        let tokens = countChatTokens(messages: [], tools: nil, caps: TemplateCaps())
        #expect(tokens >= 1)
    }
}

@Suite("Keep Alive Parsing")
struct KeepAliveTests {
    @Test func seconds() { #expect(parseKeepAlive("30s") == 30) }
    @Test func minutes() { #expect(parseKeepAlive("5m") == 300) }
    @Test func hours() { #expect(parseKeepAlive("1h") == 3600) }
    @Test func zero() { #expect(parseKeepAlive("0s") == 0) }
}

@Suite("Prompt Cache")
struct PromptCacheTests {
    @Test func setAndGet() {
        let cache = PromptCacheStore(maxSlots: 4)
        cache.set(key: "test", state: CachedPromptState(tokens: [1, 2, 3]))
        let retrieved = cache.get(key: "test")
        #expect(retrieved != nil)
        #expect(retrieved?.tokens == [1, 2, 3])
    }

    @Test func lruEviction() {
        let cache = PromptCacheStore(maxSlots: 2)
        cache.set(key: "a", state: CachedPromptState(tokens: [1]))
        cache.set(key: "b", state: CachedPromptState(tokens: [2]))
        cache.set(key: "c", state: CachedPromptState(tokens: [3]))
        #expect(cache.get(key: "a") == nil)
        #expect(cache.get(key: "b") != nil)
        #expect(cache.get(key: "c") != nil)
    }
}
