import Foundation
import OLMLX
import OLMLXTestUtils

func OLMLXTests_toolParser_bareJSON() throws {
    let parser = ToolParser.shared
    let text = #"{"name": "get_weather", "arguments": {"city": "Paris"}}"#
    let result = parser.parseModelOutput(text: text, hasTools: true)
    try expect(result.toolUses.count > 0, "should find bare JSON tool call")
}

func OLMLXTests_toolParser_qwenFormat() throws {
    let parser = ToolParser.shared
    let text = #"Some text <tool_call>{"name":"search","arguments":{"query":"test"}}</tool_call>"#
    let result = parser.parseModelOutput(text: text, hasTools: true)
    try expect(result.toolUses.count > 0, "should find Qwen tool call")
}

func OLMLXTests_toolParser_noTools() throws {
    let parser = ToolParser.shared
    let text = "Hello, how are you?"
    let result = parser.parseModelOutput(text: text, hasTools: false)
    try expectEqual(result.toolUses.count, 0)
    try expectEqual(result.visibleText, "Hello, how are you?")
}

func OLMLXTests_toolParser_mistralFormat() throws {
    let parser = ToolParser.shared
    let text = #"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Berlin"}}]"#
    let result = parser.parseModelOutput(text: text, hasTools: true)
    try expect(result.toolUses.count > 0, "should find Mistral tool call")
}

func OLMLXTests_toolParser_deepSeekFormat() throws {
    let parser = ToolParser.shared
    let text = #"<|tool_calls_begin|>[{"name": "search", "arguments": {"query": "hello"}}]<|tool_calls_end|>"#
    let result = parser.parseModelOutput(text: text, hasTools: true)
    try expect(result.toolUses.count > 0, "should find DeepSeek tool call")
}

func OLMLXTests_toolParser_llamaPythonTag() throws {
    let parser = ToolParser.shared
    let text = #"<|python_tag|>{"name": "calculator", "arguments": {"expr": "2+2"}}"#
    let result = parser.parseModelOutput(text: text, hasTools: true)
    try expect(result.toolUses.count > 0, "should find Llama python_tag tool call")
}

func OLMLXTests_toolParser_xmlFormat() throws {
    let parser = ToolParser.shared
    let text = #"<function=get_weather>{"city": "London"}</function>"#
    let result = parser.parseModelOutput(text: text, hasTools: true)
    try expect(result.toolUses.count > 0, "should find XML function format")
}

func OLMLXTests_toolParser_miniMaxFormat() throws {
    let parser = ToolParser.shared
    let text = #"<minimax:tool_call>{"name":"search","arguments":{"query":"test"}}</minimax:tool_call>"#
    let result = parser.parseModelOutput(text: text, hasTools: true)
    try expect(result.toolUses.count > 0, "should find MiniMax tool call")
}

func OLMLXTests_toolParser_noMatchReturnsVisible() throws {
    let parser = ToolParser.shared
    let text = "Just some regular text without any tool calls"
    let result = parser.parseModelOutput(text: text, hasTools: true)
    try expectEqual(result.toolUses.count, 0)
    try expectEqual(result.visibleText, text)
}

func OLMLXTests_toolParser_generateID() throws {
    let id = ToolParser.shared.generateToolUseID()
    try expect(id.hasPrefix("toolu_"), "ID should start with toolu_")
}

func OLMLXTests_templateCaps_detectTools() throws {
    let template = "{% for message in messages %}{% if message.tools %}..."
    let caps = detectCaps(tokenizerChatTemplate: template)
    try expect(caps.supportsTools, "should detect tools support")
}

func OLMLXTests_templateCaps_detectThinking() throws {
    let template = "{% if enable_thinking %}...<think>...</think>{% endif %}"
    let caps = detectCaps(tokenizerChatTemplate: template)
    try expect(caps.supportsEnableThinking, "should detect thinking support")
    try expect(caps.hasThinkingTags, "should detect thinking tags")
}

func OLMLXTests_templateCaps_nilTemplate() throws {
    let caps = detectCaps(tokenizerChatTemplate: nil)
    try expectEqual(caps.supportsTools, false)
    try expectEqual(caps.hasThinkingTags, false)
}

func OLMLXTests_inferenceOptions_fromModelOptions() throws {
    let opts = ModelOptions()
    let infOpts = InferenceOptions.from(opts)
    try expectNil(infOpts.temperature)
    try expectNil(infOpts.maxTokens)
}

func OLMLXTests_inferenceOptions_withValues() throws {
    var opts = ModelOptions()
    opts.temperature = 0.7
    opts.topP = 0.9
    opts.seed = 42
    let infOpts = InferenceOptions.from(opts)
    try expectEqual(infOpts.temperature, 0.7)
    try expectEqual(infOpts.topP, 0.9)
    try expectEqual(infOpts.seed, 42)
}

func OLMLXTests_countChatTokens_empty() throws {
    let caps = TemplateCaps()
    let tokens = countChatTokens(messages: [], tools: nil, caps: caps)
    try expect(tokens >= 1, "should have at least 1 token")
}

func OLMLXTests_parseKeepAlive_seconds() throws {
    try expectEqual(parseKeepAlive("30s"), 30)
}

func OLMLXTests_parseKeepAlive_minutes() throws {
    try expectEqual(parseKeepAlive("5m"), 300)
}

func OLMLXTests_parseKeepAlive_hours() throws {
    try expectEqual(parseKeepAlive("1h"), 3600)
}

func OLMLXTests_parseKeepAlive_zero() throws {
    try expectEqual(parseKeepAlive("0s"), 0)
}

func OLMLXTests_promptCache_setAndGet() throws {
    let cache = PromptCacheStore(maxSlots: 4)
    let state = CachedPromptState(tokens: [1, 2, 3])
    cache.set(key: "test", state: state)
    let retrieved = cache.get(key: "test")
    try expectNotNil(retrieved)
    try expectEqual(retrieved?.tokens, [1, 2, 3])
}

func OLMLXTests_promptCache_lruEviction() throws {
    let cache = PromptCacheStore(maxSlots: 2)
    cache.set(key: "a", state: CachedPromptState(tokens: [1]))
    cache.set(key: "b", state: CachedPromptState(tokens: [2]))
    cache.set(key: "c", state: CachedPromptState(tokens: [3]))
    try expectNil(cache.get(key: "a"))
    try expectNotNil(cache.get(key: "b"))
    try expectNotNil(cache.get(key: "c"))
}
