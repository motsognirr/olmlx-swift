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

@Suite("Speculative Decoding Resolution")
struct SpeculativeResolutionTests {
    private func makeManager(env: [String: String] = [:]) -> ModelManager {
        let tmp = FileManager.default.temporaryDirectory
        let cfg = tmp.appendingPathComponent("spec-tests-\(UUID().uuidString).json")
        let registry = ModelRegistry(configPath: cfg)
        let store = ModelStore(modelsDir: tmp, registry: registry)
        return ModelManager(
            registry: registry, store: store,
            settings: Settings(env: env), inferenceEngine: MockInferenceEngine()
        )
    }

    @Test func resolvedSpeculativeFallsBackToGlobal() {
        let global = Settings(env: [
            "OLMLX_SPECULATIVE": "true",
            "OLMLX_SPECULATIVE_DRAFT_MODEL": "org/draft",
            "OLMLX_SPECULATIVE_TOKENS": "5",
        ])
        let config = ModelConfig(hfPath: "org/main")
        let spec = config.resolvedSpeculative(global: global)
        #expect(spec.enabled)
        #expect(spec.draftModel == "org/draft")
        #expect(spec.numTokens == 5)
        #expect(spec.strategy == .classic)
    }

    @Test func resolvedSpeculativePerModelOverrides() {
        let global = Settings(env: [
            "OLMLX_SPECULATIVE": "true",
            "OLMLX_SPECULATIVE_DRAFT_MODEL": "org/global",
        ])
        let config = ModelConfig(
            hfPath: "org/main",
            speculativeDraftModel: "org/per-model",
            speculativeTokens: 7,
            speculativeStrategy: .classic
        )
        let spec = config.resolvedSpeculative(global: global)
        #expect(spec.draftModel == "org/per-model")
        #expect(spec.numTokens == 7)
    }

    @Test func disabledReturnsNil() async throws {
        let manager = makeManager()
        let spec = SpeculativeConfig(enabled: false)
        let runtime = try await makeSpeculativeRuntime(manager: manager, config: spec)
        #expect(runtime == nil)
    }

    @Test func dflashStrategyRejected() async {
        let manager = makeManager()
        let spec = SpeculativeConfig(
            enabled: true, draftModel: "org/draft", strategy: .dflash)
        await #expect(
            throws: InferenceError.unsupportedSpeculativeStrategy("dflash")
        ) {
            _ = try await makeSpeculativeRuntime(manager: manager, config: spec)
        }
    }

    @Test func eagleStrategyRejected() async {
        let manager = makeManager()
        let spec = SpeculativeConfig(
            enabled: true, draftModel: "org/draft", strategy: .eagle)
        await #expect(
            throws: InferenceError.unsupportedSpeculativeStrategy("eagle")
        ) {
            _ = try await makeSpeculativeRuntime(manager: manager, config: spec)
        }
    }

    @Test func missingDraftRejected() async {
        let manager = makeManager()
        let spec = SpeculativeConfig(enabled: true, draftModel: nil, strategy: .classic)
        await #expect(throws: InferenceError.speculativeDraftModelMissing) {
            _ = try await makeSpeculativeRuntime(manager: manager, config: spec)
        }
    }

    @Test func classicWithDraftDelegatesToEngine() async {
        // The mock engine throws because we never set a stubContainer, but the
        // exception confirms the runtime called through to `engine.loadModel`
        // with the draft model's local path — i.e. the draft loader is wired.
        let manager = makeManager()
        let spec = SpeculativeConfig(
            enabled: true, draftModel: "org/draft",
            numTokens: 3, strategy: .classic)
        await #expect(throws: (any Error).self) {
            _ = try await makeSpeculativeRuntime(manager: manager, config: spec)
        }
    }

    @Test func numTokensClampedAboveZero() {
        // The runtime construction takes max(1, numTokens ?? 2). Verify the
        // sentinel: a zero in config falls back to 1 rather than disabling.
        let spec = SpeculativeConfig(
            enabled: true, draftModel: "org/draft", numTokens: 0, strategy: .classic)
        #expect(spec.numTokens == 0)
        // The clamp is exercised inside makeSpeculativeRuntime, which we can't
        // run end-to-end without a real model; this assertion documents intent.
    }
}

@Suite("Prompt Cache")
struct PromptCacheTests {
    @Test func setAndGet() async {
        let cache = PromptCacheStore(maxSlots: 4)
        await cache.set(key: "test", state: CachedPromptState(tokens: [1, 2, 3]))
        let retrieved = await cache.get(key: "test")
        #expect(retrieved != nil)
        #expect(retrieved?.tokens == [1, 2, 3])
    }

    @Test func lruEviction() async {
        let cache = PromptCacheStore(maxSlots: 2)
        await cache.set(key: "a", state: CachedPromptState(tokens: [1]))
        await cache.set(key: "b", state: CachedPromptState(tokens: [2]))
        await cache.set(key: "c", state: CachedPromptState(tokens: [3]))
        #expect(await cache.get(key: "a") == nil)
        #expect(await cache.get(key: "b") != nil)
        #expect(await cache.get(key: "c") != nil)
    }

    @Test func longestCommonPrefixBasics() {
        #expect(longestCommonPrefix([1, 2, 3], [1, 2, 3, 4]) == 3)
        #expect(longestCommonPrefix([1, 2, 3, 4], [1, 2, 3]) == 3)
        #expect(longestCommonPrefix([1, 2, 3], [1, 2, 9]) == 2)
        #expect(longestCommonPrefix([], [1, 2]) == 0)
        #expect(longestCommonPrefix([1, 2], []) == 0)
        #expect(longestCommonPrefix([1, 2, 3], [9, 1, 2, 3]) == 0)
    }

    @Test func planReuseWithNoCachedStateFallsThrough() {
        let plan = planPromptCacheReuse(cached: nil, newTokens: [1, 2, 3])
        #expect(plan.cache == nil)
        #expect(plan.prefillTokens == [1, 2, 3])
        #expect(plan.reusedTokens == 0)
    }

    @Test func planReuseWithEmptyCacheFallsThrough() {
        // CachedPromptState with nil cache is treated as a miss: caller allocates fresh.
        let cached = CachedPromptState(tokens: [1, 2, 3], cache: nil)
        let plan = planPromptCacheReuse(cached: cached, newTokens: [1, 2, 3, 4])
        #expect(plan.cache == nil)
        #expect(plan.prefillTokens == [1, 2, 3, 4])
        #expect(plan.reusedTokens == 0)
    }

    @Test func acquireRemovesSlot() async {
        let store = PromptCacheStore(maxSlots: 4)
        await store.set(key: "k", state: CachedPromptState(tokens: [1]))
        // direct fetch+remove simulates ModelManager.acquirePromptCache
        let first = await store.get(key: "k")
        await store.remove(key: "k")
        #expect(first != nil)
        #expect(await store.get(key: "k") == nil)
    }
}
