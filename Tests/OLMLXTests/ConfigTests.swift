import Foundation
import Testing

@testable import OLMLX

@Suite("Config")
struct ConfigTests {
    @Test func defaults() {
        let s = Settings(env: [:])
        #expect(s.host == "0.0.0.0")
        #expect(s.port == 11434)
        #expect(s.defaultKeepAlive == "5m")
        #expect(s.maxLoadedModels == 1)
        #expect(s.modelLoadTimeout == nil)
        #expect(s.modelsDir.path.hasSuffix("models"))
        #expect(s.modelsConfig.path.hasSuffix("models.json"))
    }

    @Test func envOverride() {
        let env: [String: String] = [
            "OLMLX_HOST": "127.0.0.1",
            "OLMLX_PORT": "8080",
            "OLMLX_MAX_LOADED_MODELS": "3",
            "OLMLX_MODEL_LOAD_TIMEOUT": "120",
        ]
        let s = Settings(env: env)
        #expect(s.host == "127.0.0.1")
        #expect(s.port == 8080)
        #expect(s.maxLoadedModels == 3)
        #expect(s.modelLoadTimeout == 120.0)
    }

    @Test func portRejectsZero() {
        let s = Settings(env: ["OLMLX_PORT": "0"])
        #expect(s.port == 11434)
    }

    @Test func portAcceptsBoundaries() {
        #expect(Settings(env: ["OLMLX_PORT": "1"]).port == 1)
        #expect(Settings(env: ["OLMLX_PORT": "65535"]).port == 65535)
    }

    @Test func anthropicModelsFromEnv() {
        let env: [String: String] = [
            "OLMLX_ANTHROPIC_MODELS": #"{"haiku":"qwen3:latest","sonnet":"qwen3-8b:latest"}"#
        ]
        let s = Settings(env: env)
        #expect(s.anthropicModels["haiku"] == "qwen3:latest")
        #expect(s.anthropicModels["sonnet"] == "qwen3-8b:latest")
        #expect(s.anthropicModels["claude-sonnet"] == nil)
        #expect(s.anthropicModels["sonnet:latest"] == nil)
    }

    @Test func kvCacheQuantValid() {
        let s = Settings(env: ["OLMLX_KV_CACHE_QUANT": "turboquant:4"])
        #expect(s.kvCacheQuant == "turboquant:4")
    }

    @Test func kvCacheQuantInvalid() {
        #expect(Settings(env: ["OLMLX_KV_CACHE_QUANT": "bogus:4"]).kvCacheQuant == nil)
        #expect(Settings(env: ["OLMLX_KV_CACHE_QUANT": "turboquant:8"]).kvCacheQuant == nil)
        #expect(Settings(env: ["OLMLX_KV_CACHE_QUANT": "turboquant"]).kvCacheQuant == nil)
    }

    @Test func speculativeDefaults() {
        let s = Settings(env: [:])
        #expect(s.speculative == false)
        #expect(s.speculativeDraftModel == nil)
        #expect(s.speculativeTokens == nil)
    }

    @Test func speculativeEnvOverride() {
        let env: [String: String] = [
            "OLMLX_SPECULATIVE": "true",
            "OLMLX_SPECULATIVE_DRAFT_MODEL": "Qwen/Qwen3-0.6B",
            "OLMLX_SPECULATIVE_TOKENS": "6",
        ]
        let s = Settings(env: env)
        #expect(s.speculative == true)
        #expect(s.speculativeDraftModel == "Qwen/Qwen3-0.6B")
        #expect(s.speculativeTokens == 6)
    }

    @Test func speculativeStrategy() {
        #expect(Settings(env: [:]).speculativeStrategy == .classic)
        #expect(Settings(env: ["OLMLX_SPECULATIVE_STRATEGY": "dflash"]).speculativeStrategy == .dflash)
        #expect(Settings(env: ["OLMLX_SPECULATIVE_STRATEGY": "eagle"]).speculativeStrategy == .eagle)
        #expect(Settings(env: ["OLMLX_SPECULATIVE_STRATEGY": "bogus"]).speculativeStrategy == .classic)
    }

    @Test func syncModeOverride() {
        #expect(Settings(env: [:]).syncMode == .full)
        #expect(Settings(env: ["OLMLX_SYNC_MODE": "minimal"]).syncMode == .minimal)
        #expect(Settings(env: ["OLMLX_SYNC_MODE": "none"]).syncMode == SyncMode.none)
    }

}
