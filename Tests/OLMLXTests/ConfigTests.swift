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
        #expect(Settings(env: ["OLMLX_KV_CACHE_QUANT": "affine:2"]).kvCacheQuant == "affine:2")
        #expect(Settings(env: ["OLMLX_KV_CACHE_QUANT": "affine:4"]).kvCacheQuant == "affine:4")
        #expect(Settings(env: ["OLMLX_KV_CACHE_QUANT": "affine:8"]).kvCacheQuant == "affine:8")
    }

    @Test func kvCacheQuantInvalid() {
        // MLX-swift only implements affine KV cache quantization, so turboquant/spectral
        // (supported by the Python reference) are rejected here rather than silently
        // accepted and ignored at inference time.
        #expect(Settings(env: ["OLMLX_KV_CACHE_QUANT": "turboquant:4"]).kvCacheQuant == nil)
        #expect(Settings(env: ["OLMLX_KV_CACHE_QUANT": "spectral:4"]).kvCacheQuant == nil)
        #expect(Settings(env: ["OLMLX_KV_CACHE_QUANT": "bogus:4"]).kvCacheQuant == nil)
        #expect(Settings(env: ["OLMLX_KV_CACHE_QUANT": "affine:3"]).kvCacheQuant == nil)
        #expect(Settings(env: ["OLMLX_KV_CACHE_QUANT": "affine"]).kvCacheQuant == nil)
    }

    @Test func parseKVCacheQuantParses() {
        let result = parseKVCacheQuant("affine:4")
        #expect(result?.kind == "affine")
        #expect(result?.bits == 4)
    }

    @Test func parseKVCacheQuantRejectsUnknownKind() {
        #expect(parseKVCacheQuant("turboquant:4") == nil)
        #expect(parseKVCacheQuant("affine:5") == nil)
        #expect(parseKVCacheQuant("affine") == nil)
        #expect(parseKVCacheQuant("") == nil)
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

    @Test func maxRequestBodySizeDefault() {
        #expect(Settings(env: [:]).maxRequestBodySize == "16mb")
    }

    @Test func maxRequestBodySizeOverride() {
        #expect(Settings(env: ["OLMLX_MAX_REQUEST_BODY_SIZE": "64mb"]).maxRequestBodySize == "64mb")
        #expect(Settings(env: ["OLMLX_MAX_REQUEST_BODY_SIZE": "1gb"]).maxRequestBodySize == "1gb")
        #expect(Settings(env: ["OLMLX_MAX_REQUEST_BODY_SIZE": "1048576"]).maxRequestBodySize == "1048576")
    }

    @Test func maxRequestBodySizeRejectsInvalid() {
        #expect(Settings(env: ["OLMLX_MAX_REQUEST_BODY_SIZE": "bogus"]).maxRequestBodySize == "16mb")
        #expect(Settings(env: ["OLMLX_MAX_REQUEST_BODY_SIZE": "16xb"]).maxRequestBodySize == "16mb")
        #expect(Settings(env: ["OLMLX_MAX_REQUEST_BODY_SIZE": "-1mb"]).maxRequestBodySize == "16mb")
        #expect(Settings(env: ["OLMLX_MAX_REQUEST_BODY_SIZE": "0"]).maxRequestBodySize == "16mb")
        #expect(Settings(env: ["OLMLX_MAX_REQUEST_BODY_SIZE": ""]).maxRequestBodySize == "16mb")
    }

    @Test func validateByteCountFormatAccepts() {
        #expect(validateByteCountFormat("16mb") == "16mb")
        #expect(validateByteCountFormat("16MB") == "16mb")
        #expect(validateByteCountFormat(" 2kb ") == "2kb")
        #expect(validateByteCountFormat("1024") == "1024")
    }

    @Test func validateByteCountFormatRejects() {
        #expect(validateByteCountFormat("bogus") == nil)
        #expect(validateByteCountFormat("16xb") == nil)
        #expect(validateByteCountFormat("mb") == nil)
        #expect(validateByteCountFormat("0") == nil)
        #expect(validateByteCountFormat("-1mb") == nil)
    }

}
