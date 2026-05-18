import Foundation
import OLMLX
import OLMLXTestUtils

func OLMLXTests_config_testDefaults() throws {
    let s = Settings(env: [:])
    try expectEqual(s.host, "0.0.0.0")
    try expectEqual(s.port, 11434)
    try expectEqual(s.defaultKeepAlive, "5m")
    try expectEqual(s.maxLoadedModels, 1)
    try expectNil(s.modelLoadTimeout)
    try expect(s.modelsDir.path.hasSuffix("models"), "modelsDir should end with 'models': \(s.modelsDir.path)")
    try expect(s.modelsConfig.path.hasSuffix("models.json"), "modelsConfig should end with 'models.json': \(s.modelsConfig.path)")
}

func OLMLXTests_config_testEnvOverride() throws {
    let env: [String: String] = [
        "OLMLX_HOST": "127.0.0.1",
        "OLMLX_PORT": "8080",
        "OLMLX_MAX_LOADED_MODELS": "3",
        "OLMLX_MODEL_LOAD_TIMEOUT": "120",
    ]
    let s = Settings(env: env)
    try expectEqual(s.host, "127.0.0.1")
    try expectEqual(s.port, 8080)
    try expectEqual(s.maxLoadedModels, 3)
    try expectEqual(s.modelLoadTimeout, 120.0)
}

func OLMLXTests_config_testPortRejectsZero() throws {
    let s = Settings(env: ["OLMLX_PORT": "0"])
    try expectEqual(s.port, 11434)
}

func OLMLXTests_config_testPortAcceptsBoundaries() throws {
    do {
        let s = Settings(env: ["OLMLX_PORT": "1"])
        try expectEqual(s.port, 1)
    }
    do {
        let s = Settings(env: ["OLMLX_PORT": "65535"])
        try expectEqual(s.port, 65535)
    }
}

func OLMLXTests_config_testAnthropicModelsFromEnv() throws {
    let env: [String: String] = [
        "OLMLX_ANTHROPIC_MODELS": #"{"haiku":"qwen3:latest","sonnet":"qwen3-8b:latest"}"#
    ]
    let s = Settings(env: env)
    try expectEqual(s.anthropicModels["haiku"], "qwen3:latest")
    try expectEqual(s.anthropicModels["sonnet"], "qwen3-8b:latest")
    try expectNil(s.anthropicModels["claude-sonnet"])
    try expectNil(s.anthropicModels["sonnet:latest"])
}

func OLMLXTests_config_testKVcacheQuantValid() throws {
    let s = Settings(env: ["OLMLX_KV_CACHE_QUANT": "turboquant:4"])
    try expectEqual(s.kvCacheQuant, "turboquant:4")
}

func OLMLXTests_config_testKVcacheQuantInvalid() throws {
    let s1 = Settings(env: ["OLMLX_KV_CACHE_QUANT": "bogus:4"])
    try expectNil(s1.kvCacheQuant)
    let s2 = Settings(env: ["OLMLX_KV_CACHE_QUANT": "turboquant:8"])
    try expectNil(s2.kvCacheQuant)
    let s3 = Settings(env: ["OLMLX_KV_CACHE_QUANT": "turboquant"])
    try expectNil(s3.kvCacheQuant)
}

func OLMLXTests_config_testSpeculativeDefaults() throws {
    let s = Settings(env: [:])
    try expectEqual(s.speculative, false)
    try expectNil(s.speculativeDraftModel)
    try expectNil(s.speculativeTokens)
}

func OLMLXTests_config_testSpeculativeEnvOverride() throws {
    let env: [String: String] = [
        "OLMLX_SPECULATIVE": "true",
        "OLMLX_SPECULATIVE_DRAFT_MODEL": "Qwen/Qwen3-0.6B",
        "OLMLX_SPECULATIVE_TOKENS": "6",
    ]
    let s = Settings(env: env)
    try expectEqual(s.speculative, true)
    try expectEqual(s.speculativeDraftModel, "Qwen/Qwen3-0.6B")
    try expectEqual(s.speculativeTokens, 6)
}

func OLMLXTests_config_testSpeculativeStrategy() throws {
    let sDefault = Settings(env: [:])
    try expectEqual(sDefault.speculativeStrategy, SpeculativeStrategy.classic)
    let sDFlash = Settings(env: ["OLMLX_SPECULATIVE_STRATEGY": "dflash"])
    try expectEqual(sDFlash.speculativeStrategy, .dflash)
    let sEagle = Settings(env: ["OLMLX_SPECULATIVE_STRATEGY": "eagle"])
    try expectEqual(sEagle.speculativeStrategy, .eagle)
    let sBogus = Settings(env: ["OLMLX_SPECULATIVE_STRATEGY": "bogus"])
    try expectEqual(sBogus.speculativeStrategy, .classic)
}

func OLMLXTests_config_testSyncModeOverride() throws {
    let sDefault = Settings(env: [:])
    try expectEqual(sDefault.syncMode, SyncMode.full)
    let sMin = Settings(env: ["OLMLX_SYNC_MODE": "minimal"])
    try expectEqual(sMin.syncMode, .minimal)
    let sNone = Settings(env: ["OLMLX_SYNC_MODE": "none"])
    try expectEqual(sNone.syncMode, .none)
}

func OLMLXTests_config_testExperimentalDefaults() throws {
    let e = ExperimentalSettings(env: [:])
    try expectEqual(e.distributed, false)
    try expectEqual(e.distributedStrategy, .tensor)
    try expectEqual(e.flash, false)
    try expectEqual(e.flashSparsityThreshold, 0.5)
    try expectEqual(e.flashMinActiveNeurons, 128)
    try expectNil(e.flashMaxActiveNeurons)
}

func OLMLXTests_config_testExperimentalOverride() throws {
    let e = ExperimentalSettings(env: [
        "OLMLX_EXPERIMENTAL_FLASH": "true",
        "OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD": "0.3",
    ])
    try expectEqual(e.flash, true)
    try expectEqual(e.flashSparsityThreshold, 0.3)
}

func OLMLXTests_config_testResolveExperimental() throws {
    let base = ExperimentalSettings(env: [:])
    let result = ResolvedExperimentalSettings.resolve(
        base: base,
        overrides: [
            "flash": true,
            "flash_sparsity_threshold": 0.3,
            "flash_moe": true,
        ]
    )
    try expectEqual(result.flash, true)
    try expectEqual(result.flashSparsityThreshold, 0.3)
    try expectEqual(result.flashMOE, true)
    try expectEqual(base.flash, false)
}
