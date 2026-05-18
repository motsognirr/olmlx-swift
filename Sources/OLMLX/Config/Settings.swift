import Foundation

// MARK: - Enums

public enum SyncMode: String, Codable, Sendable {
    case full
    case minimal
    case none
}

public enum SpeculativeStrategy: String, Codable, Sendable {
    case classic
    case dflash
    case eagle
}

public enum DistributedStrategy: String, Codable, Sendable {
    case tensor
    case pipeline
}

// MARK: - KV Cache Quant Validation

public func validateKVCacheQuantFormat(_ value: String) -> String? {
    let validMethods: Set<String> = ["turboquant", "spectral"]
    let validBits: Set<String> = ["2", "4"]
    let parts = value.split(separator: ":", maxSplits: 1).map(String.init)
    guard parts.count == 2,
          validMethods.contains(parts[0]),
          validBits.contains(parts[1]) else {
        return nil
    }
    return value
}

// MARK: - Settings

public final class Settings: Codable, @unchecked Sendable {
    public static let shared = Settings()

    // MARK: Properties

    public var host: String = "0.0.0.0"
    public var port: Int = 11434
    public var modelsDir: URL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".olmlx/models")
    public var modelsConfig: URL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".olmlx/models.json")
    public var defaultKeepAlive: String = "5m"
    public var maxLoadedModels: Int = 1
    public var memoryLimitFraction: Double = 0.75
    public var modelLoadTimeout: Double? = nil
    public var logLevel: String = "INFO"
    public var promptCache: Bool = true
    public var promptCacheMaxTokens: Int? = 32768
    public var promptCacheMaxSlots: Int = 4
    public var promptCacheDisk: Bool = false
    public var promptCacheDiskPath: URL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".olmlx/cache/kv")
    public var promptCacheDiskMaxGB: Double = 10.0
    public var inferenceQueueTimeout: Double? = 300.0
    public var inferenceTimeout: Double? = nil
    public var syncMode: SyncMode = .full
    public var maxTokensLimit: Int = 131072
    public var corsOrigins: [String] = ["http://localhost:*", "http://127.0.0.1:*"]
    public var anthropicModels: [String: String] = [:]
    public var kvCacheQuant: String? = nil
    public var speculative: Bool = false
    public var speculativeStrategy: SpeculativeStrategy = .classic
    public var speculativeDraftModel: String? = nil
    public var speculativeTokens: Int? = nil

    // MARK: Init

    public init(env: [String: String] = ProcessInfo.processInfo.environment) {
        load(from: env)
    }

    // MARK: Env Loading

    private func load(from env: [String: String]) {
        let prefix = "OLMLX_"

        if let v = env["\(prefix)HOST"] { host = v }

        if let v = env["\(prefix)PORT"], let p = Int(v), (1...65535).contains(p) {
            port = p
        }

        if let v = env["\(prefix)MODELS_DIR"] {
            modelsDir = URL(fileURLWithPath: (v as NSString).expandingTildeInPath)
        }

        if let v = env["\(prefix)MODELS_CONFIG"] {
            modelsConfig = URL(fileURLWithPath: (v as NSString).expandingTildeInPath)
        }

        if let v = env["\(prefix)DEFAULT_KEEP_ALIVE"] { defaultKeepAlive = v }

        if let v = env["\(prefix)MAX_LOADED_MODELS"], let m = Int(v) { maxLoadedModels = m }

        if let v = env["\(prefix)MEMORY_LIMIT_FRACTION"], let f = Double(v), f > 0, f <= 1.0 {
            memoryLimitFraction = f
        }

        if let v = env["\(prefix)MODEL_LOAD_TIMEOUT"], let t = Double(v), t > 0 {
            modelLoadTimeout = t
        }

        if let v = env["\(prefix)LOG_LEVEL"] { logLevel = v }

        if let v = env["\(prefix)PROMPT_CACHE"] { promptCache = v.lowercased() == "true" }

        if let v = env["\(prefix)PROMPT_CACHE_MAX_TOKENS"], let t = Int(v), t > 0 {
            promptCacheMaxTokens = t
        }

        if let v = env["\(prefix)PROMPT_CACHE_MAX_SLOTS"], let s = Int(v), s > 0 {
            promptCacheMaxSlots = s
        }

        if let v = env["\(prefix)PROMPT_CACHE_DISK"] { promptCacheDisk = v.lowercased() == "true" }

        if let v = env["\(prefix)PROMPT_CACHE_DISK_PATH"] {
            promptCacheDiskPath = URL(fileURLWithPath: (v as NSString).expandingTildeInPath)
        }

        if let v = env["\(prefix)PROMPT_CACHE_DISK_MAX_GB"], let g = Double(v), g > 0 {
            promptCacheDiskMaxGB = g
        }

        if let v = env["\(prefix)INFERENCE_QUEUE_TIMEOUT"], let t = Double(v), t > 0 {
            inferenceQueueTimeout = t
        }

        if let v = env["\(prefix)INFERENCE_TIMEOUT"], let t = Double(v), t > 0 {
            inferenceTimeout = t
        }

        if let v = env["\(prefix)SYNC_MODE"], let m = SyncMode(rawValue: v) { syncMode = m }

        if let v = env["\(prefix)MAX_TOKENS_LIMIT"], let t = Int(v), t > 0 { maxTokensLimit = t }

        if let v = env["\(prefix)CORS_ORIGINS"] {
            corsOrigins = v.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        }

        if let v = env["\(prefix)ANTHROPIC_MODELS"] {
            if let data = v.data(using: .utf8),
               let decoded = try? JSONDecoder().decode([String: String].self, from: data) {
                anthropicModels = decoded.filter { key, _ in
                    !key.contains("-") && !key.contains(":")
                }
            }
        }

        if let v = env["\(prefix)KV_CACHE_QUANT"] {
            kvCacheQuant = validateKVCacheQuantFormat(v)
        }

        if let v = env["\(prefix)SPECULATIVE"] { speculative = v.lowercased() == "true" }

        if let v = env["\(prefix)SPECULATIVE_STRATEGY"], let s = SpeculativeStrategy(rawValue: v) {
            speculativeStrategy = s
        }

        if let v = env["\(prefix)SPECULATIVE_DRAFT_MODEL"] {
            let stripped = v.trimmingCharacters(in: .whitespaces)
            if !stripped.isEmpty {
                speculativeDraftModel = stripped
            }
        }

        if let v = env["\(prefix)SPECULATIVE_TOKENS"], let t = Int(v), t > 0 {
            speculativeTokens = t
        }
    }
}

// MARK: - ExperimentalSettings

public final class ExperimentalSettings: Codable, @unchecked Sendable {
    public static let shared = ExperimentalSettings()

    // Distributed
    public var distributed: Bool = false
    public var distributedStrategy: DistributedStrategy = .tensor
    public var distributedHostfile: URL = URL(fileURLWithPath: NSString(string: "~/.olmlx/hostfile.json").expandingTildeInPath)
    public var distributedBackend: String = "ring"
    public var distributedPort: Int = 32323
    public var distributedSidebandPort: Int = 32400
    public var distributedSecret: String = ""
    public var distributedRemoteWorkingDir: String = ""
    public var distributedRemotePython: String = "python"
    public var distributedPreShard: Bool = true
    public var distributedShardDir: URL = URL(fileURLWithPath: NSString(string: "~/.olmlx/shards").expandingTildeInPath)
    public var distributedWorkerShardDir: String = "~/.olmlx/shards"

    // Flash inference
    public var flash: Bool = false
    public var flashSparsityThreshold: Double = 0.5
    public var flashMinActiveNeurons: Int = 128
    public var flashMaxActiveNeurons: Int? = nil
    public var flashWindowSize: Int = 5
    public var flashIOThreads: Int = 32
    public var flashCacheBudgetNeurons: Int = 1024
    public var flashPredictorRank: Int = 128
    public var flashPredictorSensitiveLayers: Int = 0
    public var flashPredictorSensitiveRankMultiplier: Int = 4
    public var flashBypassOSCache: Bool = false
    public var flashPreallocatedBuffer: Bool = false
    public var flashMemoryBudgetFraction: Double? = nil
    public var flashPrefetch: Bool = false
    public var flashPrefetchConfidenceThreshold: Double = 0.3
    public var flashPrefetchMinNeurons: Int = 64
    public var flashPrefetchMaxNeurons: Int? = nil
    public var flashPrefetchIOThreads: Int = 16
    public var flashSpeculative: Bool = false
    public var flashSpeculativeDraftModel: String? = nil
    public var flashSpeculativeTokens: Int = 4

    // Flash MoE
    public var flashMOE: Bool = false
    public var flashMOECacheBudgetExperts: Int = 48
    public var flashMOEIOThreads: Int = 32

    public init(env: [String: String] = ProcessInfo.processInfo.environment) {
        load(from: env)
    }

    private func load(from env: [String: String]) {
        let prefix = "OLMLX_EXPERIMENTAL_"

        if let v = env["\(prefix)DISTRIBUTED"] { distributed = v.lowercased() == "true" }
        if let v = env["\(prefix)DISTRIBUTED_STRATEGY"], let s = DistributedStrategy(rawValue: v) {
            distributedStrategy = s
        }
        if let v = env["\(prefix)DISTRIBUTED_HOSTFILE"] {
            distributedHostfile = URL(fileURLWithPath: (v as NSString).expandingTildeInPath)
        }
        if let v = env["\(prefix)DISTRIBUTED_BACKEND"] { distributedBackend = v }
        if let v = env["\(prefix)DISTRIBUTED_PORT"], let p = Int(v) { distributedPort = p }
        if let v = env["\(prefix)DISTRIBUTED_SIDEBAND_PORT"], let p = Int(v) { distributedSidebandPort = p }
        if let v = env["\(prefix)DISTRIBUTED_SECRET"] { distributedSecret = v }
        if let v = env["\(prefix)DISTRIBUTED_REMOTE_WORKING_DIR"] { distributedRemoteWorkingDir = v }
        if let v = env["\(prefix)DISTRIBUTED_REMOTE_PYTHON"] { distributedRemotePython = v }
        if let v = env["\(prefix)DISTRIBUTED_PRE_SHARD"] { distributedPreShard = v.lowercased() == "true" }
        if let v = env["\(prefix)DISTRIBUTED_SHARD_DIR"] {
            distributedShardDir = URL(fileURLWithPath: (v as NSString).expandingTildeInPath)
        }
        if let v = env["\(prefix)DISTRIBUTED_WORKER_SHARD_DIR"] { distributedWorkerShardDir = v }

        // Flash inference
        if let v = env["\(prefix)FLASH"] { flash = v.lowercased() == "true" }
        if let v = env["\(prefix)FLASH_SPARSITY_THRESHOLD"], let f = Double(v), f > 0, f <= 1.0 {
            flashSparsityThreshold = f
        }
        if let v = env["\(prefix)FLASH_MIN_ACTIVE_NEURONS"], let n = Int(v), n > 0 {
            flashMinActiveNeurons = n
        }
        if let v = env["\(prefix)FLASH_MAX_ACTIVE_NEURONS"], let n = Int(v), n > 0 {
            flashMaxActiveNeurons = n
        }
        if let v = env["\(prefix)FLASH_WINDOW_SIZE"], let n = Int(v), n > 0 { flashWindowSize = n }
        if let v = env["\(prefix)FLASH_IO_THREADS"], let n = Int(v), n > 0 { flashIOThreads = n }
        if let v = env["\(prefix)FLASH_CACHE_BUDGET_NEURONS"], let n = Int(v), n >= 0 {
            flashCacheBudgetNeurons = n
        }
        if let v = env["\(prefix)FLASH_PREDICTOR_RANK"], let n = Int(v), n > 0 { flashPredictorRank = n }
        if let v = env["\(prefix)FLASH_PREDICTOR_SENSITIVE_LAYERS"], let n = Int(v), n >= 0 {
            flashPredictorSensitiveLayers = n
        }
        if let v = env["\(prefix)FLASH_PREDICTOR_SENSITIVE_RANK_MULTIPLIER"], let n = Int(v), n > 0 {
            flashPredictorSensitiveRankMultiplier = n
        }
        if let v = env["\(prefix)FLASH_BYPASS_OS_CACHE"] { flashBypassOSCache = v.lowercased() == "true" }
        if let v = env["\(prefix)FLASH_PREALLOCATED_BUFFER"] { flashPreallocatedBuffer = v.lowercased() == "true" }
        if let v = env["\(prefix)FLASH_MEMORY_BUDGET_FRACTION"], let f = Double(v), f > 0, f <= 1.0 {
            flashMemoryBudgetFraction = f
        }
        if let v = env["\(prefix)FLASH_PREFETCH"] { flashPrefetch = v.lowercased() == "true" }
        if let v = env["\(prefix)FLASH_PREFETCH_CONFIDENCE_THRESHOLD"], let f = Double(v), f > 0, f <= 1.0 {
            flashPrefetchConfidenceThreshold = f
        }
        if let v = env["\(prefix)FLASH_PREFETCH_MIN_NEURONS"], let n = Int(v), n > 0 {
            flashPrefetchMinNeurons = n
        }
        if let v = env["\(prefix)FLASH_PREFETCH_MAX_NEURONS"], let n = Int(v), n > 0 {
            flashPrefetchMaxNeurons = n
        }
        if let v = env["\(prefix)FLASH_PREFETCH_IO_THREADS"], let n = Int(v), n > 0 {
            flashPrefetchIOThreads = n
        }
        if let v = env["\(prefix)FLASH_SPECULATIVE"] { flashSpeculative = v.lowercased() == "true" }
        if let v = env["\(prefix)FLASH_SPECULATIVE_DRAFT_MODEL"] { flashSpeculativeDraftModel = v }
        if let v = env["\(prefix)FLASH_SPECULATIVE_TOKENS"], let n = Int(v), n > 0 {
            flashSpeculativeTokens = n
        }

        // Flash MoE
        if let v = env["\(prefix)FLASH_MOE"] { flashMOE = v.lowercased() == "true" }
        if let v = env["\(prefix)FLASH_MOE_CACHE_BUDGET_EXPERTS"], let n = Int(v), n >= 0 {
            flashMOECacheBudgetExperts = n
        }
        if let v = env["\(prefix)FLASH_MOE_IO_THREADS"], let n = Int(v), n > 0 {
            flashMOEIOThreads = n
        }
    }
}

// MARK: - ResolvedExperimentalSettings

public struct ResolvedExperimentalSettings {
    public static func resolve(
        base: ExperimentalSettings,
        overrides: [String: Any]
    ) -> ExperimentalSettings {
        guard !overrides.isEmpty else { return base }

        let result = ExperimentalSettings(env: [:])

        // Helper: apply overrides to the result
        let applyBool: (inout Bool, String) -> Void = { value, key in
            if let v = overrides[key] as? Bool { value = v }
        }
        let applyDouble: (inout Double, String) -> Void = { value, key in
            if let v = overrides[key] as? Double { value = v }
        }
        let applyInt: (inout Int, String) -> Void = { value, key in
            if let v = overrides[key] as? Int { value = v }
        }
        let applyOptionalInt: (inout Int?, String) -> Void = { value, key in
            if overrides.keys.contains(key) { value = overrides[key] as? Int }
        }
        let applyOptionalDouble: (inout Double?, String) -> Void = { value, key in
            if overrides.keys.contains(key) { value = overrides[key] as? Double }
        }
        let applyOptionalString: (inout String?, String) -> Void = { value, key in
            if overrides.keys.contains(key) { value = overrides[key] as? String }
        }

        // Copy base values
        result.distributed = base.distributed
        result.distributedStrategy = base.distributedStrategy
        result.distributedBackend = base.distributedBackend
        result.distributedPort = base.distributedPort
        result.distributedSidebandPort = base.distributedSidebandPort
        result.distributedSecret = base.distributedSecret
        result.distributedRemoteWorkingDir = base.distributedRemoteWorkingDir
        result.distributedRemotePython = base.distributedRemotePython
        result.distributedPreShard = base.distributedPreShard
        result.distributedWorkerShardDir = base.distributedWorkerShardDir

        result.flash = base.flash
        result.flashSparsityThreshold = base.flashSparsityThreshold
        result.flashMinActiveNeurons = base.flashMinActiveNeurons
        result.flashMaxActiveNeurons = base.flashMaxActiveNeurons
        result.flashWindowSize = base.flashWindowSize
        result.flashIOThreads = base.flashIOThreads
        result.flashCacheBudgetNeurons = base.flashCacheBudgetNeurons
        result.flashPredictorRank = base.flashPredictorRank
        result.flashPredictorSensitiveLayers = base.flashPredictorSensitiveLayers
        result.flashPredictorSensitiveRankMultiplier = base.flashPredictorSensitiveRankMultiplier
        result.flashBypassOSCache = base.flashBypassOSCache
        result.flashPreallocatedBuffer = base.flashPreallocatedBuffer
        result.flashMemoryBudgetFraction = base.flashMemoryBudgetFraction
        result.flashPrefetch = base.flashPrefetch
        result.flashPrefetchConfidenceThreshold = base.flashPrefetchConfidenceThreshold
        result.flashPrefetchMinNeurons = base.flashPrefetchMinNeurons
        result.flashPrefetchMaxNeurons = base.flashPrefetchMaxNeurons
        result.flashPrefetchIOThreads = base.flashPrefetchIOThreads
        result.flashSpeculative = base.flashSpeculative
        result.flashSpeculativeDraftModel = base.flashSpeculativeDraftModel
        result.flashSpeculativeTokens = base.flashSpeculativeTokens
        result.flashMOE = base.flashMOE
        result.flashMOECacheBudgetExperts = base.flashMOECacheBudgetExperts
        result.flashMOEIOThreads = base.flashMOEIOThreads

        // Apply overrides
        applyBool(&result.flash, "flash")
        applyDouble(&result.flashSparsityThreshold, "flash_sparsity_threshold")
        applyInt(&result.flashMinActiveNeurons, "flash_min_active_neurons")
        applyOptionalInt(&result.flashMaxActiveNeurons, "flash_max_active_neurons")
        applyInt(&result.flashWindowSize, "flash_window_size")
        applyInt(&result.flashIOThreads, "flash_io_threads")
        applyInt(&result.flashCacheBudgetNeurons, "flash_cache_budget_neurons")
        applyBool(&result.flashBypassOSCache, "flash_bypass_os_cache")
        applyBool(&result.flashPreallocatedBuffer, "flash_preallocated_buffer")
        applyOptionalDouble(&result.flashMemoryBudgetFraction, "flash_memory_budget_fraction")
        applyBool(&result.flashPrefetch, "flash_prefetch")
        applyDouble(&result.flashPrefetchConfidenceThreshold, "flash_prefetch_confidence_threshold")
        applyInt(&result.flashPrefetchMinNeurons, "flash_prefetch_min_neurons")
        applyOptionalInt(&result.flashPrefetchMaxNeurons, "flash_prefetch_max_neurons")
        applyInt(&result.flashPrefetchIOThreads, "flash_prefetch_io_threads")
        applyBool(&result.flashSpeculative, "flash_speculative")
        applyOptionalString(&result.flashSpeculativeDraftModel, "flash_speculative_draft_model")
        applyInt(&result.flashSpeculativeTokens, "flash_speculative_tokens")
        applyBool(&result.flashMOE, "flash_moe")
        applyInt(&result.flashMOECacheBudgetExperts, "flash_moe_cache_budget_experts")
        applyInt(&result.flashMOEIOThreads, "flash_moe_io_threads")

        return result
    }
}
