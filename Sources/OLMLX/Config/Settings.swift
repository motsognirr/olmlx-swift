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

// MARK: - KV Cache Quant Validation

public func validateKVCacheQuantFormat(_ value: String) -> String? {
    let validMethods: Set<String> = ["turboquant", "spectral"]
    let validBits: Set<String> = ["2", "4"]
    let parts = value.split(separator: ":", maxSplits: 1).map(String.init)
    guard parts.count == 2,
        validMethods.contains(parts[0]),
        validBits.contains(parts[1])
    else {
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
    public var promptCacheDiskPath: URL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(
        ".olmlx/cache/kv")
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
                let decoded = try? JSONDecoder().decode([String: String].self, from: data)
            {
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
