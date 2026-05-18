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
//
// MLX-swift currently only implements affine quantization for KV caches
// (see ``QuantizedKVCache`` and ``maybeQuantizeKVCache`` in MLXLMCommon).
// We reject other kinds at parse time rather than silently dropping them.

public let kvCacheQuantSupportedKinds: Set<String> = ["affine"]
public let kvCacheQuantSupportedBits: Set<Int> = [2, 4, 8]

public func parseKVCacheQuant(_ value: String) -> (kind: String, bits: Int)? {
    let parts = value.split(separator: ":", maxSplits: 1).map(String.init)
    guard parts.count == 2,
        kvCacheQuantSupportedKinds.contains(parts[0]),
        let bits = Int(parts[1]),
        kvCacheQuantSupportedBits.contains(bits)
    else {
        return nil
    }
    return (parts[0], bits)
}

public func validateKVCacheQuantFormat(_ value: String) -> String? {
    return parseKVCacheQuant(value) == nil ? nil : value
}

// MARK: - Byte Count Validation
//
// Vapor's `ByteCount` string parser fatalErrors on invalid input, so we
// pre-validate operator-supplied values before handing them to Vapor.
// Accepts a positive integer optionally suffixed with kb / mb / gb / tb
// (case-insensitive, suffixes are powers of two per Vapor's convention).

public func validateByteCountFormat(_ value: String) -> String? {
    let trimmed = value.trimmingCharacters(in: .whitespaces)
    if let n = Int(trimmed), n > 0 { return trimmed }
    let lowered = trimmed.lowercased()
    let suffixes = ["kb", "mb", "gb", "tb"]
    for suffix in suffixes where lowered.hasSuffix(suffix) {
        let numberPart = lowered.dropLast(suffix.count)
        if let n = Int(numberPart), n > 0 { return lowered }
        return nil
    }
    return nil
}

// MARK: - Settings
//
// Immutable, Sendable snapshot of server configuration loaded once at startup
// from the process environment. Pass it explicitly to consumers rather than
// reaching for a global — that way tests construct isolated instances and the
// compiler enforces that the configuration cannot change at runtime.

public struct Settings: Sendable {
    public let host: String
    public let port: Int
    public let modelsDir: URL
    public let modelsConfig: URL
    public let defaultKeepAlive: String
    public let maxLoadedModels: Int
    public let memoryLimitFraction: Double
    public let modelLoadTimeout: Double?
    public let logLevel: String
    public let promptCache: Bool
    public let promptCacheMaxTokens: Int?
    public let promptCacheMaxSlots: Int
    public let promptCacheDisk: Bool
    public let promptCacheDiskPath: URL
    public let promptCacheDiskMaxGB: Double
    public let inferenceQueueTimeout: Double?
    public let inferenceTimeout: Double?
    public let syncMode: SyncMode
    public let maxTokensLimit: Int
    public let corsOrigins: [String]
    public let anthropicModels: [String: String]
    public let kvCacheQuant: String?
    public let speculative: Bool
    public let speculativeStrategy: SpeculativeStrategy
    public let speculativeDraftModel: String?
    public let speculativeTokens: Int?
    public let maxRequestBodySize: String

    public init(env: [String: String] = ProcessInfo.processInfo.environment) {
        let prefix = "OLMLX_"
        let home = NSHomeDirectory()

        self.host = env["\(prefix)HOST"] ?? "0.0.0.0"

        if let v = env["\(prefix)PORT"], let p = Int(v), (1...65535).contains(p) {
            self.port = p
        } else {
            self.port = 11434
        }

        if let v = env["\(prefix)MODELS_DIR"] {
            self.modelsDir = URL(fileURLWithPath: (v as NSString).expandingTildeInPath)
        } else {
            self.modelsDir = URL(fileURLWithPath: home).appendingPathComponent(".olmlx/models")
        }

        if let v = env["\(prefix)MODELS_CONFIG"] {
            self.modelsConfig = URL(fileURLWithPath: (v as NSString).expandingTildeInPath)
        } else {
            self.modelsConfig = URL(fileURLWithPath: home).appendingPathComponent(".olmlx/models.json")
        }

        self.defaultKeepAlive = env["\(prefix)DEFAULT_KEEP_ALIVE"] ?? "5m"

        if let v = env["\(prefix)MAX_LOADED_MODELS"], let m = Int(v) {
            self.maxLoadedModels = m
        } else {
            self.maxLoadedModels = 1
        }

        if let v = env["\(prefix)MEMORY_LIMIT_FRACTION"], let f = Double(v), f > 0, f <= 1.0 {
            self.memoryLimitFraction = f
        } else {
            self.memoryLimitFraction = 0.75
        }

        if let v = env["\(prefix)MODEL_LOAD_TIMEOUT"], let t = Double(v), t > 0 {
            self.modelLoadTimeout = t
        } else {
            self.modelLoadTimeout = nil
        }

        self.logLevel = env["\(prefix)LOG_LEVEL"] ?? "INFO"

        if let v = env["\(prefix)PROMPT_CACHE"] {
            self.promptCache = v.lowercased() == "true"
        } else {
            self.promptCache = true
        }

        if let v = env["\(prefix)PROMPT_CACHE_MAX_TOKENS"], let t = Int(v), t > 0 {
            self.promptCacheMaxTokens = t
        } else {
            self.promptCacheMaxTokens = 32768
        }

        if let v = env["\(prefix)PROMPT_CACHE_MAX_SLOTS"], let s = Int(v), s > 0 {
            self.promptCacheMaxSlots = s
        } else {
            self.promptCacheMaxSlots = 4
        }

        self.promptCacheDisk = env["\(prefix)PROMPT_CACHE_DISK"]?.lowercased() == "true"

        if let v = env["\(prefix)PROMPT_CACHE_DISK_PATH"] {
            self.promptCacheDiskPath = URL(fileURLWithPath: (v as NSString).expandingTildeInPath)
        } else {
            self.promptCacheDiskPath = URL(fileURLWithPath: home).appendingPathComponent(".olmlx/cache/kv")
        }

        if let v = env["\(prefix)PROMPT_CACHE_DISK_MAX_GB"], let g = Double(v), g > 0 {
            self.promptCacheDiskMaxGB = g
        } else {
            self.promptCacheDiskMaxGB = 10.0
        }

        if let v = env["\(prefix)INFERENCE_QUEUE_TIMEOUT"], let t = Double(v), t > 0 {
            self.inferenceQueueTimeout = t
        } else {
            self.inferenceQueueTimeout = 300.0
        }

        if let v = env["\(prefix)INFERENCE_TIMEOUT"], let t = Double(v), t > 0 {
            self.inferenceTimeout = t
        } else {
            self.inferenceTimeout = nil
        }

        if let v = env["\(prefix)SYNC_MODE"], let m = SyncMode(rawValue: v) {
            self.syncMode = m
        } else {
            self.syncMode = .full
        }

        if let v = env["\(prefix)MAX_TOKENS_LIMIT"], let t = Int(v), t > 0 {
            self.maxTokensLimit = t
        } else {
            self.maxTokensLimit = 131072
        }

        if let v = env["\(prefix)CORS_ORIGINS"] {
            self.corsOrigins = v.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        } else {
            self.corsOrigins = ["http://localhost:*", "http://127.0.0.1:*"]
        }

        if let v = env["\(prefix)ANTHROPIC_MODELS"],
            let data = v.data(using: .utf8),
            let decoded = try? JSONDecoder().decode([String: String].self, from: data)
        {
            self.anthropicModels = decoded.filter { key, _ in
                !key.contains("-") && !key.contains(":")
            }
        } else {
            self.anthropicModels = [:]
        }

        if let v = env["\(prefix)KV_CACHE_QUANT"] {
            if let valid = validateKVCacheQuantFormat(v) {
                self.kvCacheQuant = valid
            } else {
                let kinds = kvCacheQuantSupportedKinds.sorted().joined(separator: ",")
                let bits = kvCacheQuantSupportedBits.sorted().map(String.init).joined(separator: ",")
                FileHandle.standardError.write(
                    Data(
                        "warning: ignoring OLMLX_KV_CACHE_QUANT='\(v)' — expected <kind>:<bits> with kind in {\(kinds)} and bits in {\(bits)}\n"
                            .utf8))
                self.kvCacheQuant = nil
            }
        } else {
            self.kvCacheQuant = nil
        }

        self.speculative = env["\(prefix)SPECULATIVE"]?.lowercased() == "true"

        if let v = env["\(prefix)SPECULATIVE_STRATEGY"], let s = SpeculativeStrategy(rawValue: v) {
            self.speculativeStrategy = s
        } else {
            self.speculativeStrategy = .classic
        }

        if let v = env["\(prefix)SPECULATIVE_DRAFT_MODEL"] {
            let stripped = v.trimmingCharacters(in: .whitespaces)
            self.speculativeDraftModel = stripped.isEmpty ? nil : stripped
        } else {
            self.speculativeDraftModel = nil
        }

        if let v = env["\(prefix)SPECULATIVE_TOKENS"], let t = Int(v), t > 0 {
            self.speculativeTokens = t
        } else {
            self.speculativeTokens = nil
        }

        if let v = env["\(prefix)MAX_REQUEST_BODY_SIZE"], let valid = validateByteCountFormat(v) {
            self.maxRequestBodySize = valid
        } else {
            self.maxRequestBodySize = "16mb"
        }
    }
}
