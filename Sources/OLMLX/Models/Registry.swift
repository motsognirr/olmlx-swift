import Foundation

public struct SpeculativeConfig: Sendable {
    public var enabled: Bool
    public var draftModel: String?
    public var numTokens: Int?
    public var strategy: SpeculativeStrategy

    public init(
        enabled: Bool = false, draftModel: String? = nil,
        numTokens: Int? = nil, strategy: SpeculativeStrategy = .classic
    ) {
        self.enabled = enabled
        self.draftModel = draftModel
        self.numTokens = numTokens
        self.strategy = strategy
    }
}

public struct ModelConfig: Codable, Sendable {
    public var hfPath: String
    public var keepAlive: String?
    public var options: ModelOptions?
    public var syncMode: SyncMode = .full
    public var speculative: Bool = false
    public var speculativeDraftModel: String?
    public var speculativeTokens: Int?
    public var speculativeStrategy: SpeculativeStrategy = .classic
    public var kvCacheQuant: String?
    public var experimental: [String: AnyCodable]?

    private enum CodingKeys: String, CodingKey {
        case options, speculative, experimental
        case hfPath = "hf_path"
        case keepAlive = "keep_alive"
        case syncMode = "sync_mode"
        case speculativeDraftModel = "speculative_draft_model"
        case speculativeTokens = "speculative_tokens"
        case speculativeStrategy = "speculative_strategy"
        case kvCacheQuant = "kv_cache_quant"
    }

    public init(
        hfPath: String, keepAlive: String? = nil, options: ModelOptions? = nil,
        syncMode: SyncMode = .full, speculative: Bool = false,
        speculativeDraftModel: String? = nil, speculativeTokens: Int? = nil,
        speculativeStrategy: SpeculativeStrategy = .classic,
        kvCacheQuant: String? = nil, experimental: [String: AnyCodable]? = nil
    ) {
        self.hfPath = hfPath
        self.keepAlive = keepAlive
        self.options = options
        self.syncMode = syncMode
        self.speculative = speculative
        self.speculativeDraftModel = speculativeDraftModel
        self.speculativeTokens = speculativeTokens
        self.speculativeStrategy = speculativeStrategy
        self.kvCacheQuant = kvCacheQuant
        self.experimental = experimental
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hfPath = try container.decode(String.self, forKey: .hfPath)
        keepAlive = try container.decodeIfPresent(String.self, forKey: .keepAlive)
        options = try container.decodeIfPresent(ModelOptions.self, forKey: .options)
        syncMode = try container.decodeIfPresent(SyncMode.self, forKey: .syncMode) ?? .full
        speculative = try container.decodeIfPresent(Bool.self, forKey: .speculative) ?? false
        speculativeDraftModel = try container.decodeIfPresent(String.self, forKey: .speculativeDraftModel)
        speculativeTokens = try container.decodeIfPresent(Int.self, forKey: .speculativeTokens)
        speculativeStrategy =
            try container.decodeIfPresent(SpeculativeStrategy.self, forKey: .speculativeStrategy) ?? .classic
        kvCacheQuant = try container.decodeIfPresent(String.self, forKey: .kvCacheQuant)
        experimental = try container.decodeIfPresent([String: AnyCodable].self, forKey: .experimental)
    }

    public func resolvedSpeculative() -> SpeculativeConfig {
        let globalSettings = Settings.shared
        let enabled = speculative || globalSettings.speculative
        let draft = speculativeDraftModel ?? globalSettings.speculativeDraftModel
        let tokens = speculativeTokens ?? globalSettings.speculativeTokens
        let strategy = speculativeStrategy
        return SpeculativeConfig(enabled: enabled, draftModel: draft, numTokens: tokens, strategy: strategy)
    }

    public func resolvedKVCacheQuant() -> String? {
        return kvCacheQuant ?? Settings.shared.kvCacheQuant
    }
}

public final class ModelRegistry: @unchecked Sendable {
    private var mappings: [String: ModelConfig] = [:]
    private var aliases: [String: String] = [:]
    private let configPath: URL
    private let lock = NSLock()

    public init(configPath: URL) {
        self.configPath = configPath
    }

    public static func normalizeName(_ name: String) -> String {
        if name.contains(":") { return name }
        return name + ":latest"
    }

    public func load() throws {
        lock.lock()
        defer { lock.unlock() }

        guard FileManager.default.fileExists(atPath: configPath.path) else { return }

        let data = try Data(contentsOf: configPath)

        if let json = try? JSONDecoder().decode([String: ModelConfigOrAlias].self, from: data) {
            for (key, value) in json {
                switch value {
                case .config(let config):
                    mappings[key] = config
                case .alias(let target):
                    aliases[key] = target
                case .string(let hfPath):
                    mappings[key] = ModelConfig(hfPath: hfPath)
                }
            }
        } else if let simpleJson = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            for (key, value) in simpleJson {
                if let str = value as? String {
                    mappings[key] = ModelConfig(hfPath: str)
                } else if let dict = value as? [String: Any],
                    let hfPath = dict["hf_path"] as? String
                {
                    mappings[key] = ModelConfig(hfPath: hfPath)
                }
            }
        }
    }

    public func save() throws {
        lock.lock()
        defer { lock.unlock() }

        let dir = configPath.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        var output: [String: Any] = [:]
        for (name, config) in mappings {
            output[name] = configToDict(config)
        }
        for (alias, target) in aliases {
            output[alias] = target
        }

        let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: configPath)
    }

    private func configToDict(_ config: ModelConfig) -> [String: Any] {
        var dict: [String: Any] = ["hf_path": config.hfPath]
        if let v = config.keepAlive { dict["keep_alive"] = v }
        if config.speculative { dict["speculative"] = true }
        if let v = config.speculativeDraftModel { dict["speculative_draft_model"] = v }
        if let v = config.speculativeTokens { dict["speculative_tokens"] = v }
        if let v = config.kvCacheQuant { dict["kv_cache_quant"] = v }
        return dict
    }

    public func resolve(_ name: String) -> ModelConfig? {
        lock.lock()
        defer { lock.unlock() }
        let normalized = Self.normalizeName(name)
        if let config = mappings[normalized] { return config }
        if let target = aliases[normalized] { return mappings[target] }
        return mappings[name] ?? mappings[Self.normalizeName(name)]
    }

    public func listModels() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return Array(mappings.keys).sorted()
    }

    public func search(_ query: String) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        let lowerQ = query.lowercased()
        return mappings.keys.filter { $0.lowercased().contains(lowerQ) }.sorted()
    }

    public func addMapping(name: String, config: ModelConfig) {
        lock.lock()
        defer { lock.unlock() }
        mappings[Self.normalizeName(name)] = config
    }

    public func addAlias(alias: String, target: String) {
        lock.lock()
        defer { lock.unlock() }
        aliases[Self.normalizeName(alias)] = target
    }

    public func remove(_ name: String) {
        lock.lock()
        defer { lock.unlock() }
        let normalized = Self.normalizeName(name)
        mappings.removeValue(forKey: normalized)
        aliases.removeValue(forKey: normalized)
    }
}

enum ModelConfigOrAlias: Codable {
    case config(ModelConfig)
    case alias(String)
    case string(String)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .string(str)
        } else if let config = try? container.decode(ModelConfig.self) {
            self = .config(config)
        } else {
            self = .string("")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .config(let c): try container.encode(c)
        case .alias(let a): try container.encode(a)
        case .string(let s): try container.encode(s)
        }
    }
}
