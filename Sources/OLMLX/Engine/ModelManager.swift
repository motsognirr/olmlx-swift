import Foundation
import MLXLMCommon

public struct LoadedModel: Sendable {
    public var name: String
    public var hfPath: String
    public var isVLM: Bool = false
    public var templateCaps: TemplateCaps
    public var config: ModelConfig
    public var loadedAt: Date = Date()
    public var expiresAt: Date?
    public var container: ModelContainer?

    public init(
        name: String, hfPath: String, isVLM: Bool = false,
        templateCaps: TemplateCaps = TemplateCaps(),
        config: ModelConfig, expiresAt: Date? = nil,
        container: ModelContainer? = nil
    ) {
        self.name = name
        self.hfPath = hfPath
        self.isVLM = isVLM
        self.templateCaps = templateCaps
        self.config = config
        self.loadedAt = Date()
        self.expiresAt = expiresAt
        self.container = container
    }
}

/// In-memory record for a model's prompt cache slot.
///
/// `tokens` is the full chat-templated prompt tokenization from the most recent
/// generation (not including the generated continuation). `cache` is the MLX
/// `[KVCache]` whose offset advances past `tokens` once the generated tokens
/// have been appended during decoding — i.e. `cache[0].offset >= tokens.count`.
///
/// Holds non-`Sendable` `KVCache` instances; access is gated by `PromptCacheStore`'s
/// actor isolation and by `ModelContainer.perform`'s serial access during use.
public final class CachedPromptState: @unchecked Sendable {
    public var tokens: [Int]
    public var cache: [KVCache]?

    public init(tokens: [Int], cache: [KVCache]? = nil) {
        self.tokens = tokens
        self.cache = cache
    }
}

/// Length of the longest shared prefix between two integer sequences.
public func longestCommonPrefix(_ a: [Int], _ b: [Int]) -> Int {
    let n = min(a.count, b.count)
    var i = 0
    while i < n && a[i] == b[i] { i += 1 }
    return i
}

public enum ModelLoadError: Error, Sendable {
    case notFound(String)
    case loadTimeout(String)
    case serverBusy
    case inferenceError(String)
}

public func parseKeepAlive(_ value: String) -> TimeInterval? {
    let lower = value.lowercased()
    if lower == "0" || lower == "0s" || lower == "0m" || lower == "0h" { return 0 }
    if let num = Double(lower.filter { $0.isNumber || $0 == "." }) {
        if lower.hasSuffix("s") { return num }
        if lower.hasSuffix("m") { return num * 60 }
        if lower.hasSuffix("h") { return num * 3600 }
    }
    return nil
}

public actor PromptCacheStore {
    private var store: [String: CachedPromptState] = [:]
    private let maxSlots: Int
    private var accessOrder: [String] = []

    public init(maxSlots: Int = 4) {
        self.maxSlots = maxSlots
    }

    public func get(key: String) -> CachedPromptState? {
        return store[key]
    }

    public func set(key: String, state: CachedPromptState) {
        if store[key] == nil && store.count >= maxSlots {
            if let oldest = accessOrder.first {
                store.removeValue(forKey: oldest)
                accessOrder.removeFirst()
            }
        }
        store[key] = state
        accessOrder.removeAll { $0 == key }
        accessOrder.append(key)
    }

    public func remove(key: String) {
        store.removeValue(forKey: key)
        accessOrder.removeAll { $0 == key }
    }

    public func clear() {
        store.removeAll()
        accessOrder.removeAll()
    }
}

public actor ModelManager {
    public let registry: ModelRegistry
    public let store: ModelStore
    public nonisolated let settings: Settings
    private var loadedModels: [String: LoadedModel] = [:]
    private var loadedDraftModels: [String: ModelContainer] = [:]
    private let maxLoaded: Int
    public let promptCache: PromptCacheStore
    public private(set) var inferenceEngine: (any InferenceEngineProtocol)?

    public init(
        registry: ModelRegistry,
        store: ModelStore,
        settings: Settings = Settings(),
        maxLoadedModels: Int? = nil,
        inferenceEngine: (any InferenceEngineProtocol)? = nil
    ) {
        self.registry = registry
        self.store = store
        self.settings = settings
        self.maxLoaded = maxLoadedModels ?? settings.maxLoadedModels
        self.promptCache = PromptCacheStore(maxSlots: settings.promptCacheMaxSlots)
        self.inferenceEngine = inferenceEngine
    }

    public func setInferenceEngine(_ engine: (any InferenceEngineProtocol)?) {
        self.inferenceEngine = engine
    }

    public func ensureLoaded(name: String, keepAlive: String? = nil) async throws -> LoadedModel {
        let normalized = ModelRegistry.normalizeName(name)

        if let model = loadedModels[normalized] {
            return model
        }

        guard let config = await registry.resolve(name) else {
            throw ModelLoadError.notFound(name)
        }

        let localPath = try await store.ensureDownloaded(hfPath: config.hfPath)
        let caps = detectCaps(tokenizerChatTemplate: nil)

        var model = LoadedModel(
            name: name,
            hfPath: config.hfPath,
            templateCaps: caps,
            config: config
        )

        if let engine = inferenceEngine {
            do {
                let container = try await engine.loadModel(from: localPath)
                model.container = container
            } catch {
                throw ModelLoadError.inferenceError("Failed to load MLX model: \(error)")
            }
        }

        let keepAliveStr = keepAlive ?? config.keepAlive ?? settings.defaultKeepAlive
        if let duration = parseKeepAlive(keepAliveStr) {
            model.expiresAt = duration > 0 ? Date().addingTimeInterval(duration) : nil
        }

        if loadedModels.count >= maxLoaded {
            if let oldest = loadedModels.min(by: { $0.value.loadedAt < $1.value.loadedAt }) {
                loadedModels.removeValue(forKey: oldest.key)
            }
        }
        loadedModels[normalized] = model
        return model
    }

    public func getLoaded() -> [LoadedModel] {
        return Array(loadedModels.values)
    }

    public func unload(name: String) {
        loadedModels.removeValue(forKey: ModelRegistry.normalizeName(name))
    }

    /// Loads (and caches) a draft model container by its HuggingFace path.
    ///
    /// Draft models are tracked separately from the main `loadedModels` table so a
    /// speculative-decoding request does not count against `maxLoadedModels` and
    /// does not evict the main model it pairs with. Subsequent calls with the same
    /// `hfPath` return the cached container.
    public func ensureDraftLoaded(hfPath: String) async throws -> ModelContainer {
        if let container = loadedDraftModels[hfPath] {
            return container
        }
        guard let engine = inferenceEngine else {
            throw ModelLoadError.inferenceError(
                "no inference engine configured for draft model \(hfPath)")
        }
        let localPath = try await store.ensureDownloaded(hfPath: hfPath)
        do {
            let container = try await engine.loadModel(from: localPath)
            loadedDraftModels[hfPath] = container
            return container
        } catch {
            throw ModelLoadError.inferenceError(
                "Failed to load draft MLX model \(hfPath): \(error)")
        }
    }

    public func unloadDraft(hfPath: String) {
        loadedDraftModels.removeValue(forKey: hfPath)
    }

    public func invalidatePromptCache(for model: String) async {
        await promptCache.remove(key: ModelRegistry.normalizeName(model))
    }

    /// Atomically fetches and removes a cache slot so the caller has exclusive
    /// access to the underlying `[KVCache]` while it mutates it. Pair with
    /// ``releasePromptCache(key:state:)``.
    public func acquirePromptCache(key: String) async -> CachedPromptState? {
        let state = await promptCache.get(key: key)
        if state != nil {
            await promptCache.remove(key: key)
        }
        return state
    }

    /// Returns a (possibly updated) cache slot to the store, or drops it when
    /// `state` is `nil` (e.g. the cache exceeded the configured token budget).
    public func releasePromptCache(key: String, state: CachedPromptState?) async {
        if let state {
            await promptCache.set(key: key, state: state)
        }
    }

    public nonisolated func startExpiryChecker() -> Task<Void, Never> {
        return Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { break }
                await self.expireOnce()
                try? await Task.sleep(nanoseconds: 30_000_000_000)
            }
        }
    }

    private func expireOnce() {
        let now = Date()
        let expired = loadedModels.filter {
            if let exp = $0.value.expiresAt { return now >= exp }
            return false
        }
        for (name, _) in expired {
            loadedModels.removeValue(forKey: name)
        }
    }
}
