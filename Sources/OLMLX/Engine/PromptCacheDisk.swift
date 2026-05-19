import Foundation
import MLXLMCommon

actor PromptCacheDiskStore {
    private let baseURL: URL
    private let maxBytes: UInt64
    private let fileManager: FileManager

    init(baseURL: URL, maxGB: Double, fileManager: FileManager = .default) {
        self.baseURL = baseURL
        self.maxBytes = UInt64(maxGB * 1_073_741_824)
        self.fileManager = fileManager
    }

    private func directoryURL(for key: String) -> URL {
        baseURL.appendingPathComponent(sanitizeFilename(key))
    }

    private func keyURL(for key: String) -> URL {
        directoryURL(for: key).appendingPathComponent("key.txt")
    }

    private func tokensURL(for key: String) -> URL {
        directoryURL(for: key).appendingPathComponent("tokens.json")
    }

    private func cacheURL(for key: String) -> URL {
        directoryURL(for: key).appendingPathComponent("cache.safetensors")
    }

    func save(key: String, state: CachedPromptState) async {
        guard let cache = state.cache, !cache.isEmpty else { return }

        let dir = directoryURL(for: key)
        do {
            try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)

            try key.data(using: .utf8)?.write(to: keyURL(for: key), options: .atomic)

            let tokenData = try JSONEncoder().encode(state.tokens)
            try tokenData.write(to: tokensURL(for: key), options: .atomic)

            try savePromptCache(url: cacheURL(for: key), cache: cache)

            await evictIfNeeded()
        } catch {
            FileHandle.standardError.write(
                Data("warning: failed to save prompt cache for '\(key)': \(error)\n".utf8))
            try? fileManager.removeItem(at: dir)
        }
    }

    func load(key: String) async -> CachedPromptState? {
        let dir = directoryURL(for: key)
        let tokensFile = tokensURL(for: key)
        let cacheFile = cacheURL(for: key)

        guard fileManager.fileExists(atPath: tokensFile.path),
            fileManager.fileExists(atPath: cacheFile.path)
        else {
            return nil
        }

        do {
            let tokenData = try Data(contentsOf: tokensFile)
            let tokens = try JSONDecoder().decode([Int].self, from: tokenData)

            let (loadedCache, _) = try loadPromptCache(url: cacheFile)

            touch(dir: dir)
            return CachedPromptState(tokens: tokens, cache: loadedCache)
        } catch {
            FileHandle.standardError.write(
                Data("warning: failed to load prompt cache for '\(key)': \(error)\n".utf8))
            try? fileManager.removeItem(at: dir)
            return nil
        }
    }

    func loadAll() async -> [(key: String, state: CachedPromptState)] {
        guard fileManager.fileExists(atPath: baseURL.path) else { return [] }

        guard let entries = try? fileManager.contentsOfDirectory(at: baseURL, includingPropertiesForKeys: nil)
        else { return [] }

        var results: [(key: String, state: CachedPromptState)] = []
        for url in entries where url.hasDirectoryPath {
            let keyFile = url.appendingPathComponent("key.txt")
            guard
                let originalKey = try? String(contentsOf: keyFile, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                !originalKey.isEmpty
            else {
                continue
            }
            if let state = await load(key: originalKey) {
                results.append((originalKey, state))
            }
        }
        return results
    }

    func remove(key: String) async {
        let dir = directoryURL(for: key)
        try? fileManager.removeItem(at: dir)
    }

    func clear() async {
        try? fileManager.removeItem(at: baseURL)
    }

    private struct DiskEntry {
        let url: URL
        let size: UInt64
        let date: Date
    }

    private func evictIfNeeded() async {
        guard
            let entries = try? fileManager.contentsOfDirectory(
                at: baseURL,
                includingPropertiesForKeys: [.contentModificationDateKey])
        else { return }

        var totalSize: UInt64 = 0
        var entryInfo: [DiskEntry] = []

        for url in entries where url.hasDirectoryPath {
            let size = directorySize(url)
            let date = modificationDate(url)
            totalSize += size
            entryInfo.append(DiskEntry(url: url, size: size, date: date))
        }

        guard totalSize > maxBytes else { return }

        let sorted = entryInfo.sorted { $0.date < $1.date }
        for entry in sorted {
            if totalSize <= maxBytes { break }
            try? fileManager.removeItem(at: entry.url)
            totalSize -= entry.size
        }
    }

    private func directorySize(_ url: URL) -> UInt64 {
        guard let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey])
        else { return 0 }
        var total: UInt64 = 0
        for case let fileURL as URL in enumerator {
            total += UInt64((try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    private func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? Date()
    }

    private func touch(dir: URL) {
        var components = dir
        components.appendPathComponent(".touch")
        try? Data().write(to: components)
        try? fileManager.removeItem(at: components)
    }
}

private func sanitizeFilename(_ name: String) -> String {
    let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: ".-_"))
    let sanitized = name.components(separatedBy: allowed.inverted).joined(separator: "_")
    return sanitized.isEmpty ? "unknown" : sanitized
}
