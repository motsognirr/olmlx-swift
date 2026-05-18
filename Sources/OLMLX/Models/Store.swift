import Foundation

/// Models are addressed by HuggingFace path; `ModelStore` knows where they live
/// on disk. It holds no mutable state — all methods are pure functions over
/// `modelsDir` and the registry — so the type can be plain `Sendable` rather
/// than an actor.
public final class ModelStore: Sendable {
    private let modelsDir: URL
    private let registry: ModelRegistry

    public init(modelsDir: URL, registry: ModelRegistry) {
        self.modelsDir = modelsDir
        self.registry = registry
    }

    public func localPath(for hfPath: String) -> URL {
        let safeName = hfPath.replacingOccurrences(of: "/", with: "--")
        return modelsDir.appendingPathComponent(safeName)
    }

    public func ensureDownloaded(hfPath: String) async throws -> URL {
        let path = localPath(for: hfPath)
        if FileManager.default.fileExists(atPath: path.path) {
            return path
        }
        throw ModelStoreError.notDownloaded(hfPath)
    }

    public func listLocal() throws -> [ModelManifest] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: modelsDir.path) else { return [] }

        let contents = try fm.contentsOfDirectory(at: modelsDir, includingPropertiesForKeys: nil)
        var manifests: [ModelManifest] = []

        for dir in contents {
            let manifestPath = dir.appendingPathComponent("manifest.json")
            if fm.fileExists(atPath: manifestPath.path),
                let manifest = try? ModelManifest.load(from: manifestPath)
            {
                manifests.append(manifest)
            }
        }
        return manifests
    }

    public func show(name: String) async throws -> ModelManifest? {
        guard let config = await registry.resolve(name) else { return nil }
        let path = localPath(for: config.hfPath)
        let manifestPath = path.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestPath.path) else { return nil }
        return try ModelManifest.load(from: manifestPath)
    }

    public func delete(name: String) async throws {
        guard let config = await registry.resolve(name) else {
            throw ModelStoreError.modelNotFound(name)
        }
        let path = localPath(for: config.hfPath)
        if FileManager.default.fileExists(atPath: path.path) {
            try FileManager.default.removeItem(at: path)
        }
    }

    public func hasBlob(digest: String) -> Bool {
        let blobPath = modelsDir.appendingPathComponent("blobs").appendingPathComponent(digest)
        return FileManager.default.fileExists(atPath: blobPath.path)
    }

    public func saveBlob(digest: String, data: Data) throws {
        let blobDir = modelsDir.appendingPathComponent("blobs")
        try FileManager.default.createDirectory(at: blobDir, withIntermediateDirectories: true)
        try data.write(to: blobDir.appendingPathComponent(digest))
    }
}

public enum ModelStoreError: Error, Sendable {
    case notDownloaded(String)
    case modelNotFound(String)
}
