import Foundation
import HuggingFace

/// Models are addressed by HuggingFace path; `ModelStore` knows where they live
/// on disk. It holds no mutable state — all methods are pure functions over
/// `modelsDir` and the registry — so the type can be plain `Sendable` rather
/// than an actor.
public final class ModelStore: Sendable {
    private let modelsDir: URL
    private let registry: ModelRegistry
    private let hubHost: URL

    public init(
        modelsDir: URL,
        registry: ModelRegistry,
        hubHost: URL = URL(string: "https://huggingface.co")!
    ) {
        self.modelsDir = modelsDir
        self.registry = registry
        self.hubHost = hubHost
    }

    public func localPath(for hfPath: String) -> URL {
        let safeName = hfPath.replacingOccurrences(of: "/", with: "--")
        return modelsDir.appendingPathComponent(safeName)
    }

    /// Returns the local snapshot path for `hfPath`, downloading from
    /// HuggingFace Hub if it is not already present on disk. Progress is
    /// reported via `progressHandler` if supplied.
    ///
    /// `hfPath` must be in `"namespace/name"` form. Throws
    /// `ModelStoreError.invalidHFPath` otherwise.
    public func ensureDownloaded(
        hfPath: String,
        progressHandler: (@MainActor @Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let path = localPath(for: hfPath)
        if FileManager.default.fileExists(atPath: path.path) {
            return path
        }
        return try await download(hfPath: hfPath, to: path, progressHandler: progressHandler)
    }

    /// Downloads the full snapshot of `hfPath` into `destination`, replacing
    /// any partial download.
    public func download(
        hfPath: String,
        to destination: URL? = nil,
        progressHandler: (@MainActor @Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let parts = hfPath.split(separator: "/", maxSplits: 1).map(String.init)
        guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
            throw ModelStoreError.invalidHFPath(hfPath)
        }
        let target = destination ?? localPath(for: hfPath)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)

        let repoID = Repo.ID(namespace: parts[0], name: parts[1])
        let client = HubClient(host: hubHost)
        do {
            _ = try await client.downloadSnapshot(
                of: repoID,
                to: target,
                progressHandler: progressHandler
            )
        } catch {
            // Clean up half-written directory so the next attempt starts fresh.
            try? FileManager.default.removeItem(at: target)
            throw ModelStoreError.downloadFailed(hfPath, error)
        }
        return target
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
    case invalidHFPath(String)
    case downloadFailed(String, any Error)
}
