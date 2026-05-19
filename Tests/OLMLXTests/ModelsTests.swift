import Foundation
import Testing

@testable import OLMLX

@Suite("Model Manifest")
struct ModelManifestTests {
    @Test func defaults() {
        let m = ModelManifest(name: "test", hfPath: "org/model")
        #expect(m.name == "test")
        #expect(m.hfPath == "org/model")
        #expect(m.size == 0)
        #expect(m.format == "mlx")
        #expect(m.family == "")
    }

    @Test func saveLoadRoundtrip() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let manifestDir = tmpDir.appendingPathComponent("olmlx-test-\(UUID().uuidString)")
        let manifestPath = manifestDir.appendingPathComponent("manifest.json")
        defer { try? FileManager.default.removeItem(at: manifestDir) }

        let m = ModelManifest(
            name: "llama2", hfPath: "meta-llama/Llama-2-7b",
            size: 1000, digest: "abc123", family: "llama",
            parameterSize: "7B", quantizationLevel: "Q4")
        try m.save(to: manifestPath)
        let loaded = try ModelManifest.load(from: manifestPath)
        #expect(loaded.name == "llama2")
        #expect(loaded.hfPath == "meta-llama/Llama-2-7b")
        #expect(loaded.size == 1000)
        #expect(loaded.digest == "abc123")
        #expect(loaded.family == "llama")
    }

    @Test func computeDigest() {
        let digest = ModelManifest.computeDigest(name: "test-model")
        #expect(digest.hasPrefix("sha256:"))
        #expect(digest.count == 19)  // "sha256:" + 12 hex chars
    }
}

@Suite("Model Registry")
struct ModelRegistryTests {
    @Test func normalizeName() {
        #expect(ModelRegistry.normalizeName("llama2") == "llama2:latest")
        #expect(ModelRegistry.normalizeName("llama2:7b") == "llama2:7b")
    }

    @Test func addAndResolve() async {
        let tmpDir = FileManager.default.temporaryDirectory
        let configPath = tmpDir.appendingPathComponent("test-models.json")
        let registry = ModelRegistry(configPath: configPath)

        await registry.addMapping(name: "test-model", config: ModelConfig(hfPath: "org/test"))
        let resolved = await registry.resolve("test-model")
        #expect(resolved != nil)
        #expect(resolved?.hfPath == "org/test")
    }

    @Test func search() async {
        let tmpDir = FileManager.default.temporaryDirectory
        let configPath = tmpDir.appendingPathComponent("test-models.json")
        let registry = ModelRegistry(configPath: configPath)

        await registry.addMapping(name: "llama2", config: ModelConfig(hfPath: "meta/llama2"))
        await registry.addMapping(name: "qwen3", config: ModelConfig(hfPath: "qwen/qwen3"))
        await registry.addMapping(name: "gemma", config: ModelConfig(hfPath: "google/gemma"))

        let results = await registry.search("llama")
        #expect(results.count == 1)
        #expect(results.contains("llama2:latest"))
    }

    @Test func remove() async {
        let tmpDir = FileManager.default.temporaryDirectory
        let configPath = tmpDir.appendingPathComponent("test-models.json")
        let registry = ModelRegistry(configPath: configPath)

        await registry.addMapping(name: "test-model", config: ModelConfig(hfPath: "org/test"))
        #expect(await registry.resolve("test-model") != nil)

        await registry.remove("test-model")
        #expect(await registry.resolve("test-model") == nil)
    }
}

@Suite("Model Store")
struct ModelStoreTests {
    @Test func listLocalEmpty() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let emptyDir = tmpDir.appendingPathComponent("olmlx-empty-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: emptyDir) }

        let registry = ModelRegistry(configPath: emptyDir.appendingPathComponent("models.json"))
        let store = ModelStore(modelsDir: emptyDir, registry: registry)

        #expect(try store.listLocal().count == 0)
    }

    @Test func deleteModelNotFound() async {
        let tmpDir = FileManager.default.temporaryDirectory
        let modelsDir = tmpDir.appendingPathComponent("olmlx-del-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: modelsDir) }

        let registry = ModelRegistry(configPath: modelsDir.appendingPathComponent("models.json"))
        let store = ModelStore(modelsDir: modelsDir, registry: registry)

        await #expect(throws: ModelStoreError.self) {
            try await store.delete(name: "nonexistent")
        }
    }

    @Test func hasBlob() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let modelsDir = tmpDir.appendingPathComponent("olmlx-blob-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: modelsDir) }

        let registry = ModelRegistry(configPath: modelsDir.appendingPathComponent("models.json"))
        let store = ModelStore(modelsDir: modelsDir, registry: registry)

        #expect(store.hasBlob(digest: "abc") == false)
        try store.saveBlob(digest: "abc", data: Data("hello".utf8))
        #expect(store.hasBlob(digest: "abc") == true)
    }

    @Test func downloadRejectsInvalidHFPath() async {
        let tmpDir = FileManager.default.temporaryDirectory
        let modelsDir = tmpDir.appendingPathComponent("olmlx-dl-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: modelsDir) }

        let registry = ModelRegistry(configPath: modelsDir.appendingPathComponent("models.json"))
        let store = ModelStore(modelsDir: modelsDir, registry: registry)

        // Single-segment paths are not valid HuggingFace "namespace/name" paths.
        await #expect(throws: ModelStoreError.self) {
            _ = try await store.download(hfPath: "no-slash")
        }
        await #expect(throws: ModelStoreError.self) {
            _ = try await store.download(hfPath: "")
        }
    }

    @Test func snapshotCompletenessShardedComplete() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("olmlx-snap-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let index: [String: Any] = [
            "metadata": ["total_size": 1],
            "weight_map": [
                "language_model.lm_head.weight": "model-00002-of-00002.safetensors",
                "language_model.model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            ],
        ]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        try Data().write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try Data().write(to: dir.appendingPathComponent("model-00002-of-00002.safetensors"))

        #expect(ModelStore.isSnapshotComplete(at: dir) == true)
    }

    @Test func snapshotCompletenessShardedMissing() throws {
        // Regression for issue #60: metadata-only directory (interrupted
        // download) must be reported as incomplete so callers re-download.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("olmlx-snap-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let index: [String: Any] = [
            "weight_map": [
                "language_model.lm_head.weight": "model-00004-of-00004.safetensors",
                "language_model.model.embed_tokens.weight": "model-00001-of-00004.safetensors",
            ]
        ]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: dir.appendingPathComponent("model.safetensors.index.json"))

        #expect(ModelStore.isSnapshotComplete(at: dir) == false)
    }

    @Test func snapshotCompletenessSingleFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("olmlx-snap-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(ModelStore.isSnapshotComplete(at: dir) == false)

        try Data().write(to: dir.appendingPathComponent("model.safetensors"))
        #expect(ModelStore.isSnapshotComplete(at: dir) == true)
    }
}
