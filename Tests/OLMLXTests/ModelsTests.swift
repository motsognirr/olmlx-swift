import Foundation
import OLMLX
import OLMLXTestUtils

func OLMLXTests_manifest_defaults() throws {
    let m = ModelManifest(name: "test", hfPath: "org/model")
    try expectEqual(m.name, "test")
    try expectEqual(m.hfPath, "org/model")
    try expectEqual(m.size, 0)
    try expectEqual(m.format, "mlx")
    try expectEqual(m.family, "")
}

func OLMLXTests_manifest_saveLoadRoundtrip() throws {
    let tmpDir = FileManager.default.temporaryDirectory
    let manifestDir = tmpDir.appendingPathComponent("olmlx-test-\(UUID().uuidString)")
    let manifestPath = manifestDir.appendingPathComponent("manifest.json")
    defer { try? FileManager.default.removeItem(at: manifestDir) }

    let m = ModelManifest(name: "llama2", hfPath: "meta-llama/Llama-2-7b",
                          size: 1000, digest: "abc123", family: "llama",
                          parameterSize: "7B", quantizationLevel: "Q4")
    try m.save(to: manifestPath)
    let loaded = try ModelManifest.load(from: manifestPath)
    try expectEqual(loaded.name, "llama2")
    try expectEqual(loaded.hfPath, "meta-llama/Llama-2-7b")
    try expectEqual(loaded.size, 1000)
    try expectEqual(loaded.digest, "abc123")
    try expectEqual(loaded.family, "llama")
}

func OLMLXTests_manifest_computeDigest() throws {
    let digest = ModelManifest.computeDigest(name: "test-model")
    try expect(digest.hasPrefix("sha256:"), "digest should start with sha256:")
    try expectEqual(digest.count, 19) // "sha256:" + 12 hex chars
}

func OLMLXTests_registry_normalizeName() throws {
    try expectEqual(ModelRegistry.normalizeName("llama2"), "llama2:latest")
    try expectEqual(ModelRegistry.normalizeName("llama2:7b"), "llama2:7b")
}

func OLMLXTests_registry_addAndResolve() throws {
    let tmpDir = FileManager.default.temporaryDirectory
    let configPath = tmpDir.appendingPathComponent("test-models.json")
    let registry = ModelRegistry(configPath: configPath)

    registry.addMapping(name: "test-model", config: ModelConfig(hfPath: "org/test"))
    let resolved = registry.resolve("test-model")
    try expectNotNil(resolved)
    try expectEqual(resolved?.hfPath, "org/test")
}

func OLMLXTests_registry_search() throws {
    let tmpDir = FileManager.default.temporaryDirectory
    let configPath = tmpDir.appendingPathComponent("test-models.json")
    let registry = ModelRegistry(configPath: configPath)

    registry.addMapping(name: "llama2", config: ModelConfig(hfPath: "meta/llama2"))
    registry.addMapping(name: "qwen3", config: ModelConfig(hfPath: "qwen/qwen3"))
    registry.addMapping(name: "gemma", config: ModelConfig(hfPath: "google/gemma"))

    let results = registry.search("llama")
    try expectEqual(results.count, 1)
    try expect(results.contains("llama2:latest"), "should find llama2:latest")
}

func OLMLXTests_registry_remove() throws {
    let tmpDir = FileManager.default.temporaryDirectory
    let configPath = tmpDir.appendingPathComponent("test-models.json")
    let registry = ModelRegistry(configPath: configPath)

    registry.addMapping(name: "test-model", config: ModelConfig(hfPath: "org/test"))
    try expectNotNil(registry.resolve("test-model"))

    registry.remove("test-model")
    try expectNil(registry.resolve("test-model"))
}

func OLMLXTests_store_listLocalEmpty() throws {
    let tmpDir = FileManager.default.temporaryDirectory
    let emptyDir = tmpDir.appendingPathComponent("olmlx-empty-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: emptyDir) }

    let registry = ModelRegistry(configPath: emptyDir.appendingPathComponent("models.json"))
    let store = ModelStore(modelsDir: emptyDir, registry: registry)

    let models = try store.listLocal()
    try expectEqual(models.count, 0)
}

func OLMLXTests_store_deleteModelNotFound() throws {
    let tmpDir = FileManager.default.temporaryDirectory
    let modelsDir = tmpDir.appendingPathComponent("olmlx-del-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: modelsDir) }

    let registry = ModelRegistry(configPath: modelsDir.appendingPathComponent("models.json"))
    let store = ModelStore(modelsDir: modelsDir, registry: registry)

    do {
        try store.delete(name: "nonexistent")
        throw TestError("expected error for nonexistent model")
    } catch is ModelStoreError {
    }
}

func OLMLXTests_store_hasBlob() throws {
    let tmpDir = FileManager.default.temporaryDirectory
    let modelsDir = tmpDir.appendingPathComponent("olmlx-blob-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: modelsDir) }

    let registry = ModelRegistry(configPath: modelsDir.appendingPathComponent("models.json"))
    let store = ModelStore(modelsDir: modelsDir, registry: registry)

    try expectEqual(store.hasBlob(digest: "abc"), false)
    try store.saveBlob(digest: "abc", data: "hello".data(using: .utf8)!)
    try expectEqual(store.hasBlob(digest: "abc"), true)
}
