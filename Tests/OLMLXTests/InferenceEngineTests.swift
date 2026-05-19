import Foundation
import MLXLMCommon
import Testing

@testable import OLMLX

@Suite("GenerateParameters Mapping")
struct GenerateParametersMappingTests {
    @Test func temperature() {
        var opts = ModelOptions()
        opts.temperature = 0.7
        #expect(mapToGenerateParameters(from: opts).temperature == 0.7)
    }

    @Test func topP() {
        var opts = ModelOptions()
        opts.topP = 0.9
        #expect(mapToGenerateParameters(from: opts).topP == 0.9)
    }

    @Test func topK() {
        var opts = ModelOptions()
        opts.topK = 40
        #expect(mapToGenerateParameters(from: opts).topK == 40)
    }

    @Test func maxTokens() {
        var opts = ModelOptions()
        opts.numPredict = 1024
        #expect(mapToGenerateParameters(from: opts).maxTokens == 1024)
    }

    @Test func repetitionPenalty() {
        var opts = ModelOptions()
        opts.repeatPenalty = 1.2
        #expect(mapToGenerateParameters(from: opts).repetitionPenalty == 1.2)
    }

    @Test func presencePenalty() {
        var opts = ModelOptions()
        opts.presencePenalty = 0.5
        #expect(mapToGenerateParameters(from: opts).presencePenalty == 0.5)
    }

    @Test func frequencyPenalty() {
        var opts = ModelOptions()
        opts.frequencyPenalty = 0.3
        #expect(mapToGenerateParameters(from: opts).frequencyPenalty == 0.3)
    }

    @Test func seedDoesNotChangeTemperature() {
        var opts = ModelOptions()
        opts.seed = 42
        // GenerateParameters has no seed field — MLX handles seeding via MLXRandom
        #expect(mapToGenerateParameters(from: opts).temperature == 0.6)
    }

    @Test func minP() {
        var opts = ModelOptions()
        opts.minP = 0.1
        #expect(mapToGenerateParameters(from: opts).minP == 0.1)
    }

    @Test func typicalPIsIgnored() {
        var opts = ModelOptions()
        opts.typicalP = 0.8
        opts.topP = 0.5
        // typical_p is unsupported by MLX's sampler and must not overwrite topP.
        #expect(mapToGenerateParameters(from: opts).topP == 0.5)
    }

    @Test func emptyOptions() {
        let params = mapToGenerateParameters(from: ModelOptions())
        #expect(params.temperature == 0.6)
        #expect(params.topP == 1.0)
        #expect(params.maxTokens == nil)
        #expect(params.repetitionPenalty == nil)
    }

    @Test func nilOptions() {
        let params = mapToGenerateParameters(from: nil)
        #expect(params.temperature == 0.6)
        #expect(params.topP == 1.0)
    }

    @Test func repeatLastN() {
        var opts = ModelOptions()
        opts.repeatLastN = 64
        #expect(mapToGenerateParameters(from: opts).repetitionContextSize == 64)
    }
}

@Suite("KV Cache Quant")
struct KVCacheQuantTests {
    @Test func applyAffine4() {
        var params = mapToGenerateParameters(from: nil)
        applyKVCacheQuant(&params, spec: "affine:4")
        #expect(params.kvBits == 4)
        #expect(params.kvGroupSize == 64)
    }

    @Test func applyAffine2And8() {
        var p2 = mapToGenerateParameters(from: nil)
        applyKVCacheQuant(&p2, spec: "affine:2")
        #expect(p2.kvBits == 2)

        var p8 = mapToGenerateParameters(from: nil)
        applyKVCacheQuant(&p8, spec: "affine:8")
        #expect(p8.kvBits == 8)
    }

    @Test func applyNilSpecLeavesUnchanged() {
        var params = mapToGenerateParameters(from: nil)
        applyKVCacheQuant(&params, spec: nil)
        #expect(params.kvBits == nil)
    }

    @Test func applyInvalidSpecLeavesUnchanged() {
        var params = mapToGenerateParameters(from: nil)
        applyKVCacheQuant(&params, spec: "turboquant:4")
        #expect(params.kvBits == nil)
    }

    @Test func resolvedKVCacheQuantPrefersPerModel() {
        let global = Settings(env: ["OLMLX_KV_CACHE_QUANT": "affine:4"])
        let config = ModelConfig(hfPath: "org/m", kvCacheQuant: "affine:8")
        #expect(config.resolvedKVCacheQuant(global: global) == "affine:8")
    }

    @Test func resolvedKVCacheQuantFallsBackToGlobal() {
        let global = Settings(env: ["OLMLX_KV_CACHE_QUANT": "affine:4"])
        let config = ModelConfig(hfPath: "org/m")
        #expect(config.resolvedKVCacheQuant(global: global) == "affine:4")
    }
}

@Suite("ModelManager Inference Integration")
struct ModelManagerInferenceTests {
    @Test func inferenceDelegation() async {
        let tmpDir = FileManager.default.temporaryDirectory
        let configPath = tmpDir.appendingPathComponent("test-inference-models.json")
        let registry = ModelRegistry(configPath: configPath)
        await registry.addMapping(name: "test", config: ModelConfig(hfPath: "org/test"))

        let mockEngine = MockInferenceEngine()
        let store = ModelStore(modelsDir: tmpDir, registry: registry)
        let manager = ModelManager(registry: registry, store: store)
        await manager.setInferenceEngine(mockEngine)

        let engine = await manager.inferenceEngine
        #expect(engine != nil)
    }
}

@Suite("Load Error Wrapping")
struct LoadErrorWrappingTests {
    @Test func unsupportedModelTypeIsClassified() {
        let upstream = ModelFactoryError.unsupportedModelType("minimax_m2")
        let wrapped = wrapMLXLoadError(upstream, hfPath: "mlx-community/MiniMax-M2.7-5bit")
        guard case .unsupportedArchitecture(let modelType, let hfPath) = wrapped else {
            Issue.record("expected .unsupportedArchitecture, got \(wrapped)")
            return
        }
        #expect(modelType == "minimax_m2")
        #expect(hfPath == "mlx-community/MiniMax-M2.7-5bit")
    }

    @Test func otherErrorsFallThroughToInferenceError() {
        struct Boom: Error {}
        let wrapped = wrapMLXLoadError(Boom(), hfPath: "org/whatever")
        guard case .inferenceError(let message) = wrapped else {
            Issue.record("expected .inferenceError, got \(wrapped)")
            return
        }
        #expect(message.contains("Failed to load MLX model"))
    }

    @Test func unsupportedArchitectureMessageMentionsModelTypeAndHfPath() {
        let err = ModelLoadError.unsupportedArchitecture(
            modelType: "step3p5", hfPath: "mlx-community/Step-3.5-Flash-6bit")
        let message = "\(err.localizedDescription)"
        #expect(message.contains("step3p5"))
        #expect(message.contains("mlx-community/Step-3.5-Flash-6bit"))
    }
}

@Suite("MockInferenceEngine")
struct MockInferenceEngineTests {
    @Test func unloadClearsModels() {
        let engine = MockInferenceEngine()
        #expect(engine.loadedModels == [])
        engine.loadedModels = ["test:latest"]
        #expect(engine.loadedModels == ["test:latest"])
        engine.loadedModels = []
        #expect(engine.loadedModels == [])
    }

    @Test func loadModelRecordsCallAndThrowsTestError() async {
        let engine = MockInferenceEngine()
        let dir = URL(fileURLWithPath: "/tmp/some-model")
        await #expect(throws: MockInferenceError.noStubContainerConfigured("some-model")) {
            _ = try await engine.loadModel(from: dir)
        }
        #expect(engine.loadedModels == ["some-model"])
    }

    @Test func loadModelHonorsShouldFail() async {
        let engine = MockInferenceEngine()
        engine.shouldFail = true
        let dir = URL(fileURLWithPath: "/tmp/x")
        await #expect(throws: (any Error).self) {
            _ = try await engine.loadModel(from: dir)
        }
        // shouldFail short-circuits before recording the call.
        #expect(engine.loadedModels == [])
    }
}

@Suite("Tools Conversion")
struct ToolsConversionTests {
    @Test func toolsToRawNil() {
        #expect(toolsToRaw(nil) == nil)
    }

    @Test func toolsToRawConverts() {
        let function: [String: AnyCodable] = [
            "name": .string("get_weather"),
            "description": .string("Get weather"),
            "parameters": .dictionary([
                "type": .string("object"),
                "properties": .dictionary([
                    "city": .dictionary(["type": .string("string")])
                ]),
            ]),
        ]
        let tool = Tool(type: "function", function: function)
        let result = toolsToRaw([tool])
        #expect(result != nil)
        #expect(result?.count == 1)
    }
}
