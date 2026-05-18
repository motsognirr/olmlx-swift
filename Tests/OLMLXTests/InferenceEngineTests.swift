import Foundation
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

    @Test func typicalP() {
        var opts = ModelOptions()
        opts.typicalP = 0.8
        #expect(mapToGenerateParameters(from: opts).topP == 0.8)
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

@Suite("ModelManager Inference Integration")
struct ModelManagerInferenceTests {
    @Test func inferenceDelegation() {
        let tmpDir = FileManager.default.temporaryDirectory
        let configPath = tmpDir.appendingPathComponent("test-inference-models.json")
        let registry = ModelRegistry(configPath: configPath)
        registry.addMapping(name: "test", config: ModelConfig(hfPath: "org/test"))

        let mockEngine = MockInferenceEngine()
        let store = ModelStore(modelsDir: tmpDir, registry: registry)
        let manager = ModelManager(registry: registry, store: store)
        manager.inferenceEngine = mockEngine

        #expect(manager.inferenceEngine != nil)
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
                ])
            ])
        ]
        let tool = Tool(type: "function", function: function)
        let result = toolsToRaw([tool])
        #expect(result != nil)
        #expect(result?.count == 1)
    }
}
