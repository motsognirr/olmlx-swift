import Foundation
import OLMLX
import OLMLXTestUtils

// MARK: - GenerationParameters mapping tests

func OLMLXTests_inference_mapTemperature() throws {
    var opts = ModelOptions()
    opts.temperature = 0.7
    let params = mapToGenerateParameters(from: opts)
    try expectEqual(params.temperature, 0.7)
}

func OLMLXTests_inference_mapTopP() throws {
    var opts = ModelOptions()
    opts.topP = 0.9
    let params = mapToGenerateParameters(from: opts)
    try expectEqual(params.topP, 0.9)
}

func OLMLXTests_inference_mapTopK() throws {
    var opts = ModelOptions()
    opts.topK = 40
    let params = mapToGenerateParameters(from: opts)
    try expectEqual(params.topK, 40)
}

func OLMLXTests_inference_mapMaxTokens() throws {
    var opts = ModelOptions()
    opts.numPredict = 1024
    let params = mapToGenerateParameters(from: opts)
    try expectEqual(params.maxTokens, 1024)
}

func OLMLXTests_inference_mapRepetitionPenalty() throws {
    var opts = ModelOptions()
    opts.repeatPenalty = 1.2
    let params = mapToGenerateParameters(from: opts)
    try expectEqual(params.repetitionPenalty, 1.2)
}

func OLMLXTests_inference_mapPresencePenalty() throws {
    var opts = ModelOptions()
    opts.presencePenalty = 0.5
    let params = mapToGenerateParameters(from: opts)
    try expectEqual(params.presencePenalty, 0.5)
}

func OLMLXTests_inference_mapFrequencyPenalty() throws {
    var opts = ModelOptions()
    opts.frequencyPenalty = 0.3
    let params = mapToGenerateParameters(from: opts)
    try expectEqual(params.frequencyPenalty, 0.3)
}

func OLMLXTests_inference_mapSeed() throws {
    var opts = ModelOptions()
    opts.seed = 42
    let params = mapToGenerateParameters(from: opts)
    // GenerateParameters has no seed field — MLX handles seeding via MLXRandom
    try expectEqual(params.temperature, 0.6)
}

func OLMLXTests_inference_mapMinP() throws {
    var opts = ModelOptions()
    opts.minP = 0.1
    let params = mapToGenerateParameters(from: opts)
    try expectEqual(params.minP, 0.1)
}

func OLMLXTests_inference_mapTypicalP() throws {
    var opts = ModelOptions()
    opts.typicalP = 0.8
    let params = mapToGenerateParameters(from: opts)
    try expectEqual(params.topP, 0.8)
}

func OLMLXTests_inference_mapEmptyOptions() throws {
    let params = mapToGenerateParameters(from: ModelOptions())
    try expectEqual(params.temperature, 0.6)
    try expectEqual(params.topP, 1.0)
    try expectNil(params.maxTokens)
    try expectNil(params.repetitionPenalty)
}

func OLMLXTests_inference_mapNilOptions() throws {
    let params = mapToGenerateParameters(from: nil)
    try expectEqual(params.temperature, 0.6)
    try expectEqual(params.topP, 1.0)
}

func OLMLXTests_inference_mapRepeatLastN() throws {
    var opts = ModelOptions()
    opts.repeatLastN = 64
    let params = mapToGenerateParameters(from: opts)
    try expectEqual(params.repetitionContextSize, 64)
}

// MARK: - ModelManager inference integration tests

func OLMLXTests_manager_inferenceDelegation() throws {
    let tmpDir = FileManager.default.temporaryDirectory
    let configPath = tmpDir.appendingPathComponent("test-inference-models.json")
    let registry = ModelRegistry(configPath: configPath)
    registry.addMapping(name: "test", config: ModelConfig(hfPath: "org/test"))

    let mockEngine = MockInferenceEngine()
    let store = ModelStore(modelsDir: tmpDir, registry: registry)
    let manager = ModelManager(registry: registry, store: store)
    manager.inferenceEngine = mockEngine

    // We can't easily test async ensureLoaded in sync tests,
    // but we can verify the engine is set correctly.
    try expect(manager.inferenceEngine != nil)
}

// MARK: - MockEngine unload test

func OLMLXTests_mockEngine_unloadClearsModels() throws {
    let engine = MockInferenceEngine()
    try expectEqual(engine.loadedModels, [])
    engine.loadedModels = ["test:latest"]
    try expectEqual(engine.loadedModels, ["test:latest"])
    engine.loadedModels = []
    try expectEqual(engine.loadedModels, [])
}

// MARK: - Tools conversion tests

func OLMLXTests_toolsToRawNil() throws {
    let result = toolsToRaw(nil)
    try expect(result == nil)
}

func OLMLXTests_toolsToRawConverts() throws {
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
    try expect(result != nil)
    try expectEqual(result?.count, 1)
}
