import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import OLMLX

@Suite("Extensions/Gemma4Patched/Config")
struct Gemma4PatchedConfigTests {

    private func decodeText(_ json: String) throws -> Gemma4PatchedTextConfiguration {
        try JSONDecoder().decode(
            Gemma4PatchedTextConfiguration.self, from: Data(json.utf8))
    }

    @Test func moeFieldsDecodeWhenPresent() throws {
        let cfg = try decodeText(#"""
        {"model_type":"gemma4_text","enable_moe_block":true,
         "num_experts":128,"top_k_experts":8,"moe_intermediate_size":704}
        """#)
        #expect(cfg.enableMoeBlock == true)
        #expect(cfg.numExperts == 128)
        #expect(cfg.topKExperts == 8)
        #expect(cfg.moeIntermediateSize == 704)
    }

    @Test func moeFieldsDefaultToDenseWhenAbsent() throws {
        let cfg = try decodeText(#"{"model_type":"gemma4_text"}"#)
        #expect(cfg.enableMoeBlock == false)
        #expect(cfg.numExperts == nil)
        #expect(cfg.topKExperts == nil)
        #expect(cfg.moeIntermediateSize == nil)
    }
}

@Suite("Extensions/Gemma4Patched/Projection")
struct Gemma4PatchedProjectionTests {

    @Test func perLayerModelProjectionIsAQuantizableLinear() throws {
        var cfg = try JSONDecoder().decode(
            Gemma4PatchedTextConfiguration.self,
            from: Data(#"{"model_type":"gemma4_text"}"#.utf8))
        // e2b-style: PLE enabled, small model to keep allocation cheap.
        cfg.hiddenSize = 64
        cfg.numHiddenLayers = 2
        cfg.hiddenSizePerLayerInput = 8
        cfg.vocabSize = 32
        cfg.vocabSizePerLayerInput = 32
        cfg.numKvSharedLayers = 0
        cfg.layerTypes = ["sliding_attention", "full_attention"]

        let model = Gemma4PatchedTextModelInner(cfg)
        let proj = try #require(model.perLayerModelProjection)
        // A plain Linear exposes `weight` shaped [out, in] = [layers*ple, hidden].
        #expect(proj.weight.dim(0) == cfg.numHiddenLayers * cfg.hiddenSizePerLayerInput)
        #expect(proj.weight.dim(1) == cfg.hiddenSize)
    }
}
