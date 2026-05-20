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

@Suite("Extensions/Gemma4Patched/Attention")
struct Gemma4PatchedAttentionTests {

    /// Reproduces the #59 broadcast crash: a full-attention layer with
    /// attention_k_eq_v and global KV heads. Pre-fix this aborts in SDPA.
    @Test func kEqVFullAttentionForwardProducesCorrectShape() throws {
        var cfg = try JSONDecoder().decode(
            Gemma4PatchedTextConfiguration.self,
            from: Data(#"{"model_type":"gemma4_text"}"#.utf8))
        cfg.hiddenSize = 64
        cfg.numAttentionHeads = 8
        cfg.numKeyValueHeads = 2
        cfg.numGlobalKeyValueHeads = 2
        cfg.headDim = 16
        cfg.globalHeadDim = 32
        cfg.attentionKeqV = true
        cfg.layerTypes = ["full_attention"]

        let attn = Gemma4PatchedAttention(cfg, layerIdx: 0)
        let x = MLXArray.ones([1, 5, cfg.hiddenSize]).asType(.float32)
        let (out, _, _) = attn(x, mask: .none, cache: nil)
        out.eval()
        #expect(out.dim(0) == 1)
        #expect(out.dim(1) == 5)
        #expect(out.dim(2) == cfg.hiddenSize)
    }
}

@Suite("Extensions/Gemma4Patched/Sanitize")
struct Gemma4PatchedSanitizeTests {

    private func moeConfig() throws -> Gemma4PatchedConfiguration {
        try JSONDecoder().decode(
            Gemma4PatchedConfiguration.self,
            from: Data(#"""
            {"model_type":"gemma4","text_config":{"model_type":"gemma4_text",
             "hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,
             "num_key_value_heads":1,"hidden_size_per_layer_input":0,
             "num_kv_shared_layers":0,
             "enable_moe_block":true,"num_experts":4,"top_k_experts":2,
             "moe_intermediate_size":6,"layer_types":["full_attention"]}}
            """#.utf8))
    }

    @Test func splitsFusedExpertWeights() throws {
        let model = Gemma4PatchedModel(try moeConfig())
        let base = "language_model.model.layers.0.experts"
        // gate_up_proj: [experts, 2*hidden_dims, input]
        let gateUp = MLXArray.ones([4, 12, 8])
        let down = MLXArray.ones([4, 8, 6])
        let out = model.sanitize(weights: [
            "\(base).gate_up_proj": gateUp,
            "\(base).down_proj": down,
        ])
        #expect(out["\(base).gate_up_proj"] == nil)
        #expect(out["\(base).down_proj"] == nil)
        #expect(out["\(base).switch_glu.gate_proj.weight"]?.dim(-2) == 6)
        #expect(out["\(base).switch_glu.up_proj.weight"]?.dim(-2) == 6)
        #expect(out["\(base).switch_glu.down_proj.weight"]?.dim(-1) == 6)
    }
}

@Suite("Extensions/Gemma4Patched/Registration")
struct Gemma4PatchedRegistrationTests {

    @Test func manifestRegistersGemma4Overrides() {
        let types = Set(OLMLXExtensions.manifest.map { $0.modelType })
        #expect(types.contains("gemma4"))
        #expect(types.contains("gemma4_text"))
    }

    @Test func gemma4EntryBuildsAModelFromConfig() throws {
        let entry = try #require(
            OLMLXExtensions.manifest.first { $0.modelType == "gemma4" })
        let cfg = #"""
        {"model_type":"gemma4","text_config":{"model_type":"gemma4_text",
         "hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,
         "num_key_value_heads":1,"hidden_size_per_layer_input":0,
         "num_kv_shared_layers":0,
         "layer_types":["full_attention"]}}
        """#
        let model = try entry.creator(Data(cfg.utf8))
        #expect(model is Gemma4PatchedModel)
    }
}
