import Foundation
import MLX
import MLXLMCommon
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
