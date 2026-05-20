import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import Testing

@testable import OLMLX

/// Minimal LanguageModel used to prove registration/resolution without real weights.
private final class StubLanguageModel: Module, LanguageModel {
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray { inputs }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

@Suite("Extensions/Registration")
struct RegistrationTests {

    private func testEntry() -> ExtensionEntry {
        ExtensionEntry(
            modelType: "olmlx_unit_stub",
            kind: .architecture,
            upstreamTracking: URL(string: "https://example.com/issue/1")!,
            addedOn: "2026-05-20",
            removeWhen: .upstreamReleased(version: "99.0.0"),
            notes: "unit test stub",
            creator: { _ in StubLanguageModel() }
        )
    }

    @Test func registersEntriesOntoAFreshRegistry() async throws {
        let registry = ModelTypeRegistry<any LanguageModel>()
        await OLMLXExtensions.register([testEntry()], onto: registry)

        let model = try await registry.createModel(
            configuration: Data("{}".utf8), modelType: "olmlx_unit_stub")
        #expect(model is StubLanguageModel)
    }

    @Test func unregisteredTypeStillThrows() async {
        let registry = ModelTypeRegistry<any LanguageModel>()
        await OLMLXExtensions.register([testEntry()], onto: registry)

        await #expect(throws: (any Error).self) {
            _ = try await registry.createModel(
                configuration: Data("{}".utf8), modelType: "does_not_exist")
        }
    }

    @Test func everyManifestEntryHasTrackingAndIsArchitectureOrFeature() {
        for entry in OLMLXExtensions.manifest {
            #expect(!entry.upstreamTracking.absoluteString.isEmpty)
            #expect(entry.kind == .architecture || entry.kind == .feature)
        }
    }
}

@Suite("Extensions/RemovalCondition")
struct RemovalConditionTests {

    @Test func upstreamReleasedIsRemovableWhenPinnedAtOrAboveTarget() {
        let cond = RemovalCondition.upstreamReleased(version: "3.32.0")
        #expect(cond.isRemovable(pinnedVersion: "3.32.0") == true)
        #expect(cond.isRemovable(pinnedVersion: "3.32.1") == true)
        #expect(cond.isRemovable(pinnedVersion: "3.33.0") == true)
        #expect(cond.isRemovable(pinnedVersion: "3.31.3") == false)
        #expect(cond.isRemovable(pinnedVersion: "3.4.0") == false)
    }

    @Test func upstreamMergedIsNeverAutoRemovable() {
        let cond = RemovalCondition.upstreamMerged(
            pr: URL(string: "https://github.com/ml-explore/mlx-swift-lm/pull/244")!)
        #expect(cond.isRemovable(pinnedVersion: "9.9.9") == false)
    }
}
