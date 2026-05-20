import Foundation
import MLX
import MLXNN
import MLXLMCommon
import Testing

@testable import OLMLX

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
