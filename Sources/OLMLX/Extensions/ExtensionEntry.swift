import Foundation
import MLXLMCommon

/// What kind of upstream gap a temporary extension fills.
public enum ExtensionKind: String, Sendable {
    case architecture
    case feature
}

/// Condition under which a temporary extension can be deleted.
public enum RemovalCondition: Sendable {
    /// Removable once the pinned mlx-swift-lm version reaches `version`.
    case upstreamReleased(version: String)
    /// Tracked by an upstream PR; removal requires a manual check (never auto-gated).
    case upstreamMerged(pr: URL)

    /// True only for `.upstreamReleased` when `pinnedVersion` >= the target version.
    public func isRemovable(pinnedVersion: String) -> Bool {
        switch self {
        case .upstreamMerged:
            return false
        case .upstreamReleased(let target):
            return compareVersions(pinnedVersion, target) >= 0
        }
    }
}

/// Compares dotted numeric versions. Returns -1, 0, or 1.
/// Non-numeric / missing components are treated as 0.
/// A pre-release or build suffix such as `"3.32.0-beta"` therefore compares equal to `"3.32.0"`.
func compareVersions(_ lhs: String, _ rhs: String) -> Int {
    let l = lhs.split(separator: ".").map { Int($0) ?? 0 }
    let r = rhs.split(separator: ".").map { Int($0) ?? 0 }
    for i in 0..<max(l.count, r.count) {
        let a = i < l.count ? l[i] : 0
        let b = i < r.count ? r[i] : 0
        if a != b { return a < b ? -1 : 1 }
    }
    return 0
}

/// A single temporary extension registered onto mlx-swift-lm.
public struct ExtensionEntry: Sendable {
    /// The `model_type` string from `config.json` this entry handles.
    public let modelType: String
    public let kind: ExtensionKind
    /// Upstream issue/PR this is tracking. Required — no entry without a tracking link.
    public let upstreamTracking: URL
    public let addedOn: String  // ISO date, e.g. "2026-05-20"
    public let removeWhen: RemovalCondition
    public let notes: String
    /// Builds a `LanguageModel` from raw `config.json` data.
    public let creator: @Sendable (Data) throws -> any LanguageModel

    public init(
        modelType: String,
        kind: ExtensionKind,
        upstreamTracking: URL,
        addedOn: String,
        removeWhen: RemovalCondition,
        notes: String,
        creator: @escaping @Sendable (Data) throws -> any LanguageModel
    ) {
        self.modelType = modelType
        self.kind = kind
        self.upstreamTracking = upstreamTracking
        self.addedOn = addedOn
        self.removeWhen = removeWhen
        self.notes = notes
        self.creator = creator
    }
}
