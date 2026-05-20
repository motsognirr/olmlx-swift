import ArgumentParser
import Foundation
import OLMLX

struct Ext: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ext",
        abstract: "Inspect temporary architecture/feature extensions",
        subcommands: [ExtList.self, ExtCheck.self]
    )
}

struct ExtList: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list", abstract: "List active temporary extensions")

    func run() throws {
        print(OLMLXExtensions.listSummary(OLMLXExtensions.manifest))
    }
}

struct ExtCheck: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "check",
        abstract: "Fail if any version-gated extension is covered by the given upstream version")

    @Option(name: .long, help: "Pinned mlx-swift-lm version to check against, e.g. 3.31.3")
    var pinnedVersion: String

    func run() throws {
        let removable = OLMLXExtensions.removableEntries(
            OLMLXExtensions.manifest, pinnedVersion: pinnedVersion)
        if removable.isEmpty {
            print("No removable extensions at upstream \(pinnedVersion).")
            return
        }
        let names = removable.map(\.modelType).joined(separator: ", ")
        // Reuse ValidationError purely for its non-zero exit + stderr message (CI gate), not for input validation.
        throw ValidationError(
            "These extensions can now be removed (upstream \(pinnedVersion) covers them): \(names)")
    }
}
