import Foundation
import CryptoKit

public struct ModelManifest: Codable, Sendable {
    public var name: String
    public var hfPath: String
    public var size: Int = 0
    public var modifiedAt: String = ""
    public var digest: String = ""
    public var format: String = "mlx"
    public var family: String = ""
    public var parameterSize: String = ""
    public var quantizationLevel: String = ""

    private enum CodingKeys: String, CodingKey {
        case name, size, digest, format, family
        case hfPath = "hf_path"
        case modifiedAt = "modified_at"
        case parameterSize = "parameter_size"
        case quantizationLevel = "quantization_level"
    }

    public init(name: String, hfPath: String, size: Int = 0, modifiedAt: String = "",
                digest: String = "", format: String = "mlx", family: String = "",
                parameterSize: String = "", quantizationLevel: String = "") {
        self.name = name
        self.hfPath = hfPath
        self.size = size
        self.modifiedAt = modifiedAt
        self.digest = digest
        self.format = format
        self.family = family
        self.parameterSize = parameterSize
        self.quantizationLevel = quantizationLevel
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        hfPath = try container.decode(String.self, forKey: .hfPath)
        size = try container.decodeIfPresent(Int.self, forKey: .size) ?? 0
        modifiedAt = try container.decodeIfPresent(String.self, forKey: .modifiedAt) ?? ""
        digest = try container.decodeIfPresent(String.self, forKey: .digest) ?? ""
        format = try container.decodeIfPresent(String.self, forKey: .format) ?? "mlx"
        family = try container.decodeIfPresent(String.self, forKey: .family) ?? ""
        parameterSize = try container.decodeIfPresent(String.self, forKey: .parameterSize) ?? ""
        quantizationLevel = try container.decodeIfPresent(String.self, forKey: .quantizationLevel) ?? ""
    }

    public func save(to path: URL) throws {
        let dir = path.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: path)
    }

    public static func load(from path: URL) throws -> ModelManifest {
        let data = try Data(contentsOf: path)
        return try JSONDecoder().decode(ModelManifest.self, from: data)
    }

    public static func computeDigest(name: String) -> String {
        let data = name.data(using: .utf8) ?? Data()
        let hash = SHA256.hash(data: data)
        let hex = hash.prefix(6).map { String(format: "%02x", $0) }.joined()
        return "sha256:" + hex
    }
}
