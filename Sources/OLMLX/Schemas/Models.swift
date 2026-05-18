import Foundation

public struct ModelDetails: Codable, Sendable {
    public var parentModel: String = ""
    public var format: String = "mlx"
    public var family: String = ""
    public var families: [String]?
    public var parameterSize: String = ""
    public var quantizationLevel: String = ""

    private enum CodingKeys: String, CodingKey {
        case format, family, families
        case parentModel = "parent_model"
        case parameterSize = "parameter_size"
        case quantizationLevel = "quantization_level"
    }

    public init(parentModel: String = "", format: String = "mlx", family: String = "",
                families: [String]? = nil, parameterSize: String = "", quantizationLevel: String = "") {
        self.parentModel = parentModel
        self.format = format
        self.family = family
        self.families = families
        self.parameterSize = parameterSize
        self.quantizationLevel = quantizationLevel
    }
}

public struct ModelInfo: Codable, Sendable {
    public var name: String
    public var model: String = ""
    public var modifiedAt: String = ""
    public var size: Int = 0
    public var digest: String = ""
    public var details: ModelDetails = ModelDetails()

    private enum CodingKeys: String, CodingKey {
        case name, model, size, digest, details
        case modifiedAt = "modified_at"
    }

    public init(name: String, model: String = "", modifiedAt: String = "",
                size: Int = 0, digest: String = "", details: ModelDetails = ModelDetails()) {
        self.name = name
        self.model = model
        self.modifiedAt = modifiedAt
        self.size = size
        self.digest = digest
        self.details = details
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        model = try container.decodeIfPresent(String.self, forKey: .model) ?? ""
        modifiedAt = try container.decodeIfPresent(String.self, forKey: .modifiedAt) ?? ""
        size = try container.decodeIfPresent(Int.self, forKey: .size) ?? 0
        digest = try container.decodeIfPresent(String.self, forKey: .digest) ?? ""
        details = try container.decodeIfPresent(ModelDetails.self, forKey: .details) ?? ModelDetails()
    }
}

public struct TagsResponse: Codable, Sendable {
    public var models: [ModelInfo]
}

public struct ShowRequest: Codable, Sendable {
    public var model: String
    public var verbose: Bool = false
}

public struct ShowResponse: Codable, Sendable {
    public var modelfile: String = ""
    public var parameters: String = ""
    public var template: String = ""
    public var details: ModelDetails = ModelDetails()
    public var modelInfo: [String: AnyCodable] = [:]
    public var modifiedAt: String = ""

    private enum CodingKeys: String, CodingKey {
        case modelfile, parameters, template, details
        case modelInfo = "model_info"
        case modifiedAt = "modified_at"
    }
}

public struct RunningModel: Codable, Sendable {
    public var name: String
    public var model: String = ""
    public var size: Int = 0
    public var digest: String = ""
    public var details: ModelDetails = ModelDetails()
    public var expiresAt: String = ""
    public var sizeVram: Int = 0
    public var activeRefs: Int = 0

    private enum CodingKeys: String, CodingKey {
        case name, model, size, digest, details
        case expiresAt = "expires_at"
        case sizeVram = "size_vram"
        case activeRefs = "active_refs"
    }
}

public struct PsResponse: Codable, Sendable {
    public var models: [RunningModel]
}

public struct VersionResponse: Codable, Sendable {
    public var version: String
}

public struct PullRequest: Codable, Sendable {
    public var model: String
    public var insecure: Bool = false
    public var stream: Bool = true

    public init(model: String, insecure: Bool = false, stream: Bool = true) {
        self.model = model
        self.insecure = insecure
        self.stream = stream
    }
}

public struct PullResponse: Codable, Sendable {
    public var status: String
    public var digest: String?
    public var total: Int?
    public var completed: Int?
}

public struct CopyRequest: Codable, Sendable {
    public var source: String
    public var destination: String

    public init(source: String, destination: String) {
        self.source = source
        self.destination = destination
    }
}

public struct DeleteRequest: Codable, Sendable {
    public var model: String

    public init(model: String) {
        self.model = model
    }
}

public struct CreateRequest: Codable, Sendable {
    public var model: String
    public var modelfile: String?
    public var stream: Bool = true
    public var path: String?
    public var quantize: String?

    public init(model: String, modelfile: String? = nil, stream: Bool = true,
                path: String? = nil, quantize: String? = nil) {
        self.model = model
        self.modelfile = modelfile
        self.stream = stream
        self.path = path
        self.quantize = quantize
    }
}

public struct WarmupRequest: Codable, Sendable {
    public var model: String
    public var keepAlive: String?

    private enum CodingKeys: String, CodingKey {
        case model
        case keepAlive = "keep_alive"
    }

    public init(model: String, keepAlive: String? = nil) {
        self.model = model
        self.keepAlive = keepAlive
    }
}

public struct AbortRequest: Codable, Sendable {
    public var model: String

    public init(model: String) {
        self.model = model
    }
}

public struct UnloadRequest: Codable, Sendable {
    public var model: String

    public init(model: String) {
        self.model = model
    }
}
