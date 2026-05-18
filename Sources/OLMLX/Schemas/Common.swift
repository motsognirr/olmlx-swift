import Foundation

public struct ModelOptions: Codable, Sendable {
    public var numKeep: Int?
    public var seed: Int?
    public var numPredict: Int?
    public var topK: Int?
    public var topP: Double?
    public var minP: Double?
    public var tfsZ: Double?
    public var typicalP: Double?
    public var repeatLastN: Int?
    public var temperature: Double?
    public var repeatPenalty: Double?
    public var presencePenalty: Double?
    public var frequencyPenalty: Double?
    public var mirostat: Int?
    public var mirostatTau: Double?
    public var mirostatEta: Double?
    public var penalizeNewline: Bool?
    public var stop: [String]?
    public var numa: Bool?
    public var numCtx: Int?
    public var numBatch: Int?
    public var numGpu: Int?
    public var mainGpu: Int?
    public var lowVram: Bool?
    public var vocabOnly: Bool?
    public var useMmap: Bool?
    public var useMlock: Bool?
    public var numThread: Int?

    private var extra: [String: AnyCodable] = [:]

    private enum CodingKeys: String, CodingKey {
        case numKeep = "num_keep"
        case seed
        case numPredict = "num_predict"
        case topK = "top_k"
        case topP = "top_p"
        case minP = "min_p"
        case tfsZ = "tfs_z"
        case typicalP = "typical_p"
        case repeatLastN = "repeat_last_n"
        case temperature
        case repeatPenalty = "repeat_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case mirostat
        case mirostatTau = "mirostat_tau"
        case mirostatEta = "mirostat_eta"
        case penalizeNewline = "penalize_newline"
        case stop
        case numa
        case numCtx = "num_ctx"
        case numBatch = "num_batch"
        case numGpu = "num_gpu"
        case mainGpu = "main_gpu"
        case lowVram = "low_vram"
        case vocabOnly = "vocab_only"
        case useMmap = "use_mmap"
        case useMlock = "use_mlock"
        case numThread = "num_thread"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DynamicCodingKey.self)
        let allKeys = container.allKeys

        for key in allKeys {
            switch key.stringValue {
            case "num_keep": numKeep = try container.decodeIfPresent(Int.self, forKey: key)
            case "seed": seed = try container.decodeIfPresent(Int.self, forKey: key)
            case "num_predict": numPredict = try container.decodeIfPresent(Int.self, forKey: key)
            case "top_k": topK = try container.decodeIfPresent(Int.self, forKey: key)
            case "top_p": topP = try container.decodeIfPresent(Double.self, forKey: key)
            case "min_p": minP = try container.decodeIfPresent(Double.self, forKey: key)
            case "tfs_z": tfsZ = try container.decodeIfPresent(Double.self, forKey: key)
            case "typical_p": typicalP = try container.decodeIfPresent(Double.self, forKey: key)
            case "repeat_last_n": repeatLastN = try container.decodeIfPresent(Int.self, forKey: key)
            case "temperature": temperature = try container.decodeIfPresent(Double.self, forKey: key)
            case "repeat_penalty": repeatPenalty = try container.decodeIfPresent(Double.self, forKey: key)
            case "presence_penalty": presencePenalty = try container.decodeIfPresent(Double.self, forKey: key)
            case "frequency_penalty": frequencyPenalty = try container.decodeIfPresent(Double.self, forKey: key)
            case "mirostat": mirostat = try container.decodeIfPresent(Int.self, forKey: key)
            case "mirostat_tau": mirostatTau = try container.decodeIfPresent(Double.self, forKey: key)
            case "mirostat_eta": mirostatEta = try container.decodeIfPresent(Double.self, forKey: key)
            case "penalize_newline": penalizeNewline = try container.decodeIfPresent(Bool.self, forKey: key)
            case "stop": stop = try container.decodeIfPresent([String].self, forKey: key)
            case "numa": numa = try container.decodeIfPresent(Bool.self, forKey: key)
            case "num_ctx": numCtx = try container.decodeIfPresent(Int.self, forKey: key)
            case "num_batch": numBatch = try container.decodeIfPresent(Int.self, forKey: key)
            case "num_gpu": numGpu = try container.decodeIfPresent(Int.self, forKey: key)
            case "main_gpu": mainGpu = try container.decodeIfPresent(Int.self, forKey: key)
            case "low_vram": lowVram = try container.decodeIfPresent(Bool.self, forKey: key)
            case "vocab_only": vocabOnly = try container.decodeIfPresent(Bool.self, forKey: key)
            case "use_mmap": useMmap = try container.decodeIfPresent(Bool.self, forKey: key)
            case "use_mlock": useMlock = try container.decodeIfPresent(Bool.self, forKey: key)
            case "num_thread": numThread = try container.decodeIfPresent(Int.self, forKey: key)
            default:
                if let anyVal = try container.decodeIfPresent(AnyCodable.self, forKey: key) {
                    extra[key.stringValue] = anyVal
                }
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: DynamicCodingKey.self)
        try container.encodeIfPresent(numKeep, forKey: DynamicCodingKey(stringValue: "num_keep")!)
        try container.encodeIfPresent(seed, forKey: DynamicCodingKey(stringValue: "seed")!)
        try container.encodeIfPresent(numPredict, forKey: DynamicCodingKey(stringValue: "num_predict")!)
        try container.encodeIfPresent(topK, forKey: DynamicCodingKey(stringValue: "top_k")!)
        try container.encodeIfPresent(topP, forKey: DynamicCodingKey(stringValue: "top_p")!)
        try container.encodeIfPresent(minP, forKey: DynamicCodingKey(stringValue: "min_p")!)
        try container.encodeIfPresent(tfsZ, forKey: DynamicCodingKey(stringValue: "tfs_z")!)
        try container.encodeIfPresent(typicalP, forKey: DynamicCodingKey(stringValue: "typical_p")!)
        try container.encodeIfPresent(repeatLastN, forKey: DynamicCodingKey(stringValue: "repeat_last_n")!)
        try container.encodeIfPresent(temperature, forKey: DynamicCodingKey(stringValue: "temperature")!)
        try container.encodeIfPresent(repeatPenalty, forKey: DynamicCodingKey(stringValue: "repeat_penalty")!)
        try container.encodeIfPresent(presencePenalty, forKey: DynamicCodingKey(stringValue: "presence_penalty")!)
        try container.encodeIfPresent(frequencyPenalty, forKey: DynamicCodingKey(stringValue: "frequency_penalty")!)
        try container.encodeIfPresent(mirostat, forKey: DynamicCodingKey(stringValue: "mirostat")!)
        try container.encodeIfPresent(mirostatTau, forKey: DynamicCodingKey(stringValue: "mirostat_tau")!)
        try container.encodeIfPresent(mirostatEta, forKey: DynamicCodingKey(stringValue: "mirostat_eta")!)
        try container.encodeIfPresent(penalizeNewline, forKey: DynamicCodingKey(stringValue: "penalize_newline")!)
        try container.encodeIfPresent(stop, forKey: DynamicCodingKey(stringValue: "stop")!)
        try container.encodeIfPresent(numa, forKey: DynamicCodingKey(stringValue: "numa")!)
        try container.encodeIfPresent(numCtx, forKey: DynamicCodingKey(stringValue: "num_ctx")!)
        try container.encodeIfPresent(numBatch, forKey: DynamicCodingKey(stringValue: "num_batch")!)
        try container.encodeIfPresent(numGpu, forKey: DynamicCodingKey(stringValue: "num_gpu")!)
        try container.encodeIfPresent(mainGpu, forKey: DynamicCodingKey(stringValue: "main_gpu")!)
        try container.encodeIfPresent(lowVram, forKey: DynamicCodingKey(stringValue: "low_vram")!)
        try container.encodeIfPresent(vocabOnly, forKey: DynamicCodingKey(stringValue: "vocab_only")!)
        try container.encodeIfPresent(useMmap, forKey: DynamicCodingKey(stringValue: "use_mmap")!)
        try container.encodeIfPresent(useMlock, forKey: DynamicCodingKey(stringValue: "use_mlock")!)
        try container.encodeIfPresent(numThread, forKey: DynamicCodingKey(stringValue: "num_thread")!)
        for (key, value) in extra {
            try container.encode(value, forKey: DynamicCodingKey(stringValue: key)!)
        }
    }
}

public enum AnyCodable: Codable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case dictionary([String: AnyCodable])
    case array([AnyCodable])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let v = try? container.decode(String.self) {
            self = .string(v)
        } else if let v = try? container.decode(Int.self) {
            self = .int(v)
        } else if let v = try? container.decode(Double.self) {
            self = .double(v)
        } else if let v = try? container.decode(Bool.self) {
            self = .bool(v)
        } else if let v = try? container.decode([String: AnyCodable].self) {
            self = .dictionary(v)
        } else if let v = try? container.decode([AnyCodable].self) {
            self = .array(v)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported type")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .int(let v): try container.encode(v)
        case .double(let v): try container.encode(v)
        case .bool(let v): try container.encode(v)
        case .dictionary(let v): try container.encode(v)
        case .array(let v): try container.encode(v)
        case .null: try container.encodeNil()
        }
    }
}

public struct DynamicCodingKey: CodingKey {
    public var stringValue: String
    public var intValue: Int?

    public init?(stringValue: String) { self.stringValue = stringValue }
    public init?(intValue: Int) { return nil }
}

public func validateTokenLimit(_ value: Int, fieldName: String, maxLimit: Int = 131072) throws {
    if value > maxLimit {
        throw SchemaValidationError(
            "\(fieldName) \(value) exceeds configured limit \(maxLimit)"
        )
    }
}

public func validateNonEmptyTextInput(_ value: String, fieldName: String) throws {
    if value.isEmpty {
        throw SchemaValidationError("\(fieldName) cannot be empty")
    }
}

public struct SchemaValidationError: Error, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
}
