import Foundation

public struct TimingStats: Codable, Sendable {
    public var totalDuration: Int = 0
    public var loadDuration: Int = 0
    public var promptEvalCount: Int = 0
    public var promptEvalDuration: Int = 0
    public var evalCount: Int = 0
    public var evalDuration: Int = 0

    private enum CodingKeys: String, CodingKey {
        case totalDuration = "total_duration"
        case loadDuration = "load_duration"
        case promptEvalCount = "prompt_eval_count"
        case promptEvalDuration = "prompt_eval_duration"
        case evalCount = "eval_count"
        case evalDuration = "eval_duration"
    }

    public init(
        totalDuration: Int = 0, loadDuration: Int = 0,
        promptEvalCount: Int = 0, promptEvalDuration: Int = 0,
        evalCount: Int = 0, evalDuration: Int = 0
    ) {
        self.totalDuration = totalDuration
        self.loadDuration = loadDuration
        self.promptEvalCount = promptEvalCount
        self.promptEvalDuration = promptEvalDuration
        self.evalCount = evalCount
        self.evalDuration = evalDuration
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        totalDuration = try container.decodeIfPresent(Int.self, forKey: .totalDuration) ?? 0
        loadDuration = try container.decodeIfPresent(Int.self, forKey: .loadDuration) ?? 0
        promptEvalCount = try container.decodeIfPresent(Int.self, forKey: .promptEvalCount) ?? 0
        promptEvalDuration = try container.decodeIfPresent(Int.self, forKey: .promptEvalDuration) ?? 0
        evalCount = try container.decodeIfPresent(Int.self, forKey: .evalCount) ?? 0
        evalDuration = try container.decodeIfPresent(Int.self, forKey: .evalDuration) ?? 0
    }

    public func toDict() -> [String: Int] {
        return [
            "total_duration": totalDuration,
            "load_duration": loadDuration,
            "prompt_eval_count": promptEvalCount,
            "prompt_eval_duration": promptEvalDuration,
            "eval_count": evalCount,
            "eval_duration": evalDuration,
        ]
    }
}

public final class Timer {
    private var startTime: UInt64 = 0
    private var endTime: UInt64 = 0

    public init() {}

    public func start() {
        startTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }

    public func stop() {
        endTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }

    public var durationNs: UInt64 {
        return endTime - startTime
    }

    @discardableResult
    public static func measure<T>(_ block: () throws -> T) rethrows -> (T, UInt64) {
        let timer = Timer()
        timer.start()
        let result = try block()
        timer.stop()
        return (result, timer.durationNs)
    }
}
