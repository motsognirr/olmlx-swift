import Foundation

public struct StreamToken: Sendable {
    public var text: String
    public var token: Int?
    public var promptTokens: Int
    public var generationTokens: Int
    public var promptTps: Double
    public var generationTps: Double
    public var finishReason: String?

    public init(text: String, token: Int? = nil,
                promptTokens: Int, generationTokens: Int,
                promptTps: Double, generationTps: Double,
                finishReason: String? = nil) {
        self.text = text
        self.token = token
        self.promptTokens = promptTokens
        self.generationTokens = generationTokens
        self.promptTps = promptTps
        self.generationTps = generationTps
        self.finishReason = finishReason
    }
}

// Thinking block stripping - mirrors Python strip_thinking_streaming
public struct ThinkingStripState {
    public var phase: String = "detect"
    public var buffer: String = ""
    public var expectedClose: String = ""
    public var thinkingExpected: Bool = false
    public var detectLimit: Int = 200

    public init(thinkingExpected: Bool = false) {
        self.thinkingExpected = thinkingExpected
        if thinkingExpected { detectLimit = 4096 }
    }
}

private let thinkingTagPairs: [(String, String)] = [
    ("<think>", "</think>"),
    ("<|channel>thought\n", "<channel|>"),
]

public func stripThinkingStreaming(text: String, state: inout ThinkingStripState) -> String {
    var buf = state.buffer + text
    var outParts: [String] = []
    var phase = state.phase
    var expectedClose = state.expectedClose

    let openTags = thinkingTagPairs.map { $0.0 }
    let closeForOpen = Dictionary(uniqueKeysWithValues: thinkingTagPairs)

    while !buf.isEmpty {
        if phase == "detect" {
            let (openTag, openIdx) = findEarliest(tags: openTags, in: buf)
            let (closeTag, closeIdx) = findEarliest(tags: thinkingTagPairs.map { $0.1 }, in: buf)

            if closeIdx != -1 && (openIdx == -1 || closeIdx < openIdx) && state.thinkingExpected {
                let endPos = closeIdx + closeTag.count
                buf = String(buf[buf.index(buf.startIndex, offsetBy: endPos)...])
                expectedClose = ""
                phase = "passthrough"
            } else if openIdx != -1 {
                let prefix = String(buf[..<buf.index(buf.startIndex, offsetBy: openIdx)])
                outParts.append(prefix)
                let startPos = openIdx + openTag.count
                buf = String(buf[buf.index(buf.startIndex, offsetBy: startPos)...])
                expectedClose = closeForOpen[openTag] ?? ""
                phase = "in_think"
            } else {
                if buf.count > state.detectLimit {
                    let partial = longestPartialSuffix(buf: buf, tags: openTags)
                    if partial > 0 {
                        let split = buf.count - partial
                        outParts.append(String(buf[..<buf.index(buf.startIndex, offsetBy: split)]))
                        buf = String(buf[buf.index(buf.startIndex, offsetBy: split)...])
                    } else {
                        outParts.append(buf)
                        buf = ""
                    }
                    phase = "passthrough"
                }
                break
            }
        } else if phase == "in_think" {
            if let range = buf.range(of: expectedClose) {
                buf = String(buf[range.upperBound...])
                expectedClose = ""
                phase = "passthrough"
            } else {
                let partial = longestPartialSuffix(buf: buf, tags: [expectedClose])
                if partial > 0 {
                    buf = String(buf.suffix(partial))
                } else {
                    buf = ""
                }
                break
            }
        } else {
            let (openTag, openIdx) = findEarliest(tags: openTags, in: buf)
            if openIdx == -1 {
                let partial = longestPartialSuffix(buf: buf, tags: openTags)
                if partial > 0 {
                    let split = buf.count - partial
                    outParts.append(String(buf[..<buf.index(buf.startIndex, offsetBy: split)]))
                    buf = String(buf[buf.index(buf.startIndex, offsetBy: split)...])
                    break
                } else {
                    outParts.append(buf)
                    buf = ""
                }
            } else {
                let prefix = String(buf[..<buf.index(buf.startIndex, offsetBy: openIdx)])
                outParts.append(prefix)
                let startPos = openIdx + openTag.count
                buf = String(buf[buf.index(buf.startIndex, offsetBy: startPos)...])
                expectedClose = closeForOpen[openTag] ?? ""
                phase = "in_think"
            }
        }
    }

    state.buffer = buf
    state.phase = phase
    state.expectedClose = expectedClose
    return outParts.joined()
}

public func flushThinkingBuffer(state: inout ThinkingStripState) -> String {
    let buf = state.buffer
    let phase = state.phase
    state.buffer = ""
    state.phase = "detect"
    state.expectedClose = ""
    if phase == "detect" || phase == "passthrough" { return buf }
    return ""
}

private func findEarliest(tags: [String], in s: String) -> (String, Int) {
    var bestTag = ""
    var bestIdx = -1
    for tag in tags {
        if let range = s.range(of: tag) {
            let idx = s.distance(from: s.startIndex, to: range.lowerBound)
            if bestIdx == -1 || idx < bestIdx {
                bestIdx = idx
                bestTag = tag
            }
        }
    }
    return (bestTag, bestIdx)
}

private func longestPartialSuffix(buf: String, tags: [String]) -> Int {
    var longest = 0
    for tag in tags {
        for i in stride(from: min(tag.count, buf.count), through: longest + 1, by: -1) {
            if tag.hasPrefix(buf.suffix(i)) {
                longest = i
                break
            }
        }
    }
    return longest
}
