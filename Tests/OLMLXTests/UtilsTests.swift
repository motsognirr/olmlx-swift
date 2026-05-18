import Foundation
import OLMLX
import OLMLXTestUtils

func OLMLXTests_timingStatsDefaults() throws {
    let stats = TimingStats()
    try expectEqual(stats.totalDuration, 0)
    try expectEqual(stats.loadDuration, 0)
    try expectEqual(stats.promptEvalCount, 0)
    try expectEqual(stats.promptEvalDuration, 0)
    try expectEqual(stats.evalCount, 0)
    try expectEqual(stats.evalDuration, 0)
}

func OLMLXTests_timingStatsToDict() throws {
    let stats = TimingStats(totalDuration: 100, loadDuration: 10,
                            promptEvalCount: 5, promptEvalDuration: 20,
                            evalCount: 50, evalDuration: 70)
    let dict = stats.toDict()
    try expectEqual(dict["total_duration"], 100)
    try expectEqual(dict["load_duration"], 10)
    try expectEqual(dict["prompt_eval_count"], 5)
    try expectEqual(dict["eval_count"], 50)
}

func OLMLXTests_timerMeasuresDuration() throws {
    let (_, duration) = Timer.measure {
        Thread.sleep(forTimeInterval: 0.01)
    }
    try expect(duration >= 10_000_000, "should be at least 10ms in ns")
}

func OLMLXTests_timerInitialValues() throws {
    let t = Timer()
    try expectEqual(t.durationNs, 0)
}

func OLMLXTests_streamTokenDefaults() throws {
    let tok = StreamToken(text: "hi", promptTokens: 5, generationTokens: 1,
                          promptTps: 100.0, generationTps: 50.0)
    try expectEqual(tok.text, "hi")
    try expectNil(tok.token)
    try expectNil(tok.finishReason)
}

func OLMLXTests_streamTokenWithFinishReason() throws {
    let tok = StreamToken(text: "", token: nil, promptTokens: 5, generationTokens: 10,
                          promptTps: 100.0, generationTps: 50.0, finishReason: "stop")
    try expectEqual(tok.finishReason, "stop")
}

func OLMLXTests_stripThinkingDetectPhase() throws {
    var state = ThinkingStripState()
    _ = stripThinkingStreaming(text: "hello world", state: &state)
    try expectEqual(state.phase, "detect")
    // Buffer grows in detect phase, nothing emitted until limit
}

func OLMLXTests_stripThinkingInPassthrough() throws {
    var state = ThinkingStripState()
    // Push enough data to exceed detect limit and transition to passthrough
    let bigText = String(repeating: "a", count: 250)
    let result = stripThinkingStreaming(text: bigText, state: &state)
    try expectEqual(state.phase, "passthrough")
    try expect(result.count > 0, "should emit content in passthrough")
}

func OLMLXTests_stripThinkingStripsThinkTags() throws {
    var state = ThinkingStripState(thinkingExpected: true)
    let result = stripThinkingStreaming(text: "<think>reasoning here</think>hello", state: &state)
    try expectEqual(result, "hello")
}

func OLMLXTests_stripThinkingKeepsContentAfterClose() throws {
    var state = ThinkingStripState(thinkingExpected: true)
    var result = stripThinkingStreaming(text: "</think>visible text", state: &state)
    result += stripThinkingStreaming(text: " more", state: &state)
    try expect(result.hasPrefix("visible text"), "should keep text after orphan close")
}

func OLMLXTests_flushThinkingBuffer() throws {
    var state = ThinkingStripState()
    _ = stripThinkingStreaming(text: "visible", state: &state)
    let flushed = flushThinkingBuffer(state: &state)
    try expectEqual(flushed, "visible")
    try expectEqual(state.buffer, "")
    try expectEqual(state.phase, "detect")
}

func OLMLXTests_systemMemory() throws {
    let mem = getSystemMemoryBytes()
    try expect(mem > 0, "system memory should be positive")
}

func OLMLXTests_memoryPressureNotHigh() throws {
    let result = isMemoryPressureHigh(limitFraction: 1.0)
    try expectEqual(result, false, "should not be high with 100% limit")
}
