import Foundation
import Testing

@testable import OLMLX

@Suite("Utils")
struct UtilsTests {
    @Test func timingStatsDefaults() {
        let stats = TimingStats()
        #expect(stats.totalDuration == 0)
        #expect(stats.loadDuration == 0)
        #expect(stats.promptEvalCount == 0)
        #expect(stats.promptEvalDuration == 0)
        #expect(stats.evalCount == 0)
        #expect(stats.evalDuration == 0)
    }

    @Test func timingStatsToDict() {
        let stats = TimingStats(totalDuration: 100, loadDuration: 10,
                                promptEvalCount: 5, promptEvalDuration: 20,
                                evalCount: 50, evalDuration: 70)
        let dict = stats.toDict()
        #expect(dict["total_duration"] == 100)
        #expect(dict["load_duration"] == 10)
        #expect(dict["prompt_eval_count"] == 5)
        #expect(dict["eval_count"] == 50)
    }

    @Test func timerMeasuresDuration() {
        let (_, duration) = Timer.measure {
            Thread.sleep(forTimeInterval: 0.01)
        }
        #expect(duration >= 10_000_000)
    }

    @Test func timerInitialValues() {
        let t = Timer()
        #expect(t.durationNs == 0)
    }

    @Test func streamTokenDefaults() {
        let tok = StreamToken(text: "hi", promptTokens: 5, generationTokens: 1,
                              promptTps: 100.0, generationTps: 50.0)
        #expect(tok.text == "hi")
        #expect(tok.token == nil)
        #expect(tok.finishReason == nil)
    }

    @Test func streamTokenWithFinishReason() {
        let tok = StreamToken(text: "", token: nil, promptTokens: 5, generationTokens: 10,
                              promptTps: 100.0, generationTps: 50.0, finishReason: "stop")
        #expect(tok.finishReason == "stop")
    }

    @Test func stripThinkingDetectPhase() {
        var state = ThinkingStripState()
        _ = stripThinkingStreaming(text: "hello world", state: &state)
        #expect(state.phase == "detect")
    }

    @Test func stripThinkingInPassthrough() {
        var state = ThinkingStripState()
        let bigText = String(repeating: "a", count: 250)
        let result = stripThinkingStreaming(text: bigText, state: &state)
        #expect(state.phase == "passthrough")
        #expect(result.count > 0)
    }

    @Test func stripThinkingStripsThinkTags() {
        var state = ThinkingStripState(thinkingExpected: true)
        let result = stripThinkingStreaming(text: "<think>reasoning here</think>hello", state: &state)
        #expect(result == "hello")
    }

    @Test func stripThinkingKeepsContentAfterClose() {
        var state = ThinkingStripState(thinkingExpected: true)
        var result = stripThinkingStreaming(text: "</think>visible text", state: &state)
        result += stripThinkingStreaming(text: " more", state: &state)
        #expect(result.hasPrefix("visible text"))
    }

    @Test func flushThinking() {
        var state = ThinkingStripState()
        _ = stripThinkingStreaming(text: "visible", state: &state)
        let flushed = flushThinkingBuffer(state: &state)
        #expect(flushed == "visible")
        #expect(state.buffer == "")
        #expect(state.phase == "detect")
    }

    @Test func systemMemoryPositive() {
        #expect(getSystemMemoryBytes() > 0)
    }

    @Test func memoryPressureNotHigh() {
        #expect(isMemoryPressureHigh(limitFraction: 1.0) == false)
    }
}
