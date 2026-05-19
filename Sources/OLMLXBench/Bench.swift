import ArgumentParser
import Foundation
import MLXLMCommon
import OLMLX

struct BenchPromptResult: Codable {
    let promptName: String
    let category: String
    let outputText: String
    let statusCode: Int
    let error: String?
    let evalCount: Int
    let evalDurationNs: Int
    let promptEvalCount: Int
    let promptEvalDurationNs: Int
    let totalDurationNs: Int
    let tokensPerSecond: Double
    let promptTokensPerSecond: Double

    private enum CodingKeys: String, CodingKey {
        case promptName = "prompt_name"
        case category
        case outputText = "output_text"
        case statusCode = "status_code"
        case error
        case evalCount = "eval_count"
        case evalDurationNs = "eval_duration_ns"
        case promptEvalCount = "prompt_eval_count"
        case promptEvalDurationNs = "prompt_eval_duration_ns"
        case totalDurationNs = "total_duration_ns"
        case tokensPerSecond = "tokens_per_second"
        case promptTokensPerSecond = "prompt_tokens_per_second"
    }
}

struct BenchScenario: Codable {
    let scenarioName: String
    let scenarioDescription: String
    let envOverrides: [String: String]
    let promptResults: [BenchPromptResult]
    let skipped: Bool
    let skipReason: String?

    private enum CodingKeys: String, CodingKey {
        case scenarioName = "scenario_name"
        case scenarioDescription = "scenario_description"
        case envOverrides = "env_overrides"
        case promptResults = "prompt_results"
        case skipped
        case skipReason = "skip_reason"
    }
}

struct BenchRunResult: Codable {
    let model: String
    let timestamp: String
    let gitSha: String?
    let maxTokensOverride: Int?
    let scenarios: [BenchScenario]

    private enum CodingKeys: String, CodingKey {
        case model
        case timestamp
        case gitSha = "git_sha"
        case maxTokensOverride = "max_tokens_override"
        case scenarios
    }
}

private struct BenchmarkPrompt {
    let name: String
    let category: String
    let messages: [OLMLX.Message]
}

private enum PromptSet {
    static var all: [BenchmarkPrompt] {
        [
            BenchmarkPrompt(
                name: "factual",
                category: "factual",
                messages: [Message(role: "user", content: "What is the capital of France?")]
            ),
            BenchmarkPrompt(
                name: "reasoning",
                category: "reasoning",
                messages: [
                    Message(
                        role: "user",
                        content:
                            "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"
                    )
                ]
            ),
            BenchmarkPrompt(
                name: "coding",
                category: "coding",
                messages: [
                    Message(role: "user", content: "Write a Python function to check if a string is a palindrome.")
                ]
            ),
            BenchmarkPrompt(
                name: "creative",
                category: "creative",
                messages: [
                    Message(role: "user", content: "Write a short poem about programming in the style of a haiku.")
                ]
            ),
            BenchmarkPrompt(
                name: "instruction",
                category: "instruction",
                messages: [Message(role: "user", content: "List three benefits of regular exercise.")]
            ),
        ]
    }
}

private enum BenchUtils {
    static func getGitSHA() -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["rev-parse", "HEAD"]
        let pipe = Pipe()
        process.standardOutput = pipe
        do {
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return nil
        }
    }

    static func formattedTimestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        formatter.timeZone = TimeZone(abbreviation: "UTC")
        return formatter.string(from: Date())
    }

    static func pad(_ s: String, _ width: Int) -> String {
        if s.count >= width { return String(s.prefix(width)) }
        return s + String(repeating: " ", count: width - s.count)
    }
}

enum BenchmarkError: Error, CustomStringConvertible {
    case noCompletionInfo
    var description: String { "No completion info returned from generation" }
}

public enum BenchCommands {
    public static func run(model: String, maxTokens: Int, outputDir: String?) async throws {
        let settings = Settings()
        let registry = ModelRegistry(configPath: settings.modelsConfig)
        try await registry.load()
        let store = ModelStore(modelsDir: settings.modelsDir, registry: registry)
        let engine = DefaultInferenceEngine()
        let manager = ModelManager(
            registry: registry,
            store: store,
            settings: settings,
            inferenceEngine: engine
        )

        print("Loading model \(model)...")
        let loaded = try await manager.ensureLoaded(name: model)
        guard let container = loaded.container else {
            print("Error: model loaded but no container available")
            throw ExitCode.failure
        }
        print("Model loaded. Running benchmark prompts...")

        let prompts = PromptSet.all
        var promptResults: [BenchPromptResult] = []

        for prompt in prompts {
            print("  Running prompt: \(prompt.name) (\(prompt.category))...")
            do {
                let rawMessages = messagesToRaw(prompt.messages)
                let params = GenerateParameters(
                    maxTokens: maxTokens,
                    temperature: 0.0
                )
                let (text, info) = try await runGeneration(
                    container: container,
                    messages: rawMessages,
                    tools: nil,
                    parameters: params
                )
                guard let info else {
                    throw BenchmarkError.noCompletionInfo
                }

                let promptEvalDurationNs = Int(info.promptTime * 1_000_000_000)
                let evalDurationNs = Int(info.generateTime * 1_000_000_000)
                let totalDurationNs = promptEvalDurationNs + evalDurationNs

                let tokensPerSecond =
                    evalDurationNs > 0
                    ? (Double(info.generationTokenCount) * 1_000_000_000) / Double(evalDurationNs) : 0.0

                let promptTokensPerSecond =
                    promptEvalDurationNs > 0
                    ? (Double(info.promptTokenCount) * 1_000_000_000) / Double(promptEvalDurationNs) : 0.0

                let result = BenchPromptResult(
                    promptName: prompt.name,
                    category: prompt.category,
                    outputText: text,
                    statusCode: 200,
                    error: nil,
                    evalCount: info.generationTokenCount,
                    evalDurationNs: evalDurationNs,
                    promptEvalCount: info.promptTokenCount,
                    promptEvalDurationNs: promptEvalDurationNs,
                    totalDurationNs: totalDurationNs,
                    tokensPerSecond: tokensPerSecond,
                    promptTokensPerSecond: promptTokensPerSecond
                )
                promptResults.append(result)
                print("    Tokens: \(info.generationTokenCount), Tokens/s: \(String(format: "%.1f", tokensPerSecond))")
            } catch {
                print("    Error: \(error)")
                let result = BenchPromptResult(
                    promptName: prompt.name,
                    category: prompt.category,
                    outputText: "",
                    statusCode: 500,
                    error: "\(error)",
                    evalCount: 0,
                    evalDurationNs: 0,
                    promptEvalCount: 0,
                    promptEvalDurationNs: 0,
                    totalDurationNs: 0,
                    tokensPerSecond: 0.0,
                    promptTokensPerSecond: 0.0
                )
                promptResults.append(result)
            }
        }

        let gitSHA = BenchUtils.getGitSHA()
        let timestamp = BenchUtils.formattedTimestamp()

        let scenario = BenchScenario(
            scenarioName: "baseline",
            scenarioDescription: "Default settings",
            envOverrides: [:],
            promptResults: promptResults,
            skipped: false,
            skipReason: nil
        )

        let runResult = BenchRunResult(
            model: model,
            timestamp: timestamp,
            gitSha: gitSHA,
            maxTokensOverride: maxTokens != 512 ? maxTokens : nil,
            scenarios: [scenario]
        )

        let benchDir: URL
        if let customDir = outputDir {
            benchDir = URL(fileURLWithPath: (customDir as NSString).expandingTildeInPath)
        } else {
            benchDir = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".olmlx")
                .appendingPathComponent("bench")
                .appendingPathComponent("runs")
                .appendingPathComponent(timestamp)
        }

        try FileManager.default.createDirectory(at: benchDir, withIntermediateDirectories: true)
        let jsonURL = benchDir.appendingPathComponent("results.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(runResult)
        try data.write(to: jsonURL)

        print("Benchmark results written to \(jsonURL.path)")
        print("")

        let header = [
            BenchUtils.pad("Prompt", 20), BenchUtils.pad("Tokens", 8), BenchUtils.pad("Tok/s", 10),
            BenchUtils.pad("PTok/s", 10), BenchUtils.pad("Duration", 10),
        ].joined(separator: " ")
        print(header)
        print(String(repeating: "-", count: header.count))
        for r in promptResults {
            let durationSec = String(format: "%.4f", Double(r.totalDurationNs) / 1_000_000_000.0)
            let line = [
                BenchUtils.pad(r.promptName, 20), BenchUtils.pad(String(r.evalCount), 8),
                BenchUtils.pad(String(format: "%.1f", r.tokensPerSecond), 10),
                BenchUtils.pad(String(format: "%.1f", r.promptTokensPerSecond), 10),
                BenchUtils.pad(durationSec, 10),
            ].joined(separator: " ")
            print(line)
        }
        print(String(repeating: "-", count: header.count))
    }

    public static func list(runsDir: String?) async throws {
        let dir: URL
        if let customDir = runsDir {
            dir = URL(fileURLWithPath: (customDir as NSString).expandingTildeInPath)
        } else {
            dir = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".olmlx")
                .appendingPathComponent("bench")
                .appendingPathComponent("runs")
        }

        guard FileManager.default.fileExists(atPath: dir.path) else {
            print("No benchmark runs found at \(dir.path)")
            return
        }

        let contents = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        let runDirs =
            contents
            .filter { $0.hasDirectoryPath }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }

        if runDirs.isEmpty {
            print("No benchmark runs found")
            return
        }

        print("Found \(runDirs.count) benchmark run(s):")
        print("")

        for runDir in runDirs {
            let resultsFile = runDir.appendingPathComponent("results.json")
            guard FileManager.default.fileExists(atPath: resultsFile.path),
                let data = try? Data(contentsOf: resultsFile),
                let result = try? JSONDecoder().decode(BenchRunResult.self, from: data)
            else {
                print("  \(runDir.lastPathComponent): (could not read results)")
                continue
            }

            let dateStr = runDir.lastPathComponent
            print("  Run: \(dateStr)")
            print("  Model: \(result.model)")
            if let sha = result.gitSha {
                print("  Git SHA: \(sha)")
            }
            print("")

            for scenario in result.scenarios {
                print("  Scenario: \(scenario.scenarioName)")
                let header = [
                    BenchUtils.pad("Prompt", 20), BenchUtils.pad("Tokens", 8),
                    BenchUtils.pad("Tok/s", 10), BenchUtils.pad("Status", 8),
                ].joined(separator: " ")
                print(header)
                print(String(repeating: "-", count: header.count))
                for pr in scenario.promptResults {
                    let status = pr.statusCode == 200 ? "OK" : "ERR"
                    let line = [
                        BenchUtils.pad(pr.promptName, 20), BenchUtils.pad(String(pr.evalCount), 8),
                        BenchUtils.pad(String(format: "%.1f", pr.tokensPerSecond), 10),
                        BenchUtils.pad(status, 8),
                    ].joined(separator: " ")
                    print(line)
                }
                print("")
            }
            print(String(repeating: "=", count: 60))
        }
    }
}
