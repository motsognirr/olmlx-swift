import ArgumentParser
import Foundation
import MLXLMCommon
import OLMLX
import OLMLXBench

@main
struct OLMXCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "olmlx",
        abstract: "Drop-in Ollama API replacement powered by Apple's MLX framework",
        version: "0.1.0",
        subcommands: [
            Serve.self,
            Models.self,
            Chat.self,
            Bench.self,
            Config.self,
            Service.self,
        ],
        defaultSubcommand: Serve.self
    )
}

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Start the API server"
    )

    @Option(name: .long, help: "Host to bind to")
    var host: String?

    @Option(name: .long, help: "Port to listen on")
    var port: Int?

    @Flag(name: .long, help: "Enable speculative decoding")
    var speculative: Bool = false

    @Option(name: .long, help: "KV cache quantization (e.g. turboquant:4)")
    var kvCacheQuant: String?

    mutating func run() async throws {
        // CLI flags override env. We translate the flag back into OLMLX_SPECULATIVE
        // so `Settings()` (which reads env exclusively) picks it up. Without this,
        // `--speculative` was parsed and then ignored.
        if speculative {
            setenv("OLMLX_SPECULATIVE", "true", 1)
        }

        let settings = Settings()
        let hostStr = host ?? settings.host
        let portNum = port ?? settings.port

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

        _ = manager.startExpiryChecker()

        let app = try createApp(registry: registry, store: store, manager: manager, settings: settings)

        print("olmlx v0.1.0 starting on http://\(hostStr):\(portNum)")
        app.http.server.configuration.hostname = hostStr
        app.http.server.configuration.port = portNum

        try await app.execute()
    }
}

struct Models: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Model management",
        subcommands: [ModelsList.self, ModelsShow.self, ModelsDelete.self, ModelsSearch.self]
    )
}

struct ModelsList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "list", abstract: "List registered models")

    mutating func run() async throws {
        let settings = Settings()
        let registry = ModelRegistry(configPath: settings.modelsConfig)
        try await registry.load()
        let models = await registry.listModels()
        if models.isEmpty {
            print("No models registered")
        } else {
            for name in models {
                print(name)
            }
        }
    }
}

struct ModelsShow: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "show", abstract: "Show model details")

    @Argument(help: "Model name")
    var model: String

    mutating func run() async throws {
        let settings = Settings()
        let registry = ModelRegistry(configPath: settings.modelsConfig)
        try await registry.load()
        let store = ModelStore(modelsDir: settings.modelsDir, registry: registry)

        guard let config = await registry.resolve(model) else {
            print("Model '\(model)' not found in registry")
            return
        }
        print("Model: \(model)")
        print("HF Path: \(config.hfPath)")
        if let spec = config.speculativeDraftModel { print("Draft: \(spec)") }
        if let kvq = config.resolvedKVCacheQuant(global: settings) { print("KV Cache Quant: \(kvq)") }

        if let manifest = try? await store.show(name: model) {
            print("Size: \(manifest.size) bytes")
            print("Family: \(manifest.family)")
            print("Parameters: \(manifest.parameterSize)")
            print("Quantization: \(manifest.quantizationLevel)")
        }
    }
}

struct ModelsDelete: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "delete", abstract: "Delete a model")

    @Argument(help: "Model name")
    var model: String

    mutating func run() async throws {
        let settings = Settings()
        let registry = ModelRegistry(configPath: settings.modelsConfig)
        try await registry.load()
        let store = ModelStore(modelsDir: settings.modelsDir, registry: registry)
        try await store.delete(name: model)
        await registry.remove(model)
        try await registry.save()
        print("Deleted model '\(model)'")
    }
}

struct ModelsSearch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Search models")

    @Argument(help: "Search query")
    var query: String

    mutating func run() async throws {
        let settings = Settings()
        let registry = ModelRegistry(configPath: settings.modelsConfig)
        try await registry.load()
        let results = await registry.search(query)
        if results.isEmpty {
            print("No models matching '\(query)'")
        } else {
            for name in results {
                print(name)
            }
        }
    }
}

struct Chat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Interactive terminal chat")

    @Argument(help: "Model name")
    var model: String

    @Option(name: .long, help: "System prompt")
    var system: String?

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 2048

    @Option(name: .long, help: "Temperature")
    var temperature: Double = 0.7

    mutating func run() async throws {
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

        print("olmlx chat with \(model)")
        print("Type /exit to quit, /help for commands")
        print("---")

        let loaded: LoadedModel
        do {
            loaded = try await manager.ensureLoaded(name: model)
        } catch {
            print("Failed to load model '\(model)': \(error)")
            return
        }
        guard let container = loaded.container else {
            print("Model '\(model)' has no inference container; cannot chat.")
            return
        }

        var systemPrompt: String? = system
        var conversation: [OLMLX.Message] = []
        if let sys = systemPrompt {
            conversation.append(OLMLX.Message(role: "system", content: sys))
        }

        while true {
            print("> ", terminator: "")
            guard let raw = readLine() else {
                print("")
                break
            }
            let input = raw.trimmingCharacters(in: .whitespaces)
            if input.isEmpty { continue }

            if input == "/exit" || input == "/quit" { break }
            if input == "/help" {
                print("Commands: /exit, /help, /clear, /system <prompt>")
                continue
            }
            if input == "/clear" {
                conversation.removeAll()
                if let sys = systemPrompt {
                    conversation.append(OLMLX.Message(role: "system", content: sys))
                }
                await manager.invalidatePromptCache(for: model)
                print("Conversation cleared.")
                continue
            }
            if input == "/system" || input.hasPrefix("/system ") {
                let newSys = String(input.dropFirst("/system".count))
                    .trimmingCharacters(in: .whitespaces)
                systemPrompt = newSys.isEmpty ? nil : newSys
                conversation.removeAll()
                if let sys = systemPrompt {
                    conversation.append(OLMLX.Message(role: "system", content: sys))
                    print("System prompt set; conversation cleared.")
                } else {
                    print("System prompt cleared; conversation cleared.")
                }
                await manager.invalidatePromptCache(for: model)
                continue
            }
            if input.hasPrefix("/") {
                print("Unknown command: \(input). Type /help for commands.")
                continue
            }

            conversation.append(OLMLX.Message(role: "user", content: input))

            var params = mapToGenerateParameters(from: nil)
            params.maxTokens = maxTokens
            params.temperature = Float(temperature)
            applyKVCacheQuant(
                &params,
                spec: loaded.config.resolvedKVCacheQuant(global: manager.settings)
            )

            let rawMessages = messagesToRaw(conversation)
            let cacheBinding = makePromptCacheBinding(manager: manager, modelName: model)

            print("Assistant: ", terminator: "")
            let collector = ChatTextCollector()
            do {
                try await runStreamingGeneration(
                    container: container,
                    messages: rawMessages,
                    tools: nil,
                    parameters: params,
                    onChunk: { text in
                        collector.append(text)
                        FileHandle.standardOutput.write(Data(text.utf8))
                    },
                    onComplete: { _ in },
                    promptCache: cacheBinding
                )
            } catch {
                print("\nError: \(error)")
                conversation.removeLast()
                continue
            }
            print("")
            conversation.append(OLMLX.Message(role: "assistant", content: collector.text))
        }
    }
}

final class ChatTextCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = ""

    func append(_ chunk: String) {
        lock.lock()
        buffer += chunk
        lock.unlock()
    }

    var text: String {
        lock.lock()
        defer { lock.unlock() }
        return buffer
    }
}

struct Bench: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Benchmarking",
        subcommands: [BenchRun.self, BenchList.self]
    )
}

struct BenchRun: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Run a benchmark")

    @Option(name: .long, help: "Model to benchmark")
    var model: String = ""

    @Option(name: .long, help: "Maximum tokens per prompt")
    var maxTokens: Int = 512

    @Option(name: .long, help: "Output directory for benchmark artifacts")
    var outputDir: String?

    mutating func run() async throws {
        try await BenchCommands.run(model: model, maxTokens: maxTokens, outputDir: outputDir)
    }
}

struct BenchList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "List benchmark results")

    @Option(name: .long, help: "Benchmark runs directory")
    var runsDir: String?

    mutating func run() async throws {
        try await BenchCommands.list(runsDir: runsDir)
    }
}

struct Config: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Configuration",
        subcommands: [ConfigShow.self]
    )
}

struct ConfigShow: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Show current configuration")

    mutating func run() throws {
        let s = Settings()
        print("Host: \(s.host):\(s.port)")
        print("Models Dir: \(s.modelsDir.path)")
        print("Models Config: \(s.modelsConfig.path)")
        print("Keep Alive: \(s.defaultKeepAlive)")
        print("Max Loaded: \(s.maxLoadedModels)")
        print("Memory Limit: \(s.memoryLimitFraction)")
        print("Prompt Cache: \(s.promptCache)")
        print("Speculative: \(s.speculative)")
        print("KV Cache Quant: \(s.kvCacheQuant ?? "none")")
        print("Sync Mode: \(s.syncMode.rawValue)")
        print("Log Level: \(s.logLevel)")
    }
}

// `Service` and its subcommands live in Service.swift.
