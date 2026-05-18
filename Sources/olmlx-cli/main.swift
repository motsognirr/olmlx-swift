import ArgumentParser
import Foundation
import OLMLX

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

        let app = try createApp(registry: registry, store: store, manager: manager)

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

struct Chat: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Interactive terminal chat")

    @Argument(help: "Model name")
    var model: String

    @Option(name: .long, help: "System prompt")
    var system: String?

    @Option(name: .long, help: "Maximum tokens to generate")
    var maxTokens: Int = 2048

    @Option(name: .long, help: "Temperature")
    var temperature: Double = 0.7

    mutating func run() throws {
        print("olmlx chat with \(model)")
        print("Type /exit to quit, /help for commands")
        print("---")

        while true {
            print("> ", terminator: "")
            guard let input = readLine(), !input.isEmpty else { continue }
            if input == "/exit" { break }
            if input == "/help" {
                print("Commands: /exit, /help, /clear, /system <prompt>")
                continue
            }

            print("Assistant: [response from \(model)]")
        }
    }
}

struct Bench: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Benchmarking",
        subcommands: [BenchRun.self, BenchList.self]
    )
}

struct BenchRun: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Run a benchmark")

    @Option(name: .long, help: "Model to benchmark")
    var model: String?

    @Option(name: .long, help: "Maximum tokens")
    var maxTokens: Int = 128

    mutating func run() throws {
        print("Benchmark not fully implemented yet")
    }
}

struct BenchList: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "List benchmark results")

    mutating func run() throws {
        print("No benchmark results yet")
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

struct Service: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Launchd service management",
        subcommands: [ServiceInstall.self, ServiceStatus.self]
    )
}

struct ServiceInstall: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Install as a launchd service")

    mutating func run() throws {
        print("Service installation not implemented yet")
        print("On macOS, add to ~/Library/LaunchAgents/com.olmlx.plist")
    }
}

struct ServiceStatus: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Check service status")

    mutating func run() throws {
        print("Checking service status...")
    }
}

OLMXCLI.main()
