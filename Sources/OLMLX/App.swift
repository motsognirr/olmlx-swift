import Vapor
import Foundation
import MLXLMCommon

public func createApp(registry: ModelRegistry, store: ModelStore, manager: ModelManager) throws -> Application {
    var env = try Environment.detect()
    try LoggingSystem.bootstrap(from: &env)
    let app = Application(env)

    app.storage[ModelRegistryKey.self] = registry
    app.storage[ModelStoreKey.self] = store
    app.storage[ModelManagerKey.self] = manager

    let corsConfig = CORSMiddleware.Configuration(
        allowedOrigin: .custom(["http://localhost:*", "http://127.0.0.1:*"].joined(separator: ",")),
        allowedMethods: [.GET, .POST, .PUT, .OPTIONS, .DELETE, .PATCH, .HEAD],
        allowedHeaders: [.accept, .authorization, .contentType, .origin, .xRequestedWith, .userAgent]
    )
    app.middleware.use(CORSMiddleware(configuration: corsConfig))
    app.middleware.use(RequestIDMiddleware())

    // Ollama API
    app.get { req -> String in "Ollama is running" }
    app.get("api", "version") { req -> VersionResponse in
        VersionResponse(version: "0.1.0")
    }
    app.get("api", "tags") { req async throws -> TagsResponse in
        let registry = req.application.storage[ModelRegistryKey.self]!
        let names = registry.listModels()
        let models = names.compactMap { name -> ModelInfo? in
            guard let config = registry.resolve(name) else { return nil }
            return ModelInfo(name: name, model: config.hfPath)
        }
        return TagsResponse(models: models)
    }
    app.on(.POST, "api", "show") { req async throws -> ShowResponse in
        let body = try req.content.decode(ShowRequest.self)
        let registry = req.application.storage[ModelRegistryKey.self]!
        let store = req.application.storage[ModelStoreKey.self]!

        guard let config = registry.resolve(body.model) else {
            throw Abort(.notFound, reason: "model not found")
        }

        let manifest = try? store.show(name: body.model)

        var modelInfo: [String: AnyCodable] = [:]
        if let m = manifest {
            modelInfo["general.architecture"] = .string(m.family)
            modelInfo["general.parameter_count"] = .string(m.parameterSize)
            modelInfo["general.quantization_version"] = .string(m.quantizationLevel)
            modelInfo["general.file_type"] = .string(m.format)
        }

        var details: ModelDetails?
        if let m = manifest {
            details = ModelDetails(
                parentModel: config.hfPath,
                format: m.format,
                family: m.family,
                parameterSize: m.parameterSize,
                quantizationLevel: m.quantizationLevel
            )
        }

        return ShowResponse(
            details: details ?? ModelDetails(),
            modelInfo: modelInfo,
            modifiedAt: manifest?.modifiedAt ?? ""
        )
    }
    app.on(.POST, "api", "chat") { req async throws -> Response in
        let body = try req.content.decode(ChatRequest.self)
        let manager = req.application.storage[ModelManagerKey.self]!
        let model = try await manager.ensureLoaded(name: body.model, keepAlive: body.keepAlive)

        guard let container = model.container else {
            return chatFallbackResponse(model: body.model, stream: body.stream, from: model)
        }

        let params = mapToGenerateParameters(from: body.options)
        let rawMessages = messagesToRaw(body.messages)
        let rawTools = toolsToRaw(body.tools)

        if body.stream {
            return streamingChatResponse(
                modelName: body.model, container: container,
                messages: rawMessages, tools: rawTools, parameters: params)
        } else {
            return try await fullChatResponse(
                modelName: body.model, container: container,
                messages: rawMessages, tools: rawTools, parameters: params)
        }
    }
    app.on(.POST, "api", "generate") { req async throws -> Response in
        let body = try req.content.decode(GenerateRequest.self)
        let manager = req.application.storage[ModelManagerKey.self]!
        let model = try await manager.ensureLoaded(name: body.model, keepAlive: body.keepAlive)

        guard let container = model.container else {
            return generateFallbackResponse(model: body.model, stream: body.stream, from: model)
        }

        let params = mapToGenerateParameters(from: body.options)
        let rawMessages: [[String: any Sendable]] = [
            ["role": "user", "content": body.prompt]
        ]

        if body.stream {
            return streamingChatResponse(
                modelName: body.model, container: container,
                messages: rawMessages, tools: nil, parameters: params)
        } else {
            return try await fullChatResponse(
                modelName: body.model, container: container,
                messages: rawMessages, tools: nil, parameters: params)
        }
    }
    app.on(.POST, "api", "embed") { req async throws -> EmbedResponse in
        let body = try req.content.decode(EmbedRequest.self)
        return EmbedResponse(model: body.model, embeddings: [[0.1, 0.2, 0.3]])
    }
    app.on(.POST, "api", "embeddings") { req async throws -> EmbeddingsResponse in
        return EmbeddingsResponse(embedding: [0.1, 0.2, 0.3])
    }
    app.on(.POST, "api", "pull") { req async throws -> PullResponse in
        return PullResponse(status: "not implemented", digest: nil, total: nil, completed: nil)
    }
    app.on(.POST, "api", "copy") { _ -> HTTPStatus in .ok }
    app.on(.DELETE, "api", "delete") { req -> HTTPStatus in
        let body = try req.content.decode(DeleteRequest.self)
        let store = req.application.storage[ModelStoreKey.self]!
        do { try store.delete(name: body.model) } catch { throw Abort(.notFound) }
        return .ok
    }
    app.on(.POST, "api", "create") { _ -> HTTPStatus in .ok }
    app.on(.POST, "api", "warmup") { req async throws -> HTTPStatus in
        let body = try req.content.decode(WarmupRequest.self)
        let manager = req.application.storage[ModelManagerKey.self]!
        _ = try await manager.ensureLoaded(name: body.model, keepAlive: body.keepAlive)
        return .ok
    }
    app.on(.POST, "api", "abort") { _ -> HTTPStatus in .ok }
    app.on(.POST, "api", "unload") { req -> HTTPStatus in
        let body = try req.content.decode(UnloadRequest.self)
        let manager = req.application.storage[ModelManagerKey.self]!
        manager.unload(name: body.model)
        return .ok
    }
    app.get("api", "ps") { req -> PsResponse in
        let manager = req.application.storage[ModelManagerKey.self]!
        let models = manager.getLoaded().map { lm in
            RunningModel(name: lm.name, model: lm.hfPath)
        }
        return PsResponse(models: models)
    }
    app.on(.HEAD, "api", "blobs", ":digest") { _ -> HTTPStatus in .ok }
    app.on(.POST, "api", "blobs", ":digest") { req -> HTTPStatus in
        guard let digest = req.parameters.get("digest") else { throw Abort(.badRequest) }
        let store = req.application.storage[ModelStoreKey.self]!
        if let buffer = req.body.data {
            let data = Data(buffer: buffer)
            try store.saveBlob(digest: digest, data: data)
        }
        return .ok
    }

    // OpenAI-compatible API
    app.on(.POST, "v1", "chat", "completions") { req async throws -> OpenAIChatResponse in
        let body = try req.content.decode(OpenAIChatRequest.self)
        let manager = req.application.storage[ModelManagerKey.self]!
        let model = try await manager.ensureLoaded(name: body.model, keepAlive: nil)

        guard let container = model.container else {
            return OpenAIChatResponse(
                id: "chatcmpl-\(UUID().uuidString)",
                created: Int(Date().timeIntervalSince1970),
                model: body.model,
                choices: [
                    OpenAIChoice(
                        index: 0,
                        message: OpenAIChatMessage(role: "assistant", content: "Hello!"),
                        finishReason: "stop"
                    )
                ],
                usage: OpenAIUsage(promptTokens: 10, completionTokens: 2, totalTokens: 12)
            )
        }

        var params = GenerateParameters()
        if let t = body.temperature { params.temperature = Float(t) }
        if let p = body.topP { params.topP = Float(p) }
        if let m = body.maxTokens { params.maxTokens = m }

        let rawMessages: [[String: any Sendable]] = body.messages.map { msg in
            var dict: [String: any Sendable] = ["role": msg.role]
            if let content = msg.content { dict["content"] = content }
            return dict
        }

        let (text, info) = try await runGeneration(
            container: container,
            messages: rawMessages,
            tools: nil,
            parameters: params
        )

        return OpenAIChatResponse(
            id: "chatcmpl-\(UUID().uuidString)",
            created: Int(Date().timeIntervalSince1970),
            model: body.model,
            choices: [
                OpenAIChoice(
                    index: 0,
                    message: OpenAIChatMessage(role: "assistant", content: text),
                    finishReason: "stop"
                )
            ],
            usage: OpenAIUsage(
                promptTokens: info?.promptTokenCount ?? 0,
                completionTokens: info?.generationTokenCount ?? 0,
                totalTokens: (info?.promptTokenCount ?? 0) + (info?.generationTokenCount ?? 0)
            )
        )
    }
    app.on(.POST, "v1", "completions") { req -> OpenAICompletionResponse in
        let body = try req.content.decode(OpenAICompletionRequest.self)
        return OpenAICompletionResponse(
            id: "cmpl-\(UUID().uuidString)",
            created: Int(Date().timeIntervalSince1970),
            model: body.model,
            choices: [
                OpenAICompletionChoice(index: 0, text: "Hello!", finishReason: "stop")
            ]
        )
    }
    app.get("v1", "models") { req -> OpenAIModelList in
        let registry = req.application.storage[ModelRegistryKey.self]!
        let data = registry.listModels().map { OpenAIModel(id: $0) }
        return OpenAIModelList(data: data)
    }
    app.on(.POST, "v1", "embeddings") { req -> OpenAIEmbeddingResponse in
        let body = try req.content.decode(OpenAIEmbeddingRequest.self)
        return OpenAIEmbeddingResponse(
            data: [OpenAIEmbeddingData(index: 0, embedding: [0.1, 0.2, 0.3])],
            model: body.model
        )
    }

    // Anthropic-compatible API
    app.on(.POST, "v1", "messages") { req async throws -> AnthropicMessagesResponse in
        let body = try req.content.decode(AnthropicMessagesRequest.self)
        let manager = req.application.storage[ModelManagerKey.self]!
        let model = try await manager.ensureLoaded(name: body.model, keepAlive: nil)

        guard let container = model.container else {
            return AnthropicMessagesResponse(
                id: "msg_\(UUID().uuidString)",
                content: [AnthropicContentBlock(type: "text", text: "Hello!")],
                model: body.model,
                usage: AnthropicUsage(inputTokens: 10, outputTokens: 2)
            )
        }

        var params = GenerateParameters()
        if let t = body.temperature { params.temperature = Float(t) }
        if let p = body.topP { params.topP = Float(p) }
        if let k = body.topK { params.topK = k }
        params.maxTokens = body.maxTokens

        let rawMessages: [[String: any Sendable]] = body.messages.map { msg in
            var dict: [String: any Sendable] = ["role": msg.role]
            switch msg.content {
            case .string(let s): dict["content"] = s
            case .blocks: dict["content"] = ""
            }
            return dict
        }

        let (text, info) = try await runGeneration(
            container: container,
            messages: rawMessages,
            tools: nil,
            parameters: params
        )

        return AnthropicMessagesResponse(
            id: "msg_\(UUID().uuidString)",
            content: [AnthropicContentBlock(type: "text", text: text)],
            model: body.model,
            usage: AnthropicUsage(
                inputTokens: info?.promptTokenCount ?? 0,
                outputTokens: info?.generationTokenCount ?? 0
            )
        )
    }
    app.on(.POST, "v1", "messages", "count_tokens") { req -> AnthropicTokenCountResponse in
        return AnthropicTokenCountResponse(inputTokens: 10)
    }

    return app
}

// MARK: - Chat Response Helpers

private func fullChatResponse(
    modelName: String, container: ModelContainer,
    messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
    parameters: GenerateParameters
) async throws -> Response {
    let (text, info) = try await runGeneration(
        container: container, messages: messages, tools: tools, parameters: parameters)

    let msg = ChatResponse(
        model: modelName,
        createdAt: ISO8601DateFormatter().string(from: Date()),
        message: Message(role: "assistant", content: text),
        done: true,
        doneReason: "stop",
        totalDuration: info.map { Int($0.promptTime + $0.generateTime) * 1_000_000_000 },
        promptEvalCount: info?.promptTokenCount,
        evalCount: info?.generationTokenCount
    )
    let resp = Response(status: .ok)
    try resp.content.encode(msg, as: .json)
    return resp
}

private func streamingChatResponse(
    modelName: String, container: ModelContainer,
    messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
    parameters: GenerateParameters
) -> Response {
    let response = Response(status: .ok)
    response.headers.contentType = .init(type: "application", subType: "x-ndjson")
    response.body = .init(stream: { writer in
        Task {
            let encoder = JSONEncoder()
            do {
                try await runStreamingGeneration(
                    container: container, messages: messages, tools: tools,
                    parameters: parameters,
                    onChunk: { text in
                        let msg = ChatResponse(
                            model: modelName,
                            createdAt: ISO8601DateFormatter().string(from: Date()),
                            message: Message(role: "assistant", content: text),
                            done: false
                        )
                        if let data = try? encoder.encode(msg),
                           let json = String(data: data, encoding: .utf8) {
                            _ = writer.write(.buffer(.init(string: json + "\n")))
                        }
                    },
                    onComplete: { info in
                        let msg = ChatResponse(
                            model: modelName,
                            createdAt: ISO8601DateFormatter().string(from: Date()),
                            message: Message(role: "assistant", content: ""),
                            done: true,
                            doneReason: "stop",
                            totalDuration: info.map { Int($0.promptTime + $0.generateTime) * 1_000_000_000 },
                            promptEvalCount: info?.promptTokenCount,
                            evalCount: info?.generationTokenCount
                        )
                        if let data = try? encoder.encode(msg),
                           let json = String(data: data, encoding: .utf8) {
                            _ = writer.write(.buffer(.init(string: json + "\n")))
                        }
                        _ = writer.write(.end)
                    }
                )
            } catch {
                let errMsg = ChatResponse(
                    model: modelName,
                    createdAt: ISO8601DateFormatter().string(from: Date()),
                    message: Message(role: "assistant", content: "Error: \(error)"),
                    done: true,
                    doneReason: "error"
                )
                if let data = try? encoder.encode(errMsg),
                   let json = String(data: data, encoding: .utf8) {
                    _ = writer.write(.buffer(.init(string: json + "\n")))
                }
                _ = writer.write(.end)
            }
        }
    })
    return response
}

// MARK: - Fallback Responses (no MLX engine configured)

private func chatFallbackResponse(model: String, stream: Bool, from lm: LoadedModel) -> Response {
    if stream {
        let response = Response(status: .ok)
        response.headers.contentType = .init(type: "application", subType: "x-ndjson")
        response.body = .init(stream: { writer in
            Task {
                let encoder = JSONEncoder()
                let msg = ChatResponse(
                    model: model,
                    createdAt: ISO8601DateFormatter().string(from: Date()),
                    message: Message(role: "assistant", content: "Hello from \(lm.name)"),
                    done: true,
                    doneReason: "stop"
                )
                if let data = try? encoder.encode(msg),
                   let json = String(data: data, encoding: .utf8) {
                    _ = writer.write(.buffer(.init(string: json + "\n")))
                }
                _ = writer.write(.end)
            }
        })
        return response
    } else {
        let msg = ChatResponse(
            model: model,
            createdAt: ISO8601DateFormatter().string(from: Date()),
            message: Message(role: "assistant", content: "Hello from \(lm.name)"),
            done: true,
            doneReason: "stop"
        )
        let resp = Response(status: .ok)
        try? resp.content.encode(msg, as: .json)
        return resp
    }
}

private func generateFallbackResponse(model: String, stream: Bool, from lm: LoadedModel) -> Response {
    if stream {
        let response = Response(status: .ok)
        response.headers.contentType = .init(type: "application", subType: "x-ndjson")
        response.body = .init(stream: { writer in
            Task {
                let encoder = JSONEncoder()
                let msg = GenerateResponse(
                    model: model,
                    createdAt: ISO8601DateFormatter().string(from: Date()),
                    response: "Hello from \(lm.name)",
                    done: true,
                    doneReason: "stop"
                )
                if let data = try? encoder.encode(msg),
                   let json = String(data: data, encoding: .utf8) {
                    _ = writer.write(.buffer(.init(string: json + "\n")))
                }
                _ = writer.write(.end)
            }
        })
        return response
    } else {
        let msg = GenerateResponse(
            model: model,
            createdAt: ISO8601DateFormatter().string(from: Date()),
            response: "Hello from \(lm.name)",
            done: true,
            doneReason: "stop"
        )
        let resp = Response(status: .ok)
        try? resp.content.encode(msg, as: .json)
        return resp
    }
}

// MARK: - Storage Keys

struct ModelRegistryKey: StorageKey { typealias Value = ModelRegistry }
struct ModelStoreKey: StorageKey { typealias Value = ModelStore }
struct ModelManagerKey: StorageKey { typealias Value = ModelManager }

public struct RequestIDMiddleware: AsyncMiddleware {
    public func respond(to request: Request, chainingTo next: AsyncResponder) async throws -> Response {
        let requestID = UUID().uuidString
        request.storage[RequestIDKey.self] = requestID
        let response = try await next.respond(to: request)
        response.headers.add(name: "X-Request-ID", value: requestID)
        return response
    }
}

struct RequestIDKey: StorageKey { typealias Value = String }
