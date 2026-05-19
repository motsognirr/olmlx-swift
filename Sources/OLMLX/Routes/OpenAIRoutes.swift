import Foundation
import MLXLMCommon
import Vapor

extension Application {
    func mountOpenAIRoutes() {
        on(.POST, "v1", "chat", "completions") { req async throws -> OpenAIChatResponse in
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
            applyKVCacheQuant(&params, spec: model.config.resolvedKVCacheQuant(global: manager.settings))

            let rawMessages: [[String: any Sendable]] = body.messages.map { msg in
                var dict: [String: any Sendable] = ["role": msg.role]
                if let content = msg.content { dict["content"] = content }
                return dict
            }

            let cacheBinding = makePromptCacheBinding(manager: manager, modelName: body.model)
            let speculative = try await resolveSpeculative(manager: manager, model: model)
            let (text, info) = try await runGeneration(
                container: container,
                messages: rawMessages,
                tools: nil,
                parameters: params,
                promptCache: cacheBinding,
                speculative: speculative
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
        on(.POST, "v1", "completions") { _ async throws -> Response in
            throw Abort(
                .notImplemented,
                reason: "legacy /v1/completions is not implemented; use /v1/chat/completions instead")
        }
        get("v1", "models") { req async throws -> OpenAIModelList in
            let registry = req.application.storage[ModelRegistryKey.self]!
            let data = await registry.listModels().map { OpenAIModel(id: $0) }
            return OpenAIModelList(data: data)
        }
        on(.POST, "v1", "embeddings") { _ async throws -> Response in
            throw Abort(.notImplemented, reason: "embedding endpoints are not implemented")
        }
    }
}
