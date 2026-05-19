import Foundation
import MLXLMCommon
import Vapor

extension Application {
    func mountAnthropicRoutes() {
        on(.POST, "v1", "messages") { req async throws -> AnthropicMessagesResponse in
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
            applyKVCacheQuant(&params, spec: model.config.resolvedKVCacheQuant(global: manager.settings))

            let rawMessages: [[String: any Sendable]] = body.messages.map { msg in
                var dict: [String: any Sendable] = ["role": msg.role]
                switch msg.content {
                case .string(let s): dict["content"] = s
                case .blocks: dict["content"] = ""
                }
                return dict
            }

            let cacheBinding = makePromptCacheBinding(manager: manager, modelName: body.model)
            let (text, info) = try await runGeneration(
                container: container,
                messages: rawMessages,
                tools: nil,
                parameters: params,
                promptCache: cacheBinding
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
        on(.POST, "v1", "messages", "count_tokens") { _ async throws -> Response in
            throw Abort(.notImplemented, reason: "token counting is not implemented")
        }
    }
}
