import Foundation
import MLXLMCommon
import Vapor

extension Application {
    func mountOllamaRoutes() {
        get { _ -> String in "Ollama is running" }
        get("api", "version") { _ -> VersionResponse in
            VersionResponse(version: "0.1.0")
        }
        get("api", "tags") { req async throws -> TagsResponse in
            let registry = req.application.storage[ModelRegistryKey.self]!
            let names = await registry.listModels()
            var models: [ModelInfo] = []
            for name in names {
                if let config = await registry.resolve(name) {
                    models.append(ModelInfo(name: name, model: config.hfPath))
                }
            }
            return TagsResponse(models: models)
        }
        on(.POST, "api", "show") { req async throws -> ShowResponse in
            let body = try req.content.decode(ShowRequest.self)
            let registry = req.application.storage[ModelRegistryKey.self]!
            let store = req.application.storage[ModelStoreKey.self]!

            guard let config = await registry.resolve(body.model) else {
                throw Abort(.notFound, reason: "model not found")
            }

            let manifest = try? await store.show(name: body.model)

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
        on(.POST, "api", "chat") { req async throws -> Response in
            let body = try req.content.decode(ChatRequest.self)
            let manager = req.application.storage[ModelManagerKey.self]!
            let model = try await manager.ensureLoaded(name: body.model, keepAlive: body.keepAlive)

            guard let container = model.container else {
                return chatFallbackResponse(model: body.model, stream: body.stream, from: model)
            }

            var params = mapToGenerateParameters(from: body.options)
            applyKVCacheQuant(&params, spec: model.config.resolvedKVCacheQuant(global: manager.settings))
            let rawMessages = messagesToRaw(body.messages)
            let rawTools = toolsToRaw(body.tools)
            let cacheBinding = makePromptCacheBinding(manager: manager, modelName: body.model)
            let speculative = try await resolveSpeculative(manager: manager, model: model)

            if body.stream {
                return streamingChatResponse(
                    modelName: body.model, container: container,
                    messages: rawMessages, tools: rawTools, parameters: params,
                    promptCache: cacheBinding, speculative: speculative)
            } else {
                return try await fullChatResponse(
                    modelName: body.model, container: container,
                    messages: rawMessages, tools: rawTools, parameters: params,
                    promptCache: cacheBinding, speculative: speculative)
            }
        }
        on(.POST, "api", "generate") { req async throws -> Response in
            let body = try req.content.decode(GenerateRequest.self)
            let manager = req.application.storage[ModelManagerKey.self]!
            let model = try await manager.ensureLoaded(name: body.model, keepAlive: body.keepAlive)

            guard let container = model.container else {
                return generateFallbackResponse(model: body.model, stream: body.stream, from: model)
            }

            var params = mapToGenerateParameters(from: body.options)
            applyKVCacheQuant(&params, spec: model.config.resolvedKVCacheQuant(global: manager.settings))
            let rawMessages: [[String: any Sendable]] = [
                ["role": "user", "content": body.prompt]
            ]
            let cacheBinding = makePromptCacheBinding(manager: manager, modelName: body.model)
            let speculative = try await resolveSpeculative(manager: manager, model: model)

            if body.stream {
                return streamingChatResponse(
                    modelName: body.model, container: container,
                    messages: rawMessages, tools: nil, parameters: params,
                    promptCache: cacheBinding, speculative: speculative)
            } else {
                return try await fullChatResponse(
                    modelName: body.model, container: container,
                    messages: rawMessages, tools: nil, parameters: params,
                    promptCache: cacheBinding, speculative: speculative)
            }
        }
        on(.POST, "api", "embed") { _ async throws -> Response in
            throw Abort(.notImplemented, reason: "embedding endpoints are not implemented")
        }
        on(.POST, "api", "embeddings") { _ async throws -> Response in
            throw Abort(.notImplemented, reason: "embedding endpoints are not implemented")
        }
        on(.POST, "api", "pull") { req async throws -> PullResponse in
            let body = try req.content.decode(PullRequest.self)
            let registry = req.application.storage[ModelRegistryKey.self]!
            let store = req.application.storage[ModelStoreKey.self]!

            let hfPath: String
            if let config = await registry.resolve(body.model) {
                hfPath = config.hfPath
            } else {
                hfPath = body.model
            }

            do {
                _ = try await store.ensureDownloaded(hfPath: hfPath)
            } catch ModelStoreError.invalidHFPath(let path) {
                throw Abort(.badRequest, reason: "invalid HuggingFace path: \(path)")
            } catch ModelStoreError.downloadFailed(let path, let underlying) {
                throw Abort(
                    .internalServerError,
                    reason: "failed to download \(path): \(underlying)"
                )
            }
            return PullResponse(status: "success", digest: nil, total: nil, completed: nil)
        }
        on(.POST, "api", "copy") { _ async throws -> Response in
            throw Abort(.notImplemented, reason: "model copy is not implemented")
        }
        on(.DELETE, "api", "delete") { req async throws -> HTTPStatus in
            let body = try req.content.decode(DeleteRequest.self)
            let store = req.application.storage[ModelStoreKey.self]!
            do { try await store.delete(name: body.model) } catch { throw Abort(.notFound) }
            return .ok
        }
        on(.POST, "api", "create") { _ async throws -> Response in
            throw Abort(.notImplemented, reason: "model create is not implemented")
        }
        on(.POST, "api", "warmup") { req async throws -> HTTPStatus in
            let body = try req.content.decode(WarmupRequest.self)
            let manager = req.application.storage[ModelManagerKey.self]!
            _ = try await manager.ensureLoaded(name: body.model, keepAlive: body.keepAlive)
            return .ok
        }
        on(.POST, "api", "abort") { _ async throws -> Response in
            throw Abort(.notImplemented, reason: "request abort is not implemented")
        }
        on(.POST, "api", "unload") { req async throws -> HTTPStatus in
            let body = try req.content.decode(UnloadRequest.self)
            let manager = req.application.storage[ModelManagerKey.self]!
            await manager.unload(name: body.model)
            return .ok
        }
        get("api", "ps") { req async -> PsResponse in
            let manager = req.application.storage[ModelManagerKey.self]!
            let models = await manager.getLoaded().map { lm in
                RunningModel(name: lm.name, model: lm.hfPath)
            }
            return PsResponse(models: models)
        }
        on(.HEAD, "api", "blobs", ":digest") { _ -> HTTPStatus in .ok }
        on(.POST, "api", "blobs", ":digest") { req -> HTTPStatus in
            guard let digest = req.parameters.get("digest") else { throw Abort(.badRequest) }
            let store = req.application.storage[ModelStoreKey.self]!
            if let buffer = req.body.data {
                let data = Data(buffer: buffer)
                try store.saveBlob(digest: digest, data: data)
            }
            return .ok
        }
    }
}
