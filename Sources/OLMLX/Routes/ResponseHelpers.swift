import Foundation
import MLXLMCommon
import Vapor

func fullChatResponse(
    modelName: String, container: ModelContainer,
    messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
    parameters: GenerateParameters,
    cacheContext: PromptCacheContext? = nil
) async throws -> Response {
    let (text, info) = try await runGeneration(
        container: container, messages: messages, tools: tools,
        parameters: parameters, cacheContext: cacheContext)

    let msg = ChatResponse(
        model: modelName,
        createdAt: ISO8601DateFormatter().string(from: Date()),
        message: Message(role: "assistant", content: text),
        done: true,
        doneReason: "stop",
        totalDuration: info.map { Int(($0.promptTime + $0.generateTime) * 1_000_000_000) },
        promptEvalCount: info?.promptTokenCount,
        promptEvalDuration: info.map { Int($0.promptTime * 1_000_000_000) },
        evalCount: info?.generationTokenCount,
        evalDuration: info.map { Int($0.generateTime * 1_000_000_000) }
    )
    let resp = Response(status: .ok)
    try resp.content.encode(msg, as: .json)
    return resp
}

func streamingChatResponse(
    modelName: String, container: ModelContainer,
    messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
    parameters: GenerateParameters,
    cacheContext: PromptCacheContext? = nil
) -> Response {
    let response = Response(status: .ok)
    response.headers.contentType = .init(type: "application", subType: "x-ndjson")
    response.body = .init(stream: { writer in
        Task {
            let encoder = JSONEncoder()
            do {
                try await runStreamingGeneration(
                    container: container, messages: messages, tools: tools,
                    parameters: parameters, cacheContext: cacheContext,
                    onChunk: { text in
                        let msg = ChatResponse(
                            model: modelName,
                            createdAt: ISO8601DateFormatter().string(from: Date()),
                            message: Message(role: "assistant", content: text),
                            done: false
                        )
                        if let data = try? encoder.encode(msg),
                            let json = String(data: data, encoding: .utf8)
                        {
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
                            totalDuration: info.map { Int(($0.promptTime + $0.generateTime) * 1_000_000_000) },
                            promptEvalCount: info?.promptTokenCount,
                            promptEvalDuration: info.map { Int($0.promptTime * 1_000_000_000) },
                            evalCount: info?.generationTokenCount,
                            evalDuration: info.map { Int($0.generateTime * 1_000_000_000) }
                        )
                        if let data = try? encoder.encode(msg),
                            let json = String(data: data, encoding: .utf8)
                        {
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
                    let json = String(data: data, encoding: .utf8)
                {
                    _ = writer.write(.buffer(.init(string: json + "\n")))
                }
                _ = writer.write(.end)
            }
        }
    })
    return response
}

func chatFallbackResponse(model: String, stream: Bool, from lm: LoadedModel) -> Response {
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
                    let json = String(data: data, encoding: .utf8)
                {
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

func generateFallbackResponse(model: String, stream: Bool, from lm: LoadedModel) -> Response {
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
                    let json = String(data: data, encoding: .utf8)
                {
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
