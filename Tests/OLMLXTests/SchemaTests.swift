import Foundation
import OLMLX
import OLMLXTestUtils

func OLMLXTests_schema_chatRequestValid() throws {
    let msg = Message(role: "user", content: "hello")
    let req = try ChatRequest(model: "test-model", messages: [msg])
    try expectEqual(req.model, "test-model")
    try expectEqual(req.messages.count, 1)
    try expectEqual(req.stream, true)
}

func OLMLXTests_schema_chatRequestEmptyMessages() throws {
    do {
        _ = try ChatRequest(model: "test", messages: [])
        throw TestError("expected error for empty messages")
    } catch is SchemaValidationError {
    }
}

func OLMLXTests_schema_chatResponseRoundtrip() throws {
    let json = """
    {"model":"test","created_at":"2024-01-01","message":{"role":"assistant","content":"hi"},"done":true}
    """
    let data = json.data(using: .utf8)!
    let resp = try JSONDecoder().decode(ChatResponse.self, from: data)
    try expectEqual(resp.model, "test")
    try expectEqual(resp.done, true)
    try expectEqual(resp.message.role, "assistant")
    try expectEqual(resp.message.content, "hi")
}

func OLMLXTests_schema_generateRequestValid() throws {
    let req = try GenerateRequest(model: "test", prompt: "hello")
    try expectEqual(req.model, "test")
    try expectEqual(req.prompt, "hello")
    try expectEqual(req.stream, true)
}

func OLMLXTests_schema_generateRequestEmptyPrompt() throws {
    do {
        _ = try GenerateRequest(model: "test", prompt: "")
        throw TestError("expected error for empty prompt")
    } catch is SchemaValidationError {
    }
}

func OLMLXTests_schema_generateResponseRoundtrip() throws {
    let json = """
    {"model":"test","created_at":"2024-01-01","response":"hi","done":true}
    """
    let data = json.data(using: .utf8)!
    let resp = try JSONDecoder().decode(GenerateResponse.self, from: data)
    try expectEqual(resp.model, "test")
    try expectEqual(resp.response, "hi")
    try expectEqual(resp.done, true)
}

func OLMLXTests_schema_embedRequestString() throws {
    let req = try EmbedRequest(model: "test", input: .string("hello"))
    switch req.input {
    case .string(let s): try expectEqual(s, "hello")
    case .array: throw TestError("expected string input")
    }
}

func OLMLXTests_schema_embedRequestArray() throws {
    let req = try EmbedRequest(model: "test", input: .array(["a", "b"]))
    switch req.input {
    case .array(let a): try expectEqual(a, ["a", "b"])
    case .string: throw TestError("expected array input")
    }
}

func OLMLXTests_schema_embedRequestEmpty() throws {
    do {
        _ = try EmbedRequest(model: "test", input: .string(""))
        throw TestError("expected error for empty string")
    } catch is SchemaValidationError {
    }
    do {
        _ = try EmbedRequest(model: "test", input: .array([]))
        throw TestError("expected error for empty array")
    } catch is SchemaValidationError {
    }
}

func OLMLXTests_schema_openaiChatRequestRoundtrip() throws {
    let json = """
    {"model":"test","messages":[{"role":"user","content":"hi"}],"stream":true}
    """
    let data = json.data(using: .utf8)!
    let req = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
    try expectEqual(req.model, "test")
    try expectEqual(req.messages.count, 1)
    try expectEqual(req.messages[0].role, "user")
    try expectEqual(req.stream, true)
}

func OLMLXTests_schema_openaiResponseFormat() throws {
    let schema: [String: AnyCodable] = [
        "name": .string("test_schema"),
        "schema": .dictionary(["type": .string("object")])
    ]
    let rf = try ResponseFormat(type: "json_schema", jsonSchema: schema)
    try expectEqual(rf.type, "json_schema")
}

func OLMLXTests_schema_openaiResponseFormatRequiresSchema() throws {
    do {
        _ = try ResponseFormat(type: "json_schema", jsonSchema: nil)
        throw TestError("expected error for missing json_schema")
    } catch is SchemaValidationError {
    }
}

func OLMLXTests_schema_anthropicMessageContent() throws {
    let msg = try AnthropicMessage(role: "user", content: .string("hello"))
    switch msg.content {
    case .string(let s): try expectEqual(s, "hello")
    case .blocks: throw TestError("expected string content")
    }
}

func OLMLXTests_schema_anthropicMessagesRequest() throws {
    let json = """
    {"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":100}
    """
    let data = json.data(using: .utf8)!
    let req = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: data)
    try expectEqual(req.model, "test")
    try expectEqual(req.maxTokens, 100)
    try expectEqual(req.messages.count, 1)
}

func OLMLXTests_schema_modelDetailsDefaults() throws {
    let md = ModelDetails()
    try expectEqual(md.format, "mlx")
    try expectEqual(md.family, "")
    try expectEqual(md.parentModel, "")
}

func OLMLXTests_schema_modelInfoRoundtrip() throws {
    let json = """
    {"name":"llama2","model":"llama2:latest","size":1000,"digest":"abc123"}
    """
    let data = json.data(using: .utf8)!
    let info = try JSONDecoder().decode(ModelInfo.self, from: data)
    try expectEqual(info.name, "llama2")
    try expectEqual(info.size, 1000)
    try expectEqual(info.digest, "abc123")
}

func OLMLXTests_schema_manageTypes() throws {
    let copyReq = CopyRequest(source: "src", destination: "dst")
    try expectEqual(copyReq.source, "src")
    try expectEqual(copyReq.destination, "dst")

    let delReq = DeleteRequest(model: "test")
    try expectEqual(delReq.model, "test")

    let warmReq = WarmupRequest(model: "test", keepAlive: "10m")
    try expectEqual(warmReq.model, "test")
    try expectEqual(warmReq.keepAlive, "10m")
}

func OLMLXTests_schema_pullTypes() throws {
    let req = PullRequest(model: "test-model", stream: false)
    try expectEqual(req.model, "test-model")
    try expectEqual(req.stream, false)

    let json = #"{"status":"downloading","digest":"abc","total":100,"completed":50}"#
    let data = json.data(using: .utf8)!
    let resp = try JSONDecoder().decode(PullResponse.self, from: data)
    try expectEqual(resp.status, "downloading")
    try expectEqual(resp.digest, "abc")
    try expectEqual(resp.total, 100)
    try expectEqual(resp.completed, 50)
}

func OLMLXTests_schema_modelOptionsCodable() throws {
    let json = """
    {"temperature":0.7,"top_p":0.9,"seed":42}
    """
    let data = json.data(using: .utf8)!
    let opts = try JSONDecoder().decode(ModelOptions.self, from: data)
    try expectEqual(opts.temperature, 0.7)
    try expectEqual(opts.topP, 0.9)
    try expectEqual(opts.seed, 42)
}
