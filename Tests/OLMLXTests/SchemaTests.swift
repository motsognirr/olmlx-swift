import Foundation
import Testing

@testable import OLMLX

@Suite("Chat Schema")
struct ChatSchemaTests {
    @Test func chatRequestValid() throws {
        let msg = Message(role: "user", content: "hello")
        let req = try ChatRequest(model: "test-model", messages: [msg])
        #expect(req.model == "test-model")
        #expect(req.messages.count == 1)
        #expect(req.stream == true)
    }

    @Test func chatRequestEmptyMessages() {
        #expect(throws: SchemaValidationError.self) {
            _ = try ChatRequest(model: "test", messages: [])
        }
    }

    @Test func chatResponseRoundtrip() throws {
        let json = """
        {"model":"test","created_at":"2024-01-01","message":{"role":"assistant","content":"hi"},"done":true}
        """
        let data = json.data(using: .utf8)!
        let resp = try JSONDecoder().decode(ChatResponse.self, from: data)
        #expect(resp.model == "test")
        #expect(resp.done == true)
        #expect(resp.message.role == "assistant")
        #expect(resp.message.content == "hi")
    }
}

@Suite("Generate Schema")
struct GenerateSchemaTests {
    @Test func generateRequestValid() throws {
        let req = try GenerateRequest(model: "test", prompt: "hello")
        #expect(req.model == "test")
        #expect(req.prompt == "hello")
        #expect(req.stream == true)
    }

    @Test func generateRequestEmptyPrompt() {
        #expect(throws: SchemaValidationError.self) {
            _ = try GenerateRequest(model: "test", prompt: "")
        }
    }

    @Test func generateResponseRoundtrip() throws {
        let json = """
        {"model":"test","created_at":"2024-01-01","response":"hi","done":true}
        """
        let data = json.data(using: .utf8)!
        let resp = try JSONDecoder().decode(GenerateResponse.self, from: data)
        #expect(resp.model == "test")
        #expect(resp.response == "hi")
        #expect(resp.done == true)
    }
}

@Suite("Embed Schema")
struct EmbedSchemaTests {
    @Test func embedRequestString() throws {
        let req = try EmbedRequest(model: "test", input: .string("hello"))
        if case .string(let s) = req.input {
            #expect(s == "hello")
        } else {
            Issue.record("expected string input")
        }
    }

    @Test func embedRequestArray() throws {
        let req = try EmbedRequest(model: "test", input: .array(["a", "b"]))
        if case .array(let a) = req.input {
            #expect(a == ["a", "b"])
        } else {
            Issue.record("expected array input")
        }
    }

    @Test func embedRequestEmptyString() {
        #expect(throws: SchemaValidationError.self) {
            _ = try EmbedRequest(model: "test", input: .string(""))
        }
    }

    @Test func embedRequestEmptyArray() {
        #expect(throws: SchemaValidationError.self) {
            _ = try EmbedRequest(model: "test", input: .array([]))
        }
    }
}

@Suite("OpenAI Schema")
struct OpenAISchemaTests {
    @Test func chatRequestRoundtrip() throws {
        let json = """
        {"model":"test","messages":[{"role":"user","content":"hi"}],"stream":true}
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(req.model == "test")
        #expect(req.messages.count == 1)
        #expect(req.messages[0].role == "user")
        #expect(req.stream == true)
    }

    @Test func responseFormat() throws {
        let schema: [String: AnyCodable] = [
            "name": .string("test_schema"),
            "schema": .dictionary(["type": .string("object")])
        ]
        let rf = try ResponseFormat(type: "json_schema", jsonSchema: schema)
        #expect(rf.type == "json_schema")
    }

    @Test func responseFormatRequiresSchema() {
        #expect(throws: SchemaValidationError.self) {
            _ = try ResponseFormat(type: "json_schema", jsonSchema: nil)
        }
    }
}

@Suite("Anthropic Schema")
struct AnthropicSchemaTests {
    @Test func messageContent() throws {
        let msg = try AnthropicMessage(role: "user", content: .string("hello"))
        if case .string(let s) = msg.content {
            #expect(s == "hello")
        } else {
            Issue.record("expected string content")
        }
    }

    @Test func messagesRequest() throws {
        let json = """
        {"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":100}
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: data)
        #expect(req.model == "test")
        #expect(req.maxTokens == 100)
        #expect(req.messages.count == 1)
    }
}

@Suite("Models Schema")
struct ModelsSchemaTests {
    @Test func modelDetailsDefaults() {
        let md = ModelDetails()
        #expect(md.format == "mlx")
        #expect(md.family == "")
        #expect(md.parentModel == "")
    }

    @Test func modelInfoRoundtrip() throws {
        let json = """
        {"name":"llama2","model":"llama2:latest","size":1000,"digest":"abc123"}
        """
        let data = json.data(using: .utf8)!
        let info = try JSONDecoder().decode(ModelInfo.self, from: data)
        #expect(info.name == "llama2")
        #expect(info.size == 1000)
        #expect(info.digest == "abc123")
    }

    @Test func manageTypes() {
        let copyReq = CopyRequest(source: "src", destination: "dst")
        #expect(copyReq.source == "src")
        #expect(copyReq.destination == "dst")

        let delReq = DeleteRequest(model: "test")
        #expect(delReq.model == "test")

        let warmReq = WarmupRequest(model: "test", keepAlive: "10m")
        #expect(warmReq.model == "test")
        #expect(warmReq.keepAlive == "10m")
    }

    @Test func pullTypes() throws {
        let req = PullRequest(model: "test-model", stream: false)
        #expect(req.model == "test-model")
        #expect(req.stream == false)

        let json = #"{"status":"downloading","digest":"abc","total":100,"completed":50}"#
        let data = json.data(using: .utf8)!
        let resp = try JSONDecoder().decode(PullResponse.self, from: data)
        #expect(resp.status == "downloading")
        #expect(resp.digest == "abc")
        #expect(resp.total == 100)
        #expect(resp.completed == 50)
    }

    @Test func modelOptionsCodable() throws {
        let json = """
        {"temperature":0.7,"top_p":0.9,"seed":42}
        """
        let data = json.data(using: .utf8)!
        let opts = try JSONDecoder().decode(ModelOptions.self, from: data)
        #expect(opts.temperature == 0.7)
        #expect(opts.topP == 0.9)
        #expect(opts.seed == 42)
    }
}
