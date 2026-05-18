import Foundation
import Vapor

struct ModelRegistryKey: StorageKey { typealias Value = ModelRegistry }
struct ModelStoreKey: StorageKey { typealias Value = ModelStore }
struct ModelManagerKey: StorageKey { typealias Value = ModelManager }
struct RequestIDKey: StorageKey { typealias Value = String }

public struct RequestIDMiddleware: AsyncMiddleware {
    public func respond(to request: Request, chainingTo next: AsyncResponder) async throws -> Response {
        let requestID = UUID().uuidString
        request.storage[RequestIDKey.self] = requestID
        let response = try await next.respond(to: request)
        response.headers.add(name: "X-Request-ID", value: requestID)
        return response
    }
}

extension Application {
    func installDefaultMiddleware() {
        let corsConfig = CORSMiddleware.Configuration(
            allowedOrigin: .originBased,
            allowedMethods: [.GET, .POST, .PUT, .OPTIONS, .DELETE, .PATCH, .HEAD],
            allowedHeaders: [.accept, .authorization, .contentType, .origin, .xRequestedWith, .userAgent]
        )
        middleware.use(CORSMiddleware(configuration: corsConfig))
        middleware.use(RequestIDMiddleware())
    }
}
