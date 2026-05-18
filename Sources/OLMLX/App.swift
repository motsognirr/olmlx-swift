import Foundation
import Vapor

public func createApp(
    registry: ModelRegistry,
    store: ModelStore,
    manager: ModelManager,
    settings: Settings = Settings()
) throws -> Application {
    var env = try Environment.detect()
    try LoggingSystem.bootstrap(from: &env)
    let app = Application(env)

    app.storage[ModelRegistryKey.self] = registry
    app.storage[ModelStoreKey.self] = store
    app.storage[ModelManagerKey.self] = manager

    app.installDefaultMiddleware(settings: settings)

    app.mountOllamaRoutes()
    app.mountOpenAIRoutes()
    app.mountAnthropicRoutes()

    return app
}
