// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "olmlx-swift",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "OLMLX", targets: ["OLMLX"]),
        .executable(name: "olmlx", targets: ["olmlx-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/vapor/vapor", from: "4.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "OLMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Vapor", package: "vapor"),
            ]
        ),
        .testTarget(
            name: "OLMLXTests",
            dependencies: ["OLMLX"],
            path: "Tests/OLMLXTests"
        ),
        .executableTarget(
            name: "olmlx-cli",
            dependencies: [
                "OLMLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ]
)
