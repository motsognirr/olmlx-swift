// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "olmlx-swift",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "OLMLX", targets: ["OLMLX"]),
        .library(name: "OLMLXBench", targets: ["OLMLXBench"]),
        .executable(name: "olmlx", targets: ["olmlx-cli"]),
    ],
    dependencies: [
        // mlx-swift and mlx-swift-lm are tightly version-coupled (matching minor
        // numbers, 0.31.x / 3.31.x). Pin both to the next minor bump only.
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.31.3")),
        // Pre-1.0: a minor bump can still break the API, so allow patches only.
        .package(url: "https://github.com/huggingface/swift-huggingface", .upToNextMinor(from: "0.9.0")),
        // Post-1.0 packages: allow same-major versions.
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMajor(from: "1.3.0")),
        .package(url: "https://github.com/vapor/vapor", .upToNextMajor(from: "4.121.0")),
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMajor(from: "1.5.0")),
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
        .target(
            name: "OLMLXBench",
            dependencies: [
                "OLMLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "olmlx-cli",
            dependencies: [
                "OLMLX",
                "OLMLXBench",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
    ]
)
