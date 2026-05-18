# Installation & Build

`olmlx-swift` targets **macOS 14+ on Apple Silicon** and is built with the
Swift 6 toolchain. There is no prebuilt binary at the moment — you build from
source.

## 1. Prerequisites

- **Hardware:** an Apple Silicon Mac (M1 or newer). MLX requires the unified
  GPU; Intel Macs cannot run inference.
- **OS:** macOS 14 (Sonoma) or later.
- **Toolchain:** Xcode 16 or the matching command-line tools (Swift 6.0).
  Verify with `swift --version`.
- **CMake:** required once to build the MLX metallib. Install with
  `brew install cmake` if you do not already have it.

## 2. Clone and build

```sh
git clone https://github.com/your-fork/olmlx-swift.git
cd olmlx-swift
swift build -c release
```

The Swift package builds the `olmlx` executable at
`.build/release/olmlx`. A debug build (`swift build` without `-c release`)
works for development; it lives at `.build/debug/olmlx`.

You can also use the bundled `Makefile`:

```sh
make build       # debug build
make test        # run unit tests
make lint        # swift-format + swiftlint
make format      # auto-format Sources/Tests/Package.swift
```

## 3. Build the Metal kernel library (required)

`mlx-swift` does not ship a `metallib` through SwiftPM today. A vanilla
`swift build` produces an executable that **links** against MLX but
**throws `MLX error: Failed to load the default metallib`** the first time
any GPU operation runs — which means the very first inference request.

You only need to do this once per checkout. After the SwiftPM dependencies are
resolved (the build above downloads them), compile the metallib via CMake and
drop it next to the binary:

```sh
REPO=$(pwd)

cd /tmp && rm -rf mlx-build && mkdir mlx-build && cd mlx-build
cmake "$REPO/.build/checkouts/mlx-swift/Source/Cmlx/mlx" \
  -DMLX_BUILD_METAL=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --target mlx-metallib

cp mlx/backend/metal/kernels/mlx.metallib "$REPO/.build/release/mlx.metallib"
```

The metallib must live in one of:

- next to the binary as `mlx.metallib` or `Resources/mlx.metallib`
- inside a SwiftPM bundle named `mlx-swift_Cmlx.bundle`
- at the path baked into the binary as `default_mtllib_path`

Plain `swift build` produces none of those, so this step is unavoidable until
mlx-swift gains a SwiftPM build plugin. Compiling the metallib takes a couple
of minutes the first time; subsequent runs are instant.

## 4. (Optional) Install on PATH

```sh
sudo cp .build/release/olmlx /usr/local/bin/olmlx
sudo cp .build/release/mlx.metallib /usr/local/bin/mlx.metallib
```

The metallib must travel with the binary — copy them together.

## 5. Verify the install

```sh
olmlx --version
# olmlx 0.1.0

olmlx config show
# Host: 0.0.0.0:11434
# Models Dir: /Users/<you>/.olmlx/models
# ...
```

If you see `olmlx 0.1.0` and a config dump, the build is good. Move on to
[Quickstart](quickstart.md).

## Common build issues

- **`error: Missing required modules: 'Cmlx'`** — the SwiftPM dependency graph
  isn't resolved. Run `swift package resolve` and try again.
- **`Failed to load the default metallib` at runtime** — you skipped step 3,
  or copied the binary without its sibling `mlx.metallib`.
- **`OLMXCLI` prints help and exits 0 for every subcommand** — see the
  [troubleshooting page](troubleshooting.md#every-subcommand-prints-help).
