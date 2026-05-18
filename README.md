# olmlx

A drop-in Ollama API replacement powered by Apple's MLX framework.

## Building

Use `make build` (release) or `make dev-build` (debug). Both run `swift
build` and then stage `mlx.metallib` into `.build/<config>/` next to the
produced executable.

MLX loads `mlx.metallib` from the directory containing the running binary
at the first inference call. The `mlx-swift` SwiftPM package does not
declare the metallib as a SwiftPM resource, so plain `swift build` will
produce an executable but inference will then fail with:

> Failed to load the default metallib

Use `make build` to avoid this. The build script
(`scripts/build-metallib.sh`) compiles the in-tree mlx kernel sources
from the resolved `mlx-swift` checkout into `mlx.metallib`. The only
extra requirement beyond `swift build` is the Xcode command-line tools
(for `xcrun -sdk macosx metal` and `xcrun -sdk macosx metallib`).

### Common tasks

| Task | Command |
|------|---------|
| Release build with metallib | `make build` |
| Debug build with metallib | `make dev-build` |
| Run the server | `.build/release/olmlx serve` |
| Run the full check (lint + test + build + verify) | `make ci` |
| Clean everything | `make clean` |

## Running

```
.build/release/olmlx serve
```

See `olmlx --help` and `olmlx <subcommand> --help` for options.
