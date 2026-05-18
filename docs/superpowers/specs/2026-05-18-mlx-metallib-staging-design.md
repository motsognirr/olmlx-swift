# Stage `mlx.metallib` next to the `olmlx` binary at build time

Tracks: GitHub issue [#43](https://github.com/dpalmqvist/olmlx-swift/issues/43).

## Problem

After `swift build -c release`, running `.build/release/olmlx` fails at the first
inference call with `Failed to load the default metallib`. The `mlx-swift`
SwiftPM package does not declare `mlx.metallib` as a SwiftPM resource on the
`Cmlx` target, so it is never staged into the consumer's build output. MLX's
runtime then has nowhere to find the kernel library and Metal initialization
aborts.

The issue text proposes locating an existing `mlx.metallib` under
`.build/checkouts/`, but the mlx-swift checkout does **not** contain a
prebuilt `.metallib` — only the 9 `.metal` source files under
`Source/Cmlx/mlx-generated/metal/` together with their headers. Producing the
runtime artifact therefore requires compiling those sources ourselves.

## Goals

- `swift build -c release && .build/release/olmlx serve` works on a clean
  checkout, with no Python `mlx` install required, no manually copied files,
  and no edits to the `mlx-swift` package.
- The same fix works for `-c debug` builds.
- The build remains hermetic: only Xcode command-line tools (`xcrun -sdk
  macosx metal`, `metallib`) are required beyond what `swift build` already
  needs.
- The fix is discoverable from the existing `Makefile` and documented in the
  README.

## Non-goals

- Implementing `olmlx service install` (still a stub; out of scope).
- Filing an upstream PR against `ml-explore/mlx-swift` to ship the metallib as
  a SwiftPM resource. We will note this as a follow-up but not block on it.
- Homebrew bottle / tarball / GitHub release-artifact packaging.
- Linux / CUDA paths. macOS arm64 only.
- Adding any Swift-level unit test for build artifacts.

## Approach

A new `Makefile` target `metallib` compiles the mlx kernel sources from the
resolved `mlx-swift` checkout into `mlx.metallib` and stages it next to the
SwiftPM-produced executable.

### Why a Makefile target, not a SwiftPM build-tool plugin

- A `BuildToolPlugin` would have to reach into another package's checkout
  (`.build/checkouts/mlx-swift/...`), which SwiftPM does not officially
  support and which is fragile across SwiftPM versions.
- The repo already uses a `Makefile` as the front door (`make build`,
  `make ci`); adding one more target there matches the existing pattern.
- Plain `swift build` will continue to work, but won't stage the metallib;
  the README directs contributors to `make build` as the supported path.

### Why compile, not copy from a Python wheel

The issue's documented workaround (copying from an installed Python `mlx`
wheel) requires a working Python + mlx install — exactly the precondition we
want to remove. Compiling from the in-tree `.metal` sources is hermetic and
reproducible from a clean checkout.

## Design

### File layout

| Path | Purpose |
|------|---------|
| `Makefile` | Adds the `metallib` target and wires it into `build` / `ci`. |
| `scripts/build-metallib.sh` | Encapsulates the multi-step metal compile. Easier to read and to invoke from CI than embedding it inline in the Makefile. |
| `README.md` | Adds a short "Building" section that explains `make build` is the supported path because of the metallib staging step. |

The `scripts/` directory does not exist yet; this spec introduces it. We
prefer a script file over a long recipe because the recipe contains a
per-source loop and shell quoting that is awkward in `make`.

### Makefile changes

Targets added (`metallib`, `dev-metallib`, `verify-metallib`); existing
`build` / `ci` chain to them.

```make
CONFIG ?= release

build:
	swift build -c $(CONFIG)
	$(MAKE) metallib CONFIG=$(CONFIG)

dev-build:
	$(MAKE) build CONFIG=debug

metallib:
	scripts/build-metallib.sh $(CONFIG)

verify-metallib:
	@test -s .build/$(CONFIG)/mlx.metallib \
	  || (echo "missing or empty .build/$(CONFIG)/mlx.metallib" && exit 1)

ci: lint test build verify-metallib
```

Notes:

- `CONFIG ?= release` matches `swift build`'s default-when-asked-explicitly
  convention. The plain `swift build` (no `-c`) defaults to `debug`, so we
  expose `make dev-build` to call the debug path explicitly. Contributors who
  run plain `swift build` get a clear README pointer.
- `verify-metallib` is the only new "test" — a file-existence + non-empty
  check. It is sufficient because the failure mode in #43 is "the file is
  absent". We do not load a model in CI.

### `scripts/build-metallib.sh`

Inputs:

- `$1` — build config (`release` or `debug`). Required.

Behavior:

1. Resolve `CHECKOUT := .build/checkouts/mlx-swift`. If absent, exit with the
   message `mlx-swift checkout not found — run "swift package resolve" first`.
2. Resolve `METAL_SRC := $CHECKOUT/Source/Cmlx/mlx-generated/metal`. This
   directory is populated by mlx-swift's own `fix-metal-includes.sh` as part
   of the package layout (includes are pre-rewritten to be relative to this
   root).
3. Resolve `OUT_DIR := .build/<config>` (e.g. `.build/release`). This is a
   SwiftPM-managed symlink to the per-arch directory; writing into it stages
   the file alongside the produced binary.
4. Resolve `WORK := .build/metallib-build/<config>`. Create if absent.
5. The kernel source list is the 9 files enumerated by mlx-swift's
   `tools/fix-metal-includes.sh` (paths are relative to `METAL_SRC`):
   - `arg_reduce.metal`
   - `conv.metal`
   - `gemv.metal`
   - `layer_norm.metal`
   - `random.metal`
   - `rms_norm.metal`
   - `rope.metal`
   - `scaled_dot_product_attention.metal`
   - `steel/attn/kernels/steel_attention.metal`
6. **Up-to-date check** — if `$OUT_DIR/mlx.metallib` exists and is newer
   than every `.metal` and `.h` under `$METAL_SRC`, exit 0 with the message
   `mlx.metallib up to date`. (Implemented with a single `find -newer`
   comparison, not per-file `make` rules — the script is simpler that way.)
7. For each kernel source, run:
   ```
   xcrun -sdk macosx metal \
     -c \
     -ffast-math \
     -gline-tables-only \
     -frecord-sources \
     -I "$METAL_SRC" \
     "$METAL_SRC/<src>" \
     -o "$WORK/<basename-with-flat-name>.air"
   ```
   The basename is flattened (`/` → `_`) so `steel/attn/kernels/...` does
   not collide with the work dir layout.
8. Link all `.air` outputs into a single metallib:
   ```
   xcrun -sdk macosx metallib "$WORK"/*.air -o "$OUT_DIR/mlx.metallib"
   ```
9. Print `wrote $OUT_DIR/mlx.metallib (<size>)` on success.

The script uses `set -euo pipefail`. On any failure (missing checkout,
missing source, compile error, link error) it exits non-zero with a clear
message — no silent fallbacks.

### Include flags

`xcrun -sdk macosx metal -I "$METAL_SRC"` is sufficient: the includes in the
9 source files reference paths relative to `$METAL_SRC` (e.g.
`#include "steel/conv/params.h"`, `#include "utils.h"`), and that rewrite is
performed by mlx-swift's `fix-metal-includes.sh` as part of preparing the
checkout layout we consume.

If `metal` reports a missing header on first run we expand the include set
accordingly. We expect this not to be necessary, but call it out so it isn't
a surprise during implementation.

### README changes

Add a "Building" section:

```
## Building

Use `make build` (or `make dev-build` for a debug build). This runs
`swift build` and then stages `mlx.metallib` into `.build/<config>/`
alongside the executable; the MLX runtime requires the metallib to be
located next to the binary at runtime.

Plain `swift build` will produce an executable but without the metallib
present inference will fail at the first call with
"Failed to load the default metallib".
```

## Error handling

| Condition | Behavior |
|-----------|----------|
| `.build/checkouts/mlx-swift` missing | Exit with "run `swift package resolve` first". |
| `xcrun -sdk macosx metal` not found | Surface `xcrun`'s own error; `set -e` aborts. |
| Any `.metal` compile fails | Print the failing command + `metal` output, exit non-zero. |
| `metallib` link fails | Exit non-zero with the tool's error. |
| `.build/<config>` symlink missing (no SwiftPM build yet) | Exit with "run `swift build -c <config>` first". |

## Testing strategy

- **Manual smoke (documented in the PR description):**
  1. `make clean && swift package resolve && make build`.
  2. `test -s .build/release/mlx.metallib`.
  3. `.build/release/olmlx --help` runs without crashing.
  4. Optionally: `.build/release/olmlx serve` against a known model returns a
     completion. This requires a downloaded model and is not part of CI.
- **CI:** `make ci` now includes `verify-metallib`. CI does not load a model.
- **No Swift unit test.** This is a build-artifact concern; a unit test would
  not add signal beyond `verify-metallib`.

## Risks and follow-ups

1. **Set of kernel sources may drift.** When we bump `mlx-swift`, the 9-file
   list in `scripts/build-metallib.sh` may go stale. Mitigation: cross-check
   against `.build/checkouts/mlx-swift/tools/fix-metal-includes.sh` on each
   bump. A stronger mitigation — discover the list dynamically by globbing
   `*.metal` under `$METAL_SRC` — is tempting but the upstream script is the
   authoritative manifest; globbing risks accidentally including files that
   were excluded from the official build. We pick the safer, explicit list.
2. **Compile time.** Template instantiation produces a ~131 MB metallib; the
   first build is slow. Subsequent builds skip via the up-to-date check.
3. **No upstream fix yet.** A follow-up issue against `ml-explore/mlx-swift`
   should propose declaring the metallib as a SwiftPM resource on `Cmlx`.
   That work is out of scope for this PR.

## Acceptance criteria

- `git clean -fdx && swift package resolve && make build` produces
  `.build/release/olmlx` and `.build/release/mlx.metallib` (non-empty).
- `make dev-build` produces `.build/debug/olmlx` and
  `.build/debug/mlx.metallib`.
- `make ci` passes, including `verify-metallib`.
- README has a "Building" section explaining why `make build` is the
  supported path.
- `make clean` removes the produced metallib (it lives under `.build`, which
  `make clean` already wipes).
