# mlx.metallib Build-Time Staging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile `mlx.metallib` from the in-tree mlx-swift kernel sources and stage it next to the `olmlx` executable, so `make build && .build/<config>/olmlx serve` works on a clean checkout with no Python dependency. Fixes [GH #43](https://github.com/dpalmqvist/olmlx-swift/issues/43).

**Architecture:** A new shell script (`scripts/build-metallib.sh`) compiles the 9 `.metal` kernel sources under `.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal/` using `xcrun -sdk macosx metal` and links them with `xcrun -sdk macosx metallib`, staging the result in `.build/<config>/mlx.metallib`. The `Makefile` gains `metallib`, `dev-build`, and `verify-metallib` targets and wires them into `build` / `ci`. A new `README.md` documents `make build` as the supported entry point. No edits to `Package.swift` or `mlx-swift`.

**Tech Stack:** macOS, Bash, GNU Make, `xcrun` (`metal`, `metallib`), SwiftPM (consumer).

**Spec:** [`docs/superpowers/specs/2026-05-18-mlx-metallib-staging-design.md`](../specs/2026-05-18-mlx-metallib-staging-design.md)

---

## Preconditions

- Working directory is `/Users/daniel/devel/olmlx-swift` (the repo root).
- The base of this branch is `main` at commit `b2b087f` or newer (the spec commit).
- `xcrun -sdk macosx metal --version` and `xcrun -sdk macosx metallib --version` both succeed (Xcode CLT installed).
- `swift package resolve` has been run at least once so `.build/checkouts/mlx-swift/` exists. If not, run it before Task 1 (it's a one-time setup, not part of any task).

## File Structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `scripts/build-metallib.sh` | create | Single source of truth for the metallib compile. Reads config arg, checks preconditions, compiles `.metal` → `.air`, links `.air` → `mlx.metallib`. |
| `Makefile` | modify | Adds `metallib`, `dev-build`, `verify-metallib` targets; wires them into `build` and `ci`. |
| `README.md` | create | Top-level project README with a "Building" section explaining the metallib staging step. |
| `.github/workflows/ci.yml` | modify (optional task at end) | Adds a job that runs `make build && make verify-metallib` to gate CI on the fix. |

---

### Task 1: Add `scripts/build-metallib.sh` (preconditions only)

**Goal:** Produce a runnable script that validates its inputs and exits cleanly with a clear error when preconditions aren't met. No compilation yet — this is the scaffolding step so we can exercise the failure paths first.

**Files:**
- Create: `scripts/build-metallib.sh`

- [ ] **Step 1: Create the script with arg + precondition checks**

Create `scripts/build-metallib.sh` with the following exact content:

```bash
#!/usr/bin/env bash
# Build mlx.metallib from the resolved mlx-swift checkout's .metal sources
# and stage it next to the SwiftPM-produced binary in .build/<config>/.
#
# Usage: scripts/build-metallib.sh <config>
#   <config> -- "release" or "debug"
#
# Exits non-zero with a clear message on any failure.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <release|debug>" >&2
    exit 2
fi

CONFIG="$1"
if [[ "$CONFIG" != "release" && "$CONFIG" != "debug" ]]; then
    echo "error: config must be 'release' or 'debug', got '$CONFIG'" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKOUT="$REPO_ROOT/.build/checkouts/mlx-swift"
METAL_SRC="$CHECKOUT/Source/Cmlx/mlx-generated/metal"
OUT_DIR="$REPO_ROOT/.build/$CONFIG"
WORK="$REPO_ROOT/.build/metallib-build/$CONFIG"
OUT="$OUT_DIR/mlx.metallib"

if [[ ! -d "$CHECKOUT" ]]; then
    echo "error: mlx-swift checkout not found at $CHECKOUT" >&2
    echo "       run 'swift package resolve' first" >&2
    exit 1
fi

if [[ ! -d "$METAL_SRC" ]]; then
    echo "error: metal source dir not found at $METAL_SRC" >&2
    echo "       the mlx-swift checkout layout may have changed" >&2
    exit 1
fi

if [[ ! -d "$OUT_DIR" ]]; then
    echo "error: $OUT_DIR not found" >&2
    echo "       run 'swift build -c $CONFIG' first" >&2
    exit 1
fi

mkdir -p "$WORK"

# Kernel sources (paths relative to $METAL_SRC). Authoritative manifest:
#   .build/checkouts/mlx-swift/tools/fix-metal-includes.sh
KERNELS=(
    arg_reduce.metal
    conv.metal
    gemv.metal
    layer_norm.metal
    random.metal
    rms_norm.metal
    rope.metal
    scaled_dot_product_attention.metal
    steel/attn/kernels/steel_attention.metal
)

echo "build-metallib: preconditions OK (config=$CONFIG)"
echo "  CHECKOUT=$CHECKOUT"
echo "  METAL_SRC=$METAL_SRC"
echo "  OUT=$OUT"
echo "  WORK=$WORK"
echo "  kernels=${#KERNELS[@]}"
```

- [ ] **Step 2: Make it executable**

Run:

```bash
chmod +x scripts/build-metallib.sh
```

- [ ] **Step 3: Verify it rejects bad args**

Run:

```bash
scripts/build-metallib.sh
```

Expected: exit code 2, stderr message `usage: scripts/build-metallib.sh <release|debug>`.

Run:

```bash
scripts/build-metallib.sh foo
```

Expected: exit code 2, stderr message `error: config must be 'release' or 'debug', got 'foo'`.

- [ ] **Step 4: Verify it detects missing OUT_DIR**

If `.build/release/` does not exist yet (because no release build has been done), run:

```bash
scripts/build-metallib.sh release
```

Expected: exit code 1, stderr includes `run 'swift build -c release' first`.

If `.build/release/` does exist, skip this check — we'll re-exercise it in Task 5 after a clean build.

- [ ] **Step 5: Verify it accepts good args (with existing build artifacts)**

If `.build/release/` exists from a prior build, run:

```bash
scripts/build-metallib.sh release
```

Expected: exit code 0, stdout includes `build-metallib: preconditions OK (config=release)` and `kernels=9`.

- [ ] **Step 6: Commit**

```bash
git add scripts/build-metallib.sh
git commit -m "$(cat <<'EOF'
Add: scripts/build-metallib.sh skeleton with precondition checks (#43)

First step toward fixing mlx.metallib staging: a guarded script that
validates its arg and the mlx-swift checkout layout. Compile and link
steps will follow in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Implement compile + link in `build-metallib.sh`

**Goal:** Compile each `.metal` source to a `.air` and link them into `mlx.metallib`. Add an up-to-date check so re-runs are fast.

**Files:**
- Modify: `scripts/build-metallib.sh` (append to the bottom — keep the precondition block from Task 1 intact)

- [ ] **Step 1: Replace the trailing diagnostic block with the real build**

Open `scripts/build-metallib.sh`. Remove the trailing diagnostic block (the four `echo "  ..."` lines and the `echo "build-metallib: preconditions OK ..."` line) and append the following exact content at the end of the file:

```bash
# Up-to-date check: skip the whole build if OUT is newer than every input.
if [[ -f "$OUT" ]]; then
    newest_input="$(find "$METAL_SRC" \( -name '*.metal' -o -name '*.h' \) -newer "$OUT" -print -quit)"
    if [[ -z "$newest_input" ]]; then
        echo "build-metallib: $OUT up to date"
        exit 0
    fi
fi

echo "build-metallib: compiling ${#KERNELS[@]} kernels (config=$CONFIG)"

air_files=()
for src in "${KERNELS[@]}"; do
    src_path="$METAL_SRC/$src"
    if [[ ! -f "$src_path" ]]; then
        echo "error: missing kernel source: $src_path" >&2
        exit 1
    fi
    # Flatten path separators so steel/attn/kernels/steel_attention.metal
    # becomes a single .air file in $WORK without nested dirs.
    flat="${src//\//_}"
    air="$WORK/${flat%.metal}.air"
    echo "  metal -c $src"
    xcrun -sdk macosx metal \
        -c \
        -ffast-math \
        -gline-tables-only \
        -frecord-sources \
        -I "$METAL_SRC" \
        "$src_path" \
        -o "$air"
    air_files+=("$air")
done

echo "build-metallib: linking $OUT"
xcrun -sdk macosx metallib "${air_files[@]}" -o "$OUT"

size="$(stat -f '%z' "$OUT")"
echo "build-metallib: wrote $OUT ($size bytes)"
```

- [ ] **Step 2: Pre-condition the build dir**

Before running the full compile, make sure `.build/release/` exists by running:

```bash
swift build -c release
```

Expected: SwiftPM build succeeds (this is unchanged from current behavior; the metallib won't be staged yet).

- [ ] **Step 3: Run the compile end-to-end**

Run:

```bash
scripts/build-metallib.sh release
```

Expected:
- Stdout shows `compiling 9 kernels`, then `metal -c <each>` for all 9 sources.
- Stdout ends with `wrote <repo>/.build/release/mlx.metallib (<size> bytes)`.
- Exit code 0.
- File `.build/release/mlx.metallib` is non-zero (expected ~130 MB after template instantiation).

If any kernel fails to compile with a missing-header error, the spec calls this out as the only realistic discovery-time fixup — add an additional `-I` flag pointing at the specific subdirectory the error names, and re-run. Do *not* silently rename or skip the failing source.

- [ ] **Step 4: Verify the up-to-date check**

Re-run immediately:

```bash
scripts/build-metallib.sh release
```

Expected: stdout contains `up to date`, exit code 0, finishes in well under a second.

- [ ] **Step 5: Verify it rebuilds when a source changes**

Touch a kernel header and re-run:

```bash
touch .build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal/utils.h
scripts/build-metallib.sh release
```

Expected: stdout shows `compiling 9 kernels` again (the up-to-date check correctly detected the changed input).

- [ ] **Step 6: Commit**

```bash
git add scripts/build-metallib.sh
git commit -m "$(cat <<'EOF'
Add: compile + link logic to scripts/build-metallib.sh (#43)

Compiles the 9 in-tree mlx-swift kernel sources to .air and links them
into a single mlx.metallib, staged at .build/<config>/mlx.metallib. An
up-to-date check skips the work when no source has changed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Wire `metallib`, `dev-build`, `verify-metallib` into the Makefile

**Goal:** `make build` runs `swift build` then stages the metallib. `make dev-build` does the same for debug. `make verify-metallib` is a CI-friendly file-existence assertion.

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Replace the Makefile with the new version**

Open `Makefile`. The current contents (25 lines) are:

```make
.PHONY: build test lint format format-check ci clean

SOURCES := Sources Tests Package.swift

build:
	swift build

test:
	swift test

lint:
	swift format lint --strict --recursive $(SOURCES)
	swiftlint lint --strict

format:
	swift format format --in-place --recursive $(SOURCES)

format-check:
	swift format lint --strict --recursive $(SOURCES)

ci: lint test

clean:
	swift package clean
	rm -rf .build
```

Replace the entire file with:

```make
.PHONY: build dev-build metallib verify-metallib test lint format format-check ci clean

SOURCES := Sources Tests Package.swift
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
	@echo "verify-metallib: .build/$(CONFIG)/mlx.metallib present"

test:
	swift test

lint:
	swift format lint --strict --recursive $(SOURCES)
	swiftlint lint --strict

format:
	swift format format --in-place --recursive $(SOURCES)

format-check:
	swift format lint --strict --recursive $(SOURCES)

ci: lint test build verify-metallib

clean:
	swift package clean
	rm -rf .build
```

Notes for the engineer:
- `CONFIG ?= release` makes `make build` default to `release` (matches the issue repro, which uses `swift build -c release`). Plain `swift build` (without `-c`) still works directly but won't stage the metallib — the README documents this.
- `$(MAKE) metallib CONFIG=$(CONFIG)` is used (not a Make dependency) because `metallib` must run **after** `swift build -c $(CONFIG)`, and the recipe-line ordering plus the recursive invoke is the simplest way to guarantee that.
- Indentation in Makefiles must be **tab characters**, not spaces.

- [ ] **Step 2: Run `make verify-metallib` (file should already exist from Task 2)**

Run:

```bash
make verify-metallib
```

Expected: exit code 0, stdout `verify-metallib: .build/release/mlx.metallib present`.

- [ ] **Step 3: Run `make build` (clean to force a fresh compile)**

Run:

```bash
rm -rf .build/metallib-build .build/release/mlx.metallib
make build
```

Expected: SwiftPM build runs, then `build-metallib: compiling 9 kernels`, then `wrote .build/release/mlx.metallib (...)`. Exit code 0.

- [ ] **Step 4: Run `make dev-build`**

Run:

```bash
make dev-build
```

Expected: SwiftPM debug build runs, then `compiling 9 kernels` again (different config => different `$WORK` and different output path), then `wrote .build/debug/mlx.metallib (...)`. Exit code 0. Confirm:

```bash
test -s .build/debug/mlx.metallib && echo OK
```

Should print `OK`.

- [ ] **Step 5: Run `make ci`**

Run:

```bash
make ci
```

Expected: lint passes, tests pass, build runs (or no-ops if up-to-date), `verify-metallib` passes. Exit code 0.

If lint or tests fail due to unrelated in-progress work in the tree, stash those changes before running `make ci`, or run only `make build && make verify-metallib` to validate this task's scope.

- [ ] **Step 6: Commit**

```bash
git add Makefile
git commit -m "$(cat <<'EOF'
Add: metallib + dev-build + verify-metallib Make targets (#43)

make build now runs swift build and then stages mlx.metallib next to the
produced binary. dev-build is the debug-config shortcut. verify-metallib
is a file-existence gate used by ci.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Create top-level `README.md` with a Building section

**Goal:** Document that `make build` is the supported entry point and why. There is no README yet, so we add a minimal one — but only what's needed for the build story; we are not writing a feature tour.

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

Create `README.md` with exactly this content:

```markdown
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
```

- [ ] **Step 2: Smoke-render the markdown (optional)**

Open `README.md` in a viewer that renders GitHub-flavored markdown, or run a quick syntax check:

```bash
test -s README.md && head -1 README.md
```

Expected: prints `# olmlx`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
Docs: add README with Building section (#43)

Documents make build / make dev-build as the supported entry points and
explains why plain swift build is not enough (mlx.metallib staging).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: End-to-end clean-build smoke test

**Goal:** Prove the fix works from a fully clean state, matching the issue's repro.

**Files:** none modified — this is a verification task with no code changes.

- [ ] **Step 1: Wipe the build tree**

Run:

```bash
make clean
```

Expected: `swift package clean` runs, `.build` is removed. Confirm:

```bash
test ! -e .build && echo "clean OK"
```

Should print `clean OK`.

- [ ] **Step 2: Resolve packages**

Run:

```bash
swift package resolve
```

Expected: exit code 0, `.build/checkouts/mlx-swift/` re-created.

- [ ] **Step 3: Run `make build`**

Run:

```bash
make build
```

Expected: SwiftPM compiles the whole project, then `scripts/build-metallib.sh release` runs and stages the metallib. Exit code 0.

- [ ] **Step 4: Verify the binary + metallib are co-located**

Run:

```bash
ls -la .build/release/olmlx .build/release/mlx.metallib
```

Expected: both files exist, `mlx.metallib` is non-empty (likely ~130 MB).

- [ ] **Step 5: Run the binary to confirm it starts**

Run:

```bash
.build/release/olmlx --help
```

Expected: usage text, no `Failed to load the default metallib` error.

If a model is available locally, optionally exercise `serve`:

```bash
.build/release/olmlx serve &
SERVE_PID=$!
sleep 2
curl -sS http://127.0.0.1:11434/api/tags
kill $SERVE_PID
```

The exact response depends on registered models — the success signal is that the server starts and responds without the metallib error.

- [ ] **Step 6: Nothing to commit; record in PR description**

This task produces no commits. In the PR description, paste the output of:

```bash
ls -la .build/release/olmlx .build/release/mlx.metallib
.build/release/olmlx --help | head -5
```

as evidence the clean-build smoke passed.

---

### Task 6 (optional): Gate CI on `verify-metallib`

**Goal:** Catch any future regression in CI rather than waiting for a user report. Optional — skip if you'd rather keep this PR minimal and add CI gating in a follow-up.

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add a `build` job to ci.yml**

Append the following job to `.github/workflows/ci.yml` (after the existing `test:` job, with the same two-space indent under `jobs:`):

```yaml
  build:
    name: Build (release + metallib)
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4

      - name: Show toolchain
        run: swift --version

      - name: Cache SwiftPM
        uses: actions/cache@v4
        with:
          path: |
            .build
            ~/Library/Caches/org.swift.swiftpm
          key: ${{ runner.os }}-spm-build-${{ hashFiles('**/Package.resolved') }}
          restore-keys: |
            ${{ runner.os }}-spm-build-
            ${{ runner.os }}-spm-

      - name: make build
        run: make build

      - name: make verify-metallib
        run: make verify-metallib
```

Note: the cache key is intentionally different from the `test:` job's (`-spm-build-` vs `-spm-`) so the two jobs don't fight over the same cache entry. We accept the duplicate fetch on first run.

- [ ] **Step 2: Validate the YAML locally**

Run:

```bash
python3 -c 'import yaml,sys; yaml.safe_load(open(".github/workflows/ci.yml")); print("ok")'
```

Expected: prints `ok`. If Python isn't installed, skip this step — GitHub will surface the error on first push, and the change is small enough to eyeball.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
CI: add build job gating on make build + verify-metallib (#43)

Catches regressions in mlx.metallib staging in CI rather than waiting
for a user report.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review against the spec

- **Build the metallib from in-tree .metal sources, not Python wheel** — Task 2 (`scripts/build-metallib.sh` compile loop).
- **Hermetic, no Python dependency** — Task 1's preconditions only check the SwiftPM checkout; Task 2 uses only `xcrun`.
- **`make build` chains swift build + metallib; `make dev-build` for debug; `make verify-metallib` for CI** — Task 3.
- **Idempotent up-to-date check** — Task 2 Step 1 (the `find -newer` block) and Step 4 (verification).
- **Clear error messages on missing checkout / missing OUT_DIR / missing source** — Task 1 (precondition errors), Task 2 (missing-source error).
- **README "Building" section** — Task 4.
- **CI runs `verify-metallib`** — Task 3 wires it into `make ci`; Task 6 wires it into `.github/workflows/ci.yml` (optional).
- **Acceptance criteria from spec** — Task 5 covers all four bullets (`make clean && swift package resolve && make build` produces both files; `make dev-build` produces debug versions; `make ci` passes; README present).
- **No Swift unit test for build artifacts** — confirmed; the only "test" is `verify-metallib`.
- **Out of scope (service install, upstream PR, packaging, Linux/CUDA)** — not touched.

No placeholders. All file paths, commands, and code blocks are concrete. The 9-kernel list and the `xcrun` flags are consistent between the script (Task 1/2) and the spec.
