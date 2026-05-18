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

# Up-to-date check: skip the whole build if OUT is newer than every input.
if [[ -f "$OUT" ]]; then
    newest_input="$(find "$METAL_SRC" \( -name '*.metal' -o -name '*.h' \) -newer "$OUT" -print -quit)"
    if [[ -z "$newest_input" ]]; then
        echo "build-metallib: $OUT up to date"
        exit 0
    fi
fi

# Metal compile flags match upstream mlx-swift's CMake config
# (Source/Cmlx/mlx/cmake/extension.cmake, mlx/backend/metal/kernels/CMakeLists.txt).
# Debug builds also embed line tables and source records for shader debugging.
METAL_FLAGS=(-Wall -Wextra -fno-fast-math -Wno-c++17-extensions)
if [[ "$CONFIG" == "debug" ]]; then
    METAL_FLAGS+=(-gline-tables-only -frecord-sources)
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
        "${METAL_FLAGS[@]}" \
        -I "$METAL_SRC" \
        "$src_path" \
        -o "$air"
    air_files+=("$air")
done

echo "build-metallib: linking $OUT"
xcrun -sdk macosx metallib "${air_files[@]}" -o "$OUT"

size="$(stat -f '%z' "$OUT")"
echo "build-metallib: wrote $OUT ($size bytes)"
