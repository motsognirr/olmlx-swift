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
