#!/usr/bin/env bash
# Copyright Xingyu Chen.
# Configure and build the _channel extension of this worktree for local development on Linux.
#
# Usage: scripts/dev_build.sh [--rayd DIR] [--clean]
#
#   --rayd DIR   RayD checkout to source-link. Its HEAD must equal the commit in
#                dependencies/rayd.lock.json (a dirty RayD tree is allowed). Default:
#                $RAYD_SOURCE_DIR, else the sibling directory ../rayd-integ.
#   --clean      delete the build directory first.
#
# Environment:
#   CHANNEL_DEV_CUDA_ARCH   one SM number, default 86; the build goes to build/dev-<arch>
#   CHANNEL_DEV_CONDA_ENV   conda env used read-only, default solvers
#   CHANNEL_DEV_JOBS        parallel compile jobs, default nproc
#
# The script never pip-installs anything and never writes outside build/dev-<arch>.
# It re-runs CMake configure every time because the build fingerprint records the
# Channel and RayD Git HEAD and dirty state at configure time; an unchanged tree
# reconfigures in seconds and Ninja then rebuilds only what changed. A lock file in
# the build directory serializes concurrent builds of the same worktree.
#
# Afterwards, `source scripts/dev_env.sh` exports the developer-override variables
# that make tests and Python in this worktree load the extension just built.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RAYD_DIR=${RAYD_SOURCE_DIR:-$(dirname "$ROOT")/rayd-integ}
CLEAN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rayd) RAYD_DIR=$2; shift 2 ;;
        --rayd=*) RAYD_DIR=${1#--rayd=}; shift ;;
        --clean) CLEAN=1; shift ;;
        -h|--help) sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "dev_build.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done
RAYD_DIR=$(cd "$RAYD_DIR" && pwd)

ARCH=${CHANNEL_DEV_CUDA_ARCH:-86}
if [[ ! $ARCH =~ ^[0-9]{2,3}$ ]]; then
    echo "dev_build.sh: CHANNEL_DEV_CUDA_ARCH must be one SM number such as 86, got '$ARCH'" >&2
    exit 2
fi
BUILD_DIR=$ROOT/build/dev-$ARCH
JOBS=${CHANNEL_DEV_JOBS:-$(nproc)}

# Activate the conda env read-only (nothing is installed into it).
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CHANNEL_DEV_CONDA_ENV:-solvers}"
set -u
E=$CONDA_PREFIX

# Conda's CUDA keeps headers and libraries under targets/x86_64-linux, which the
# legacy FindCUDA used by Torch's Caffe2 config does not find on its own; point it
# there and use the real nvcc (the targets/ nvcc symlink has no nvcc.profile).
export CUDA_NVCC_EXECUTABLE=$E/bin/nvcc
export CUDACXX=$E/bin/nvcc
# One architecture, stated once: CMAKE_CUDA_ARCHITECTURES and TORCH_CUDA_ARCH_LIST
# must agree, and RayD refuses RAYD_CUDA_ARCHS next to either of them.
export TORCH_CUDA_ARCH_LIST="${ARCH:0:${#ARCH}-1}.${ARCH: -1}"
unset CUDAARCHS RAYD_CUDA_ARCHS CMAKE_CUDA_ARCHITECTURES

mkdir -p "$BUILD_DIR"
exec 9>"$ROOT/build/.dev-$ARCH.lock"
if ! flock -n 9; then
    echo "dev_build.sh: another build of $BUILD_DIR is running; waiting for it" >&2
    flock 9
fi
if [[ $CLEAN == 1 ]]; then
    find "$BUILD_DIR" -mindepth 1 -delete
fi

start=$(date +%s)
cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH-real" \
    -DCMAKE_CUDA_COMPILER="$E/bin/nvcc" \
    -DCUDA_TOOLKIT_ROOT_DIR="$E/targets/x86_64-linux" \
    -DPython_EXECUTABLE="$E/bin/python" \
    -DRAYD_SOURCE_DIR="$RAYD_DIR" \
    -DBUILD_TESTING=OFF \
    >"$BUILD_DIR/dev_configure.log" 2>&1 || {
        tail -n 40 "$BUILD_DIR/dev_configure.log" >&2
        echo "dev_build.sh: configure failed; full log: $BUILD_DIR/dev_configure.log" >&2
        exit 1
    }
cmake --build "$BUILD_DIR" --target _channel -- -j "$JOBS"
end=$(date +%s)

extension=$(find "$BUILD_DIR" -maxdepth 1 -name '_channel*.so' -print -quit)
if [[ -z $extension || ! -f $BUILD_DIR/_channel.build-fingerprint ]]; then
    echo "dev_build.sh: build finished but $BUILD_DIR has no _channel extension and fingerprint" >&2
    exit 1
fi
echo "dev_build.sh: built $extension in $((end - start)) s"
echo "dev_build.sh: fingerprint $(cat "$BUILD_DIR/_channel.build-fingerprint")"
echo "dev_build.sh: now run: source scripts/dev_env.sh"
