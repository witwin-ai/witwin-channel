# Copyright Xingyu Chen.
# Source this file to load this worktree's developer build of _channel in Python and pytest.
#
# Usage (from any directory, after scripts/dev_build.sh):
#
#   source scripts/dev_env.sh
#
# It activates the conda env read-only (CHANNEL_DEV_CONDA_ENV, default solvers),
# exports WITWIN_CHANNEL_DEVELOPER_OVERRIDE=1, WITWIN_CHANNEL_EXTENSION_PATH and
# WITWIN_CHANNEL_EXPECTED_FINGERPRINT for build/dev-<arch>/ of this worktree
# (CHANNEL_DEV_CUDA_ARCH, default 86), and sets CUDA_VISIBLE_DEVICES=1 unless it is
# already set. Run Python and pytest from the worktree root so this worktree's
# witwin/channel is the package that is imported. Re-source it after each rebuild:
# the fingerprint changes with every configure that sees a new Git HEAD or dirty state.
#
# The CI tiers run `python -m ruff` and `python -m mypy`. When the conda env lacks
# them and CHANNEL_DEV_TOOLS_BIN names a directory holding `ruff` and `mypy`
# executables, a forwarding shim under build/dev-tools is put on PYTHONPATH.

_channel_dev_env() {
    local root arch build extension
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    arch=${CHANNEL_DEV_CUDA_ARCH:-86}
    build=$root/build/dev-$arch
    extension=$(find "$build" -maxdepth 1 -name '_channel*.so' -print -quit 2>/dev/null)
    if [[ -z $extension || ! -f $build/_channel.build-fingerprint ]]; then
        echo "dev_env.sh: no built extension in $build; run scripts/dev_build.sh first" >&2
        return 1
    fi
    if [[ ${CONDA_DEFAULT_ENV:-} != "${CHANNEL_DEV_CONDA_ENV:-solvers}" ]]; then
        local had_u=0
        [[ $- == *u* ]] && had_u=1
        set +u
        # shellcheck disable=SC1091
        source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
        conda activate "${CHANNEL_DEV_CONDA_ENV:-solvers}"
        [[ $had_u == 1 ]] && set -u
    fi
    export WITWIN_CHANNEL_DEVELOPER_OVERRIDE=1
    export WITWIN_CHANNEL_EXTENSION_PATH=$extension
    WITWIN_CHANNEL_EXPECTED_FINGERPRINT=$(tr -d '[:space:]' <"$build/_channel.build-fingerprint")
    export WITWIN_CHANNEL_EXPECTED_FINGERPRINT
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
    if [[ -n ${CHANNEL_DEV_TOOLS_BIN:-} ]] && ! python -c 'import ruff, mypy' 2>/dev/null; then
        local shim=$root/build/dev-tools tool
        for tool in ruff mypy; do
            mkdir -p "$shim/$tool"
            : >"$shim/$tool/__init__.py"
            # The tool must not see this PYTHONPATH, or mypy would import the shim again.
            printf '%s\n' 'import os, sys' 'env = dict(os.environ)' 'env.pop("PYTHONPATH", None)' \
                "os.execve('$CHANNEL_DEV_TOOLS_BIN/$tool', ['$tool', *sys.argv[1:]], env)" \
                >"$shim/$tool/__main__.py"
        done
        case ":${PYTHONPATH:-}:" in
            *":$shim:"*) ;;
            *) export PYTHONPATH=$shim${PYTHONPATH:+:$PYTHONPATH} ;;
        esac
    fi
    echo "dev_env.sh: $extension (fingerprint ${WITWIN_CHANNEL_EXPECTED_FINGERPRINT:0:12}...," \
        "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
}
if ! _channel_dev_env; then
    unset -f _channel_dev_env
    return 1
fi
unset -f _channel_dev_env
