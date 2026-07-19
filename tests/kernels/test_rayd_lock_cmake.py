import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "dependencies" / "rayd.lock.json"
DEFAULT_TORCH_CUDA_ARCH_LIST = "7.5 8.0 8.6 8.9 12.0+PTX"


def _run(
    *args: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def _rayd_source() -> Path:
    configured = os.environ.get("RAYD_SOURCE_DIR")
    source = Path(configured) if configured else ROOT.parent.parent / "RayDi"
    assert (source / ".git").exists(), f"locked RayD checkout is missing: {source}"
    return source


def _clone_locked_rayd(destination: Path) -> dict[str, object]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    git = shutil.which("git")
    assert git is not None
    clone = _run(
        git, "clone", "--shared", "--no-checkout", str(_rayd_source()), str(destination)
    )
    assert clone.returncode == 0, clone.stderr
    assert (
        _run(
            git,
            "remote",
            "set-url",
            "origin",
            str(lock["repository_url"]),
            cwd=destination,
        ).returncode
        == 0
    )
    assert (
        _run(git, "sparse-checkout", "init", "--no-cone", cwd=destination).returncode
        == 0
    )
    abi_path = str(lock["integration_abi"]["path"])
    sparse = _run(
        git,
        "sparse-checkout",
        "set",
        "backends/torch/CMakeLists.txt",
        abi_path,
        cwd=destination,
    )
    assert sparse.returncode == 0, sparse.stderr
    checkout = _run(git, "checkout", "--detach", str(lock["commit"]), cwd=destination)
    assert checkout.returncode == 0, checkout.stderr
    return lock


def _configure(
    rayd: Path,
    build: Path,
    *,
    release: bool | None = None,
    torch_cuda_arch_list: str = DEFAULT_TORCH_CUDA_ARCH_LIST,
    cmake_cuda_architectures: str | None = None,
    skbuild_state: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = torch_cuda_arch_list
    arguments = [
        sys.executable,
        "-m",
        "cmake",
        "-S",
        str(ROOT),
        "-B",
        str(build),
        "-DCHANNEL_NATIVE_VALIDATE_RAYD_ONLY=ON",
        f"-DRAYD_SOURCE_DIR={rayd}",
    ]
    if release is not None:
        arguments.append(f"-DCHANNEL_NATIVE_RELEASE_BUILD={'ON' if release else 'OFF'}")
    if cmake_cuda_architectures is not None:
        arguments.append(f"-DCMAKE_CUDA_ARCHITECTURES={cmake_cuda_architectures}")
    if skbuild_state is not None:
        arguments.append(f"-DSKBUILD_STATE={skbuild_state}")
    return _run(*arguments, env=env)


def test_rayd_lock_is_machine_readable_and_complete():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    assert lock == {
        "schema_version": 1,
        "repository_url": "https://github.com/Asixa/RayD.git",
        "commit": "3988f0934fec7b521ee5190b0defc0883c84b9e6",
        "integration_abi": {
            "kind": "source-header-sha256",
            "path": "backends/torch/include/rayd/torch/integration_v2.h",
            "sha256": "6cb18f682e08cb0bb0853507e3b4b82a68e681bb1dad89dc8c36518705f74989",
        },
    }


def test_cmake_accepts_the_locked_rayd_checkout(tmp_path: Path):
    rayd = tmp_path / "rayd"
    lock = _clone_locked_rayd(rayd)

    configured = _configure(rayd, tmp_path / "build")

    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert f"Validated locked RayD checkout {lock['commit']}" in configured.stdout
    assert (
        "Channel Native CUDA architectures: "
        "75-real;80-real;86-real;89-real;120-real;120-virtual"
    ) in configured.stdout


def test_cmake_normalizes_explicit_torch_cuda_arch_override(tmp_path: Path):
    rayd = tmp_path / "rayd"
    _clone_locked_rayd(rayd)

    configured = _configure(
        rayd, tmp_path / "build", torch_cuda_arch_list="8.6 12.0+PTX"
    )

    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert (
        "Channel Native CUDA architectures: 86-real;120-real;120-virtual"
        in configured.stdout
    )
    cache = (tmp_path / "build" / "CMakeCache.txt").read_text(encoding="utf-8")
    assert "CMAKE_CUDA_ARCHITECTURES:STRING=86-real;120-real;120-virtual" in cache


def test_cmake_accepts_matching_explicit_architecture_inputs(tmp_path: Path):
    rayd = tmp_path / "rayd"
    _clone_locked_rayd(rayd)

    configured = _configure(
        rayd,
        tmp_path / "build",
        torch_cuda_arch_list="12.0",
        cmake_cuda_architectures="120-real",
    )

    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert "Channel Native CUDA architectures: 120-real" in configured.stdout


def test_cmake_rejects_conflicting_architecture_inputs(tmp_path: Path):
    rayd = tmp_path / "rayd"
    _clone_locked_rayd(rayd)

    configured = _configure(
        rayd,
        tmp_path / "build",
        torch_cuda_arch_list="12.0+PTX",
        cmake_cuda_architectures="120-real",
    )

    output = configured.stdout + configured.stderr
    assert configured.returncode != 0, output
    assert "CMAKE_CUDA_ARCHITECTURES and TORCH_CUDA_ARCH_LIST disagree" in output


def test_cmake_rejects_disabled_torch_cuda_architectures(tmp_path: Path):
    rayd = tmp_path / "rayd"
    _clone_locked_rayd(rayd)

    configured = _configure(rayd, tmp_path / "build", torch_cuda_arch_list="OFF")

    output = configured.stdout + configured.stderr
    assert configured.returncode != 0, output
    assert "TORCH_CUDA_ARCH_LIST must not be OFF" in output


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("commit", "RayD revision mismatch"),
        ("remote", "RayD repository mismatch"),
        ("abi", "RayD integration ABI mismatch"),
        ("dirty-release", "forbids a dirty RayD checkout"),
    ],
)
def test_cmake_rejects_unlocked_rayd(
    tmp_path: Path, mutation: str, expected_error: str
):
    rayd = tmp_path / "rayd"
    lock = _clone_locked_rayd(rayd)
    git = shutil.which("git")
    assert git is not None
    release = mutation == "dirty-release"
    if mutation == "commit":
        assert (
            _run(git, "config", "user.name", "Channel Native Test", cwd=rayd).returncode
            == 0
        )
        assert (
            _run(
                git, "config", "user.email", "test@example.invalid", cwd=rayd
            ).returncode
            == 0
        )
        assert (
            _run(
                git, "commit", "--allow-empty", "-m", "wrong revision", cwd=rayd
            ).returncode
            == 0
        )
    elif mutation == "remote":
        assert (
            _run(
                git,
                "remote",
                "set-url",
                "origin",
                "https://example.invalid/RayD.git",
                cwd=rayd,
            ).returncode
            == 0
        )
    elif mutation == "abi":
        abi_path = rayd / str(lock["integration_abi"]["path"])
        abi_path.write_text(
            abi_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
    else:
        cmake_file = rayd / "backends" / "torch" / "CMakeLists.txt"
        cmake_file.write_text(
            cmake_file.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )

    configured = _configure(rayd, tmp_path / "build", release=release)

    output = configured.stdout + configured.stderr
    assert configured.returncode != 0, output
    assert expected_error in output


def test_wheel_configuration_enables_clean_release_guard_by_default(tmp_path: Path):
    rayd = tmp_path / "rayd"
    _clone_locked_rayd(rayd)
    cmake_file = rayd / "backends" / "torch" / "CMakeLists.txt"
    cmake_file.write_text(
        cmake_file.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    configured = _configure(
        rayd,
        tmp_path / "build",
        skbuild_state="wheel",
    )

    output = configured.stdout + configured.stderr
    assert configured.returncode != 0, output
    assert "CHANNEL_NATIVE_RELEASE_BUILD forbids a dirty RayD checkout" in output
