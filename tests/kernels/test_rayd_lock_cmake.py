# Copyright Xingyu Chen.
# Tests rayd lock cmake.

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
    *args: str, cwd: Path | None = None, env: dict[str, str] | None = None,
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
    platform_root = ROOT.parent.parent if ROOT.parent.name == ".worktrees" else ROOT.parent
    source = Path(configured) if configured else platform_root / "RayD"
    assert (source / ".git").exists(), f"locked RayD checkout is missing: {source}"
    return source


def _clone_locked_rayd(destination: Path) -> dict[str, object]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    git = shutil.which("git")
    assert git is not None
    clone = _run(git, "clone", "--shared", "--no-checkout", str(_rayd_source()), str(destination))
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
    assert _run(git, "sparse-checkout", "init", "--no-cone", cwd=destination).returncode == 0
    abi_paths = [header["path"] for header in lock["integration_abi"]["headers"]]
    sparse = _run(
        git,
        "sparse-checkout",
        "set",
        "torch/CMakeLists.txt",
        "src/field_transport_ad.cuh",
        "src/transmission_device.cuh",
        *abi_paths,
        cwd=destination,
    )
    assert sparse.returncode == 0, sparse.stderr
    checkout = _run(git, "checkout", "--detach", str(lock["commit"]), cwd=destination)
    assert checkout.returncode == 0, checkout.stderr
    return lock


def _configure(
    rayd: Path, build: Path, *, release: bool | None = None,
    torch_cuda_arch_list: str = DEFAULT_TORCH_CUDA_ARCH_LIST,
    cmake_cuda_architectures: str | None = None, skbuild_state: str | None = None,
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
        "-DCHANNEL_VALIDATE_RAYD_ONLY=ON",
        f"-DRAYD_SOURCE_DIR={rayd}",
    ]
    if release is not None:
        arguments.append(f"-DCHANNEL_RELEASE_BUILD={'ON' if release else 'OFF'}")
    if cmake_cuda_architectures is not None:
        arguments.append(f"-DCMAKE_CUDA_ARCHITECTURES={cmake_cuda_architectures}")
    if skbuild_state is not None:
        arguments.append(f"-DSKBUILD_STATE={skbuild_state}")
    return _run(*arguments, env=env)


def test_rayd_lock_is_machine_readable_and_complete():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    assert lock["schema_version"] == 2
    assert lock["repository_url"] == "https://github.com/Asixa/RayD"
    assert lock["commit"] == "c7a99979d0fdcc67b2ec8a12246a7df597603409"
    assert lock["integration_abi"] == {
        "kind": "source-header-set-sha256",
        "entrypoint": "include/rayd/integration.h",
        "headers": [
            {
                "path": "include/rayd/diffraction.h",
                "sha256": "40a3800a1f8019108c0f5182d11c90698419da1df18d5b948dba83f2fc2867a9",
            },
            {
                "path": "include/rayd/integration.h",
                "sha256": "57a48596403ecb81a6064f908bd945637a00d353f292acb904c851af389e3a67",
            },
            {
                "path": "include/rayd/penetration.h",
                "sha256": "363554e2c9d33b7a039ea252b5bfdbfcc5a70eceb4a1e5e9d0a23ef0aba13a57",
            },
            {
                "path": "include/rayd/reflection.h",
                "sha256": "e7decb547d45cbf03fe810785aabc059f6ed43eda1a48165aa18297bf0746d73",
            },
            {
                "path": "include/rayd/scattering.h",
                "sha256": "67bfa850df5f4ef95f24f4456124c9b5d4bdc9d97b4292f419b078ee8e63e20b",
            },
            {
                "path": "include/rayd/scene.h",
                "sha256": "0ad8a15be68f98a9ae9646d85cd8e77b4b255b158ddfc10820503dd900d36480",
            },
            {
                "path": "include/rayd/transmission.h",
                "sha256": "2ee0a6b5170f371a74e2ed69cde429b43ec8cee5f31f5f0366fa8337168372e1",
            },
            {
                "path": "include/rayd/visibility.h",
                "sha256": "95796526b7a94c580b28cca652d1859b63f9792c9eaf0115bd1d44ca75e85125",
            },
        ],
        "sha256": "db48cdb91b31c00a14259f912f8b504eb2485a031b036c6f79688cb5452670c4",
        "api_version": 8,
        "identity": "rayd.torch.integration",
    }
    assert lock["source_bundle"] == {
        "distribution": "rayd-torch",
        "distribution_version": "0.8.0",
        "metadata_path": "rayd/torch/_source/rayd-source.json",
        "manifest_sha256": "c4fb39b27eeea2588615d1493c6fe3f4fc0202017341fa320dafdcacb595b1c1",
    }


def test_invalid_explicit_rayd_source_never_falls_back_to_package(tmp_path: Path):
    configured = _configure(tmp_path / "missing", tmp_path / "build")

    output = configured.stdout + configured.stderr
    assert configured.returncode != 0, output
    assert "package discovery is not a fallback" in output


def test_cmake_accepts_the_locked_rayd_checkout(tmp_path: Path):
    rayd = tmp_path / "rayd"
    lock = _clone_locked_rayd(rayd)

    configured = _configure(rayd, tmp_path / "build")

    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert f"Validated locked RayD git-checkout source {lock['commit']}" in configured.stdout
    assert (
        "Channel CUDA architectures: 75-real;80-real;86-real;89-real;120-real;120-virtual"
    ) in configured.stdout


def test_cmake_normalizes_explicit_torch_cuda_arch_override(tmp_path: Path):
    rayd = tmp_path / "rayd"
    _clone_locked_rayd(rayd)

    configured = _configure(rayd, tmp_path / "build", torch_cuda_arch_list="8.6 12.0+PTX")

    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert "Channel CUDA architectures: 86-real;120-real;120-virtual" in configured.stdout
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
    assert "Channel CUDA architectures: 120-real" in configured.stdout


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
        ("abi", "RayD integration header changed"),
        ("dirty-release", "forbids a dirty RayD checkout"),
    ],
)
def test_cmake_rejects_unlocked_rayd(tmp_path: Path, mutation: str, expected_error: str):
    rayd = tmp_path / "rayd"
    lock = _clone_locked_rayd(rayd)
    git = shutil.which("git")
    assert git is not None
    release = mutation == "dirty-release"
    if mutation == "commit":
        assert _run(git, "config", "user.name", "Channel Test", cwd=rayd).returncode == 0
        assert _run(git, "config", "user.email", "test@example.invalid", cwd=rayd).returncode == 0
        assert (
            _run(git, "commit", "--allow-empty", "-m", "wrong revision", cwd=rayd).returncode == 0
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
        abi_path = rayd / str(lock["integration_abi"]["entrypoint"])
        abi_path.write_text(abi_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        cmake_file = rayd / "torch" / "CMakeLists.txt"
        cmake_file.write_text(cmake_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    configured = _configure(rayd, tmp_path / "build", release=release)

    output = configured.stdout + configured.stderr
    assert configured.returncode != 0, output
    assert expected_error in output


def test_wheel_configuration_enables_clean_release_guard_by_default(tmp_path: Path):
    rayd = tmp_path / "rayd"
    _clone_locked_rayd(rayd)
    cmake_file = rayd / "torch" / "CMakeLists.txt"
    cmake_file.write_text(cmake_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    configured = _configure(
        rayd,
        tmp_path / "build",
        skbuild_state="wheel",
    )

    output = configured.stdout + configured.stderr
    assert configured.returncode != 0, output
    assert "CHANNEL_RELEASE_BUILD forbids a dirty RayD checkout" in output
