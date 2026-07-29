# Copyright Xingyu Chen.
# Tests release contract performance evidence.

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from benchmarks.phase13_phase12.artifacts import ArtifactStore
from benchmarks.phase13_phase12.contracts import EvidenceError
from benchmarks.phase13_phase12.release import (
    TORCH_WHEEL_ARCHITECTURES,
    WHEEL_ARCHITECTURES,
    WHEEL_NATIVE_MEMBER,
    WHEEL_SMOKE_NAME,
    _build_fingerprint,
    _retain_wheel_native,
    _runner_channel_environment,
    _validate_extracted_pe_audit,
    _validate_wheel_smoke,
)


def _config(tmp_path: Path) -> tuple[SimpleNamespace, dict[str, object]]:
    checkout = tmp_path / "validation"
    extension = (
        checkout
        / "witwin"
        / "channel"
        / "_channel.cp311-win_amd64.pyd"
    )
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"runner extension")
    site_packages = tmp_path / "runner-install" / "site-packages"
    site_packages.mkdir(parents=True)
    rayd = tmp_path / "rayd"
    rayd.mkdir()
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    tools = SimpleNamespace(
        nvcc=tmp_path / "nvcc.exe",
        cl=tmp_path / "cl.exe",
        link=tmp_path / "link.exe",
        ninja=tmp_path / "ninja.exe",
    )
    candidate = SimpleNamespace(python_executable=python)
    config = SimpleNamespace(
        rayd_checkout=rayd,
        tools=tools,
        runner_build_environment={"PATH": str(tmp_path)},
        variant=lambda *_args: candidate,
    )
    validation = {
        "checkout": str(checkout),
        "site_packages_source": str(site_packages),
        "packaged_extension_path": str(extension),
        "build_fingerprint": "a" * 64,
    }
    return config, validation


def _multiarch_build_info() -> dict[str, object]:
    info: dict[str, object] = {
        "build_type": "Release",
        "channel_abi_version": 1,
        "channel_git_dirty": False,
        "channel_git_sha": "1" * 40,
        "compiler": "MSVC-19.44",
        "cuda_architectures": list(WHEEL_ARCHITECTURES),
        "cuda_compiler_version": "12.9",
        "cuda_version": "12.8",
        "cxx_abi": "msvc",
        "rayd_dirty": False,
        "rayd_commit": "2" * 40,
        "rayd_integration_abi_sha256": "3" * 64,
        "rayd_integration_abi_kind": "source-header-sha256",
        "rayd_integration_abi_path": (
            "backends/torch/include/rayd/torch/integration.h"
        ),
        "rayd_repository_url": "https://github.com/Asixa/RayD.git",
        "rayd_source_kind": "git-checkout",
        "rayd_source_manifest_sha256": "4" * 64,
        "torch_version": "2.10.0",
    }
    info["build_fingerprint"] = _build_fingerprint(info)
    return info


def test_release_tiers_bind_packaged_checkout_without_developer_override(
    tmp_path: Path,
) -> None:
    config, validation = _config(tmp_path)

    environment, binding = _runner_channel_environment(config, validation)  # type: ignore[arg-type]

    assert binding["mode"] == "runner-owned-packaged-validation-checkout"
    assert environment["PYTHONPATH"] == str(tmp_path / "validation")
    assert environment["TORCH_CUDA_ARCH_LIST"] == TORCH_WHEEL_ARCHITECTURES
    assert f'-DRAYD_SOURCE_DIR="{(tmp_path / "rayd").as_posix()}"' in environment[
        "CMAKE_ARGS"
    ]
    assert "75-real;80-real;86-real;89-real;120-real;120-virtual" in environment[
        "CMAKE_ARGS"
    ]
    assert not any(name.startswith("WITWIN_CHANNEL_") for name in environment)


def test_wheel_fingerprint_is_multiarch_identity_not_sm120_timing_identity() -> None:
    info = _multiarch_build_info()
    implementation = {
        "groups": {"diffraction": {"candidate_commit": "1" * 40}},
        "rayd_commit": "2" * 40,
        "integration_header_sha256": "3" * 64,
        "rayd_repository_url": "https://github.com/Asixa/RayD.git",
        "final_build_fingerprint": "f" * 64,
    }
    smoke = {
        "wheel_smoke": True,
        "wheel_sha256": "4" * 64,
        "distribution": {},
        "package_origin": "package",
        "native_origin": "native",
        "build_info": info,
        "pe_audit": {"sha256": "5" * 64},
    }

    validated = _validate_wheel_smoke(
        smoke, implementation=implementation, wheel_sha256="4" * 64
    )

    assert validated["build_info"]["build_fingerprint"] == _build_fingerprint(info)  # type: ignore[index]
    assert info["build_fingerprint"] != implementation["final_build_fingerprint"]


def _wheel_bytes(*, duplicate_casefold_member: bool = False) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w") as archive:
        archive.writestr(WHEEL_NATIVE_MEMBER, b"multiarch extension")
        if duplicate_casefold_member:
            archive.writestr(WHEEL_NATIVE_MEMBER.upper(), b"alternate")
    return stream.getvalue()


def test_wheel_native_is_retained_from_one_exact_safe_member(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "raw")
    wheel = store.write_bytes("input/release.whl", _wheel_bytes(), allow_empty=False)

    native = _retain_wheel_native(store, wheel)

    assert native["member"] == WHEEL_NATIVE_MEMBER
    artifact = native["artifact"]
    assert isinstance(artifact, dict)
    assert store.read_verified(artifact, label="native") == b"multiarch extension"


def test_wheel_native_rejects_casefold_duplicate_member(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "raw")
    wheel = store.write_bytes(
        "input/release.whl",
        _wheel_bytes(duplicate_casefold_member=True),
        allow_empty=False,
    )

    with pytest.raises(EvidenceError, match="one canonical"):
        _retain_wheel_native(store, wheel)


def test_direct_pe_audit_must_target_retained_wheel_native(tmp_path: Path) -> None:
    store = ArtifactStore.create(tmp_path / "raw")
    artifact = store.write_bytes(
        "release/wheel-native/_channel.cp311-win_amd64.pyd",
        b"multiarch extension",
        allow_empty=False,
    )
    native = {"member": WHEEL_NATIVE_MEMBER, "artifact": artifact}
    path = store.root / str(artifact["path"])
    payload = {
        "schema_version": 1,
        "path": str(path),
        "sha256": artifact["sha256"],
        "dependencies": ["python311.dll"],
        "export_count": 1,
        "exports_sha256": "6" * 64,
        "python_init_export": "PyInit__channel",
    }
    wheel_pe = {
        key: value for key, value in payload.items() if key != "path"
    } | {"wheel_member": WHEEL_NATIVE_MEMBER}

    assert (
        _validate_extracted_pe_audit(
            payload, native=native, wheel_pe_audit=wheel_pe, store=store
        )
        == wheel_pe
    )


def test_release_uses_stable_wheel_smoke_artifact_name() -> None:
    assert WHEEL_SMOKE_NAME == "wheel-smoke-pe-audit.v1.json"