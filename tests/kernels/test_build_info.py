import hashlib
import json
import re

import torch

from witwin.channel_native.core.kernels.extension import build_info


def test_build_info_contract():
    info = build_info()

    assert info["backend"] == "channel-native"
    assert info["uses_dr_jit"] is False
    assert isinstance(info["uses_rayd_native"], bool)
    assert isinstance(info["uses_path_native"], bool)
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["optix_available"], bool)
    assert info["channel_native_abi_version"] == 1
    assert len(info["channel_native_git_sha"]) == 40
    assert isinstance(info["channel_native_git_dirty"], bool)
    assert info["rayd_repository_url"] == "https://github.com/Asixa/RayD.git"
    assert info["rayd_commit"] == "4cb400acbfcc2da7fda4110d1298d311816905f1"
    assert isinstance(info["rayd_dirty"], bool)
    assert info["rayd_integration_abi_kind"] == "source-header-sha256"
    assert (
        info["rayd_integration_abi_path"]
        == "backends/torch/include/rayd/torch/integration_v2.h"
    )
    assert (
        info["rayd_integration_abi_sha256"]
        == "c8e162c55a0e5abe789e4f1b19cd6ab00ee4ef59d70244cfc55d58166aeb646b"
    )
    assert isinstance(info["torch_version"], str) and info["torch_version"]
    assert isinstance(info["cuda_version"], str) and info["cuda_version"]
    assert isinstance(info["cuda_compiler_version"], str)
    assert isinstance(info["compiler"], str) and info["compiler"]
    assert info["cxx_abi"] in {"msvc", "cxx11", "pre-cxx11", "unknown"}
    architectures = info["cuda_architectures"]
    assert architectures
    assert len(architectures) == len(set(architectures))
    assert all(
        re.fullmatch(r"(?:75|80|86|89|120)-(?:real|virtual)", value)
        for value in architectures
    )
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        assert f"{major}{minor}-real" in architectures
    assert info["build_type"]
    assert len(info["build_fingerprint"]) == 64
    assert info["torch_version"] == torch.__version__.split("+", 1)[0]
    assert info["cuda_version"] == torch.version.cuda


def test_build_fingerprint_covers_compiled_identity():
    info = build_info()
    identity_keys = {
        "build_type",
        "channel_native_abi_version",
        "channel_native_git_dirty",
        "channel_native_git_sha",
        "compiler",
        "cuda_architectures",
        "cuda_compiler_version",
        "cuda_version",
        "cxx_abi",
        "rayd_commit",
        "rayd_dirty",
        "rayd_integration_abi_kind",
        "rayd_integration_abi_path",
        "rayd_integration_abi_sha256",
        "rayd_repository_url",
        "torch_version",
    }
    canonical = json.dumps(
        {key: info[key] for key in identity_keys},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    assert hashlib.sha256(canonical).hexdigest() == info["build_fingerprint"]
