# Copyright Xingyu Chen.
# Tests build info.

import hashlib
import json
import re

import torch

from witwin.channel.deployment import build_info


def test_build_info_contract():
    info = build_info()

    assert info["backend"] == "channel"
    assert info["uses_dr_jit"] is False
    assert isinstance(info["uses_rayd_native"], bool)
    assert isinstance(info["uses_path_native"], bool)
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["optix_available"], bool)
    assert info["channel_abi_version"] == 1
    assert len(info["channel_git_sha"]) == 40
    assert isinstance(info["channel_git_dirty"], bool)
    assert info["rayd_repository_url"] == "https://github.com/Asixa/RayD.git"
    assert info["rayd_commit"] == "94cf6eaf39f3625af482bb3fd8cba1377a804ecc"
    assert isinstance(info["rayd_dirty"], bool)
    assert info["rayd_integration_abi_kind"] == "source-header-sha256"
    assert (
        info["rayd_integration_abi_path"]
        == "backends/torch/include/rayd/torch/integration.h"
    )
    assert (
        info["rayd_integration_abi_sha256"]
        == "57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14"
    )
    assert info["rayd_source_kind"] in {"git-checkout", "python-package"}
    assert (
        info["rayd_source_manifest_sha256"]
        == "c00942ec28b760407b638b5ab06ead894fade6ca506a84c163550283aa471cc4"
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
        re.fullmatch(
            r"(?:70|75|80|86|87|89|90|100|101|120)-(?:real|virtual)",
            value,
        )
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
        "channel_abi_version",
        "channel_git_dirty",
        "channel_git_sha",
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
        "rayd_source_kind",
        "rayd_source_manifest_sha256",
        "torch_version",
    }
    canonical = json.dumps(
        {key: info[key] for key in identity_keys},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

    assert hashlib.sha256(canonical).hexdigest() == info["build_fingerprint"]