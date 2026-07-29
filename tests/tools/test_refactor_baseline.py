# Copyright Xingyu Chen.
# Tests refactor baseline.

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import refactor_baseline as baseline


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_api_manifest_resolves_reexported_dataclass_without_importing(tmp_path: Path):
    _write(
        tmp_path / "witwin/channel/__init__.py",
        "from .api import Config, solve\n__all__ = ['Config', 'solve']\n",
    )
    _write(
        tmp_path / "witwin/channel/api.py",
        """
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Config:
    count: int = 3
    labels: tuple[str, ...] = field(default_factory=tuple)

def solve(scene, config: Config | None = None):
    return scene
""",
    )

    manifest = baseline.api_manifest(tmp_path, ["witwin.channel"])

    exported = manifest["modules"][0]["objects"]
    assert [item["target"] for item in exported] == [
        "witwin.channel.api.Config",
        "witwin.channel.api.solve",
    ]
    assert manifest["schemas"] == [
        {
            "export": "witwin.channel.Config",
            "target": "witwin.channel.api.Config",
            "kind": "class",
            "bases": [],
            "decorators": ["dataclass(frozen=True)"],
            "dataclass": True,
            "fields": [
                {
                    "name": "count",
                    "annotation": "int",
                    "default_kind": "value",
                    "default": "3",
                },
                {
                    "name": "labels",
                    "annotation": "tuple[str, ...]",
                    "default_kind": "factory",
                    "default": "tuple",
                },
            ],
            "enum_values": [],
            "exception_type": False,
        }
    ]


def test_api_manifest_keeps_class_definition_module_without_override(tmp_path: Path):
    _write(
        tmp_path / "witwin/channel/api.py",
        "class Config:\n    pass\n\n__all__ = ['Config']\n",
    )

    manifest = baseline.api_manifest(tmp_path, ["witwin.channel.api"])

    assert manifest["modules"][0]["objects"][0]["target"] == (
        "witwin.channel.api.Config"
    )


def test_api_manifest_uses_literal_class_module_override(tmp_path: Path):
    _write(
        tmp_path / "witwin/channel/api.py",
        """
class Config:
    pass

Config.__module__ = "legacy.compat.path"
__all__ = ["Config"]
""",
    )

    manifest = baseline.api_manifest(tmp_path, ["witwin.channel.api"])

    assert manifest["modules"][0]["objects"][0]["target"] == (
        "legacy.compat.path.Config"
    )


def test_api_manifest_ignores_nonliteral_and_unrelated_assignments(tmp_path: Path):
    _write(
        tmp_path / "witwin/channel/api.py",
        """
compatibility_module = "legacy.compat.path"

class Config:
    pass

Config.__module__ = compatibility_module
Config.compatibility_module = "legacy.compat.unrelated"
__all__ = ["Config"]
""",
    )

    manifest = baseline.api_manifest(tmp_path, ["witwin.channel.api"])

    assert manifest["modules"][0]["objects"][0]["target"] == (
        "witwin.channel.api.Config"
    )


def test_api_manifest_resolves_literal_module_override_through_reexport(
    tmp_path: Path,
):
    _write(
        tmp_path / "witwin/channel/__init__.py",
        "from .api import Config\n__all__ = ['Config']\n",
    )
    _write(
        tmp_path / "witwin/channel/api.py",
        """
class Config:
    pass

Config.__module__ = "legacy.compat.path"
""",
    )

    manifest = baseline.api_manifest(tmp_path, ["witwin.channel"])

    assert manifest["modules"][0]["objects"][0]["target"] == (
        "legacy.compat.path.Config"
    )


def test_python_body_hash_ignores_locations_comments_and_formatting(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write(
        first / "witwin/channel/module.py",
        "def compute(value):\n    return value + 1\n",
    )
    _write(
        second / "witwin/channel/module.py",
        "\n# moved during refactor\ndef compute( value ):\n\n    return value+1  # same body\n",
    )

    first_hash = baseline.python_body_hashes(first)[0]
    second_hash = baseline.python_body_hashes(second)[0]

    assert first_hash["body_sha256"] == second_hash["body_sha256"]
    assert first_hash["normalized_ast_sha256"] == second_hash["normalized_ast_sha256"]


def test_native_body_hash_uses_comment_and_whitespace_free_tokens(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write(
        first / "native/channel/op.cu",
        "int add_one(int value) { return value + 1; }\n",
    )
    _write(
        second / "native/channel/op.cu",
        "int add_one( int value ) { /* preserved math */ return value+1; }\n",
    )

    first_hash = baseline.cpp_body_hashes(first)[0]
    second_hash = baseline.cpp_body_hashes(second)[0]

    assert first_hash["name"] == "add_one"
    assert first_hash["body_sha256"] == second_hash["body_sha256"]


def test_native_body_hash_captures_multiline_return_and_signature(tmp_path: Path):
    _write(
        tmp_path / "native/channel/op.cu",
        """
std::tuple<torch::Tensor, torch::Tensor>
channel_pair_cuda(
    torch::Tensor first,
    torch::Tensor second)
{
    return {first, second};
}
""",
    )

    entries = baseline.cpp_body_hashes(tmp_path)

    assert [entry["name"] for entry in entries] == ["channel_pair_cuda"]
    assert entries[0]["token_count"] == 7


def test_adr033_identity_normalization_preserves_frozen_native_hashes(
    tmp_path: Path,
):
    former_function = "c" + "n_op"
    former_namespace = "channel" + "_native"
    former_macro = "CHANNEL" + "_NATIVE_WITH_RAYD"
    _write(
        tmp_path / "former/native/channel/op.cu",
        f"int {former_function}(int value) "
        f"{{ return {former_namespace}::adjust(value) + {former_macro}; }}\n",
    )
    _write(
        tmp_path / "current/native/channel/op.cu",
        "int channel_op(int value) "
        "{ return channel::adjust(value) + CHANNEL_WITH_RAYD; }\n",
    )

    former = baseline.cpp_body_hashes(tmp_path / "former")[0]
    current = baseline.cpp_body_hashes(
        tmp_path / "current", adr033_predecessor_identity=True
    )[0]

    assert {
        key: former[key]
        for key in ("name", "token_count", "signature_sha256", "body_sha256")
    } == {
        key: current[key]
        for key in ("name", "token_count", "signature_sha256", "body_sha256")
    }


def test_binding_manifest_records_cpp_signature_and_pybind_defaults(tmp_path: Path):
    _write(
        tmp_path / "native/channel/bindings.cpp",
        """
#include <tuple>
std::tuple<int, float> channel_pair(torch::Tensor value, int count = 2) {
    return {count, 1.0f};
}
PYBIND11_MODULE(_channel, module) {
    module.def(
        "pair",
        &channel_pair,
        pybind11::arg("value"),
        pybind11::arg("count") = 2);
}
""",
    )

    manifest = baseline.binding_manifest(tmp_path)

    assert manifest["duplicate_symbols"] == []
    assert manifest["symbols"] == [
        {
            "name": "pair",
            "target": "channel_pair",
            "path": "native/channel/bindings.cpp",
            "line": 7,
            "return_type": "std::tuple<int, float>",
            "return_arity": 2,
            "parameters": [
                {"name": "value", "type": "torch::Tensor", "default": None},
                {"name": "count", "type": "int", "default": "2"},
            ],
            "pybind_arguments": [
                {"name": "value", "default": None},
                {"name": "count", "default": "2"},
            ],
            "doc": None,
        }
    ]


def test_pytest_marker_manifest_keeps_reason_condition_and_strict(tmp_path: Path):
    _write(
        tmp_path / "tests/test_contract.py",
        """
import pytest

@pytest.mark.skipif(not AVAILABLE, reason="optional runtime")
def test_optional():
    pass

@pytest.mark.xfail(reason="known AD gap", strict=True)
def test_gap():
    pass
""",
    )

    entries = baseline.pytest_marker_manifest(tmp_path)["entries"]

    assert [(entry["kind"], entry["reason"]) for entry in entries] == [
        ("skipif", "optional runtime"),
        ("xfail", "known AD gap"),
    ]
    assert entries[0]["condition"] == "not AVAILABLE"
    assert entries[1]["strict"] is True


def test_runtime_artifacts_reject_secret_bearing_payloads(tmp_path: Path):
    artifact = tmp_path / "runtime.json"
    artifact.write_text(json.dumps({"api_token": "must-not-leak"}), encoding="utf-8")

    with pytest.raises(baseline.BaselineError, match="sensitive key"):
        baseline.runtime_artifact_manifest([f"solver-results={artifact}"])


def test_environment_manifest_records_presence_but_not_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("RAYD_SOURCE_DIR", "do-not-record-this-path")
    monkeypatch.setattr(
        baseline,
        "_git_state",
        lambda _repo: {"sha": "a" * 40, "dirty": False, "worktree_status": []},
    )
    monkeypatch.setattr(
        baseline,
        "_rayd_state",
        lambda _repo, _rayd: {"available": False, "dirty": None, "sha": None},
    )
    monkeypatch.setattr(baseline, "_command_versions", lambda _repo: {})
    monkeypatch.setattr(
        baseline,
        "_torch_runtime_manifest",
        lambda _repo: {
            "status": "ok",
            "torch_version": "fixture",
            "cuda_runtime": "fixture",
            "cuda_available": False,
            "devices": [],
        },
    )
    monkeypatch.setattr(baseline, "_package_version", lambda _name: None)

    manifest = baseline.environment_manifest(tmp_path)
    serialized = json.dumps(manifest)

    assert manifest["environment_variables_present"]["RAYD_SOURCE_DIR"] is True
    assert "do-not-record-this-path" not in serialized
    assert manifest["privacy"]["environment_values_collected"] is False


def test_git_state_collapses_private_claude_paths(monkeypatch: pytest.MonkeyPatch):
    def fake_git(_repo, *args, **_kwargs):
        if args == ("rev-parse", "HEAD"):
            return "d" * 40
        return "\n".join(
            [
                "?? .claude/settings.local.json",
                "?? .claude/private/notes.md",
                "?? tools/refactor_baseline.py",
            ]
        )

    monkeypatch.setattr(baseline, "_git", fake_git)

    state = baseline._git_state(Path("fixture"))

    assert state["worktree_status"] == [
        "?? .claude/",
        "?? tools/refactor_baseline.py",
    ]
    assert state["private_untracked_roots"] == [".claude/"]
    assert "settings.local" not in json.dumps(state)
    assert "notes.md" not in json.dumps(state)


def test_torch_runtime_query_is_isolated_from_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    recorded = {}

    def fake_run(command, *, cwd, timeout):
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["timeout"] = timeout
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                '{"cuda_available":true,"cuda_runtime":"12.9",'
                '"devices":[{"capability":[12,0],"index":0,"sm":120}],'
                '"torch_version":"2.8.0"}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(baseline, "_run", fake_run)

    manifest = baseline._torch_runtime_manifest(tmp_path)

    assert manifest == {
        "status": "ok",
        "cuda_available": True,
        "cuda_runtime": "12.9",
        "devices": [{"capability": [12, 0], "index": 0, "sm": 120}],
        "torch_version": "2.8.0",
    }
    assert "import torch" in recorded["command"][2]
    assert "channel" not in recorded["command"][2]


def test_build_manifest_records_compile_contract_and_redacts_cache_paths(
    tmp_path: Path,
):
    _write(
        tmp_path / "CMakeLists.txt",
        """
set(CHANNEL_DEFAULT_CUDA_ARCHITECTURES "80-real;90-virtual")
set(CHANNEL_DEFAULT_TORCH_CUDA_ARCH_LIST "8.0 9.0+PTX")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "build type")
target_compile_definitions(_channel PRIVATE FEATURE_A=1 FEATURE_B=1)
set_source_files_properties(
    native/channel/bridge.cpp
    PROPERTIES COMPILE_OPTIONS "/EHc-")
""",
    )
    _write(tmp_path / "pyproject.toml", "[build-system]\nrequires=[]\n")
    cache = tmp_path / "CMakeCache.txt"
    _write(
        cache,
        """
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_CACHE_MAJOR_VERSION:INTERNAL=3
CMAKE_CACHE_MINOR_VERSION:INTERNAL=31
CMAKE_CACHE_PATCH_VERSION:INTERNAL=6
CMAKE_COMMAND:INTERNAL=C:\\Users\\Alice\\cmake.exe
CMAKE_CXX_FLAGS:STRING=/DWIN32 /I"C:\\Users\\Alice\\private-include"
CMAKE_CXX_COMPILER:FILEPATH=C:\\Users\\Alice\\compiler.exe
RAYD_SOURCE_DIR:PATH=C:\\Users\\Alice\\RayDi
""",
    )
    _write(
        tmp_path / "CMakeFiles/3.31.6/CMakeCXXCompiler.cmake",
        """
set(CMAKE_CXX_COMPILER "C:/Users/Alice/MSVC/cl.exe")
set(CMAKE_CXX_COMPILER_ID "MSVC")
set(CMAKE_CXX_COMPILER_VERSION "19.44.35219.0")
set(MSVC_CXX_ARCHITECTURE_ID x64)
""",
    )
    _write(
        tmp_path / "CMakeFiles/3.31.6/CMakeCUDACompiler.cmake",
        """
set(CMAKE_CUDA_COMPILER "C:/Users/Alice/CUDA/bin/nvcc.exe")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "12.9.41")
set(CMAKE_CUDA_SIMULATE_ID "MSVC")
set(CMAKE_CUDA_SIMULATE_VERSION "19.44")
""",
    )

    manifest = baseline.build_manifest(tmp_path, cmake_cache=cache)
    serialized = json.dumps(manifest)

    assert manifest["cmake_source_contract"] == {
        "cxx_standard": "17",
        "default_build_type": "Release",
        "default_cuda_architectures": "80-real;90-virtual",
        "default_torch_cuda_arch_list": "8.0 9.0+PTX",
        "target_compile_definitions": ["FEATURE_A=1", "FEATURE_B=1"],
        "per_source_compile_options": [
            {
                "path": "native/channel/bridge.cpp",
                "options": ["/EHc-"],
            }
        ],
    }
    assert [value["key"] for value in manifest["cmake_cache_contract"]["values"]] == [
        "CMAKE_BUILD_TYPE",
        "CMAKE_CACHE_MAJOR_VERSION",
        "CMAKE_CACHE_MINOR_VERSION",
        "CMAKE_CACHE_PATCH_VERSION",
        "CMAKE_COMMAND",
        "CMAKE_CXX_FLAGS",
    ]
    assert manifest["cmake_cache_contract"]["cmake_version"] == "3.31.6"
    compilers = manifest["cmake_toolchain_contract"]["compilers"]
    assert [compiler["kind"] for compiler in compilers] == ["cuda", "cxx"]
    assert compilers[0]["values"] == {
        "CMAKE_CUDA_COMPILER_ID": "NVIDIA",
        "CMAKE_CUDA_COMPILER_VERSION": "12.9.41",
        "CMAKE_CUDA_SIMULATE_ID": "MSVC",
        "CMAKE_CUDA_SIMULATE_VERSION": "19.44",
    }
    assert compilers[1]["values"] == {
        "CMAKE_CXX_COMPILER_ID": "MSVC",
        "CMAKE_CXX_COMPILER_VERSION": "19.44.35219.0",
        "MSVC_CXX_ARCHITECTURE_ID": "x64",
    }
    assert "private-include" not in serialized
    assert "compiler.exe" not in serialized
    assert "RayDi" not in serialized
    assert "Users/Alice" not in serialized
    assert manifest["missing_build_inputs"] == ["native-binary"]


def test_pytest_collection_uses_repository_local_basetemp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    recorded: dict[str, object] = {}

    def fake_run(command, *, cwd, timeout, env):
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["env"] = env
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="tests/test_contract.py::test_one\n\n1 test collected\n",
            stderr="",
        )

    monkeypatch.setattr(baseline, "_run", fake_run)
    scratch = tmp_path / "artifacts/pytest-baseline"

    manifest = baseline.collect_pytest_manifest(tmp_path, scratch)

    assert f"--basetemp={scratch}" in recorded["command"]
    assert recorded["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert manifest["command"][-1] == "--basetemp=artifacts/pytest-baseline"
    assert manifest["nodeids"] == ["tests/test_contract.py::test_one"]


def test_freeze_is_immutable_and_marks_partial_artifact_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    _write(repo / "pyproject.toml", "[project]\nname='fixture'\nversion='1'\n")
    (repo / ".git").mkdir()
    sha = "b" * 40
    monkeypatch.setattr(
        baseline,
        "_git_state",
        lambda _repo: {"sha": sha, "dirty": False, "worktree_status": []},
    )
    monkeypatch.setattr(baseline, "environment_manifest", lambda *_args: {})
    monkeypatch.setattr(baseline, "build_manifest", lambda *_args: {})
    monkeypatch.setattr(
        baseline,
        "api_manifest",
        lambda *_args: {"schema_version": 1, "modules": [], "schemas": []},
    )
    monkeypatch.setattr(baseline, "binding_manifest", lambda *_args: {})
    monkeypatch.setattr(baseline, "import_graph_manifest", lambda *_args: {})
    monkeypatch.setattr(baseline, "body_hash_manifest", lambda *_args: {})
    monkeypatch.setattr(baseline, "pytest_marker_manifest", lambda *_args: {})

    destination = baseline.freeze_baseline(repo, output, collect_tests=False)

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert destination == output / sha
    assert manifest["complete"] is False
    assert "schemas.json" in manifest["files"]
    with pytest.raises(
        baseline.BaselineError, match="immutable baseline already exists"
    ):
        baseline.freeze_baseline(repo, output, collect_tests=False)