# Copyright Xingyu Chen.
# Tests the generated native source inventory and the native layout rules.

from __future__ import annotations

from pathlib import Path

import pytest

from ci import check_native_inventory as check


ROOT = Path(__file__).resolve().parents[1]

_CMAKE = """\
Python_add_library(
    _channel
    MODULE
    native/channel/binding/module.cpp
    native/channel/kernels/demo.cu
)
set_source_files_properties(
    native/channel/kernels/demo.cu
    PROPERTIES COMPILE_OPTIONS "--fmad=false")
"""

_KERNEL = """\
#include "torch_cuda.h"
// demo_kernel<<<1, 1>>>() in a comment is not a launch.
__global__ void demo_kernel(float* out) { out[0] = 1.0f; }
template <int N>
__global__ void __launch_bounds__(256) templated_kernel(float* out) { out[0] = N; }
void launch(float* out, cudaStream_t stream) {
    demo_kernel<<<1, 1, 0, stream>>>(out);
    templated_kernel<4><<<1, 1, 0, stream>>>(out);
    cudaStreamSynchronize(stream);
}
"""


def _repo(tmp_path: Path) -> Path:
    kernels = tmp_path / "native/channel/kernels"
    binding = tmp_path / "native/channel/binding"
    kernels.mkdir(parents=True)
    binding.mkdir(parents=True)
    (tmp_path / "ci").mkdir()
    (kernels / "demo.cu").write_text(_KERNEL, encoding="utf-8")
    (kernels / "torch_cuda.h").write_text("#include <ATen/ATen.h>\n", encoding="utf-8")
    (binding / "module.cpp").write_text("PYBIND11_MODULE(_channel, module) {}\n", encoding="utf-8")
    (tmp_path / "CMakeLists.txt").write_text(_CMAKE, encoding="utf-8")
    return tmp_path


def test_inventory_lists_names_without_lines_or_counts(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    entry = check.generate_inventory(repo)["files"]["native/channel/kernels/demo.cu"]  # type: ignore[index]
    assert entry == {
        "kernels": ["demo_kernel", "templated_kernel"],
        "launches": ["demo_kernel", "templated_kernel"],
        "host_synchronization": ["cudaStreamSynchronize"],
        "device_trap": False,
        "compile_options": ["--fmad=false"],
    }
    assert check.check_layout(repo) == []


def test_layout_rejects_an_unregistered_or_twice_registered_source(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "native/channel/kernels/extra.cu").write_text("", encoding="utf-8")
    cmake = repo / "CMakeLists.txt"
    cmake.write_text(_CMAKE.replace("    native/channel/kernels/demo.cu\n)", "    native/channel/kernels/demo.cu\n    native/channel/kernels/demo.cu\n)"), encoding="utf-8")

    problems = check.check_layout(repo)
    assert "native/channel/kernels/extra.cu is listed 0 times in the _channel target (expected once)" in problems
    assert "native/channel/kernels/demo.cu is listed 2 times in the _channel target (expected once)" in problems


@pytest.mark.parametrize(
    ("relative", "text", "message"),
    [
        ("native/channel/binding/fields.cpp", "void f() { k<<<1, 1>>>(); }", "launches a kernel"),
        ("native/channel/binding/fields.cpp", "PYBIND11_MODULE(x, m) {}", "defines a pybind module"),
        ("native/channel/kernels/common.cuh", "", "header names are owner nouns"),
        ("native/channel/kernels/bad.cuh", "torch::Tensor x;", "names bare torch::"),
        ("native/channel/kernels/bad.cuh", "auto t = at::kFloat32;", "names at::kFloat32"),
        ("native/channel/kernels/bad.cuh", "#include <torch/extension.h>\n", "reaches <torch/extension.h>"),
    ],
)
def test_layout_rejects_each_forbidden_pattern(tmp_path: Path, relative: str, text: str, message: str) -> None:
    repo = _repo(tmp_path)
    (repo / relative).write_text(text, encoding="utf-8")
    if relative.endswith(".cpp"):
        cmake = repo / "CMakeLists.txt"
        cmake.write_text(cmake.read_text(encoding="utf-8").replace("MODULE\n", f"MODULE\n    {relative}\n"), encoding="utf-8")

    assert any(message in problem for problem in check.check_layout(repo))


def test_layout_rejects_fast_math(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    cmake = repo / "CMakeLists.txt"
    cmake.write_text(cmake.read_text(encoding="utf-8") + 'add_compile_options("--use_fast_math")\n', encoding="utf-8")

    assert "CMakeLists.txt enables --use_fast_math for Channel sources" in check.check_layout(repo)


def test_the_repository_inventory_is_current() -> None:
    assert check.main(["--repo", str(ROOT)]) == 0
