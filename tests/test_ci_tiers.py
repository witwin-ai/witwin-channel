from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib

from ci import run_ci_tier as tiers


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _ids(tier: str) -> list[str]:
    return [gate.id for gate in tiers.TIER_GATES[tier]]


def _base_ids(tier: str) -> list[str]:
    ids = _ids(tier)
    assert ids[-1] == f"{tier}.repository-hygiene-final"
    return ids[:-1]


def test_four_tiers_are_cumulative_and_end_with_a_hygiene_recheck() -> None:
    assert tuple(tiers.TIER_GATES) == ("quick", "cuda", "nightly", "release")
    quick = _base_ids("quick")
    cuda = _base_ids("cuda")
    nightly = _base_ids("nightly")
    release = _base_ids("release")

    assert cuda[: len(quick)] == quick
    assert nightly[: len(cuda)] == cuda
    assert release[: len(nightly)] == nightly

    for tier in tiers.TIER_GATES:
        full = _ids(tier)
        assert full[-1] == f"{tier}.repository-hygiene-final"
        assert len(full) == len(set(full))


def test_tiers_cover_the_required_gate_families() -> None:
    assert {
        "quick.ruff",
        "quick.mypy",
        "quick.import-graph",
        "quick.contract-coverage",
        "quick.public-api-binding-contract-manifests",
        "quick.production-dependencies",
        "quick.product-identity",
        "quick.repository-hygiene",
        "quick.secret-scan",
        "quick.maintenance-budgets",
        "quick.import-no-native",
        "quick.cpu-import-config",
    } <= set(_ids("quick"))
    assert {
        "cuda.unit-contract-acceptance",
        "cuda.four-solver-smoke",
        "cuda.no-fallback",
        "cuda.ad-core",
    } <= set(_ids("cuda"))
    assert {
        "nightly.coverage-run-full-suite",
        "nightly.coverage-json",
        "nightly.coverage-policy",
        "nightly.munich-parity",
        "nightly.full-ad",
        "nightly.statistics-gate",
        "nightly.wheel-build-py311-cu128-win-x64",
        "nightly.wheel-smoke-py311-cu128-win-x64",
        "nightly.duplication",
    } <= set(_ids("nightly"))
    assert {
        "release.performance",
        "release.peak-memory",
        "release.cold-start",
        "release.scaling",
        "release.fresh-checkout-wheel-build",
        "release.fresh-checkout-wheel-smoke",
        "release.rayd-lock-build-identity",
    } <= set(_ids("release"))
    performance = next(
        gate for gate in tiers.RELEASE_GATES if gate.id == "release.performance"
    )
    assert performance.args[performance.args.index("--profile") + 1] == "full"
    scaling = next(gate for gate in tiers.RELEASE_GATES if gate.id == "release.scaling")
    assert scaling.args[scaling.args.index("--gpu-budget-gib") + 1] == "16"

    nightly = _ids("nightly")
    coverage = [
        nightly.index("nightly.coverage-run-full-suite"),
        nightly.index("nightly.coverage-json"),
        nightly.index("nightly.coverage-policy"),
    ]
    assert coverage == list(range(coverage[0], coverage[0] + 3))
    wheel_builds = [
        gate
        for gate in tiers.TIER_GATES["release"]
        if gate.id.endswith("wheel-build")
        or "wheel-build-py311-cu128-win-x64" in gate.id
    ]
    assert len(wheel_builds) == 2
    assert all("--no-isolation" in gate.args for gate in wheel_builds)
    wheel_smokes = {
        gate.id: gate.args
        for gate in tiers.TIER_GATES["release"]
        if "wheel-smoke" in gate.id
    }
    assert wheel_smokes == {
        "nightly.wheel-smoke-py311-cu128-win-x64": (
            "ci/wheel_smoke.py",
            "artifacts/nightly/wheel",
            "--output",
            "artifacts/nightly/wheel-smoke-pe-audit.v1.json",
        ),
        "release.fresh-checkout-wheel-smoke": (
            "ci/wheel_smoke.py",
            "artifacts/release/wheel",
            "--output",
            "artifacts/release/wheel-smoke-pe-audit.v1.json",
        ),
    }


def test_all_python_entry_points_in_the_registry_exist() -> None:
    paths = (
        argument.removeprefix("--ignore=")
        for gate in tiers.TIER_GATES["release"]
        for argument in gate.args
        if argument.endswith(".py")
    )
    missing = sorted(path for path in paths if not (ROOT / path).is_file())

    assert missing == []


def test_list_and_dry_run_never_spawn_processes(monkeypatch, capsys) -> None:
    def unexpected_run(*args, **kwargs):
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(tiers.subprocess, "run", unexpected_run)

    assert tiers.main(["quick", "--list", "--python", "python311"]) == 0
    listed = capsys.readouterr().out
    assert "quick.ruff\tpython311 -m ruff check" in listed
    assert tiers.main(["quick", "--dry-run", "--python", "python311"]) == 0
    dry_run = capsys.readouterr().out
    assert "[DRY-RUN] quick.ruff: python311 -m ruff check" in dry_run


def test_actual_execution_is_fail_fast(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], Path, bool]] = []
    return_codes = iter((0, 7, 0))

    class Result:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def fake_run(argv, *, cwd, check):
        calls.append((argv, cwd, check))
        return Result(next(return_codes))

    monkeypatch.setattr(tiers.subprocess, "run", fake_run)
    gates = (
        tiers.Gate("first", ("first.py",)),
        tiers.Gate("second", ("second.py",)),
        tiers.Gate("never", ("never.py",)),
    )

    assert tiers.run_gates(gates, python="python311", root=tmp_path) == 7
    assert calls == [
        (("python311", "first.py"), tmp_path, False),
        (("python311", "second.py"), tmp_path, False),
    ]


def test_paid_wheel_workflow_is_hosted_complete_and_opt_in() -> None:
    with (ROOT / "ci" / "support-matrix.toml").open("rb") as stream:
        runtime = tomllib.load(stream)["runtime"][0]
    assert runtime["python"] == "3.11"
    assert runtime["cuda_runtime"] == "12.8"
    assert runtime["cuda_toolkit"] == "12.8.1"
    assert runtime["verified_sm"] == [120]
    assert runtime["declared_unverified_sm"] == [
        70,
        75,
        80,
        86,
        87,
        89,
        90,
        100,
        101,
    ]

    assert {path.name for path in WORKFLOWS.glob("*.yml")} == {
        "publish-witwin-channel.yml"
    }
    workflow = (WORKFLOWS / "publish-witwin-channel.yml").read_text(
        encoding="utf-8"
    )
    assert "\n  push:" not in workflow
    assert "\n  pull_request:" not in workflow
    assert "\n  schedule:" not in workflow
    assert "\n  release:\n    types: [published]" in workflow
    assert "\n  workflow_dispatch:" in workflow
    assert "self-hosted" not in workflow
    assert "runs-on: windows-2022" in workflow
    assert "runs-on: ubuntu-22.04" in workflow
    assert "manylinux_2_28" in workflow

    locked_rayd = "49c58c4cb8212f6babb920cc88fb937509826cc5"
    lock = json.loads(
        (ROOT / "dependencies" / "rayd.lock.json").read_text(encoding="utf-8")
    )
    assert lock["commit"] == locked_rayd
    assert lock["source_bundle"]["distribution_version"] == "0.7.0"
    assert f"RAYD_COMMIT: {locked_rayd}" in workflow
    assert "repository: Asixa/RayD" in workflow
    assert "RAYD_SOURCE_DIR=" in workflow
    assert "-DCHANNEL_RELEASE_BUILD=ON" in workflow

    full_arches = (
        "70-real;75-real;80-real;86-real;87-real;89-real;"
        "90-real;100-real;101-real;120-real;120-virtual"
    )
    assert full_arches in workflow
    assert "--expected-sass 70,75,80,86,87,89,90,100,101,120" in workflow
    assert "--expected-ptx 120" in workflow
    assert "CHANNEL_CUDA_GENCODE_FLAGS=" in workflow
    assert "RAYD_TORCH_CUDA_GENCODE_FLAGS=" in workflow
    assert "CMAKE_CUDA_COMPILER_LAUNCHER: \"\"" in workflow
    assert "CMAKE_BUILD_PARALLEL_LEVEL: \"3\"" in workflow
    assert ".Path.Replace('\\', '/')" in workflow
    assert "actions/cache@v5" in workflow
    assert "sub-packages:" in workflow
    assert "safe.directory /project/channel" in workflow
    assert "safe.directory /host${{ github.workspace }}/rayd" in workflow
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "CHANNEL_CUDA_GENCODE_FLAGS" in cmake
    second_torch_find = cmake.index("find_package(Torch REQUIRED)", cmake.index("add_subdirectory("))
    channel_gencode_cleanup = cmake.index(
        "Channel CUDA flags after removing Torch gencode flags"
    )
    channel_target = cmake.index("Python_add_library(")
    assert second_torch_find < channel_gencode_cleanup < channel_target
    assert (
        '"(^|[ \\t])-gencode[ \\t]+arch=[^ \\t]+,code=[^ \\t]+"'
        in cmake
    )
    assert "set_target_properties(_channel PROPERTIES CUDA_ARCHITECTURES OFF)" in cmake
    assert "target_compile_options(" in cmake

    kernels = ROOT / "native" / "channel" / "kernels"
    cuda_sources = [*kernels.rglob("*.cu"), *kernels.rglob("*.cuh")]
    native_root = ROOT / "native" / "channel"
    pending = list(cuda_sources)
    reachable_sources: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in reachable_sources:
            continue
        reachable_sources.add(source)
        text = source.read_text(encoding="utf-8")
        for include in re.findall(r'^\s*#include\s+"([^"]+)"', text, re.MULTILINE):
            candidate = (source.parent / include).resolve()
            if candidate.is_file() and candidate.is_relative_to(native_root):
                pending.append(candidate)
    extension_includes = [
        path
        for path in reachable_sources
        if "#include <torch/extension.h>" in path.read_text(encoding="utf-8")
    ]
    assert extension_includes == []
    minimal_header = (kernels / "torch_cuda_minimal.h").read_text(encoding="utf-8")
    assert "#include <ATen/ATen.h>" in minimal_header
    assert "#include <torch/csrc/utils/pybind.h>" in minimal_header
    assert "#include <torch/types.h>" not in minimal_header
    assert all(
        not re.search(r"(?<!rayd::)torch::", path.read_text(encoding="utf-8"))
        for path in cuda_sources
    )

    publish_guard = (
        "github.event_name == 'release' && github.event.action == 'published'"
    )
    assert publish_guard in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "id-token: write" in workflow
