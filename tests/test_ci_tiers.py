# Copyright Xingyu Chen.
# Tests ci tiers.

from __future__ import annotations

import json
from pathlib import Path
import re

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
        "quick.native-bindings",
        "quick.native-inventory",
        "quick.duplication",
        "quick.compact-signatures",
        "quick.public-api-and-generated-checks",
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
        "cuda.survival",
        "cuda.reference-accuracy",
        "cuda.launch-budgets",
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
        "nightly.core-wheel-build",
        "nightly.wheel-smoke-py311-cu128-win-x64",
    } <= set(_ids("nightly"))
    assert {
        "release.performance",
        "release.peak-memory",
        "release.cold-start",
        "release.scaling",
        "release.fresh-checkout-wheel-build",
        "release.core-wheel-build",
        "release.fresh-checkout-wheel-smoke",
        "release.rayd-lock-build-identity",
    } <= set(_ids("release"))
    performance = next(gate for gate in tiers.RELEASE_GATES if gate.id == "release.performance")
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
        if gate.id.endswith("wheel-build") or "wheel-build-py311-cu128-win-x64" in gate.id
    ]
    # Four, not two: each tier that smokes a Channel wheel also has to produce
    # the Core wheel that goes into the same isolated target. `wheel_smoke.py`
    # requires --core-wheel, so before those two gates existed both smokes
    # exited in argparse and neither tier could reach the gate it names.
    assert len(wheel_builds) == 4
    assert all("--no-isolation" in gate.args for gate in wheel_builds)
    wheel_smokes = {
        gate.id: gate.args for gate in tiers.TIER_GATES["release"] if "wheel-smoke" in gate.id
    }
    assert wheel_smokes == {
        "nightly.wheel-smoke-py311-cu128-win-x64": (
            "ci/wheel_smoke.py",
            "artifacts/nightly/wheel",
            "--core-wheel",
            "artifacts/nightly/core-wheel",
            "--output",
            "artifacts/nightly/wheel-smoke-pe-audit.v1.json",
        ),
        "release.fresh-checkout-wheel-smoke": (
            "ci/wheel_smoke.py",
            "artifacts/release/wheel",
            "--core-wheel",
            "artifacts/release/core-wheel",
            "--output",
            "artifacts/release/wheel-smoke-pe-audit.v1.json",
        ),
    }
    # Every wheel smoke must receive the Core wheel produced by the gate that
    # runs immediately before it, in the same tier. Pinning the ORDER is the
    # part that matters: a smoke scheduled before its producer would fail on a
    # missing directory instead of on the artifact it is supposed to audit.
    ordered = _ids("release")
    for smoke_id, core_gate_id in (
        ("nightly.wheel-smoke-py311-cu128-win-x64", "nightly.core-wheel-build"),
        ("release.fresh-checkout-wheel-smoke", "release.core-wheel-build"),
    ):
        assert ordered.index(core_gate_id) < ordered.index(smoke_id)


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
    from ci import wheel_matrix

    matrix = wheel_matrix.load()
    assert matrix["cuda_runtime"] == "12.8"
    assert matrix["cuda_toolkit"] == "12.8.1"
    # x86_64 wheels carry no Jetson-only cubins; their exact-match SMs cannot run on x86_64.
    assert not set(matrix["sass"]) & {72, 87, 101, 110}

    assert {path.name for path in WORKFLOWS.glob("*.yml")} == {
        "gpu-tier.yml",
        "publish-witwin-channel.yml",
        "quick-tier.yml",
    }
    workflow = (WORKFLOWS / "publish-witwin-channel.yml").read_text(encoding="utf-8")
    assert "\n  push:" not in workflow
    assert "\n  pull_request:" not in workflow
    assert "\n  schedule:" not in workflow
    assert "\n  release:\n    types: [published]" in workflow
    assert "\n  workflow_dispatch:" in workflow
    assert "          - windows-smoke" in workflow
    assert "          - full" in workflow
    assert "self-hosted" not in workflow
    assert "runs-on: windows-2022" in workflow
    assert "runs-on: ubuntu-22.04" in workflow
    assert "manylinux_2_28" in workflow

    locked_rayd = "5b6e10dc4b85ca82e2cd995508296a391b9170ed"
    locked_core = "46c826d969654912461d2e18b6cb87df9cc3df8f"
    lock = json.loads((ROOT / "dependencies" / "rayd.lock.json").read_text(encoding="utf-8"))
    assert lock["commit"] == locked_rayd
    assert lock["source_bundle"]["distribution_version"] == "0.10.0"
    assert f"RAYD_COMMIT: {locked_rayd}" in workflow
    assert f"CORE_COMMIT: {locked_core}" in workflow
    assert "repository: witwin-ai/witwin-core" in workflow
    assert "python -m pip install --no-deps ./core" in workflow
    assert "repository: Asixa/RayD" in workflow
    assert "RAYD_SOURCE_DIR=" in workflow
    assert "-DCHANNEL_RELEASE_BUILD=ON" in workflow

    # Every architecture, flag, size limit and matrix row comes from ci/support-matrix.toml.
    assert "python ci/wheel_matrix.py github-outputs" in workflow
    assert workflow.count("include: ${{ fromJSON(needs.metadata.outputs.build) }}") == 2
    assert (
        workflow.count(
            "-DCMAKE_CUDA_ARCHITECTURES=${{ needs.metadata.outputs.cuda_architectures }}"
        )
        == 2
    )
    assert not re.search(r"\b\d+-real\b|sm_\d+|compute_\d+|--expected-sass \d", workflow)
    assert workflow.count("--expected-sass ${{ needs.metadata.outputs.expected_sass }}") == 1
    assert workflow.count("--expected-sass $env:EXPECTED_SASS") == 1
    assert workflow.count("rayd/drjit/scripts/verify_cuda_binary_arches.py") == 4
    assert "rayd/backends/drjit/scripts" not in workflow
    assert workflow.count("wheel_matrix.py check-size") == 2
    assert "check-wheels dist/pypi --channel pypi" in workflow
    assert "check-wheels dist/release-asset --channel release-asset" in workflow
    assert workflow.count("wheel_matrix.py select-torch") == 2
    assert "torch==${{ matrix.torch }} --index-url https://download.pytorch.org/whl/cu128" in workflow
    assert "CIBW_BUILD: ${{ matrix.python_tag }}-manylinux_x86_64" in workflow
    assert "python-version: ${{ matrix.python }}" in workflow
    assert "CHANNEL_CUDA_GENCODE_FLAGS" in workflow
    assert "RAYD_TORCH_CUDA_GENCODE_FLAGS" in workflow
    assert 'CMAKE_CUDA_COMPILER_LAUNCHER: ""' in workflow
    assert 'CMAKE_BUILD_PARALLEL_LEVEL: "3"' in workflow
    assert "CMAKE_BUILD_PARALLEL_LEVEL=2" in workflow
    assert ".Path.Replace('\\', '/')" in workflow
    assert "actions/cache@v5" in workflow
    assert "sub-packages:" in workflow
    assert "safe.directory /project/channel" in workflow
    assert "safe.directory /host${{ github.workspace }}/rayd" in workflow
    assert "python channel/ci/wheel_smoke.py" in workflow
    assert "import witwin.channel as channel" in workflow
    assert "from witwin.channel.propagation import consumer" in workflow
    assert '"enumerated_canonical_compact"' in workflow
    assert '"enumerated_exact_pair_metadata"' in workflow
    assert '"consumer_los_jones"' in workflow
    assert '"runtime_probe"' in workflow
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "CHANNEL_CUDA_GENCODE_FLAGS" in cmake
    second_torch_find = cmake.index(
        "find_package(Torch REQUIRED)", cmake.index("add_subdirectory(")
    )
    channel_gencode_cleanup = cmake.index("Channel CUDA flags after removing Torch gencode flags")
    channel_target = cmake.index("Python_add_library(")
    assert second_torch_find < channel_gencode_cleanup < channel_target
    assert '"(^|[ \\t])-gencode[ \\t]+arch=[^ \\t]+,code=[^ \\t]+"' in cmake
    assert "set_target_properties(_channel PROPERTIES CUDA_ARCHITECTURES OFF)" in cmake
    assert "torch._C._GLIBCXX_USE_CXX11_ABI" in cmake
    assert 'TORCH_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI' not in cmake
    suffix_probe = cmake.index("OUTPUT_VARIABLE CHANNEL_PYTHON_EXTENSION_SUFFIX")
    windows_block = cmake.index("if(WIN32)", suffix_probe)
    suffix_target = cmake.index(
        'PROPERTIES SUFFIX "${CHANNEL_PYTHON_EXTENSION_SUFFIX}"',
    )
    assert suffix_probe < windows_block < suffix_target
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
    minimal_header = (kernels / "torch_cuda.h").read_text(encoding="utf-8")
    assert "#include <ATen/ATen.h>" in minimal_header
    assert "#include <torch/csrc/utils/pybind.h>" in minimal_header
    assert "#include <torch/types.h>" not in minimal_header
    assert all(
        not re.search(r"(?<!rayd::)torch::", path.read_text(encoding="utf-8"))
        for path in cuda_sources
    )
    assert all("at::kFloat32" not in path.read_text(encoding="utf-8") for path in cuda_sources)
    raw_typed_range = re.compile(
        r"for\s*\([^)]*:\s*\{(?P<items>.*?)\}\)\s*\{",
        re.DOTALL,
    )
    ambiguous_ranges = []
    for path in cuda_sources:
        text = path.read_text(encoding="utf-8")
        for match in raw_typed_range.finditer(text):
            items = match.group("items")
            if ("std::pair" in items or "std::tuple" in items) and re.search(
                r'^\s*\{"', items, re.MULTILINE
            ):
                ambiguous_ranges.append(path)
    assert ambiguous_ranges == []

    publish_guard = "github.event_name == 'release' && github.event.action == 'published'"
    assert workflow.count(publish_guard) == 2
    assert workflow.count("needs.validate-wheels.result == 'success'") == 2
    publish = workflow[workflow.index("\n  publish:") : workflow.index("\n  release-assets:")]
    assert "pattern: channel-wheel-pypi-*" in publish
    assert "release-asset" not in publish
    assert "pypa/gh-action-pypi-publish@release/v1" in publish
    assert "id-token: write" in publish
    assets = workflow[workflow.index("\n  release-assets:") :]
    assert "pattern: channel-wheel-release-asset-*" in assets
    assert "contents: write" in assets
    assert 'gh release upload "$RELEASE_TAG"' in assets


def test_quick_tier_workflow_runs_on_every_push_and_pull_request() -> None:
    workflow = (WORKFLOWS / "quick-tier.yml").read_text(encoding="utf-8")
    publish = (WORKFLOWS / "publish-witwin-channel.yml").read_text(encoding="utf-8")

    assert "\n  push:\n    branches:\n      - main" in workflow
    assert "\n  pull_request:" in workflow
    assert "run: python ci/run_ci_tier.py quick" in workflow
    assert "self-hosted" not in workflow
    # The quick tier never loads the extension, so it must not pay for CUDA wheels.
    assert "download.pytorch.org/whl/cpu" in workflow
    assert "cu128" not in workflow
    core_pin = re.search(r"CORE_COMMIT: ([0-9a-f]{40})", workflow)
    assert core_pin is not None
    assert f"CORE_COMMIT: {core_pin.group(1)}" in publish


def test_gpu_tier_workflow_builds_with_the_dev_scripts_on_a_gpu_runner() -> None:
    workflow = (WORKFLOWS / "gpu-tier.yml").read_text(encoding="utf-8")

    assert "\n  push:\n    branches:\n      - main" in workflow
    assert "\n  pull_request:" in workflow
    assert "runs-on: [self-hosted, linux, cuda]" in workflow
    assert "ref: ${{ steps.lock.outputs.commit }}" in workflow
    assert 'scripts/dev_build.sh --rayd "$GITHUB_WORKSPACE/rayd"' in workflow
    assert "source scripts/dev_env.sh" in workflow
    assert "python ci/run_ci_tier.py cuda" in workflow
    assert "pip install" not in workflow
