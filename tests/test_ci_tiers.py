from __future__ import annotations

import json
from pathlib import Path
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


def test_workflows_use_only_the_verified_windows_cuda_runner() -> None:
    with (ROOT / "ci" / "support-matrix.toml").open("rb") as stream:
        runtime = tomllib.load(stream)["runtime"][0]
    assert runtime["python"] == "3.11"
    assert runtime["cuda_runtime"] == "12.8"
    assert runtime["verified_sm"] == [120]

    expected = {
        "quick-pr.yml": ("pull_request:", "quick"),
        "cuda-pr.yml": ("pull_request:", "cuda"),
        "nightly.yml": ("schedule:", "nightly"),
        "release.yml": ("release:", "release"),
    }
    assert {path.name for path in WORKFLOWS.glob("*.yml")} == set(expected)
    for name, (trigger, tier) in expected.items():
        source = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert trigger in source
        assert "workflow_dispatch:" in source
        assert "runs-on: [self-hosted, Windows, X64, CUDA]" in source
        assert "${{ vars.WITWIN2_PYTHON }}" in source
        assert "Repository variable WITWIN2_PYTHON is required" in source
        assert "C:\\Users\\" not in source
        assert "sys.version_info[:2] == (3, 11)" in source
        assert "torch.version.cuda == '12.8'" in source
        assert f"ci/run_ci_tier.py {tier}" in source

    quick = (WORKFLOWS / "quick-pr.yml").read_text(encoding="utf-8")
    assert "torch.cuda.is_available()" not in quick
    assert "torch.cuda.get_device_capability()" not in quick
    assert "SM120" not in quick

    locked_rayd = "5df6497d36ac941bbc88b26dc3c16e373ee37705"
    lock = json.loads(
        (ROOT / "dependencies" / "rayd.lock.json").read_text(encoding="utf-8")
    )
    assert lock["commit"] == locked_rayd
    for name in ("cuda-pr.yml", "nightly.yml", "release.yml"):
        source = (WORKFLOWS / name).read_text(encoding="utf-8")
        rayd_checkout = source.split("- name: Checkout locked RayD", maxsplit=1)[
            1
        ].split("- name: Verify immutable RayD checkout", maxsplit=1)[0]
        assert "repository: Asixa/RayD" in source
        assert [
            line.strip()
            for line in rayd_checkout.splitlines()
            if line.strip().startswith("ref:")
        ] == [f"ref: {locked_rayd}"]
        assert f"$expected = '{locked_rayd}'" in source
        assert "dependencies/rayd.lock.json" in source
        assert "ConvertFrom-Json" in source
        assert "git -C $rayd rev-parse HEAD" in source
        assert source.index(f"ref: {locked_rayd}") < source.index(
            "dependencies/rayd.lock.json"
        )
        assert "torch.cuda.is_available()" in source
        assert "torch.cuda.get_device_capability() == (12, 0)" in source
        assert "-m cmake -S . -B $buildDir" in source
        assert "-m cmake --build $buildDir" in source
        assert "WITWIN_CHANNEL_NATIVE_DEVELOPER_OVERRIDE=1" in source
        assert "WITWIN_CHANNEL_NATIVE_EXTENSION_PATH=" in source
        assert "WITWIN_CHANNEL_NATIVE_EXPECTED_FINGERPRINT=" in source
        assert "CMAKE_ARGS=-DRAYD_SOURCE_DIR=$rayd" in source
        assert "WITWIN_RAYD_DIR" in source

    cuda = (WORKFLOWS / "cuda-pr.yml").read_text(encoding="utf-8")
    nightly_source = (WORKFLOWS / "nightly.yml").read_text(encoding="utf-8")
    release = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert "-DCHANNEL_NATIVE_RELEASE_BUILD=OFF" in cuda
    assert "-DCHANNEL_NATIVE_RELEASE_BUILD=ON" not in cuda
    assert "-DCHANNEL_NATIVE_RELEASE_BUILD=ON" in nightly_source
    assert "-DCHANNEL_NATIVE_RELEASE_BUILD=OFF" not in nightly_source
    assert "-DCHANNEL_NATIVE_RELEASE_BUILD=ON" in release
    assert "-DCHANNEL_NATIVE_RELEASE_BUILD=OFF" not in release
    assert "actions/upload-artifact@v4" in nightly_source
    assert "artifacts/nightly/wheel-smoke-pe-audit.v1.json" in nightly_source
    assert "actions/upload-artifact@v4" in release
    assert "artifacts/nightly/wheel-smoke-pe-audit.v1.json" in release
    assert "artifacts/release/wheel-smoke-pe-audit.v1.json" in release
    assert "schedule:" in release
    assert "types: [published]" in release
    assert "github.run_id" in release
    assert "fetch-depth: 0" in release
