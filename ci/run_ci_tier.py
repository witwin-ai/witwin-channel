"""List or execute the repository's four fail-fast CI tiers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True, slots=True)
class Gate:
    id: str
    args: tuple[str, ...]

    def argv(self, python: str) -> tuple[str, ...]:
        return (python, *self.args)


QUICK_GATES = (
    Gate("quick.ruff", ("-m", "ruff", "check", "src", "tests", "benchmarks", "ci")),
    Gate("quick.mypy", ("-m", "mypy")),
    Gate("quick.import-graph", ("ci/check_import_graph.py",)),
    Gate("quick.contract-coverage", ("ci/check_contract_coverage.py",)),
    Gate(
        "quick.public-api-binding-contract-manifests",
        (
            "-m",
            "pytest",
            "-q",
            "tests/test_public_api_snapshot.py",
            "tests/test_binding_manifest_contract.py",
            "tests/test_phase10_legacy_dead_binding_audit.py",
        ),
    ),
    Gate("quick.production-dependencies", ("ci/check_production_dependencies.py",)),
    Gate("quick.product-identity", ("ci/check_product_identity.py",)),
    Gate("quick.repository-hygiene", ("ci/check_repository_hygiene.py",)),
    Gate("quick.secret-scan", ("ci/check_secrets.py",)),
    Gate("quick.maintenance-budgets", ("ci/check_maintenance_budgets.py",)),
    Gate(
        "quick.import-no-native",
        (
            "-c",
            "import sys; sys.path.insert(0, 'src'); "
            "import torch, witwin.channel; "
            "assert 'witwin.channel._channel' not in sys.modules; "
            "assert not torch.cuda.is_initialized()",
        ),
    ),
    Gate(
        "quick.cpu-import-config",
        (
            "-m",
            "pytest",
            "-q",
            "tests/test_import_contract.py",
            "tests/test_supported_runtime_matrix.py",
            "tests/test_capabilities.py",
            "tests/deterministic/test_config.py",
            "tests/montecarlo/basic/test_basic_config.py",
            "tests/montecarlo/bdpt/test_config.py",
            "tests/path/test_path_config.py",
            "tests/test_solver_config_metadata.py",
        ),
    ),
)

CUDA_GATES = (
    Gate(
        "cuda.unit-contract-acceptance",
        (
            "-m",
            "pytest",
            "-q",
            "tests",
            "--ignore=tests/ad",
            "--ignore=tests/performance",
            "--ignore=tests/deterministic/test_munich_deterministic_parity.py",
            "--ignore=tests/montecarlo/bdpt/test_munich_bdpt_parity.py",
            "--ignore=tests/scene/test_munich_loader_parity.py",
        ),
    ),
    Gate(
        "cuda.four-solver-smoke",
        (
            "-m",
            "pytest",
            "-q",
            "tests/path/test_path_solver_smoke.py",
            "tests/deterministic/test_los_empty_space.py",
            "tests/montecarlo/basic/test_basic_solver_smoke.py",
            "tests/montecarlo/bdpt/test_los_empty_space.py",
        ),
    ),
    Gate(
        "cuda.no-fallback",
        (
            "-m",
            "pytest",
            "-q",
            "tests/kernels/test_native_loader_no_fallback.py",
            "tests/scattering/test_sampling.py",
        ),
    ),
    Gate(
        "cuda.ad-core",
        (
            "-m",
            "pytest",
            "-q",
            "tests/runtime/test_autograd_contracts.py",
            "tests/propagation/geometry/test_autograd_kernels.py",
            "tests/propagation/fields/test_autograd_kernels.py",
            "tests/ad/test_field_em_ad.py",
            "tests/ad/test_rayd_geometry_ad.py",
        ),
    ),
)

NIGHTLY_GATES = (
    Gate(
        "nightly.coverage-run-full-suite",
        ("-m", "coverage", "run", "--branch", "-m", "pytest", "-q", "tests"),
    ),
    Gate(
        "nightly.coverage-json",
        ("-m", "coverage", "json", "-o", ".coverage.nightly.json"),
    ),
    Gate(
        "nightly.coverage-policy",
        ("ci/check_coverage.py", ".coverage.nightly.json"),
    ),
    Gate(
        "nightly.munich-parity",
        (
            "-m",
            "pytest",
            "-q",
            "tests/scene/test_munich_loader_parity.py",
            "tests/deterministic/test_munich_deterministic_parity.py",
            "tests/montecarlo/bdpt/test_munich_bdpt_parity.py",
            "tests/ad/test_munich_ad_smoke.py",
        ),
    ),
    Gate("nightly.full-ad", ("-m", "pytest", "-q", "tests/ad")),
    Gate(
        "nightly.statistics-gate",
        (
            "benchmarks/bench_phase_c_statistics.py",
            "--mode",
            "full",
            "--json",
            "artifacts/nightly/phase-c-statistics.v1.json",
        ),
    ),
    Gate(
        "nightly.wheel-build-py311-cu128-win-x64",
        (
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            "artifacts/nightly/wheel",
        ),
    ),
    # The smoke installs Core and Channel into ONE isolated target with
    # --no-deps, so it needs the Core wheel as an input and refuses to run
    # without it. Until this gate existed the smoke below died in argparse and
    # the tier reported the failure of a gate it had never actually reached.
    Gate(
        "nightly.core-wheel-build",
        (
            "ci/build_core_wheel.py",
            "--outdir",
            "artifacts/nightly/core-wheel",
            "--no-isolation",
        ),
    ),
    Gate(
        "nightly.wheel-smoke-py311-cu128-win-x64",
        (
            "ci/wheel_smoke.py",
            "artifacts/nightly/wheel",
            "--core-wheel",
            "artifacts/nightly/core-wheel",
            "--output",
            "artifacts/nightly/wheel-smoke-pe-audit.v1.json",
        ),
    ),
    Gate("nightly.duplication", ("ci/check_duplication.py",)),
)

RELEASE_GATES = (
    Gate(
        "release.performance",
        (
            "benchmarks/bench_phase_e_acceptance.py",
            "--profile",
            "full",
            "--fail-on-gate",
            "--output",
            "artifacts/release/phase-e-acceptance.v1.json",
        ),
    ),
    Gate(
        "release.peak-memory",
        (
            "benchmarks/bench_solver_peak_memory.py",
            "--tx",
            "1",
            "--rx",
            "1024",
            "--depth",
            "3",
            "--gpu-budget-gib",
            "16",
            "--output",
            "artifacts/release/solver-peak-memory.v1.json",
        ),
    ),
    Gate(
        "release.cold-start",
        (
            "benchmarks/bench_solver_cold_start.py",
            "--solvers",
            "path,deterministic,basic,bdpt",
            "--repeats",
            "3",
            "--output",
            "artifacts/release/solver-cold-start.v1.json",
        ),
    ),
    Gate(
        "release.scaling",
        (
            "benchmarks/bench_solver_scaling.py",
            "--solvers",
            "path,deterministic,basic,bdpt",
            "--gpu-budget-gib",
            "16",
            "--output",
            "artifacts/release/solver-scaling.v1.json",
        ),
    ),
    # Host-side compile cost against scene size. The solver benchmarks above
    # compile one scene once, outside their timing loops, so none of them can
    # see what a per-solve compile costs on a large scene.
    Gate(
        "release.compile-scaling",
        (
            "benchmarks/bench_compile_scaling.py",
            "--sizes",
            "256,1024",
            "--output",
            "artifacts/release/compile-scaling.v1.json",
        ),
    ),
    Gate(
        "release.fresh-checkout-wheel-build",
        (
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            "artifacts/release/wheel",
        ),
    ),
    # Same defect as the nightly pair, and it would have surfaced later and
    # cost more: a release tier that cannot reach its own wheel smoke.
    Gate(
        "release.core-wheel-build",
        (
            "ci/build_core_wheel.py",
            "--outdir",
            "artifacts/release/core-wheel",
            "--no-isolation",
        ),
    ),
    Gate(
        "release.fresh-checkout-wheel-smoke",
        (
            "ci/wheel_smoke.py",
            "artifacts/release/wheel",
            "--core-wheel",
            "artifacts/release/core-wheel",
            "--output",
            "artifacts/release/wheel-smoke-pe-audit.v1.json",
        ),
    ),
    Gate(
        "release.rayd-lock-build-identity",
        (
            "-m",
            "pytest",
            "-q",
            "tests/kernels/test_rayd_lock_cmake.py",
            "tests/kernels/test_build_identity_cmake.py",
        ),
    ),
)

def _tier(name: str, *gate_groups: tuple[Gate, ...]) -> tuple[Gate, ...]:
    """Compose a tier and append a trailing repository-hygiene re-check.

    The quick tier already runs ``repository-hygiene`` first, so the tree is
    proven clean before the tests. Re-running the same check last makes any
    test that dirties the worktree (a stray artifact, an unrestored fixture)
    fail its tier instead of leaking state into the next gate.
    """

    gates = tuple(gate for group in gate_groups for gate in group)
    trailing = Gate(
        f"{name}.repository-hygiene-final", ("ci/check_repository_hygiene.py",)
    )
    return (*gates, trailing)


TIER_GATES = {
    "quick": _tier("quick", QUICK_GATES),
    "cuda": _tier("cuda", QUICK_GATES, CUDA_GATES),
    "nightly": _tier("nightly", QUICK_GATES, CUDA_GATES, NIGHTLY_GATES),
    "release": _tier(
        "release", QUICK_GATES, CUDA_GATES, NIGHTLY_GATES, RELEASE_GATES
    ),
}


def format_gate(gate: Gate, python: str) -> str:
    return subprocess.list2cmdline(gate.argv(python))


def run_gates(
    gates: tuple[Gate, ...], *, python: str, root: Path, dry_run: bool = False
) -> int:
    for gate in gates:
        command = format_gate(gate, python)
        prefix = "DRY-RUN" if dry_run else "RUN"
        print(f"[{prefix}] {gate.id}: {command}", flush=True)
        if dry_run:
            continue
        completed = subprocess.run(gate.argv(python), cwd=root, check=False)
        if completed.returncode:
            print(
                f"[FAIL] {gate.id}: exit code {completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=tuple(TIER_GATES))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", dest="list_only")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    gates = TIER_GATES[args.tier]
    if args.list_only:
        for gate in gates:
            print(f"{gate.id}\t{format_gate(gate, args.python)}")
        return 0
    return run_gates(
        gates,
        python=args.python,
        root=args.root.resolve(),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
