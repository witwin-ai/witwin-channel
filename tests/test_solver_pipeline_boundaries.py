from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"
SOLVER_ROOTS = (
    PACKAGE_ROOT / "path",
    PACKAGE_ROOT / "deterministic",
    PACKAGE_ROOT / "montecarlo" / "basic",
    PACKAGE_ROOT / "montecarlo" / "bdpt",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    return modules


def test_solver_facades_remain_thin_and_have_pipeline_owners() -> None:
    for root in SOLVER_ROOTS:
        facade = root / "solver.py"
        pipeline = root / "pipeline.py"

        assert pipeline.is_file()
        assert len(facade.read_text(encoding="utf-8").splitlines()) <= 200


def test_monte_carlo_pipelines_do_not_depend_on_enumerated_engine() -> None:
    for root in SOLVER_ROOTS[2:]:
        imports = _imports(root / "pipeline.py")
        assert not any(
            module.startswith("witwin.channel.propagation.enumerated")
            for module in imports
        )


def test_solver_modules_do_not_import_another_solver() -> None:
    solver_prefixes = (
        "witwin.channel.path",
        "witwin.channel.deterministic",
        "witwin.channel.montecarlo.basic",
        "witwin.channel.montecarlo.bdpt",
    )
    for owner, root in zip(solver_prefixes, SOLVER_ROOTS, strict=True):
        for path in (root / "solver.py", root / "pipeline.py"):
            imports = _imports(path)
            assert not any(
                module.startswith(prefix)
                for prefix in solver_prefixes
                if prefix != owner
                for module in imports
            )
