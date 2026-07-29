# Copyright Xingyu Chen.
# Tests solver pipeline boundaries.

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "witwin" / "channel"
SOLVER_ROOTS = (
    PACKAGE_ROOT / "path",
    PACKAGE_ROOT / "deterministic",
    PACKAGE_ROOT / "montecarlo" / "basic",
    PACKAGE_ROOT / "montecarlo" / "bdpt",
)


def _solver_sources(root: Path) -> tuple[Path, ...]:
    """Every source file that owns one solver.

    A solver is either a collapsed single module beside its former package
    directory, or a package whose facade delegates to a pipeline owner.
    """

    module = root.with_suffix(".py")
    if module.is_file():
        return (module,)
    return (root / "solver.py", root / "pipeline.py")


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
        sources = _solver_sources(root)
        if len(sources) == 1:
            # A collapsed solver owns its whole pipeline in one module, so
            # there is no second file for a facade to stay thin against.
            assert sources[0].is_file()
            assert not root.is_dir()
            continue
        facade, pipeline = sources

        assert pipeline.is_file()
        assert len(facade.read_text(encoding="utf-8").splitlines()) <= 200


def test_monte_carlo_pipelines_do_not_depend_on_enumerated_engine() -> None:
    for root in SOLVER_ROOTS[2:]:
        for source in _solver_sources(root):
            if source.name == "solver.py":
                continue
            imports = _imports(source)
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
        for path in _solver_sources(root):
            imports = _imports(path)
            assert not any(
                module.startswith(prefix)
                for prefix in solver_prefixes
                if prefix != owner
                for module in imports
            )