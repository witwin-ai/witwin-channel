# Copyright Xingyu Chen.
# Tests import contract.

import importlib
import sys


FORBIDDEN_MODULES = (
    "drjit",
    "mitsuba",
    "sionna",
    "rayd",
)

FORBIDDEN_INTERNAL_MODULES = (
    "witwin.channel.path.rayd_export",
)


def test_deterministic_import_does_not_import_forbidden_solver_stacks():
    # The fresh import is checked in isolation and the original module objects
    # are restored afterwards, so later tests keep one Config class identity.
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "witwin.channel.deterministic"
        or name.startswith("witwin.channel.deterministic.")
        or name in FORBIDDEN_MODULES
        or name in FORBIDDEN_INTERNAL_MODULES
    }
    for module_name in saved:
        sys.modules.pop(module_name, None)
    try:
        importlib.import_module("witwin.channel.deterministic")

        for module_name in FORBIDDEN_MODULES:
            assert module_name not in sys.modules
        for module_name in FORBIDDEN_INTERNAL_MODULES:
            assert module_name not in sys.modules
    finally:
        for module_name in list(sys.modules):
            if module_name == "witwin.channel.deterministic" or module_name.startswith(
                "witwin.channel.deterministic."
            ):
                sys.modules.pop(module_name, None)
        sys.modules.update(saved)
        package = sys.modules.get("witwin.channel")
        if package is not None and "witwin.channel.deterministic" in saved:
            package.deterministic = saved["witwin.channel.deterministic"]  # type: ignore[attr-defined]


def test_solver_facade_delegates_to_pipeline_owner():
    from witwin.channel import deterministic

    # The solver collapsed into one module, so the facade and the pipeline
    # owner are two functions in it rather than two modules. The public entry
    # still only compiles the scene and hands off; the pipeline owner is the
    # one that does the work, and neither is re-exported from elsewhere.
    assert deterministic.solve.__module__ == deterministic.__name__
    assert deterministic._solve_pipeline.__module__ == deterministic.__name__
    assert deterministic._metadata.__module__ == deterministic.__name__
    assert deterministic.__all__ == ["Config", "PathTable", "Result", "solve"]