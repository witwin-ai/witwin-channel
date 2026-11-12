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


def test_radiomap_import_does_not_import_forbidden_solver_stacks():
    # The fresh import is checked in isolation and the original module objects
    # are restored afterwards, so later tests keep one Config class identity.
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "witwin.channel.radiomap"
        or name.startswith("witwin.channel.radiomap.")
        or name in FORBIDDEN_MODULES
        or name in FORBIDDEN_INTERNAL_MODULES
    }
    for module_name in saved:
        sys.modules.pop(module_name, None)
    try:
        importlib.import_module("witwin.channel.radiomap")

        for module_name in FORBIDDEN_MODULES:
            assert module_name not in sys.modules
        for module_name in FORBIDDEN_INTERNAL_MODULES:
            assert module_name not in sys.modules
    finally:
        for module_name in list(sys.modules):
            if module_name == "witwin.channel.radiomap" or module_name.startswith(
                "witwin.channel.radiomap."
            ):
                sys.modules.pop(module_name, None)
        sys.modules.update(saved)
        package = sys.modules.get("witwin.channel")
        if package is not None and "witwin.channel.radiomap" in saved:
            package.radiomap = saved["witwin.channel.radiomap"]  # type: ignore[attr-defined]


def test_solver_facade_delegates_to_pipeline_owner():
    from witwin.channel import radiomap
    from witwin.channel.radiomap import _coherent

    # radiomap.solve only dispatches on the config type; the Coherent estimator's own solve
    # compiles the scene and hands off to its pipeline owner, which does the work.
    assert radiomap.solve.__module__ == radiomap.__name__
    assert _coherent.solve.__module__ == _coherent.__name__
    assert _coherent._solve_pipeline.__module__ == _coherent.__name__
    assert _coherent._metadata.__module__ == _coherent.__name__
    assert _coherent.__all__ == [
        "Coherent", "CoherentRadioMap", "CoherentRadioMapMetadata", "PathTable", "solve",
    ]
