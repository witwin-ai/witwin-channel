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
    for module_name in list(sys.modules):
        if module_name == "witwin.channel.deterministic" or module_name.startswith(
            "witwin.channel.deterministic."
        ):
            sys.modules.pop(module_name, None)
    for module_name in FORBIDDEN_MODULES:
        sys.modules.pop(module_name, None)
    for module_name in FORBIDDEN_INTERNAL_MODULES:
        sys.modules.pop(module_name, None)

    importlib.import_module("witwin.channel.deterministic")

    for module_name in FORBIDDEN_MODULES:
        assert module_name not in sys.modules
    for module_name in FORBIDDEN_INTERNAL_MODULES:
        assert module_name not in sys.modules


def test_solver_facade_delegates_to_pipeline_owner():
    from witwin.channel.deterministic import pipeline, solver

    assert solver._metadata is pipeline._metadata
    assert solver.solve.__module__ == solver.__name__
    assert pipeline.solve.__module__ == pipeline.__name__
