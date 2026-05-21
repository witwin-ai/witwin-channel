"""Shared pytest configuration for the channel test suite."""

from __future__ import annotations
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

import pytest
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
CORE_ROOT = REPO_ROOT.parent / "core"
SUPPORT_DIR = TESTS_DIR / "support"

for root in (CORE_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_LOCAL_PACKAGE_ROOTS = {
    "witwin.core": CORE_ROOT / "witwin" / "core",
}


class _LocalWitwinFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for prefix, root in _LOCAL_PACKAGE_ROOTS.items():
            if fullname == prefix:
                init_py = root / "__init__.py"
                if init_py.exists():
                    return importlib.util.spec_from_file_location(
                        fullname,
                        init_py,
                        submodule_search_locations=[str(root)],
                    )
            elif fullname.startswith(prefix + "."):
                relative_parts = fullname.split(".")[len(prefix.split(".")):]
                module_root = root.joinpath(*relative_parts)
                init_py = module_root / "__init__.py"
                if init_py.exists():
                    return importlib.util.spec_from_file_location(
                        fullname,
                        init_py,
                        submodule_search_locations=[str(module_root)],
                    )
                module_py = module_root.with_suffix(".py")
                if module_py.exists():
                    return importlib.util.spec_from_file_location(fullname, module_py)
        return None


for module_name in list(sys.modules):
    if module_name == "witwin.core" or module_name.startswith("witwin.core."):
        sys.modules.pop(module_name, None)

if not any(isinstance(finder, _LocalWitwinFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _LocalWitwinFinder())


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def pytest_addoption(parser):
    gpu_default_on = _cuda_available()
    for option, help_text, default in (
        ("--gpu", "Run GPU-only channel tests (default: on when CUDA is available).", gpu_default_on),
        ("--acceptance", "Run acceptance-only channel validation tests.", False),
        ("--run-optimize", "Run the opt-in optimization visual test.", False),
    ):
        try:
            parser.addoption(option, action="store_true", default=default, help=help_text)
        except ValueError:
            pass


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires a CUDA-capable environment")
    config.addinivalue_line("markers", "acceptance: end-to-end acceptance or validation coverage")
    config.addinivalue_line("markers", "validation: solver/reference validation coverage")
    config.addinivalue_line("markers", "optimize: long-running optimization coverage, skipped unless --run-optimize is set")


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))

    try:
        path.relative_to(SUPPORT_DIR)
        return True
    except ValueError:
        pass

    if not config.getoption("--acceptance"):
        return False
    if path.suffix != ".py":
        return False
    if not (path.name.startswith("test_") or path.name.endswith("_test.py")):
        return False

    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False

    return "acceptance" not in source


def pytest_collection_modifyitems(config, items):
    run_gpu = config.getoption("--gpu") and _cuda_available()
    run_acceptance = config.getoption("--acceptance")
    run_optimize = config.getoption("--run-optimize")

    skip_gpu = pytest.mark.skip(reason="needs --gpu flag and CUDA device")
    skip_acceptance = pytest.mark.skip(reason="needs --acceptance flag")
    skip_optimize = pytest.mark.skip(reason="needs --run-optimize flag")
    selected_items = []
    deselected_items = []

    for item in items:
        if run_acceptance and "acceptance" not in item.keywords:
            deselected_items.append(item)
            continue
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)
        if "acceptance" in item.keywords and not run_acceptance:
            item.add_marker(skip_acceptance)
        if "optimize" in item.keywords and not run_optimize:
            item.add_marker(skip_optimize)
        selected_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = selected_items


