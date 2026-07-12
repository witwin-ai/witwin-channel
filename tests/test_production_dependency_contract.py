from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ci" / "check_production_dependencies.py"
SPEC = importlib.util.spec_from_file_location("check_production_dependencies", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


def test_production_sources_do_not_import_legacy_channel_stacks():
    assert contract.scan_roots([ROOT]) == []


def test_contract_detects_direct_and_from_imports():
    violations = contract.scan_source(
        "import drjit\nfrom witwin import channel\nfrom sionna.rt import Scene\n"
    )

    assert [violation.module for violation in violations] == [
        "drjit",
        "witwin.channel",
        "sionna.rt",
    ]


def test_relative_native_raydn_module_is_not_python_raydn_dependency():
    assert contract.scan_source("from .raydn import RayDNScene\n") == []


def test_consumer_scan_allows_independent_radar_stack_but_rejects_old_channel():
    assert contract.scan_source(
        "import drjit\n", forbidden_modules=contract.LEGACY_CHANNEL_MODULES
    ) == []

    violations = contract.scan_source(
        "from witwin import channel\n",
        forbidden_modules=contract.LEGACY_CHANNEL_MODULES,
    )
    assert [violation.module for violation in violations] == ["witwin.channel"]
