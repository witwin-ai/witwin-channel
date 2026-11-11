# Copyright Xingyu Chen.
# Tests the consumer contract's import, ownership and capability boundaries.

from __future__ import annotations

import ast
import re
from pathlib import Path



ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ROOT / "witwin" / "channel"
CONSUMER_MODULE = PACKAGE_ROOT / "propagation" / "consumer.py"


def _all_imports(path: Path) -> set[str]:
    """Every module this file imports, at any nesting level."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return imports


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return imports


def test_public_consumer_import_does_not_eagerly_load_internal_definitions() -> None:
    # The consumer is one module, so the discovery route, the fixed-topology
    # replay, and the time-varying surface all share one top-level import list
    # and the solver-neutrality rule binds every one of them at once. It is
    # prose in CLAUDE.md and unenforced by ci/check_import_graph.py, which
    # carries no consumer rule at all.
    consumer_imports = _top_level_imports(CONSUMER_MODULE)
    assert "witwin.channel.propagation.enumerated" not in consumer_imports
    assert "witwin.channel.propagation.rows" not in consumer_imports
    assert "witwin.channel.propagation.penetration" not in consumer_imports
    # ``witwin.channel.kernels.topology`` reaches ``propagation.penetration``,
    # so the compact finalizer's AD companions stay behind a call-time import
    # for exactly the same reason.
    assert "witwin.channel.kernels.topology" not in consumer_imports
    assert not any(
        module.startswith("witwin.channel.path")
        or module.startswith("witwin.channel.deterministic")
        or module.startswith("witwin.channel.montecarlo")
        or module.startswith("witwin.channel.propagation.enumerated")
        for module in consumer_imports
    )


def test_consumer_governance_is_radar_neutral() -> None:
    # The whole consumer module, not one of its sections: collapsing the
    # package widened this from the orchestration file to every line of the
    # boundary. The acronyms are matched as words because "unique" and
    # "broadcasts" contain "iq" and "adc"; a substring hit on either is
    # noise, and the rule is about radar vocabulary, not letter runs.
    source = CONSUMER_MODULE.read_text(encoding="utf-8").lower()
    for forbidden in ("waveform", "rcs", "iq", "adc", "cfar", "detection"):
        assert re.search(rf"\b{forbidden}\b", source) is None, forbidden


def test_compact_autograd_native_companions_have_one_topology_owner() -> None:
    symbols = {
        "evaluated_paths_compact_finalize_backward",
        "evaluated_paths_compact_finalize_jvp",
    }
    owners: dict[str, list[Path]] = {symbol: [] for symbol in symbols}
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            # A symbol is dispatched by name or declared once as a native op of its facade.
            if (
                isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == "_required_native_op")
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "op")
                )
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in symbols
            ):
                owners[node.args[0].value].append(path)

    facade = PACKAGE_ROOT / "kernels" / "topology.py"
    assert owners == {symbol: [facade] for symbol in symbols}
    # The consumer reaches both companions through that one facade. The
    # import is call-time rather than top level because
    # ``witwin.channel.kernels.topology`` reaches
    # ``witwin.channel.propagation.penetration``, which the cold public
    # import of the consumer must not load.
    assert "witwin.channel.kernels.topology" in _all_imports(CONSUMER_MODULE)


def test_consumer_publishes_frequency_offset_capability_and_laws() -> None:
    """The capability record and convention publish the wideband limits and laws."""

    from witwin.channel.constants import (
        NARROWBAND_FREQUENCY_OFFSET_ERROR_LAW,
        NARROWBAND_FREQUENCY_OFFSET_LAW,
    )
    from witwin.channel.propagation.consumer import (
        PropagationConvention,
        capabilities,
    )

    record = capabilities()
    assert record.supports_wideband_offsets is True
    assert record.wideband_dispersive_materials is False
    assert record.wideband_rough_materials is False

    convention = PropagationConvention()
    assert (
        convention.narrowband_frequency_offset_law
        == NARROWBAND_FREQUENCY_OFFSET_LAW
    )
    assert "delay_s" in convention.narrowband_frequency_offset_law
    assert (
        convention.narrowband_frequency_offset_error_law
        == NARROWBAND_FREQUENCY_OFFSET_ERROR_LAW
    )
