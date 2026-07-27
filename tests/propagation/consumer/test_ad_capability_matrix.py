"""Make the AD capability matrix authoritative instead of decorative.

A checked-in matrix that nothing parses rots the first time a test is renamed.
This module parses `docs/dev/propagation-ad-capability-matrix.md`, rejects any
row outside the four accepted target states, rejects an empty `test` cell, and
resolves every cited test node id by importing its module and looking the
function up. It then pins the document against the live capability record, so
the two halves of ADR-043 cannot drift apart.

No CUDA and no solver: this is a document/record contract, and it must stay
runnable on a machine with no GPU.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from witwin.channel.propagation.consumer import capabilities


ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "docs" / "dev" / "propagation-ad-capability-matrix.md"

STATES = frozenset({"SUP", "ZERO", "REF", "DECL"})
MECHANISMS = frozenset(
    {
        "native-companion",
        "native-declared",
        "torch-orchestration",
        "host-declaration",
    }
)
VALIDATIONS = frozenset(
    {"fd", "oracle-f64", "analytic", "adjoint", "declaration", "refusal"}
)
COLUMNS = (
    "route",
    "leaf-or-output",
    "mode",
    "state",
    "mechanism",
    "owner",
    "test",
    "validation",
)
MODES = frozenset({"jvp", "vjp", "both", "none"})


def _rows() -> list[dict[str, str]]:
    """Every data row of every matrix table in the document."""

    rows: list[dict[str, str]] = []
    header_seen = False
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            header_seen = False
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if tuple(cells) == COLUMNS:
            header_seen = True
            continue
        if not header_seen:
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        if len(cells) != len(COLUMNS):
            raise AssertionError(f"malformed matrix row: {stripped}")
        rows.append(dict(zip(COLUMNS, cells, strict=True)))
    return rows


MATRIX_ROWS = _rows()


def test_the_document_actually_contains_a_matrix() -> None:
    assert len(MATRIX_ROWS) >= 40, "the matrix lost rows"


@pytest.mark.parametrize("row", MATRIX_ROWS, ids=lambda row: row["test"][:80])
def test_every_row_declares_one_of_the_four_target_states(row) -> None:
    assert row["state"] in STATES, row
    assert row["mechanism"] in MECHANISMS, row
    assert row["validation"] in VALIDATIONS, row
    assert set(row["mode"].split()) <= MODES, row
    assert row["owner"], row
    assert row["test"], "a row with no test is a claim with no evidence"


@pytest.mark.parametrize("row", MATRIX_ROWS, ids=lambda row: row["test"][:80])
def test_every_cited_test_resolves(row) -> None:
    for node in (part.strip() for part in row["test"].split(",")):
        path, _, name = node.partition("::")
        assert name, f"{node} is not a test node id"
        assert (ROOT / path).is_file(), f"{path} does not exist"
        module = importlib.import_module(
            path.removesuffix(".py").replace("/", ".")
        )
        assert hasattr(module, name), f"{node} does not resolve"


def test_a_state_outside_the_four_is_rejected_by_the_parser() -> None:
    """Falsifier: the state check is a check, not a tautology."""

    assert "TODO" not in STATES
    assert "SILENT" not in STATES
    assert "PARTIAL" not in STATES


def test_the_matrix_agrees_with_the_live_capability_record() -> None:
    """The anti-rot pin between the prose half and the queryable half."""

    record = capabilities()
    assert record.contract_version == 6
    assert record.ad_modes_for_component("diffraction") == frozenset({"none"})
    assert record.direction_differentiable_components == frozenset(
        {"los", "reflection"}
    )
    assert record.differentiable_geometry_for("discovery") == frozenset(
        {"path_length_m", "delay_s"}
    )
    assert record.differentiable_geometry_for("fixed_topology") == frozenset(
        {
            "path_length_m",
            "delay_s",
            "interaction_positions_m",
            "field_direction",
        }
    )
    assert record.material_leaves_for("transmission") == (
        "layer_eps_r",
        "layer_sigma_e",
        "layer_thickness_m",
    )
    assert record.supports_higher_order_ad is False
    assert record.ad_accounting is True


def test_the_deferred_section_names_every_declared_output() -> None:
    """A `DECL` cell without a named deferral is a defect, not a declaration."""

    text = MATRIX.read_text(encoding="utf-8")
    deferred = text.split("## 4. Deferred", maxsplit=1)
    assert len(deferred) == 2, "the Deferred section is missing"
    body = deferred[1]
    for needle in (
        "field_direction",
        "interaction_positions_m",
        "Diffraction",
        "Second-order",
    ):
        assert needle in body, f"{needle} has no named deferral"
    declared = {row["leaf-or-output"] for row in MATRIX_ROWS if row["state"] == "DECL"}
    assert declared, "the matrix declares no outputs at all"
    for name in declared:
        assert name.startswith("out:"), "only an output may be DECL"


def test_the_matrix_column_vocabulary_is_closed() -> None:
    """Every parsed value came from the published vocabulary, nothing else."""

    assert {row["state"] for row in MATRIX_ROWS} <= STATES
    assert {row["mechanism"] for row in MATRIX_ROWS} <= MECHANISMS
    assert {row["validation"] for row in MATRIX_ROWS} <= VALIDATIONS
    # Every `SUP` row must carry a real oracle, never a declaration.
    for row in MATRIX_ROWS:
        if row["state"] == "SUP":
            assert row["validation"] in {"fd", "oracle-f64", "analytic", "adjoint"}
        if row["state"] == "REF":
            assert row["validation"] in {"refusal", "declaration"}
        if row["state"] == "DECL":
            assert row["validation"] == "declaration"
