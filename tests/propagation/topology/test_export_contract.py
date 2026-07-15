from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tests.propagation.test_topology_batch_adapter import (
    _GEOMETRY_FIELDS,
    _PATH_FIELDS,
    _TOPOLOGY_FIELDS,
    _assert_exact_tensor,
    _batch,
)
from witwin.channel_native.propagation.models import adapters
from witwin.channel_native.propagation.topology import export


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_canonical_export_owns_legacy_compatibility_aliases():
    assert (
        adapters.evaluated_paths_from_topology_batch
        is export.export_evaluated_rows
    )
    assert adapters.PathExecutionStats is export.PathExecutionStats
    assert adapters.TopologyBatchSidecars is export.EvaluatedPathSidecars
    assert export.export_evaluated_rows.__module__ == export.__name__
    assert export.PathExecutionStats.__module__ == export.__name__
    assert export.EvaluatedPathSidecars.__module__ == export.__name__


def test_canonical_export_preserves_all_22_tensor_objects_and_sidecars():
    source = _batch()

    evaluated, sidecars = export.export_evaluated_rows(source)

    for contract, names in (
        (evaluated.topology, _TOPOLOGY_FIELDS),
        (evaluated.geometry, _GEOMETRY_FIELDS),
        (evaluated.fields, _PATH_FIELDS),
    ):
        for name in names:
            _assert_exact_tensor(getattr(contract, name), getattr(source, name))
    assert evaluated.geometry.row_identity is evaluated.topology.row_identity
    assert evaluated.fields.row_identity is evaluated.topology.row_identity
    assert sidecars.execution == export.PathExecutionStats(
        launch_count=11,
        visibility_rejection_count=12,
        selected_edge_count=13,
        candidate_count=14,
        guardrail_count=15,
        ad_companion_launches=16,
        ad_tape_bytes=17,
    )
    _assert_exact_tensor(
        sidecars.diffraction_vector_field,
        source.diffraction_vector_field,
    )


def test_canonical_export_calls_no_tensor_allocation_or_transform_api(monkeypatch):
    source = _batch()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("canonical export must only name existing tensors")

    for name in ("clone", "empty", "zeros", "as_tensor", "stack", "cat"):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ("clone", "contiguous", "to", "detach", "cpu", "cuda"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)

    evaluated, sidecars = export.export_evaluated_rows(source)

    assert evaluated.topology.valid is source.valid
    assert sidecars.diffraction_vector_field is source.diffraction_vector_field


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.propagation.topology import export; "
            "from witwin.channel_native.propagation.models import adapters"
        ),
        (
            "from witwin.channel_native.propagation.models import adapters; "
            "from witwin.channel_native.propagation.topology import export"
        ),
    ),
)
def test_fresh_import_order_preserves_canonical_export_identity(imports: str):
    code = (
        f"{imports}; "
        "assert adapters.evaluated_paths_from_topology_batch is export.export_evaluated_rows; "
        "assert adapters.PathExecutionStats is export.PathExecutionStats; "
        "assert adapters.TopologyBatchSidecars is export.EvaluatedPathSidecars"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(REPOSITORY_ROOT / "src"),
            environment.get("PYTHONPATH"),
        )
        if value
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
