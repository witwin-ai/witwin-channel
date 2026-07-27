from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest
import torch


def test_consumer_cold_import_is_solver_neutral() -> None:
    code = """
import sys
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if "witwin_channel_editable" not in type(finder).__module__
]
import witwin.channel.propagation.consumer as consumer
forbidden = (
    "witwin.channel.path",
    "witwin.channel.deterministic",
    "witwin.channel.montecarlo",
    "witwin.channel.propagation.enumerated",
    "witwin.channel.propagation.models",
)
assert consumer.CONTRACT_VERSION == 6
assert not any(name.startswith(forbidden) for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stderr


def _pop_consumer_modules() -> dict[str, object]:
    popped = {}
    for name in tuple(sys.modules):
        if name.startswith("witwin.channel.propagation.consumer"):
            popped[name] = sys.modules.pop(name)
    return popped


def test_consumer_import_is_solver_neutral() -> None:
    # Restore the original module objects afterwards. A re-import leaves a
    # second copy of every consumer class behind, and an isinstance check in a
    # later test file then compares a value built by one copy against the
    # other.
    original = _pop_consumer_modules()
    try:
        before = set(sys.modules)
        consumer = importlib.import_module("witwin.channel.propagation.consumer")
        loaded = set(sys.modules) - before
        assert consumer.CONTRACT_VERSION == 6
        assert not any(
            name.startswith(
                (
                    "witwin.channel.path",
                    "witwin.channel.deterministic",
                    "witwin.channel.montecarlo",
                    "witwin.channel.propagation.enumerated",
                    "witwin.channel.propagation.models",
                )
            )
            for name in loaded
        )
    finally:
        _pop_consumer_modules()
        sys.modules.update(original)


def test_pair_layout_convention_is_explicit() -> None:
    from witwin.channel.propagation.consumer import PropagationConvention

    assert PropagationConvention().pair_layout == (
        "sink_major_source_minor:"
        "pair_index=sink_index*source_count+source_index"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_endpoint_batch_preserves_input_tensor_objects() -> None:
    from witwin.channel.propagation.consumer import EndpointBatch

    stable_ids = torch.tensor([7, 9], device="cuda", dtype=torch.int64)
    positions = torch.randn((2, 3), device="cuda", dtype=torch.float32)
    polarizations = torch.randn((2, 3), device="cuda", dtype=torch.float32)
    basis = torch.randn((2, 2, 3), device="cuda", dtype=torch.float32)
    powers = torch.ones((2,), device="cuda", dtype=torch.float32)

    endpoints = EndpointBatch(
        stable_ids=stable_ids,
        positions_m=positions,
        polarizations=polarizations,
        polarization_basis=basis,
        powers_w=powers,
    )

    assert endpoints.stable_ids is stable_ids
    assert endpoints.positions_m is positions
    assert endpoints.polarizations is polarizations
    assert endpoints.polarization_basis is basis
    assert endpoints.powers_w is powers


def test_jones_contract_rejects_renamed_two_component_field() -> None:
    from witwin.channel.propagation.consumer import JonesTransport

    with pytest.raises(ValueError, match="rank 3"):
        JonesTransport(
            matrix=torch.empty((3, 2), dtype=torch.complex64),
            source_basis=torch.empty((3, 2, 3), dtype=torch.float32),
            sink_basis=torch.empty((3, 2, 3), dtype=torch.float32),
        )
