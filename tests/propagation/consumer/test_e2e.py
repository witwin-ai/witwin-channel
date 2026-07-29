# Copyright Xingyu Chen.
# Tests e2e.

from __future__ import annotations

import pytest
import torch

from witwin.channel.propagation.consumer import (
    Complex3Transport,
    EndpointBatch,
    FixedTopologyRequest,
    JonesTransport,
    PropagationRequest,
    ScalarTransport,
    evaluate,
    reevaluate,
)
from witwin.channel.scene import compile
from witwin.core import Scene


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_FREQUENCY_HZ = 1.0e9


def _endpoints(
    source_positions: torch.Tensor | None = None,
) -> tuple[EndpointBatch, EndpointBatch]:
    basis = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    source = EndpointBatch(
        stable_ids=torch.tensor([101], device="cuda", dtype=torch.int64),
        positions_m=(
            source_positions
            if source_positions is not None
            else torch.tensor([[0.0, 0.0, 0.0]], device="cuda")
        ),
        polarizations=torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        polarization_basis=basis,
        powers_w=torch.ones((1,), device="cuda"),
    )
    sink = EndpointBatch(
        stable_ids=torch.tensor([707], device="cuda", dtype=torch.int64),
        positions_m=torch.tensor([[0.0, 0.0, 2.0]], device="cuda"),
        polarizations=torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        polarization_basis=basis,
    )
    return source, sink


def _discover(compiled, response: str):
    sources, sinks = _endpoints()
    return evaluate(
        compiled,
        PropagationRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=_FREQUENCY_HZ,
            components=frozenset({"los"}),
            max_depth=0,
            response=response,
            topology_mode="discover",
            ad_mode="none",
        ),
    )


def test_consumer_scalar_complex3_and_jones_end_to_end() -> None:
    compiled = compile(Scene(), reference_frequency_hz=_FREQUENCY_HZ)

    scalar = _discover(compiled, "scalar_transport")
    complex3 = _discover(compiled, "complex3_transport")
    jones = _discover(compiled, "polarimetric_transport")

    assert isinstance(scalar.paths.transport, ScalarTransport)
    assert isinstance(complex3.paths.transport, Complex3Transport)
    assert isinstance(jones.paths.transport, JonesTransport)
    for result in (scalar, complex3, jones):
        assert result.paths.path_count == 1
        torch.testing.assert_close(
            result.paths.pair_offsets,
            torch.tensor([0, 1], device="cuda", dtype=torch.int64),
        )
        assert result.diagnostics.compact_count_d2h_copies == 0
        assert result.diagnostics.compact_sync_count == 0
    coefficient = scalar.paths.transport.coefficient
    torch.testing.assert_close(
        complex3.paths.transport.field[:, 0], coefficient
    )
    torch.testing.assert_close(
        complex3.paths.transport.field[:, 1:],
        torch.zeros((1, 2), device="cuda", dtype=torch.complex64),
    )
    torch.testing.assert_close(
        jones.paths.transport.matrix,
        torch.diag_embed(coefficient[:, None].expand(-1, 2)),
    )


def test_consumer_general_owner_publishes_exact_pair_layout_once() -> None:
    compiled = compile(Scene(), reference_frequency_hz=_FREQUENCY_HZ)
    sources = EndpointBatch(
        stable_ids=torch.tensor([101, 102], device="cuda", dtype=torch.int64),
        positions_m=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda"
        ),
        polarizations=torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda"
        ),
        powers_w=torch.ones((2,), device="cuda"),
    )
    sinks = EndpointBatch(
        stable_ids=torch.tensor([701, 702], device="cuda", dtype=torch.int64),
        positions_m=torch.tensor(
            [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], device="cuda"
        ),
        polarizations=torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="cuda"
        ),
    )
    result = evaluate(
        compiled,
        PropagationRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=_FREQUENCY_HZ,
            components=frozenset({"los", "reflection"}),
            max_depth=1,
            response="scalar_transport",
            topology_mode="discover",
            ad_mode="none",
        ),
    )

    assert result.paths.path_count == 4
    torch.testing.assert_close(
        result.paths.pair_index,
        torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int64),
    )
    torch.testing.assert_close(
        result.paths.pair_offsets,
        torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.int64),
    )
    torch.testing.assert_close(
        result.paths.topology.source_id,
        torch.tensor([101, 102, 101, 102], device="cuda", dtype=torch.int64),
    )
    torch.testing.assert_close(
        result.paths.topology.sink_id,
        torch.tensor([701, 701, 702, 702], device="cuda", dtype=torch.int64),
    )
    assert result.diagnostics.compact_count_d2h_copies == 1
    assert result.diagnostics.compact_count_d2h_bytes == 8
    assert result.diagnostics.compact_sync_count == 1


def test_fixed_los_reevaluation_vjp_and_jvp_end_to_end() -> None:
    compiled = compile(Scene(), reference_frequency_hz=_FREQUENCY_HZ)
    discovered = _discover(compiled, "scalar_transport")
    topology = discovered.paths.topology

    source_positions = torch.tensor(
        [[0.1, 0.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    sources, sinks = _endpoints(source_positions)
    fixed = reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=_FREQUENCY_HZ,
            topology=topology,
            response="scalar_transport",
            ad_mode="vjp",
        ),
    )
    assert fixed.paths.topology is topology
    assert fixed.diagnostics.discovery_launch_count == 0
    assert fixed.diagnostics.validation_d2h_copies == 1
    assert fixed.diagnostics.validation_d2h_bytes == 4
    assert fixed.diagnostics.validation_sync_count == 1
    fixed.paths.transport.coefficient.real.sum().backward()
    assert source_positions.grad is not None
    assert torch.isfinite(source_positions.grad).all()
    assert torch.count_nonzero(source_positions.grad).item() > 0

    primal_positions = source_positions.detach()
    tangent_positions = torch.tensor(
        [[0.0, 0.0, 0.25]], device="cuda", dtype=torch.float32
    )
    with torch.autograd.forward_ad.dual_level():
        dual_positions = torch.autograd.forward_ad.make_dual(
            primal_positions, tangent_positions
        )
        dual_sources, dual_sinks = _endpoints(dual_positions)
        dual_result = reevaluate(
            compiled,
            FixedTopologyRequest(
                sources=dual_sources,
                sinks=dual_sinks,
                reference_frequency_hz=_FREQUENCY_HZ,
                topology=topology,
                response="scalar_transport",
                ad_mode="jvp",
            ),
        )
        primal, tangent = torch.autograd.forward_ad.unpack_dual(
            dual_result.paths.transport.coefficient
        )
    assert torch.isfinite(primal).all()
    assert tangent is not None
    assert torch.isfinite(tangent).all()
    assert torch.count_nonzero(tangent).item() > 0