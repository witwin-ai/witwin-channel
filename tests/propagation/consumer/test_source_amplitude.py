"""ADR-039: what ``sources.powers_w`` does to every published surface.

Before ADR-039 the consumer required ``powers_w`` and then published a value
that could not depend on it. These tests pin the convention on both sides of
the boundary: scalar and complex3 transport scale by ``sqrt(powers_w)`` per
row and per source, while the Jones operator, the solver results, and the
internal ``PathFields`` contract keep their own conventions. A change that
applied the amplitude twice would fail here as loudly as one that dropped it.
"""

from __future__ import annotations

import pytest
import torch

from witwin.channel.propagation.consumer import (
    EndpointBatch,
    FixedTopologyRequest,
    PropagationRequest,
    evaluate,
    prepare_fixed_topology,
    reevaluate,
)
from witwin.channel.scene import compile
from witwin.core import Scene

from . import _reflection_world as world


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_FREQUENCY_HZ = 1.0e9
_POWERS_W = (4.0, 9.0)


def _endpoints(powers: tuple[float, ...]) -> tuple[EndpointBatch, EndpointBatch]:
    """One source per declared power and a single shared sink."""

    count = len(powers)
    basis = (
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device="cuda")
        .expand(count, 2, 3)
        .contiguous()
    )
    sources = EndpointBatch(
        stable_ids=torch.arange(101, 101 + count, device="cuda", dtype=torch.int64),
        positions_m=torch.stack(
            [
                torch.tensor([0.0, 0.3 * index, 0.0], device="cuda")
                for index in range(count)
            ]
        ).contiguous(),
        polarizations=torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
        .expand(count, 3)
        .contiguous(),
        polarization_basis=basis,
        powers_w=torch.tensor(powers, device="cuda", dtype=torch.float32),
    )
    sinks = EndpointBatch(
        stable_ids=torch.tensor([707], device="cuda", dtype=torch.int64),
        positions_m=torch.tensor([[0.0, 0.0, 2.0]], device="cuda"),
        polarizations=torch.tensor([[1.0, 0.0, 0.0]], device="cuda"),
        polarization_basis=basis[:1].contiguous(),
    )
    return sources, sinks


def _request(sources, sinks, response: str, ad_mode: str = "none"):
    return PropagationRequest(
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=_FREQUENCY_HZ,
        components=frozenset({"los"}),
        max_depth=0,
        response=response,
        topology_mode="discover",
        ad_mode=ad_mode,
    )


def _unit_powers(sources: EndpointBatch) -> EndpointBatch:
    return EndpointBatch(
        stable_ids=sources.stable_ids,
        positions_m=sources.positions_m,
        polarizations=sources.polarizations,
        polarization_basis=sources.polarization_basis,
        powers_w=torch.ones_like(sources.powers_w),
    )


def _row_amplitude(result, sources: EndpointBatch) -> torch.Tensor:
    """Expected per-row amplitude from each row's own transmitting source."""

    row_source = result.paths.topology.source_index.to(dtype=torch.int64)
    return sources.powers_w.index_select(0, row_source).sqrt()


def test_discovery_scalar_transport_scales_by_each_row_own_source() -> None:
    compiled = compile(Scene(), reference_frequency_hz=_FREQUENCY_HZ)
    sources, sinks = _endpoints(_POWERS_W)

    excited = evaluate(compiled, _request(sources, sinks, "scalar_transport"))
    unit = evaluate(
        compiled, _request(_unit_powers(sources), sinks, "scalar_transport")
    )

    assert excited.paths.path_count == len(_POWERS_W)
    amplitude = _row_amplitude(excited, sources)
    assert not torch.allclose(amplitude, torch.ones_like(amplitude))
    torch.testing.assert_close(
        excited.paths.transport.coefficient,
        unit.paths.transport.coefficient * amplitude,
    )


def test_discovery_complex3_transport_scales_by_each_row_own_source() -> None:
    compiled = compile(Scene(), reference_frequency_hz=_FREQUENCY_HZ)
    sources, sinks = _endpoints(_POWERS_W)

    excited = evaluate(compiled, _request(sources, sinks, "complex3_transport"))
    unit = evaluate(
        compiled, _request(_unit_powers(sources), sinks, "complex3_transport")
    )

    amplitude = _row_amplitude(excited, sources)
    torch.testing.assert_close(
        excited.paths.transport.field,
        unit.paths.transport.field * amplitude[:, None],
    )


def test_discovery_complex3_projects_onto_the_scalar_coefficient() -> None:
    """The two excited responses describe one field, not two conventions."""

    compiled = compile(Scene(), reference_frequency_hz=_FREQUENCY_HZ)
    sources, sinks = _endpoints(_POWERS_W)

    scalar = evaluate(compiled, _request(sources, sinks, "scalar_transport"))
    complex3 = evaluate(compiled, _request(sources, sinks, "complex3_transport"))

    # The endpoint polarizations are +x and the paths run along +z, so the
    # receiver projection is the x component.
    torch.testing.assert_close(
        complex3.paths.transport.field[:, 0],
        scalar.paths.transport.coefficient,
    )


def test_polarimetric_transport_stays_excitation_free() -> None:
    """The Jones operator is a basis map, so it must not carry amplitude."""

    compiled = compile(Scene(), reference_frequency_hz=_FREQUENCY_HZ)
    sources, sinks = _endpoints(_POWERS_W)

    excited = evaluate(compiled, _request(sources, sinks, "polarimetric_transport"))
    unit = evaluate(
        compiled, _request(_unit_powers(sources), sinks, "polarimetric_transport")
    )

    torch.testing.assert_close(
        excited.paths.transport.matrix,
        unit.paths.transport.matrix,
        rtol=0.0,
        atol=0.0,
    )


def _fixed_request(sources, sinks, topology, response: str, ad_mode: str = "none"):
    return FixedTopologyRequest(
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=_FREQUENCY_HZ,
        topology=topology,
        response=response,
        ad_mode=ad_mode,
    )


@pytest.mark.parametrize(
    "response", ["scalar_transport", "complex3_transport"]
)
def test_raw_fixed_los_reevaluation_scales_by_each_row_own_source(
    response: str,
) -> None:
    compiled = compile(Scene(), reference_frequency_hz=_FREQUENCY_HZ)
    sources, sinks = _endpoints(_POWERS_W)
    topology = evaluate(
        compiled, _request(sources, sinks, response)
    ).paths.topology

    excited = reevaluate(
        compiled, _fixed_request(sources, sinks, topology, response)
    )
    unit = reevaluate(
        compiled, _fixed_request(_unit_powers(sources), sinks, topology, response)
    )

    amplitude = _row_amplitude(excited, sources)
    if response == "scalar_transport":
        torch.testing.assert_close(
            excited.paths.transport.coefficient,
            unit.paths.transport.coefficient * amplitude,
        )
    else:
        torch.testing.assert_close(
            excited.paths.transport.field,
            unit.paths.transport.field * amplitude[:, None],
        )


@pytest.mark.parametrize(
    "response", ["scalar_transport", "complex3_transport"]
)
def test_prepared_reflection_reevaluation_scales_by_the_source_amplitude(
    response: str,
) -> None:
    """A reflection row carries the amplitude through the whole bounce chain."""

    compiled = world.smooth_wall_scene()
    sources, sinks = world.endpoints()
    powered = EndpointBatch(
        stable_ids=sources.stable_ids,
        positions_m=sources.positions_m,
        polarizations=sources.polarizations,
        polarization_basis=sources.polarization_basis,
        powers_w=torch.full_like(sources.powers_w, 9.0),
    )
    discovered = world.discover(compiled, sources, sinks, response=response)
    prepared = prepare_fixed_topology(discovered.paths.topology)
    assert any(bucket.component == "reflection" for bucket in prepared.buckets)

    excited = reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=powered,
            sinks=sinks,
            reference_frequency_hz=world.FREQUENCY_HZ,
            topology=prepared,
            response=response,
            ad_mode="none",
        ),
    )
    unit = reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=world.FREQUENCY_HZ,
            topology=prepared,
            response=response,
            ad_mode="none",
        ),
    )

    if response == "scalar_transport":
        torch.testing.assert_close(
            excited.paths.transport.coefficient,
            unit.paths.transport.coefficient * 3.0,
        )
    else:
        torch.testing.assert_close(
            excited.paths.transport.field,
            unit.paths.transport.field * 3.0,
        )


@pytest.mark.parametrize(
    "response", ["scalar_transport", "complex3_transport"]
)
def test_prepared_reflection_reevaluation_scales_each_row_by_its_own_source(
    response: str,
) -> None:
    """Distinct powers on a reflection topology: one factor per row.

    The uniform-power reflection case above cannot separate a per-row gather
    from a single global amplitude, and the per-row discovery cases above carry
    no reflection rows. Two sources with different declared powers, reevaluated
    on a prepared topology that holds both components, pin both at once.
    """

    compiled = world.smooth_wall_scene()
    sources, sinks = world.multi_endpoints()
    powered = EndpointBatch(
        stable_ids=sources.stable_ids,
        positions_m=sources.positions_m,
        polarizations=sources.polarizations,
        polarization_basis=sources.polarization_basis,
        powers_w=torch.tensor(_POWERS_W, device="cuda", dtype=torch.float32),
    )
    discovered = world.discover(compiled, sources, sinks, response=response)
    prepared = prepare_fixed_topology(discovered.paths.topology)
    assert any(bucket.component == "reflection" for bucket in prepared.buckets)

    excited = reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=powered,
            sinks=sinks,
            reference_frequency_hz=world.FREQUENCY_HZ,
            topology=prepared,
            response=response,
            ad_mode="none",
        ),
    )
    unit = reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=world.FREQUENCY_HZ,
            topology=prepared,
            response=response,
            ad_mode="none",
        ),
    )

    amplitude = _row_amplitude(excited, powered)
    assert torch.unique(amplitude).numel() == len(_POWERS_W)
    if response == "scalar_transport":
        torch.testing.assert_close(
            excited.paths.transport.coefficient,
            unit.paths.transport.coefficient * amplitude,
        )
    else:
        torch.testing.assert_close(
            excited.paths.transport.field,
            unit.paths.transport.field * amplitude[:, None],
        )


@pytest.mark.parametrize("ad_mode", ["none", "jvp", "vjp"])
@pytest.mark.parametrize(
    "response", ["scalar_transport", "complex3_transport"]
)
def test_every_ad_mode_publishes_the_same_excited_primal(
    ad_mode: str, response: str
) -> None:
    compiled = compile(Scene(), reference_frequency_hz=_FREQUENCY_HZ)
    sources, sinks = _endpoints(_POWERS_W)
    topology = evaluate(
        compiled, _request(sources, sinks, response)
    ).paths.topology
    reference = reevaluate(
        compiled, _fixed_request(sources, sinks, topology, response)
    )

    result = reevaluate(
        compiled, _fixed_request(sources, sinks, topology, response, ad_mode)
    )

    published = (
        result.paths.transport.coefficient
        if response == "scalar_transport"
        else result.paths.transport.field
    )
    expected = (
        reference.paths.transport.coefficient
        if response == "scalar_transport"
        else reference.paths.transport.field
    )
    primal = torch.autograd.forward_ad.unpack_dual(published).primal
    torch.testing.assert_close(primal.detach(), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "response", ["scalar_transport", "complex3_transport"]
)
def test_endpoint_gradients_carry_the_source_amplitude(response: str) -> None:
    """The amplitude is a constant factor, so it scales the whole derivative."""

    compiled = compile(Scene(), reference_frequency_hz=_FREQUENCY_HZ)
    base_sources, sinks = _endpoints((4.0,))
    topology = evaluate(
        compiled, _request(base_sources, sinks, response)
    ).paths.topology

    def _grad(power: float) -> torch.Tensor:
        positions = base_sources.positions_m.detach().clone().requires_grad_(True)
        sources = EndpointBatch(
            stable_ids=base_sources.stable_ids,
            positions_m=positions,
            polarizations=base_sources.polarizations,
            polarization_basis=base_sources.polarization_basis,
            powers_w=torch.full_like(base_sources.powers_w, power),
        )
        result = reevaluate(
            compiled, _fixed_request(sources, sinks, topology, response, "vjp")
        )
        published = (
            result.paths.transport.coefficient
            if response == "scalar_transport"
            else result.paths.transport.field
        )
        published.real.sum().backward()
        assert positions.grad is not None
        return positions.grad

    excited = _grad(4.0)
    unit = _grad(1.0)
    assert torch.count_nonzero(excited).item() > 0
    torch.testing.assert_close(excited, unit * 2.0)


@pytest.mark.parametrize(
    "response", ["scalar_transport", "complex3_transport"]
)
def test_forward_tangents_carry_the_source_amplitude(response: str) -> None:
    compiled = compile(Scene(), reference_frequency_hz=_FREQUENCY_HZ)
    base_sources, sinks = _endpoints((4.0,))
    topology = evaluate(
        compiled, _request(base_sources, sinks, response)
    ).paths.topology
    tangent_positions = torch.tensor([[0.0, 0.0, 0.25]], device="cuda")

    def _tangent(power: float) -> torch.Tensor:
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(
                base_sources.positions_m.detach(), tangent_positions
            )
            sources = EndpointBatch(
                stable_ids=base_sources.stable_ids,
                positions_m=dual,
                polarizations=base_sources.polarizations,
                polarization_basis=base_sources.polarization_basis,
                powers_w=torch.full_like(base_sources.powers_w, power),
            )
            result = reevaluate(
                compiled, _fixed_request(sources, sinks, topology, response, "jvp")
            )
            published = (
                result.paths.transport.coefficient
                if response == "scalar_transport"
                else result.paths.transport.field
            )
            tangent = torch.autograd.forward_ad.unpack_dual(published).tangent
        assert tangent is not None
        return tangent

    excited = _tangent(4.0)
    unit = _tangent(1.0)
    assert torch.count_nonzero(excited).item() > 0
    torch.testing.assert_close(excited, unit * 2.0)


def test_path_fields_and_solver_results_keep_their_own_conventions() -> None:
    """Nothing outside the consumer moved, so no other reader double-counts."""

    from witwin.channel.deterministic import Config as DeterministicConfig
    from witwin.channel.deterministic import solve as deterministic_solve
    from witwin.channel.path import Config as PathConfig
    from witwin.channel.path import solve as path_solve

    from tests.support.core_world import make_receiver, make_transmitter

    def _scene(power: float) -> Scene:
        return Scene(
            structures=[],
            endpoints=[
                make_transmitter(
                    torch.tensor([0.0, 0.0, 0.0]), power_w=power
                ),
                make_receiver(torch.tensor([0.0, 0.0, 2.0])),
            ],
        )

    unit_paths = path_solve(
        _scene(1.0),
        PathConfig(components=frozenset({"los"}), max_depth=0),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    excited_paths = path_solve(
        _scene(4.0),
        PathConfig(components=frozenset({"los"}), max_depth=0),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    # PathResult.a stays unit excitation: it is the shared internal
    # PathFields.coefficient that Deterministic and BDPT also read.
    torch.testing.assert_close(excited_paths.a, unit_paths.a, rtol=0.0, atol=0.0)
    assert (
        excited_paths.metadata["coefficient_semantics"]
        == "unit_excitation_dimensionless_receiver_projection"
    )

    unit_det = deterministic_solve(
        _scene(1.0),
        DeterministicConfig(components=frozenset({"los"}), max_depth=0),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    excited_det = deterministic_solve(
        _scene(4.0),
        DeterministicConfig(components=frozenset({"los"}), max_depth=0),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    # Deterministic already published the excited field before ADR-039.
    torch.testing.assert_close(excited_det.field, unit_det.field * 2.0)
    torch.testing.assert_close(excited_det.path_gain, unit_det.path_gain * 4.0)
