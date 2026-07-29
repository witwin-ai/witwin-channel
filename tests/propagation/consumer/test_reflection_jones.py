# Copyright Xingyu Chen.
# Tests reflection jones.

from __future__ import annotations

import pytest
import torch

from witwin.channel.propagation.consumer import (
    FixedTopologyRequest,
    JonesTransport,
    PropagationRequest,
    evaluate,
    prepare_fixed_topology,
    reevaluate,
)

from tests.propagation.consumer._reflection_world import (
    FREQUENCY_HZ,
    discover,
    endpoints,
    multi_endpoints,
    rotated_endpoints,
    smooth_wall_scene,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_REFLECTION_ROW = 1


def _jones(compiled, sources, sinks, *, ad_mode: str = "none"):
    discovered = discover(compiled, sources, sinks, response="complex3_transport")
    prepared = prepare_fixed_topology(discovered.paths.topology)
    return reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=prepared,
            response="polarimetric_transport",
            ad_mode=ad_mode,
        ),
    )


def _reflection_arguments(compiled, result, sources, sinks):
    """Rebuild the exact native transport arguments of the reflection row."""

    from witwin.channel.materials import face_material_field_bundle

    row = torch.tensor([_REFLECTION_ROW], device="cuda", dtype=torch.int64)
    geometry = result.paths.geometry
    topology = result.paths.topology
    face_id = topology.primitive_sequence.index_select(0, row).to(torch.int64)
    bundle = face_material_field_bundle(compiled, device=torch.device("cuda"))
    return {
        "source": sources.positions_m.expand(1, 3).contiguous(),
        "target": sinks.positions_m.expand(1, 3).contiguous(),
        "positions": geometry.interaction_positions_m.index_select(
            0, row
        ).contiguous(),
        "normals": geometry.interaction_normals.index_select(0, row).contiguous(),
        "power": torch.ones((1,), device="cuda"),
        "material": tuple(
            bundle[name][face_id].contiguous()
            for name in ("eps_r", "sigma_e", "mu_r", "gain", "thickness")
        ),
    }


def _transport(arguments, tx_polarization, rx_polarization):
    from witwin.channel.kernels import fields as field_kernels

    return field_kernels.field_reflection_sequence(
        arguments["source"],
        arguments["target"],
        arguments["positions"],
        arguments["normals"],
        arguments["power"],
        tx_polarization.contiguous(),
        rx_polarization.contiguous(),
        *arguments["material"],
        frequency_hz=FREQUENCY_HZ,
    )


@pytest.mark.parametrize("mixture", [(0.5**0.5, 0.5**0.5), (0.6, -0.8)])
def test_reflection_jones_matches_the_superposition_oracle(mixture) -> None:
    """The decisive test: excite a mixture the composition never evaluated.

 The published operator claims that the native reflection transport is
 linear in the transmit polarization and that its rows and columns are
 indexed sink-then-source. Driving the SAME production transport with a
 mixture of the two source basis vectors and comparing against the same
 mixture of matrix entries falsifies a non-transverse basis and any scale
 error, and - because the world reads the two ends out in two different
 rotated frames - a transposed index convention as well.

 The reference frame matters. With the reference basis aligned to the
 incidence plane the operator is diagonal, and a transposed matrix is then
 numerically identical to the correct one, so the same oracle proves
 nothing. The guard below refuses to let that happen silently.
 """

    compiled = smooth_wall_scene()
    sources, sinks = rotated_endpoints()
    result = _jones(compiled, sources, sinks)
    transport = result.paths.transport
    assert isinstance(transport, JonesTransport)

    arguments = _reflection_arguments(compiled, result, sources, sinks)
    basis = transport.source_basis[_REFLECTION_ROW]
    sink_basis = transport.sink_basis[_REFLECTION_ROW]
    first, second = mixture
    excitation = (first * basis[0] + second * basis[1]).reshape(1, 3)
    matrix = transport.matrix[_REFLECTION_ROW]

    asymmetry = float((matrix - matrix.transpose(0, 1)).abs().max())
    assert asymmetry > float(matrix.abs().max()), (
        "this world publishes a near-symmetric operator, so the oracle below "
        "cannot tell a transposed convention from the correct one"
    )

    for sink_component in (0, 1):
        evaluated = _transport(
            arguments, excitation, sink_basis[sink_component].reshape(1, 3)
        )
        expected = (
            first * matrix[sink_component, 0] + second * matrix[sink_component, 1]
        )
        torch.testing.assert_close(
            evaluated["coefficient"].reshape(()),
            expected,
            rtol=2.0e-5,
            atol=1.0e-9,
        )
        transposed = (
            first * matrix[0, sink_component] + second * matrix[1, sink_component]
        )
        assert not torch.allclose(
            evaluated["coefficient"].reshape(()),
            transposed,
            rtol=1.0e-2,
            atol=1.0e-9,
        ), "the transposed convention is indistinguishable in this world"


def test_composed_los_jones_reproduces_the_fused_native_operator() -> None:
    """A depth-0 row must land on the shipped native LoS Jones values exactly.

 This is what pins the composed transverse bases to the native
 endpoint-basis owner: the composition and the fused operator evaluate the
 identical native expressions, so anything but equality means the composed
 route built a different basis or a different direction.
 """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    composed = _jones(compiled, sources, sinks)
    fused = evaluate(
        compiled,
        PropagationRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            components=frozenset({"los"}),
            max_depth=0,
            response="polarimetric_transport",
            topology_mode="discover",
            ad_mode="none",
        ),
    ).paths.transport

    assert torch.equal(composed.paths.transport.matrix[:1], fused.matrix)
    assert torch.equal(composed.paths.transport.source_basis[:1], fused.source_basis)
    assert torch.equal(composed.paths.transport.sink_basis[:1], fused.sink_basis)


def test_evaluate_composed_jones_ad_route_reproduces_the_fused_primal() -> None:
    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    request = dict(
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=FREQUENCY_HZ,
        components=frozenset({"los"}),
        max_depth=0,
        response="polarimetric_transport",
        topology_mode="discover",
    )
    fused = evaluate(compiled, PropagationRequest(ad_mode="none", **request))
    composed = evaluate(compiled, PropagationRequest(ad_mode="vjp", **request))

    assert torch.equal(
        composed.paths.transport.matrix, fused.paths.transport.matrix
    )
    assert torch.equal(
        composed.paths.transport.source_basis, fused.paths.transport.source_basis
    )
    assert torch.equal(
        composed.paths.transport.sink_basis, fused.paths.transport.sink_basis
    )


def test_jones_bases_are_transverse_to_their_own_leg_and_direction_is_frozen(
) -> None:
    """The invariant the whole operator rests on, tied to its AD consequence.

 A reflection row launches on one direction and arrives on another. If a
 basis stops being transverse to its own leg, the native projection
 silently shortens it and the published operator stops being the operator
 in the published basis - with no exception anywhere. The same invariant is
 what makes the frozen ``direction`` edge exact: the dropped derivative term
 carries a factor ``sink_basis . direction``, which is zero here.
 """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    result = _jones(compiled, sources, sinks, ad_mode="vjp")
    transport = result.paths.transport
    geometry = result.paths.geometry

    launch = torch.nn.functional.normalize(
        geometry.interaction_positions_m[_REFLECTION_ROW, 0]
        - sources.positions_m[0],
        dim=-1,
    )
    arrival = geometry.field_direction[_REFLECTION_ROW]
    for component in (0, 1):
        assert (
            transport.source_basis[_REFLECTION_ROW, component] * launch
        ).sum().abs() < 1.0e-6
        assert (
            transport.sink_basis[_REFLECTION_ROW, component] * arrival
        ).sum().abs() < 1.0e-6
    # The launch and arrival legs really are different planes for this row, so
    # a single shared basis could not have satisfied both assertions above.
    assert float((launch * arrival).sum().abs()) < 0.999
    assert geometry.field_direction.requires_grad is False


def test_launch_direction_excitation_produces_no_field() -> None:
    """An analytic falsifier for the launch direction the basis is built on.

 ``project_to_wedge_plane(d, d)`` is exactly zero, so exciting the frozen
 transport with the launch direction itself must radiate nothing. If the
 basis had been built on the arrival direction, or on the straight
 source-to-sink direction, this would be a plausible non-zero number.
 """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    result = _jones(compiled, sources, sinks)
    arguments = _reflection_arguments(compiled, result, sources, sinks)
    launch = torch.nn.functional.normalize(
        arguments["positions"][:, 0] - arguments["source"], dim=-1
    )

    evaluated = _transport(arguments, launch, launch)
    reference = _transport(
        arguments,
        result.paths.transport.source_basis[_REFLECTION_ROW, 0].reshape(1, 3),
        result.paths.transport.sink_basis[_REFLECTION_ROW, 0].reshape(1, 3),
    )
    scale = float(reference["field_vector"].abs().max())
    assert scale > 0.0
    assert float(evaluated["field_vector"].abs().max()) < 1.0e-6 * scale


def test_single_mirror_jones_separates_the_two_polarizations() -> None:
    """One flat plate in the incidence plane gives a diagonal operator.

 The reference basis is chosen so that ``u`` projects into the incidence
 plane and ``v`` is the out-of-plane axis, which are the TM and TE
 responses. They must not mix, and they must not be the same number: a
 composition that accidentally published one polarization twice, or that
 rotated one basis into the other, fails here.
 """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    matrix = _jones(compiled, sources, sinks).paths.transport.matrix[
        _REFLECTION_ROW
    ]

    diagonal = torch.stack((matrix[0, 0].abs(), matrix[1, 1].abs()))
    cross = torch.stack((matrix[0, 1].abs(), matrix[1, 0].abs()))
    assert float(cross.max()) < 1.0e-5 * float(diagonal.min())
    assert float(diagonal.min()) > 0.0
    assert abs(float(diagonal[0] / diagonal[1]) - 1.0) > 1.0e-3


def test_reflection_jones_supports_reverse_mode_endpoint_gradients() -> None:
    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    discovered = discover(compiled, sources, sinks)
    prepared = prepare_fixed_topology(discovered.paths.topology)

    sink_positions = torch.tensor(
        [[0.0, 0.5, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True
    )
    ad_sources, ad_sinks = endpoints(sink_positions=sink_positions)
    result = reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=ad_sources,
            sinks=ad_sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=prepared,
            response="polarimetric_transport",
            ad_mode="vjp",
        ),
    )
    matrix = result.paths.transport.matrix
    assert torch.isfinite(matrix).all()
    matrix.real.sum().backward()
    assert sink_positions.grad is not None
    assert torch.isfinite(sink_positions.grad).all()
    assert int(torch.count_nonzero(sink_positions.grad)) > 0


def test_polarization_basis_gradients_are_rejected_before_any_native_work(
) -> None:
    """The frozen-basis contract, enforced rather than documented.

 The composition hands both bases to native companions that reject
 gradients on the transmit and receive polarization, so a differentiable
 basis can only ever produce a silently incomplete derivative.
 """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    discovered = discover(compiled, sources, sinks)
    prepared = prepare_fixed_topology(discovered.paths.topology)
    basis = torch.tensor(
        [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        device="cuda",
        requires_grad=True,
    )
    # ``type(sinks)`` rather than a fresh import: a sibling test reloads the
    # consumer package to prove cold-import neutrality, so a re-imported class
    # object is not always the one this batch was built from.
    differentiable = type(sinks)(
        stable_ids=sinks.stable_ids,
        positions_m=sinks.positions_m,
        polarizations=sinks.polarizations,
        polarization_basis=basis,
    )

    with pytest.raises(NotImplementedError, match="primal-only"):
        reevaluate(
            compiled,
            FixedTopologyRequest(
                sources=sources,
                sinks=differentiable,
                reference_frequency_hz=FREQUENCY_HZ,
                topology=prepared,
                response="polarimetric_transport",
                ad_mode="vjp",
            ),
        )


def test_reflection_jones_reverse_mode_matches_central_differences() -> None:
    """Finite, non-zero, and correct are three different claims.

 The direct source-and-target dependence of the native transport keeps a
 reflection gradient finite and non-zero even if the moving stationary point
 contributed nothing, so only a numeric reference can tell whether the
 specular-motion term is really in the answer. Central differences of the
 same reevaluation are that reference.
 """

    compiled = smooth_wall_scene()
    sources, sinks = rotated_endpoints()
    discovered = discover(compiled, sources, sinks)
    prepared = prepare_fixed_topology(discovered.paths.topology)

    def matrix_at(position: torch.Tensor) -> torch.Tensor:
        moved = rotated_endpoints(sink_positions=position)[1]
        return reevaluate(
            compiled,
            FixedTopologyRequest(
                sources=sources,
                sinks=moved,
                reference_frequency_hz=FREQUENCY_HZ,
                topology=prepared,
                response="polarimetric_transport",
                ad_mode="vjp",
            ),
        ).paths.transport.matrix[_REFLECTION_ROW]

    def objective(position: torch.Tensor) -> torch.Tensor:
        matrix = matrix_at(position)
        return matrix.real.sum() + matrix.imag.sum()

    base = torch.tensor([[0.0, 0.5, 0.0]], device="cuda", dtype=torch.float32)
    step = 2.0e-3
    reference = torch.zeros((1, 3), device="cuda")
    for axis in range(3):
        offset = torch.zeros((1, 3), device="cuda")
        offset[0, axis] = step
        reference[0, axis] = (
            objective(base + offset) - objective(base - offset)
        ).detach() / (2 * step)

    differentiable = base.clone().requires_grad_()
    objective(differentiable).backward()

    scale = float(reference.abs().max())
    assert scale > 0.0
    assert float((differentiable.grad - reference).abs().max()) < 1.0e-2 * scale

    # The specular point itself has to be on the graph; a detached hit table
    # would leave the gradient above plausible but the motion term missing.
    hit = matrix_at(differentiable)
    assert hit.requires_grad


def test_reflection_jones_supports_forward_mode_under_the_declared_convention(
) -> None:
    """``jvp`` is declared for polarimetric_transport and nothing else uses it."""

    import torch.autograd.forward_ad as forward_ad

    compiled = smooth_wall_scene()
    sources, sinks = rotated_endpoints()
    prepared = prepare_fixed_topology(
        discover(compiled, sources, sinks).paths.topology
    )

    def matrix_at(y: float) -> torch.Tensor:
        moved = rotated_endpoints(
            sink_positions=torch.tensor([[0.0, y, 0.0]], device="cuda")
        )[1]
        return reevaluate(
            compiled,
            FixedTopologyRequest(
                sources=sources,
                sinks=moved,
                reference_frequency_hz=FREQUENCY_HZ,
                topology=prepared,
                response="polarimetric_transport",
                ad_mode="none",
            ),
        ).paths.transport.matrix

    step = 1.0e-3
    reference = (matrix_at(0.5 + step) - matrix_at(0.5 - step)) / (2 * step)

    primal = torch.tensor(
        [[0.0, 0.5, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True
    )
    tangent = torch.tensor([[0.0, 1.0, 0.0]], device="cuda")
    with forward_ad.dual_level():
        dual_sinks = rotated_endpoints(
            sink_positions=forward_ad.make_dual(primal, tangent)
        )[1]
        result = reevaluate(
            compiled,
            FixedTopologyRequest(
                sources=sources,
                sinks=dual_sinks,
                reference_frequency_hz=FREQUENCY_HZ,
                topology=prepared,
                response="polarimetric_transport",
                ad_mode="jvp",
            ),
        )
        matrix_tangent = forward_ad.unpack_dual(
            result.paths.transport.matrix
        ).tangent
        matrix_tangent = None if matrix_tangent is None else matrix_tangent.clone()

    assert matrix_tangent is not None
    scale = float(reference.abs().max())
    assert scale > 0.0
    assert float((matrix_tangent - reference).abs().max()) < 1.0e-2 * scale


def test_multi_pair_jones_reproduces_the_single_pair_operator_row_for_row(
) -> None:
    """Two sources and three sinks, so the diagonal pair-index trick is used.

 ``transverse_basis`` hands the native endpoint-basis owner per-row tables
 and the index ``k * (N + 1)``. With one row that index is 0 and the trick
 is untested; with twelve rows a mis-strided read lands on another row's
 endpoints. The reference is the same rows evaluated one pair at a time.
 """

    compiled = smooth_wall_scene()
    sources, sinks = multi_endpoints()
    batched = _jones(compiled, sources, sinks)
    topology = batched.paths.topology
    matrix = batched.paths.transport.matrix

    assert batched.paths.path_count == 12
    for row in range(batched.paths.path_count):
        source_index = int(topology.source_index[row])
        sink_index = int(topology.sink_index[row])
        single_sources, single_sinks = rotated_endpoints(
            source_positions=sources.positions_m[source_index : source_index + 1],
            sink_positions=sinks.positions_m[sink_index : sink_index + 1],
        )
        single = _jones(compiled, single_sources, single_sinks)
        depth = int(topology.depth[row])
        expected = single.paths.transport.matrix[depth]
        torch.testing.assert_close(
            matrix[row], expected, rtol=1.0e-5, atol=1.0e-9
        )


@pytest.mark.parametrize(
    "field", ["powers_w", "polarizations", "polarization_basis"]
)
def test_every_declared_frozen_polarimetric_input_is_enforced_on_evaluate(
    field,
) -> None:
    """A declared frozen input that nobody checks is not a contract.

 ``polarimetric_frozen_ad_inputs`` names tx_power, the endpoint
 polarizations, and both bases. The composed operator is excited by the two
 basis vectors, so an endpoint polarization never reaches the transport and
 tx_power reaches a companion that does not differentiate it - a request
 carrying either could only ever get an empty gradient back. The discovery
 entry point must refuse them, not just the fixed-topology one.
 """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    values = {
        "stable_ids": sources.stable_ids,
        "positions_m": sources.positions_m,
        "polarizations": sources.polarizations,
        "polarization_basis": sources.polarization_basis,
        "powers_w": sources.powers_w,
    }
    values[field] = values[field].clone().requires_grad_()

    with pytest.raises(NotImplementedError, match="primal-only"):
        evaluate(
            compiled,
            PropagationRequest(
                sources=type(sources)(**values),
                sinks=sinks,
                reference_frequency_hz=FREQUENCY_HZ,
                components=frozenset({"los"}),
                max_depth=0,
                response="polarimetric_transport",
                topology_mode="discover",
                ad_mode="vjp",
            ),
        )


def test_a_dead_reflection_row_publishes_an_inert_operator() -> None:
    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    discovered = discover(compiled, sources, sinks)
    prepared = prepare_fixed_topology(discovered.paths.topology)
    moved_sinks = endpoints(
        sink_positions=torch.tensor([[0.0, 30.0, 0.0]], device="cuda")
    )[1]

    result = reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=moved_sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=prepared,
            response="polarimetric_transport",
            ad_mode="none",
        ),
    )

    assert result.row_valid.tolist() == [True, False]
    transport = result.paths.transport
    assert torch.equal(
        transport.matrix[_REFLECTION_ROW],
        torch.zeros((2, 2), device="cuda", dtype=torch.complex64),
    )
    assert torch.equal(
        transport.source_basis[_REFLECTION_ROW],
        torch.zeros((2, 3), device="cuda"),
    )
    assert torch.equal(
        transport.sink_basis[_REFLECTION_ROW], torch.zeros((2, 3), device="cuda")
    )
    # The living row still carries a real operator.
    assert float(transport.matrix[0].abs().max()) > 0.0