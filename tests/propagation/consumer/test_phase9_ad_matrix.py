"""Every Phase-9 consumer AD cell that ADR-043 decided, proved at the boundary.

The survey that opened Phase 9 found 295 consumer cells, 122 of which produced a
genuine derivative and only 28 of which had a named test at the surface that
publishes them. This module closes the cells ADR-043 declares `SUP`, `ZERO`, or
`REF`; the four target states and their required evidence are defined in
`docs/dev/propagation-ad-capability-matrix.md`, and every row of that document
cites a test in this file or beside it.

Finite differences appear here and nowhere else. They are the oracle for a
`SUP` row, exactly as CLAUDE.md allows under `tests/`.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

from witwin.channel.propagation.consumer import (
    EndpointBatch,
    FixedTopologyRequest,
    PropagationRequest,
    TimeVaryingRequest,
    capabilities,
    evaluate,
    evaluate_time_varying,
    prepare_fixed_topology,
    reevaluate,
    replicate_over_slots,
)
from witwin.channel.scene import compile as compile_scene
from witwin.core import PhysicalMaterial, Scene

from tests.propagation.consumer._multi_endpoint_world import (
    FREQUENCY_HZ as SLOT_FREQUENCY_HZ,
    SINK_POSITIONS,
    SOURCE_POSITIONS,
    compiled_world,
    cuda_positions,
    frozen_topology,
    replay,
    sink_batch,
    slot_offsets,
    source_batch,
    stack_over_slots,
)
from tests.propagation.consumer._reflection_world import (
    FREQUENCY_HZ,
    WALL_X_M,
    discover,
    endpoints,
    smooth_wall_scene,
)
from tests.support.core_world import make_receiver, make_transmitter
from tests.support.scenes import (
    transmission_wall_structure,
    wedge_diffraction_scene,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_FD_STEP = 1.0e-3
_SINK = [[0.0, 0.5, 0.0]]


def _cuda(values) -> torch.Tensor:
    return torch.tensor(values, device="cuda", dtype=torch.float32)


def _prepared(compiled, sources, sinks, **kwargs):
    return prepare_fixed_topology(
        discover(compiled, sources, sinks, **kwargs).paths.topology
    )


def _fixed(compiled, prepared, sources, sinks, **kwargs):
    return reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=prepared,
            response=kwargs.pop("response", "scalar_transport"),
            ad_mode=kwargs.pop("ad_mode", "none"),
            **kwargs,
        ),
    )


def _central_difference(base: torch.Tensor, loss) -> torch.Tensor:
    """One central-difference gradient of ``loss`` with respect to ``base``."""

    out = torch.zeros_like(base)
    flat = out.reshape(-1)
    for index in range(base.numel()):
        plus = base.clone().reshape(-1)
        plus[index] += _FD_STEP
        minus = base.clone().reshape(-1)
        minus[index] -= _FD_STEP
        flat[index] = (
            loss(plus.reshape(base.shape)) - loss(minus.reshape(base.shape))
        ) / (2.0 * _FD_STEP)
    return out


# ---------------------------------------------------------------------------
# field_direction: SUP on the fixed-topology route, DECL on discovery.
# ---------------------------------------------------------------------------


def _direction_world():
    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    return compiled, sources, sinks, _prepared(compiled, sources, sinks)


def _direction_loss(compiled, prepared, sources, weight):
    def loss(sink_positions: torch.Tensor, ad_mode: str = "none"):
        moved = endpoints(sink_positions=sink_positions)[1]
        result = _fixed(compiled, prepared, sources, moved, ad_mode=ad_mode)
        return (result.paths.geometry.field_direction * weight).sum(), result

    return loss


def test_the_arrival_direction_carries_a_reverse_gradient_matching_fd() -> None:
    """`field_direction` is live on the fixed-topology route and is correct.

    Before ADR-043 this tensor was ``mark_non_differentiable`` in both branches
    of both field companions, so an AoA or beam-steering loss differentiated
    through it got an exact zero with no warning. The adjoint existed inside the
    kernel and was dropped at the ABI boundary; the seed now reaches it.
    """

    compiled, sources, _, prepared = _direction_world()
    generator = torch.Generator(device="cuda").manual_seed(9)
    weight = torch.randn(
        (prepared.row_count, 3), device="cuda", generator=generator
    )
    loss = _direction_loss(compiled, prepared, sources, weight)

    base = _cuda(_SINK)
    leaf = base.clone().requires_grad_(True)
    value, result = loss(leaf, "vjp")
    assert result.paths.geometry.field_direction.requires_grad is True
    value.backward()

    reference = _central_difference(base, lambda pos: loss(pos)[0].item())
    torch.testing.assert_close(leaf.grad, reference, atol=2.0e-3, rtol=2.0e-3)
    assert float(reference.abs().max()) > 1.0e-2, "the oracle must be nonzero"


def test_the_arrival_direction_tangent_agrees_with_the_reverse_gradient() -> None:
    """jvp and vjp answer the same question about the same frozen rows."""

    compiled, sources, _, prepared = _direction_world()
    generator = torch.Generator(device="cuda").manual_seed(11)
    weight = torch.randn(
        (prepared.row_count, 3), device="cuda", generator=generator
    )
    loss = _direction_loss(compiled, prepared, sources, weight)

    base = _cuda(_SINK)
    direction = _cuda([[0.3, 1.0, -0.4]])
    with forward_ad.dual_level():
        _, result = loss(forward_ad.make_dual(base, direction), "jvp")
        tangent = forward_ad.unpack_dual(
            result.paths.geometry.field_direction
        ).tangent
    assert tangent is not None, "the direction tangent must be published"
    forward = float((tangent * weight).sum())

    leaf = base.clone().requires_grad_(True)
    loss(leaf, "vjp")[0].backward()
    reverse = float((leaf.grad * direction).sum())
    assert forward == pytest.approx(reverse, rel=2.0e-3, abs=2.0e-5)


def test_the_arrival_direction_stays_declared_dead_on_the_discovery_route() -> None:
    """Discovery re-solves the topology, so its direction is a DECL output.

    The capability record says so per route rather than leaving a caller to
    discover it from a zero. This pins the declaration against the behaviour.
    """

    record = capabilities()
    assert "field_direction" not in record.differentiable_geometry_for("discovery")
    assert "field_direction" in record.differentiable_geometry_for("fixed_topology")

    compiled = smooth_wall_scene()
    sources, sinks = endpoints(
        sink_positions=_cuda(_SINK).clone().requires_grad_(True)
    )
    result = discover(compiled, sources, sinks, ad_mode="vjp")
    direction = result.paths.geometry.field_direction
    assert direction.requires_grad is False
    assert direction.grad_fn is None


def test_interaction_positions_are_declared_dead_on_the_discovery_route() -> None:
    """The same declaration, for the other route-dependent geometry tensor."""

    record = capabilities()
    discovery = record.differentiable_geometry_for("discovery")
    assert "interaction_positions_m" not in discovery
    assert "path_length_m" in discovery and "delay_s" in discovery

    compiled = smooth_wall_scene()
    sources, sinks = endpoints(
        sink_positions=_cuda(_SINK).clone().requires_grad_(True)
    )
    result = discover(compiled, sources, sinks, ad_mode="vjp")
    assert result.paths.geometry.interaction_positions_m.requires_grad is False


def test_direction_liveness_is_one_decision_for_the_whole_result() -> None:
    """Never live for some rows and silently dead for others.

    The batch below carries both a line-of-sight and a reflection bucket, and
    both are inside the Channel-owned direction set, so the single decision is
    "live" and it holds for every row of both buckets.
    """

    compiled, sources, _, prepared = _direction_world()
    components = {bucket.component for bucket in prepared.buckets}
    assert components == {"los", "reflection"}
    assert components <= capabilities().direction_differentiable_components

    leaf = _cuda(_SINK).clone().requires_grad_(True)
    moved = endpoints(sink_positions=leaf)[1]
    result = _fixed(compiled, prepared, sources, moved, ad_mode="vjp")
    direction = result.paths.geometry.field_direction
    assert direction.requires_grad is True
    # Every row contributes: a row whose direction had been silently detached
    # would drop out of this sum and leave the gradient short.
    for row in range(direction.shape[0]):
        (grad,) = torch.autograd.grad(
            direction[row].sum(), leaf, retain_graph=True
        )
        assert float(grad.abs().sum()) > 0.0, f"row {row} carries no derivative"


# ---------------------------------------------------------------------------
# Pre-compute refusals: the primal-only inputs, on every route and both modes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
@pytest.mark.parametrize(
    "leaf", ["sources.powers_w", "sources.polarizations", "sinks.polarizations"]
)
def test_a_primal_only_input_is_refused_before_discovery_produces_a_result(
    ad_mode: str, leaf: str
) -> None:
    """REF, on the discovery route, in both modes, before any native work.

    These used to publish a complete ``PropagationEvaluation`` and only raise
    from inside ``backward()`` - a partial result for an unsupported request.
    """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    batch, field = leaf.split(".")
    target = sources if batch == "sources" else sinks
    values = {
        "stable_ids": target.stable_ids,
        "positions_m": target.positions_m,
        "polarizations": target.polarizations,
        "polarization_basis": target.polarization_basis,
    }
    if batch == "sources":
        values["powers_w"] = target.powers_w
    values[field] = values[field].detach().clone().requires_grad_()
    seeded = EndpointBatch(**values)

    produced: list[object] = []
    with pytest.raises(NotImplementedError, match=f"{leaf} is primal-only"):
        produced.append(
            discover(
                compiled,
                seeded if batch == "sources" else sources,
                sinks if batch == "sources" else seeded,
                ad_mode=ad_mode,
            )
        )
    assert produced == [], "no result object may exist for a refused request"


@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_a_primal_only_material_is_refused_before_any_native_work(
    ad_mode: str,
) -> None:
    """`mu_r` is named by the capability record and refused by that record."""

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    compiled.materials.mu_r.requires_grad_(True)
    try:
        with pytest.raises(
            NotImplementedError, match="materials.mu_r is primal-only"
        ):
            discover(compiled, sources, sinks, ad_mode=ad_mode)
    finally:
        compiled.materials.mu_r.requires_grad_(False)


def test_the_capability_record_names_every_pre_compute_refusal() -> None:
    record = capabilities()
    assert set(record.primal_only_ad_inputs) == {
        "materials.layer_mu_r",
        "materials.mu_r",
        "sinks.polarization_basis",
        "sinks.polarizations",
        "sources.polarization_basis",
        "sources.polarizations",
        "sources.powers_w",
    }
    # The polarimetric vocabulary names the same physics and stays published.
    assert "tx_power" in record.polarimetric_frozen_ad_inputs


# ---------------------------------------------------------------------------
# Structural zeros: the material leaf a component does not read.
# ---------------------------------------------------------------------------


def test_a_reflection_scene_reads_the_per_face_material_leaves() -> None:
    """SUP: the leaves the record names for `reflection` really are live."""

    assert capabilities().material_leaves_for("reflection") == (
        "eps_r",
        "sigma_e",
        "thickness_m",
        "gain",
    )
    assert capabilities().material_leaves_for("los") == ()

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    prepared = _prepared(compiled, sources, sinks)
    leaf = compiled.materials.eps_r
    leaf.requires_grad_(True)
    try:
        result = _fixed(compiled, prepared, sources, sinks, ad_mode="vjp")
        result.paths.transport.coefficient.abs().sum().backward()
        assert leaf.grad is not None and float(leaf.grad.abs().sum()) > 0.0
    finally:
        leaf.grad = None
        leaf.requires_grad_(False)


def test_a_layer_leaf_contributes_exactly_zero_to_a_reflection_scene() -> None:
    """ZERO, asserted exactly, and discoverable from the record beforehand."""

    assert "layer_eps_r" not in capabilities().material_leaves_for("reflection")

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    prepared = _prepared(compiled, sources, sinks)
    wrong = compiled.materials.layer_eps_r
    wrong.requires_grad_(True)
    try:
        result = _fixed(compiled, prepared, sources, sinks, ad_mode="vjp")
        loss = result.paths.transport.coefficient.abs().sum()
        # No graph at all is the exact zero: the leaf never entered this
        # physics, so there is nothing to accumulate rather than something
        # small. Asserting "no grad_fn" is a stronger statement than
        # "grad == 0" would be.
        assert loss.requires_grad is False
        assert loss.grad_fn is None
    finally:
        wrong.requires_grad_(False)

    # Falsifier: the same expression over the leaf this component DOES read
    # carries a graph, so the zero above is a fact about the leaf and not about
    # the route.
    right = compiled.materials.eps_r
    right.requires_grad_(True)
    try:
        result = _fixed(compiled, prepared, sources, sinks, ad_mode="vjp")
        assert result.paths.transport.coefficient.requires_grad is True
    finally:
        right.requires_grad_(False)


# ---------------------------------------------------------------------------
# Diffraction: the advertisement is narrowed; the primal defect is recorded.
# ---------------------------------------------------------------------------


def _wedge_endpoints():
    sources = EndpointBatch(
        stable_ids=torch.tensor([1], device="cuda"),
        positions_m=_cuda([[0.0, -1.0, 0.5]]),
        polarizations=_cuda([[0.0, 0.0, 1.0]]),
        powers_w=torch.ones((1,), device="cuda"),
    )
    sinks = EndpointBatch(
        stable_ids=torch.tensor([2], device="cuda"),
        positions_m=_cuda([[3.0, 1.0, 0.5]]),
        polarizations=_cuda([[0.0, 0.0, 1.0]]),
    )
    return sources, sinks


def _wedge_request(ad_mode: str, sources, sinks) -> PropagationRequest:
    return PropagationRequest(
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=3.0e9,
        components=frozenset({"diffraction"}),
        max_depth=1,
        response="scalar_transport",
        topology_mode="discover",
        ad_mode=ad_mode,
    )


@pytest.mark.parametrize("ad_mode", ["jvp", "vjp"])
def test_diffraction_ad_is_refused_before_any_native_work(ad_mode: str) -> None:
    """REF by narrowed advertisement, with no route-specific refusal code."""

    assert capabilities().ad_modes_for_component("diffraction") == frozenset(
        {"none"}
    )
    compiled = compile_scene(wedge_diffraction_scene(), reference_frequency_hz=3.0e9)
    sources, sinks = _wedge_endpoints()
    with pytest.raises(NotImplementedError, match="unsupported for components"):
        evaluate(compiled, _wedge_request(ad_mode, sources, sinks))


def test_the_diffraction_primal_defect_is_pinned_rather_than_fixed() -> None:
    """A recorded upstream gap, not an AD cell.

    ``_solver_scene`` builds ``SolverScene(transmitters=(), receivers=())``
    because the consumer takes explicit endpoint batches, while the order-1
    diffraction topology owner indexes ``tx_polarizations[tx_index]``. That is a
    primal reachability defect: it fires at ``ad_mode='none'`` too. ADR-043
    deliberately does not fix it, because fixing it would silently re-open an AD
    column nobody has validated. This test pins the failure by exception type
    and site so a future fix is a deliberate decision rather than a side effect.
    """

    compiled = compile_scene(wedge_diffraction_scene(), reference_frequency_hz=3.0e9)
    sources, sinks = _wedge_endpoints()
    with pytest.raises(IndexError) as raised:
        evaluate(compiled, _wedge_request("none", sources, sinks))
    frames = traceback.extract_tb(raised.value.__traceback__)
    sites = {(Path(frame.filename).name, frame.name) for frame in frames}
    assert ("diffraction.py", "_diffraction_topology_order1") in sites
    assert "index 0 is out of bounds" in str(raised.value)


# ---------------------------------------------------------------------------
# Higher order: refused everywhere, loudly, before any partial result.
# ---------------------------------------------------------------------------


def test_the_record_declares_that_no_route_supports_higher_order_ad() -> None:
    assert capabilities().supports_higher_order_ad is False


def test_create_graph_through_a_reevaluation_names_the_owner() -> None:
    """Reverse-over-reverse fails inside the backward it asked to differentiate.

    It used to return a silently detached first gradient and fail one step
    later with a generic Torch message that named Torch, not Channel.
    """

    compiled, sources, _, prepared = _direction_world()
    leaf = _cuda(_SINK).clone().requires_grad_(True)
    moved = endpoints(sink_positions=leaf)[1]
    result = _fixed(compiled, prepared, sources, moved, ad_mode="vjp")
    loss = result.paths.transport.coefficient.abs().sum()
    with pytest.raises(NotImplementedError, match="first-order only") as raised:
        torch.autograd.grad(loss, leaf, create_graph=True)
    assert "backward" in str(raised.value)


def test_create_graph_through_a_discovery_names_the_owner() -> None:
    compiled = smooth_wall_scene()
    leaf = _cuda(_SINK).clone().requires_grad_(True)
    sources, sinks = endpoints(sink_positions=leaf)
    result = discover(compiled, sources, sinks, ad_mode="vjp")
    loss = result.paths.transport.coefficient.abs().sum()
    with pytest.raises(NotImplementedError, match="first-order only"):
        torch.autograd.grad(loss, leaf, create_graph=True)


def test_a_forward_over_reverse_request_is_refused_before_any_result() -> None:
    """The worst silent cell the survey found, now a pre-compute refusal.

    A reverse gradient taken inside a dual level came back with the correct
    first-order value and ``unpack_dual(grad).tangent is None``: a mixed second
    derivative read as an exact zero, with no error anywhere.
    """

    compiled = smooth_wall_scene()
    base = _cuda(_SINK)
    produced: list[object] = []
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(base, torch.ones_like(base))
        dual.requires_grad_(True)
        sources, sinks = endpoints(sink_positions=dual)
        with pytest.raises(NotImplementedError, match="second-order request"):
            produced.append(discover(compiled, sources, sinks, ad_mode="vjp"))
    assert produced == []


def test_a_forward_over_reverse_reevaluation_is_refused_before_any_result() -> None:
    compiled, sources, _, prepared = _direction_world()
    base = _cuda(_SINK)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(base, torch.ones_like(base))
        dual.requires_grad_(True)
        moved = endpoints(sink_positions=dual)[1]
        with pytest.raises(NotImplementedError, match="second-order request"):
            _fixed(compiled, prepared, sources, moved, ad_mode="vjp")


def test_a_requires_grad_primal_under_a_forward_dual_stays_supported() -> None:
    """The symmetric rule was verified and deliberately NOT enforced.

    ADR-038's declared convention is that a forward-only dual and a
    ``requires_grad`` leaf agree bit for bit, and the field facades run one
    Function for both modes. Refusing ``jvp`` + ``requires_grad`` would have
    broken that convention, so reverse-over-reverse is caught where it becomes
    wrong instead - inside the backward.
    """

    compiled, sources, _, prepared = _direction_world()
    base = _cuda(_SINK).clone().requires_grad_(True)
    with forward_ad.dual_level():
        moved = endpoints(
            sink_positions=forward_ad.make_dual(base, _cuda([[0.0, 1.0, 0.0]]))
        )[1]
        result = _fixed(compiled, prepared, sources, moved, ad_mode="jvp")
        tangent = forward_ad.unpack_dual(result.paths.geometry.delay_s).tangent
    assert tangent is not None and float(tangent.abs().sum()) > 0.0


def test_nested_forward_levels_raise_from_torch_and_that_is_the_owner() -> None:
    """One composition Channel does not wrap: Torch owns it and says so."""

    compiled, sources, _, prepared = _direction_world()
    base = _cuda(_SINK)
    with forward_ad.dual_level():
        outer = forward_ad.make_dual(base, torch.ones_like(base))
        with pytest.raises(RuntimeError, match="Nested forward mode AD"):
            with forward_ad.dual_level():
                inner = forward_ad.make_dual(outer, torch.ones_like(base))
                _fixed(
                    compiled,
                    prepared,
                    sources,
                    endpoints(sink_positions=inner)[1],
                    ad_mode="jvp",
                )


# ---------------------------------------------------------------------------
# AD accounting on both consumer routes.
# ---------------------------------------------------------------------------


def test_the_record_declares_ad_accounting_and_both_routes_publish_it() -> None:
    assert capabilities().ad_accounting is True

    compiled, sources, sinks, prepared = _direction_world()
    primal = _fixed(compiled, prepared, sources, sinks, ad_mode="none")
    assert primal.diagnostics.ad_companion_launches == 0
    assert primal.diagnostics.ad_tape_bytes == 0

    leaf = _cuda(_SINK).clone().requires_grad_(True)
    reverse = _fixed(
        compiled, prepared, sources, endpoints(sink_positions=leaf)[1],
        ad_mode="vjp",
    )
    assert reverse.diagnostics.ad_companion_launches == len(prepared.buckets)
    assert reverse.diagnostics.ad_tape_bytes > 0

    with forward_ad.dual_level():
        moved = endpoints(
            sink_positions=forward_ad.make_dual(
                _cuda(_SINK), _cuda([[0.0, 1.0, 0.0]])
            )
        )[1]
        forward = _fixed(compiled, prepared, sources, moved, ad_mode="jvp")
    # Forward mode retains nothing past the solve, which is the ledger's own
    # contract; the solver metadata layer applies the same gate.
    assert forward.diagnostics.ad_companion_launches == len(prepared.buckets)
    assert forward.diagnostics.ad_tape_bytes == 0


def test_the_discovery_route_publishes_the_ledger_it_already_built() -> None:
    compiled = smooth_wall_scene()
    leaf = _cuda(_SINK).clone().requires_grad_(True)
    sources, sinks = endpoints(sink_positions=leaf)
    reverse = discover(compiled, sources, sinks, ad_mode="vjp")
    assert reverse.diagnostics.ad_companion_launches > 0
    assert reverse.diagnostics.ad_tape_bytes > 0

    primal = discover(compiled, *endpoints(), ad_mode="none")
    assert primal.diagnostics.ad_companion_launches == 0
    assert primal.diagnostics.ad_tape_bytes == 0


def test_a_wideband_sweep_accounts_every_column_it_launches() -> None:
    """The honest launch law: each column drives its own companions."""

    compiled, sources, _, prepared = _direction_world()
    leaf = _cuda(_SINK).clone().requires_grad_(True)
    moved = endpoints(sink_positions=leaf)[1]
    single = _fixed(compiled, prepared, sources, moved, ad_mode="vjp")
    swept = _fixed(
        compiled,
        prepared,
        sources,
        moved,
        ad_mode="vjp",
        frequency_offsets_hz=(-4.0e6, 0.0, 4.0e6),
    )
    assert swept.diagnostics.frequency_column_count == 3
    assert swept.diagnostics.ad_companion_launches == (
        4 * single.diagnostics.ad_companion_launches
    )


# ---------------------------------------------------------------------------
# Previously untested but working cells: vertices, materials, combined inputs,
# Jones forward mode, slots, time-varying reverse mode.
# ---------------------------------------------------------------------------


def test_a_mesh_vertex_gradient_matches_the_image_source_closed_form() -> None:
    """SUP: the vertex leaf is live on the reevaluate route, and it is right.

    The oracle is analytic rather than a finite difference. For one planar wall
    at ``x = w``, the specular path length from source ``S`` to sink ``R`` is
    ``|S' - R|`` with the image ``S' = (2w - S_x, S_y, S_z)``, so

        d|S' - R| / dw = 2 * (S'_x - R_x) / |S' - R|.

    A rigid ``+x`` translation of the wall moves every vertex by the same
    amount, so the analytic answer for that direction is the sum of the
    per-vertex ``x`` gradients. That checks the accumulation across vertices as
    well as the per-vertex value, and needs no recompiled scene.
    """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    prepared = _prepared(compiled, sources, sinks)
    reflection = next(
        bucket for bucket in prepared.buckets if bucket.depth == 1
    )

    vertices = compiled.structures[0].vertices
    vertices.requires_grad_(True)
    try:
        result = _fixed(compiled, prepared, sources, sinks, ad_mode="vjp")
        length = result.paths.geometry.path_length_m[reflection.rows].sum()
        length.backward()
        analytic = float(vertices.grad[:, 0].sum())
        measured_length = float(length)
    finally:
        vertices.requires_grad_(False)
        vertices.grad = None

    source = sources.positions_m[0]
    sink = sinks.positions_m[0]
    image = torch.stack(
        (2.0 * torch.tensor(WALL_X_M, device="cuda") - source[0], source[1], source[2])
    )
    expected_length = float(torch.linalg.vector_norm(image - sink))
    assert measured_length == pytest.approx(expected_length, rel=1.0e-5)

    expected = 2.0 * float(image[0] - sink[0]) / expected_length
    assert analytic == pytest.approx(expected, rel=1.0e-3, abs=1.0e-5)


def test_a_combined_request_equals_the_sum_of_its_single_leaf_gradients() -> None:
    """Several leaves live at once, checked against one leaf at a time.

    A wrong accumulation, a shared buffer, or a leaf that silently stops
    contributing once another one becomes live all fail here and nowhere else.
    The endpoint half additionally carries a finite-difference oracle.
    """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    prepared = _prepared(compiled, sources, sinks)
    sink_base = _cuda(_SINK)
    eps = compiled.materials.eps_r

    def loss(sink_positions: torch.Tensor) -> torch.Tensor:
        moved = endpoints(sink_positions=sink_positions)[1]
        result = _fixed(compiled, prepared, sources, moved, ad_mode="vjp")
        return result.paths.transport.coefficient.abs().sum()

    try:
        sink_leaf = sink_base.clone().requires_grad_(True)
        eps.requires_grad_(True)
        loss(sink_leaf).backward()
        combined_sink = sink_leaf.grad.clone()
        combined_eps = eps.grad.clone()
        eps.grad = None

        # The material pass runs while the leaf is still live: CompiledScene
        # caches its scene-static tables only for a primal scene, so a primal
        # pass in between would publish a cached table that the material leaf
        # is no longer part of.
        loss(sink_base).backward()
        alone_eps = eps.grad.clone()

        eps.requires_grad_(False)
        alone = sink_base.clone().requires_grad_(True)
        loss(alone).backward()
        alone_sink = alone.grad.clone()
    finally:
        eps.requires_grad_(False)
        eps.grad = None

    assert float(combined_sink.abs().sum()) > 0.0
    assert float(combined_eps.abs().sum()) > 0.0
    torch.testing.assert_close(combined_sink, alone_sink, atol=0.0, rtol=0.0)
    torch.testing.assert_close(combined_eps, alone_eps, atol=0.0, rtol=0.0)

    reference = _central_difference(
        sink_base, lambda pos: float(loss(pos).item())
    )
    torch.testing.assert_close(
        combined_sink, reference, atol=5.0e-3, rtol=5.0e-2
    )


def test_the_jones_operator_carries_forward_tangents() -> None:
    """Forward mode on the composed operator, which had no test at all."""

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    prepared = _prepared(compiled, sources, sinks)
    base = _cuda(_SINK)
    with forward_ad.dual_level():
        moved = endpoints(
            sink_positions=forward_ad.make_dual(base, _cuda([[0.0, 1.0, 0.0]]))
        )[1]
        result = _fixed(
            compiled,
            prepared,
            sources,
            moved,
            ad_mode="jvp",
            response="polarimetric_transport",
        )
        tangent = forward_ad.unpack_dual(result.paths.transport.matrix).tangent
    assert tangent is not None
    assert float(tangent.abs().sum()) > 0.0


def _slot_world(slot_count: int):
    """The Phase-7 multi-endpoint world, stacked slot-major with a leaf."""

    compiled = compiled_world()
    prepared = frozen_topology(compiled)
    offsets = slot_offsets(slot_count)
    del prepared
    per_slot = frozen_topology(compiled)
    leaf = (
        stack_over_slots(cuda_positions(SINK_POSITIONS), offsets)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    sources = source_batch(
        stack_over_slots(cuda_positions(SOURCE_POSITIONS), offsets)
    )
    return compiled, per_slot, sources, sink_batch(leaf), leaf


def test_a_slot_batched_replay_carries_reverse_gradients() -> None:
    """Reverse mode across the block-diagonal slot layout, previously untested.

    Only forward mode had a slot test, so nothing checked that a reverse pass
    scatters its endpoint gradient back into the right slot.
    """

    compiled, prepared, sources, sinks, leaf = _slot_world(3)
    result = replay(
        compiled,
        replicate_over_slots(
            prepared, 3, source_count=len(SOURCE_POSITIONS),
            sink_count=len(SINK_POSITIONS),
        ),
        sources,
        sinks,
        slot_count=3,
        ad_mode="vjp",
    )
    coefficient = result.paths.transport.coefficient
    rows_per_slot = coefficient.shape[0] // 3
    coefficient.abs().sum().backward(retain_graph=True)
    assert leaf.grad is not None
    per_slot = leaf.grad.reshape(3, -1, 3).abs().sum(dim=(1, 2))
    for slot in range(3):
        assert float(per_slot[slot]) > 0.0, f"slot {slot} carries no gradient"

    # Slots never cross-pair, so a loss that reads one slot's rows must leave
    # every other slot's endpoints at an exact zero. A gather that leaked
    # across the block diagonal would fail here and nowhere else.
    leaf.grad = None
    middle = coefficient[rows_per_slot : 2 * rows_per_slot]
    middle.abs().sum().backward()
    isolated = leaf.grad.reshape(3, -1, 3).abs().sum(dim=(1, 2))
    assert float(isolated[1]) > 0.0
    assert float(isolated[0]) == 0.0
    assert float(isolated[2]) == 0.0


def test_a_time_varying_replay_carries_reverse_gradients() -> None:
    """Reverse mode on the CIR route, previously covered in forward mode only."""

    compiled, prepared, sources, sinks, leaf = _slot_world(3)
    result = evaluate_time_varying(
        compiled,
        TimeVaryingRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=SLOT_FREQUENCY_HZ,
            topology=prepared,
            times_s=torch.arange(3, dtype=torch.float64) * 1.0e-3,
            response="scalar_transport",
            ad_mode="vjp",
        ),
    )
    result.transport.coefficient.abs().sum().backward()
    assert leaf.grad is not None
    per_slot = leaf.grad.reshape(3, -1, 3).abs().sum(dim=(1, 2))
    for slot in range(3):
        assert float(per_slot[slot]) > 0.0, f"instant {slot} carries no gradient"


# ---------------------------------------------------------------------------
# transmission: advertised by the capability record on both AD modes, and now
# proved at the consumer boundary the record describes.
# ---------------------------------------------------------------------------


_TRANSMISSION_FREQUENCY_HZ = 3.0e9
_TRANSMISSION_SOURCE = [[0.0, 0.0, 0.5]]
_TRANSMISSION_SINK = [[4.0, 0.0, 0.5]]
_TRANSMISSION_WEIGHT = (0.7, -1.3)


def _transmission_world():
    """One thin sheet between one transmitter and one receiver."""

    wall = transmission_wall_structure(
        2.0,
        PhysicalMaterial(eps_r=4.0, sigma_e=0.01, thickness_m=0.1),
        name="sheet",
        surface_id=7,
    )
    scene = Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(torch.tensor([0.0, 0.0, 0.5])),
            make_receiver(torch.tensor([4.0, 0.0, 0.5])),
        ],
    )
    return compile_scene(
        scene, reference_frequency_hz=_TRANSMISSION_FREQUENCY_HZ
    )


def _transmission_loss(compiled):
    """Weighted real/imaginary loss: a magnitude loss cannot see the phase."""

    weight_re, weight_im = _TRANSMISSION_WEIGHT

    def loss(source_positions: torch.Tensor, ad_mode: str = "none"):
        sources = EndpointBatch(
            stable_ids=torch.tensor([1], dtype=torch.int64, device="cuda"),
            positions_m=source_positions,
            polarizations=_cuda([[0.0, 0.0, 1.0]]),
            powers_w=torch.ones(1, device="cuda"),
        )
        sinks = EndpointBatch(
            stable_ids=torch.tensor([2], dtype=torch.int64, device="cuda"),
            positions_m=_cuda(_TRANSMISSION_SINK),
            polarizations=_cuda([[0.0, 0.0, 1.0]]),
        )
        result = evaluate(
            compiled,
            PropagationRequest(
                sources=sources,
                sinks=sinks,
                reference_frequency_hz=_TRANSMISSION_FREQUENCY_HZ,
                components=frozenset({"transmission"}),
                max_depth=2,
                response="scalar_transport",
                topology_mode="discover",
                ad_mode=ad_mode,
            ),
        )
        coefficient = result.paths.transport.coefficient
        assert int(coefficient.shape[0]) == 1, "the sheet must publish one row"
        value = (coefficient.real * weight_re + coefficient.imag * weight_im).sum()
        return value, result

    return loss


def test_a_transmission_scene_carries_a_reverse_gradient_matching_fd() -> None:
    """The one advertised component that had no consumer-boundary row.

    `capabilities().ad_modes_for_component("transmission")` has always carried
    `jvp` and `vjp`, and `response_components` reaches transmission from both
    transport responses, but the only coverage sat one layer below the consumer
    in the enumerated engine. An advertised cell whose only evidence is a
    different surface is exactly the silent class this matrix exists to remove.
    """

    compiled = _transmission_world()
    loss = _transmission_loss(compiled)

    base = _cuda(_TRANSMISSION_SOURCE)
    leaf = base.clone().requires_grad_(True)
    value, result = loss(leaf, "vjp")
    assert result.paths.transport.coefficient.requires_grad is True
    value.backward()

    reference = _central_difference(base, lambda pos: loss(pos)[0].item())
    assert float(reference.abs().max()) > 1.0e-2, "the oracle must be nonzero"
    torch.testing.assert_close(leaf.grad, reference, atol=2.0e-3, rtol=2.0e-2)


def test_the_transmission_tangent_agrees_with_its_reverse_gradient() -> None:
    """jvp and vjp answer the same question about the same transmitted row."""

    compiled = _transmission_world()
    loss = _transmission_loss(compiled)

    base = _cuda(_TRANSMISSION_SOURCE)
    leaf = base.clone().requires_grad_(True)
    loss(leaf, "vjp")[0].backward()

    generator = torch.Generator(device="cuda").manual_seed(43)
    direction = torch.randn(base.shape, device="cuda", generator=generator)
    with forward_ad.dual_level():
        dual = forward_ad.make_dual(base.clone(), direction)
        value, _ = loss(dual, "jvp")
        tangent = forward_ad.unpack_dual(value).tangent
    assert tangent is not None, "the forward route published no tangent"

    expected = float((leaf.grad * direction).sum())
    assert abs(expected) > 1.0e-2, "the adjoint identity must be nontrivial"
    assert abs(float(tangent) - expected) <= 2.0e-3 * abs(expected)
