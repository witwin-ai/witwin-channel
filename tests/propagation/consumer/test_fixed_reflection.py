from __future__ import annotations

import pytest
import torch

from witwin.channel.propagation.consumer import (
    FixedTopologyRequest,
    capabilities,
    prepare_fixed_topology,
    reevaluate,
)

from tests.propagation.consumer._reflection_world import (
    FREQUENCY_HZ,
    DeviceReadCounter,
    discover,
    endpoints,
    flat_phase_screen_wall_scene,
    los_blocker_scene,
    multi_endpoints,
    occluder_scene,
    smooth_wall_scene,
    two_wall_scene,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _prepared(compiled, sources, sinks, **kwargs):
    discovered = discover(compiled, sources, sinks, **kwargs)
    return discovered, prepare_fixed_topology(discovered.paths.topology)


def _reevaluate(compiled, prepared, sources, sinks, **kwargs):
    return reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=prepared,
            response=kwargs.pop("response", "complex3_transport"),
            ad_mode=kwargs.pop("ad_mode", "none"),
        ),
    )


def test_prepare_fixed_topology_buckets_cover_every_row_in_frozen_order() -> None:
    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    discovered, prepared = _prepared(compiled, sources, sinks)

    assert prepared.topology is discovered.paths.topology
    assert [(bucket.component, bucket.depth) for bucket in prepared.buckets] == [
        ("los", 0),
        ("reflection", 1),
    ]
    covered = torch.cat([bucket.rows for bucket in prepared.buckets])
    assert covered.numel() == prepared.row_count
    assert torch.equal(
        torch.sort(covered).values,
        torch.arange(prepared.row_count, device="cuda", dtype=torch.int64),
    )
    for bucket in prepared.buckets:
        assert torch.equal(bucket.rows, torch.sort(bucket.rows).values)
    assert prepared.prepare_synchronizations == prepared.prepare_d2h_copies
    assert prepared.prepare_d2h_copies > 0


def test_prepare_fixed_topology_rejects_malformed_interaction_padding() -> None:
    from witwin.channel.propagation.consumer import PropagationTopology

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    topology = discover(compiled, sources, sinks).paths.topology
    broken = PropagationTopology(
        source_index=topology.source_index,
        sink_index=topology.sink_index,
        source_id=topology.source_id,
        sink_id=topology.sink_id,
        # Claim depth 1 on the LoS row while its sequence slot still holds the
        # -1 padding sentinel.
        depth=torch.ones_like(topology.depth),
        component_id=topology.component_id,
        primitive_id=topology.primitive_id,
        edge_id=topology.edge_id,
        material_id=topology.material_id,
        primitive_sequence=topology.primitive_sequence,
        material_sequence=topology.material_sequence,
        interaction_type=topology.interaction_type,
    )

    with pytest.raises(ValueError, match="interaction sequence padding"):
        prepare_fixed_topology(broken)


def test_fixed_reflection_reproduces_discovery_at_unchanged_endpoints() -> None:
    """The frozen route is the same owners on the same geometry, so it is exact.

    Not ``assert_close``: a difference here would mean the reevaluation used a
    different operator, a different stationary point, or different materials,
    and any of those is a defect rather than a tolerance question.
    """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    discovered, prepared = _prepared(
        compiled, sources, sinks, response="complex3_transport"
    )
    fixed = _reevaluate(compiled, prepared, sources, sinks)

    assert fixed.paths.path_count == discovered.paths.path_count
    assert torch.equal(
        fixed.paths.transport.field, discovered.paths.transport.field
    )
    assert torch.equal(
        fixed.paths.transport.direction, discovered.paths.transport.direction
    )
    for name in ("path_length_m", "delay_s", "interaction_positions_m"):
        assert torch.equal(
            getattr(fixed.paths.geometry, name),
            getattr(discovered.paths.geometry, name),
        ), name
    assert torch.equal(fixed.paths.pair_index, discovered.paths.pair_index)
    assert torch.equal(fixed.paths.pair_offsets, discovered.paths.pair_offsets)
    assert bool(fixed.row_valid.all())


def test_fixed_reflection_publishes_the_los_validation_budget_only() -> None:
    """What the published diagnostics claim about this route.

    These numbers are constants the route reports about itself, so on their
    own they only pin the declared budget and the absence of a compaction
    stage. The measured half of the claim is the sibling test below; keep both.
    """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    _, prepared = _prepared(compiled, sources, sinks)

    fixed = _reevaluate(compiled, prepared, sources, sinks)

    assert fixed.diagnostics.validation_d2h_copies == 1
    assert fixed.diagnostics.validation_d2h_bytes == 4
    assert fixed.diagnostics.validation_sync_count == 1
    assert fixed.diagnostics.compact_count_d2h_copies == 0
    assert fixed.diagnostics.compact_count_d2h_bytes == 0
    assert fixed.diagnostics.compact_sync_count == 0
    assert fixed.diagnostics.discovery_launch_count == 0


def test_the_prepared_route_reads_exactly_one_device_value_per_call() -> None:
    """The measured budget, not the self-reported one.

    ``validation_d2h_copies == 1`` is a constant the route publishes about
    itself and cannot fail on a regression. This counts the host reads of CUDA
    tensors that actually happen inside one warm reevaluation: the single
    contract bitmask ``.item()`` and nothing else. A second count read, a
    Boolean compaction, or a host-side row decision fails here.
    """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    _, prepared = _prepared(compiled, sources, sinks)
    _reevaluate(compiled, prepared, sources, sinks)

    with DeviceReadCounter() as counter:
        _reevaluate(compiled, prepared, sources, sinks)

    assert counter.counts.get("item", 0) == 1, counter.counts
    assert counter.total == 1, counter.counts


def test_a_frozen_topology_with_no_rows_reevaluates_to_an_empty_answer() -> None:
    from witwin.channel.propagation.consumer import PropagationTopology

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    base = discover(compiled, sources, sinks).paths.topology
    empty = PropagationTopology(
        **{
            name: getattr(base, name)[:0]
            for name in (
                "source_index",
                "sink_index",
                "source_id",
                "sink_id",
                "depth",
                "component_id",
                "primitive_id",
                "edge_id",
                "material_id",
                "primitive_sequence",
                "material_sequence",
                "interaction_type",
            )
        }
    )
    prepared = prepare_fixed_topology(empty)

    assert prepared.row_count == 0
    assert prepared.buckets == ()
    assert prepared.prepare_d2h_copies == 0
    result = _reevaluate(compiled, prepared, sources, sinks)
    assert result.paths.path_count == 0
    # No bucket carries a validity-publishing component, so there is no mask.
    assert result.row_valid is None
    assert result.diagnostics.validation_d2h_copies == 0


def test_a_dead_reflection_row_is_answered_not_raised() -> None:
    """A frozen path can stop existing; that is an answer, not a failure.

    The mask is the sole authority. The surviving rows must be untouched, so a
    caller does not have to rediscover the whole batch because one path died.
    """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    _, prepared = _prepared(compiled, sources, sinks, response="complex3_transport")
    baseline = _reevaluate(compiled, prepared, sources, sinks)

    moved_sinks = endpoints(
        sink_positions=torch.tensor([[0.0, 30.0, 0.0]], device="cuda")
    )[1]
    moved = _reevaluate(compiled, prepared, sources, moved_sinks)

    assert moved.row_valid is not None
    assert moved.row_valid.tolist() == [True, False]
    dead = 1
    zero3 = torch.zeros((3,), device="cuda")
    assert torch.equal(
        moved.paths.transport.field[dead],
        zero3.to(torch.complex64),
    )
    assert float(moved.paths.geometry.path_length_m[dead]) == 0.0
    assert float(moved.paths.geometry.delay_s[dead]) == 0.0
    assert torch.equal(moved.paths.geometry.field_direction[dead], zero3)
    assert torch.equal(
        moved.paths.geometry.interaction_positions_m[dead],
        torch.zeros((1, 3), device="cuda"),
    )
    assert torch.equal(
        moved.paths.geometry.interaction_normals[dead],
        torch.zeros((1, 3), device="cuda"),
    )
    # The living LoS row is a real answer at the new endpoint, not the baseline
    # value, and it is finite.
    assert torch.isfinite(moved.paths.transport.field).all()
    assert float(moved.paths.geometry.path_length_m[0]) == pytest.approx(
        30.5, rel=1.0e-6
    )
    assert float(baseline.paths.geometry.path_length_m[0]) == pytest.approx(
        1.0, rel=1.0e-6
    )


def test_an_invalid_row_contributes_exact_zero_to_every_gradient() -> None:
    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    _, prepared = _prepared(compiled, sources, sinks)

    far = torch.tensor(
        [[0.0, 30.0, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True
    )
    moved_sources, moved_sinks = endpoints(sink_positions=far)
    result = reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=moved_sources,
            sinks=moved_sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=prepared,
            response="scalar_transport",
            ad_mode="vjp",
        ),
    )
    assert result.row_valid.tolist() == [True, False]
    # Only the LoS row is alive, so d(sum path_length)/d(sink) is exactly the
    # unit vector along that one segment. Any leakage from the dead reflection
    # row would show up as an x or z component.
    result.paths.geometry.path_length_m.sum().backward()
    torch.testing.assert_close(
        far.grad, torch.tensor([[0.0, 1.0, 0.0]], device="cuda")
    )


def test_prepared_los_route_agrees_with_the_native_los_gather_route() -> None:
    """The prepared route must not change the answer for a LoS-only topology.

    The raw-topology route keeps its fused native gather and its unchanged
    all-or-nothing validation. This pins the new structural row selection to
    that shipped behavior.
    """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    discovered = discover(
        compiled,
        sources,
        sinks,
        components=frozenset({"los"}),
        max_depth=0,
        response="complex3_transport",
    )
    topology = discovered.paths.topology
    request = dict(
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=FREQUENCY_HZ,
        response="complex3_transport",
        ad_mode="none",
    )
    raw = reevaluate(compiled, FixedTopologyRequest(topology=topology, **request))
    prepared = reevaluate(
        compiled,
        FixedTopologyRequest(
            topology=prepare_fixed_topology(topology), **request
        ),
    )

    assert torch.equal(raw.paths.transport.field, prepared.paths.transport.field)
    assert torch.equal(raw.paths.pair_index, prepared.paths.pair_index)
    assert torch.equal(raw.paths.pair_offsets, prepared.paths.pair_offsets)
    assert torch.equal(
        raw.paths.geometry.path_length_m, prepared.paths.geometry.path_length_m
    )
    assert raw.row_valid is None
    assert prepared.row_valid is None


def test_prepared_gather_rejects_endpoint_identity_drift_before_native_work() -> None:
    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    _, prepared = _prepared(compiled, sources, sinks)
    # ``type(sinks)`` rather than a fresh import: a sibling test reloads the
    # consumer package to prove cold-import neutrality.
    renamed = type(sinks)(
        stable_ids=torch.tensor([999], device="cuda", dtype=torch.int64),
        positions_m=sinks.positions_m,
        polarizations=sinks.polarizations,
        polarization_basis=sinks.polarization_basis,
    )

    with pytest.raises(ValueError, match="error bitmask 16"):
        _reevaluate(compiled, prepared, sources, renamed)


def test_reevaluate_rejects_a_rough_scene_before_any_native_work() -> None:
    """Rough-surface attenuation belongs to the discovery field loop.

    Reproducing that host-gated policy here would duplicate another owner, and
    silently disagreeing with ``evaluate`` is worse than refusing.
    """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    _, prepared = _prepared(compiled, sources, sinks)
    rough = smooth_wall_scene(rms_height_m=0.01)

    with pytest.raises(NotImplementedError, match="requires a smooth scene"):
        _reevaluate(rough, prepared, sources, sinks)


def test_raw_topology_with_interactions_points_at_prepare_fixed_topology() -> None:
    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    topology = discover(compiled, sources, sinks).paths.topology

    with pytest.raises(NotImplementedError, match="prepare_fixed_topology"):
        reevaluate(
            compiled,
            FixedTopologyRequest(
                sources=sources,
                sinks=sinks,
                reference_frequency_hz=FREQUENCY_HZ,
                topology=topology,
                response="scalar_transport",
                ad_mode="none",
            ),
        )


def test_reflection_transport_is_finite_on_the_inert_zero_geometry() -> None:
    """The precondition the row-validity contract rests on.

    An invalid row reaches the native transport with RayD's zeroed hit
    geometry. Its outputs are never published, but they must stay finite so a
    zero cotangent cannot become a NaN and poison a living row through the
    shared material tensors.
    """

    from witwin.channel.propagation.fields.kernels import (
        autograd as field_autograd,
    )

    source = torch.tensor(
        [[0.0, -0.5, 0.0]], device="cuda", requires_grad=True
    )
    target = torch.tensor([[0.0, 0.5, 0.0]], device="cuda", requires_grad=True)
    zeros = torch.zeros((1, 1, 3), device="cuda")
    material = tuple(
        torch.full((1, 1), value, device="cuda")
        for value in (4.0, 0.01, 1.0, 1.0, 0.1)
    )
    out = field_autograd.field_reflection_sequence_ad(
        source,
        target,
        zeros,
        zeros,
        torch.ones((1,), device="cuda"),
        torch.zeros((1, 3), device="cuda"),
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
        *material,
        frequency=FREQUENCY_HZ,
        frequency_value=FREQUENCY_HZ,
    )
    for name, value in out.items():
        assert torch.isfinite(value).all(), name
    out["coefficient"].real.sum().backward()
    assert torch.isfinite(source.grad).all()
    assert torch.isfinite(target.grad).all()


def test_the_specular_point_moves_with_the_sink_and_that_motion_is_differentiable(
) -> None:
    """The term this whole capability exists to provide, pinned to a number.

    A frozen reflection row is a face sequence, so at a new sink the
    stationary point slides along the wall and everything downstream of it -
    incidence angle, Fresnel coefficients, arrival direction - moves with it.
    An implementation that detached the EPC hit geometry would still publish a
    plausible finite gradient through the direct endpoint dependence, so
    "finite and non-zero" proves nothing here. These are the exact image-source
    values: the wall is at x = 2 with the source and sink both at x = 0, so the
    stationary point is the midpoint in y and z, and the x sensitivity follows
    from the image at ``4 - x_sink``.
    """

    compiled = smooth_wall_scene()
    sink_positions = torch.tensor(
        [[0.0, 0.5, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True
    )
    sources, sinks = endpoints()
    _, prepared = _prepared(compiled, sources, sinks)
    ad_sources, ad_sinks = endpoints(sink_positions=sink_positions)
    result = reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=ad_sources,
            sinks=ad_sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=prepared,
            response="complex3_transport",
            ad_mode="vjp",
        ),
    )
    hit = result.paths.geometry.interaction_positions_m[1, 0]
    torch.testing.assert_close(
        hit, torch.tensor([2.0, 0.0, 0.0], device="cuda"), rtol=0, atol=1.0e-5
    )
    assert hit.requires_grad, "the specular point carries no gradient at all"

    hit.sum().backward()
    # d(x_hit + y_hit + z_hit)/d(sink): the hit stays on the plane so the x
    # row is zero; y moves at half the sink's y and at 0.125 per unit sink x;
    # z moves at half the sink's z.
    torch.testing.assert_close(
        sink_positions.grad,
        torch.tensor([[0.125, 0.5, 0.5]], device="cuda"),
        rtol=1.0e-4,
        atol=1.0e-6,
    )


def test_depth_two_rows_reproduce_discovery_and_die_inertly() -> None:
    """Multibounce buckets, which a single-wall world never reaches.

    Two facing walls give depth-2 rows, so this is the first thing that
    exercises a second bucket launch, the ``[N, depth]`` material gather, the
    interaction padding, and - once the rows die - a dead row whose two
    coincident zeroed hit points form a zero-length middle segment. The
    unmasked field outputs are published straight from the kernel, so a NaN
    there would reach a caller.
    """

    compiled = two_wall_scene()
    sources, sinks = endpoints()
    discovered = discover(
        compiled, sources, sinks, max_depth=2, response="complex3_transport"
    )
    prepared = prepare_fixed_topology(discovered.paths.topology)
    assert [(bucket.component, bucket.depth) for bucket in prepared.buckets] == [
        ("los", 0),
        ("reflection", 1),
        ("reflection", 2),
    ]
    depth_two = prepared.buckets[2].rows
    assert depth_two.numel() == 2

    same = _reevaluate(compiled, prepared, sources, sinks)
    assert torch.equal(
        same.paths.transport.field, discovered.paths.transport.field
    )
    assert torch.equal(
        same.paths.geometry.interaction_positions_m,
        discovered.paths.geometry.interaction_positions_m,
    )
    assert bool(same.row_valid.all())

    far = endpoints(sink_positions=torch.tensor([[0.0, 5.0, 0.0]], device="cuda"))[1]
    dead = _reevaluate(compiled, prepared, sources, far)
    assert dead.row_valid.tolist() == [True, False, False, False, False]
    assert torch.isfinite(dead.paths.transport.field).all()
    assert torch.isfinite(dead.paths.geometry.path_length_m).all()
    invalid = ~dead.row_valid
    assert float(dead.paths.transport.field[invalid].abs().max()) == 0.0
    assert float(dead.paths.geometry.path_length_m[invalid].abs().max()) == 0.0
    assert float(
        dead.paths.geometry.interaction_positions_m[invalid].abs().max()
    ) == 0.0


def test_an_occluded_reflection_row_is_invalid_and_the_occluder_is_the_cause(
) -> None:
    """The other way a frozen row dies, and a control that isolates it.

    Both shipped dead-row tests kill their row by sliding the stationary point
    off its facet. This one keeps the stationary point at ``(2, 0.5, 0)``,
    comfortably inside the facet, and blocks the arrival leg instead. The
    control replays the identical frozen row and identical endpoints in a scene
    without the plate: it stays valid there, so the plate is what killed it.
    """

    compiled = occluder_scene()
    sources, sinks = endpoints()
    _, prepared = _prepared(compiled, sources, sinks, response="complex3_transport")
    assert [(bucket.component, bucket.depth) for bucket in prepared.buckets] == [
        ("los", 0),
        ("reflection", 1),
    ]
    moved = endpoints(sink_positions=torch.tensor([[0.0, 1.5, 0.0]], device="cuda"))[1]

    blocked = _reevaluate(compiled, prepared, sources, moved)
    assert blocked.row_valid.tolist() == [True, False]
    assert float(blocked.paths.transport.field[1].abs().max()) == 0.0

    control = _reevaluate(
        smooth_wall_scene(),
        prepare_fixed_topology(
            discover(
                smooth_wall_scene(), sources, sinks, response="complex3_transport"
            ).paths.topology
        ),
        sources,
        moved,
    )
    assert control.row_valid.tolist() == [True, True]
    # The stationary point really is on the facet in the control, so the only
    # difference between the two runs is the plate.
    torch.testing.assert_close(
        control.paths.geometry.interaction_positions_m[1, 0],
        torch.tensor([2.0, 0.5, 0.0], device="cuda"),
        rtol=0,
        atol=1.0e-5,
    )
    # Fresh discovery agrees the blocked path is gone.
    assert discover(compiled, sources, moved).paths.path_count == 1


def test_a_frozen_los_row_is_never_invalidated_by_an_occluder() -> None:
    """A declared limit, pinned so it cannot drift into a silent surprise.

    Row validity covers reflection only. A frozen LoS row is replayed as pure
    free space and is not re-tested for visibility, so a sink that moves behind
    a wall still publishes a full-strength LoS answer while fresh discovery
    drops the row entirely. A caller that needs blockage on LoS has to
    rediscover; the mask will not tell it.
    """

    compiled = los_blocker_scene()
    clear_sinks = endpoints(
        sink_positions=torch.tensor([[0.0, -0.2, 0.0]], device="cuda")
    )[1]
    sources, _ = endpoints()
    discovered = discover(
        compiled, sources, clear_sinks, response="complex3_transport"
    )
    depths = discovered.paths.topology.depth.tolist()
    assert depths.count(0) == 1, depths
    los_row = depths.index(0)
    prepared = prepare_fixed_topology(discovered.paths.topology)

    behind = endpoints(
        sink_positions=torch.tensor([[0.0, 0.5, 0.0]], device="cuda")
    )[1]
    replayed = _reevaluate(compiled, prepared, sources, behind)
    fresh = discover(compiled, sources, behind, response="complex3_transport")

    assert 0 not in fresh.paths.topology.depth.tolist(), (
        "discovery still finds the blocked LoS row"
    )
    assert bool(replayed.row_valid[los_row])
    assert float(replayed.paths.transport.field[los_row].abs().max()) > 0.0


def test_a_multi_pair_batch_reproduces_discovery_row_for_row() -> None:
    """Twelve rows over six pairs, none of them interchangeable.

    Every earlier test runs one source, one sink, and one row per bucket, so
    the interleaved scatter back into frozen row order, the pair segmentation,
    and the source/sink index roles are all unconstrained. Here the two sources
    and three sinks are off-axis in different ways, so swapping any of that
    changes values rather than permuting equal rows.
    """

    compiled = smooth_wall_scene()
    sources, sinks = multi_endpoints()
    discovered = discover(compiled, sources, sinks, response="complex3_transport")
    prepared = prepare_fixed_topology(discovered.paths.topology)

    assert prepared.row_count == 12
    assert [(bucket.component, bucket.depth) for bucket in prepared.buckets] == [
        ("los", 0),
        ("reflection", 1),
    ]
    assert prepared.buckets[0].rows.tolist() == [0, 2, 4, 6, 8, 10]
    assert prepared.buckets[1].rows.tolist() == [1, 3, 5, 7, 9, 11]

    fixed = _reevaluate(compiled, prepared, sources, sinks)
    assert torch.equal(
        fixed.paths.transport.field, discovered.paths.transport.field
    )
    assert torch.equal(
        fixed.paths.geometry.interaction_positions_m,
        discovered.paths.geometry.interaction_positions_m,
    )
    assert torch.equal(fixed.paths.pair_index, discovered.paths.pair_index)
    assert torch.equal(fixed.paths.pair_offsets, discovered.paths.pair_offsets)
    assert bool(fixed.row_valid.all())
    # No two rows of a bucket agree, so nothing here could be masked by symmetry.
    reflection = fixed.paths.geometry.interaction_positions_m[1::2, 0]
    assert int(torch.unique(reflection, dim=0).shape[0]) == 6


def test_forward_mode_publishes_geometry_tangents_under_the_declared_convention(
) -> None:
    """The derivative family a Doppler consumer reads, checked against FD.

    ``delay_rate`` comes from the forward-mode tangent of ``delay_s``, so this
    is the one AD mode the capability record declares that nothing else
    exercises. The reference is central differences of the same reevaluation.
    """

    import torch.autograd.forward_ad as forward_ad

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    _, prepared = _prepared(compiled, sources, sinks)

    def path_lengths(sink_y: float) -> torch.Tensor:
        moved = endpoints(
            sink_positions=torch.tensor([[0.0, sink_y, 0.0]], device="cuda")
        )[1]
        return _reevaluate(
            compiled, prepared, sources, moved
        ).paths.geometry.path_length_m

    step = 1.0e-3
    reference = (path_lengths(0.5 + step) - path_lengths(0.5 - step)) / (2 * step)

    primal = torch.tensor(
        [[0.0, 0.5, 0.0]], device="cuda", dtype=torch.float32, requires_grad=True
    )
    tangent = torch.tensor([[0.0, 1.0, 0.0]], device="cuda")
    with forward_ad.dual_level():
        dual_sinks = endpoints(
            sink_positions=forward_ad.make_dual(primal, tangent)
        )[1]
        result = reevaluate(
            compiled,
            FixedTopologyRequest(
                sources=sources,
                sinks=dual_sinks,
                reference_frequency_hz=FREQUENCY_HZ,
                topology=prepared,
                response="scalar_transport",
                ad_mode="jvp",
            ),
        )
        path_tangent = forward_ad.unpack_dual(
            result.paths.geometry.path_length_m
        ).tangent
        delay_tangent = forward_ad.unpack_dual(
            result.paths.geometry.delay_s
        ).tangent
        coefficient_tangent = forward_ad.unpack_dual(
            result.paths.transport.coefficient
        ).tangent
        path_tangent = None if path_tangent is None else path_tangent.clone()
        delay_tangent = None if delay_tangent is None else delay_tangent.clone()
        has_coefficient_tangent = coefficient_tangent is not None

    assert path_tangent is not None, "geometry tangent silently dropped"
    assert delay_tangent is not None
    assert has_coefficient_tangent
    torch.testing.assert_close(path_tangent, reference, rtol=1.0e-3, atol=1.0e-4)
    # delay is path length over c, exactly, so the two tangents share that ratio.
    torch.testing.assert_close(
        delay_tangent * 299792458.0, path_tangent, rtol=1.0e-5, atol=1.0e-9
    )


def test_a_forward_only_dual_is_rejected_rather_than_partially_differentiated(
) -> None:
    """A partial derivative is not an acceptable answer.

    ``Function.apply`` unpacks a dual before ``setup_context`` runs, so the
    shared native field companions cannot see a forward-only tangent and
    publish ``path_length_m`` and ``delay_s`` without one while the transport
    still carries its tangent. Radar reads exactly those two outputs, so the
    prepared route refuses the call instead of returning a silently incomplete
    derivative.
    """

    import torch.autograd.forward_ad as forward_ad

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    _, prepared = _prepared(compiled, sources, sinks)
    primal = torch.tensor([[0.0, 0.5, 0.0]], device="cuda", dtype=torch.float32)
    tangent = torch.tensor([[0.0, 1.0, 0.0]], device="cuda")

    with forward_ad.dual_level():
        dual_sinks = endpoints(
            sink_positions=forward_ad.make_dual(primal, tangent)
        )[1]
        with pytest.raises(NotImplementedError, match="requires_grad"):
            reevaluate(
                compiled,
                FixedTopologyRequest(
                    sources=sources,
                    sinks=dual_sinks,
                    reference_frequency_hz=FREQUENCY_HZ,
                    topology=prepared,
                    response="scalar_transport",
                    ad_mode="jvp",
                ),
            )


def test_reevaluate_rejects_a_realization_coherent_screen_before_native_work(
) -> None:
    """The second branch of the smooth-scene gate.

    A flat realization_coherent screen leaves ``scatter_model_id`` at 0, so it
    slips past the roughness check and has to be rejected on its own terms.
    """

    compiled = smooth_wall_scene()
    sources, sinks = endpoints()
    _, prepared = _prepared(compiled, sources, sinks)

    with pytest.raises(NotImplementedError, match="realization_coherent"):
        _reevaluate(flat_phase_screen_wall_scene(), prepared, sources, sinks)


def test_row_validity_capability_is_declared_for_reflection_only() -> None:
    record = capabilities()

    assert record.fixed_topology_components == frozenset({"los", "reflection"})
    assert record.fixed_topology_row_validity_components == frozenset(
        {"reflection"}
    )
    assert "polarimetric_transport" in record.fixed_topology_responses
