"""World provenance and fixed-topology staleness (ADR-040).

Before this contract, replaying a frozen topology against a world that had
moved returned a full-strength old answer with ``row_valid=[True, True]`` and
no warning at all. These tests pin the refusal, the one declared motion mode
that is legitimate, and the two limitations that stay documented rather than
fixed: a replay can never gain a row, and a compiled scene that is internally
consistent but describes an older instant is indistinguishable from a current
one unless the caller revalidates it against its live source world.
"""

from __future__ import annotations

import pytest
import torch

from witwin.channel.propagation.consumer import (
    EndpointBatch,
    FixedTopologyRequest,
    PropagationRequest,
    PropagationTopology,
    WorldProvenance,
    capabilities,
    evaluate,
    prepare_fixed_topology,
    rediscovery_required,
    reevaluate,
)
from witwin.channel.scene import compile as compile_scene
from witwin.core import (
    DeformationState,
    DynamicScene,
    LinearTrajectory,
    Mesh,
    PhysicalMaterial,
    Scene,
    Structure,
)

from tests.propagation.consumer._reflection_world import DeviceReadCounter


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

FREQUENCY_HZ = 77.0e9
WALL_X_M = 4.0
WALL_HALF_Y_M = 1.2
WALL_HALF_Z_M = 3.0
WALL_VERTICES = (
    (WALL_X_M, -WALL_HALF_Y_M, -WALL_HALF_Z_M),
    (WALL_X_M, WALL_HALF_Y_M, -WALL_HALF_Z_M),
    (WALL_X_M, WALL_HALF_Y_M, WALL_HALF_Z_M),
    (WALL_X_M, -WALL_HALF_Y_M, WALL_HALF_Z_M),
)
WALL_FACES = ((0, 1, 2), (0, 2, 3))
CONCRETE = dict(eps_r=5.24, sigma_e=0.0462)

# The single-pair world of the staleness probe: one wall in the specular
# corridor between a transmitter at the origin and a sink 15 cm along +x.
TX_POSITIONS = ((0.0, 0.0, 0.0),)
TX_IDS = (10,)
RX_POSITIONS = ((0.15, 0.0, 0.0),)
RX_IDS = (30,)

# The two-source, two-sink world of the invalidation probe, where an arriving
# wall both kills line-of-sight rows and creates a reflection row.
MULTI_TX_POSITIONS = ((0.0, 0.0, 0.0), (6.0, -1.0, 0.0))
MULTI_TX_IDS = (10, 11)
MULTI_SITE_POSITIONS = ((2.0, 0.6, 0.0), (2.0, 2.4, 0.0))
MULTI_SITE_IDS = (20, 21)


def _wall_mesh() -> Mesh:
    # recenter=False: witwin.core.Mesh otherwise silently rewrites the authored
    # world coordinates and the wall stops being where this world says it is.
    return Mesh(
        vertices=torch.tensor(WALL_VERTICES, dtype=torch.float32),
        faces=torch.tensor(WALL_FACES, dtype=torch.int64),
        recenter=False,
        fill_mode="surface",
        topology_diagnostics=False,
    )


def _wall_structure(
    mesh: Mesh,
    material: PhysicalMaterial,
    *,
    assignment_id: int = 1,
) -> Structure:
    return Structure(
        geometry=mesh,
        material=material,
        structure_id=1,
        material_id=1,
        assignment_id=assignment_id,
        surface_id=1,
    )


def _wall_world(
    *,
    eps_r: float = CONCRETE["eps_r"],
    assignment_id: int = 1,
) -> tuple[Scene, Mesh, PhysicalMaterial]:
    mesh = _wall_mesh()
    material = PhysicalMaterial(
        name="concrete", eps_r=eps_r, sigma_e=CONCRETE["sigma_e"]
    )
    scene = Scene(
        structures=(_wall_structure(mesh, material, assignment_id=assignment_id),)
    )
    return scene, mesh, material


def _batch(positions, stable_ids, *, power_w: float | None = None) -> EndpointBatch:
    device = torch.device("cuda")
    count = len(stable_ids)
    return EndpointBatch(
        stable_ids=torch.tensor(stable_ids, dtype=torch.int64, device=device),
        positions_m=torch.tensor(positions, dtype=torch.float32, device=device),
        polarizations=torch.tensor(
            [(0.0, 0.0, 1.0)] * count, dtype=torch.float32, device=device
        ),
        powers_w=(
            None
            if power_w is None
            else torch.full((count,), power_w, dtype=torch.float32, device=device)
        ),
    )


def _pair() -> tuple[EndpointBatch, EndpointBatch]:
    return (
        _batch(TX_POSITIONS, TX_IDS, power_w=0.01),
        _batch(RX_POSITIONS, RX_IDS),
    )


def _multi_pair() -> tuple[EndpointBatch, EndpointBatch]:
    return (
        _batch(MULTI_TX_POSITIONS, MULTI_TX_IDS, power_w=0.01),
        _batch(MULTI_SITE_POSITIONS, MULTI_SITE_IDS),
    )


def _discover(compiled, sources, sinks, *, max_depth: int = 1):
    return evaluate(
        compiled,
        PropagationRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            components=frozenset({"los", "reflection"}),
            max_depth=max_depth,
            response="scalar_transport",
            topology_mode="discover",
            ad_mode="none",
        ),
    )


def _replay(compiled, topology, sources, sinks, *, world_motion="frozen_world"):
    return reevaluate(
        compiled,
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=topology,
            response="scalar_transport",
            ad_mode="none",
            world_motion=world_motion,
        ),
    )


def _moved_wall_scene(scene: Scene, *, metres: float, time_s: float = 1.0):
    """The same wall, translated along +y by ``metres`` at ``time_s``."""

    dynamic = DynamicScene(
        scene,
        structure_trajectories={
            1: LinearTrajectory(
                origin=torch.zeros(3),
                velocity=torch.tensor([0.0, metres / time_s, 0.0]),
            )
        },
    )
    return dynamic, dynamic.at(time_s)


def test_a_fresh_replay_passes() -> None:
    scene, _mesh, _material = _wall_world()
    compiled = compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ)
    sources, sinks = _pair()

    discovered = _discover(compiled, sources, sinks)
    assert discovered.paths.path_count == 2
    prepared = prepare_fixed_topology(discovered.paths.topology)

    result = _replay(compiled, prepared, sources, sinks)

    assert result.row_valid.tolist() == [True, True]
    assert rediscovery_required(compiled, prepared) is None


def test_the_topology_carries_the_world_it_was_discovered_against() -> None:
    scene, _mesh, _material = _wall_world()
    compiled = compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ)
    sources, sinks = _pair()

    discovered = _discover(compiled, sources, sinks)
    provenance = discovered.paths.topology.provenance

    assert isinstance(provenance, WorldProvenance)
    assert provenance.moved_domain(WorldProvenance.of(compiled)) is None
    assert provenance.topology_version == compiled.topology_version
    assert provenance.geometry_version == compiled.geometry_version
    assert provenance.material_version == compiled.material_version
    assert provenance.assignment_version == compiled.assignment_version
    # prepare_fixed_topology forwards it verbatim and changes no signature.
    prepared = prepare_fixed_topology(discovered.paths.topology)
    assert prepared.provenance is provenance


def test_a_moved_structure_fails_loudly_by_default() -> None:
    scene, _mesh, _material = _wall_world()
    compiled = compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ)
    sources, sinks = _pair()
    prepared = prepare_fixed_topology(
        _discover(compiled, sources, sinks).paths.topology
    )

    _dynamic, moved = _moved_wall_scene(scene, metres=5.0)
    moved_compiled = compile_scene(moved, reference_frequency_hz=FREQUENCY_HZ)
    assert moved_compiled.geometry_version != compiled.geometry_version
    assert moved_compiled.topology_version == compiled.topology_version

    with pytest.raises(ValueError, match="geometry_version"):
        _replay(moved_compiled, prepared, sources, sinks)

    assert rediscovery_required(moved_compiled, prepared) == "geometry_version"


def test_a_declared_fixed_winner_replay_is_allowed_and_correct() -> None:
    scene, _mesh, _material = _wall_world()
    compiled = compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ)
    sources, sinks = _pair()
    prepared = prepare_fixed_topology(
        _discover(compiled, sources, sinks).paths.topology
    )

    _dynamic, moved = _moved_wall_scene(scene, metres=5.0)
    moved_compiled = compile_scene(moved, reference_frequency_hz=FREQUENCY_HZ)

    result = _replay(
        moved_compiled,
        prepared,
        sources,
        sinks,
        world_motion="fixed_winner_replay",
    )

    # The wall left the specular corridor, so the reflection row stops
    # existing. That is a complete answer, published inert at the input.
    assert result.row_valid.tolist() == [True, False]
    dead = 1
    zero = torch.zeros((), device=result.paths.geometry.delay_s.device)
    assert torch.equal(result.paths.geometry.delay_s[dead], zero)
    assert torch.equal(
        result.paths.transport.coefficient[dead].abs(), zero
    )
    assert torch.equal(result.paths.geometry.path_length_m[dead], zero)
    # The surviving line-of-sight row is untouched by the wall motion.
    assert float(result.paths.geometry.delay_s[0]) > 0.0


def test_material_and_assignment_mismatch_are_never_replayable() -> None:
    scene, mesh, material = _wall_world()
    compiled = compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ)
    sources, sinks = _pair()
    prepared = prepare_fixed_topology(
        _discover(compiled, sources, sinks).paths.topology
    )

    # A different material on the same mesh: only material_version moves.
    # witwin.core hashes a material by (material_id, version) plus its tensor
    # fields, so a scalar respecification announces itself through `version`.
    respecified = PhysicalMaterial(
        name="concrete", eps_r=4.0, sigma_e=CONCRETE["sigma_e"], version=1
    )
    material_scene = Scene(structures=(_wall_structure(mesh, respecified),))
    material_compiled = compile_scene(
        material_scene, reference_frequency_hz=FREQUENCY_HZ
    )
    assert material_compiled.material_version != compiled.material_version

    # A different assignment on the same mesh and the same material object.
    assignment_scene = Scene(
        structures=(_wall_structure(mesh, material, assignment_id=2),)
    )
    assignment_compiled = compile_scene(
        assignment_scene, reference_frequency_hz=FREQUENCY_HZ
    )
    assert assignment_compiled.assignment_version != compiled.assignment_version

    for target, domain in (
        (material_compiled, "material_version"),
        (assignment_compiled, "assignment_version"),
    ):
        for motion in ("frozen_world", "fixed_winner_replay"):
            with pytest.raises(ValueError, match=domain):
                _replay(target, prepared, sources, sinks, world_motion=motion)
        assert rediscovery_required(target, prepared) == domain


def test_an_unstamped_topology_is_replayable() -> None:
    """A hand-built topology has no world, so it can never be stale.

    This is the one documented escape from the freshness rule. It is pinned so
    it cannot silently widen to cover a discovery-produced topology.
    """

    scene, _mesh, _material = _wall_world()
    sources, sinks = _pair()
    device = torch.device("cuda")
    rows = 1
    empty = torch.empty((rows, 0), device=device, dtype=torch.int32)
    fabricated = PropagationTopology(
        source_index=torch.zeros((rows,), device=device, dtype=torch.int32),
        sink_index=torch.zeros((rows,), device=device, dtype=torch.int32),
        source_id=torch.tensor(TX_IDS, device=device, dtype=torch.int64),
        sink_id=torch.tensor(RX_IDS, device=device, dtype=torch.int64),
        depth=torch.zeros((rows,), device=device, dtype=torch.int32),
        component_id=torch.zeros((rows,), device=device, dtype=torch.int32),
        primitive_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        edge_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        material_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        primitive_sequence=empty,
        material_sequence=empty,
        interaction_type=empty,
    )
    assert fabricated.provenance is None

    _dynamic, moved = _moved_wall_scene(scene, metres=5.0)
    moved_compiled = compile_scene(moved, reference_frequency_hz=FREQUENCY_HZ)

    result = _replay(moved_compiled, fabricated, sources, sinks)

    assert result.paths.path_count == rows
    assert rediscovery_required(moved_compiled, fabricated) is None


def test_a_compiled_scene_that_drifted_from_its_live_world_is_reported() -> None:
    """The mutation staleness class, which the recorded versions cannot see.

    A compiled scene and the rows discovered on it always agree with each
    other, so mutating the live world in place leaves the pair internally
    consistent and the default freshness check silent. ``revalidate_source``
    is the caller-priced signal for exactly this case; it recomputes the four
    domains from the live world, which is O(scene) host work and belongs on a
    motion-event cadence rather than in a replay loop.
    """

    scene, mesh, _material = _wall_world()
    compiled = compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ)
    sources, sinks = _pair()
    prepared = prepare_fixed_topology(
        _discover(compiled, sources, sinks).paths.topology
    )
    reference = _replay(compiled, prepared, sources, sinks)

    with torch.no_grad():
        mesh.vertices[:, 1] += 5.0
    try:
        assert rediscovery_required(compiled, prepared) is None
        assert (
            rediscovery_required(compiled, prepared, revalidate_source=True)
            == "geometry_version"
        )
        # Without the caller asking, the stale pair still answers for the world
        # it describes. That is the documented boundary, not a wrong answer.
        stale = _replay(compiled, prepared, sources, sinks)
        assert torch.equal(
            stale.paths.geometry.delay_s, reference.paths.geometry.delay_s
        )
        # A recompile of the mutated world is refused by the default rule.
        current = compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ)
        assert current is not compiled
        with pytest.raises(ValueError, match="geometry_version"):
            _replay(current, prepared, sources, sinks)
    finally:
        with torch.no_grad():
            mesh.vertices[:, 1] -= 5.0


def test_an_old_compiled_scene_of_an_unmutated_world_is_not_detectable() -> None:
    """The documented limit: Channel is never told the caller moved on.

    Driving motion through ``DynamicScene.at(t)`` leaves the source ``Scene``
    untouched, so an old ``CompiledScene`` plus the rows discovered on it are a
    complete, self-consistent world. Nothing in the request names the instant
    the caller meant. The signal is the new compiled scene: once the caller
    compiles the new snapshot, the default rule refuses the frozen replay.
    """

    scene, _mesh, _material = _wall_world()
    compiled = compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ)
    sources, sinks = _pair()
    prepared = prepare_fixed_topology(
        _discover(compiled, sources, sinks).paths.topology
    )

    _dynamic, moved = _moved_wall_scene(scene, metres=5.0)
    moved_compiled = compile_scene(moved, reference_frequency_hz=FREQUENCY_HZ)

    assert rediscovery_required(compiled, prepared, revalidate_source=True) is None
    replayed = _replay(compiled, prepared, sources, sinks)
    assert replayed.row_valid.tolist() == [True, True]

    assert rediscovery_required(moved_compiled, prepared) == "geometry_version"


def test_rediscovery_required_is_host_only_and_free() -> None:
    scene, _mesh, _material = _wall_world()
    compiled = compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ)
    sources, sinks = _pair()
    prepared = prepare_fixed_topology(
        _discover(compiled, sources, sinks).paths.topology
    )
    _dynamic, moved = _moved_wall_scene(scene, metres=5.0)
    moved_compiled = compile_scene(moved, reference_frequency_hz=FREQUENCY_HZ)

    torch.cuda.synchronize()
    with DeviceReadCounter() as counter:
        assert rediscovery_required(compiled, prepared) is None
        assert rediscovery_required(moved_compiled, prepared) == "geometry_version"
    assert counter.total == 0
    # Nothing was queued on the stream either, so a per-frame poll costs the
    # caller four integer comparisons and no device work at all.
    assert torch.cuda.current_stream().query()


def test_the_adr032_budget_is_unchanged() -> None:
    scene, _mesh, _material = _wall_world()
    compiled = compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ)
    sources, sinks = _pair()
    prepared = prepare_fixed_topology(
        _discover(compiled, sources, sinks).paths.topology
    )

    diagnostics = _replay(compiled, prepared, sources, sinks).diagnostics

    assert diagnostics.validation_d2h_copies == 1
    assert diagnostics.validation_d2h_bytes == 4
    assert diagnostics.validation_sync_count == 1
    assert diagnostics.compact_count_d2h_copies == 0
    assert diagnostics.compact_count_d2h_bytes == 0
    assert diagnostics.compact_sync_count == 0


def test_a_born_row_is_absent_from_a_replay() -> None:
    """Fixed-topology replay is subtractive: rows die, rows are never born.

    A documented limitation with no birth signal, pinned so a later change
    cannot quietly claim otherwise (ADR-040).
    """

    scene, _mesh, _material = _wall_world()
    sources, sinks = _multi_pair()
    arriving = DynamicScene(
        scene,
        structure_trajectories={
            1: LinearTrajectory(
                origin=torch.tensor([0.0, 40.0, 0.0]),
                velocity=torch.tensor([0.0, -40.0, 0.0]),
            )
        },
    )
    far_snapshot = arriving.at(0.0)
    near_snapshot = arriving.at(1.0)
    far = compile_scene(far_snapshot, reference_frequency_hz=FREQUENCY_HZ)
    near = compile_scene(near_snapshot, reference_frequency_hz=FREQUENCY_HZ)

    far_rows = _discover(far, sources, sinks)
    near_rows = _discover(near, sources, sinks)
    assert far_rows.paths.path_count == 4
    assert far_rows.paths.topology.component_id.tolist() == [0, 0, 0, 0]
    assert near_rows.paths.path_count == 3
    assert 1 in near_rows.paths.topology.component_id.tolist()

    prepared = prepare_fixed_topology(far_rows.paths.topology)
    replayed = _replay(
        near, prepared, sources, sinks, world_motion="fixed_winner_replay"
    )

    assert replayed.paths.path_count == 4
    assert replayed.row_valid.tolist() == [True, False, True, False]
    # The reflection row that fresh discovery finds is simply not here.
    assert replayed.paths.topology.component_id.tolist() == [0, 0, 0, 0]
    assert 1 not in replayed.paths.topology.component_id.tolist()


def test_time_s_round_trips_from_the_snapshot_to_the_compiled_scene() -> None:
    scene, _mesh, _material = _wall_world()
    dynamic, snapshot = _moved_wall_scene(scene, metres=3.5, time_s=0.7)

    assert compile_scene(scene, reference_frequency_hz=FREQUENCY_HZ).time_s is None
    assert (
        compile_scene(snapshot, reference_frequency_hz=FREQUENCY_HZ).time_s == 0.7
    )

    tensor_time = torch.tensor(0.5, dtype=torch.float32)
    tensor_snapshot = dynamic.at(tensor_time)
    tensor_compiled = compile_scene(
        tensor_snapshot, reference_frequency_hz=FREQUENCY_HZ
    )
    assert tensor_compiled.time_s is tensor_time

    # It is metadata, never a gate: two instants of one static world replay
    # against each other without complaint.
    static = DynamicScene(scene)
    first = compile_scene(static.at(0.0), reference_frequency_hz=FREQUENCY_HZ)
    second = compile_scene(static.at(9.0), reference_frequency_hz=FREQUENCY_HZ)
    assert first.time_s == 0.0
    assert second.time_s == 9.0
    sources, sinks = _pair()
    prepared = prepare_fixed_topology(
        _discover(first, sources, sinks).paths.topology
    )
    assert rediscovery_required(second, prepared) is None
    assert _replay(second, prepared, sources, sinks).row_valid.tolist() == [
        True,
        True,
    ]


def test_compile_applies_rigid_motion_and_deformation_from_a_dynamic_scene() -> None:
    """The first Channel test that drives ``compile`` from a ``DynamicScene``.

    The compile cache aliases every moving snapshot of one source scene to a
    single RayD input identity and relies on Core's geometry hash to keep the
    BVH honest. That coupling spans two repositories and had no test.
    """

    scene, _mesh, _material = _wall_world()
    moving = DynamicScene(
        scene,
        structure_trajectories={
            1: LinearTrajectory(
                origin=torch.zeros(3),
                velocity=torch.tensor([0.0, 5.0, 0.0]),
            )
        },
    )
    rest = compile_scene(moving.at(0.0), reference_frequency_hz=FREQUENCY_HZ)
    moved = compile_scene(moving.at(1.0), reference_frequency_hz=FREQUENCY_HZ)

    assert rest.geometry.vertices[:, 1].tolist() == pytest.approx(
        [-1.20, 1.20, 1.20, -1.20], abs=1e-5
    )
    assert moved.geometry.vertices[:, 1].tolist() == pytest.approx(
        [3.80, 6.20, 6.20, 3.80], abs=1e-5
    )
    assert rest.rayd is not moved.rayd
    assert rest.geometry_version != moved.geometry_version
    assert rest.topology_version == moved.topology_version

    class _TranslateY:
        def at(self, time_s):
            offsets = torch.zeros((4, 3), dtype=torch.float32)
            offsets[:, 1] = 5.0 * float(time_s)
            return DeformationState(offsets=offsets)

    deforming = DynamicScene(scene, structure_deformations={1: _TranslateY()})
    deformed = compile_scene(deforming.at(1.0), reference_frequency_hz=FREQUENCY_HZ)
    assert deformed.geometry.vertices[:, 1].tolist() == pytest.approx(
        [3.80, 6.20, 6.20, 3.80], abs=1e-5
    )

    # Both routes produce a world the frozen topology refuses by default and
    # accepts once the caller declares the fixed-winner replay.
    sources, sinks = _pair()
    prepared = prepare_fixed_topology(
        _discover(rest, sources, sinks).paths.topology
    )
    for target in (moved, deformed):
        with pytest.raises(ValueError, match="geometry_version"):
            _replay(target, prepared, sources, sinks)
        result = _replay(
            target, prepared, sources, sinks, world_motion="fixed_winner_replay"
        )
        assert result.row_valid.tolist() == [True, False]


def test_the_capability_record_publishes_the_world_motion_vocabulary() -> None:
    record = capabilities()

    assert record.contract_version == 4
    assert record.world_motions == frozenset(
        {"frozen_world", "fixed_winner_replay"}
    )
    assert record.world_version_domains == (
        "topology_version",
        "material_version",
        "assignment_version",
        "geometry_version",
    )
    with pytest.raises(NotImplementedError, match="world_motion"):
        sources, sinks = _pair()
        FixedTopologyRequest(
            sources=sources,
            sinks=sinks,
            reference_frequency_hz=FREQUENCY_HZ,
            topology=PropagationTopology(
                source_index=torch.zeros((0,), device="cuda", dtype=torch.int32),
                sink_index=torch.zeros((0,), device="cuda", dtype=torch.int32),
                source_id=torch.zeros((0,), device="cuda", dtype=torch.int64),
                sink_id=torch.zeros((0,), device="cuda", dtype=torch.int64),
                depth=torch.zeros((0,), device="cuda", dtype=torch.int32),
                component_id=torch.zeros((0,), device="cuda", dtype=torch.int32),
                primitive_id=torch.zeros((0,), device="cuda", dtype=torch.int32),
                edge_id=torch.zeros((0,), device="cuda", dtype=torch.int32),
                material_id=torch.zeros((0,), device="cuda", dtype=torch.int32),
                primitive_sequence=torch.zeros(
                    (0, 0), device="cuda", dtype=torch.int32
                ),
                material_sequence=torch.zeros(
                    (0, 0), device="cuda", dtype=torch.int32
                ),
                interaction_type=torch.zeros(
                    (0, 0), device="cuda", dtype=torch.int32
                ),
            ),
            response="scalar_transport",
            ad_mode="none",
            world_motion="whatever_i_like",
        )
