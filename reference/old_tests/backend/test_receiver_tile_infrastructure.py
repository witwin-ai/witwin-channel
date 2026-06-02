import drjit as dr
import pytest
import sys

import witwin as wt
from witwin.channel import native_extension_available
from witwin.channel.kernels.monitors.common.receiver_tiles import (
    build_receiver_tiles,
    compact_tile_tasks,
    deduplicate_tile_tasks,
)
from witwin.channel.kernels.monitors.common.utd_state_tiles import build_utd_state_tile_plan
from witwin.channel.kernels.trace.reflection import drjit_impl as reflection_drjit
from witwin.channel.kernels.monitors.common.reflection_family_tiles import build_reflection_family_tile_plan
from witwin.channel.kernels.trace.reflection import native_impl as reflection_native
from witwin.channel.kernels.monitors.field.reflection_grid import native_impl as reflection_grid_native
from witwin.channel.kernels.monitors.common.suffix_grid import native_impl as suffix_grid_native
from witwin.channel.kernels.monitors.common.suffix_grid.segment_tiles import (
    build_suffix_segment_tile_plan,
    build_suffix_tile_packet_plan,
)
from witwin.channel.kernels.trace.utd import native_impl as utd_native
from witwin.channel.monitors.field import Field
from witwin.channel.utils import to_numpy
from tests.backend.test_native_kernel_consistency import (
    _build_reflection_case,
    _build_reflection_grid_case,
    _build_suffix_grid_case,
    _build_utd_case,
)


def _max_abs_diff(lhs, rhs) -> float:
    lhs_np = to_numpy(lhs)
    rhs_np = to_numpy(rhs)
    if lhs_np.size == 0 and rhs_np.size == 0:
        return 0.0
    return float(abs(lhs_np - rhs_np).max())


def _assert_close(lhs, rhs, *, atol: float = 1.0e-6):
    assert _max_abs_diff(lhs, rhs) <= atol


def _complex_total(value):
    return dr.sum(value.real) + dr.sum(value.imag)


def _vector_total(value):
    return sum(dr.sum(value[axis].real) + dr.sum(value[axis].imag) for axis in ("x", "y", "z"))


def _utd_total(outputs):
    return (
        _complex_total(outputs[0])
        + _complex_total(outputs[1])
        + _vector_total(outputs[2])
        + _vector_total(outputs[3])
    )


def _reflection_total(outputs):
    return sum(_vector_total(value) for value in outputs)


def _suffix_total(field, vector):
    return _complex_total(field) + _vector_total(vector)


def _build_packetized_suffix_case():
    grid = Field(bounds=((-1.5, 1.5), (0.6, 2.4)), size=(4, 4), axis="y", position=0.35)
    seg_origin = wt.Point3f(
        [-1.20, -1.05, -0.90, -0.75, 1.20, 1.05, 0.90, 0.75],
        [0.35] * 8,
        [0.85, 0.95, 1.05, 1.15, 0.85, 0.95, 1.05, 1.15],
    )
    seg_dir = wt.Vector3f(
        [0.45, 0.47, 0.43, 0.44, -0.46, -0.48, -0.42, -0.41],
        [0.60] * 8,
        [0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.03, 0.02],
    )
    blocker_dist = wt.Float([3.0] * 8)
    seg_field = wt.Complex2f(
        [0.50, 0.48, 0.46, 0.44, 0.52, 0.50, 0.48, 0.46],
        [0.02, 0.01, 0.00, -0.01, 0.01, 0.00, -0.01, -0.02],
    )
    seg_vector = {
        "x": wt.Complex2f(
            [0.18, 0.17, 0.16, 0.15, 0.18, 0.17, 0.16, 0.15],
            [0.01, 0.00, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00],
        ),
        "y": wt.Complex2f(
            [0.08, 0.09, 0.10, 0.11, 0.08, 0.09, 0.10, 0.11],
            [0.00, 0.01, 0.00, -0.01, 0.00, 0.01, 0.00, -0.01],
        ),
        "z": wt.Complex2f(
            [0.04, 0.05, 0.04, 0.05, 0.04, 0.05, 0.04, 0.05],
            [0.01, 0.00, -0.01, 0.00, 0.01, 0.00, -0.01, 0.00],
        ),
    }
    state_idx = wt.UInt32([0, 0, 0, 0, 1, 1, 1, 1])
    receiver_tiles = build_receiver_tiles(
        grid=grid,
        plane_position=grid.position,
        grid_data=grid.get_coordinates(),
        tile_shape=(4, 4),
    )
    return {
        "grid": grid,
        "grid_data": grid.get_coordinates(),
        "receiver_tiles": receiver_tiles,
        "seg_origin": seg_origin,
        "seg_dir": seg_dir,
        "blocker_dist": blocker_dist,
        "seg_field": seg_field,
        "seg_vector": seg_vector,
        "state_idx": state_idx,
        "n_states": 2,
        "active": wt.Bool([True] * 8),
        "wavelength": 299792458.0 / 1.0e9,
        "k": float(2.0 * dr.pi / (299792458.0 / 1.0e9)),
    }


@pytest.mark.gpu
def test_receiver_tile_builder_matches_field_geometry():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    field = Field(bounds=((-2.0, 2.0), (-1.5, 1.5)), size=(5, 4), axis="z", position=1.25)
    descriptor = build_receiver_tiles(grid=field, tile_shape=(2, 3))

    assert descriptor.builder_backend == "native_cuda"
    assert descriptor.n_tiles == 6
    assert descriptor.n_tiles_0 == 3
    assert descriptor.n_tiles_1 == 2
    assert to_numpy(descriptor.tile_i0).tolist() == [0, 2, 4, 0, 2, 4]
    assert to_numpy(descriptor.tile_i1).tolist() == [0, 0, 0, 3, 3, 3]
    assert to_numpy(descriptor.tile_extent_0).tolist() == [2, 2, 1, 2, 2, 1]
    assert to_numpy(descriptor.tile_extent_1).tolist() == [3, 3, 3, 1, 1, 1]

    coords = field.get_coordinates()
    x_coords = to_numpy(coords["x_coords"]).tolist()
    y_coords = to_numpy(coords["y_coords"]).tolist()
    assert to_numpy(descriptor.tile_coord_0_min).tolist() == [
        x_coords[0],
        x_coords[2],
        x_coords[4],
        x_coords[0],
        x_coords[2],
        x_coords[4],
    ]
    assert to_numpy(descriptor.tile_coord_0_max).tolist() == [
        x_coords[1],
        x_coords[3],
        x_coords[4],
        x_coords[1],
        x_coords[3],
        x_coords[4],
    ]
    assert to_numpy(descriptor.tile_coord_1_min).tolist() == [
        y_coords[0],
        y_coords[0],
        y_coords[0],
        y_coords[3],
        y_coords[3],
        y_coords[3],
    ]
    assert to_numpy(descriptor.tile_coord_1_max).tolist() == [
        y_coords[2],
        y_coords[2],
        y_coords[2],
        y_coords[3],
        y_coords[3],
        y_coords[3],
    ]
    assert to_numpy(descriptor.tile_aabb_min.z).tolist() == [1.25] * descriptor.n_tiles
    assert to_numpy(descriptor.tile_aabb_max.z).tolist() == [1.25] * descriptor.n_tiles


@pytest.mark.gpu
def test_receiver_tile_task_helpers_compact_and_deduplicate():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    family_idx = wt.UInt32([1, 1, 2, 2, 2, 3])
    tile_idx = wt.UInt32([4, 4, 5, 5, 7, 4])
    active = wt.Bool([True, False, True, True, False, True])

    compact_family, compact_tile = compact_tile_tasks(family_idx, tile_idx, active)
    assert to_numpy(compact_family).tolist() == [1, 2, 2, 3]
    assert to_numpy(compact_tile).tolist() == [4, 5, 5, 4]

    unique_family, unique_tile = deduplicate_tile_tasks(
        compact_family,
        compact_tile,
        n_tiles=8,
    )
    assert to_numpy(unique_family).tolist() == [1, 2, 3]
    assert to_numpy(unique_tile).tolist() == [4, 5, 4]


@pytest.mark.gpu
def test_native_reflection_exact_replay_accepts_receiver_tiles_descriptor():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_case()
    field = Field(bounds=((-3.0, 3.0), (-3.0, 3.0)), size=(5, 5), axis="z", position=1.5)
    receiver_tiles = build_receiver_tiles(
        grid=field,
        plane_position=1.5,
        receiver_positions=case["rx_pos"],
        tile_shape=(2, 2),
    )

    ref = reflection_drjit.reflection_accumulate_forward(**case)
    got = reflection_native.reflection_accumulate_forward(**case, receiver_tiles=receiver_tiles)

    assert len(ref) == len(got)
    for ref_bounce, got_bounce in zip(ref, got):
        for axis in ("x", "y", "z"):
            _assert_close(ref_bounce[axis].real, got_bounce[axis].real)
            _assert_close(ref_bounce[axis].imag, got_bounce[axis].imag)


@pytest.mark.gpu
def test_reflection_family_tile_plan_prunes_full_grid_pairs():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_case()
    field = Field(bounds=((-12.0, 12.0), (-12.0, 12.0)), size=(25, 25), axis="y", position=0.0)
    receiver_tiles = build_receiver_tiles(
        grid=field,
        plane_position=0.0,
        receiver_positions=field.receiver_positions_3d(position=0.0),
        tile_shape=(2, 2),
    )

    first_bounce_paths = case["source_paths_per_bounce"][0]
    plan = build_reflection_family_tile_plan(
        paths=first_bounce_paths,
        scene=case["scene"],
        receiver_tiles=receiver_tiles,
    )

    assert plan is not None
    assert plan.n_tiles == receiver_tiles.n_tiles
    assert plan.tile_task_count > 0
    assert plan.tile_task_count < plan.n_families * plan.n_tiles
    assert plan.estimated_pair_count < plan.n_families * field.n_cells


@pytest.mark.gpu
def test_native_reflection_grid_accepts_receiver_tiles_descriptor():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_grid_case("x")
    receiver_tiles = build_receiver_tiles(
        grid=case["grid"],
        plane_position=case["grid"].position,
        grid_data=case["grid_data"],
    )

    ref = reflection_grid_native.accumulate_reflection_grid(
        grid=case["grid"],
        plane_position=case["grid"].position,
        grid_data=case["grid_data"],
        ray_origin=case["ray_origin"],
        ray_dir=case["ray_dir"],
        active=case["active"],
        blocker_dist=case["blocker_dist"],
        prev_refl_p=case["prev_refl_p"],
        prev_refl_n=case["prev_refl_n"],
        prev_tx=case["prev_tx"],
        prev_weight=case["prev_weight"],
        prev_polarization=case["prev_polarization"],
        prev_prim_idx=case["prev_prim_idx"],
        wavelength=case["wavelength"],
        k=case["k"],
        validate_paths=False,
        tri_data=None,
    )
    got = reflection_grid_native.accumulate_reflection_grid(
        grid=case["grid"],
        plane_position=case["grid"].position,
        grid_data=case["grid_data"],
        receiver_tiles=receiver_tiles,
        ray_origin=case["ray_origin"],
        ray_dir=case["ray_dir"],
        active=case["active"],
        blocker_dist=case["blocker_dist"],
        prev_refl_p=case["prev_refl_p"],
        prev_refl_n=case["prev_refl_n"],
        prev_tx=case["prev_tx"],
        prev_weight=case["prev_weight"],
        prev_polarization=case["prev_polarization"],
        prev_prim_idx=case["prev_prim_idx"],
        wavelength=case["wavelength"],
        k=case["k"],
        validate_paths=False,
        tri_data=None,
    )

    for ref_component, got_component in zip(ref, got):
        _assert_close(ref_component, got_component)


@pytest.mark.gpu
def test_native_utd_receiver_tiles_are_execution_noop():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    field = Field(bounds=((-2.0, 2.0), (-2.0, 2.0)), size=(4, 4), axis="z", position=1.5)
    receiver_tiles = build_receiver_tiles(
        grid=field,
        plane_position=1.5,
        receiver_positions=case["rx_pos"],
        tile_shape=(2, 2),
    )

    ref = utd_native.utd_accumulate_forward(
        case["state_arrays"],
        case["rx_pos"],
        case["k"],
        case["n_edges"],
        False,
        scene=case["scene"],
        wavelength=case["wavelength"],
    )
    got = utd_native.utd_accumulate_forward(
        case["state_arrays"],
        case["rx_pos"],
        case["k"],
        case["n_edges"],
        False,
        scene=case["scene"],
        wavelength=case["wavelength"],
        receiver_tiles=receiver_tiles,
    )

    _assert_close(ref[0].real, got[0].real, atol=5.0e-5)
    _assert_close(ref[0].imag, got[0].imag, atol=5.0e-5)
    _assert_close(ref[1].real, got[1].real, atol=5.0e-5)
    _assert_close(ref[1].imag, got[1].imag, atol=5.0e-5)
    for axis in ("x", "y", "z"):
        _assert_close(ref[2][axis].real, got[2][axis].real, atol=5.0e-5)
        _assert_close(ref[2][axis].imag, got[2][axis].imag, atol=5.0e-5)
        _assert_close(ref[3][axis].real, got[3][axis].real, atol=5.0e-5)
        _assert_close(ref[3][axis].imag, got[3][axis].imag, atol=5.0e-5)


@pytest.mark.gpu
def test_utd_state_tile_plan_prunes_shadow_tiles():
    case = _build_utd_case()
    field = Field(bounds=((-12.0, 12.0), (-12.0, 12.0)), size=(25, 25), axis="z", position=1.5)
    receiver_tiles = build_receiver_tiles(
        grid=field,
        plane_position=1.5,
        receiver_positions=field.receiver_positions_3d(position=1.5),
        tile_shape=(2, 2),
    )

    plan = build_utd_state_tile_plan(
        state_arrays=case["state_arrays"],
        receiver_tiles=receiver_tiles,
    )

    assert plan is not None
    assert plan.n_tiles == receiver_tiles.n_tiles
    assert plan.tile_task_count > 0
    assert plan.tile_task_count < plan.n_states * plan.n_tiles
    assert plan.estimated_pair_count < plan.n_states * field.n_cells


@pytest.mark.gpu
def test_utd_state_tile_plan_native_matches_drjit_fallback(monkeypatch):
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_utd_case()
    field = Field(bounds=((-12.0, 12.0), (-12.0, 12.0)), size=(25, 25), axis="z", position=1.5)
    receiver_tiles = build_receiver_tiles(
        grid=field,
        plane_position=1.5,
        receiver_positions=field.receiver_positions_3d(position=1.5),
        tile_shape=(2, 2),
    )

    native_plan = build_utd_state_tile_plan(
        state_arrays=case["state_arrays"],
        receiver_tiles=receiver_tiles,
    )
    assert native_plan is not None
    assert native_plan.planner_backend == "native_cuda_halfspace_exact"

    utd_state_tiles = sys.modules[build_utd_state_tile_plan.__module__]
    monkeypatch.setattr(utd_state_tiles, "_native_utd_state_tile_planner", lambda: None)
    fallback_plan = build_utd_state_tile_plan(
        state_arrays=case["state_arrays"],
        receiver_tiles=receiver_tiles,
    )
    assert fallback_plan is not None
    assert fallback_plan.planner_backend == "wedge_exterior_halfspace_drjit"
    assert fallback_plan.n_states == native_plan.n_states
    assert fallback_plan.n_tiles == native_plan.n_tiles
    assert fallback_plan.tile_task_count == native_plan.tile_task_count
    assert fallback_plan.estimated_pair_count == native_plan.estimated_pair_count
    assert fallback_plan.max_states_per_tile == native_plan.max_states_per_tile
    assert to_numpy(fallback_plan.state_tile_counts).tolist() == to_numpy(native_plan.state_tile_counts).tolist()
    assert to_numpy(fallback_plan.tile_task_state_idx).tolist() == to_numpy(native_plan.tile_task_state_idx).tolist()
    assert to_numpy(fallback_plan.tile_task_tile_idx).tolist() == to_numpy(native_plan.tile_task_tile_idx).tolist()
    assert (
        to_numpy(fallback_plan.tile_task_counts_per_tile).tolist()
        == to_numpy(native_plan.tile_task_counts_per_tile).tolist()
    )


@pytest.mark.gpu
def test_native_suffix_grid_accepts_receiver_tiles_descriptor():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_suffix_grid_case()
    receiver_tiles = build_receiver_tiles(
        grid=case["grid"],
        plane_position=case["grid"].position,
        grid_data=case["grid_data"],
        tile_shape=(2, 2),
    )

    ref_field, ref_vector = suffix_grid_native.accumulate_reflected_segment_fields_batched(
        grid=case["grid"],
        grid_data=case["grid_data"],
        seg_origin=case["seg_origin"],
        seg_dir=case["seg_dir"],
        blocker_dist=case["blocker_dist"],
        seg_field=case["seg_field"],
        seg_vector=case["seg_vector"],
        state_idx=case["state_idx"],
        n_states=case["n_states"],
        wavelength=case["wavelength"],
        k=case["k"],
        active=case["active"],
        execution={"suffix_backend": "native"},
    )
    got_field, got_vector = suffix_grid_native.accumulate_reflected_segment_fields_batched(
        grid=case["grid"],
        grid_data=case["grid_data"],
        receiver_tiles=receiver_tiles,
        seg_origin=case["seg_origin"],
        seg_dir=case["seg_dir"],
        blocker_dist=case["blocker_dist"],
        seg_field=case["seg_field"],
        seg_vector=case["seg_vector"],
        state_idx=case["state_idx"],
        n_states=case["n_states"],
        wavelength=case["wavelength"],
        k=case["k"],
        active=case["active"],
        execution={"suffix_backend": "native"},
    )

    _assert_close(ref_field.real, got_field.real)
    _assert_close(ref_field.imag, got_field.imag)
    for axis in ("x", "y", "z"):
        _assert_close(ref_vector[axis].real, got_vector[axis].real)
        _assert_close(ref_vector[axis].imag, got_vector[axis].imag)


@pytest.mark.gpu
def test_suffix_segment_tile_plan_prunes_full_grid_pairs():
    case = _build_suffix_grid_case()
    receiver_tiles = build_receiver_tiles(
        grid=case["grid"],
        plane_position=case["grid"].position,
        grid_data=case["grid_data"],
        tile_shape=(2, 2),
    )

    plan = build_suffix_segment_tile_plan(
        seg_origin=case["seg_origin"],
        seg_dir=case["seg_dir"],
        blocker_dist=case["blocker_dist"],
        active=case["active"],
        receiver_tiles=receiver_tiles,
    )

    active_count = int(sum(bool(value) for value in to_numpy(case["active"]).tolist()))
    assert plan is not None
    assert plan.n_tiles == receiver_tiles.n_tiles
    assert plan.tile_task_count > 0
    assert plan.tile_task_count < active_count * plan.n_tiles
    assert plan.estimated_cell_count < active_count * case["grid"].n_cells


def test_suffix_tile_packet_plan_groups_coherent_segments():
    case = _build_packetized_suffix_case()
    plan = build_suffix_segment_tile_plan(
        seg_origin=case["seg_origin"],
        seg_dir=case["seg_dir"],
        blocker_dist=case["blocker_dist"],
        active=case["active"],
        receiver_tiles=case["receiver_tiles"],
    )
    packet_plan = build_suffix_tile_packet_plan(
        seg_dir=case["seg_dir"],
        receiver_tiles=case["receiver_tiles"],
        tile_plan=plan,
        state_idx=case["state_idx"],
    )

    assert plan is not None
    assert packet_plan is not None
    assert packet_plan.packet_count > 0
    assert packet_plan.max_packets_per_tile == 1
    assert packet_plan.max_segments_per_packet == plan.max_segments_per_tile


@pytest.mark.gpu
def test_native_suffix_packetized_tile_replay_matches_unpacketized_result():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_packetized_suffix_case()
    ref_field, ref_vector = suffix_grid_native.accumulate_reflected_segment_fields_batched(
        grid=case["grid"],
        grid_data=case["grid_data"],
        seg_origin=case["seg_origin"],
        seg_dir=case["seg_dir"],
        blocker_dist=case["blocker_dist"],
        seg_field=case["seg_field"],
        seg_vector=case["seg_vector"],
        state_idx=case["state_idx"],
        n_states=case["n_states"],
        wavelength=case["wavelength"],
        k=case["k"],
        active=case["active"],
        execution={"suffix_backend": "native"},
    )
    got_field, got_vector = suffix_grid_native.accumulate_reflected_segment_fields_batched(
        grid=case["grid"],
        grid_data=case["grid_data"],
        receiver_tiles=case["receiver_tiles"],
        seg_origin=case["seg_origin"],
        seg_dir=case["seg_dir"],
        blocker_dist=case["blocker_dist"],
        seg_field=case["seg_field"],
        seg_vector=case["seg_vector"],
        state_idx=case["state_idx"],
        n_states=case["n_states"],
        wavelength=case["wavelength"],
        k=case["k"],
        active=case["active"],
        execution={"suffix_backend": "native"},
    )

    _assert_close(ref_field.real, got_field.real, atol=1.0e-6)
    _assert_close(ref_field.imag, got_field.imag, atol=1.0e-6)
    for axis in ("x", "y", "z"):
        _assert_close(ref_vector[axis].real, got_vector[axis].real, atol=1.0e-6)
        _assert_close(ref_vector[axis].imag, got_vector[axis].imag, atol=1.0e-6)


@pytest.mark.gpu
def test_native_utd_receiver_tiles_do_not_change_backward_ad():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    ref_case = _build_utd_case()
    tiled_case = _build_utd_case()
    field = Field(bounds=((-2.0, 2.0), (-2.0, 2.0)), size=(4, 4), axis="z", position=1.5)
    receiver_tiles = build_receiver_tiles(
        grid=field,
        plane_position=1.5,
        receiver_positions=tiled_case["rx_pos"],
        tile_shape=(2, 2),
    )

    dr.enable_grad(ref_case["state_arrays"]["edge_pos"].x)
    ref_outputs = utd_native.utd_accumulate_forward(
        ref_case["state_arrays"],
        ref_case["rx_pos"],
        ref_case["k"],
        ref_case["n_edges"],
        False,
        scene=ref_case["scene"],
        wavelength=ref_case["wavelength"],
    )
    dr.backward(_utd_total(ref_outputs), flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    ref_grad = dr.grad(ref_case["state_arrays"]["edge_pos"].x)

    dr.enable_grad(tiled_case["state_arrays"]["edge_pos"].x)
    tiled_outputs = utd_native.utd_accumulate_forward(
        tiled_case["state_arrays"],
        tiled_case["rx_pos"],
        tiled_case["k"],
        tiled_case["n_edges"],
        False,
        scene=tiled_case["scene"],
        wavelength=tiled_case["wavelength"],
        receiver_tiles=receiver_tiles,
    )
    dr.backward(_utd_total(tiled_outputs), flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    tiled_grad = dr.grad(tiled_case["state_arrays"]["edge_pos"].x)

    _assert_close(ref_grad, tiled_grad, atol=5.0e-5)


@pytest.mark.gpu
def test_native_reflection_family_tiled_path_supports_backward_ad():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    case = _build_reflection_case()
    field = Field(bounds=((-12.0, 12.0), (-12.0, 12.0)), size=(25, 25), axis="y", position=0.0)
    receiver_tiles = build_receiver_tiles(
        grid=field,
        plane_position=0.0,
        receiver_positions=field.receiver_positions_3d(position=0.0),
        tile_shape=(2, 2),
    )
    plan = build_reflection_family_tile_plan(
        paths=case["source_paths_per_bounce"][0],
        scene=case["scene"],
        receiver_tiles=receiver_tiles,
    )
    assert plan is not None
    assert plan.estimated_pair_count < plan.n_families * field.n_cells

    ref_rx_pos = field.receiver_positions_3d(position=0.0)
    tiled_rx_pos = wt.Point3f(
        dr.detach(ref_rx_pos.x),
        dr.detach(ref_rx_pos.y),
        dr.detach(ref_rx_pos.z),
    )

    dr.enable_grad(ref_rx_pos.x)
    ref_outputs = reflection_drjit.reflection_accumulate_forward(
        rx_pos=ref_rx_pos,
        scene=case["scene"],
        wavelength=case["wavelength"],
        k=case["k"],
        source_paths_per_bounce=case["source_paths_per_bounce"],
        reflection_detail=case["reflection_detail"],
        tx_polarization=case["tx_polarization"],
    )
    dr.backward(_reflection_total(ref_outputs), flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    ref_grad = dr.grad(ref_rx_pos.x)

    dr.enable_grad(tiled_rx_pos.x)
    tiled_outputs = reflection_native.reflection_accumulate_forward(
        rx_pos=tiled_rx_pos,
        scene=case["scene"],
        wavelength=case["wavelength"],
        k=case["k"],
        source_paths_per_bounce=case["source_paths_per_bounce"],
        reflection_detail=case["reflection_detail"],
        tx_polarization=case["tx_polarization"],
        receiver_tiles=receiver_tiles,
    )
    dr.backward(_reflection_total(tiled_outputs), flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    tiled_grad = dr.grad(tiled_rx_pos.x)

    _assert_close(ref_grad, tiled_grad, atol=5.0e-5)


@pytest.mark.gpu
def test_native_suffix_tiled_path_supports_backward_ad():
    if not native_extension_available():
        pytest.skip("native extension is not built in the current environment")

    ref_case = _build_suffix_grid_case()
    tiled_case = _build_suffix_grid_case()
    receiver_tiles = build_receiver_tiles(
        grid=tiled_case["grid"],
        plane_position=tiled_case["grid"].position,
        grid_data=tiled_case["grid_data"],
        tile_shape=(2, 2),
    )

    dr.enable_grad(ref_case["seg_origin"].x)
    ref_field, ref_vector = suffix_grid_native.accumulate_reflected_segment_fields_batched(
        grid=ref_case["grid"],
        grid_data=ref_case["grid_data"],
        seg_origin=ref_case["seg_origin"],
        seg_dir=ref_case["seg_dir"],
        blocker_dist=ref_case["blocker_dist"],
        seg_field=ref_case["seg_field"],
        seg_vector=ref_case["seg_vector"],
        state_idx=ref_case["state_idx"],
        n_states=ref_case["n_states"],
        wavelength=ref_case["wavelength"],
        k=ref_case["k"],
        active=ref_case["active"],
        execution={"suffix_backend": "native"},
    )
    dr.backward(_suffix_total(ref_field, ref_vector), flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    ref_grad = dr.grad(ref_case["seg_origin"].x)

    dr.enable_grad(tiled_case["seg_origin"].x)
    tiled_field, tiled_vector = suffix_grid_native.accumulate_reflected_segment_fields_batched(
        grid=tiled_case["grid"],
        grid_data=tiled_case["grid_data"],
        receiver_tiles=receiver_tiles,
        seg_origin=tiled_case["seg_origin"],
        seg_dir=tiled_case["seg_dir"],
        blocker_dist=tiled_case["blocker_dist"],
        seg_field=tiled_case["seg_field"],
        seg_vector=tiled_case["seg_vector"],
        state_idx=tiled_case["state_idx"],
        n_states=tiled_case["n_states"],
        wavelength=tiled_case["wavelength"],
        k=tiled_case["k"],
        active=tiled_case["active"],
        execution={"suffix_backend": "native"},
    )
    dr.backward(_suffix_total(tiled_field, tiled_vector), flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    tiled_grad = dr.grad(tiled_case["seg_origin"].x)

    _assert_close(ref_grad, tiled_grad, atol=1.0e-5)

