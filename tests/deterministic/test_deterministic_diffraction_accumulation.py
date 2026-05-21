from types import SimpleNamespace

import drjit as dr
import numpy as np
import pytest

from witwin.channel.core.physics.polarization import path_basis, vector_from_jones
from witwin.channel.core.runtime import Material, Tx, Wave
from witwin.channel.core.geometry.diffraction import wedge_exterior_mask
from witwin.channel.deterministic import types as wt
from witwin.channel.deterministic.trace import diffraction as trace_diffraction
from witwin.channel.deterministic.kernels.utd import utd_pair_vectors
from witwin.channel.deterministic.kernels.utd.native_impl import _shadow_support_mask
from witwin.channel.deterministic.diffraction.forward import ForwardEval
from witwin.channel.deterministic.diffraction import accumulation
from witwin.channel.deterministic.diffraction.state import SOURCE_TYPE_DIRECT_TX, State


class _EmptyClosedScene:
    def _triangle_runtime(self):
        return None


def test_receiver_tile_plan_uses_single_tile_outside_memory_safe():
    plan = accumulation.receiver_tile_plan(
        SimpleNamespace(memory_profile="default"),
        n_rx=100,
    )

    assert plan["enabled"] is False
    assert plan["tile_size"] == 100
    assert plan["tile_count"] == 1


def test_receiver_tile_plan_chunks_memory_safe_workloads():
    plan = accumulation.receiver_tile_plan(
        SimpleNamespace(memory_profile="memory_safe", receiver_tile_size=32),
        n_rx=100,
    )

    assert plan["enabled"] is True
    assert plan["tile_size"] == 32
    assert plan["tile_count"] == 4


def test_axis_aligned_z_receiver_grouping_avoids_numpy_copy(monkeypatch):
    positions = wt.Point3f(
        wt.Float([0.0, 1.0, 2.0]),
        wt.Float([0.0, 0.0, 0.0]),
        wt.Float([1.5, 1.5, 1.5]),
    )
    runtime = SimpleNamespace(
        tx=object(),
        rx=SimpleNamespace(positions=positions),
        wave=object(),
        diffraction=object(),
        reflection=object(),
    )

    def with_rx(group_positions, *, polarization):
        return SimpleNamespace(
            tx=runtime.tx,
            rx=SimpleNamespace(positions=group_positions),
            wave=runtime.wave,
            diffraction=runtime.diffraction,
            reflection=runtime.reflection,
        )

    runtime.with_rx = with_rx
    prepare_calls = []

    def fake_prepare(*args, **kwargs):
        prepare_calls.append(args)
        return ({}, {"n_edges": 0}, {"n_states": 0}, {"final_state_count": 0})

    def fail_asarray(*args, **kwargs):
        raise AssertionError("axis-aligned z grids should not copy receiver z values to NumPy")

    monkeypatch.setattr(accumulation.builders, "prepare", fake_prepare)
    monkeypatch.setattr(accumulation.np, "asarray", fail_asarray)

    raw = accumulation.trace_diffraction_raw_collections(
        runtime=runtime,
        scene=object(),
        config=SimpleNamespace(
            enable_rd_diffraction=False,
            rx_polarization=None,
        ),
        solver_controls={
            "effective": {
                "max_diffractions": 1,
                "reflection_n_rays": 0,
                "reflection_max_bounces": 0,
                "diffraction_state_budget": None,
                "inserted_reflection_state_budget": None,
                "max_inserted_reflections_per_path": None,
                "memory_profile": "default",
            },
            "selected": "accuracy",
        },
        spec=SimpleNamespace(
            axis="z",
            position=1.5,
            surface_mode="axis_aligned",
            ray_mode="3d",
        ),
        reflection_detail=None,
        state_layout="path_export_reduced",
    )

    assert len(raw) == 1
    assert prepare_calls[0][1] == 1.5
    assert dr.width(raw[0]["rx_positions"].x) == 3
    receiver_index_map = raw[0]["receiver_index_map"]
    dr.eval(receiver_index_map)
    assert [int(receiver_index_map[i]) for i in range(3)] == [0, 1, 2]


def test_accumulate_coherent_vector_only_skips_scalar_total(monkeypatch):
    class ScalarSentinel:
        def __add__(self, other):
            raise AssertionError("vector-only accumulation should not assemble scalar totals")

        __radd__ = __add__

    positions = wt.Point3f(
        wt.Float([0.0, 1.0]),
        wt.Float([0.0, 0.0]),
        wt.Float([0.0, 0.0]),
    )
    zero = wt.Complex2f(wt.Float([0.0, 0.0]), wt.Float([0.0, 0.0]))
    vector = {"x": zero, "y": zero, "z": zero}
    return_scalar_flags = []

    def fake_utd_accumulate_forward(*args, **kwargs):
        return_scalar_flags.append(kwargs.get("return_scalar"))
        return ScalarSentinel(), ScalarSentinel(), vector, vector, []

    monkeypatch.setattr(
        trace_diffraction,
        "utd_accumulate_forward",
        fake_utd_accumulate_forward,
    )

    total, total_vector = trace_diffraction.accumulate_coherent(
        state_arrays={"n_states": 1},
        edge_data={"n_edges": 1},
        sample_grid=SimpleNamespace(axis="z"),
        rx=SimpleNamespace(positions=positions),
        tx=object(),
        scene=object(),
        wave=object(),
        material=object(),
        suffix=SimpleNamespace(enabled=False),
        execution=object(),
        return_vector=True,
        return_scalar=False,
        receiver_axis="z",
    )

    assert total is None
    assert total_vector is not None
    assert return_scalar_flags == [False]


def test_raw_diffraction_support_rejects_deep_non_exterior_targets_by_default():
    edge_pos = wt.Point3f(0.0, 0.0, 0.0)
    edge_dir = wt.Vector3f(0.0, 0.0, 1.0)
    n0 = wt.Vector3f(1.0, 0.0, 0.0)
    nn = wt.Vector3f(0.0, 1.0, 0.0)
    source_pos = wt.Point3f(1.0, 1.0, 0.0)
    rx = wt.Point3f(
        wt.Float([1.0, -1.0]),
        wt.Float([1.0, -1.0]),
        wt.Float([0.0, 0.0]),
    )
    target_exterior = wedge_exterior_mask(rx - edge_pos, edge_dir, n0, nn)

    mask = _shadow_support_mask(
        batch_rx=rx,
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=n0,
        nn=nn,
        source_pos=source_pos,
        wedge_n=wt.Float(1.5),
        visible=wt.Bool([True, True]),
        target_exterior=target_exterior,
        scene=_EmptyClosedScene(),
        shadow_support_cutoff_db=None,
    )

    dr.eval(mask)
    assert np.asarray(mask).tolist() == [True, False]


def test_raw_diffraction_support_cutoff_keeps_only_boundary_shadow_band():
    edge_pos = wt.Point3f(0.0, 0.0, 0.0)
    edge_dir = wt.Vector3f(0.0, 0.0, 1.0)
    n0 = wt.Vector3f(1.0, 0.0, 0.0)
    nn = wt.Vector3f(0.0, 1.0, 0.0)
    source_pos = wt.Point3f(1.0, 1.0, 0.0)
    rx = wt.Point3f(
        wt.Float([-1.0, -1.0]),
        wt.Float([-0.1, -1.0]),
        wt.Float([0.0, 0.0]),
    )
    target_exterior = wedge_exterior_mask(rx - edge_pos, edge_dir, n0, nn)

    mask = _shadow_support_mask(
        batch_rx=rx,
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=n0,
        nn=nn,
        source_pos=source_pos,
        wedge_n=wt.Float(1.5),
        visible=wt.Bool([True, True]),
        target_exterior=target_exterior,
        scene=_EmptyClosedScene(),
        shadow_support_cutoff_db=25.0,
    )

    dr.eval(mask)
    assert np.asarray(mask).tolist() == [True, False]


@pytest.mark.gpu
def test_native_utd_pair_vectors_are_continuous_across_rsb():
    xs = np.asarray([-0.02, -0.01, -0.005, 0.005, 0.01, 0.02], dtype=np.float32)
    width = int(xs.size)
    values = lambda value: wt.Float([float(value)] * width)
    zeros = wt.Complex2f(values(0.0), values(0.0))
    ones = wt.Complex2f(values(1.0), values(0.0))
    edge_pos = wt.Point3f(values(0.0), values(0.0), values(0.0))
    edge_dir = wt.Vector3f(values(0.0), values(0.0), values(1.0))
    source_pos = wt.Point3f(values(1.0), values(1.0), values(0.0))
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    incident_vector = vector_from_jones({"u": zeros, "v": ones}, incident_basis)

    state_arrays = State.make(
        edge_idx=wt.UInt32([0] * width),
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=wt.Vector3f(values(1.0), values(0.0), values(0.0)),
        nn=wt.Vector3f(values(0.0), values(1.0), values(0.0)),
        wedge_n=values(1.5),
        adjacent_face0=wt.Int32([0] * width),
        adjacent_face1=wt.Int32([1] * width),
        source_pos=source_pos,
        incident_field=ones,
        incident_normal_derivative=zeros,
        incident_vector=incident_vector,
        incident_normal_derivative_vector={"x": zeros, "y": zeros, "z": zeros},
        edge_line_min=values(-20.0),
        edge_line_max=values(20.0),
        r0=zeros,
        rn=zeros,
    )
    target_pos = wt.Point3f(wt.Float(xs), values(-1.0), values(0.0))

    _, field_vector = utd_pair_vectors(
        state_arrays,
        target_pos,
        wave=Wave(wavelength=1.0),
        material=Material(1.0),
    )
    power = sum(
        np.abs(np.asarray(field_vector[axis])) ** 2
        for axis in ("x", "y", "z")
    )

    assert float(power.min()) > 1.0e-4
    assert float(power.max() / power.min()) < 2.0


@pytest.mark.gpu
def test_native_utd_pair_vectors_select_stationary_point_matches_drjit_endpoint_completion():
    width = 2
    values = lambda value: wt.Float([float(value)] * width)
    zeros = wt.Complex2f(values(0.0), values(0.0))
    ones = wt.Complex2f(values(1.0), values(0.0))
    edge_pos = wt.Point3f(values(0.0), values(0.0), values(0.0))
    edge_dir = wt.Vector3f(values(0.0), values(0.0), values(1.0))
    source_pos = wt.Point3f(values(-2.0), values(1.0), values(0.0))
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    incident_vector = vector_from_jones({"u": ones, "v": zeros}, incident_basis)
    face_material = {
        "eta_r": values(10000.0),
        "sigma": values(0.0),
        "gain": values(1.0),
        "use_fresnel": wt.Bool([True] * width),
    }

    state_arrays = State.make(
        edge_idx=wt.UInt32([0] * width),
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=wt.Vector3f(values(1.0), values(0.0), values(0.0)),
        nn=wt.Vector3f(values(0.0), values(1.0), values(0.0)),
        wedge_n=values(1.5),
        adjacent_face0=wt.Int32([0] * width),
        adjacent_face1=wt.Int32([1] * width),
        source_pos=source_pos,
        incident_field=ones,
        incident_normal_derivative=zeros,
        incident_vector=incident_vector,
        incident_normal_derivative_vector={"x": zeros, "y": zeros, "z": zeros},
        incident_basis=incident_basis,
        face0_material=face_material,
        face1_material=face_material,
        edge_line_min=values(-1.0),
        edge_line_max=values(1.0),
        r0=zeros,
        rn=zeros,
        source_type_code=wt.UInt32([SOURCE_TYPE_DIRECT_TX] * width),
        order=wt.UInt32([1] * width),
    )
    target_pos = wt.Point3f(
        wt.Float([2.0, 2.0]),
        wt.Float([1.0, 1.0]),
        wt.Float([-4.0, 4.0]),
    )
    wave = Wave(wavelength=0.299792458)
    material = Material(1.0)
    tx = Tx(position=(-2.0, 1.0, 0.0), polarization=(1.0, 0.0, 0.0))

    _, native_vector = utd_pair_vectors(
        state_arrays,
        target_pos,
        wave=wave,
        material=material,
        select_diffraction_point=True,
    )
    _, drjit_vector = ForwardEval.to_targets(
        state_arrays,
        target_pos,
        wave,
        return_vector=True,
        material=material,
        scene=None,
        tx=tx,
        select_diffraction_point=True,
        enable_segment_visibility=False,
    )

    native_values = np.stack(
        [
            np.asarray(native_vector[axis], dtype=np.complex64)
            for axis in ("x", "y", "z")
        ],
        axis=1,
    )
    drjit_values = np.stack(
        [
            np.asarray(drjit_vector[axis], dtype=np.complex64)
            for axis in ("x", "y", "z")
        ],
        axis=1,
    )

    np.testing.assert_allclose(native_values, drjit_values, rtol=2.0e-3, atol=2.0e-5)


@pytest.mark.gpu
def test_native_utd_pair_vectors_keep_wrapped_rsb_shadow_jump():
    xs = np.asarray([-0.03125, 0.03125], dtype=np.float32)
    width = int(xs.size)
    values = lambda value: wt.Float([float(value)] * width)
    zeros = wt.Complex2f(values(0.0), values(0.0))
    edge_pos = wt.Point3f(values(0.0), values(0.0), values(0.0))
    edge_dir = wt.Vector3f(values(0.0), values(0.0), values(1.0))
    source_pos = wt.Point3f(values(1.0), values(-4.0), values(3.0))
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    incident_jones = {
        "u": wt.Complex2f(values(-0.0006798789254389703), values(3.633725646068342e-05)),
        "v": wt.Complex2f(values(0.0046222880482673645), values(-0.000247045885771513)),
    }
    face_material = {
        "eta_r": values(10000.0),
        "sigma": values(0.0),
        "gain": values(1.0),
        "use_fresnel": wt.Bool([True] * width),
    }

    state_arrays = State.make(
        edge_idx=wt.UInt32([0] * width),
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=wt.Vector3f(values(-1.0), values(0.0), values(0.0)),
        nn=wt.Vector3f(values(0.0), values(-1.0), values(0.0)),
        wedge_n=values(1.5),
        adjacent_face0=wt.Int32([0] * width),
        adjacent_face1=wt.Int32([1] * width),
        source_pos=source_pos,
        incident_field=wt.Complex2f(values(1.0), values(0.0)),
        incident_normal_derivative=zeros,
        incident_jones=incident_jones,
        incident_derivative_jones={"u": zeros, "v": zeros},
        incident_basis=incident_basis,
        face0_material=face_material,
        face1_material=face_material,
        edge_line_min=values(-20.0),
        edge_line_max=values(20.0),
        r0=zeros,
        rn=zeros,
    )
    target_pos = wt.Point3f(wt.Float(xs), values(5.0), values(0.0))

    _, field_vector = utd_pair_vectors(
        state_arrays,
        target_pos,
        wave=Wave(wavelength=0.299792458),
        material=Material(1.0),
    )
    power = sum(
        np.abs(np.asarray(field_vector[axis])) ** 2
        for axis in ("x", "y", "z")
    )

    assert float(power.min()) > 1.0e-8
    ratio = float(power.max() / power.min())
    assert ratio > 2.0
    assert ratio < 10.0


@pytest.mark.gpu
def test_native_utd_face_operator_keeps_physical_shadow_phase_jump():
    xs = np.asarray([0.75, 0.7530770301818848], dtype=np.float32)
    width = int(xs.size)
    values = lambda value: wt.Float([float(value)] * width)
    zeros = wt.Complex2f(values(0.0), values(0.0))
    edge_pos = wt.Point3f(values(0.0), values(0.0), values(0.0))
    edge_dir = wt.Vector3f(values(0.0), values(0.0), values(1.0))
    source_pos = wt.Point3f(values(-1.0), values(-4.0), values(3.0))
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    incident_jones = {
        "u": wt.Complex2f(values(0.0006798789254389703), values(-3.633725646068342e-05)),
        "v": wt.Complex2f(values(0.0046222880482673645), values(-0.000247045885771513)),
    }
    face_material = {
        "eta_r": values(10000.0),
        "sigma": values(0.0),
        "gain": values(1.0),
        "use_fresnel": wt.Bool([True] * width),
    }

    state_arrays = State.make(
        edge_idx=wt.UInt32([0] * width),
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=wt.Vector3f(values(0.0), values(-1.0), values(0.0)),
        nn=wt.Vector3f(values(1.0), values(0.0), values(0.0)),
        wedge_n=values(1.5),
        adjacent_face0=wt.Int32([0] * width),
        adjacent_face1=wt.Int32([1] * width),
        source_pos=source_pos,
        incident_field=wt.Complex2f(values(1.0), values(0.0)),
        incident_normal_derivative=zeros,
        incident_jones=incident_jones,
        incident_derivative_jones={"u": zeros, "v": zeros},
        incident_basis=incident_basis,
        face0_material=face_material,
        face1_material=face_material,
        edge_line_min=values(-20.0),
        edge_line_max=values(20.0),
        r0=zeros,
        rn=zeros,
    )
    target_pos = wt.Point3f(wt.Float(xs), values(-3.0), values(0.0))

    _, field_vector = utd_pair_vectors(
        state_arrays,
        target_pos,
        wave=Wave(wavelength=0.299792458),
        material=Material(1.0),
    )
    samples = [
        np.asarray(field_vector[axis], dtype=np.complex64)
        for axis in ("x", "y", "z")
    ]
    power = sum(np.abs(values_) ** 2 for values_ in samples)
    coherence = sum(np.conj(values_[0]) * values_[1] for values_ in samples)
    normalized_coherence = float(
        np.real(coherence) / np.sqrt(float(power[0]) * float(power[1]))
    )

    assert float(power.max() / power.min()) < 2.0
    assert normalized_coherence < -0.5


@pytest.mark.gpu
def test_native_utd_shadow_sector_chart_does_not_switch_mid_sector():
    xs = np.asarray([0.49990234, 0.50009763], dtype=np.float32)
    width = int(xs.size)
    values = lambda value: wt.Float([float(value)] * width)
    zeros = wt.Complex2f(values(0.0), values(0.0))
    edge_pos = wt.Point3f(values(-1.0), values(0.0), values(2.5))
    edge_dir = wt.Vector3f(values(0.0), values(1.0), values(0.0))
    source_pos = wt.Point3f(values(0.0), values(-5.0), values(4.0))
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    incident_jones = {
        "u": wt.Complex2f(values(0.0040), values(-0.0002)),
        "v": wt.Complex2f(values(0.0015), values(0.0001)),
    }
    face_material = {
        "eta_r": values(10000.0),
        "sigma": values(0.0),
        "gain": values(1.0),
        "use_fresnel": wt.Bool([True] * width),
    }

    state_arrays = State.make(
        edge_idx=wt.UInt32([0] * width),
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=wt.Vector3f(values(-1.0), values(0.0), values(0.0)),
        nn=wt.Vector3f(values(0.0), values(0.0), values(1.0)),
        wedge_n=values(1.5),
        adjacent_face0=wt.Int32([0] * width),
        adjacent_face1=wt.Int32([1] * width),
        source_pos=source_pos,
        incident_field=wt.Complex2f(values(1.0), values(0.0)),
        incident_normal_derivative=zeros,
        incident_jones=incident_jones,
        incident_derivative_jones={"u": zeros, "v": zeros},
        incident_basis=incident_basis,
        face0_material=face_material,
        face1_material=face_material,
        edge_line_min=values(-20.0),
        edge_line_max=values(20.0),
        r0=zeros,
        rn=zeros,
    )
    target_pos = wt.Point3f(wt.Float(xs), values(4.0), values(1.0))

    _, field_vector = utd_pair_vectors(
        state_arrays,
        target_pos,
        wave=Wave(wavelength=0.299792458),
        material=Material(1.0),
    )
    samples = [
        np.asarray(field_vector[axis], dtype=np.complex64)
        for axis in ("x", "y", "z")
    ]
    power = sum(np.abs(values_) ** 2 for values_ in samples)
    coherence = sum(np.conj(values_[0]) * values_[1] for values_ in samples)
    normalized_coherence = float(
        np.real(coherence) / np.sqrt(float(power[0]) * float(power[1]))
    )

    assert float(power.max() / power.min()) < 1.05
    assert normalized_coherence > 0.99


@pytest.mark.gpu
def test_native_utd_face_n_boundary_uses_continuous_beta_branches():
    xs = np.asarray([0.99980462, 1.00019526], dtype=np.float32)
    width = int(xs.size)
    values = lambda value: wt.Float([float(value)] * width)
    zeros = wt.Complex2f(values(0.0), values(0.0))
    edge_pos = wt.Point3f(values(1.0), values(-1.0), values(1.0))
    edge_dir = wt.Vector3f(values(0.0), values(0.0), values(1.0))
    source_pos = wt.Point3f(values(0.0), values(-5.0), values(4.0))
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    incident_jones = {
        "u": wt.Complex2f(values(0.0040), values(-0.0002)),
        "v": wt.Complex2f(values(0.0015), values(0.0001)),
    }
    face_material = {
        "eta_r": values(10000.0),
        "sigma": values(0.0),
        "gain": values(1.0),
        "use_fresnel": wt.Bool([True] * width),
    }

    state_arrays = State.make(
        edge_idx=wt.UInt32([0] * width),
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=wt.Vector3f(values(0.0), values(-1.0), values(0.0)),
        nn=wt.Vector3f(values(1.0), values(0.0), values(0.0)),
        wedge_n=values(1.5),
        adjacent_face0=wt.Int32([0] * width),
        adjacent_face1=wt.Int32([1] * width),
        source_pos=source_pos,
        incident_field=wt.Complex2f(values(1.0), values(0.0)),
        incident_normal_derivative=zeros,
        incident_jones=incident_jones,
        incident_derivative_jones={"u": zeros, "v": zeros},
        incident_basis=incident_basis,
        face0_material=face_material,
        face1_material=face_material,
        edge_line_min=values(-20.0),
        edge_line_max=values(20.0),
        r0=zeros,
        rn=zeros,
    )
    target_pos = wt.Point3f(wt.Float(xs), values(4.0), values(1.0))

    _, field_vector = utd_pair_vectors(
        state_arrays,
        target_pos,
        wave=Wave(wavelength=0.299792458),
        material=Material(1.0),
    )
    samples = [
        np.asarray(field_vector[axis], dtype=np.complex64)
        for axis in ("x", "y", "z")
    ]
    power = sum(np.abs(values_) ** 2 for values_ in samples)
    coherence = sum(np.conj(values_[0]) * values_[1] for values_ in samples)
    normalized_coherence = float(
        np.real(coherence) / np.sqrt(float(power[0]) * float(power[1]))
    )

    assert float(power.max() / power.min()) < 1.1
    assert normalized_coherence > 0.99


@pytest.mark.gpu
def test_native_utd_reflected_face_terms_remain_phase_continuous_at_symmetric_sum_branch():
    xs = np.asarray([-0.00009766, 0.00009766], dtype=np.float32)
    width = int(xs.size)
    values = lambda value: wt.Float([float(value)] * width)
    zeros = wt.Complex2f(values(0.0), values(0.0))
    edge_pos = wt.Point3f(values(-1.0), values(0.0), values(2.5))
    edge_dir = wt.Vector3f(values(0.0), values(1.0), values(0.0))
    source_pos = wt.Point3f(values(0.0), values(-5.0), values(4.0))
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    incident_jones = {
        "u": wt.Complex2f(values(0.0040), values(-0.0002)),
        "v": wt.Complex2f(values(0.0015), values(0.0001)),
    }
    face_material = {
        "eta_r": values(10000.0),
        "sigma": values(0.0),
        "gain": values(1.0),
        "use_fresnel": wt.Bool([True] * width),
    }

    state_arrays = State.make(
        edge_idx=wt.UInt32([9] * width),
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=wt.Vector3f(values(-1.0), values(0.0), values(0.0)),
        nn=wt.Vector3f(values(0.0), values(0.0), values(1.0)),
        wedge_n=values(1.5),
        adjacent_face0=wt.Int32([0] * width),
        adjacent_face1=wt.Int32([1] * width),
        source_pos=source_pos,
        incident_field=wt.Complex2f(values(1.0), values(0.0)),
        incident_normal_derivative=zeros,
        incident_jones=incident_jones,
        incident_derivative_jones={"u": zeros, "v": zeros},
        incident_basis=incident_basis,
        face0_material=face_material,
        face1_material=face_material,
        edge_line_min=values(-20.0),
        edge_line_max=values(20.0),
        r0=zeros,
        rn=zeros,
    )
    target_pos = wt.Point3f(wt.Float(xs), values(4.0), values(1.0))

    _, field_vector = utd_pair_vectors(
        state_arrays,
        target_pos,
        wave=Wave(wavelength=0.299792458),
        material=Material(1.0),
    )
    samples = [
        np.asarray(field_vector[axis], dtype=np.complex64)
        for axis in ("x", "y", "z")
    ]
    power = sum(np.abs(values_) ** 2 for values_ in samples)
    coherence = sum(np.conj(values_[0]) * values_[1] for values_ in samples)
    normalized_coherence = float(
        np.real(coherence) / np.sqrt(float(power[0]) * float(power[1]))
    )

    assert float(power.max() / power.min()) < 1.1
    assert normalized_coherence > 0.99


@pytest.mark.gpu
@pytest.mark.parametrize(
    (
        "xs",
        "y_value",
        "edge_idx",
        "edge_pos_tuple",
        "edge_dir_tuple",
        "n0_tuple",
        "nn_tuple",
    ),
    [
        (
            [-0.25004882, -0.24995117],
            4.0,
            4,
            (1.0, -1.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        (
            [-0.00004883, 0.00004883],
            4.0,
            10,
            (1.0, 0.0, 2.5),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
        (
            [-1.75014651, -1.74985349],
            -4.0,
            2,
            (-1.0, -1.0, 1.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ),
    ],
)
def test_native_utd_sum_branch_keeps_reflected_face_shadow_jump(
    xs,
    y_value,
    edge_idx,
    edge_pos_tuple,
    edge_dir_tuple,
    n0_tuple,
    nn_tuple,
):
    xs = np.asarray(xs, dtype=np.float32)
    width = int(xs.size)
    values = lambda value: wt.Float([float(value)] * width)
    vector = lambda items: wt.Vector3f(
        values(items[0]),
        values(items[1]),
        values(items[2]),
    )
    point = lambda items: wt.Point3f(
        values(items[0]),
        values(items[1]),
        values(items[2]),
    )
    zeros = wt.Complex2f(values(0.0), values(0.0))
    edge_pos = point(edge_pos_tuple)
    edge_dir = vector(edge_dir_tuple)
    source_pos = point((0.0, -5.0, 4.0))
    incident_basis = path_basis(edge_pos - source_pos, preferred=edge_dir)
    incident_jones = {
        "u": wt.Complex2f(values(0.0040), values(-0.0002)),
        "v": wt.Complex2f(values(0.0015), values(0.0001)),
    }
    face_material = {
        "eta_r": values(10000.0),
        "sigma": values(0.0),
        "gain": values(1.0),
        "use_fresnel": wt.Bool([True] * width),
    }

    state_arrays = State.make(
        edge_idx=wt.UInt32([edge_idx] * width),
        edge_pos=edge_pos,
        edge_dir=edge_dir,
        n0=vector(n0_tuple),
        nn=vector(nn_tuple),
        wedge_n=values(1.5),
        adjacent_face0=wt.Int32([0] * width),
        adjacent_face1=wt.Int32([1] * width),
        source_pos=source_pos,
        incident_field=wt.Complex2f(values(1.0), values(0.0)),
        incident_normal_derivative=zeros,
        incident_jones=incident_jones,
        incident_derivative_jones={"u": zeros, "v": zeros},
        incident_basis=incident_basis,
        face0_material=face_material,
        face1_material=face_material,
        edge_line_min=values(-20.0),
        edge_line_max=values(20.0),
        r0=zeros,
        rn=zeros,
    )
    target_pos = wt.Point3f(wt.Float(xs), values(y_value), values(1.0))

    _, field_vector = utd_pair_vectors(
        state_arrays,
        target_pos,
        wave=Wave(wavelength=0.299792458),
        material=Material(1.0),
    )
    samples = [
        np.asarray(field_vector[axis], dtype=np.complex64)
        for axis in ("x", "y", "z")
    ]
    power = sum(np.abs(values_) ** 2 for values_ in samples)
    coherence = sum(np.conj(values_[0]) * values_[1] for values_ in samples)
    normalized_coherence = float(
        np.real(coherence) / np.sqrt(float(power[0]) * float(power[1]))
    )

    ratio = float(power.max() / power.min())
    assert ratio > 1.1 or normalized_coherence < -0.5
