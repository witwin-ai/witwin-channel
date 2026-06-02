from __future__ import annotations

import sys
import time
from pathlib import Path

import drjit as dr
import torch
ROOT = Path(__file__).resolve().parents[3]
CORE_ROOT = ROOT.parent / "core"
for root in (CORE_ROOT, ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tests.backend.test_native_kernel_consistency import (
    _build_reflection_grid_case,
    _build_suffix_grid_case,
    _clone_reflection_grid_inputs,
    _clone_suffix_grid_inputs,
    _enable_grad_complex,
    _enable_grad_point3f,
    _enable_grad_vector3f,
    _enable_grad_vector_complex,
    _reflection_grid_loss,
    _run_reflection_grid_reference,
    _suffix_grid_loss,
)
from tests._scene_helpers import box_geometry, build_scene as build_test_scene
from witwin.channel import Field, native_extension_available
import witwin as wt
from witwin.channel.config import DiffractionExecutionConfig
from witwin.channel.kernels.trace.cartesian_filter import drjit_impl as cartesian_filter_drjit
from witwin.channel.kernels.trace.cartesian_filter import native_impl as cartesian_filter_native
from witwin.channel.kernels.trace.packed_state import drjit_impl as packed_state_drjit
from witwin.channel.kernels.trace.pruning_sort import drjit_impl as pruning_sort_drjit
from witwin.channel.kernels.trace.pruning_sort import native_impl as pruning_sort_native
from witwin.channel.kernels.trace.reflection import drjit_impl as reflection_drjit
from witwin.channel.kernels.trace.reflection import native_impl as reflection_native
from witwin.channel.kernels.monitors.field.reflection_grid import native_impl as reflection_grid_native
from witwin.channel.kernels.monitors.common.suffix_grid import drjit_impl as suffix_grid_drjit
from witwin.channel.kernels.monitors.common.suffix_grid import native_impl as suffix_grid_native
from witwin.channel.kernels.trace.utd import drjit_impl as utd_drjit
from witwin.channel.kernels.trace.utd import native_impl as utd_native
from witwin.channel.trace.diffraction import _build_tx_first_order_state_arrays
from witwin.channel.trace.diffraction.constants import STATE_STATIC_KEYS, _state_history_size
from witwin.channel.trace.diffraction.state.arrays import _materialize_state_history
from witwin.channel.trace.reflection.api import compute_reflection_field
from witwin.channel.trace.materials import coerce_reflection_trace_detail
from witwin.channel.utils import drjit_to_torch_view
FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad


def _sync(value):
    if isinstance(value, dict):
        for item in value.values():
            _sync(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _sync(item)
        return
    try:
        dr.eval(value)
    except Exception:
        return


def _time_call(fn, *, warmup=3, repeats=10):
    for _ in range(warmup):
        result = fn()
        _sync(result)
        dr.sync_thread()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        _sync(result)
        dr.sync_thread()
        samples.append((time.perf_counter() - start) * 1000.0)
    return sum(samples) / len(samples)


def _time_and_peak(fn, *, warmup=3, repeats=10):
    for _ in range(warmup):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = fn()
        _sync(result)
        dr.sync_thread()

    samples = []
    peak_mb = 0.0
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = fn()
        _sync(result)
        dr.sync_thread()
        samples.append((time.perf_counter() - start) * 1000.0)
        if torch.cuda.is_available():
            peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    return sum(samples) / len(samples), peak_mb


def _max_abs_diff(lhs, rhs) -> float:
    return float(dr.max(dr.abs(lhs - rhs))[0])


def _torch_max_abs_diff(lhs, rhs) -> float:
    lhs_t = drjit_to_torch_view(lhs, detach=True)
    rhs_t = drjit_to_torch_view(rhs, detach=True)
    if lhs_t.numel() == 0:
        return 0.0
    if lhs_t.dtype == torch.bool:
        return float((lhs_t != rhs_t).any().item())
    if lhs_t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint32):
        diff = (lhs_t.to(torch.int64) - rhs_t.to(torch.int64)).abs()
        return float(diff.max().item())
    diff = (lhs_t.to(torch.float32) - rhs_t.to(torch.float32)).abs()
    return float(diff.max().item())


def _try_component(value, name: str):
    try:
        return getattr(value, name)
    except Exception:
        return None


def _state_value_max_abs_diff(lhs, rhs) -> float:
    x = _try_component(lhs, "x")
    y = _try_component(lhs, "y")
    if x is not None and y is not None:
        diffs = [
            _state_value_max_abs_diff(lhs.x, rhs.x),
            _state_value_max_abs_diff(lhs.y, rhs.y),
        ]
        z = _try_component(lhs, "z")
        if z is not None:
            diffs.append(_state_value_max_abs_diff(lhs.z, rhs.z))
        return max(diffs)
    real = _try_component(lhs, "real")
    imag = _try_component(lhs, "imag")
    if real is not None and imag is not None and real is not lhs and imag is not lhs:
        return max(
            _state_value_max_abs_diff(lhs.real, rhs.real),
            _state_value_max_abs_diff(lhs.imag, rhs.imag),
        )
    return _torch_max_abs_diff(lhs, rhs)


def _state_arrays_diff(ref: dict, got: dict) -> float:
    max_diff = abs(int(ref["n_states"]) - int(got["n_states"]))
    for key in STATE_STATIC_KEYS:
        if key not in ref and key not in got:
            continue
        max_diff = max(max_diff, _state_value_max_abs_diff(ref[key], got[key]))
    history_size = _state_history_size(ref)
    ref_edge_history, ref_reflection_history = _materialize_state_history(ref)
    got_edge_history, got_reflection_history = _materialize_state_history(got)
    for slot in range(history_size):
        max_diff = max(
            max_diff,
            _state_value_max_abs_diff(ref_edge_history[slot], got_edge_history[slot]),
        )
        max_diff = max(
            max_diff,
            _state_value_max_abs_diff(
                ref_reflection_history[slot],
                got_reflection_history[slot],
            ),
        )
    return max_diff


def _build_reflection_case():
    wavelength = 299792458.0 / 1.0e9
    k = float(2.0 * dr.pi / wavelength)
    scene = build_test_scene(
        box_geometry(center=(-2.0, -2.0, 1.5), size=2.0),
        box_geometry(center=(2.0, 1.5, 1.5), size=2.0),
    )
    field = Field(bounds=((-3.0, 3.0), (-3.0, 3.0)), size=(12, 12))
    tx = wt.Point3f(0.0, -4.0, 1.5)
    _, _, reflection_detail = compute_reflection_field(
        grid=field,
        rx_z=1.5,
        tx_pos=tx,
        scene=scene,
        wavelength=wavelength,
        k=k,
        n_rays=128,
        max_reflections=1,
        mode="3d",
        reflection_coef=1.0,
        tx_polarization=(1.0, 0.0, 0.0),
        reflection_material=None,
        use_scene_materials=False,
        return_per_bounce=False,
    )
    detail = coerce_reflection_trace_detail(reflection_detail)
    return {
        "rx_pos": field.receiver_positions_3d(position=1.5),
        "scene": scene,
        "wavelength": wavelength,
        "k": k,
        "source_paths_per_bounce": list(detail.source_paths_per_bounce),
        "reflection_detail": detail,
        "tx_polarization": (1.0, 0.0, 0.0),
    }


def _build_utd_case():
    wavelength = 299792458.0 / 1.0e9
    k_dr = 2.0 * dr.pi / wavelength
    k = float(k_dr)
    scene = build_test_scene(
        box_geometry(center=(-1.8, -1.2, 1.5), size=2.0),
        box_geometry(center=(1.8, 1.2, 1.5), size=2.0),
    )
    edge_data = scene.get_edge_data(1.5)["edge_data"]
    tx = wt.Point3f(0.0, -4.0, 1.5)
    state_arrays = _build_tx_first_order_state_arrays(tx, edge_data, wavelength, k_dr)
    field = Field(bounds=((-4.0, 4.0), (-4.0, 4.0)), size=(24, 24))
    coords = field.get_coordinates()
    rx_pos = wt.Point3f(coords["X"], coords["Y"], wt.Float(1.5))
    return {
        "state_arrays": state_arrays,
        "rx_pos": rx_pos,
        "k": k,
        "wavelength": wavelength,
        "n_edges": edge_data["n_edges"],
    }


def _build_cartesian_filter_case():
    n_prev = 2048
    n_edges = 512
    prev_idx = dr.arange(wt.UInt32, n_prev)
    prev_edge_idx = prev_idx % wt.UInt32(n_edges)
    hist0 = dr.select(
        prev_idx > wt.UInt32(0),
        wt.Int32((prev_idx - wt.UInt32(1)) % wt.UInt32(n_edges)),
        wt.Int32(-1),
    )
    hist1 = dr.select(
        prev_idx > wt.UInt32(1),
        wt.Int32((prev_idx - wt.UInt32(2)) % wt.UInt32(n_edges)),
        wt.Int32(-1),
    )
    hist2 = dr.select(
        prev_idx > wt.UInt32(2),
        wt.Int32((prev_idx - wt.UInt32(3)) % wt.UInt32(n_edges)),
        wt.Int32(-1),
    )
    prev_power = dr.select(
        (prev_idx % wt.UInt32(11)) == wt.UInt32(0),
        wt.Float(1.0e-30),
        wt.Float(1.0),
    )
    return {
        "prev_edge_idx": prev_edge_idx,
        "prev_edge_history": [hist0, hist1, hist2],
        "prev_power": prev_power,
        "n_prev": n_prev,
        "n_edges": n_edges,
        "min_power": 1.0e-20,
    }


def _build_pruning_case():
    utd_case = _build_utd_case()
    state_arrays = packed_state_drjit.concat_state_arrays([utd_case["state_arrays"]] * 4)
    return {
        "state_arrays": state_arrays,
        "budget": max(1, int(state_arrays["n_states"]) // 2),
    }


def _reflection_diff(ref, got) -> float:
    max_diff = 0.0
    for axis in ("x", "y", "z"):
        max_diff = max(max_diff, _max_abs_diff(ref[0][axis].real, got[0][axis].real))
        max_diff = max(max_diff, _max_abs_diff(ref[0][axis].imag, got[0][axis].imag))
    return max_diff


def _utd_diff(ref, got) -> float:
    max_diff = 0.0
    for lhs, rhs in (
        (ref[0].real, got[0].real),
        (ref[0].imag, got[0].imag),
        (ref[2]["x"].real, got[2]["x"].real),
        (ref[2]["x"].imag, got[2]["x"].imag),
        (ref[2]["y"].real, got[2]["y"].real),
        (ref[2]["y"].imag, got[2]["y"].imag),
    ):
        max_diff = max(max_diff, _max_abs_diff(lhs, rhs))
    return max_diff


def _reflection_grid_diff(ref, got) -> float:
    max_diff = 0.0
    for ref_component, got_component in zip(ref, got):
        max_diff = max(max_diff, _max_abs_diff(ref_component, got_component))
    return max_diff


def _suffix_grid_diff(ref_field, ref_vector, got_field, got_vector) -> float:
    max_diff = max(
        _max_abs_diff(ref_field.real, got_field.real),
        _max_abs_diff(ref_field.imag, got_field.imag),
    )
    for axis in ("x", "y", "z"):
        max_diff = max(max_diff, _max_abs_diff(ref_vector[axis].real, got_vector[axis].real))
        max_diff = max(max_diff, _max_abs_diff(ref_vector[axis].imag, got_vector[axis].imag))
    return max_diff


def _make_reflection_grid_reference_runner(case: dict):
    return lambda: _run_reflection_grid_reference(case)


def _make_reflection_grid_native_runner(case: dict):
    return lambda: reflection_grid_native.accumulate_reflection_grid(
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


def _make_reflection_grid_backward_runner(case: dict, *, backend: str):
    def run():
        inputs = _clone_reflection_grid_inputs(case)
        _enable_grad_point3f(inputs["prev_refl_p"])
        _enable_grad_vector3f(inputs["prev_refl_n"])
        _enable_grad_point3f(inputs["prev_tx"])
        _enable_grad_complex(inputs["prev_weight"])
        _enable_grad_vector_complex(inputs["prev_polarization"])

        if backend == "native":
            outputs = reflection_grid_native.accumulate_reflection_grid(
                grid=case["grid"],
                plane_position=case["grid"].position,
                grid_data=case["grid_data"],
                ray_origin=case["ray_origin"],
                ray_dir=case["ray_dir"],
                active=case["active"],
                blocker_dist=case["blocker_dist"],
                prev_refl_p=inputs["prev_refl_p"],
                prev_refl_n=inputs["prev_refl_n"],
                prev_tx=inputs["prev_tx"],
                prev_weight=inputs["prev_weight"],
                prev_polarization=inputs["prev_polarization"],
                prev_prim_idx=case["prev_prim_idx"],
                wavelength=case["wavelength"],
                k=case["k"],
                validate_paths=False,
                tri_data=None,
            )
        else:
            outputs = _run_reflection_grid_reference({**case, **inputs})
        loss = _reflection_grid_loss(outputs)
        dr.backward(loss, flags=FLAGS)
        return (
            dr.grad(inputs["prev_refl_p"]),
            dr.grad(inputs["prev_refl_n"]),
            dr.grad(inputs["prev_tx"]),
            dr.grad(inputs["prev_weight"]),
            dr.grad(inputs["prev_polarization"]["x"]),
            dr.grad(inputs["prev_polarization"]["y"]),
            dr.grad(inputs["prev_polarization"]["z"]),
        )

    return run


def _make_suffix_grid_reference_runner(case: dict):
    execution = DiffractionExecutionConfig(suffix_backend="drjit", suffix_dda="symbolic")
    return lambda: suffix_grid_drjit.accumulate_reflected_segment_fields_batched(
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
        execution=execution,
    )


def _make_suffix_grid_native_runner(case: dict):
    execution = DiffractionExecutionConfig(suffix_backend="native", suffix_dda="symbolic")
    return lambda: suffix_grid_native.accumulate_reflected_segment_fields_batched(
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
        execution=execution,
    )


def _make_native_suffix_grid_backward_runner(case: dict):
    execution = DiffractionExecutionConfig(suffix_backend="native", suffix_dda="symbolic")

    def run():
        inputs = _clone_suffix_grid_inputs(case)
        _enable_grad_point3f(inputs["seg_origin"])
        _enable_grad_complex(inputs["seg_field"])
        _enable_grad_vector_complex(inputs["seg_vector"])
        field, vector = suffix_grid_native.accumulate_reflected_segment_fields_chunk(
            grid=case["grid"],
            grid_data=case["grid_data"],
            seg_origin=inputs["seg_origin"],
            seg_dir=case["seg_dir"],
            blocker_dist=case["blocker_dist"],
            seg_field=inputs["seg_field"],
            seg_vector=inputs["seg_vector"],
            wavelength=case["wavelength"],
            k=case["k"],
            active=case["active"],
            execution=execution,
        )
        loss = _suffix_grid_loss(field, vector)
        dr.backward(loss, flags=FLAGS)
        return (
            dr.grad(inputs["seg_origin"]),
            dr.grad(inputs["seg_field"]),
            dr.grad(inputs["seg_vector"]["x"]),
            dr.grad(inputs["seg_vector"]["y"]),
            dr.grad(inputs["seg_vector"]["z"]),
        )

    return run


def main():
    if not native_extension_available():
        raise SystemExit("native extension is not available in the current environment")

    reflection_case = _build_reflection_case()
    reflection_grid_case = _build_reflection_grid_case("x")
    suffix_grid_case = _build_suffix_grid_case()
    utd_case = _build_utd_case()
    cartesian_case = _build_cartesian_filter_case()
    pruning_case = _build_pruning_case()

    reflection_ref = lambda: reflection_drjit.reflection_accumulate_forward(**reflection_case)
    reflection_nat = lambda: reflection_native.reflection_accumulate_forward(**reflection_case)
    reflection_grid_ref = _make_reflection_grid_reference_runner(reflection_grid_case)
    reflection_grid_nat = _make_reflection_grid_native_runner(reflection_grid_case)
    reflection_grid_ref_backward = _make_reflection_grid_backward_runner(
        reflection_grid_case,
        backend="drjit",
    )
    reflection_grid_nat_backward = _make_reflection_grid_backward_runner(
        reflection_grid_case,
        backend="native",
    )
    suffix_grid_ref = _make_suffix_grid_reference_runner(suffix_grid_case)
    suffix_grid_nat = _make_suffix_grid_native_runner(suffix_grid_case)
    suffix_grid_nat_backward = _make_native_suffix_grid_backward_runner(suffix_grid_case)
    utd_ref = lambda: utd_drjit.utd_accumulate_forward(
        utd_case["state_arrays"],
        utd_case["rx_pos"],
        utd_case["k"],
        utd_case["n_edges"],
        False,
        wavelength=utd_case["wavelength"],
    )
    utd_nat = lambda: utd_native.utd_accumulate_forward(
        utd_case["state_arrays"],
        utd_case["rx_pos"],
        utd_case["k"],
        utd_case["n_edges"],
        False,
        wavelength=utd_case["wavelength"],
    )
    cartesian_ref = lambda: cartesian_filter_drjit.cartesian_filter_bruteforce(**cartesian_case)
    cartesian_nat = lambda: cartesian_filter_native.cartesian_filter_bruteforce(**cartesian_case)
    pruning_ref = lambda: pruning_sort_drjit.prune_state_arrays_by_budget(
        pruning_case["state_arrays"],
        pruning_case["budget"],
        "bench",
    )
    pruning_nat = lambda: pruning_sort_native.prune_state_arrays_by_budget(
        pruning_case["state_arrays"],
        pruning_case["budget"],
        "bench",
    )

    reflection_ref_result = reflection_ref()
    reflection_nat_result = reflection_nat()
    reflection_grid_ref_result = reflection_grid_ref()
    reflection_grid_nat_result = reflection_grid_nat()
    suffix_grid_ref_result = suffix_grid_ref()
    suffix_grid_nat_result = suffix_grid_nat()
    utd_ref_result = utd_ref()
    utd_nat_result = utd_nat()
    cartesian_ref_result = cartesian_ref()
    cartesian_nat_result = cartesian_nat()
    pruning_ref_result = pruning_ref()
    pruning_nat_result = pruning_nat()

    reflection_ref_ms = _time_call(reflection_ref)
    reflection_nat_ms = _time_call(reflection_nat)
    reflection_grid_ref_ms, reflection_grid_ref_peak_mb = _time_and_peak(reflection_grid_ref)
    reflection_grid_nat_ms, reflection_grid_nat_peak_mb = _time_and_peak(reflection_grid_nat)
    reflection_grid_ref_backward_ms, reflection_grid_ref_backward_peak_mb = _time_and_peak(
        reflection_grid_ref_backward
    )
    reflection_grid_nat_backward_ms, reflection_grid_nat_backward_peak_mb = _time_and_peak(
        reflection_grid_nat_backward
    )
    suffix_grid_ref_ms, suffix_grid_ref_peak_mb = _time_and_peak(suffix_grid_ref)
    suffix_grid_nat_ms, suffix_grid_nat_peak_mb = _time_and_peak(suffix_grid_nat)
    suffix_grid_nat_backward_ms, suffix_grid_nat_backward_peak_mb = _time_and_peak(
        suffix_grid_nat_backward
    )
    utd_ref_ms = _time_call(utd_ref)
    utd_nat_ms = _time_call(utd_nat)
    cartesian_ref_ms = _time_call(cartesian_ref)
    cartesian_nat_ms = _time_call(cartesian_nat)
    pruning_ref_ms = _time_call(pruning_ref)
    pruning_nat_ms = _time_call(pruning_nat)

    print("reflection")
    print(f"  drjit_ms={reflection_ref_ms:.3f}")
    print(f"  native_ms={reflection_nat_ms:.3f}")
    print(f"  speedup={reflection_ref_ms / reflection_nat_ms:.3f}x")
    print(f"  max_abs_diff={_reflection_diff(reflection_ref_result, reflection_nat_result):.6e}")

    print("reflection_grid_forward")
    print(f"  drjit_ms={reflection_grid_ref_ms:.3f}")
    print(f"  native_ms={reflection_grid_nat_ms:.3f}")
    print(f"  speedup={reflection_grid_ref_ms / reflection_grid_nat_ms:.3f}x")
    print(f"  drjit_peak_mb={reflection_grid_ref_peak_mb:.3f}")
    print(f"  native_peak_mb={reflection_grid_nat_peak_mb:.3f}")
    print(f"  max_abs_diff={_reflection_grid_diff(reflection_grid_ref_result, reflection_grid_nat_result):.6e}")

    print("reflection_grid_backward")
    print(f"  drjit_ms={reflection_grid_ref_backward_ms:.3f}")
    print(f"  native_ms={reflection_grid_nat_backward_ms:.3f}")
    print(f"  speedup={reflection_grid_ref_backward_ms / reflection_grid_nat_backward_ms:.3f}x")
    print(f"  drjit_peak_mb={reflection_grid_ref_backward_peak_mb:.3f}")
    print(f"  native_peak_mb={reflection_grid_nat_backward_peak_mb:.3f}")

    print("suffix_grid_forward")
    print(f"  drjit_ms={suffix_grid_ref_ms:.3f}")
    print(f"  native_ms={suffix_grid_nat_ms:.3f}")
    print(f"  speedup={suffix_grid_ref_ms / suffix_grid_nat_ms:.3f}x")
    print(f"  drjit_peak_mb={suffix_grid_ref_peak_mb:.3f}")
    print(f"  native_peak_mb={suffix_grid_nat_peak_mb:.3f}")
    print(
        "  max_abs_diff="
        f"{_suffix_grid_diff(suffix_grid_ref_result[0], suffix_grid_ref_result[1], suffix_grid_nat_result[0], suffix_grid_nat_result[1]):.6e}"
    )

    print("suffix_grid_backward")
    print(f"  native_ms={suffix_grid_nat_backward_ms:.3f}")
    print("  drjit_ms=n/a (drjit symbolic suffix baseline does not support AD-sensitive inputs)")
    print(f"  native_peak_mb={suffix_grid_nat_backward_peak_mb:.3f}")

    print("utd")
    print(f"  drjit_ms={utd_ref_ms:.3f}")
    print(f"  native_ms={utd_nat_ms:.3f}")
    print(f"  speedup={utd_ref_ms / utd_nat_ms:.3f}x")
    print(f"  max_abs_diff={_utd_diff(utd_ref_result, utd_nat_result):.6e}")

    print("cartesian_filter")
    print(f"  drjit_ms={cartesian_ref_ms:.3f}")
    print(f"  native_ms={cartesian_nat_ms:.3f}")
    print(f"  speedup={cartesian_ref_ms / cartesian_nat_ms:.3f}x")
    print(f"  prev_idx_max_abs_diff={_torch_max_abs_diff(cartesian_ref_result[0], cartesian_nat_result[0]):.6e}")
    print(f"  edge_idx_max_abs_diff={_torch_max_abs_diff(cartesian_ref_result[1], cartesian_nat_result[1]):.6e}")

    print("pruning_sort")
    print(f"  drjit_ms={pruning_ref_ms:.3f}")
    print(f"  native_ms={pruning_nat_ms:.3f}")
    print(f"  speedup={pruning_ref_ms / pruning_nat_ms:.3f}x")
    print(f"  max_state_diff={_state_arrays_diff(pruning_ref_result[0], pruning_nat_result[0]):.6e}")


if __name__ == "__main__":
    main()

