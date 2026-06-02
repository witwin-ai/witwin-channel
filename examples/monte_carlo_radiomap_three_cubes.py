from __future__ import annotations

from collections import Counter
import contextlib
from dataclasses import dataclass
import gc
import importlib
import io
import os
from pprint import pformat
import re
import subprocess
import threading
import time

import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import witwin.channel as wt

from witwin.channel.core.scene import Mesh as ChannelMesh
from witwin.channel.core.scene import EdgePolicy, ReceiverGrid, Scene as ChannelScene, Transmitter
from witwin.channel.core.scene import SionnaAdaptor
from witwin.channel.core.geometry.mesh_buffers import to_point3f, to_vector3u
from witwin.core import Box, Material, Structure
from witwin.channel.montecarlo import (
    Config,
    IntegratorOptions,
    Tuning,
    solve,
)
import witwin.channel.montecarlo.integrators.basic as package_rm_integrator


DEFAULT_BOUNDS = ((-10.0, 10.0), (-10.0, 10.0))
DEFAULT_GRID_SHAPE = (256, 256)
DEFAULT_PLANE_Z = 1.0
DEFAULT_FREQUENCY_HZ = 1.0e9
DEFAULT_TX_POS = (0.0, -5.0, 4.0)
DEFAULT_RELATIVE_PERMITTIVITY = 1.0e4
DEFAULT_CUBE_CENTERS = (
    (-2.5, -3.0, 1.5),
    (2.0, 0.5, 1.5),
    (-0.5, 3.5, 1.5),
)
DEFAULT_CUBE_SIZE = 2.0
DEFAULT_FORWARD_REFLECTION_N_RAYS = 384
DEFAULT_GRADIENT_REFLECTION_N_RAYS = 512
DEFAULT_SAMPLES_PER_TX = 1_000_000
DEFAULT_FD_STEP = 1.0e-3
DEFAULT_DB_MIN = -70.0
DEFAULT_DB_MAX = -20.0
DEFAULT_GRADIENT_DB_FLOOR = -160.0
DEFAULT_SIONNA_MAX_DEPTH = 3
_GRAD_FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad
_GRADIENT_COMPONENTS = ("los", "reflection", "diffraction")
_MIB = 1024.0 * 1024.0
_DEVICE_ALLOCATOR_RE = re.compile(
    r"- device\s*:\s*(?P<used>[^/]+)/(?P<reserved>[^ ]+\s+[A-Za-z]+) used \(peak:\s*(?P<peak>[^)]+)\)\."
)
_MEMORY_VALUE_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[KMGT]?i?B|B)\s*$")
_MEMORY_UNIT_SCALE = {
    "B": 1,
    "KiB": 1024,
    "MiB": 1024 ** 2,
    "GiB": 1024 ** 3,
    "TiB": 1024 ** 4,
}


@dataclass(frozen=True, slots=True)
class SolveSnapshot:
    path_gain: np.ndarray
    coords_x: np.ndarray
    coords_y: np.ndarray
    metadata: dict
    components: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class GradientSnapshot:
    parameter: str
    forward: SolveSnapshot
    jvp: np.ndarray
    fd: np.ndarray
    delta: np.ndarray
    component_jvp: dict[str, np.ndarray]
    component_fd: dict[str, np.ndarray]
    component_delta: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class BackwardSnapshot:
    parameter: str
    vjp: float
    jvp_sum: float
    fd_sum: float
    vjp_delta: float
    fd_delta: float


@dataclass(frozen=True, slots=True)
class SionnaComparison:
    standalone: np.ndarray
    sionna: np.ndarray
    delta: np.ndarray
    standalone_metadata: dict


@dataclass(frozen=True, slots=True)
class KernelHistorySnapshot:
    label: str
    records: list
    summary: dict
    memory_before: dict
    memory_after: dict
    process_gpu_peak_mib: float | None

    @property
    def count(self) -> int:
        return len(self.records)

    def pretty(self) -> str:
        return pformat(self.summary, sort_dicts=False, width=120)


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    label: str
    samples_ms: tuple[float, ...]
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    drjit_allocator: dict
    process_gpu_peak_mib: float | None

    @property
    def peak_memory_mib(self) -> float | None:
        if self.process_gpu_peak_mib is not None:
            return float(self.process_gpu_peak_mib)
        peak_bytes = self.drjit_allocator.get("device_peak_bytes")
        if peak_bytes is None:
            return None
        return float(peak_bytes) / _MIB


@dataclass(frozen=True, slots=True)
class PerformanceComparison:
    witwin: PerformanceSnapshot
    sionna: PerformanceSnapshot


def _clear_drjit_ad_state() -> None:
    try:
        dr.clear_grad()
    except TypeError:
        pass
    dr.sync_thread()
    gc.collect()


def _memory_text_to_bytes(text: str | None) -> int | None:
    if text is None:
        return None
    match = _MEMORY_VALUE_RE.match(text)
    if match is None:
        return None
    value = float(match.group("value"))
    unit = match.group("unit")
    return int(round(value * _MEMORY_UNIT_SCALE[unit]))


def _capture_drjit_allocator(*, include_raw: bool = False) -> dict:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        dr.whos()
    raw = stream.getvalue()
    match = _DEVICE_ALLOCATOR_RE.search(raw)
    if match is None:
        return {"raw": raw} if include_raw else {}

    used = match.group("used").strip()
    reserved = match.group("reserved").strip()
    peak = match.group("peak").strip()
    allocator = {
        "device_used": used,
        "device_reserved": reserved,
        "device_peak": peak,
        "device_used_bytes": _memory_text_to_bytes(used),
        "device_reserved_bytes": _memory_text_to_bytes(reserved),
        "device_peak_bytes": _memory_text_to_bytes(peak),
    }
    if include_raw:
        allocator["raw"] = raw
    return allocator


def _kernel_record(entry: dict) -> dict:
    return {
        "type": str(entry.get("type", "unknown")),
        "size": int(entry.get("size", 0) or 0),
        "execution_time_ms": float(entry.get("execution_time", 0.0) or 0.0),
        "operation_count": int(entry.get("operation_count", 0) or 0),
        "hash": str(entry.get("hash", "")),
    }


def _summarize_kernel_history(
    history: list[dict],
    *,
    top_k: int = 8,
    small_kernel_max_size: int = 128,
    small_kernel_min_count: int = 3,
) -> dict:
    by_type: dict[str, dict] = {}
    repeated_small = Counter()
    records = [_kernel_record(entry) for entry in history]
    for record in records:
        type_name = str(record["type"])
        bucket = by_type.setdefault(
            type_name,
            {
                "count": 0,
                "size_sum": 0,
                "size_max": 0,
                "execution_time_ms_sum": 0.0,
                "small_count_le_128": 0,
                "small_count_le_1024": 0,
                "large_count_gt_1m": 0,
            },
        )
        size = int(record["size"])
        bucket["count"] += 1
        bucket["size_sum"] += size
        bucket["size_max"] = max(int(bucket["size_max"]), size)
        bucket["execution_time_ms_sum"] += float(record["execution_time_ms"])
        if size <= 128:
            bucket["small_count_le_128"] += 1
        if size <= 1024:
            bucket["small_count_le_1024"] += 1
        if size > (1 << 20):
            bucket["large_count_gt_1m"] += 1
        if size <= int(small_kernel_max_size):
            repeated_small[(type_name, size, int(record["operation_count"]), str(record["hash"]))] += 1

    top_by_execution = sorted(
        records,
        key=lambda item: (float(item["execution_time_ms"]), int(item["size"])),
        reverse=True,
    )[: int(top_k)]
    top_by_size = sorted(
        records,
        key=lambda item: (int(item["size"]), float(item["execution_time_ms"])),
        reverse=True,
    )[: int(top_k)]
    frequent_small_kernels = []
    for (type_name, size, operation_count, hash_value), count in repeated_small.most_common():
        if int(count) < int(small_kernel_min_count):
            continue
        frequent_small_kernels.append(
            {
                "type": str(type_name),
                "size": int(size),
                "operation_count": int(operation_count),
                "hash": str(hash_value),
                "count": int(count),
            }
        )
        if len(frequent_small_kernels) >= int(top_k):
            break

    return {
        "total_count": int(len(records)),
        "by_type": by_type,
        "top_by_execution_ms": top_by_execution,
        "top_by_size": top_by_size,
        "frequent_small_kernels": frequent_small_kernels,
    }


def _sync_drjit() -> None:
    dr.sync_thread()


def _reset_drjit_memory_stats() -> None:
    detail = getattr(dr, "detail", None)
    if detail is not None and hasattr(detail, "malloc_clear_statistics"):
        detail.malloc_clear_statistics()


def _flush_drjit_caches() -> None:
    _sync_drjit()
    if hasattr(dr, "flush_malloc_cache"):
        dr.flush_malloc_cache()
    _sync_drjit()


def _query_process_gpu_memory_mib(*, pid: int | None = None) -> float | None:
    resolved_pid = os.getpid() if pid is None else int(pid)
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
    except Exception:
        return None

    peak_mib = None
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            used_mib = float(parts[1])
        except ValueError:
            continue
        if row_pid != resolved_pid:
            continue
        peak_mib = used_mib if peak_mib is None else max(peak_mib, used_mib)
    return peak_mib


def _run_with_process_gpu_peak(operation, *, sync_result=None):
    peak_mib = _query_process_gpu_memory_mib()
    stop = threading.Event()

    def _poll():
        nonlocal peak_mib
        while not stop.wait(0.02):
            current = _query_process_gpu_memory_mib()
            if current is None:
                continue
            peak_mib = current if peak_mib is None else max(peak_mib, current)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    try:
        result = operation()
        if sync_result is not None:
            sync_result(result)
    finally:
        stop.set()
        thread.join(timeout=0.5)
        current = _query_process_gpu_memory_mib()
        if current is not None:
            peak_mib = current if peak_mib is None else max(peak_mib, current)
    return result, peak_mib


def _sync_witwin_result(result) -> None:
    dr.eval(result.path_gain)
    _sync_drjit()


def _sync_sionna_result(result) -> None:
    _ = np.asarray(result.path_gain, dtype=np.float64)
    _sync_drjit()


def _force_monte_carlo_backend_contract() -> None:
    package_rm_integrator.NativeExtension.native_extension_available = lambda: True


def _origin_box_mesh() -> tuple[wt.Point3f, wt.Vector3u]:
    vertices, faces = Box(
        position=(0.0, 0.0, 0.0),
        size=(DEFAULT_CUBE_SIZE, DEFAULT_CUBE_SIZE, DEFAULT_CUBE_SIZE),
        device="cuda",
    ).to_mesh()
    return to_point3f(vertices), to_vector3u(faces)


def _translate_vertices(vertices: wt.Point3f, center) -> wt.Point3f:
    center = wt.Point3f(*center) if not hasattr(center, "x") else center
    return wt.Point3f(
        vertices.x + center.x,
        vertices.y + center.y,
        vertices.z + center.z,
    )


def _grid_extent(bounds):
    return (
        float(bounds[0][0]),
        float(bounds[0][1]),
        float(bounds[1][0]),
        float(bounds[1][1]),
    )


def _finite_difference(forward_plus: np.ndarray, forward_minus: np.ndarray, step: float) -> np.ndarray:
    return (forward_plus - forward_minus) / (2.0 * float(step))


def _scalar_grad(value) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _profile_kernel_history(label: str, operation, *, warmup: int = 1) -> KernelHistorySnapshot:
    _flush_drjit_caches()
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        for _ in range(max(1, int(warmup))):
            operation()
            _sync_drjit()
        dr.kernel_history_clear()
        _reset_drjit_memory_stats()
        memory_before = _capture_drjit_allocator()
        _, process_gpu_peak_mib = _run_with_process_gpu_peak(
            operation,
            sync_result=lambda _: _sync_drjit(),
        )
        records = list(dr.kernel_history())
    memory_after = _capture_drjit_allocator()
    return KernelHistorySnapshot(
        label=label,
        records=records,
        summary=_summarize_kernel_history(records),
        memory_before=memory_before,
        memory_after=memory_after,
        process_gpu_peak_mib=process_gpu_peak_mib,
    )


def _timed_benchmark(label: str, operation, *, sync_result, warmup: int = 1, repeats: int = 3) -> PerformanceSnapshot:
    _flush_drjit_caches()
    for _ in range(int(warmup)):
        result = operation()
        sync_result(result)

    samples_ms: list[float] = []
    peak_process_gpu_mib = None
    peak_allocator_bytes = None
    peak_allocator = {}
    for _ in range(int(repeats)):
        _reset_drjit_memory_stats()
        start = time.perf_counter()
        _, run_peak_process_mib = _run_with_process_gpu_peak(operation, sync_result=sync_result)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples_ms.append(float(elapsed_ms))
        allocator = _capture_drjit_allocator()
        allocator_peak_bytes = allocator.get("device_peak_bytes")
        if allocator_peak_bytes is not None and (
            peak_allocator_bytes is None or int(allocator_peak_bytes) > int(peak_allocator_bytes)
        ):
            peak_allocator_bytes = int(allocator_peak_bytes)
            peak_allocator = allocator
        elif not peak_allocator and allocator:
            peak_allocator = allocator
        if run_peak_process_mib is not None:
            peak_process_gpu_mib = (
                float(run_peak_process_mib)
                if peak_process_gpu_mib is None
                else max(float(peak_process_gpu_mib), float(run_peak_process_mib))
            )

    return PerformanceSnapshot(
        label=label,
        samples_ms=tuple(samples_ms),
        median_ms=float(np.median(samples_ms)),
        mean_ms=float(np.mean(samples_ms)),
        min_ms=float(np.min(samples_ms)),
        max_ms=float(np.max(samples_ms)),
        drjit_allocator=peak_allocator,
        process_gpu_peak_mib=peak_process_gpu_mib,
    )


def _path_gain_db(values: np.ndarray, floor: float = 1.0e-20) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(values, dtype=np.float64), float(floor)))


def _signed_gradient_db(
    values: np.ndarray,
    *,
    floor_db: float = DEFAULT_GRADIENT_DB_FLOOR,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    floor_linear = 10.0 ** (float(floor_db) / 10.0)
    magnitude_db = 10.0 * np.log10(np.maximum(np.abs(array), floor_linear))
    encoded = np.sign(array) * np.maximum(magnitude_db - float(floor_db), 0.0)
    encoded = np.where(array == 0.0, 0.0, encoded)
    return encoded


def _gradient_display_limit(*value_groups: np.ndarray, floor_db: float = DEFAULT_GRADIENT_DB_FLOOR) -> float:
    encoded = [
        np.abs(_signed_gradient_db(values, floor_db=floor_db)).reshape(-1)
        for values in value_groups
    ]
    if not encoded:
        return 1.0
    all_values = np.concatenate(encoded)
    vmax = float(np.nanpercentile(all_values, 99.0))
    return max(vmax, 1.0)


def _flatten_axes(axes) -> np.ndarray:
    return np.asarray(axes, dtype=object).reshape(-1)


class ThreeCubeExperiment:
    def __init__(
        self,
        *,
        bounds=DEFAULT_BOUNDS,
        grid_shape=DEFAULT_GRID_SHAPE,
        plane_z: float = DEFAULT_PLANE_Z,
        tx_pos=DEFAULT_TX_POS,
        forward_num_samples: int = DEFAULT_FORWARD_REFLECTION_N_RAYS,
        gradient_num_samples: int = DEFAULT_GRADIENT_REFLECTION_N_RAYS,
        samples_per_tx: int = DEFAULT_SAMPLES_PER_TX,
        seed: int = 7,
    ) -> None:
        _force_monte_carlo_backend_contract()
        self.bounds = bounds
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.plane_z = float(plane_z)
        self.tx_pos = tuple(float(value) for value in tx_pos)
        self.base_centers = tuple(tuple(float(value) for value in center) for center in DEFAULT_CUBE_CENTERS)
        self.base_box_vertices, self.base_box_faces = _origin_box_mesh()
        self.grid = ReceiverGrid(
            "rm",
            axis="z",
            position=self.plane_z,
            bounds=self.bounds,
            grid_shape=self.grid_shape,
        )
        self.base_channel_scene = self._build_channel_scene()
        self.forward_config = Config(
            num_samples=int(forward_num_samples),
            max_bounces=3,
            max_diffraction_order=1,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(enable_rd_diffraction=True),
            integrator_options=IntegratorOptions(
                integrator="basic",
                samples_per_tx=int(samples_per_tx),
                seed=int(seed),
                accumulation_backend="auto",
            ),
        )
        self.gradient_config = Config(
            num_samples=int(gradient_num_samples),
            max_bounces=3,
            max_diffraction_order=1,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(enable_rd_diffraction=True, shadow_boundary_mode="none"),
            integrator_options=IntegratorOptions(
                integrator="basic",
                samples_per_tx=int(samples_per_tx),
                seed=int(seed),
                accumulation_backend="auto",
            ),
        )

    def _cube_center(self, index: int, cube1_x=None):
        center = self.base_centers[index]
        if index != 0 or cube1_x is None:
            return wt.Point3f(*center)
        return wt.Point3f(cube1_x, center[1], center[2])

    def _cube_mesh(self, index: int, *, cube1_x=None) -> ChannelMesh:
        return ChannelMesh(
            vertices=_translate_vertices(self.base_box_vertices, self._cube_center(index, cube1_x=cube1_x)),
            faces=self.base_box_faces,
        )

    def _build_channel_scene(self, *, cube1_x=None) -> ChannelScene:
        material = Material(eps_r=DEFAULT_RELATIVE_PERMITTIVITY, sigma_e=0.0)
        structures = [
            Structure(
                name=f"cube_{index}",
                geometry=self._cube_mesh(index, cube1_x=cube1_x),
                material=material,
            )
            for index, _center in enumerate(self.base_centers)
        ]
        return ChannelScene(
            structures=structures,
            transmitters=[
                Transmitter("tx", wt.Point3f(self.tx_pos[0], self.tx_pos[1], self.tx_pos[2])),
            ],
            receivers=[
                self.grid,
            ],
            frequency=DEFAULT_FREQUENCY_HZ,
            device="cuda",
        )

    def reset_geometry(self) -> None:
        return None

    def _solve(self, *, tx_x=None, cube1_x=None, config: Config | None = None):
        resolved_tx_x = wt.Float(self.tx_pos[0]) if tx_x is None else tx_x
        resolved_scene = (
            self.base_channel_scene
            if cube1_x is None
            else self._build_channel_scene(cube1_x=cube1_x)
        )
        result = solve(
            scene=resolved_scene,
            transmitter=Transmitter("tx", wt.Point3f(resolved_tx_x, self.tx_pos[1], self.tx_pos[2])),
            receiver="rm",
            config=self.forward_config if config is None else config,
        )
        return result

    @staticmethod
    def _snapshot(result) -> SolveSnapshot:
        result = result.squeeze_tx(0)
        path_gain = np.asarray(result.path_gain, dtype=np.float64)
        shadow_boundary_correction = (
            np.asarray(result.incoherent["shadow_boundary_correction"], dtype=np.float64)
            if "shadow_boundary_correction" in result.incoherent
            else np.zeros_like(path_gain)
        )
        return SolveSnapshot(
            path_gain=path_gain,
            coords_x=np.asarray(result.coords.grid_x, dtype=np.float64),
            coords_y=np.asarray(result.coords.grid_y, dtype=np.float64),
            metadata=dict(result.metadata),
            components={
                "los": np.asarray(result.incoherent["los"], dtype=np.float64),
                "reflection": np.asarray(result.incoherent["reflection"], dtype=np.float64),
                "diffraction": np.asarray(result.incoherent["diffraction"], dtype=np.float64),
                "shadow_boundary_correction": shadow_boundary_correction,
            },
        )

    def forward(self) -> SolveSnapshot:
        self.reset_geometry()
        result = self._solve(config=self.forward_config)
        return self._snapshot(result)

    def gradient(self, parameter: str, *, fd_step: float = DEFAULT_FD_STEP) -> GradientSnapshot:
        if parameter not in {"tx_x", "cube1_x"}:
            raise ValueError("parameter must be 'tx_x' or 'cube1_x'.")

        _clear_drjit_ad_state()

        def _solve_with_grad_parameter():
            if parameter == "tx_x":
                variable = wt.Float(self.tx_pos[0])
                dr.enable_grad(variable)
                return self._solve(tx_x=variable, config=self.gradient_config), variable
            variable = wt.Float(self.base_centers[0][0])
            dr.enable_grad(variable)
            return self._solve(cube1_x=variable, config=self.gradient_config), variable

        def _solve_fd(offset: float):
            if parameter == "tx_x":
                return self._solve(
                    tx_x=wt.Float(self.tx_pos[0] + float(offset)),
                    config=self.gradient_config,
                )
            return self._solve(
                cube1_x=wt.Float(self.base_centers[0][0] + float(offset)),
                config=self.gradient_config,
            )

        self.reset_geometry()
        result, parameter_value = _solve_with_grad_parameter()
        dr.set_grad(parameter_value, 1.0)
        jvp_np = np.asarray(
            dr.forward_to(
                result.path_gain,
                flags=_GRAD_FLAGS,
            ),
            dtype=np.float64,
        )[0]

        self.reset_geometry()
        component_result, component_parameter_value = _solve_with_grad_parameter()
        dr.set_grad(component_parameter_value, 1.0)
        component_jvp_values = dr.forward_to(
            component_result.incoherent["los"],
            component_result.incoherent["reflection"],
            component_result.incoherent["diffraction"],
            flags=_GRAD_FLAGS,
        )
        if not isinstance(component_jvp_values, tuple):
            component_jvp_values = (component_jvp_values,)

        plus = _solve_fd(fd_step)
        minus = _solve_fd(-fd_step)

        forward = self._snapshot(result)
        component_jvp = {
            name: np.asarray(values, dtype=np.float64)[0]
            for name, values in zip(_GRADIENT_COMPONENTS, component_jvp_values, strict=True)
        }
        _clear_drjit_ad_state()

        fd_np = _finite_difference(
            np.asarray(plus.path_gain, dtype=np.float64)[0],
            np.asarray(minus.path_gain, dtype=np.float64)[0],
            fd_step,
        )
        component_fd = {
            name: _finite_difference(
                np.asarray(plus.incoherent[name], dtype=np.float64)[0],
                np.asarray(minus.incoherent[name], dtype=np.float64)[0],
                fd_step,
            )
            for name in _GRADIENT_COMPONENTS
        }
        self.reset_geometry()
        return GradientSnapshot(
            parameter=parameter,
            forward=forward,
            jvp=jvp_np,
            fd=fd_np,
            delta=jvp_np - fd_np,
            component_jvp=component_jvp,
            component_fd=component_fd,
            component_delta={
                name: component_jvp[name] - component_fd[name]
                for name in _GRADIENT_COMPONENTS
            },
        )

    def backward(self, parameter: str, *, fd_step: float = DEFAULT_FD_STEP) -> BackwardSnapshot:
        if parameter not in {"tx_x", "cube1_x"}:
            raise ValueError("parameter must be 'tx_x' or 'cube1_x'.")

        gradient = self.gradient(parameter, fd_step=fd_step)

        _clear_drjit_ad_state()
        variable, loss = self.scalar_loss(parameter=parameter)
        dr.backward(loss, flags=_GRAD_FLAGS)
        vjp = _scalar_grad(dr.grad(variable))
        _clear_drjit_ad_state()

        jvp_sum = float(np.sum(gradient.jvp))
        fd_sum = float(np.sum(gradient.fd))
        return BackwardSnapshot(
            parameter=parameter,
            vjp=vjp,
            jvp_sum=jvp_sum,
            fd_sum=fd_sum,
            vjp_delta=vjp - jvp_sum,
            fd_delta=vjp - fd_sum,
        )

    def scalar_loss(self, *, parameter: str):
        if parameter == "tx_x":
            variable = wt.Float(self.tx_pos[0])
            dr.enable_grad(variable)
            result = self._solve(tx_x=variable, config=self.gradient_config)
        elif parameter == "cube1_x":
            variable = wt.Float(self.base_centers[0][0])
            dr.enable_grad(variable)
            result = self._solve(cube1_x=variable, config=self.gradient_config)
        else:
            raise ValueError("parameter must be 'tx_x' or 'cube1_x'.")
        loss = dr.sum(result.path_gain)
        dr.eval(loss)
        dr.sync_thread()
        return variable, loss

    def _prepare_sionna_context(self):
        rt = SionnaAdaptor.load_rt(prefer_local=True)
        mi = importlib.import_module("mitsuba")
        sionna_scene = self.base_channel_scene.to_sionna(prefer_local=True)
        sionna_scene.frequency = DEFAULT_FREQUENCY_HZ
        sionna_scene.tx_array = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            pattern="iso",
            polarization="V",
        )
        sionna_scene.add(rt.Transmitter("tx", position=mi.Point3f(*self.tx_pos), power_dbm=0.0))
        solver = rt.RadioMapSolver()
        span_x = float(self.bounds[0][1] - self.bounds[0][0])
        span_y = float(self.bounds[1][1] - self.bounds[1][0])
        solver_kwargs = {
            "center": mi.Point3f(
                0.5 * (float(self.bounds[0][0]) + float(self.bounds[0][1])),
                0.5 * (float(self.bounds[1][0]) + float(self.bounds[1][1])),
                self.plane_z,
            ),
            "orientation": mi.Point3f(0.0, 0.0, 0.0),
            "size": mi.Point2f(span_x, span_y),
            "cell_size": mi.Point2f(span_x / float(self.grid_shape[0]), span_y / float(self.grid_shape[1])),
            "samples_per_tx": int(self.forward_config.integrator_options.samples_per_tx),
            "max_depth": DEFAULT_SIONNA_MAX_DEPTH,
            "los": True,
            "specular_reflection": True,
            "diffraction": True,
            "edge_diffraction": True,
            "refraction": False,
            "seed": int(self.forward_config.integrator_options.seed),
        }
        return solver, sionna_scene, solver_kwargs

    def compare_sionna(self) -> SionnaComparison:
        self.reset_geometry()
        solver, sionna_scene, solver_kwargs = self._prepare_sionna_context()
        sionna_result = solver(sionna_scene, **solver_kwargs)
        forward = self.forward()
        sionna_map = np.asarray(sionna_result.path_gain, dtype=np.float64)[0]
        return SionnaComparison(
            standalone=forward.path_gain,
            sionna=sionna_map,
            delta=forward.path_gain - sionna_map,
            standalone_metadata=forward.metadata,
        )

    def benchmark_forward_vs_sionna(self, *, warmup: int = 1, repeats: int = 3) -> PerformanceComparison:
        solver, sionna_scene, solver_kwargs = self._prepare_sionna_context()

        def _witwin_operation():
            self.reset_geometry()
            return self._solve(config=self.forward_config)

        witwin = _timed_benchmark(
            "witwin",
            _witwin_operation,
            sync_result=_sync_witwin_result,
            warmup=warmup,
            repeats=repeats,
        )
        sionna = _timed_benchmark(
            "sionna",
            lambda: solver(sionna_scene, **solver_kwargs),
            sync_result=_sync_sionna_result,
            warmup=warmup,
            repeats=repeats,
        )
        return PerformanceComparison(witwin=witwin, sionna=sionna)

    def forward_kernel_history(self) -> KernelHistorySnapshot:
        self.reset_geometry()

        def _operation():
            result = self._solve(config=self.forward_config)
            dr.eval(result.path_gain)

        return _profile_kernel_history("forward", _operation)

    def backward_kernel_history(self, *, parameter: str = "tx_x") -> KernelHistorySnapshot:
        if parameter == "tx_x":
            self.reset_geometry()

        def _operation():
            variable, loss = self.scalar_loss(parameter=parameter)
            dr.backward(loss, flags=_GRAD_FLAGS)
            dr.eval(dr.grad(variable))

        return _profile_kernel_history(f"backward_{parameter}", _operation)


def plot_forward(snapshot: SolveSnapshot, *, bounds=DEFAULT_BOUNDS, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.0))
    image = ax.imshow(
        _path_gain_db(snapshot.path_gain),
        origin="lower",
        extent=_grid_extent(bounds),
        cmap="viridis",
        interpolation="nearest",
        vmin=DEFAULT_DB_MIN,
        vmax=DEFAULT_DB_MAX,
    )
    ax.set_title("Monte Carlo Radiomap Forward (dB)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    return ax


def plot_forward_components(snapshot: SolveSnapshot, *, bounds=DEFAULT_BOUNDS, axes=None):
    panel_specs = [
        ("los", "LoS (dB)", "power"),
        ("reflection", "Reflection (dB)", "power"),
        ("diffraction", "Diffraction (dB)", "power"),
    ]
    if "shadow_boundary_correction" in snapshot.components:
        panel_specs.append(
            (
                "shadow_boundary_correction",
                "Shadow Boundary Correction (signed dB)",
                "signed",
            )
        )
    if axes is None:
        _, axes = plt.subplots(
            1,
            len(panel_specs),
            figsize=(5.0 * len(panel_specs), 4.5),
            constrained_layout=True,
        )
    axes_flat = _flatten_axes(axes)
    active_specs = panel_specs[: len(axes_flat)]
    for ax, (component_name, title, scale) in zip(
        axes_flat[: len(active_specs)],
        active_specs,
        strict=True,
    ):
        values = snapshot.components[component_name]
        if scale == "signed":
            encoded = _signed_gradient_db(values, floor_db=DEFAULT_GRADIENT_DB_FLOOR)
            vmax = _gradient_display_limit(values, floor_db=DEFAULT_GRADIENT_DB_FLOOR)
            image = ax.imshow(
                encoded,
                origin="lower",
                extent=_grid_extent(bounds),
                cmap="RdBu_r",
                interpolation="nearest",
                vmin=-vmax,
                vmax=vmax,
            )
            colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            colorbar.set_label(f"Signed dB above {float(DEFAULT_GRADIENT_DB_FLOOR):.0f} dB")
        else:
            image = ax.imshow(
                _path_gain_db(values),
                origin="lower",
                extent=_grid_extent(bounds),
                cmap="viridis",
                interpolation="nearest",
                vmin=DEFAULT_DB_MIN,
                vmax=DEFAULT_DB_MAX,
            )
            plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    return axes


def plot_shadow_boundary(
    snapshot: SolveSnapshot,
    *,
    bounds=DEFAULT_BOUNDS,
    ax=None,
    floor_db: float = DEFAULT_GRADIENT_DB_FLOOR,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.0))
    values = snapshot.components.get(
        "shadow_boundary_correction",
        np.zeros_like(snapshot.path_gain),
    )
    encoded = _signed_gradient_db(values, floor_db=floor_db)
    vmax = _gradient_display_limit(values, floor_db=floor_db)
    image = ax.imshow(
        encoded,
        origin="lower",
        extent=_grid_extent(bounds),
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title("Shadow Boundary Correction (signed dB)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(f"Signed dB above {float(floor_db):.0f} dB")
    return ax


def plot_gradient(
    snapshot: GradientSnapshot,
    *,
    bounds=DEFAULT_BOUNDS,
    axes=None,
    floor_db: float = DEFAULT_GRADIENT_DB_FLOOR,
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    encoded_jvp = _signed_gradient_db(snapshot.jvp, floor_db=floor_db)
    encoded_fd = _signed_gradient_db(snapshot.fd, floor_db=floor_db)
    encoded_delta = _signed_gradient_db(snapshot.delta, floor_db=floor_db)
    vmax = _gradient_display_limit(
        snapshot.jvp,
        snapshot.fd,
        snapshot.delta,
        floor_db=floor_db,
    )
    panels = (
        (axes[0], encoded_jvp, f"JVP d path_gain / d {snapshot.parameter} (signed dB)"),
        (axes[1], encoded_fd, f"FD d path_gain / d {snapshot.parameter} (signed dB)"),
        (axes[2], encoded_delta, f"JVP - FD ({snapshot.parameter}, signed dB)"),
    )
    for ax, values, title in panels:
        image = ax.imshow(
            values,
            origin="lower",
            extent=_grid_extent(bounds),
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(f"Signed dB above {float(floor_db):.0f} dB")
    return axes


def plot_gradient_components(
    snapshot: GradientSnapshot,
    *,
    bounds=DEFAULT_BOUNDS,
    axes=None,
    floor_db: float = DEFAULT_GRADIENT_DB_FLOOR,
):
    if axes is None:
        _, axes = plt.subplots(3, 3, figsize=(15.0, 12.0), constrained_layout=True)
    vmax = _gradient_display_limit(
        *(snapshot.component_jvp[name] for name in _GRADIENT_COMPONENTS),
        *(snapshot.component_fd[name] for name in _GRADIENT_COMPONENTS),
        *(snapshot.component_delta[name] for name in _GRADIENT_COMPONENTS),
        floor_db=floor_db,
    )
    column_titles = ("JVP", "FD", "JVP - FD")
    row_titles = {
        "los": "LoS",
        "reflection": "Reflection",
        "diffraction": "Diffraction",
    }
    for row, component in enumerate(_GRADIENT_COMPONENTS):
        encoded_values = (
            _signed_gradient_db(snapshot.component_jvp[component], floor_db=floor_db),
            _signed_gradient_db(snapshot.component_fd[component], floor_db=floor_db),
            _signed_gradient_db(snapshot.component_delta[component], floor_db=floor_db),
        )
        for col, values in enumerate(encoded_values):
            ax = axes[row][col]
            image = ax.imshow(
                values,
                origin="lower",
                extent=_grid_extent(bounds),
                cmap="RdBu_r",
                interpolation="nearest",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_title(f"{row_titles[component]} {column_titles[col]} (signed dB)")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            colorbar.set_label(f"Signed dB above {float(floor_db):.0f} dB")
    return axes


def plot_sionna_comparison(comparison: SionnaComparison, *, bounds=DEFAULT_BOUNDS, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    standalone_db = _path_gain_db(comparison.standalone)
    sionna_db = _path_gain_db(comparison.sionna)
    delta_db = standalone_db - sionna_db
    delta_vmax = float(np.nanpercentile(np.abs(delta_db), 99.0))
    delta_vmax = max(delta_vmax, 1.0e-12)
    panels = (
        (axes[0], standalone_db, "Standalone Monte Carlo (dB)", "viridis", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[1], sionna_db, "Sionna RT (dB)", "viridis", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[2], delta_db, "Standalone - Sionna (dB)", "RdBu_r", -delta_vmax, delta_vmax),
    )
    for ax, values, title, cmap, vmin, vmax in panels:
        image = ax.imshow(
            values,
            origin="lower",
            extent=_grid_extent(bounds),
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    return axes


def plot_performance_comparison(comparison: PerformanceComparison, *, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    labels = ["witwin", "sionna"]
    timing_ms = [comparison.witwin.median_ms, comparison.sionna.median_ms]
    memory_mib = [
        np.nan if comparison.witwin.peak_memory_mib is None else comparison.witwin.peak_memory_mib,
        np.nan if comparison.sionna.peak_memory_mib is None else comparison.sionna.peak_memory_mib,
    ]
    colors = ["#2A6F97", "#C97B63"]

    axes[0].bar(labels, timing_ms, color=colors)
    axes[0].set_title("Forward Runtime")
    axes[0].set_ylabel("Median time (ms)")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, memory_mib, color=colors)
    axes[1].set_title("GPU Memory")
    axes[1].set_ylabel("Peak memory (MiB)")
    axes[1].grid(axis="y", alpha=0.25)

    return axes


__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_DB_MAX",
    "DEFAULT_DB_MIN",
    "DEFAULT_FD_STEP",
    "DEFAULT_FORWARD_REFLECTION_N_RAYS",
    "DEFAULT_GRADIENT_DB_FLOOR",
    "DEFAULT_GRADIENT_REFLECTION_N_RAYS",
    "DEFAULT_GRID_SHAPE",
    "BackwardSnapshot",
    "GradientSnapshot",
    "KernelHistorySnapshot",
    "PerformanceComparison",
    "PerformanceSnapshot",
    "SolveSnapshot",
    "SionnaComparison",
    "ThreeCubeExperiment",
    "plot_forward",
    "plot_forward_components",
    "plot_shadow_boundary",
    "plot_gradient",
    "plot_gradient_components",
    "plot_performance_comparison",
    "plot_sionna_comparison",
]
