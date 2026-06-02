from __future__ import annotations

from dataclasses import dataclass, replace
import gc
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import witwin.channel as wt

from witwin.channel.core.scene import Mesh as ChannelMesh
from witwin.channel.core.scene import EdgePolicy, ReceiverGrid, Scene as ChannelScene, Transmitter
from witwin.channel.core.geometry.mesh_buffers import to_point3f, to_vector3u
from witwin.core import Box, Material, Structure
from witwin.channel.deterministic import (
    Config,
    Tuning,
    solve,
)


DEFAULT_BOUNDS = ((-10.0, 10.0), (-10.0, 10.0))
DEFAULT_GRID_SHAPE = (128, 128)
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
DEFAULT_FORWARD_NUM_SAMPLES = 384
DEFAULT_GRADIENT_NUM_SAMPLES = 512
DEFAULT_MAX_BOUNCES = 3
DEFAULT_MAX_DIFFRACTION_ORDER = 1
DEFAULT_SHADOW_BOUNDARY_CORRECTION = False
DEFAULT_SHADOW_SUPPORT_CUTOFF_DB = 25.0
DEFAULT_FD_STEP = 1.0e-3
DEFAULT_DB_MIN = -90.0
DEFAULT_DB_MAX = -40.0
DEFAULT_GRADIENT_DB_FLOOR = -160.0
_GRAD_FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad
_GRADIENT_COMPONENTS = ("los", "reflection", "diffraction")


def _clear_drjit_ad_state() -> None:
    try:
        dr.clear_grad()
    except TypeError:
        pass
    dr.sync_thread()
    gc.collect()


@dataclass(frozen=True, slots=True)
class SolveSnapshot:
    path_gain: np.ndarray
    coords_x: np.ndarray
    coords_y: np.ndarray
    component_space: str
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
class ShadowBoundaryCorrectionComparison:
    with_correction: SolveSnapshot
    without_correction: SolveSnapshot


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


def _map2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    return array


def _path_gain_db(values: np.ndarray, floor: float = 1.0e-20) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(_map2d(values).astype(np.float64, copy=False), float(floor)))


def _signed_gradient_db(
    values: np.ndarray,
    *,
    floor_db: float = DEFAULT_GRADIENT_DB_FLOOR,
) -> np.ndarray:
    array = _map2d(values).astype(np.float64, copy=False)
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


def _component_space_for_config(config: Config) -> str:
    del config
    return "components"


def _component_arrays(result) -> tuple[str, dict[str, np.ndarray]]:
    component_space = "components"
    metrics = result.components
    return (
        component_space,
        {
            name: np.asarray(metrics[name], dtype=np.float64)
            for name in _GRADIENT_COMPONENTS
        },
    )


def _component_label_prefix(component_space: str) -> str:
    del component_space
    return "Component Power"


class ThreeCubeExperiment:
    def __init__(
        self,
        *,
        bounds=DEFAULT_BOUNDS,
        grid_shape=DEFAULT_GRID_SHAPE,
        plane_z: float = DEFAULT_PLANE_Z,
        tx_pos=DEFAULT_TX_POS,
        forward_num_samples: int = DEFAULT_FORWARD_NUM_SAMPLES,
        gradient_num_samples: int = DEFAULT_GRADIENT_NUM_SAMPLES,
        max_bounces: int = DEFAULT_MAX_BOUNCES,
        max_diffraction_order: int = DEFAULT_MAX_DIFFRACTION_ORDER,
        shadow_boundary_correction: bool = DEFAULT_SHADOW_BOUNDARY_CORRECTION,
        shadow_support_cutoff_db: float | None = DEFAULT_SHADOW_SUPPORT_CUTOFF_DB,
        seed: int = 7,
    ) -> None:
        self.bounds = bounds
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.plane_z = float(plane_z)
        self.tx_pos = tuple(float(value) for value in tx_pos)
        self.max_bounces = int(max_bounces)
        self.max_diffraction_order = int(max_diffraction_order)
        self.shadow_boundary_correction = bool(shadow_boundary_correction)
        self.seed = int(seed)
        self.shadow_support_cutoff_db = (
            None if shadow_support_cutoff_db is None else float(shadow_support_cutoff_db)
        )
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
            max_bounces=self.max_bounces,
            shadow_boundary_correction=self.shadow_boundary_correction,
            max_diffraction_order=self.max_diffraction_order,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(
                shadow_support_cutoff_db=self.shadow_support_cutoff_db,
                enable_rd_diffraction=True,
            ),
        )
        self.gradient_config = Config(
            num_samples=int(gradient_num_samples),
            max_bounces=self.max_bounces,
            shadow_boundary_correction=self.shadow_boundary_correction,
            max_diffraction_order=self.max_diffraction_order,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(
                shadow_support_cutoff_db=self.shadow_support_cutoff_db,
                enable_rd_diffraction=True,
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

    @staticmethod
    def _cube_name(index: int) -> str:
        return f"cube{index + 1}"

    def _build_channel_scene(self, *, cube1_x=None, cube1_eps=None) -> ChannelScene:
        material = Material(eps_r=DEFAULT_RELATIVE_PERMITTIVITY, sigma_e=0.0)
        structures = [
            Structure(
                name=self._cube_name(index),
                geometry=self._cube_mesh(index, cube1_x=cube1_x),
                material=material,
            )
            for index, _center in enumerate(self.base_centers)
        ]
        scene = ChannelScene(
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
        if cube1_eps is not None:
            scene.structure("cube1").set_material_parameters(eps_r=cube1_eps)
        return scene

    def reset_geometry(self) -> None:
        return None

    def _solve(self, *, tx_x=None, cube1_x=None, cube1_eps=None, config: Config | None = None):
        resolved_tx_x = wt.Float(self.tx_pos[0]) if tx_x is None else tx_x
        resolved_scene = (
            self.base_channel_scene
            if cube1_x is None and cube1_eps is None
            else self._build_channel_scene(cube1_x=cube1_x, cube1_eps=cube1_eps)
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
        component_space, components = _component_arrays(result)
        return SolveSnapshot(
            path_gain=np.asarray(result.path_gain, dtype=np.float64),
            coords_x=np.asarray(result.coords.grid_x, dtype=np.float64),
            coords_y=np.asarray(result.coords.grid_y, dtype=np.float64),
            component_space=component_space,
            components=components,
        )

    def forward(self) -> SolveSnapshot:
        self.reset_geometry()
        result = self._solve(config=self.forward_config)
        return self._snapshot(result)

    def forward_with_shadow_boundary_correction(self, enabled: bool) -> SolveSnapshot:
        self.reset_geometry()
        result = self._solve(
            config=replace(
                self.forward_config,
                shadow_boundary_correction=bool(enabled),
            )
        )
        return self._snapshot(result)

    def shadow_boundary_correction_comparison(self) -> ShadowBoundaryCorrectionComparison:
        return ShadowBoundaryCorrectionComparison(
            with_correction=self.forward_with_shadow_boundary_correction(True),
            without_correction=self.forward_with_shadow_boundary_correction(False),
        )

    def gradient(self, parameter: str, *, fd_step: float = DEFAULT_FD_STEP) -> GradientSnapshot:
        if parameter not in {"tx_x", "cube1_x", "cube1_eps"}:
            raise ValueError("parameter must be 'tx_x', 'cube1_x', or 'cube1_eps'.")

        _clear_drjit_ad_state()

        def _solve_with_grad_parameter():
            if parameter == "tx_x":
                variable = wt.Float(self.tx_pos[0])
                dr.enable_grad(variable)
                return self._solve(tx_x=variable, config=self.gradient_config), variable
            if parameter == "cube1_x":
                variable = wt.Float(self.base_centers[0][0])
                dr.enable_grad(variable)
                return self._solve(cube1_x=variable, config=self.gradient_config), variable
            variable = wt.Float(DEFAULT_RELATIVE_PERMITTIVITY)
            dr.enable_grad(variable)
            return self._solve(cube1_eps=variable, config=self.gradient_config), variable

        def _solve_fd(offset: float):
            if parameter == "tx_x":
                return self._solve(
                    tx_x=wt.Float(self.tx_pos[0] + float(offset)),
                    config=self.gradient_config,
                )
            if parameter == "cube1_x":
                return self._solve(
                    cube1_x=wt.Float(self.base_centers[0][0] + float(offset)),
                    config=self.gradient_config,
                )
            return self._solve(
                cube1_eps=wt.Float(DEFAULT_RELATIVE_PERMITTIVITY + float(offset)),
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
        )

        self.reset_geometry()
        component_result, component_parameter_value = _solve_with_grad_parameter()
        dr.set_grad(component_parameter_value, 1.0)
        component_space = _component_space_for_config(self.gradient_config)
        component_metrics = getattr(component_result, component_space)
        component_jvp_values = dr.forward_to(
            component_metrics["los"],
            component_metrics["reflection"],
            component_metrics["diffraction"],
            flags=_GRAD_FLAGS,
        )
        if not isinstance(component_jvp_values, tuple):
            component_jvp_values = (component_jvp_values,)

        component_jvp = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in zip(_GRADIENT_COMPONENTS, component_jvp_values, strict=True)
        }
        _clear_drjit_ad_state()

        plus = _solve_fd(fd_step)
        minus = _solve_fd(-fd_step)

        forward = self._snapshot(result)
        fd_np = _finite_difference(
            np.asarray(plus.path_gain, dtype=np.float64),
            np.asarray(minus.path_gain, dtype=np.float64),
            fd_step,
        )
        plus_component_metrics = getattr(plus, component_space)
        minus_component_metrics = getattr(minus, component_space)
        component_fd = {
            name: _finite_difference(
                np.asarray(plus_component_metrics[name], dtype=np.float64),
                np.asarray(minus_component_metrics[name], dtype=np.float64),
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
        if parameter not in {"tx_x", "cube1_x", "cube1_eps"}:
            raise ValueError("parameter must be 'tx_x', 'cube1_x', or 'cube1_eps'.")

        gradient = self.gradient(parameter, fd_step=fd_step)

        _clear_drjit_ad_state()
        if parameter == "tx_x":
            variable = wt.Float(self.tx_pos[0])
            dr.enable_grad(variable)
            result = self._solve(tx_x=variable, config=self.gradient_config)
        elif parameter == "cube1_x":
            variable = wt.Float(self.base_centers[0][0])
            dr.enable_grad(variable)
            result = self._solve(cube1_x=variable, config=self.gradient_config)
        else:
            variable = wt.Float(DEFAULT_RELATIVE_PERMITTIVITY)
            dr.enable_grad(variable)
            result = self._solve(cube1_eps=variable, config=self.gradient_config)

        loss = dr.sum(result.path_gain)
        dr.eval(loss)
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
    ax.set_title("Deterministic Radiomap Forward (dB)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    return ax


def plot_forward_components(snapshot: SolveSnapshot, *, bounds=DEFAULT_BOUNDS, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    prefix = _component_label_prefix(snapshot.component_space)
    panels = (
        (axes[0], snapshot.components["los"], f"LoS {prefix} (dB)"),
        (axes[1], snapshot.components["reflection"], f"Reflection {prefix} (dB)"),
        (axes[2], snapshot.components["diffraction"], f"Diffraction {prefix} (dB)"),
    )
    for ax, values, title in panels:
        image = ax.imshow(
            _path_gain_db(values),
            origin="lower",
            extent=_grid_extent(bounds),
            cmap="viridis",
            interpolation="nearest",
            vmin=DEFAULT_DB_MIN,
            vmax=DEFAULT_DB_MAX,
        )
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    return axes


def plot_shadow_boundary_correction_comparison(
    comparison: ShadowBoundaryCorrectionComparison,
    *,
    bounds=DEFAULT_BOUNDS,
    axes=None,
    component: str | None = None,
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    if component is None:
        with_values = comparison.with_correction.path_gain
        without_values = comparison.without_correction.path_gain
        label = "Path Gain"
    else:
        with_values = comparison.with_correction.components[component]
        without_values = comparison.without_correction.components[component]
        label = f"{component.title()} Component"
    with_db = _path_gain_db(with_values)
    without_db = _path_gain_db(without_values)
    delta_db = with_db - without_db
    delta_limit = max(float(np.nanpercentile(np.abs(delta_db), 99.0)), 1.0)
    panels = (
        (axes[0], with_db, f"{label} With Correction (dB)", "viridis", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[1], without_db, f"{label} Without Correction (dB)", "viridis", DEFAULT_DB_MIN, DEFAULT_DB_MAX),
        (axes[2], delta_db, f"{label} Delta With - Without (dB)", "RdBu_r", -delta_limit, delta_limit),
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
    prefix = _component_label_prefix(snapshot.forward.component_space)
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
            ax.set_title(f"{row_titles[component]} {prefix} {column_titles[col]} (signed dB)")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            colorbar.set_label(f"Signed dB above {float(floor_db):.0f} dB")
    return axes


def material_gradient_summary(snapshot: GradientSnapshot, backward: BackwardSnapshot | None = None) -> dict[str, float]:
    summary = {
        "total_l1": float(np.sum(np.abs(snapshot.jvp))),
        "diffraction_l1": float(np.sum(np.abs(snapshot.component_jvp["diffraction"]))),
    }
    if backward is not None:
        summary["backward_scalar_gradient"] = float(backward.vjp)
    return summary


__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_DB_MAX",
    "DEFAULT_DB_MIN",
    "DEFAULT_FD_STEP",
    "DEFAULT_FORWARD_NUM_SAMPLES",
    "DEFAULT_GRADIENT_DB_FLOOR",
    "DEFAULT_GRADIENT_NUM_SAMPLES",
    "DEFAULT_GRID_SHAPE",
    "BackwardSnapshot",
    "GradientSnapshot",
    "ShadowBoundaryCorrectionComparison",
    "SolveSnapshot",
    "ThreeCubeExperiment",
    "plot_shadow_boundary_correction_comparison",
    "plot_forward",
    "plot_forward_components",
    "plot_gradient",
    "plot_gradient_components",
    "material_gradient_summary",
]

