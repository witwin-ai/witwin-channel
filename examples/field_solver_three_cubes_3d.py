"""witwin.channel.deterministic.solve_field elevated-TX three-cube example."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
import sys

import drjit as dr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import witwin.channel as wt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from witwin.channel.core.scene import Mesh as ChannelMesh
from witwin.channel.core.scene import EdgePolicy, Scene as ChannelScene
from witwin.channel.core.geometry.mesh_buffers import to_point3f, to_vector3u
from witwin.core import Box, Material, Structure
from witwin.channel.deterministic import Config, FieldResult, FieldSpec, Tuning, solve_field


DEFAULT_BOUNDS = ((-10.0, 10.0), (-10.0, 10.0))
DEFAULT_GRID_SHAPE = (256, 256)
DEFAULT_SMOKE_GRID_SHAPE = (16, 16)
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
DEFAULT_GRADIENT_REFLECTION_N_RAYS = 128
DEFAULT_REFLECTION_MAX_BOUNCES = 3
DEFAULT_MAX_DIFFRACTION_ORDER = 1
DEFAULT_FD_STEP = 1.0e-3
DEFAULT_POWER_DB_MIN = -90.0
DEFAULT_POWER_DB_MAX = -40.0
DEFAULT_FIELD_DB_MIN = -120.0
DEFAULT_GRADIENT_DB_FLOOR = -160.0
_GRAD_FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad
_POWER_COMPONENTS = ("los", "reflection", "diffraction")


def _clear_drjit_ad_state() -> None:
    try:
        dr.clear_grad()
    except TypeError:
        pass
    dr.sync_thread()
    gc.collect()


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


def _power_db(values: np.ndarray, floor_db: float = DEFAULT_POWER_DB_MIN) -> np.ndarray:
    floor_linear = 10.0 ** (float(floor_db) / 10.0)
    return 10.0 * np.log10(np.maximum(np.asarray(values, dtype=np.float64), floor_linear))


def _field_magnitude_db(values: np.ndarray, floor_db: float = DEFAULT_FIELD_DB_MIN) -> np.ndarray:
    floor_linear = 10.0 ** (float(floor_db) / 20.0)
    return 20.0 * np.log10(np.maximum(np.abs(np.asarray(values)), floor_linear))


def _signed_gradient_db(
    values: np.ndarray,
    *,
    floor_db: float = DEFAULT_GRADIENT_DB_FLOOR,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    floor_linear = 10.0 ** (float(floor_db) / 10.0)
    magnitude_db = 10.0 * np.log10(np.maximum(np.abs(array), floor_linear))
    encoded = np.sign(array) * np.maximum(magnitude_db - float(floor_db), 0.0)
    return np.where(array == 0.0, 0.0, encoded)


def _gradient_display_limit(*value_groups: np.ndarray, floor_db: float = DEFAULT_GRADIENT_DB_FLOOR) -> float:
    encoded = [
        np.abs(_signed_gradient_db(values, floor_db=floor_db)).reshape(-1)
        for values in value_groups
    ]
    if not encoded:
        return 1.0
    return max(float(np.nanpercentile(np.concatenate(encoded), 99.0)), 1.0)


def _decorate_axis(ax, *, bounds=DEFAULT_BOUNDS, tx_pos=DEFAULT_TX_POS, cube1_x=None, title: str | None = None) -> None:
    for index, center in enumerate(DEFAULT_CUBE_CENTERS):
        cx, cy, _cz = center
        if index == 0 and cube1_x is not None:
            cx = float(cube1_x)
        size = DEFAULT_CUBE_SIZE
        ax.add_patch(
            Rectangle(
                (cx - size / 2.0, cy - size / 2.0),
                size,
                size,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
            )
        )
    ax.scatter(
        [float(tx_pos[0])],
        [float(tx_pos[1])],
        marker="*",
        s=70,
        color="gold",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title is not None:
        ax.set_title(title)


@dataclass(frozen=True, slots=True)
class FieldSnapshot:
    total_field: np.ndarray
    total_power: np.ndarray
    coords_x: np.ndarray
    coords_y: np.ndarray
    power_components: dict[str, np.ndarray]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class GradientSnapshot:
    parameter: str
    forward: FieldSnapshot
    ad: np.ndarray
    fd: np.ndarray
    delta: np.ndarray
    component_ad: dict[str, np.ndarray]
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


class ThreeCubeField3DExperiment:
    """Elevated-TX 3D field-solver experiment matching the radiomap scene."""

    def __init__(
        self,
        *,
        bounds=DEFAULT_BOUNDS,
        grid_shape=DEFAULT_GRID_SHAPE,
        plane_z: float = DEFAULT_PLANE_Z,
        tx_pos=DEFAULT_TX_POS,
        forward_num_samples: int = DEFAULT_FORWARD_REFLECTION_N_RAYS,
        gradient_num_samples: int = DEFAULT_GRADIENT_REFLECTION_N_RAYS,
        reflection_max_bounces: int = DEFAULT_REFLECTION_MAX_BOUNCES,
        max_diffraction_order: int = DEFAULT_MAX_DIFFRACTION_ORDER,
    ) -> None:
        self.bounds = bounds
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.plane_z = float(plane_z)
        self.tx_pos = tuple(float(value) for value in tx_pos)
        self.forward_num_samples = int(forward_num_samples)
        self.gradient_num_samples = int(gradient_num_samples)
        self.reflection_max_bounces = int(reflection_max_bounces)
        self.max_diffraction_order = int(max_diffraction_order)
        self.base_centers = tuple(tuple(float(value) for value in center) for center in DEFAULT_CUBE_CENTERS)
        self.base_box_vertices, self.base_box_faces = _origin_box_mesh()
        self.base_scene = self.build_scene()

    def field_spec(self, *, grid_shape: tuple[int, int] | None = None) -> FieldSpec:
        return FieldSpec(
            axis="z",
            position=self.plane_z,
            bounds=self.bounds,
            grid_shape=self.grid_shape if grid_shape is None else tuple(int(value) for value in grid_shape),
            ray_mode="3d",
            ray_sampling="full_sphere",
        )

    def config(
        self,
        *,
        num_samples: int,
        max_diffraction_order: int | None = None,
        enable_rd_diffraction: bool = True,
    ) -> Config:
        return Config(
            num_samples=int(num_samples),
            max_bounces=self.reflection_max_bounces,
            max_diffraction_order=self.max_diffraction_order if max_diffraction_order is None else int(max_diffraction_order),
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(enable_rd_diffraction=bool(enable_rd_diffraction)),
        )

    def cube_center(self, index: int, *, cube1_x=None):
        center = self.base_centers[index]
        if index != 0 or cube1_x is None:
            return wt.Point3f(*center)
        return wt.Point3f(cube1_x, center[1], center[2])

    def cube_mesh(self, index: int, *, cube1_x=None) -> ChannelMesh:
        return ChannelMesh(
            vertices=_translate_vertices(self.base_box_vertices, self.cube_center(index, cube1_x=cube1_x)),
            faces=self.base_box_faces,
        )

    def build_scene(self, *, cube1_x=None) -> ChannelScene:
        material = Material(eps_r=DEFAULT_RELATIVE_PERMITTIVITY, sigma_e=0.0)
        return ChannelScene(
            structures=[
                Structure(
                    name=f"cube_{index}",
                    geometry=self.cube_mesh(index, cube1_x=cube1_x),
                    material=material,
                )
                for index, _center in enumerate(self.base_centers)
            ],
            device="cuda",
        )

    def solve(
        self,
        *,
        tx_x=None,
        cube1_x=None,
        grid_shape: tuple[int, int] | None = None,
        num_samples: int | None = None,
        max_diffraction_order: int | None = None,
        enable_rd_diffraction: bool = True,
    ) -> FieldResult:
        resolved_tx_x = wt.Float(self.tx_pos[0]) if tx_x is None else tx_x
        scene = self.base_scene if cube1_x is None else self.build_scene(cube1_x=cube1_x)
        return solve_field(
            scene=scene,
            frequency=DEFAULT_FREQUENCY_HZ,
            tx_pos=wt.Point3f(resolved_tx_x, self.tx_pos[1], self.tx_pos[2]),
            field=self.field_spec(grid_shape=grid_shape),
            config=self.config(
                num_samples=(
                    self.forward_num_samples
                    if num_samples is None else int(num_samples)
                ),
                max_diffraction_order=max_diffraction_order,
                enable_rd_diffraction=enable_rd_diffraction,
            ),
        )

    @staticmethod
    def snapshot(result: FieldResult) -> FieldSnapshot:
        shape = (int(result.grid_shape[1]), int(result.grid_shape[0]))
        total = result.field.total
        total_field = np.asarray(total.real, dtype=np.float64).reshape(shape) + 1j * np.asarray(
            total.imag,
            dtype=np.float64,
        ).reshape(shape)
        return FieldSnapshot(
            total_field=total_field,
            total_power=np.asarray(result.power["total"], dtype=np.float64).reshape(shape),
            coords_x=np.asarray(result.coords.grid_x, dtype=np.float64).reshape(shape),
            coords_y=np.asarray(result.coords.grid_y, dtype=np.float64).reshape(shape),
            power_components={
                name: np.asarray(result.power[name], dtype=np.float64).reshape(shape)
                for name in _POWER_COMPONENTS
            },
            metadata=dict(result.metadata),
        )

    def forward(self, *, grid_shape: tuple[int, int] | None = None) -> FieldSnapshot:
        return self.snapshot(
            self.solve(
                grid_shape=grid_shape,
                num_samples=self.forward_num_samples,
                max_diffraction_order=self.max_diffraction_order,
                enable_rd_diffraction=True,
            )
        )

    def gradient(
        self,
        parameter: str,
        *,
        grid_shape: tuple[int, int] | None = None,
        num_samples: int | None = None,
        fd_step: float = DEFAULT_FD_STEP,
    ) -> GradientSnapshot:
        if parameter not in {"tx_x", "cube1_x"}:
            raise ValueError("parameter must be 'tx_x' or 'cube1_x'.")
        resolved_grid_shape = self.grid_shape if grid_shape is None else tuple(int(value) for value in grid_shape)
        resolved_num_samples = (
            self.gradient_num_samples if num_samples is None else int(num_samples)
        )

        _clear_drjit_ad_state()
        if parameter == "tx_x":
            variable = wt.Float(self.tx_pos[0])
            dr.enable_grad(variable)
            result = self.solve(
                tx_x=variable,
                grid_shape=resolved_grid_shape,
                num_samples=resolved_num_samples,
                max_diffraction_order=self.max_diffraction_order,
                enable_rd_diffraction=True,
            )
        else:
            variable = wt.Float(self.base_centers[0][0])
            dr.enable_grad(variable)
            result = self.solve(
                cube1_x=variable,
                grid_shape=resolved_grid_shape,
                num_samples=resolved_num_samples,
                max_diffraction_order=self.max_diffraction_order,
                enable_rd_diffraction=True,
            )

        dr.set_grad(variable, 1.0)
        ad = np.asarray(
            dr.forward_to(result.power["total"], flags=_GRAD_FLAGS),
            dtype=np.float64,
        ).reshape((resolved_grid_shape[1], resolved_grid_shape[0]))
        component_values = dr.forward_to(
            *(result.power[name] for name in _POWER_COMPONENTS),
            flags=_GRAD_FLAGS,
        )
        if not isinstance(component_values, tuple):
            component_values = (component_values,)
        component_ad = {
            name: np.asarray(values, dtype=np.float64).reshape((resolved_grid_shape[1], resolved_grid_shape[0]))
            for name, values in zip(_POWER_COMPONENTS, component_values, strict=True)
        }
        _clear_drjit_ad_state()

        def _fd(offset: float) -> FieldResult:
            if parameter == "tx_x":
                return self.solve(
                    tx_x=wt.Float(self.tx_pos[0] + float(offset)),
                    grid_shape=resolved_grid_shape,
                    num_samples=resolved_num_samples,
                    max_diffraction_order=self.max_diffraction_order,
                    enable_rd_diffraction=True,
                )
            return self.solve(
                cube1_x=wt.Float(self.base_centers[0][0] + float(offset)),
                grid_shape=resolved_grid_shape,
                num_samples=resolved_num_samples,
                max_diffraction_order=self.max_diffraction_order,
                enable_rd_diffraction=True,
            )

        plus = _fd(fd_step)
        minus = _fd(-fd_step)
        fd = (
            np.asarray(plus.power["total"], dtype=np.float64).reshape((resolved_grid_shape[1], resolved_grid_shape[0]))
            - np.asarray(minus.power["total"], dtype=np.float64).reshape((resolved_grid_shape[1], resolved_grid_shape[0]))
        ) / (2.0 * float(fd_step))
        component_fd = {
            name: (
                np.asarray(plus.power[name], dtype=np.float64).reshape((resolved_grid_shape[1], resolved_grid_shape[0]))
                - np.asarray(minus.power[name], dtype=np.float64).reshape((resolved_grid_shape[1], resolved_grid_shape[0]))
            ) / (2.0 * float(fd_step))
            for name in _POWER_COMPONENTS
        }
        return GradientSnapshot(
            parameter=parameter,
            forward=self.snapshot(result),
            ad=ad,
            fd=fd,
            delta=ad - fd,
            component_ad=component_ad,
            component_fd=component_fd,
            component_delta={
                name: component_ad[name] - component_fd[name]
                for name in _POWER_COMPONENTS
            },
        )

    def backward(
        self,
        parameter: str,
        *,
        grid_shape: tuple[int, int] | None = None,
        num_samples: int | None = None,
        fd_step: float = DEFAULT_FD_STEP,
    ) -> BackwardSnapshot:
        if parameter not in {"tx_x", "cube1_x"}:
            raise ValueError("parameter must be 'tx_x' or 'cube1_x'.")

        gradient = self.gradient(
            parameter,
            grid_shape=grid_shape,
            num_samples=num_samples,
            fd_step=fd_step,
        )
        resolved_grid_shape = self.grid_shape if grid_shape is None else tuple(int(value) for value in grid_shape)
        resolved_num_samples = (
            self.gradient_num_samples if num_samples is None else int(num_samples)
        )

        _clear_drjit_ad_state()
        if parameter == "tx_x":
            variable = wt.Float(self.tx_pos[0])
            dr.enable_grad(variable)
            result = self.solve(
                tx_x=variable,
                grid_shape=resolved_grid_shape,
                num_samples=resolved_num_samples,
                max_diffraction_order=self.max_diffraction_order,
                enable_rd_diffraction=True,
            )
        else:
            variable = wt.Float(self.base_centers[0][0])
            dr.enable_grad(variable)
            result = self.solve(
                cube1_x=variable,
                grid_shape=resolved_grid_shape,
                num_samples=resolved_num_samples,
                max_diffraction_order=self.max_diffraction_order,
                enable_rd_diffraction=True,
            )

        loss = dr.sum(result.power["total"])
        dr.eval(loss)
        dr.backward(loss, flags=_GRAD_FLAGS)
        vjp = float(np.asarray(dr.grad(variable), dtype=np.float64).reshape(-1)[0])
        _clear_drjit_ad_state()

        jvp_sum = float(np.sum(gradient.ad))
        fd_sum = float(np.sum(gradient.fd))
        return BackwardSnapshot(
            parameter=parameter,
            vjp=vjp,
            jvp_sum=jvp_sum,
            fd_sum=fd_sum,
            vjp_delta=vjp - jvp_sum,
            fd_delta=vjp - fd_sum,
        )


def smoke_profile() -> dict[str, object]:
    experiment = ThreeCubeField3DExperiment(
        grid_shape=DEFAULT_SMOKE_GRID_SHAPE,
        forward_num_samples=32,
        gradient_num_samples=16,
        max_diffraction_order=0,
    )
    forward = experiment.forward(grid_shape=DEFAULT_SMOKE_GRID_SHAPE)
    tx_grad = experiment.gradient(
        "tx_x",
        grid_shape=(8, 8),
        num_samples=16,
    )
    return {
        "forward_shape": forward.total_power.shape,
        "tx_x_ad_l1": float(np.sum(np.abs(tx_grad.ad))),
    }


def plot_forward(snapshot: FieldSnapshot, *, bounds=DEFAULT_BOUNDS, tx_pos=DEFAULT_TX_POS, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    panels = (
        (
            axes[0],
            _power_db(snapshot.total_power),
            "Total Power (dB)",
            "viridis",
            DEFAULT_POWER_DB_MIN,
            DEFAULT_POWER_DB_MAX,
        ),
        (
            axes[1],
            _field_magnitude_db(snapshot.total_field),
            "Total Field Magnitude (dB)",
            "magma",
            DEFAULT_FIELD_DB_MIN,
            None,
        ),
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
        _decorate_axis(ax, bounds=bounds, tx_pos=tx_pos, title=title)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    return axes


def plot_forward_components(snapshot: FieldSnapshot, *, bounds=DEFAULT_BOUNDS, tx_pos=DEFAULT_TX_POS, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    panels = (
        (axes[0], snapshot.power_components["los"], "LoS Power (dB)"),
        (axes[1], snapshot.power_components["reflection"], "Reflection Power (dB)"),
        (axes[2], snapshot.power_components["diffraction"], "Diffraction Power (dB)"),
    )
    for ax, values, title in panels:
        image = ax.imshow(
            _power_db(values),
            origin="lower",
            extent=_grid_extent(bounds),
            cmap="viridis",
            interpolation="nearest",
            vmin=DEFAULT_POWER_DB_MIN,
            vmax=DEFAULT_POWER_DB_MAX,
        )
        _decorate_axis(ax, bounds=bounds, tx_pos=tx_pos, title=title)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    return axes


def plot_gradient(
    snapshot: GradientSnapshot,
    *,
    bounds=DEFAULT_BOUNDS,
    tx_pos=DEFAULT_TX_POS,
    axes=None,
    floor_db: float = DEFAULT_GRADIENT_DB_FLOOR,
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    encoded_ad = _signed_gradient_db(snapshot.ad, floor_db=floor_db)
    encoded_fd = _signed_gradient_db(snapshot.fd, floor_db=floor_db)
    encoded_delta = _signed_gradient_db(snapshot.delta, floor_db=floor_db)
    vmax = _gradient_display_limit(snapshot.ad, snapshot.fd, snapshot.delta, floor_db=floor_db)
    panels = (
        (axes[0], encoded_ad, f"AD d power / d {snapshot.parameter}"),
        (axes[1], encoded_fd, f"FD d power / d {snapshot.parameter}"),
        (axes[2], encoded_delta, f"AD - FD ({snapshot.parameter})"),
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
        _decorate_axis(ax, bounds=bounds, tx_pos=tx_pos, title=f"{title} (signed dB)")
        colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(f"Signed dB above {float(floor_db):.0f} dB")
    return axes


def plot_gradient_components(
    snapshot: GradientSnapshot,
    *,
    bounds=DEFAULT_BOUNDS,
    tx_pos=DEFAULT_TX_POS,
    axes=None,
    floor_db: float = DEFAULT_GRADIENT_DB_FLOOR,
):
    if axes is None:
        _, axes = plt.subplots(3, 3, figsize=(15.0, 12.0), constrained_layout=True)
    vmax = _gradient_display_limit(
        *(snapshot.component_ad[name] for name in _POWER_COMPONENTS),
        *(snapshot.component_fd[name] for name in _POWER_COMPONENTS),
        *(snapshot.component_delta[name] for name in _POWER_COMPONENTS),
        floor_db=floor_db,
    )
    column_titles = ("AD", "FD", "AD - FD")
    for row, component in enumerate(_POWER_COMPONENTS):
        encoded = (
            _signed_gradient_db(snapshot.component_ad[component], floor_db=floor_db),
            _signed_gradient_db(snapshot.component_fd[component], floor_db=floor_db),
            _signed_gradient_db(snapshot.component_delta[component], floor_db=floor_db),
        )
        for col, values in enumerate(encoded):
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
            _decorate_axis(
                ax,
                bounds=bounds,
                tx_pos=tx_pos,
                title=f"{component} {column_titles[col]} (signed dB)",
            )
            colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            colorbar.set_label(f"Signed dB above {float(floor_db):.0f} dB")
    return axes


__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_FD_STEP",
    "DEFAULT_FORWARD_REFLECTION_N_RAYS",
    "DEFAULT_GRADIENT_REFLECTION_N_RAYS",
    "DEFAULT_GRID_SHAPE",
    "DEFAULT_MAX_DIFFRACTION_ORDER",
    "DEFAULT_PLANE_Z",
    "DEFAULT_TX_POS",
    "BackwardSnapshot",
    "FieldSnapshot",
    "GradientSnapshot",
    "ThreeCubeField3DExperiment",
    "plot_forward",
    "plot_forward_components",
    "plot_gradient",
    "plot_gradient_components",
    "smoke_profile",
]


if __name__ == "__main__":
    print(smoke_profile())



