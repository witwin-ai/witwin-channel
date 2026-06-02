"""witwin.channel.deterministic.solve_field three-cube multipath example."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
import sys

import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import witwin.channel as wt
from matplotlib.patches import Rectangle

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from witwin.channel.core.scene import Mesh as ChannelMesh
from witwin.channel.core.scene import Scene as ChannelScene
from witwin.channel.core.geometry.mesh_buffers import to_point3f, to_vector3u
from witwin.core import Box, Material, Structure
from witwin.channel.deterministic import Config, FieldResult, FieldSpec, Tuning, solve_field


DEFAULT_BOUNDS = ((-6.0, 6.0), (-6.0, 6.0))
DEFAULT_GRID_SHAPE = (256, 256)
DEFAULT_PLANE_Z = 1.5
DEFAULT_FREQUENCY_HZ = 1.0e9
DEFAULT_TX_POS = (0.0, -5.0, 1.5)
DEFAULT_RELATIVE_PERMITTIVITY = 1.0e4
DEFAULT_CUBE_CENTERS = (
    (-2.5, -3.0, 1.5),
    (2.0, 0.5, 1.5),
    (-0.5, 3.5, 1.5),
)
DEFAULT_CUBE_SIZE = 2.0
DEFAULT_REFLECTION_N_RAYS = 1280
DEFAULT_REFLECTION_MAX_BOUNCES = 3
DEFAULT_MAX_DIFFRACTIONS = 2
DEFAULT_FD_STEP = 1.0e-3
DEFAULT_POWER_DB_MIN = -140.0
DEFAULT_FIELD_DB_MIN = -140.0
DEFAULT_GRADIENT_DB_FLOOR = -180.0
_GRAD_FLAGS = dr.ADFlag.Default | dr.ADFlag.AllowNoGrad


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


def _decorate_scene_axis(ax, title: str, *, bounds=DEFAULT_BOUNDS) -> None:
    for center in DEFAULT_CUBE_CENTERS:
        cx, cy, _cz = center
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
    ax.plot(
        [DEFAULT_TX_POS[0]],
        [DEFAULT_TX_POS[1]],
        marker="*",
        markersize=8,
        color="gold",
        markeredgecolor="black",
    )
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


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
    all_values = np.concatenate(encoded)
    vmax = float(np.nanpercentile(all_values, 99.0))
    return max(vmax, 1.0)


@dataclass(frozen=True, slots=True)
class FieldSnapshot:
    total_field: np.ndarray
    total_power: np.ndarray
    coords_x: np.ndarray
    coords_y: np.ndarray


@dataclass(frozen=True, slots=True)
class GradientSnapshot:
    parameter: str
    forward: FieldSnapshot
    ad: np.ndarray
    fd: np.ndarray
    delta: np.ndarray
    component_ad: dict[str, np.ndarray]
    component_l1: dict[str, float]


@dataclass(frozen=True, slots=True)
class BackwardSnapshot:
    parameter: str
    vjp: float
    jvp_sum: float
    fd_sum: float
    vjp_delta: float
    fd_delta: float


class ThreeCubeFieldExperiment:
    """Small reusable experiment for notebooks and smoke tests."""

    def __init__(
        self,
        *,
        bounds=DEFAULT_BOUNDS,
        grid_shape=DEFAULT_GRID_SHAPE,
        plane_z: float = DEFAULT_PLANE_Z,
        tx_pos=DEFAULT_TX_POS,
        num_samples: int = DEFAULT_REFLECTION_N_RAYS,
        reflection_max_bounces: int = DEFAULT_REFLECTION_MAX_BOUNCES,
        max_diffraction_order: int = DEFAULT_MAX_DIFFRACTIONS,
    ) -> None:
        self.bounds = bounds
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.plane_z = float(plane_z)
        self.tx_pos = tuple(float(value) for value in tx_pos)
        self.num_samples = int(num_samples)
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
            grid_shape=self.grid_shape if grid_shape is None else grid_shape,
            ray_mode="2d",
            ray_sampling="full_sphere",
        )

    def config(
        self,
        *,
        num_samples: int | None = None,
        max_diffraction_order: int | None = None,
        enable_rd_diffraction: bool = True,
    ) -> Config:
        return Config(
            num_samples=self.num_samples if num_samples is None else int(num_samples),
            max_bounces=self.reflection_max_bounces,
            max_diffraction_order=self.max_diffraction_order if max_diffraction_order is None else int(max_diffraction_order),
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

    @staticmethod
    def cube_name(index: int) -> str:
        return f"cube{index + 1}"

    def build_scene(self, *, cube1_x=None, cube1_eps=None) -> ChannelScene:
        material = Material(eps_r=DEFAULT_RELATIVE_PERMITTIVITY, sigma_e=0.0)
        structures = [
            Structure(
                name=self.cube_name(index),
                geometry=self.cube_mesh(index, cube1_x=cube1_x),
                material=material,
            )
            for index, _center in enumerate(self.base_centers)
        ]
        scene = ChannelScene(
            structures=structures,
            device="cuda",
        )
        if cube1_eps is not None:
            scene.structure("cube1").set_material_parameters(eps_r=cube1_eps)
        return scene

    def solve(
        self,
        *,
        tx_x=None,
        cube1_x=None,
        cube1_eps=None,
        grid_shape: tuple[int, int] | None = None,
        num_samples: int | None = None,
        max_diffraction_order: int | None = None,
        enable_rd_diffraction: bool = True,
    ) -> FieldResult:
        resolved_tx_x = wt.Float(self.tx_pos[0]) if tx_x is None else tx_x
        scene = (
            self.base_scene
            if cube1_x is None and cube1_eps is None
            else self.build_scene(cube1_x=cube1_x, cube1_eps=cube1_eps)
        )
        return solve_field(
            scene=scene,
            frequency=DEFAULT_FREQUENCY_HZ,
            tx_pos=wt.Point3f(resolved_tx_x, self.tx_pos[1], self.tx_pos[2]),
            field=self.field_spec(grid_shape=grid_shape),
            config=self.config(
                num_samples=num_samples,
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
        )

    def forward(
        self,
        *,
        grid_shape: tuple[int, int] | None = None,
        num_samples: int | None = None,
        max_diffraction_order: int | None = None,
        enable_rd_diffraction: bool = True,
    ) -> FieldSnapshot:
        result = self.solve(
            grid_shape=grid_shape,
            num_samples=num_samples,
            max_diffraction_order=max_diffraction_order,
            enable_rd_diffraction=enable_rd_diffraction,
        )
        return self.snapshot(result)

    def gradient(
        self,
        parameter: str,
        *,
        grid_shape: tuple[int, int] | None = None,
        num_samples: int | None = None,
        max_diffraction_order: int | None = None,
        enable_rd_diffraction: bool = True,
        fd_step: float = DEFAULT_FD_STEP,
    ) -> GradientSnapshot:
        if parameter not in {"tx_x", "cube1_x", "cube1_eps"}:
            raise ValueError("parameter must be 'tx_x', 'cube1_x', or 'cube1_eps'.")

        resolved_grid_shape = self.grid_shape if grid_shape is None else tuple(int(value) for value in grid_shape)
        resolved_num_samples = (
            self.num_samples if num_samples is None else int(num_samples)
        )
        resolved_max_diffraction_order = (
            self.max_diffraction_order if max_diffraction_order is None else int(max_diffraction_order)
        )

        _clear_drjit_ad_state()
        if parameter == "tx_x":
            variable = wt.Float(self.tx_pos[0])
            dr.enable_grad(variable)
            result = self.solve(
                tx_x=variable,
                grid_shape=resolved_grid_shape,
                num_samples=resolved_num_samples,
                max_diffraction_order=resolved_max_diffraction_order,
                enable_rd_diffraction=enable_rd_diffraction,
            )
        elif parameter == "cube1_x":
            variable = wt.Float(self.base_centers[0][0])
            dr.enable_grad(variable)
            result = self.solve(
                cube1_x=variable,
                grid_shape=resolved_grid_shape,
                num_samples=resolved_num_samples,
                max_diffraction_order=resolved_max_diffraction_order,
                enable_rd_diffraction=enable_rd_diffraction,
            )
        else:
            variable = wt.Float(DEFAULT_RELATIVE_PERMITTIVITY)
            dr.enable_grad(variable)
            result = self.solve(
                cube1_eps=variable,
                grid_shape=resolved_grid_shape,
                num_samples=resolved_num_samples,
                max_diffraction_order=resolved_max_diffraction_order,
                enable_rd_diffraction=enable_rd_diffraction,
            )

        dr.set_grad(variable, 1.0)
        total_jvp, diffraction_jvp = dr.forward_to(
            result.power["total"],
            result.power["diffraction"],
            flags=_GRAD_FLAGS,
        )
        ad = np.asarray(total_jvp, dtype=np.float64).reshape((resolved_grid_shape[1], resolved_grid_shape[0]))
        component_ad = {
            "total": ad,
            "diffraction": np.asarray(diffraction_jvp, dtype=np.float64).reshape(
                (resolved_grid_shape[1], resolved_grid_shape[0])
            ),
        }
        component_l1 = {
            name: float(np.sum(np.abs(values)))
            for name, values in component_ad.items()
        }
        forward_snapshot = self.snapshot(result)
        # Drop the AD-backed solve before launching the two FD solves.
        del result, variable
        _clear_drjit_ad_state()

        def _fd(offset: float) -> np.ndarray:
            if parameter == "tx_x":
                solved = self.solve(
                    tx_x=wt.Float(self.tx_pos[0] + offset),
                    grid_shape=resolved_grid_shape,
                    num_samples=resolved_num_samples,
                    max_diffraction_order=resolved_max_diffraction_order,
                    enable_rd_diffraction=enable_rd_diffraction,
                )
            elif parameter == "cube1_x":
                solved = self.solve(
                    cube1_x=wt.Float(self.base_centers[0][0] + offset),
                    grid_shape=resolved_grid_shape,
                    num_samples=resolved_num_samples,
                    max_diffraction_order=resolved_max_diffraction_order,
                    enable_rd_diffraction=enable_rd_diffraction,
                )
            else:
                solved = self.solve(
                    cube1_eps=wt.Float(DEFAULT_RELATIVE_PERMITTIVITY + offset),
                    grid_shape=resolved_grid_shape,
                    num_samples=resolved_num_samples,
                    max_diffraction_order=resolved_max_diffraction_order,
                    enable_rd_diffraction=enable_rd_diffraction,
                )
            power = np.asarray(solved.power["total"], dtype=np.float64).reshape(
                (resolved_grid_shape[1], resolved_grid_shape[0])
            ).copy()
            del solved
            _clear_drjit_ad_state()
            return power

        fd = (_fd(fd_step) - _fd(-fd_step)) / (2.0 * float(fd_step))
        return GradientSnapshot(
            parameter=parameter,
            forward=forward_snapshot,
            ad=ad,
            fd=fd,
            delta=ad - fd,
            component_ad=component_ad,
            component_l1=component_l1,
        )

    def backward(
        self,
        parameter: str,
        *,
        grid_shape: tuple[int, int] | None = None,
        num_samples: int | None = None,
        max_diffraction_order: int | None = None,
        enable_rd_diffraction: bool = True,
        fd_step: float = DEFAULT_FD_STEP,
    ) -> BackwardSnapshot:
        if parameter not in {"tx_x", "cube1_x", "cube1_eps"}:
            raise ValueError("parameter must be 'tx_x', 'cube1_x', or 'cube1_eps'.")

        gradient = self.gradient(
            parameter,
            grid_shape=grid_shape,
            num_samples=num_samples,
            max_diffraction_order=max_diffraction_order,
            enable_rd_diffraction=enable_rd_diffraction,
            fd_step=fd_step,
        )
        resolved_grid_shape = self.grid_shape if grid_shape is None else tuple(int(value) for value in grid_shape)
        resolved_num_samples = (
            self.num_samples if num_samples is None else int(num_samples)
        )
        resolved_max_diffraction_order = (
            self.max_diffraction_order if max_diffraction_order is None else int(max_diffraction_order)
        )

        _clear_drjit_ad_state()
        if parameter == "tx_x":
            variable = wt.Float(self.tx_pos[0])
            dr.enable_grad(variable)
            result = self.solve(
                tx_x=variable,
                grid_shape=resolved_grid_shape,
                num_samples=resolved_num_samples,
                max_diffraction_order=resolved_max_diffraction_order,
                enable_rd_diffraction=enable_rd_diffraction,
            )
        elif parameter == "cube1_x":
            variable = wt.Float(self.base_centers[0][0])
            dr.enable_grad(variable)
            result = self.solve(
                cube1_x=variable,
                grid_shape=resolved_grid_shape,
                num_samples=resolved_num_samples,
                max_diffraction_order=resolved_max_diffraction_order,
                enable_rd_diffraction=enable_rd_diffraction,
            )
        else:
            variable = wt.Float(DEFAULT_RELATIVE_PERMITTIVITY)
            dr.enable_grad(variable)
            result = self.solve(
                cube1_eps=variable,
                grid_shape=resolved_grid_shape,
                num_samples=resolved_num_samples,
                max_diffraction_order=resolved_max_diffraction_order,
                enable_rd_diffraction=enable_rd_diffraction,
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
    experiment = ThreeCubeFieldExperiment(grid_shape=(16, 16), num_samples=64, max_diffraction_order=1)
    forward = experiment.forward(
        grid_shape=(16, 16),
        num_samples=64,
        max_diffraction_order=1,
        enable_rd_diffraction=False,
    )
    tx_grad = experiment.gradient(
        "tx_x",
        grid_shape=(8, 8),
        num_samples=32,
        max_diffraction_order=0,
        enable_rd_diffraction=False,
    )
    cube_grad = experiment.gradient(
        "cube1_eps",
        grid_shape=(8, 8),
        num_samples=32,
        max_diffraction_order=1,
        enable_rd_diffraction=False,
    )
    return {
        "forward_shape": forward.total_power.shape,
        "tx_x_ad_l1": float(np.sum(np.abs(tx_grad.ad))),
        "cube1_eps_ad_l1": float(np.sum(np.abs(cube_grad.ad))),
        "cube1_eps_diffraction_ad_l1": cube_grad.component_l1["diffraction"],
    }


def plot_forward(snapshot: FieldSnapshot, *, bounds=DEFAULT_BOUNDS, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    panels = (
        (
            axes[0],
            _power_db(snapshot.total_power),
            "Total Power (dB)",
            "jet",
            -60.0,
            -20.0,
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
        ax.set_title(title)
        _decorate_scene_axis(ax, title, bounds=bounds)
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
        _, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    encoded_ad = _signed_gradient_db(snapshot.ad, floor_db=floor_db)
    encoded_fd = _signed_gradient_db(snapshot.fd, floor_db=floor_db)
    encoded_delta = _signed_gradient_db(snapshot.delta, floor_db=floor_db)
    vmax = _gradient_display_limit(
        snapshot.ad,
        snapshot.fd,
        snapshot.delta,
        floor_db=floor_db,
    )
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
        ax.set_title(f"{title} (signed dB)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(f"Signed dB above {float(floor_db):.0f} dB")
    return axes


def plot_material_gradient(
    snapshot: GradientSnapshot,
    *,
    bounds=DEFAULT_BOUNDS,
    axes=None,
    floor_db: float = DEFAULT_GRADIENT_DB_FLOOR,
):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), constrained_layout=True)
    vmax = _gradient_display_limit(
        snapshot.component_ad["total"],
        snapshot.component_ad["diffraction"],
        floor_db=floor_db,
    )
    panels = (
        (axes[0], snapshot.component_ad["total"], f"Total d power / d {snapshot.parameter}"),
        (axes[1], snapshot.component_ad["diffraction"], f"Diffraction d power / d {snapshot.parameter}"),
    )
    for ax, values, title in panels:
        image = ax.imshow(
            _signed_gradient_db(values, floor_db=floor_db),
            origin="lower",
            extent=_grid_extent(bounds),
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"{title} (signed dB)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(f"Signed dB above {float(floor_db):.0f} dB")
    return axes


def material_gradient_summary(snapshot: GradientSnapshot, backward: BackwardSnapshot | None = None) -> dict[str, float]:
    summary = {
        "total_l1": float(snapshot.component_l1["total"]),
        "diffraction_l1": float(snapshot.component_l1["diffraction"]),
    }
    if backward is not None:
        summary["backward_scalar_gradient"] = float(backward.vjp)
    return summary


__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_FIELD_DB_MIN",
    "DEFAULT_FD_STEP",
    "DEFAULT_FREQUENCY_HZ",
    "DEFAULT_GRID_SHAPE",
    "DEFAULT_GRADIENT_DB_FLOOR",
    "DEFAULT_POWER_DB_MIN",
    "BackwardSnapshot",
    "FieldSnapshot",
    "GradientSnapshot",
    "ThreeCubeFieldExperiment",
    "plot_forward",
    "plot_gradient",
    "plot_material_gradient",
    "material_gradient_summary",
    "smoke_profile",
]


if __name__ == "__main__":
    print(smoke_profile())


