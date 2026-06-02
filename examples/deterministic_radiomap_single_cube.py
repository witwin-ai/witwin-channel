"""Single-cube deterministic radiomap example with diffraction enabled."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import witwin.channel as wt

from witwin.channel.core.scene import Mesh as ChannelMesh
from witwin.channel.core.scene import EdgePolicy, ReceiverGrid, Scene as ChannelScene, Transmitter
from witwin.channel.core.geometry.mesh_buffers import to_point3f, to_vector3u
from witwin.core import Box, Material, Structure
from witwin.channel.deterministic import Config, Tuning, solve


DEFAULT_BOUNDS = ((-8.0, 8.0), (-8.0, 8.0))
DEFAULT_GRID_SHAPE = (256, 256)
DEFAULT_PLANE_Z = 1.0
DEFAULT_FREQUENCY_HZ = 1.0e9
DEFAULT_TX_POS = (-2.0, -5.0, 4.0)
DEFAULT_CUBE_CENTER = (0.0, 0.0, 1.5)
DEFAULT_CUBE_SIZE = 2.0
DEFAULT_RELATIVE_PERMITTIVITY = 1.0e4
DEFAULT_FORWARD_NUM_SAMPLES = 192
DEFAULT_MAX_BOUNCES = 2
DEFAULT_MAX_DIFFRACTION_ORDER = 1
DEFAULT_SHADOW_BOUNDARY_CORRECTION = False
DEFAULT_DB_MIN = -90.0
DEFAULT_DB_MAX = -40.0
DEFAULT_LINE_PROFILE_YS = (-4.0, 4.0)
_POWER_COMPONENTS = ("los", "reflection", "diffraction")
_COMPONENT_COLORS = {
    "los": "#2f6fed",
    "reflection": "#dd7c22",
    "diffraction": "#2a9d54",
}


@dataclass(frozen=True, slots=True)
class SolveSnapshot:
    path_gain: np.ndarray
    coords_x: np.ndarray
    coords_y: np.ndarray
    components: dict[str, np.ndarray]
    metadata: dict[str, object]


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


def _path_gain_db(values: np.ndarray, floor: float = 1.0e-20) -> np.ndarray:
    return 10.0 * np.log10(
        np.maximum(np.asarray(values, dtype=np.float64), float(floor))
    )


def _line_axis(
    coords: np.ndarray, *, axis: str, row_index: int | None = None
) -> np.ndarray:
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim == 1:
        return array
    if axis == "x":
        if row_index is None:
            raise ValueError("row_index is required for x-axis line extraction.")
        return array[int(row_index), :]
    return array[:, 0]


def _decorate_topdown_axis(
    ax,
    *,
    bounds=DEFAULT_BOUNDS,
    tx_pos=DEFAULT_TX_POS,
    cube_center=DEFAULT_CUBE_CENTER,
    cube_size: float = DEFAULT_CUBE_SIZE,
    title: str | None = None,
) -> None:
    cx, cy, _cz = cube_center
    size = float(cube_size)
    ax.add_patch(
        Rectangle(
            (float(cx) - size / 2.0, float(cy) - size / 2.0),
            size,
            size,
            fill=False,
            edgecolor="black",
            linewidth=1.1,
        )
    )
    ax.scatter(
        [float(tx_pos[0])],
        [float(tx_pos[1])],
        marker="*",
        s=85,
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


def _cube_faces(
    center=DEFAULT_CUBE_CENTER, size: float = DEFAULT_CUBE_SIZE
) -> list[list[tuple[float, float, float]]]:
    cx, cy, cz = (float(value) for value in center)
    half = float(size) / 2.0
    vertices = [
        (cx - half, cy - half, cz - half),
        (cx + half, cy - half, cz - half),
        (cx + half, cy + half, cz - half),
        (cx - half, cy + half, cz - half),
        (cx - half, cy - half, cz + half),
        (cx + half, cy - half, cz + half),
        (cx + half, cy + half, cz + half),
        (cx - half, cy + half, cz + half),
    ]
    face_indices = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4),
    )
    return [[vertices[index] for index in face] for face in face_indices]


class SingleCubeExperiment:
    def __init__(
        self,
        *,
        bounds=DEFAULT_BOUNDS,
        grid_shape=DEFAULT_GRID_SHAPE,
        plane_z: float = DEFAULT_PLANE_Z,
        tx_pos=DEFAULT_TX_POS,
        forward_num_samples: int = DEFAULT_FORWARD_NUM_SAMPLES,
        max_bounces: int = DEFAULT_MAX_BOUNCES,
        max_diffraction_order: int = DEFAULT_MAX_DIFFRACTION_ORDER,
        shadow_boundary_correction: bool = DEFAULT_SHADOW_BOUNDARY_CORRECTION,
    ) -> None:
        self.bounds = bounds
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.plane_z = float(plane_z)
        self.tx_pos = tuple(float(value) for value in tx_pos)
        self.cube_center = tuple(float(value) for value in DEFAULT_CUBE_CENTER)
        self.max_bounces = int(max_bounces)
        self.max_diffraction_order = int(max_diffraction_order)
        self.base_box_vertices, self.base_box_faces = _origin_box_mesh()
        self.grid = ReceiverGrid(
            "rm",
            axis="z",
            position=self.plane_z,
            bounds=self.bounds,
            grid_shape=self.grid_shape,
        )
        self.scene = self._build_scene()
        self.forward_config = Config(
            num_samples=int(forward_num_samples),
            max_bounces=self.max_bounces,
            shadow_boundary_correction=bool(shadow_boundary_correction),
            max_diffraction_order=self.max_diffraction_order,
            edge_policy=EdgePolicy(edge_selection_mode="all_edges"),
            tuning=Tuning(enable_rd_diffraction=True),
        )

    def _cube_mesh(self) -> ChannelMesh:
        return ChannelMesh(
            vertices=_translate_vertices(self.base_box_vertices, self.cube_center),
            faces=self.base_box_faces,
        )

    def _build_scene(self) -> ChannelScene:
        material = Material(eps_r=DEFAULT_RELATIVE_PERMITTIVITY, sigma_e=0.0)
        return ChannelScene(
            structures=[
                Structure(
                    name="center_cube",
                    geometry=self._cube_mesh(),
                    material=material,
                )
            ],
            transmitters=[
                Transmitter("tx", wt.Point3f(*self.tx_pos)),
            ],
            receivers=[
                self.grid,
            ],
            frequency=DEFAULT_FREQUENCY_HZ,
            device="cuda",
        )

    def solve(self):
        return solve(
            scene=self.scene,
            transmitter="tx",
            receiver="rm",
            config=self.forward_config,
        )

    @staticmethod
    def snapshot(result) -> SolveSnapshot:
        result = result.squeeze_tx(0)
        return SolveSnapshot(
            path_gain=np.asarray(result.path_gain, dtype=np.float64),
            coords_x=np.asarray(result.coords.grid_x, dtype=np.float64),
            coords_y=np.asarray(result.coords.grid_y, dtype=np.float64),
            components={
                name: np.asarray(result.components[name], dtype=np.float64)
                for name in _POWER_COMPONENTS
            },
            metadata=dict(result.metadata),
        )

    def forward(self) -> SolveSnapshot:
        return self.snapshot(self.solve())


def plot_scene_layout(experiment: SingleCubeExperiment, *, ax=None):
    if ax is None:
        figure = plt.figure(figsize=(6.2, 5.2))
        ax = figure.add_subplot(111, projection="3d")
    faces = _cube_faces(center=experiment.cube_center)
    cube = Poly3DCollection(
        faces,
        facecolors="#7ca6c8",
        edgecolors="#1f2d3a",
        linewidths=0.9,
        alpha=0.45,
    )
    ax.add_collection3d(cube)
    tx = experiment.tx_pos
    ax.scatter(
        [tx[0]], [tx[1]], [tx[2]], marker="*", s=150, color="gold", edgecolor="black"
    )
    ax.scatter([tx[0]], [tx[1]], [experiment.plane_z], marker="x", s=55, color="black")
    ax.set_xlim(experiment.bounds[0])
    ax.set_ylim(experiment.bounds[1])
    ax.set_zlim(0.0, max(float(tx[2]) + 1.0, 5.0))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Single Cube Scene Layout")
    ax.view_init(elev=25.0, azim=-58.0)
    return ax


def plot_forward(snapshot: SolveSnapshot, *, experiment: SingleCubeExperiment, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.0))
    image = ax.imshow(
        _path_gain_db(snapshot.path_gain),
        origin="lower",
        extent=_grid_extent(experiment.bounds),
        cmap="viridis",
        interpolation="nearest",
        vmin=DEFAULT_DB_MIN,
        vmax=DEFAULT_DB_MAX,
    )
    _decorate_topdown_axis(
        ax,
        bounds=experiment.bounds,
        tx_pos=experiment.tx_pos,
        cube_center=experiment.cube_center,
        title="Path Gain With Diffraction (dB)",
    )
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    return ax


def plot_forward_components(
    snapshot: SolveSnapshot, *, experiment: SingleCubeExperiment, axes=None
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    panels = (
        (axes[0], snapshot.components["los"], "LoS Power (dB)"),
        (axes[1], snapshot.components["reflection"], "Reflection Power (dB)"),
        (axes[2], snapshot.components["diffraction"], "Diffraction Power (dB)"),
    )
    for ax, values, title in panels:
        image = ax.imshow(
            _path_gain_db(values),
            origin="lower",
            extent=_grid_extent(experiment.bounds),
            cmap="viridis",
            interpolation="nearest",
            vmin=DEFAULT_DB_MIN,
            vmax=DEFAULT_DB_MAX,
        )
        _decorate_topdown_axis(
            ax,
            bounds=experiment.bounds,
            tx_pos=experiment.tx_pos,
            cube_center=experiment.cube_center,
            title=title,
        )
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    return axes


def line_power_profiles(
    snapshot: SolveSnapshot,
    *,
    y_values=DEFAULT_LINE_PROFILE_YS,
) -> dict[float, dict[str, object]]:
    y_axis = _line_axis(snapshot.coords_y, axis="y")
    profiles = {}
    for target_y in y_values:
        row_index = int(np.argmin(np.abs(y_axis - float(target_y))))
        x = _line_axis(snapshot.coords_x, axis="x", row_index=row_index)
        path_gain = np.asarray(snapshot.path_gain[row_index, :], dtype=np.float64)
        components = {
            name: np.asarray(snapshot.components[name][row_index, :], dtype=np.float64)
            for name in _POWER_COMPONENTS
        }
        component_sum = np.zeros_like(path_gain)
        for values in components.values():
            component_sum = component_sum + values
        component_fraction = {
            name: np.divide(
                values,
                component_sum,
                out=np.zeros_like(values),
                where=component_sum > 0.0,
            )
            for name, values in components.items()
        }
        profiles[float(target_y)] = {
            "target_y": float(target_y),
            "sampled_y": float(y_axis[row_index]),
            "row_index": row_index,
            "x": x,
            "path_gain": path_gain,
            "components": components,
            "component_sum": component_sum,
            "component_fraction": component_fraction,
        }
    return profiles


def summarize_line_power_profiles(
    profiles: dict[float, dict[str, object]],
) -> dict[str, dict[str, object]]:
    summary = {}
    for target_y, profile in profiles.items():
        path_gain = np.asarray(profile["path_gain"], dtype=np.float64)
        components = profile["components"]
        fractions = profile["component_fraction"]
        summary[f"y={target_y:g}"] = {
            "sampled_y": float(profile["sampled_y"]),
            "row_index": int(profile["row_index"]),
            "path_gain_max_db": float(_path_gain_db([np.max(path_gain)])[0]),
            "path_gain_mean_db": float(_path_gain_db([np.mean(path_gain)])[0]),
            "component_max_db": {
                name: float(_path_gain_db([np.max(values)])[0])
                for name, values in components.items()
            },
            "component_mean_db": {
                name: float(_path_gain_db([np.mean(values)])[0])
                for name, values in components.items()
            },
            "component_fraction_mean": {
                name: float(np.mean(values)) for name, values in fractions.items()
            },
            "component_nonzero_fraction": {
                name: float(np.mean(np.asarray(values) > 0.0))
                for name, values in components.items()
            },
        }
    return summary


def plot_line_power_profiles(profiles: dict[float, dict[str, object]], *, axes=None):
    items = list(profiles.items())
    if axes is None:
        _, axes = plt.subplots(
            len(items),
            2,
            figsize=(12.0, 3.7 * len(items)),
            squeeze=False,
            constrained_layout=True,
        )

    for row, (_target_y, profile) in enumerate(items):
        x = np.asarray(profile["x"], dtype=np.float64)
        sampled_y = float(profile["sampled_y"])
        power_ax = axes[row, 0]
        power_ax.plot(
            x,
            _path_gain_db(profile["path_gain"]),
            color="black",
            linewidth=1.8,
            label="Total",
        )
        for name in _POWER_COMPONENTS:
            values = profile["components"][name]
            power_ax.plot(
                x,
                _path_gain_db(values),
                color=_COMPONENT_COLORS[name],
                linewidth=1.2,
                label=name.title(),
            )
        power_ax.set_title(f"Power Line at y={sampled_y:.2f} m")
        power_ax.set_xlabel("x (m)")
        power_ax.set_ylabel("Power (dB)")
        power_ax.set_ylim(DEFAULT_DB_MIN, DEFAULT_DB_MAX)
        power_ax.grid(True, alpha=0.25)
        power_ax.legend(loc="lower right", fontsize=8)

        mix_ax = axes[row, 1]
        baseline = np.zeros_like(x)
        for name in _POWER_COMPONENTS:
            values = np.asarray(profile["component_fraction"][name], dtype=np.float64)
            mix_ax.fill_between(
                x,
                baseline,
                baseline + values,
                color=_COMPONENT_COLORS[name],
                alpha=0.75,
                label=name.title(),
            )
            baseline = baseline + values
        mix_ax.set_title(f"Component Mix at y={sampled_y:.2f} m")
        mix_ax.set_xlabel("x (m)")
        mix_ax.set_ylabel("Power Fraction")
        mix_ax.set_ylim(0.0, 1.0)
        mix_ax.grid(True, alpha=0.25)
        mix_ax.legend(loc="upper right", fontsize=8)
    return axes


def smoke_profile() -> dict[str, object]:
    experiment = SingleCubeExperiment(
        grid_shape=(16, 16),
        forward_num_samples=64,
        max_bounces=2,
        max_diffraction_order=1,
        shadow_boundary_correction=False,
    )
    snapshot = experiment.forward()
    profiles = line_power_profiles(snapshot)
    return {
        "path_gain_shape": snapshot.path_gain.shape,
        "finite": bool(np.isfinite(snapshot.path_gain).all()),
        "max_diffraction_order": experiment.forward_config.max_diffraction_order,
        "enable_rd_diffraction": experiment.forward_config.tuning.enable_rd_diffraction,
        "line_profile_rows": {
            label: profile["row_index"]
            for label, profile in summarize_line_power_profiles(profiles).items()
        },
    }


__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_DB_MAX",
    "DEFAULT_DB_MIN",
    "DEFAULT_GRID_SHAPE",
    "DEFAULT_LINE_PROFILE_YS",
    "DEFAULT_FORWARD_NUM_SAMPLES",
    "DEFAULT_MAX_BOUNCES",
    "DEFAULT_MAX_DIFFRACTION_ORDER",
    "DEFAULT_PLANE_Z",
    "DEFAULT_TX_POS",
    "SingleCubeExperiment",
    "SolveSnapshot",
    "line_power_profiles",
    "plot_forward",
    "plot_forward_components",
    "plot_line_power_profiles",
    "plot_scene_layout",
    "smoke_profile",
    "summarize_line_power_profiles",
]


if __name__ == "__main__":
    print(smoke_profile())
