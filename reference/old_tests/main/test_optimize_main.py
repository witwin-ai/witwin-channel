"""Opt-in standalone optimization visual test."""

from __future__ import annotations

import random
from dataclasses import dataclass
import os
from pathlib import Path

import drjit as dr
import numpy as np
import pytest
import witwin as wt
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import DEFAULT_VARIANT, POWER_DB_FLOOR, FieldMonitor, Tracer
if os.environ.get("WITWIN_CHANNEL_MAIN_SHOW", "0") != "1":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

pytestmark = [pytest.mark.gpu, pytest.mark.optimize]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "optimize.png"
DEFAULT_MONITOR_NAME = "main_optimize_grid"


def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    wt.set_log_level(wt.LogLevel.Warn)
    try:
        dr.seed(seed)
    except (AttributeError, TypeError):
        pass
    try:
        wt.register_sampler_seed(seed)
    except AttributeError:
        pass


def _scalar_height(value) -> float:
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    try:
        return float(value[0])
    except (TypeError, ValueError, IndexError, KeyError):
        return float(value)


def _monitor_height(tx_pos) -> float:
    if hasattr(tx_pos, "z"):
        return _scalar_height(tx_pos.z)
    return _scalar_height(tx_pos[2])


def _dr_scalar(value) -> float:
    array = np.asarray(dr.detach(value), dtype=np.float64).reshape(-1)
    return float(array[0])


def _assert_plane_monitor_result(result, monitor: FieldMonitor) -> None:
    payload = result.monitor(monitor.name)
    sampling = payload.metadata["receiver_sampling"]
    assert sampling["sample_positions"] == "boundary_points"
    assert sampling["index_partitioning"] == "span_over_n_bins"
    assert tuple(payload.grid_shape) == tuple(monitor.grid_shape)
    assert tuple(payload.range_x) == monitor.bounds[0]
    assert tuple(payload.range_y) == monitor.bounds[1]
    assert abs(float(payload.plane_position) - float(monitor.position)) < 1e-6


@dataclass
class OptimizeConfig:
    grid_size: int = 96
    frequency: float = 1e9
    range_x: tuple[float, float] = (-8.0, 8.0)
    range_y: tuple[float, float] = (-8.0, 8.0)
    n_rays: int = 20_000
    max_reflections: int = 1
    reflection_coef: float = 1.0
    cube_size: float = 4.0
    target_center: tuple[float, float, float] = (3.0, -2.5, 2.0)
    target_rotation: float = np.pi / 5 + np.pi * 1.5
    init_center: tuple[float, float, float] = (0.0, 0.0, 2.0)
    init_tx: tuple[float, float, float] = (-5.0, 5.0, 1.5)
    init_rotation: float = 0.0
    learning_rate: float = 0.2
    lr_min: float = 1e-3
    use_cosine_annealing: bool = True
    n_iterations: int = 80
    output_path: str = str(OUTPUT_PATH)
    seed: int = 42


class RadioFieldOptimizer:
    def __init__(self, config: OptimizeConfig):
        self.config = config
        self.losses: list[float] = []
        self.centers: list[tuple[float, float, float]] = []
        self.rotations: list[float] = []
        self.target_field = None
        self.target_field_db_dr = None
        self.init_field = None
        self.final_field = None
        self.initial_loss = None
        self.final_loss = None
        self.final_center = None
        self.final_rotation = None
        self._loss_trace = []
        self._base_vertices = None
        self._base_faces = None
        self._scene = None
        self._tracer = None

    def _create_base_mesh(self):
        cfg = self.config
        base_center = wt.Point3f(0.0, 0.0, float(cfg.init_center[2]))
        vertices, faces = box_drjit_geometry(
            center=base_center,
            size=cfg.cube_size,
            rotation=None,
        ).to_mesh()
        self._base_vertices = vertices
        self._base_faces = faces

    def _apply_transformation(self, center_xy, rotation):
        translate_vec = wt.Vector3f(center_xy.x, center_xy.y, 0.0)
        rotate_axis = wt.Vector3f(0.0, 0.0, 1.0)
        rotation_deg = rotation * wt.Float(180.0 / np.pi)
        trafo = wt.Transform4f().translate(translate_vec).rotate(rotate_axis, rotation_deg)
        return trafo @ self._base_vertices

    def _compute_field(self, center, tx, rotation):
        cfg = self.config
        center_pt = wt.Point3f(float(center[0]), float(center[1]), float(center[2]))
        tx_pt = wt.Point3f(float(tx[0]), float(tx[1]), float(tx[2]))
        rotation_val = wt.Float(float(rotation))
        scene = build_test_scene(box_drjit_geometry(center=center_pt, size=cfg.cube_size, rotation=rotation_val))
        monitor = FieldMonitor(
            DEFAULT_MONITOR_NAME,
            axis="z",
            position=_monitor_height(tx),
            bounds=(cfg.range_x, cfg.range_y),
            grid_size=cfg.grid_size,
        )
        scene.add_monitor(monitor)
        tracer = Tracer(
            frequency=cfg.frequency,
            scene=scene,
            reflection_n_rays=cfg.n_rays,
            reflection_max_bounces=cfg.max_reflections,
            reflection_coef=cfg.reflection_coef,
        )
        result = tracer.trace(tx_pos=tx_pt)
        _assert_plane_monitor_result(result, monitor)
        a_tot = result.primary.field.total
        field_real = np.array(a_tot.real)
        field_imag = np.array(a_tot.imag)
        field_mag = np.sqrt(field_real**2 + field_imag**2)
        return result, field_mag

    def prepare(self):
        cfg = self.config
        _set_seed(cfg.seed)
        self._create_base_mesh()

        self._scene = build_test_scene((self._base_vertices, self._base_faces))
        self._scene.add_monitor(
            FieldMonitor(
                DEFAULT_MONITOR_NAME,
                axis="z",
                position=_monitor_height(cfg.init_tx),
                bounds=(cfg.range_x, cfg.range_y),
                grid_size=cfg.grid_size,
            )
        )
        self._tracer = Tracer(
            frequency=cfg.frequency,
            scene=self._scene,
            reflection_n_rays=cfg.n_rays,
            reflection_max_bounces=cfg.max_reflections,
            reflection_coef=cfg.reflection_coef,
        )

        _, self.target_field = self._compute_field(cfg.target_center, cfg.init_tx, cfg.target_rotation)
        target_field_db = 20 * np.log10(self.target_field + POWER_DB_FLOOR)
        self.target_field_db_dr = wt.Float(target_field_db)

        _, self.init_field = self._compute_field(cfg.init_center, cfg.init_tx, cfg.init_rotation)
        init_field_db = 20 * np.log10(self.init_field + POWER_DB_FLOOR)
        init_loss = float(np.mean((init_field_db - target_field_db) ** 2))

        self.initial_loss = init_loss
        self.losses = [init_loss]
        self.centers = [tuple(cfg.init_center)]
        self.rotations = [cfg.init_rotation]
        self._loss_trace = []

    def optimize(self):
        cfg = self.config
        param_center = wt.Point2f(cfg.init_center[0], cfg.init_center[1])
        param_rotation = wt.Float(cfg.init_rotation)
        dr.enable_grad(param_center, param_rotation)

        opt = wt.ad.Adam(lr=cfg.learning_rate)
        opt["center"] = param_center
        opt["rotation"] = param_rotation

        for i in range(cfg.n_iterations):
            if cfg.use_cosine_annealing:
                lr = cfg.lr_min + 0.5 * (cfg.learning_rate - cfg.lr_min) * (1 + np.cos(np.pi * i / cfg.n_iterations))
                opt.set_learning_rate(lr)

            param_center = opt["center"]
            param_rotation = opt["rotation"]
            dr.set_grad(param_center, wt.Point2f(0.0, 0.0))
            dr.set_grad(param_rotation, wt.Float(0.0))

            vertices = self._apply_transformation(param_center, param_rotation)
            self._scene.update_vertices(vertices, recompute_edges=True)
            result = self._tracer.trace(tx_pos=wt.Point3f(*cfg.init_tx))

            a_tot = result.primary.field.total
            field_mag = dr.sqrt(a_tot.real * a_tot.real + a_tot.imag * a_tot.imag + wt.Float(POWER_DB_FLOOR))
            field_db = dr.log(field_mag + POWER_DB_FLOOR) * (20.0 / np.log(10.0))
            diff = field_db - self.target_field_db_dr
            loss = dr.mean(diff * diff)
            self._loss_trace.append(dr.detach(loss))
            dr.backward(loss)
            opt.step()

        final_center_xy = opt["center"]
        final_rotation = opt["rotation"]
        self.final_center = (_dr_scalar(final_center_xy.x), _dr_scalar(final_center_xy.y), cfg.init_center[2])
        self.final_rotation = _dr_scalar(final_rotation)
        _, self.final_field = self._compute_field(self.final_center, cfg.init_tx, self.final_rotation)
        target_field_db = 20 * np.log10(self.target_field + POWER_DB_FLOOR)
        final_field_db = 20 * np.log10(self.final_field + POWER_DB_FLOOR)
        self.final_loss = float(np.mean((final_field_db - target_field_db) ** 2))
        self.losses = [self.initial_loss] + [_dr_scalar(item) for item in self._loss_trace]
        if self.losses:
            self.losses[-1] = self.final_loss
        self.centers = [tuple(cfg.init_center), self.final_center]
        self.rotations = [cfg.init_rotation, self.final_rotation]

    def visualize(self):
        cfg = self.config
        final_center = self.final_center
        final_rotation = self.final_rotation
        extent = [cfg.range_x[0], cfg.range_x[1], cfg.range_y[0], cfg.range_y[1]]

        gs = cfg.grid_size
        init_db = 20 * np.log10(self.init_field.reshape(gs, gs) + POWER_DB_FLOOR)
        target_db = 20 * np.log10(self.target_field.reshape(gs, gs) + POWER_DB_FLOOR)
        final_db = 20 * np.log10(self.final_field.reshape(gs, gs) + POWER_DB_FLOOR)
        diff_db = 20 * np.log10(np.abs(self.final_field - self.target_field).reshape(gs, gs) + POWER_DB_FLOOR)

        fig = plt.figure(figsize=(16, 8))
        ax_init = fig.add_subplot(2, 4, 1)
        ax_target = fig.add_subplot(2, 4, 2)
        ax_final = fig.add_subplot(2, 4, 3)
        ax_diff = fig.add_subplot(2, 4, 4)
        ax_loss = fig.add_subplot(2, 4, 5)
        ax_traj = fig.add_subplot(2, 4, 6)
        ax_rot = fig.add_subplot(2, 4, 7, projection="polar")
        ax_err = fig.add_subplot(2, 4, 8)

        for ax, image, title in (
            (ax_init, init_db, "Initial Field"),
            (ax_target, target_db, "Target Field"),
            (ax_final, final_db, "Optimized Field"),
        ):
            im = ax.imshow(image, extent=extent, origin="lower", cmap="jet", vmin=-80, vmax=-20)
            ax.scatter([cfg.init_tx[0]], [cfg.init_tx[1]], c="white", s=100, marker="*", edgecolors="black", zorder=5)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlim(cfg.range_x)
            ax.set_ylim(cfg.range_y)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=10)

        im = ax_diff.imshow(diff_db, extent=extent, origin="lower", cmap="RdBu_r", vmin=-100, vmax=-40)
        plt.colorbar(im, ax=ax_diff, shrink=0.8)
        ax_diff.set_xlim(cfg.range_x)
        ax_diff.set_ylim(cfg.range_y)
        ax_diff.set_aspect("equal")
        ax_diff.set_title("|Final - Target| (dB)", fontsize=10)

        ax_loss.plot(self.losses, linewidth=2)
        ax_loss.scatter([0, len(self.losses) - 1], [self.losses[0], self.losses[-1]], c=["green", "blue"], zorder=3)
        ax_loss.set_title("Loss Curve")
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("MSE")
        ax_loss.grid(True, alpha=0.3)

        cx = [item[0] for item in self.centers]
        cy = [item[1] for item in self.centers]
        ax_traj.plot(cx, cy, "b.-", linewidth=1.5, markersize=6)
        ax_traj.scatter([cfg.init_center[0]], [cfg.init_center[1]], c="green", s=120, marker="o", label="Initial")
        ax_traj.scatter([cfg.target_center[0]], [cfg.target_center[1]], c="red", s=160, marker="*", label="Target")
        ax_traj.scatter([final_center[0]], [final_center[1]], c="blue", s=120, marker="s", label="Final")
        ax_traj.set_xlim(cfg.range_x)
        ax_traj.set_ylim(cfg.range_y)
        ax_traj.set_aspect("equal")
        ax_traj.set_title("Center Trajectory")
        ax_traj.grid(True, alpha=0.3)
        ax_traj.legend(fontsize=8)

        radii = np.linspace(0.2, 1.0, len(self.rotations))
        ax_rot.plot(self.rotations, radii, linewidth=1.5)
        ax_rot.scatter([cfg.init_rotation], [0.2], c="green", s=100, marker="o")
        ax_rot.scatter([cfg.target_rotation], [1.0], c="red", s=130, marker="*")
        ax_rot.scatter([final_rotation], [1.0], c="blue", s=100, marker="s")
        ax_rot.set_title("Rotation")

        center_err = [
            np.sqrt((cfg.init_center[0] - cfg.target_center[0]) ** 2 + (cfg.init_center[1] - cfg.target_center[1]) ** 2),
            np.sqrt((final_center[0] - cfg.target_center[0]) ** 2 + (final_center[1] - cfg.target_center[1]) ** 2),
        ]
        rot_err = [
            abs(np.degrees(cfg.init_rotation - cfg.target_rotation)),
            abs(np.degrees(final_rotation - cfg.target_rotation)),
        ]
        ax_err.plot(["Initial", "Final"], center_err, label="Center", marker="o")
        ax_err.plot(["Initial", "Final"], rot_err, label="Rotation (deg)", marker="s")
        ax_err.set_title("Parameter Error")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(fontsize=8)

        fig.suptitle(
            "Main Optimization Visual Test\n"
            f"target_center={cfg.target_center[:2]}, final_center=({final_center[0]:.2f}, {final_center[1]:.2f}), "
            f"target_rot={np.degrees(cfg.target_rotation):.1f}, final_rot={np.degrees(final_rotation):.1f}",
            fontsize=12,
        )
        plt.tight_layout()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.output_path, dpi=160)
        plt.close(fig)

    def run(self):
        self.prepare()
        self.optimize()
        self.visualize()


def test_optimize_main_visual():
    optimizer = RadioFieldOptimizer(OptimizeConfig())
    optimizer.run()

    assert OUTPUT_PATH.exists()
    assert OUTPUT_PATH.stat().st_size > 0
    assert optimizer.final_loss is not None
    assert optimizer.initial_loss is not None
    assert optimizer.final_loss < optimizer.initial_loss
