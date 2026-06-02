"""Inverse optimization via differentiable simulation."""
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Tuple, List

try:
    from ._paths import FIGURES_DIR, maybe_show
    from ._monitor import DEFAULT_MONITOR_NAME, assert_plane_monitor_result, monitor_height
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _paths import FIGURES_DIR, maybe_show
    from _monitor import DEFAULT_MONITOR_NAME, assert_plane_monitor_result, monitor_height

import matplotlib.pyplot as plt
import numpy as np
import witwin as wt
import drjit as dr
from tqdm import tqdm
from tests._scene_helpers import (
    box_drjit_geometry,
    build_scene as build_test_scene,
    prism_drjit_geometry,
)
from witwin.channel import (
    DEFAULT_VARIANT,
    FieldMonitor,
    POWER_DB_FLOOR,
    Tracer,
    scalar,
)


def set_seed(seed: int = 42):
    """Freeze all random number generators for reproducibility."""
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


@dataclass
class OptimizeConfig:
    """Optimization config."""

    # Simulation parameters
    grid_size: int = 256
    frequency: float = 1e9
    range_x: Tuple[float, float] = (-8, 8)
    range_y: Tuple[float, float] = (-8, 8)
    n_rays: int = 100000
    max_reflections: int = 1
    reflection_coef: float = 1.0

    # Geometry parameters
    geometry_type: str = "cube"  # "cube" or "prism"
    cube_size: float = 4.0       # for cube: side length
    prism_n_sides: int = 5       # for prism: number of sides (3=triangle, 4=square, 5=pentagon, etc.)
    prism_radius: float = 2.0    # for prism: circumradius
    prism_height: float = 4.0    # for prism: height

    # Optimization toggles
    optimize_center: bool = True
    optimize_tx: bool = True
    optimize_rotation: bool = True

    # Target parameters
    target_center: Tuple[float, float, float] = (3.0, -2.5, 2.0)
    target_tx: Tuple[float, float, float] = (-3.0, -4.0, 1.5)
    target_rotation: float = np.pi / 5  # 36 deg

    # Initial parameters
    init_center: Tuple[float, float, float] = (0.0, 0.0, 2.0)
    init_tx: Tuple[float, float, float] = (-5.0, 5.0, 1.5)
    init_rotation: float = 0.0

    # Default values (used when not optimizing)
    default_center: Tuple[float, float, float] = (0.0, 0.0, 2.0)
    default_tx: Tuple[float, float, float] = (-5.0, 5.0, 1.5)
    default_rotation: float = 0.0

    # Optimizer settings
    learning_rate: float = 0.2
    lr_min: float = 1e-3
    use_cosine_annealing: bool = False
    n_iterations: int = 500

    # Output
    output_path: str = str(FIGURES_DIR / "optimize.png")
    seed: int = 42

    def __post_init__(self):
        if not self.optimize_center:
            self.target_center = self.default_center
            self.init_center = self.default_center
        if not self.optimize_tx:
            self.target_tx = self.default_tx
            self.init_tx = self.default_tx
        if not self.optimize_rotation:
            self.target_rotation = self.default_rotation
            self.init_rotation = self.default_rotation

    @property
    def optimized_params(self) -> List[str]:
        params = []
        if self.optimize_center:
            params.append("Center")
        if self.optimize_tx:
            params.append("TX")
        if self.optimize_rotation:
            params.append("Rotation")
        return params


class RadioFieldOptimizer:
    """Optimize scene parameters from target field."""

    def __init__(self, config: OptimizeConfig):
        self.config = config
        self.losses: List[float] = []
        self.centers: List[Tuple[float, float, float]] = []
        self.tx_positions: List[Tuple[float, float, float]] = []
        self.rotations: List[float] = []
        self.stopped_early = False

        # Fields computed during preparation
        self.target_field: Optional[np.ndarray] = None
        self.target_field_db_dr: Optional[wt.Float] = None
        self.init_field: Optional[np.ndarray] = None
        self.final_field: Optional[np.ndarray] = None

        # Base mesh (created once, transformed during optimization)
        self._base_vertices: Optional[wt.Point3f] = None  # Vertices at origin with no rotation
        self._base_faces: Optional[wt.Vector3u] = None    # Topology (unchanged)
        self._base_center_z: float = 0.0                  # Z offset for base mesh

        # Tracer instance (created once, updated via update_scene())
        self._tracer: Optional[Tracer] = None

    def _create_base_mesh(self):
        """Create base mesh at origin with no rotation (called once)."""
        cfg = self.config
        # Create mesh centered at (0, 0, center_z) with no rotation
        base_center = wt.Point3f(0.0, 0.0, float(cfg.init_center[2]))
        self._base_center_z = float(cfg.init_center[2])

        if cfg.geometry_type == "cube":
            vertices, faces = box_drjit_geometry(
                center=base_center,
                size=cfg.cube_size,
                rotation=None  # No rotation for base mesh
            ).to_mesh()
        elif cfg.geometry_type == "prism":
            vertices, faces = prism_drjit_geometry(
                n_sides=cfg.prism_n_sides,
                center=base_center,
                radius=cfg.prism_radius,
                height=cfg.prism_height,
                rotation=None  # No rotation for base mesh
            ).to_mesh()
        else:
            raise ValueError(f"Unknown geometry_type: {cfg.geometry_type}")

        self._base_vertices = vertices
        self._base_faces = faces

    def _apply_transformation(self, center_xy, rotation):
        """
        Apply transformation to base vertices.

        Args:
            center_xy: wt.Point2f - XY translation (Z is fixed from base mesh)
            rotation: wt.Float - Rotation angle around Z-axis in radians

        Returns:
            vertices: wt.Point3f - Transformed vertices (differentiable)
            faces: wt.Vector3u - Faces (unchanged topology)
        """
        # Build transformation: first rotate around Z-axis, then translate.
        translate_vec = wt.Vector3f(center_xy.x, center_xy.y, 0.0)
        rotate_axis = wt.Vector3f(0.0, 0.0, 1.0)
        if rotation is not None:
            rotation_deg = rotation * wt.Float(180.0 / np.pi)
            trafo = wt.Transform4f().translate(translate_vec).rotate(rotate_axis, rotation_deg)
        else:
            trafo = wt.Transform4f().translate(translate_vec)

        # Apply transformation to base vertices
        transformed_vertices = trafo @ self._base_vertices

        return transformed_vertices, self._base_faces

    def _create_mesh(self, center_pt, rotation):
        """Create mesh based on geometry_type config for one-shot field evaluation."""
        cfg = self.config
        if cfg.geometry_type == "cube":
            return box_drjit_geometry(center=center_pt, size=cfg.cube_size, rotation=rotation).to_mesh()
        elif cfg.geometry_type == "prism":
            return prism_drjit_geometry(
                n_sides=cfg.prism_n_sides,
                center=center_pt,
                radius=cfg.prism_radius,
                height=cfg.prism_height,
                rotation=rotation
            ).to_mesh()
        else:
            raise ValueError(f"Unknown geometry_type: {cfg.geometry_type}")

    def _compute_field(self, center, tx, rotation) -> Tuple[dict, np.ndarray]:
        cfg = self.config
        center_pt = wt.Point3f(float(center[0]), float(center[1]), float(center[2]))
        tx_pt = wt.Point3f(float(tx[0]), float(tx[1]), float(tx[2]))
        rot = wt.Float(float(rotation)) if rotation is not None else None

        vertices, faces = self._create_mesh(center_pt, rot)
        scene = build_test_scene((vertices, faces))
        monitor = FieldMonitor(
            DEFAULT_MONITOR_NAME,
            axis="z",
            position=monitor_height(tx),
            bounds=(cfg.range_x, cfg.range_y),
            grid_size=cfg.grid_size,
        )
        scene.add_monitor(monitor)
        tracer = Tracer(
            frequency=cfg.frequency,
            scene=scene,
            reflection_n_rays=cfg.n_rays,
            reflection_max_bounces=cfg.max_reflections,
            reflection_coef=cfg.reflection_coef
        )
        result = tracer.trace(tx_pos=tx_pt)
        assert_plane_monitor_result(result, monitor)

        a_tot = result.primary.field.total
        field_real = np.array(a_tot.real)
        field_imag = np.array(a_tot.imag)
        field_mag = np.sqrt(field_real**2 + field_imag**2)

        return result, field_mag

    def prepare(self):
        cfg = self.config
        set_seed(cfg.seed)

        lr_str = f"{cfg.learning_rate}->{cfg.lr_min}" if cfg.use_cosine_annealing else f"{cfg.learning_rate}"
        geom_str = cfg.geometry_type
        if cfg.geometry_type == "prism":
            geom_str = f"prism({cfg.prism_n_sides}-sided)"
        print(f"[Config] {cfg.grid_size}x{cfg.grid_size}, {cfg.frequency/1e9:.1f}GHz, geom={geom_str}, "
              f"opt=[{','.join(cfg.optimized_params)}], lr={lr_str}, iter={cfg.n_iterations}")

        # Create base mesh once (at origin, no rotation)
        self._create_base_mesh()

        # Create Scene + Tracer once (scene reused, vertices updated each step)
        self._scene = build_test_scene((self._base_vertices, self._base_faces))
        self._scene.add_monitor(
            FieldMonitor(
                DEFAULT_MONITOR_NAME,
                axis="z",
                position=monitor_height(cfg.init_tx),
                bounds=(cfg.range_x, cfg.range_y),
                grid_size=cfg.grid_size,
            )
        )
        self._tracer = Tracer(
            frequency=cfg.frequency,
            scene=self._scene,
            reflection_n_rays=cfg.n_rays,
            reflection_max_bounces=cfg.max_reflections,
            reflection_coef=cfg.reflection_coef
        )

        # Compute target and initial fields
        print("[1/2] Computing target field...", end=" ")
        _, self.target_field = self._compute_field(cfg.target_center, cfg.target_tx, cfg.target_rotation)
        target_field_db = 20 * np.log10(self.target_field + POWER_DB_FLOOR)
        self.target_field_db_dr = wt.Float(target_field_db)
        print("done")

        print("[2/2] Computing initial field...", end=" ")
        _, self.init_field = self._compute_field(cfg.init_center, cfg.init_tx, cfg.init_rotation)
        init_field_db = 20 * np.log10(self.init_field + POWER_DB_FLOOR)
        init_loss = float(np.mean((init_field_db - target_field_db)**2))
        print(f"done (loss={init_loss:.4f})")

        # Initialize history
        self.losses = [init_loss]
        self.centers = [tuple(cfg.init_center)]
        self.tx_positions = [tuple(cfg.init_tx)]
        self.rotations = [cfg.init_rotation]

    def optimize(self):
        cfg = self.config

        # Create optimizable parameters
        param_center = wt.Point2f(cfg.init_center[0], cfg.init_center[1])
        param_cz = wt.Float(cfg.init_center[2])
        param_tx = wt.Point2f(cfg.init_tx[0], cfg.init_tx[1])
        param_tx_z = wt.Float(cfg.init_tx[2])
        param_rotation = wt.Float(cfg.init_rotation)

        # Enable gradients
        if cfg.optimize_center:
            dr.enable_grad(param_center)
        if cfg.optimize_tx:
            dr.enable_grad(param_tx)
        if cfg.optimize_rotation:
            dr.enable_grad(param_rotation)

        # Create Adam optimizer
        opt = wt.ad.Adam(lr=cfg.learning_rate)
        if cfg.optimize_center:
            opt['center'] = param_center
        if cfg.optimize_tx:
            opt['tx'] = param_tx
        if cfg.optimize_rotation:
            opt['rotation'] = param_rotation

        # Optimization loop
        print("\n[3] Running Adam optimization...")
        pbar = tqdm(range(cfg.n_iterations), desc="Optimizing", unit="iter")

        # Timing accumulators
        timing_accum = defaultdict(float)
        timing_count = 0
        timing_interval = 10  # Print every N iterations

        for i in pbar:
            try:
                iter_t0 = time.perf_counter()

                if cfg.use_cosine_annealing:
                    lr = cfg.lr_min + 0.5 * (cfg.learning_rate - cfg.lr_min) * (
                        1 + np.cos(np.pi * i / cfg.n_iterations)
                    )
                    opt.set_learning_rate(lr)

                if cfg.optimize_center:
                    dr.set_grad(param_center, wt.Point2f(0.0, 0.0))
                if cfg.optimize_tx:
                    dr.set_grad(param_tx, wt.Point2f(0.0, 0.0))
                if cfg.optimize_rotation:
                    dr.set_grad(param_rotation, wt.Float(0.0))

                # Use detached values when not optimizing to prevent gradient accumulation
                if cfg.optimize_tx:
                    tx_pos = wt.Point3f(param_tx.x, param_tx.y, param_tx_z)
                else:  # Use fixed constant values
                    tx_pos = wt.Point3f(cfg.init_tx[0], cfg.init_tx[1], cfg.init_tx[2]) 

                rotation = param_rotation if cfg.optimize_rotation else None

                if cfg.optimize_center:
                    center_for_transform = param_center
                else:  # Use fixed constant values - prevents any gradient flow to center
                    center_for_transform = wt.Point2f(cfg.init_center[0], cfg.init_center[1]) 

                # Apply transformation to base mesh (no mesh recreation)
                vertices, _ = self._apply_transformation(center_for_transform, rotation)

                # Update scene vertices (reuse Scene + Tracer, avoid rebuild)
                t0 = time.perf_counter()
                self._scene.update_vertices(vertices, recompute_edges=True)
                timing_accum['scene_update'] += time.perf_counter() - t0

                # Trace with updated scene (with timing)
                result = self._tracer.trace(
                    tx_pos=tx_pos,
                    return_timing=True,
                )

                # Accumulate trace timing
                if result.primary.timing is not None:
                    for k, v in result.primary.timing.items():
                        timing_accum[k] += v

                # Loss computation
                t0 = time.perf_counter()
                a_tot = result.primary.field.total
                field_mag = dr.sqrt(a_tot.real * a_tot.real + a_tot.imag * a_tot.imag + wt.Float(POWER_DB_FLOOR))
                field_db = dr.log(field_mag + POWER_DB_FLOOR) * (20.0 / np.log(10.0))
                diff = field_db - self.target_field_db_dr
                loss = dr.mean(diff * diff)
                timing_accum['loss_compute'] += time.perf_counter() - t0

                # Backward pass
                t0 = time.perf_counter()
                backward_ok = True
                try:
                    dr.backward(loss)
                except RuntimeError as e:
                    if "AllowNoGrad" in str(e) or "does not depend" in str(e):
                        print(f"    [WARN] Iter {i+1}: No gradient, skipping update")
                        backward_ok = False
                    else:
                        raise
                timing_accum['backward'] += time.perf_counter() - t0

                # Optimizer step
                t0 = time.perf_counter()
                if backward_ok:
                    opt.step()
                    if cfg.optimize_center:
                        param_center = opt['center']
                    if cfg.optimize_tx:
                        param_tx = opt['tx']
                    if cfg.optimize_rotation:
                        param_rotation = opt['rotation']
                timing_accum['opt_step'] += time.perf_counter() - t0

                # Scalar extraction (for logging)
                t0 = time.perf_counter()
                loss_val = scalar(loss)
                cx = scalar(param_center.x)
                cy = scalar(param_center.y)
                cz = scalar(param_cz)
                tx_x = scalar(param_tx.x)
                tx_y = scalar(param_tx.y)
                tx_z = scalar(param_tx_z)
                rot = scalar(param_rotation)
                timing_accum['scalar_sync'] += time.perf_counter() - t0

                self.losses.append(loss_val)
                self.centers.append((cx, cy, cz))
                self.tx_positions.append((tx_x, tx_y, tx_z))
                self.rotations.append(rot)

                timing_accum['total'] += time.perf_counter() - iter_t0
                timing_count += 1

                # Print timing stats every N iterations
                if (i + 1) % timing_interval == 0:
                    avg_total = timing_accum['total'] / timing_count * 1000
                    abbrev = {'scene_update': 'scene', 'edge_cache': 'edge', 'reflection': 'ref',
                              'diffraction': 'dif', 'loss_compute': 'loss', 'backward': 'bwd',
                              'opt_step': 'opt', 'scalar_sync': 'sync'}
                    stats = []
                    for key in ['scene_update', 'los', 'reflection', 'diffraction', 'backward']:
                        if key in timing_accum:
                            avg_ms = timing_accum[key] / timing_count * 1000
                            pct = timing_accum[key] / timing_accum['total'] * 100
                            stats.append(f"{abbrev.get(key, key)}={avg_ms:.0f}({pct:.0f}%)")
                    tqdm.write(f"  [T@{i+1}] {avg_total:.0f}ms | " + " ".join(stats))

                pbar.set_postfix(loss=f"{loss_val:.4f}")

            except RuntimeError as e:
                print(f"\n[!] Stopped at iter {i+1}: {e}")
                self.stopped_early = True
                break

        print("Computing final field...", end=" ")
        final_center = self.centers[-1]
        final_tx = self.tx_positions[-1]
        final_rotation = self.rotations[-1]
        _, self.final_field = self._compute_field(final_center, final_tx, final_rotation)
        print("done")
        self._print_summary()

    def _print_summary(self):
        """Print optimization results summary."""
        cfg = self.config
        fc, ft, fr = self.centers[-1], self.tx_positions[-1], self.rotations[-1]
        status = " (early stop)" if self.stopped_early else ""

        # Center error
        c_err = (abs(fc[0]-cfg.target_center[0]), abs(fc[1]-cfg.target_center[1]))
        print(f"[Result{status}] Center: ({fc[0]:.2f},{fc[1]:.2f}) err=({c_err[0]:.3f},{c_err[1]:.3f})", end="")

        if cfg.optimize_tx:
            tx_err = (abs(ft[0]-cfg.target_tx[0]), abs(ft[1]-cfg.target_tx[1]))
            print(f" | TX: ({ft[0]:.2f},{ft[1]:.2f}) err=({tx_err[0]:.3f},{tx_err[1]:.3f})", end="")

        if cfg.optimize_rotation:
            rot_err = abs(np.degrees(fr - cfg.target_rotation))
            print(f" | Rot: {np.degrees(fr):.1f} err={rot_err:.2f}", end="")

        reduction = (1 - self.losses[-1]/self.losses[0])*100
        print(f"\n[Loss] {self.losses[0]:.4f} -> {self.losses[-1]:.4f} ({reduction:.1f}% reduction)")

    def visualize(self):
        """Generate visualization plots."""
        cfg = self.config
        print("Generating visualization...", end=" ")

        final_center = self.centers[-1]
        final_tx = self.tx_positions[-1]
        final_rotation = self.rotations[-1]
        extent = [cfg.range_x[0], cfg.range_x[1], cfg.range_y[0], cfg.range_y[1]]

        # Convert to dB
        gs = cfg.grid_size
        init_db = 20 * np.log10(self.init_field.reshape(gs, gs) + POWER_DB_FLOOR)
        target_db = 20 * np.log10(self.target_field.reshape(gs, gs) + POWER_DB_FLOOR)
        final_db = 20 * np.log10(self.final_field.reshape(gs, gs) + POWER_DB_FLOOR)
        diff_db = 20 * np.log10(np.abs(self.final_field - self.target_field).reshape(gs, gs) + POWER_DB_FLOOR)
        vmin, vmax = -80, -20

        def rot_str(r):
            return f" rot={np.degrees(r):.0f}" if cfg.optimize_rotation else ""

        def draw_scene(ax, center, tx, show_target=False):
            ax.scatter([tx[0]], [tx[1]], c='white', s=100, marker='*', edgecolors='black', zorder=5)
            ax.scatter([center[0]], [center[1]], c='red', s=150, marker='x', linewidths=3, zorder=6)
            if show_target:
                if cfg.optimize_tx:
                    ax.scatter([cfg.target_tx[0]], [cfg.target_tx[1]], c='yellow', s=80, marker='*',
                              edgecolors='orange', zorder=4, alpha=0.7)
                ax.scatter([cfg.target_center[0]], [cfg.target_center[1]], c='yellow', s=100, marker='x',
                          linewidths=2, zorder=4, alpha=0.7)
            ax.set_xlim(cfg.range_x)
            ax.set_ylim(cfg.range_y)
            ax.set_aspect('equal')

        fig = plt.figure(figsize=(20, 10))

        ax_init = fig.add_subplot(2, 4, 1)
        ax_target = fig.add_subplot(2, 4, 2)
        ax_final = fig.add_subplot(2, 4, 3)
        ax_diff = fig.add_subplot(2, 4, 4)

        ax_loss = fig.add_subplot(2, 4, 5)
        ax_traj = fig.add_subplot(2, 4, 6)
        ax_rot = fig.add_subplot(2, 4, 7, projection='polar') if cfg.optimize_rotation else fig.add_subplot(2, 4, 7)
        ax_err = fig.add_subplot(2, 4, 8)

        im1 = ax_init.imshow(init_db, extent=extent, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        draw_scene(ax_init, cfg.init_center, cfg.init_tx)
        ax_init.set_title(f'Initial Field\nc=({cfg.init_center[0]:.1f},{cfg.init_center[1]:.1f}) '
                         f'tx=({cfg.init_tx[0]:.1f},{cfg.init_tx[1]:.1f}){rot_str(cfg.init_rotation)}', fontsize=10)
        plt.colorbar(im1, ax=ax_init, shrink=0.8)

        im2 = ax_target.imshow(target_db, extent=extent, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        draw_scene(ax_target, cfg.target_center, cfg.target_tx)
        ax_target.set_title(f'Target Field\nc=({cfg.target_center[0]:.1f},{cfg.target_center[1]:.1f}) '
                           f'tx=({cfg.target_tx[0]:.1f},{cfg.target_tx[1]:.1f}){rot_str(cfg.target_rotation)}', fontsize=10)
        plt.colorbar(im2, ax=ax_target, shrink=0.8)

        im3 = ax_final.imshow(final_db, extent=extent, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        draw_scene(ax_final, final_center, final_tx, show_target=True)
        suffix = " (early stop)" if self.stopped_early else ""
        ax_final.set_title(f'Optimized Field{suffix}\nc=({final_center[0]:.2f},{final_center[1]:.2f}) '
                          f'tx=({final_tx[0]:.2f},{final_tx[1]:.2f}){rot_str(final_rotation)}', fontsize=10)
        plt.colorbar(im3, ax=ax_final, shrink=0.8)

        im4 = ax_diff.imshow(diff_db, extent=extent, origin='lower', cmap='RdBu_r', vmin=-100, vmax=-40)
        ax_diff.set_title('|Final - Target| (dB)', fontsize=11)
        ax_diff.set_xlim(cfg.range_x)
        ax_diff.set_ylim(cfg.range_y)
        ax_diff.set_aspect('equal')
        plt.colorbar(im4, ax=ax_diff, shrink=0.8)

        ax_loss.semilogy(self.losses, 'b-', linewidth=2)
        if len(self.losses) > 10:
            y_min = min(self.losses[10:]) * 0.9
            y_max = max(self.losses[10:]) * 1.1
            ax_loss.set_ylim(y_min, y_max)
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('MSE Loss (dB)')
        ax_loss.set_title('Loss Curve')
        ax_loss.grid(True, alpha=0.3)

        cx = [c[0] for c in self.centers]
        cy = [c[1] for c in self.centers]
        ax_traj.plot(cx, cy, 'b.-', linewidth=1, markersize=3, alpha=0.6)
        ax_traj.scatter([cfg.init_center[0]], [cfg.init_center[1]], c='green', s=150, marker='o', zorder=6)
        ax_traj.scatter([final_center[0]], [final_center[1]], c='blue', s=150, marker='s', zorder=7)
        ax_traj.scatter([cfg.target_center[0]], [cfg.target_center[1]], c='red', s=200, marker='*', zorder=10)

        if cfg.optimize_tx:
            tx_x = [t[0] for t in self.tx_positions]
            tx_y = [t[1] for t in self.tx_positions]
            ax_traj.plot(tx_x, tx_y, 'm.-', linewidth=1, markersize=3, alpha=0.6)
            ax_traj.scatter([cfg.init_tx[0]], [cfg.init_tx[1]], c='green', s=150, marker='^', zorder=6)
            ax_traj.scatter([final_tx[0]], [final_tx[1]], c='blue', s=150, marker='^', zorder=7)
            ax_traj.scatter([cfg.target_tx[0]], [cfg.target_tx[1]], c='red', s=200, marker='^', zorder=10)

        ax_traj.scatter([], [], c='green', s=80, marker='o', label='Initial')
        ax_traj.scatter([], [], c='red', s=80, marker='*', label='Target')
        ax_traj.scatter([], [], c='blue', s=80, marker='s', label='Final')
        ax_traj.set_xlim(cfg.range_x)
        ax_traj.set_ylim(cfg.range_y)
        ax_traj.set_xlabel('X')
        ax_traj.set_ylabel('Y')
        ax_traj.set_title('Parameter Trajectories')
        ax_traj.legend(fontsize=8, loc='upper right')
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_aspect('equal')

        if cfg.optimize_rotation:
            rotations_rad = np.array(self.rotations)
            n_points = len(rotations_rad)
            radii = np.linspace(0.3, 1.0, n_points)
            colors = plt.cm.viridis(np.linspace(0, 1, n_points))
            for i in range(n_points - 1):
                ax_rot.plot([rotations_rad[i], rotations_rad[i+1]], [radii[i], radii[i+1]], color=colors[i], linewidth=2)
            ax_rot.scatter([cfg.init_rotation], [0.3], c='green', s=150, marker='o', zorder=6, label='Initial')
            ax_rot.scatter([final_rotation], [1.0], c='blue', s=150, marker='s', zorder=7, label='Final')
            ax_rot.scatter([cfg.target_rotation], [1.0], c='red', s=200, marker='*', zorder=10, label='Target')
            ax_rot.plot([cfg.target_rotation, cfg.target_rotation], [0, 1.1], 'r--', linewidth=2, alpha=0.7)
            ax_rot.set_title('Rotation (polar)')
            ax_rot.set_ylim(0, 1.2)
            ax_rot.legend(fontsize=7, loc='upper right')
        else:
            ax_rot.axis('off')

        center_err = [np.sqrt((c[0]-cfg.target_center[0])**2 + (c[1]-cfg.target_center[1])**2) for c in self.centers]
        ax_err.plot(center_err, 'b-', linewidth=2, label='Center')
        if cfg.optimize_tx:
            tx_err = [np.sqrt((t[0]-cfg.target_tx[0])**2 + (t[1]-cfg.target_tx[1])**2) for t in self.tx_positions]
            ax_err.plot(tx_err, 'm-', linewidth=2, label='TX')
        if cfg.optimize_rotation:
            rot_err = [abs(np.degrees(r - cfg.target_rotation)) for r in self.rotations]
            ax_err.plot(rot_err, 'g-', linewidth=2, label='Rotation (deg)')
        ax_err.set_xlabel('Iteration')
        ax_err.set_ylabel('Error')
        ax_err.set_title('Parameter Error')
        ax_err.legend(fontsize=8)
        ax_err.grid(True, alpha=0.3)

        status = " (early stop)" if self.stopped_early else ""
        params_str = " + ".join(cfg.optimized_params)
        title = f'Inverse Optimization: {params_str}{status}\n'
        title += f'Center: ({cfg.target_center[0]:.1f},{cfg.target_center[1]:.1f})->({final_center[0]:.2f},{final_center[1]:.2f})'
        if cfg.optimize_tx:
            title += f', TX: ({cfg.target_tx[0]:.1f},{cfg.target_tx[1]:.1f})->({final_tx[0]:.2f},{final_tx[1]:.2f})'
        if cfg.optimize_rotation:
            title += f', Rot: {np.degrees(cfg.target_rotation):.0f}->{np.degrees(final_rotation):.1f}'

        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"done -> {output_path}")
        maybe_show()

    def run(self):
        """Run full optimization pipeline: prepare -> optimize -> visualize."""
        self.prepare()
        self.optimize()
        self.visualize()


def main():
    config = OptimizeConfig(
        # Simulation
        grid_size=256,
        frequency=1e9,
        n_rays=1000000,

        # Geometry: "cube" or "prism"
        geometry_type="cube",       # or "prism"
        cube_size=4.0,              # for cube
        prism_n_sides=5,            # for prism: 3=triangle, 4=square, 5=pentagon, 6=hexagon, etc.
        prism_radius=2.0,           # for prism
        prism_height=4.0,           # for prism

        # What to optimize
        optimize_center=True,
        optimize_tx=False,
        optimize_rotation=True,

        # Target scene
        target_center=(3.0, -2.5, 2.0),
        target_tx=(-3.0, -4.0, 1.5),
        target_rotation=np.pi / 5 + np.pi * 1.5,  # 36 deg + 270 deg

        # Initial guess
        init_center=(0.0, 0.0, 2.0),
        init_tx=(-5.0, 5.0, 1.5),
        init_rotation=0.0,

        # Optimizer
        learning_rate=0.2,
        n_iterations=500,
        use_cosine_annealing=True,
    )

    optimizer = RadioFieldOptimizer(config)
    optimizer.run()


if __name__ == "__main__":
    main()


