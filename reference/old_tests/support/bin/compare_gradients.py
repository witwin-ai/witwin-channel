"""Compare AD gradient magnitude for position, rotation, transmitter."""
from pathlib import Path
import sys
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
from scipy.ndimage import uniform_filter
from tests._scene_helpers import box_drjit_geometry, build_scene as build_test_scene
from witwin.channel import DEFAULT_VARIANT, FieldMonitor, Scene, Tracer, draw_scene, to_numpy, to_power_db
def compute_ssim(img1, img2, data_range=1.0, win_size=7):
    """Compute SSIM between two images."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = uniform_filter(img1.astype(np.float64), size=win_size)
    mu2 = uniform_filter(img2.astype(np.float64), size=win_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1.astype(np.float64) ** 2, size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(img2.astype(np.float64) ** 2, size=win_size) - mu2_sq
    sigma12 = uniform_filter(img1.astype(np.float64) * img2.astype(np.float64), size=win_size) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def compute_psnr(img_true, img_pred, data_range=1.0):
    """Compute PSNR between two images."""
    mse = np.mean((img_true - img_pred) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10((data_range ** 2) / mse))


# =============================================================================
# Common parameters
# =============================================================================
GRID_SIZE = 512
FREQ = 1e9
TX_POS = (-5.0, 5.0, 1.5)
RANGE_X, RANGE_Y = (-8, 8), (-8, 8)
CENTER = (0.0, 0.0, 2.0)
SIZE = 4.0
N_RAYS = 1000000
MAX_REFLECTIONS = 1
REFLECTION_COEF = 1.0
ROTATION_VAL = np.deg2rad(15)


# =============================================================================
# Position gradient functions
# =============================================================================
def compute_field_position(center, size, freq, tx_pos, range_x, range_y, grid_size,
                           n_rays, max_reflections, reflection_coef):
    scene = build_test_scene(box_drjit_geometry(center=center, size=size))
    monitor = FieldMonitor(
        DEFAULT_MONITOR_NAME,
        axis="z",
        position=monitor_height(tx_pos),
        bounds=(range_x, range_y),
        grid_size=grid_size,
    )
    scene.add_monitor(monitor)
    tracer = Tracer(
        frequency=freq, scene=scene, reflection_n_rays=n_rays,
        reflection_max_bounces=max_reflections, reflection_coef=reflection_coef
    )
    result = tracer.trace(tx_pos=tx_pos)
    assert_plane_monitor_result(result, monitor)
    return result, scene


def compute_ad_grad_position(center_vals):
    def compute_component_grad(target_part):
        center = wt.Point3f(float(center_vals[0]), float(center_vals[1]), float(center_vals[2]))
        dr.enable_grad(center)
        dr.set_grad(center, wt.Vector3f(1.0, 0.0, 0.0))
        scene = build_test_scene(box_drjit_geometry(center=center, size=SIZE))
        monitor = FieldMonitor(
            DEFAULT_MONITOR_NAME,
            axis="z",
            position=monitor_height(TX_POS),
            bounds=(RANGE_X, RANGE_Y),
            grid_size=GRID_SIZE,
        )
        scene.add_monitor(monitor)
        tracer = Tracer(frequency=FREQ, scene=scene, reflection_n_rays=N_RAYS,
                        reflection_max_bounces=MAX_REFLECTIONS, reflection_coef=REFLECTION_COEF)
        result = tracer.trace(tx_pos=TX_POS)
        assert_plane_monitor_result(result, monitor)
        a_tot = result.primary.field.total
        field = a_tot.real if target_part == 'real' else a_tot.imag
        dr.forward_to(field, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        grad = dr.grad(field)
        return result, scene, to_numpy(grad) if grad is not None else np.zeros(GRID_SIZE * GRID_SIZE)

    result, scene, grad_re = compute_component_grad('real')
    _, _, grad_im = compute_component_grad('imag')
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)
    return result, scene, grad_mag


def compute_fd_grad_position(center_vals, delta=0.01):
    center_base = wt.Point3f(float(center_vals[0]), float(center_vals[1]), float(center_vals[2]))
    result_base, scene = compute_field_position(center_base, SIZE, FREQ, TX_POS, RANGE_X, RANGE_Y,
                                                 GRID_SIZE, N_RAYS, MAX_REFLECTIONS, REFLECTION_COEF)
    a_tot_base = result_base.primary.field.total
    re_base = to_numpy(a_tot_base.real)
    im_base = to_numpy(a_tot_base.imag)

    center_perturbed = wt.Point3f(float(center_vals[0]) + delta, float(center_vals[1]), float(center_vals[2]))
    result_perturbed, _ = compute_field_position(center_perturbed, SIZE, FREQ, TX_POS, RANGE_X, RANGE_Y,
                                                  GRID_SIZE, N_RAYS, MAX_REFLECTIONS, REFLECTION_COEF)
    a_tot_perturbed = result_perturbed.primary.field.total
    re_perturbed = to_numpy(a_tot_perturbed.real)
    im_perturbed = to_numpy(a_tot_perturbed.imag)

    grad_re = (re_perturbed - re_base) / delta
    grad_im = (im_perturbed - im_base) / delta
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)
    return result_base, scene, grad_mag


# =============================================================================
# Rotation gradient functions
# =============================================================================
def compute_field_rotation(rotation, center, size, freq, tx_pos, range_x, range_y, grid_size,
                           n_rays, max_reflections, reflection_coef):
    scene = build_test_scene(box_drjit_geometry(center=wt.Point3f(*center), size=size, rotation=rotation))
    monitor = FieldMonitor(
        DEFAULT_MONITOR_NAME,
        axis="z",
        position=monitor_height(tx_pos),
        bounds=(range_x, range_y),
        grid_size=grid_size,
    )
    scene.add_monitor(monitor)
    tracer = Tracer(frequency=freq, scene=scene, reflection_n_rays=n_rays,
                    reflection_max_bounces=max_reflections, reflection_coef=reflection_coef)
    result = tracer.trace(tx_pos=tx_pos)
    assert_plane_monitor_result(result, monitor)
    return result, scene


def compute_ad_grad_rotation(rotation_val):
    def compute_component_grad(target_part):
        rotation = wt.Float(float(rotation_val))
        dr.enable_grad(rotation)
        dr.set_grad(rotation, wt.Float(1.0))
        result, scene = compute_field_rotation(rotation, CENTER, SIZE, FREQ, TX_POS, RANGE_X, RANGE_Y,
                                                GRID_SIZE, N_RAYS, MAX_REFLECTIONS, REFLECTION_COEF)
        a = result.primary.field.total
        field = a.real if target_part == 'real' else a.imag
        dr.forward_to(field, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        grad = dr.grad(field)
        return result, scene, to_numpy(grad) if grad is not None else np.zeros(GRID_SIZE * GRID_SIZE)

    result, scene, grad_re = compute_component_grad('real')
    _, _, grad_im = compute_component_grad('imag')
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)
    return result, scene, grad_mag


def compute_fd_grad_rotation(rotation_val, delta=0.01):
    rotation_base = wt.Float(float(rotation_val))
    result_base, scene = compute_field_rotation(rotation_base, CENTER, SIZE, FREQ, TX_POS, RANGE_X, RANGE_Y,
                                                 GRID_SIZE, N_RAYS, MAX_REFLECTIONS, REFLECTION_COEF)
    a_base = result_base.primary.field.total
    re_base = to_numpy(a_base.real)
    im_base = to_numpy(a_base.imag)

    rotation_perturbed = wt.Float(float(rotation_val) + delta)
    result_perturbed, _ = compute_field_rotation(rotation_perturbed, CENTER, SIZE, FREQ, TX_POS, RANGE_X, RANGE_Y,
                                                  GRID_SIZE, N_RAYS, MAX_REFLECTIONS, REFLECTION_COEF)
    a_perturbed = result_perturbed.primary.field.total
    re_perturbed = to_numpy(a_perturbed.real)
    im_perturbed = to_numpy(a_perturbed.imag)

    grad_re = (re_perturbed - re_base) / delta
    grad_im = (im_perturbed - im_base) / delta
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)
    return result_base, scene, grad_mag


# =============================================================================
# Transmitter gradient functions
# =============================================================================
def compute_field_tx(tx_pos, center, size, freq, range_x, range_y, grid_size,
                     n_rays, max_reflections, reflection_coef):
    scene = build_test_scene(box_drjit_geometry(center=wt.Point3f(*center), size=size))
    monitor = FieldMonitor(
        DEFAULT_MONITOR_NAME,
        axis="z",
        position=monitor_height(tx_pos),
        bounds=(range_x, range_y),
        grid_size=grid_size,
    )
    scene.add_monitor(monitor)
    tracer = Tracer(frequency=freq, scene=scene, reflection_n_rays=n_rays,
                    reflection_max_bounces=max_reflections, reflection_coef=reflection_coef)
    result = tracer.trace(tx_pos=tx_pos)
    assert_plane_monitor_result(result, monitor)
    return result, scene


def compute_ad_grad_tx(tx_vals):
    def compute_component_grad(target_part):
        tx_pos = wt.Point3f(float(tx_vals[0]), float(tx_vals[1]), float(tx_vals[2]))
        dr.enable_grad(tx_pos)
        dr.set_grad(tx_pos, wt.Vector3f(1.0, 0.0, 0.0))
        result, scene = compute_field_tx(tx_pos, CENTER, SIZE, FREQ, RANGE_X, RANGE_Y,
                                          GRID_SIZE, N_RAYS, MAX_REFLECTIONS, REFLECTION_COEF)
        a_tot = result.primary.field.total
        field = a_tot.real if target_part == 'real' else a_tot.imag
        dr.forward_to(field, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        grad = dr.grad(field)
        return result, scene, to_numpy(grad) if grad is not None else np.zeros(GRID_SIZE * GRID_SIZE)

    result, scene, grad_re = compute_component_grad('real')
    _, _, grad_im = compute_component_grad('imag')
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)
    return result, scene, grad_mag


def compute_fd_grad_tx(tx_vals, delta=0.01):
    tx_base = wt.Point3f(float(tx_vals[0]), float(tx_vals[1]), float(tx_vals[2]))
    result_base, scene = compute_field_tx(tx_base, CENTER, SIZE, FREQ, RANGE_X, RANGE_Y,
                                           GRID_SIZE, N_RAYS, MAX_REFLECTIONS, REFLECTION_COEF)
    a_base = result_base.primary.field.total
    re_base = to_numpy(a_base.real)
    im_base = to_numpy(a_base.imag)

    tx_perturbed = wt.Point3f(float(tx_vals[0]) + delta, float(tx_vals[1]), float(tx_vals[2]))
    result_perturbed, _ = compute_field_tx(tx_perturbed, CENTER, SIZE, FREQ, RANGE_X, RANGE_Y,
                                            GRID_SIZE, N_RAYS, MAX_REFLECTIONS, REFLECTION_COEF)
    a_perturbed = result_perturbed.primary.field.total
    re_perturbed = to_numpy(a_perturbed.real)
    im_perturbed = to_numpy(a_perturbed.imag)

    grad_re = (re_perturbed - re_base) / delta
    grad_im = (im_perturbed - im_base) / delta
    grad_mag = np.sqrt(grad_re**2 + grad_im**2)
    return result_base, scene, grad_mag


# =============================================================================
# Metrics computation
# =============================================================================
def compute_rmse(img_true, img_pred):
    """Compute RMSE between two images."""
    return float(np.sqrt(np.mean((img_true - img_pred) ** 2)))


def compute_metrics_pair(ad_grad, fd_grad):
    """Compute SSIM, PSNR and RMSE between AD and FD gradients."""
    # Convert to dB scale for comparison
    ad_db = 20 * np.log10(ad_grad + 1e-20)
    fd_db = 20 * np.log10(fd_grad + 1e-20)

    # Normalize to [0, 1] for SSIM/PSNR computation
    combined_min = min(ad_db.min(), fd_db.min())
    combined_max = max(ad_db.max(), fd_db.max())

    ad_norm = (ad_db - combined_min) / (combined_max - combined_min + 1e-10)
    fd_norm = (fd_db - combined_min) / (combined_max - combined_min + 1e-10)

    # Compute SSIM
    ssim_val = compute_ssim(ad_norm, fd_norm, data_range=1.0)

    # Compute PSNR (use normalized values)
    psnr_val = compute_psnr(fd_norm, ad_norm, data_range=1.0)

    # Compute RMSE (use normalized values)
    rmse_val = compute_rmse(fd_norm, ad_norm)

    # Compute RMSE in dB
    rmse_db = compute_rmse(fd_db, ad_db)

    return ssim_val, psnr_val, rmse_val, rmse_db


def main():
    print("=" * 70)
    print("Comparing AD Gradient Magnitude: Position, Rotation, Transmitter")
    print("=" * 70)

    # Compute all gradients
    print("\n[1/6] Computing AD gradient for POSITION...")
    _, _, grad_ad_position = compute_ad_grad_position(CENTER)

    print("[2/6] Computing FD gradient for POSITION...")
    result_pos, scene_pos, grad_fd_position = compute_fd_grad_position(CENTER)

    print("[3/6] Computing AD gradient for ROTATION...")
    _, _, grad_ad_rotation = compute_ad_grad_rotation(ROTATION_VAL)

    print("[4/6] Computing FD gradient for ROTATION...")
    result_rot, scene_rot, grad_fd_rotation = compute_fd_grad_rotation(ROTATION_VAL)

    print("[5/6] Computing AD gradient for TRANSMITTER...")
    _, _, grad_ad_tx = compute_ad_grad_tx(TX_POS)

    print("[6/6] Computing FD gradient for TRANSMITTER...")
    result_tx, scene_tx, grad_fd_tx = compute_fd_grad_tx(TX_POS)

    # Reshape to 2D
    grad_ad_position_2d = grad_ad_position.reshape(GRID_SIZE, GRID_SIZE)
    grad_fd_position_2d = grad_fd_position.reshape(GRID_SIZE, GRID_SIZE)
    grad_ad_rotation_2d = grad_ad_rotation.reshape(GRID_SIZE, GRID_SIZE)
    grad_fd_rotation_2d = grad_fd_rotation.reshape(GRID_SIZE, GRID_SIZE)
    grad_ad_tx_2d = grad_ad_tx.reshape(GRID_SIZE, GRID_SIZE)
    grad_fd_tx_2d = grad_fd_tx.reshape(GRID_SIZE, GRID_SIZE)

    # Compute metrics
    print("\n" + "=" * 70)
    print("METRICS: SSIM, PSNR and RMSE (AD vs FD)")
    print("=" * 70)

    ssim_pos, psnr_pos, rmse_pos, rmse_db_pos = compute_metrics_pair(grad_ad_position_2d, grad_fd_position_2d)
    ssim_rot, psnr_rot, rmse_rot, rmse_db_rot = compute_metrics_pair(grad_ad_rotation_2d, grad_fd_rotation_2d)
    ssim_tx, psnr_tx, rmse_tx, rmse_db_tx = compute_metrics_pair(grad_ad_tx_2d, grad_fd_tx_2d)

    print(f"\nPOSITION gradient:")
    print(f"  SSIM: {ssim_pos:.4f}")
    print(f"  PSNR: {psnr_pos:.2f} dB")
    print(f"  RMSE: {rmse_pos:.4f}")
    print(f"  RMSE (dB): {rmse_db_pos:.2f} dB")

    print(f"\nROTATION gradient:")
    print(f"  SSIM: {ssim_rot:.4f}")
    print(f"  PSNR: {psnr_rot:.2f} dB")
    print(f"  RMSE: {rmse_rot:.4f}")
    print(f"  RMSE (dB): {rmse_db_rot:.2f} dB")

    print(f"\nTRANSMITTER gradient:")
    print(f"  SSIM: {ssim_tx:.4f}")
    print(f"  PSNR: {psnr_tx:.2f} dB")
    print(f"  RMSE: {rmse_tx:.4f}")
    print(f"  RMSE (dB): {rmse_db_tx:.2f} dB")

    # Convert to dB for visualization
    grad_ad_position_db = 20 * np.log10(grad_ad_position_2d + 1e-20)
    grad_ad_rotation_db = 20 * np.log10(grad_ad_rotation_2d + 1e-20)
    grad_ad_tx_db = 20 * np.log10(grad_ad_tx_2d + 1e-20)

    # Get |E| field (use position result)
    field_db = to_numpy(to_power_db(result_pos.primary.field.total)).reshape(GRID_SIZE, GRID_SIZE)

    # Get edges for scene overlay (different for each scene)
    edges_pos = scene_pos.get_edge_data(result_pos.primary.plane_position)['edges_2d']
    edges_rot = scene_rot.get_edge_data(result_rot.primary.plane_position)['edges_2d']
    edges_tx = scene_tx.get_edge_data(result_tx.primary.plane_position)['edges_2d']

    # Plot 1x4 figure: |E| field + AD gradient magnitude for all three
    print("\n[7/7] Generating figure...")
    fig = plt.figure(figsize=(17, 4))
    gs = fig.add_gridspec(1, 6, width_ratios=[0.05, 1, 1, 1, 1, 0.05], wspace=0.08)

    cax_field = fig.add_subplot(gs[0, 0])
    ax_field = fig.add_subplot(gs[0, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(2, 5)]
    cax_grad = fig.add_subplot(gs[0, 5])

    extent = [RANGE_X[0], RANGE_X[1], RANGE_Y[0], RANGE_Y[1]]
    field_vmin, field_vmax = -60, -20
    grad_vmin, grad_vmax = -80, 0

    # |E| field
    im_field = ax_field.imshow(field_db, extent=extent, origin='lower',
                                cmap='inferno', vmin=field_vmin, vmax=field_vmax)
    draw_scene(ax_field, edges_pos, TX_POS, RANGE_X, RANGE_Y)
    ax_field.set_xticks([])
    ax_field.set_yticks([])
    plt.colorbar(im_field, cax=cax_field, label='dB')

    # Position gradient
    im1 = axes[0].imshow(grad_ad_position_db, extent=extent, origin='lower',
                          cmap='RdBu_r', vmin=grad_vmin, vmax=grad_vmax)
    draw_scene(axes[0], edges_pos, TX_POS, RANGE_X, RANGE_Y)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Rotation gradient (use rotated scene edges)
    im2 = axes[1].imshow(grad_ad_rotation_db, extent=extent, origin='lower',
                          cmap='RdBu_r', vmin=grad_vmin, vmax=grad_vmax)
    draw_scene(axes[1], edges_rot, TX_POS, RANGE_X, RANGE_Y)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Transmitter gradient
    im3 = axes[2].imshow(grad_ad_tx_db, extent=extent, origin='lower',
                          cmap='RdBu_r', vmin=grad_vmin, vmax=grad_vmax)
    draw_scene(axes[2], edges_tx, TX_POS, RANGE_X, RANGE_Y)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Colorbar for gradients
    plt.colorbar(im3, cax=cax_grad, label='dB')

    png_path = FIGURES_DIR / "compare_gradients.png"
    svg_path = FIGURES_DIR / "compare_gradients.svg"
    plt.savefig(png_path, dpi=150)
    plt.savefig(svg_path)
    print(f"\n[OK] Figure saved to {png_path}")
    print(f"[OK] Figure saved to {svg_path}")

    maybe_show()


if __name__ == "__main__":
    main()


