"""
Multi-path gradient test: TX -> Cube1 -> Cube2 -> RX reflection path

This script tests:
1. Two-bounce reflection between two cubes
2. Visualization of electric field
3. Gradient computation and visualization
"""

from pathlib import Path
import sys

try:
    from ._paths import FIGURES_DIR, maybe_show
    from ._monitor import (
        assert_boundary_point_sampling,
        assert_plane_monitor_result,
    )
    from ._multipath_benchmark import create_grad_multipath_case
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from _paths import FIGURES_DIR, maybe_show
    from _monitor import (
        assert_boundary_point_sampling,
        assert_plane_monitor_result,
    )
    from _multipath_benchmark import create_grad_multipath_case

import witwin as wt
from witwin.channel import (
    DEFAULT_VARIANT,
    FieldMonitor,
    Tracer,
    compute_diffraction_field,
    compute_reflection_field,
    scalar,
)
from witwin.channel.trace.diffraction.geometry import _evaluate_reflection_prefix_chain, _point_source_field
from witwin.channel.trace.materials import coerce_reflection_trace_detail
from witwin.channel.trace.diffraction import _accumulate_state_subset_field, _prepare_diffraction_state_arrays
from witwin.channel.kernels.trace.packed_state import subset_state_arrays
from witwin.channel.config import ReflectionSuffixConfig

_NO_SUFFIX = ReflectionSuffixConfig()

import drjit as dr
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Multi-path Gradient Test: TX -> Cube1 -> Cube2 -> RX")
print("=" * 60)

_GRAD_TRACE_CONFIG = {
    "trace": {
        "diffraction_execution": {
            "suffix_dda": "symbolic",
        }
    }
}

# -----------------------------------------------------------------------------
# Scene Setup: Two staggered cubes (offset in Y)
# -----------------------------------------------------------------------------
#
#                              Cube2 (2.5, 1.5)
#                              +-------+
#                              |       |
#           TX (0, -5)         |       |
#             *                +-------+
#              \
#               \    reflection path
#                \
#      +-------+  \
#      |       |   \
#      |       |    ---> RX Grid
#      +-------+
#   Cube1 (-2.5, -1.5)
#
# Multi-bounce path: TX -> Cube1 -> Cube2 -> RX
#

case = create_grad_multipath_case()
scene = case.scene
print(f"Scene created: {len(scene.vertical_edges)} diffraction edges")

# Reference scene with Cube1 only for row-2 comparison
scene_c1_only = case.scene_c1_only
print(f"Cube1-only scene created: {len(scene_c1_only.vertical_edges)} diffraction edges")

# -----------------------------------------------------------------------------
# Tracer Setup
# -----------------------------------------------------------------------------
frequency = case.frequency  # 1 GHz
tracer = case.tracer
tracer_c1_only = case.tracer_c1_only

# TX position - between the two cubes, offset in Y
tx_x = case.tx_x
tx_y = case.tx_y
tx_z = case.tx_z
tx_pos = case.tx_pos
tx_pos_eval = case.tx_pos_eval

# Grid parameters
grid_size = case.grid_size
range_xy = case.range_xy
base_monitor = case.monitor

print(f"\nTX position: ({tx_x[0]}, {tx_y[0]}, {tx_z[0]})")
print(f"Grid: {grid_size}x{grid_size}, range: {range_xy}")

# Diagnostic toggle for manual comparison of merged vs unmerged reflection-path
# accumulation in the selected coherent-sum panels.
MERGE_DIAGNOSTIC_REFLECTION_PATHS = True

# -----------------------------------------------------------------------------
# Forward Pass: Compute Fields
# -----------------------------------------------------------------------------
print("\nComputing fields...")

result = tracer.trace(
    tx_pos,
    verbose=True,
    return_timing=True,
)
assert_plane_monitor_result(result, base_monitor)
payload = result.primary

print(f"Timing: {payload.timing}")

print("\nComputing Cube1-only comparison fields...")
result_c1_only = tracer_c1_only.trace(
    tx_pos_eval,
    verbose=False,
    return_timing=True,
)
assert_plane_monitor_result(result_c1_only, base_monitor)
payload_c1_only = result_c1_only.primary
print(f"Cube1-only timing: {payload_c1_only.timing}")

# Extract fields
X = payload.coords.grid_x.numpy().reshape(grid_size, grid_size)
Y = payload.coords.grid_y.numpy().reshape(grid_size, grid_size)

def get_power_db(field):
    power = dr.abs(field.real)**2 + dr.abs(field.imag)**2
    power_np = power.numpy().reshape(grid_size, grid_size)
    return 10 * np.log10(power_np + 1e-20)

los_db = get_power_db(payload.field.los)
ref_db = get_power_db(payload.field.reflection)
tot_db = get_power_db(payload.field.total)
los_c1_only_db = get_power_db(payload_c1_only.field.los)
dif_c1_only_db = get_power_db(payload_c1_only.field.diffraction)
ref_c1_only_db = get_power_db(payload_c1_only.field.reflection)
tot_c1_only_db = get_power_db(payload_c1_only.field.total)

# -----------------------------------------------------------------------------
# Backward Pass: Compute Gradients
# -----------------------------------------------------------------------------
print("\nComputing gradients...")

# Loss: sum of squared field magnitude (total power)
a_tot = payload.field.total
loss = dr.sum(a_tot.real * a_tot.real + a_tot.imag * a_tot.imag)

# Backward pass
dr.backward(loss)

# Get gradients
grad_x = float(dr.grad(tx_x)[0]) if dr.width(dr.grad(tx_x)) > 0 else 0.0
grad_y = float(dr.grad(tx_y)[0]) if dr.width(dr.grad(tx_y)) > 0 else 0.0
grad_z = float(dr.grad(tx_z)[0]) if dr.width(dr.grad(tx_z)) > 0 else 0.0

print(f"\nGradients of total power w.r.t. TX position:")
print(f"  d(loss)/d(tx_x) = {grad_x:.6e}")
print(f"  d(loss)/d(tx_y) = {grad_y:.6e}")
print(f"  d(loss)/d(tx_z) = {grad_z:.6e}")

# Gradient magnitude and direction
grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
print(f"  |grad| = {grad_mag:.6e}")

# -----------------------------------------------------------------------------
# Compute Spatial Gradient Field (for visualization)
# -----------------------------------------------------------------------------
print("\nComputing spatial gradient field...")

# We'll compute the gradient of total power at each grid point
# This shows how the field changes spatially

tot_power = (a_tot.real * a_tot.real + a_tot.imag * a_tot.imag).numpy().reshape(grid_size, grid_size)

# Numerical gradient (spatial)
grad_power_x, grad_power_y = np.gradient(tot_power)
grad_power_mag = np.sqrt(grad_power_x**2 + grad_power_y**2)

# -----------------------------------------------------------------------------
# Per-edge Diffraction Computation (direct TX -> edge -> RX)
# -----------------------------------------------------------------------------
print("\nComputing per-edge diffraction fields...")

dif_real_all, dif_imag_all, per_edge_list = compute_diffraction_field(
    payload.coords.grid_x, payload.coords.grid_y, 1.5, tx_pos_eval,
    scene, tracer.wavelength, tracer.k,
    return_per_edge=True
)

# Get edge positions for labelling
edge_cache = scene.get_edge_data(1.5)
dif_points = edge_cache['diffraction_points']
n_edges = len(per_edge_list)
print(f"  {n_edges} diffraction edges")

# Per-edge power in dB
per_edge_db = []
edge_labels = []
for i, (er, ei) in enumerate(per_edge_list):
    p = (er * er + ei * ei).numpy().reshape(grid_size, grid_size)
    per_edge_db.append(10 * np.log10(p + 1e-20))
    dp = dif_points[i]
    px, py = scalar(dp.position.x), scalar(dp.position.y)
    cube_id = 'C1' if px < 0 else 'C2'
    edge_labels.append(f'{cube_id} ({px:+.1f}, {py:+.1f})')

# Split into Cube1 and Cube2 edges
c1_idx = [i for i in range(n_edges) if scalar(dif_points[i].position.x) < 0]
c2_idx = [i for i in range(n_edges) if scalar(dif_points[i].position.x) >= 0]

# -----------------------------------------------------------------------------
# Per-bounce Reflection Fields
# -----------------------------------------------------------------------------
print("\nComputing per-bounce reflection fields...")

field = base_monitor.to_field(1.0)
coords = field.get_coordinates()
assert_boundary_point_sampling(
    coords["x_coords"],
    coords["y_coords"],
    bounds=base_monitor.bounds,
    grid_size=base_monitor.grid_shape,
)

a_ref_total_pb, a_ref_list, reflection_detail = compute_reflection_field(
    grid=field,
    rx_z=1.5,
    tx_pos=tx_pos_eval,
    scene=scene,
    wavelength=tracer.wavelength,
    k=tracer.k,
    n_rays=tracer.reflection_n_rays,
    max_reflections=tracer.reflection_max_bounces,
    mode='2d',
    reflection_coef=tracer.reflection_coef,
    return_per_bounce=True,
    grid_data=coords,
)

n_bounces = len(a_ref_list)
print(f"  {n_bounces} bounce levels")

# Per-bounce reflection dB
ref_bounce_db = []
for i, a_ref_b in enumerate(a_ref_list):
    p = (a_ref_b.real * a_ref_b.real + a_ref_b.imag * a_ref_b.imag).numpy().reshape(grid_size, grid_size)
    ref_bounce_db.append(10 * np.log10(p + 1e-20))
    print(f"  Bounce {i+1}: ref peak = {ref_bounce_db[-1].max():.1f} dB")

# Per-bounce total field (ref + dif) in dB
# Bounce 0: LoS + direct/mixed diffraction + first-bounce reflection
# Bounce b>0: cumulative LoS + diffraction + reflections up to bounce b
bounce_total_db = []
cumulative_real = payload.field.los.real + payload.field.diffraction.real
cumulative_imag = payload.field.los.imag + payload.field.diffraction.imag
for b in range(n_bounces):
    ref_b = a_ref_list[b]
    cumulative_real = cumulative_real + ref_b.real
    cumulative_imag = cumulative_imag + ref_b.imag
    tot_r = cumulative_real
    tot_i = cumulative_imag
    p = (tot_r * tot_r + tot_i * tot_i).numpy().reshape(grid_size, grid_size)
    bounce_total_db.append(10 * np.log10(p + 1e-20))

# ---------------------------------------------------------------------------
# Selected component combination:
#   - Cube1 first-order reflection
#   - Cube2 second-order reflection
#   - Cube2 S -> R -> D diffraction only
# ---------------------------------------------------------------------------
print("\nComputing selected reflection/diffraction combination...")

rx_pos = wt.Point3f(coords['X'], coords['Y'], wt.Float(1.5))
zero_field = wt.Complex2f(dr.zeros(wt.Float, field.n_cells), dr.zeros(wt.Float, field.n_cells))

def add_fields(*fields):
    total_real = dr.zeros(wt.Float, field.n_cells)
    total_imag = dr.zeros(wt.Float, field.n_cells)
    for item in fields:
        total_real = total_real + item.real
        total_imag = total_imag + item.imag
    return wt.Complex2f(total_real, total_imag)

def subtract_fields(field_a, field_b):
    return wt.Complex2f(field_a.real - field_b.real, field_a.imag - field_b.imag)

def merge_reflection_path_indices(paths, selected_indices, position_tol=1e-5):
    if paths is None or len(selected_indices) == 0:
        return []

    chain_depth = int(paths.get('chain_depth', 0))
    merged = {}
    ordered_keys = []

    for idx in selected_indices:
        idx_int = int(idx)
        gather_idx = wt.UInt32(idx_int)
        image_source = dr.gather(wt.Point3f, paths['image_source'], gather_idx)
        chain = tuple(int(paths[f'path_prim_idx_{slot}'][idx_int]) for slot in range(chain_depth))
        quantized_pos = (
            int(round(float(image_source.x[0]) / position_tol)),
            int(round(float(image_source.y[0]) / position_tol)),
            int(round(float(image_source.z[0]) / position_tol)),
        )
        key = (chain, quantized_pos)
        discovery_count = int(paths['discovery_count'][idx_int])

        if key not in merged:
            merged[key] = {
                'representative_idx': idx_int,
                'best_count': discovery_count,
            }
            ordered_keys.append(key)
            continue

        if discovery_count > merged[key]['best_count']:
            merged[key]['representative_idx'] = idx_int
            merged[key]['best_count'] = discovery_count

    return [merged[key]['representative_idx'] for key in ordered_keys]

def accumulate_reflection_paths(paths, selected_indices, merge_duplicates=MERGE_DIAGNOSTIC_REFLECTION_PATHS):
    if paths is None or len(selected_indices) == 0:
        return zero_field

    active_indices = merge_reflection_path_indices(paths, selected_indices) if merge_duplicates else list(selected_indices)
    total = zero_field
    chain_depth = int(paths.get('chain_depth', 0))
    reflection_context = coerce_reflection_trace_detail(reflection_detail)
    for idx in active_indices:
        gather_idx = wt.UInt32(int(idx))
        image_source = dr.gather(wt.Point3f, paths['image_source'], gather_idx)
        chain = [
            dr.gather(wt.Int32, paths[f'path_prim_idx_{slot}'], gather_idx)
            for slot in range(chain_depth)
        ]
        valid, source_weight, _ = _evaluate_reflection_prefix_chain(
            image_source=image_source,
            target_pos=rx_pos,
            chain_prim_indices=chain,
            scene=scene,
            target_adjacent_faces=(),
            material_override=reflection_context.reflection_material,
            reflection_gain=reflection_context.reflection_gain,
            use_scene_materials=reflection_context.use_scene_materials,
            wavelength=tracer.wavelength,
            k=tracer.k,
        )
        field_path = _point_source_field(
            image_source,
            source_weight,
            rx_pos,
            tracer.wavelength,
            tracer.k,
        )
        field_path = wt.Complex2f(
            dr.select(valid, field_path.real, wt.Float(0.0)),
            dr.select(valid, field_path.imag, wt.Float(0.0)),
        )
        total = add_fields(total, field_path)
    return total

_, edge_data_selected, state_arrays_selected, _ = _prepare_diffraction_state_arrays(
    tx_pos=tx_pos_eval,
    rx_z=1.5,
    scene=scene,
    wavelength=tracer.wavelength,
    k=tracer.k,
    reflection_detail=reflection_detail,
    material_detail=None,
    reflection_n_rays=tracer.reflection_n_rays,
    reflection_max_bounces=tracer.reflection_max_bounces,
    reflection_coef=tracer.reflection_coef,
    reflection_mode='2d',
    max_diffractions=tracer.max_diffractions,
)

bounce1_paths = reflection_detail['source_paths_per_bounce'][0] if len(reflection_detail['source_paths_per_bounce']) >= 1 else None
bounce2_paths = reflection_detail['source_paths_per_bounce'][1] if len(reflection_detail['source_paths_per_bounce']) >= 2 else None

tri_v0x = scene.tri_data_gpu['v0'].x.numpy()
tri_v1x = scene.tri_data_gpu['v1'].x.numpy()
tri_v2x = scene.tri_data_gpu['v2'].x.numpy()

def prim_center_x(prim_idx):
    prim_idx = int(prim_idx)
    return float((tri_v0x[prim_idx] + tri_v1x[prim_idx] + tri_v2x[prim_idx]) / 3.0)

bounce1_idx_c1 = []
if bounce1_paths is not None and int(bounce1_paths.get('n_paths', 0)) > 0:
    bounce1_idx_c1 = [
        path_idx
        for path_idx in range(int(bounce1_paths['n_paths']))
        if prim_center_x(bounce1_paths['path_prim_idx_0'][path_idx]) < 0.0
    ]

bounce2_idx_c2 = []
if bounce2_paths is not None and int(bounce2_paths.get('n_paths', 0)) > 0:
    bounce2_idx_c2 = [
        path_idx
        for path_idx in range(int(bounce2_paths['n_paths']))
        if prim_center_x(bounce2_paths['path_prim_idx_1'][path_idx]) > 0.0
    ]

bounce1_idx_c1_merged = merge_reflection_path_indices(bounce1_paths, bounce1_idx_c1)
bounce2_idx_c2_merged = merge_reflection_path_indices(bounce2_paths, bounce2_idx_c2)
a_ref_bounce1_c1_unmerged = accumulate_reflection_paths(bounce1_paths, bounce1_idx_c1, merge_duplicates=False)
a_ref_bounce2_c2_unmerged = accumulate_reflection_paths(bounce2_paths, bounce2_idx_c2, merge_duplicates=False)
a_ref_bounce1_c1_merged = accumulate_reflection_paths(bounce1_paths, bounce1_idx_c1, merge_duplicates=True)
a_ref_bounce2_c2_merged = accumulate_reflection_paths(bounce2_paths, bounce2_idx_c2, merge_duplicates=True)
a_ref_bounce1_c1 = a_ref_bounce1_c1_merged if MERGE_DIAGNOSTIC_REFLECTION_PATHS else a_ref_bounce1_c1_unmerged
a_ref_bounce2_c2 = a_ref_bounce2_c2_merged if MERGE_DIAGNOSTIC_REFLECTION_PATHS else a_ref_bounce2_c2_unmerged

srd_c2_mask = (
    (state_arrays_selected['order'] == wt.UInt32(1))
    & (state_arrays_selected['prefix_reflection_depth'] > wt.UInt32(0))
    & (state_arrays_selected['intermediate_reflection_depth'] == wt.UInt32(0))
    & (state_arrays_selected['suffix_reflection_depth'] == wt.UInt32(0))
    & (state_arrays_selected['edge_pos'].x >= wt.Float(0.0))
)
srd_c2_states = subset_state_arrays(state_arrays_selected, srd_c2_mask)
a_dif_srd_c2 = _accumulate_state_subset_field(
    state_arrays=srd_c2_states,
    rx_pos=rx_pos,
    scene=scene,
    wavelength=tracer.wavelength,
    k=tracer.k,
    n_edges=0 if edge_data_selected is None else edge_data_selected['n_edges'],
    material_detail=None,
    suffix=_NO_SUFFIX,
) if int(srd_c2_states['n_states']) > 0 else zero_field

sd_mask = (
    (state_arrays_selected['order'] == wt.UInt32(1))
    & (state_arrays_selected['prefix_reflection_depth'] == wt.UInt32(0))
    & (state_arrays_selected['intermediate_reflection_depth'] == wt.UInt32(0))
    & (state_arrays_selected['suffix_reflection_depth'] == wt.UInt32(0))
)
sd_states = subset_state_arrays(state_arrays_selected, sd_mask)
a_dif_sd = _accumulate_state_subset_field(
    state_arrays=sd_states,
    rx_pos=rx_pos,
    scene=scene,
    wavelength=tracer.wavelength,
    k=tracer.k,
    n_edges=0 if edge_data_selected is None else edge_data_selected['n_edges'],
    material_detail=None,
    suffix=_NO_SUFFIX,
) if int(sd_states['n_states']) > 0 else zero_field

srd_mask = (
    (state_arrays_selected['order'] == wt.UInt32(1))
    & (state_arrays_selected['prefix_reflection_depth'] > wt.UInt32(0))
    & (state_arrays_selected['intermediate_reflection_depth'] == wt.UInt32(0))
    & (state_arrays_selected['suffix_reflection_depth'] == wt.UInt32(0))
)
srd_states = subset_state_arrays(state_arrays_selected, srd_mask)
a_dif_srd = _accumulate_state_subset_field(
    state_arrays=srd_states,
    rx_pos=rx_pos,
    scene=scene,
    wavelength=tracer.wavelength,
    k=tracer.k,
    n_edges=0 if edge_data_selected is None else edge_data_selected['n_edges'],
    material_detail=None,
    suffix=_NO_SUFFIX,
) if int(srd_states['n_states']) > 0 else zero_field

filtered_total_field = add_fields(payload.field.los, payload.field.reflection, a_dif_sd, a_dif_srd)
selected_combo_unmerged = add_fields(
    payload.field.los,
    a_dif_sd,
    a_ref_bounce1_c1_unmerged,
    a_ref_bounce2_c2_unmerged,
    a_dif_srd_c2,
)
selected_combo_merged = add_fields(
    payload.field.los,
    a_dif_sd,
    a_ref_bounce1_c1_merged,
    a_ref_bounce2_c2_merged,
    a_dif_srd_c2,
)
a_selected_combo = add_fields(
    payload.field.los,
    a_dif_sd,
    a_ref_bounce1_c1,
    a_ref_bounce2_c2,
    a_dif_srd_c2,
)
a_selected_combo_no_los_sd_unmerged = add_fields(
    a_ref_bounce1_c1_unmerged,
    a_ref_bounce2_c2_unmerged,
    a_dif_srd_c2,
)
a_selected_combo_no_los_sd_merged = add_fields(
    a_ref_bounce1_c1_merged,
    a_ref_bounce2_c2_merged,
    a_dif_srd_c2,
)
a_selected_combo_no_los_sd = add_fields(
    a_ref_bounce1_c1,
    a_ref_bounce2_c2,
    a_dif_srd_c2,
)

sd_db = get_power_db(a_dif_sd)
srd_db = get_power_db(a_dif_srd)
filtered_tot_db = get_power_db(filtered_total_field)
ref1_c1_db = get_power_db(a_ref_bounce1_c1)
ref2_c2_db = get_power_db(a_ref_bounce2_c2)
srd_c2_db = get_power_db(a_dif_srd_c2)
selected_combo_db = get_power_db(a_selected_combo)
selected_combo_no_los_sd_db = get_power_db(a_selected_combo_no_los_sd)
selected_combo_unmerged_db = get_power_db(selected_combo_unmerged)
selected_combo_merged_db = get_power_db(selected_combo_merged)
selected_combo_no_los_sd_unmerged_db = get_power_db(a_selected_combo_no_los_sd_unmerged)
selected_combo_no_los_sd_merged_db = get_power_db(a_selected_combo_no_los_sd_merged)
cube2_delta_field = subtract_fields(a_selected_combo, payload_c1_only.field.total)
cube2_delta_db = get_power_db(cube2_delta_field)

print(f"  S->D states = {int(sd_states['n_states'])}")
print(f"  S->R->D states = {int(srd_states['n_states'])}")
print(f"  S->D peak = {sd_db.max():.1f} dB")
print(f"  S->R->D peak = {srd_db.max():.1f} dB")
print(f"  Cube1 bounce-1 reflection paths = {len(bounce1_idx_c1)}")
print(f"  Cube1 bounce-1 merged physical paths = {len(bounce1_idx_c1_merged)}")
print(f"  Cube2 bounce-2 reflection paths = {len(bounce2_idx_c2)}")
print(f"  Cube2 bounce-2 merged physical paths = {len(bounce2_idx_c2_merged)}")
print(f"  merge_reflection_path_indices enabled = {MERGE_DIAGNOSTIC_REFLECTION_PATHS}")
print(f"  Cube2 S->R->D states = {int(srd_c2_states['n_states'])}")
print(f"  Cube1 bounce-1 reflection peak = {ref1_c1_db.max():.1f} dB")
print(f"  Cube2 bounce-2 reflection peak = {ref2_c2_db.max():.1f} dB")
print(f"  Cube2 S->R->D diffraction peak = {srd_c2_db.max():.1f} dB")
print(f"  Selected coherent sum (LoS + S->D + selected high-order terms) peak = {selected_combo_db.max():.1f} dB")
print(f"  Selected high-order-only sum peak = {selected_combo_no_los_sd_db.max():.1f} dB")
print(f"  Selected coherent sum peak [merge off] = {selected_combo_unmerged_db.max():.1f} dB")
print(f"  Selected coherent sum peak [merge on] = {selected_combo_merged_db.max():.1f} dB")
print(f"  Selected high-order-only peak [merge off] = {selected_combo_no_los_sd_unmerged_db.max():.1f} dB")
print(f"  Selected high-order-only peak [merge on] = {selected_combo_no_los_sd_merged_db.max():.1f} dB")
print(f"  Cube1-only total-field peak = {tot_c1_only_db.max():.1f} dB")
print(f"  Cube2 net delta peak = {cube2_delta_db.max():.1f} dB")

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
print("\nGenerating plots...")

def axis_aligned_box_outline_xy(center_xy, size):
    half_size = 0.5 * size
    cx, cy = center_xy
    return np.array(
        [
            [cx - half_size, cy - half_size],
            [cx + half_size, cy - half_size],
            [cx + half_size, cy + half_size],
            [cx - half_size, cy + half_size],
            [cx - half_size, cy - half_size],
        ]
    )


cube1_outline = axis_aligned_box_outline_xy(center_xy=(-2.5, -3.0), size=2.0)
cube2_outline = axis_aligned_box_outline_xy(center_xy=(2.0, 0.5), size=2.0)

def draw_overlay(ax):
    ax.plot(cube1_outline[:, 0], cube1_outline[:, 1], 'w-', linewidth=1.5)
    ax.plot(cube2_outline[:, 0], cube2_outline[:, 1], 'w--', linewidth=1.5)
    ax.plot(tx_x[0], tx_y[0], 'r*', markersize=10)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

extent = [range_xy[0], range_xy[1], range_xy[0], range_xy[1]]
field_vmin, field_vmax = -90, -20
dif_vmin, dif_vmax = -80, -30
grad_vmin, grad_vmax = -80, 0

# ===== Figure 1: Main field overview (1 row) =====
fig1, axes1 = plt.subplots(1, 7, figsize=(21, 3.5))

main_fields = [
    (los_db, 'LoS', 'inferno', field_vmin, field_vmax),
    (ref_db, 'Reflection', 'inferno', -80, -30),
    (sd_db, 'Diffraction (S -> D)', 'inferno', dif_vmin, dif_vmax),
    (srd_db, 'Diffraction (S -> R -> D)', 'inferno', -90, -40),
    (tot_db, 'Total Field', 'inferno', field_vmin, field_vmax),
    (filtered_tot_db, 'Total (No S -> D -> D)', 'inferno', field_vmin, field_vmax),
]

for col, (data, title, cmap, vmin, vmax) in enumerate(main_fields):
    ax = axes1[col]
    im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    draw_overlay(ax)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Gradient
grad_db = 20 * np.log10(grad_power_mag + 1e-20)
ax_grad = axes1[6]
im_grad = ax_grad.imshow(grad_db, extent=extent, origin='lower',
                          cmap='RdBu_r', vmin=grad_vmin, vmax=grad_vmax)
draw_overlay(ax_grad)
arrow_scale = 0.5 / (grad_mag + 1e-10)
ax_grad.arrow(float(tx_x[0]), float(tx_y[0]),
              grad_x * arrow_scale, grad_y * arrow_scale,
              head_width=0.3, head_length=0.2, fc='cyan', ec='cyan', linewidth=2)
ax_grad.set_title(f'|grad(P)|  TX:({grad_x:.1e},{grad_y:.1e})', fontsize=9)
plt.colorbar(im_grad, ax=ax_grad, fraction=0.046, pad=0.04)

fig1.suptitle('Multi-path Test: TX -> Cube1 -> Cube2 -> RX\n'
              f'TX=({tx_x[0]:.1f}, {tx_y[0]:.1f}), '
              f'Gradient magnitude: {grad_mag:.4e}', fontsize=12)
fig1.tight_layout()
fig1_path = FIGURES_DIR / "grad_multipath.png"
fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {fig1_path}")

# ===== Figure 2: Per-bounce reflection + direct diffraction per edge =====
# Layout per row: [Cumulative total] [Ref bounce] [direct diffraction per edge...]
n_bounce_cols = 2 + n_edges  # col 0: total, col 1: ref, col 2..N+1: per edge
fig2 = plt.figure(figsize=(2.8 * n_bounce_cols + 0.6, 3.0 * n_bounces + 0.8))
gs2 = fig2.add_gridspec(n_bounces, n_bounce_cols + 1,
                        width_ratios=[1] * n_bounce_cols + [0.05],
                        hspace=0.30, wspace=0.10)

ref_vmin, ref_vmax = -80, -30
dif_edge_vmin, dif_edge_vmax = -80, -30
rd_edge_vmin, rd_edge_vmax = -90, -40
tot_vmin, tot_vmax = -60, -20
last_im2 = None

for b in range(n_bounces):
    # Column 0: Total (ref + dif) for this bounce
    ax = fig2.add_subplot(gs2[b, 0])
    ax.imshow(bounce_total_db[b], extent=extent, origin='lower',
              cmap='inferno', vmin=ref_vmin, vmax=ref_vmax)
    draw_overlay(ax)
    ax.set_title(f'Bounce {b+1} Total', fontsize=9)
    ax.set_ylabel(f'Bounce {b+1}', fontsize=10, fontweight='bold')

    # Column 1: This bounce's reflection field
    ax = fig2.add_subplot(gs2[b, 1])
    last_im2 = ax.imshow(ref_bounce_db[b], extent=extent, origin='lower',
                         cmap='inferno', vmin=ref_vmin, vmax=ref_vmax)
    draw_overlay(ax)
    ax.set_title(f'Bounce {b+1} Ref', fontsize=9)

    # Columns 2..N+1: per-edge diffraction
    for ei_dif in range(n_edges):
        ax = fig2.add_subplot(gs2[b, 2 + ei_dif])
        last_im2 = ax.imshow(per_edge_db[ei_dif], extent=extent,
                             origin='lower', cmap='inferno',
                             vmin=dif_edge_vmin, vmax=dif_edge_vmax)
        if b == 0:
            ax.set_title(f'Dif {edge_labels[ei_dif]}', fontsize=8)

        draw_overlay(ax)
        dp = dif_points[ei_dif]
        ax.plot(scalar(dp.position.x), scalar(dp.position.y),
                'co', markersize=7, markeredgecolor='w', markeredgewidth=1.2)

# Colorbar
if last_im2 is not None:
    cax = fig2.add_subplot(gs2[:, n_bounce_cols])
    plt.colorbar(last_im2, cax=cax, label='dB')

fig2.suptitle('Per-bounce: [Total] [Ref] [Diffraction per Edge]\n'
              'Direct per-edge diffraction is repeated for context across bounce rows',
              fontsize=11)
fig2_path = FIGURES_DIR / "grad_multipath_per_bounce.png"
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig2_path}")

# ===== Figure 3: Cross-section Analysis =====
fig3, axes4 = plt.subplots(1, 3, figsize=(15, 4))

mid_y_idx = grid_size // 2
x_line = X[mid_y_idx, :]

axes4[0].plot(x_line, los_db[mid_y_idx, :], 'b-', label='LoS', linewidth=2)
axes4[0].plot(x_line, ref_db[mid_y_idx, :], 'g-', label='Reflection', linewidth=2)
axes4[0].plot(x_line, sd_db[mid_y_idx, :], 'c--', label='S -> D', linewidth=1.5)
axes4[0].plot(x_line, srd_db[mid_y_idx, :], color='m', linestyle='-.', label='S -> R -> D', linewidth=1.5)
axes4[0].plot(x_line, tot_db[mid_y_idx, :], 'k-', label='Total', linewidth=1.5, alpha=0.7)
axes4[0].axvline(-2.5, color='gray', linestyle='--', alpha=0.5, label='Cube1')
axes4[0].axvline(2.5, color='gray', linestyle=':', alpha=0.5, label='Cube2')
axes4[0].set_xlabel('X (m)')
axes4[0].set_ylabel('Power (dB)')
axes4[0].set_title('Cross-section at Y=0')
axes4[0].legend(fontsize=8)
axes4[0].grid(True, alpha=0.3)
axes4[0].set_ylim(-90, -20)

mid_x_idx = grid_size // 2
y_line = Y[:, mid_x_idx]

axes4[1].plot(y_line, los_db[:, mid_x_idx], 'b-', label='LoS', linewidth=2)
axes4[1].plot(y_line, ref_db[:, mid_x_idx], 'g-', label='Reflection', linewidth=2)
axes4[1].plot(y_line, tot_db[:, mid_x_idx], 'k-', label='Total', linewidth=1.5, alpha=0.7)
axes4[1].axvline(tx_y[0], color='red', linestyle='--', alpha=0.5, label='TX')
axes4[1].set_xlabel('Y (m)')
axes4[1].set_ylabel('Power (dB)')
axes4[1].set_title('Cross-section at X=0')
axes4[1].legend(fontsize=8)
axes4[1].grid(True, alpha=0.3)
axes4[1].set_ylim(-90, -20)

axes4[2].plot(x_line, grad_db[mid_y_idx, :], 'r-', linewidth=2)
axes4[2].axvline(-2.5, color='gray', linestyle='--', alpha=0.5)
axes4[2].axvline(2.5, color='gray', linestyle=':', alpha=0.5)
axes4[2].set_xlabel('X (m)')
axes4[2].set_ylabel('|grad(Power)| (dB)')
axes4[2].set_title('Gradient Magnitude at Y=0')
axes4[2].grid(True, alpha=0.3)

fig3.tight_layout()
fig3_path = FIGURES_DIR / "grad_multipath_crosssection.png"
fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig3_path}")

# ===== Figure 4: Selected component combination =====
fig4, axes5 = plt.subplots(2, 5, figsize=(19.2, 7.6))

selected_fields_row1 = [
    (ref1_c1_db, 'Cube1 Ref (bounce 1)', 'inferno', -90, -40),
    (ref2_c2_db, 'Cube2 Ref (bounce 2)', 'inferno', -90, -40),
    (srd_c2_db, 'Cube2 Dif (S -> R -> D)', 'inferno', -90, -40),
    (selected_combo_db, 'LoS + S->D + selected sum', 'inferno', -90, -20),
    (selected_combo_no_los_sd_db, 'Selected sum (no LoS / S->D)', 'inferno', -90, -35),
]

selected_fields_row2 = [
    (los_c1_only_db, 'Cube1-only LoS', 'inferno', -90, -20),
    (dif_c1_only_db, 'Cube1-only diffraction', 'inferno', -90, -35),
    (ref_c1_only_db, 'Cube1-only reflection', 'inferno', -90, -40),
    (tot_c1_only_db, 'Cube1-only total', 'inferno', -90, -20),
]

for ax, (data, title, cmap, vmin, vmax) in zip(axes5[0], selected_fields_row1):
    im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    draw_overlay(ax)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax, (data, title, cmap, vmin, vmax) in zip(axes5[1], selected_fields_row2):
    im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    draw_overlay(ax)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

axes5[1, 4].axis('off')
axes5[1, 4].text(
    0.5,
    0.5,
    'Cube1-only row has no\nmatched high-order-only panel',
    ha='center',
    va='center',
    fontsize=10,
)

fig4.suptitle(
    'Row 1: selected full sum plus a high-order-only variant without LoS or first-order diffraction\n'
    'Row 2: Separate Cube1-only scene',
    fontsize=12,
)
fig4.tight_layout()
fig4_path = FIGURES_DIR / "grad_multipath_second_order_combo.png"
fig4.savefig(fig4_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig4_path}")

# ===== Figure 5: Cube2 net delta diagnostics =====
fig5, axes6 = plt.subplots(1, 3, figsize=(12.6, 4.0))

delta_fields = [
    (selected_combo_db, 'Full selected coherent sum', 'inferno', -90, -20),
    (tot_c1_only_db, 'Cube1-only total', 'inferno', -90, -20),
    (cube2_delta_db, 'Cube2 net delta |full - Cube1-only|', 'magma', -90, -35),
]

for ax, (data, title, cmap, vmin, vmax) in zip(axes6, delta_fields):
    im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    draw_overlay(ax)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig5.suptitle(
    'Cube2 net impact diagnostic: coherent field difference between the full selected sum and the Cube1-only total scene',
    fontsize=12,
)
fig5.tight_layout()
fig5_path = FIGURES_DIR / "grad_multipath_cube2_delta.png"
fig5.savefig(fig5_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig5_path}")

# ===== Figure 6: merge_reflection_path_indices toggle comparison =====
fig6, axes7 = plt.subplots(2, 2, figsize=(8.8, 8.2))

merge_compare_fields = [
    [
        (selected_combo_unmerged_db, 'merge OFF: LoS + S->D + selected sum', 'inferno', -90, -20),
        (selected_combo_no_los_sd_unmerged_db, 'merge OFF: selected sum (no LoS / S->D)', 'inferno', -90, -35),
    ],
    [
        (selected_combo_merged_db, 'merge ON: LoS + S->D + selected sum', 'inferno', -90, -20),
        (selected_combo_no_los_sd_merged_db, 'merge ON: selected sum (no LoS / S->D)', 'inferno', -90, -35),
    ],
]

for row_idx, row_fields in enumerate(merge_compare_fields):
    for col_idx, (data, title, cmap, vmin, vmax) in enumerate(row_fields):
        ax = axes7[row_idx, col_idx]
        im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        draw_overlay(ax)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig6.suptitle(
    'merge_reflection_path_indices toggle comparison for the selected coherent-sum panels',
    fontsize=12,
)
fig6.tight_layout()
fig6_path = FIGURES_DIR / "grad_multipath_merge_toggle_compare.png"
fig6.savefig(fig6_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig6_path}")

# -----------------------------------------------------------------------------
# Finite Difference Gradient Verification
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("R-D Gradient Consistency Check (AD vs FD)")
print("=" * 60)

eps = 1e-3
grad_check_grid = 32
grad_check_n_rays = min(640, tracer.reflection_n_rays)
rel_tol_percent = 12.0

gradient_check_tracer = Tracer(
    frequency=frequency,
    scene=scene,
    config=_GRAD_TRACE_CONFIG,
    reflection_n_rays=grad_check_n_rays,
    reflection_max_bounces=tracer.reflection_max_bounces,
    reflection_coef=tracer.reflection_coef,
    enable_rd_diffraction=tracer.enable_rd_diffraction,
)

def compute_total_power_loss(tracer_obj, x, y, z, check_grid_size, check_range_xy, check_height):
    """Compute sum(|a_tot|^2) at a given TX position."""
    with dr.suspend_grad():
        pos = wt.Point3f(wt.Float(x), wt.Float(y), wt.Float(z))
        monitor = FieldMonitor(
            "grad_check_plane",
            axis="z",
            position=check_height,
            bounds=(check_range_xy, check_range_xy),
            grid_size=check_grid_size,
        )
        res = tracer_obj.trace(pos, monitor=monitor, verbose=False)
        assert_plane_monitor_result(res, monitor)
        a = res.primary.field.total
        return float(dr.sum(a.real * a.real + a.imag * a.imag)[0])

def compute_tx_ad_fd(tracer_obj, base_tx, delta, check_grid_size, check_range_xy, check_height):
    """Compute AD and FD gradients w.r.t. (tx_x, tx_y, tx_z)."""
    base_x, base_y, base_z = base_tx

    # AD gradient
    tx_x_ad = wt.Float(base_x)
    tx_y_ad = wt.Float(base_y)
    tx_z_ad = wt.Float(base_z)
    dr.enable_grad(tx_x_ad)
    dr.enable_grad(tx_y_ad)
    dr.enable_grad(tx_z_ad)

    tx_pos_ad = wt.Point3f(tx_x_ad, tx_y_ad, tx_z_ad)
    monitor = FieldMonitor(
        "grad_check_plane",
        axis="z",
        position=check_height,
        bounds=(check_range_xy, check_range_xy),
        grid_size=check_grid_size,
    )
    res_ad = tracer_obj.trace(
        tx_pos_ad,
        monitor=monitor,
        verbose=False,
    )
    assert_plane_monitor_result(res_ad, monitor)
    a_ad = res_ad.primary.field.total
    loss_ad = dr.sum(a_ad.real * a_ad.real + a_ad.imag * a_ad.imag)
    dr.backward(loss_ad)

    ad_grad_x = float(dr.grad(tx_x_ad)[0]) if dr.width(dr.grad(tx_x_ad)) > 0 else 0.0
    ad_grad_y = float(dr.grad(tx_y_ad)[0]) if dr.width(dr.grad(tx_y_ad)) > 0 else 0.0
    ad_grad_z = float(dr.grad(tx_z_ad)[0]) if dr.width(dr.grad(tx_z_ad)) > 0 else 0.0

    # FD gradient
    print("  FD axis tx_x...")
    fd_grad_x = (
        compute_total_power_loss(tracer_obj, base_x + delta, base_y, base_z, check_grid_size, check_range_xy, check_height)
        - compute_total_power_loss(tracer_obj, base_x - delta, base_y, base_z, check_grid_size, check_range_xy, check_height)
    ) / (2 * delta)
    print("  FD axis tx_y...")
    fd_grad_y = (
        compute_total_power_loss(tracer_obj, base_x, base_y + delta, base_z, check_grid_size, check_range_xy, check_height)
        - compute_total_power_loss(tracer_obj, base_x, base_y - delta, base_z, check_grid_size, check_range_xy, check_height)
    ) / (2 * delta)
    print("  FD axis tx_z...")
    fd_grad_z = (
        compute_total_power_loss(tracer_obj, base_x, base_y, base_z + delta, check_grid_size, check_range_xy, check_height)
        - compute_total_power_loss(tracer_obj, base_x, base_y, base_z - delta, check_grid_size, check_range_xy, check_height)
    ) / (2 * delta)

    ad = np.array([ad_grad_x, ad_grad_y, ad_grad_z], dtype=np.float64)
    fd = np.array([fd_grad_x, fd_grad_y, fd_grad_z], dtype=np.float64)
    return ad, fd

def report_tx_gradient_consistency(ad, fd, rel_tol=12.0, near_zero_fd=1e-4, near_zero_abs_tol=1e-4):
    """Print AD/FD comparison and pass/fail summary."""
    print(f"\nAcceptance threshold: relative error < {rel_tol:.2f}%")
    all_pass = True
    axis_names = ['tx_x', 'tx_y', 'tx_z']

    for i, axis_name in enumerate(axis_names):
        ad_val = float(ad[i])
        fd_val = float(fd[i])
        print(f"\n  {axis_name}:")
        print(f"    AD = {ad_val:.6e}")
        print(f"    FD = {fd_val:.6e}")

        if abs(fd_val) <= near_zero_fd:
            abs_err = abs(ad_val - fd_val)
            print(f"    near-zero FD axis, abs error = {abs_err:.6e}")
            passed = abs_err <= near_zero_abs_tol
            print(f"    pass(abs<{near_zero_abs_tol:.1e}) = {passed}")
        else:
            rel_err = abs(ad_val - fd_val) / abs(fd_val) * 100.0
            passed = rel_err < rel_tol
            print(f"    rel error = {rel_err:.3f}%")
            print(f"    pass(<{rel_tol:.2f}%) = {passed}")

        all_pass = all_pass and passed

    print("\nR-D AD/FD consistency:", "PASS" if all_pass else "FAIL")
    return all_pass

base_tx = (0.0, -5.0, 1.5)
print(f"\nComputing AD/FD at grid={grad_check_grid}, n_rays={grad_check_n_rays}, eps={eps}...")
rd_ad, rd_fd = compute_tx_ad_fd(
    gradient_check_tracer, base_tx, eps, grad_check_grid, range_xy, 1.5
)
_rd_grad_check_pass = report_tx_gradient_consistency(
    rd_ad, rd_fd, rel_tol=rel_tol_percent
)

# -----------------------------------------------------------------------------
# Deterministic Gradient Test (LoS + Diffraction only)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Deterministic Gradient Test (LoS + Diffraction)")
print("=" * 60)
print("Note: Reflection uses Monte Carlo sampling, which causes noise.")
print("Testing deterministic components only...\n")

# Create tracer with minimal reflections for deterministic test
tracer_det = Tracer(
    frequency=frequency,
    scene=scene,
    reflection_n_rays=1,  # Minimal rays (reflection will be negligible)
    reflection_max_bounces=0,  # No bounces
    enable_rd_diffraction=False
)

# Re-enable gradients
tx_x_det = wt.Float(0.0)
tx_y_det = wt.Float(-5.0)
tx_z_det = wt.Float(1.5)
dr.enable_grad(tx_x_det)
dr.enable_grad(tx_y_det)
dr.enable_grad(tx_z_det)

tx_pos_det = wt.Point3f(tx_x_det, tx_y_det, tx_z_det)
det_monitor = FieldMonitor(
    "det_plane",
    axis="z",
    position=1.5,
    bounds=(range_xy, range_xy),
    grid_size=64,
)

# Forward pass
result_det = tracer_det.trace(
    tx_pos_det,
    monitor=det_monitor,
    verbose=False,
)
assert_plane_monitor_result(result_det, det_monitor)
payload_det = result_det.primary

# Compute loss (LoS + Diffraction)
a_los_det = payload_det.field.los
a_dif_det = payload_det.field.diffraction
a_det = wt.Complex2f(a_los_det.real + a_dif_det.real, a_los_det.imag + a_dif_det.imag)
loss_det = dr.sum(a_det.real * a_det.real + a_det.imag * a_det.imag)

# Backward
dr.backward(loss_det)

grad_det_x = float(dr.grad(tx_x_det)[0])
grad_det_y = float(dr.grad(tx_y_det)[0])
grad_det_z = float(dr.grad(tx_z_det)[0])

print(f"AD Gradients (LoS + Dif):")
print(f"  d(loss)/d(tx_x) = {grad_det_x:.6e}")
print(f"  d(loss)/d(tx_y) = {grad_det_y:.6e}")
print(f"  d(loss)/d(tx_z) = {grad_det_z:.6e}")

# FD verification for deterministic case
def compute_det_loss(x, y, z):
    with dr.suspend_grad():
        pos = wt.Point3f(wt.Float(x), wt.Float(y), wt.Float(z))
        monitor = FieldMonitor(
            "det_plane",
            axis="z",
            position=1.5,
            bounds=(range_xy, range_xy),
            grid_size=64,
        )
        res = tracer_det.trace(
            pos,
            monitor=monitor,
            verbose=False,
        )
        assert_plane_monitor_result(res, monitor)
        a_l = res.primary.field.los
        a_d = res.primary.field.diffraction
        a = wt.Complex2f(a_l.real + a_d.real, a_l.imag + a_d.imag)
        return float(dr.sum(a.real * a.real + a.imag * a.imag)[0])

fd_det_x = (compute_det_loss(0.0 + eps, -5.0, 1.5) - compute_det_loss(0.0 - eps, -5.0, 1.5)) / (2 * eps)
fd_det_y = (compute_det_loss(0.0, -5.0 + eps, 1.5) - compute_det_loss(0.0, -5.0 - eps, 1.5)) / (2 * eps)
fd_det_z = (compute_det_loss(0.0, -5.0, 1.5 + eps) - compute_det_loss(0.0, -5.0, 1.5 - eps)) / (2 * eps)

print(f"\nFD Gradients (LoS + Dif):")
print(f"  d(loss)/d(tx_x) = {fd_det_x:.6e}")
print(f"  d(loss)/d(tx_y) = {fd_det_y:.6e}")
print(f"  d(loss)/d(tx_z) = {fd_det_z:.6e}")

print(f"\nRelative Errors (Deterministic):")
for name, ad, fd in [('tx_x', grad_det_x, fd_det_x),
                      ('tx_y', grad_det_y, fd_det_y),
                      ('tx_z', grad_det_z, fd_det_z)]:
    if abs(fd) > 1e-10:
        rel_err = abs(ad - fd) / abs(fd) * 100
        print(f"  {name}: {rel_err:.2f}%")
    else:
        print(f"  {name}: FD~0, AD={ad:.2e}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)

maybe_show()


