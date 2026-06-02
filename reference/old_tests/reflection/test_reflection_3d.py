"""Test script for 3D reflection tracing implementation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from witwin.channel import FieldMonitor, Tracer
from witwin.channel.utils import to_power_db, to_numpy
from tests._scene_helpers import box_geometry, build_scene
# Test parameters
frequency = 2.4e9  # 2.4 GHz (WiFi frequency)
tx_pos = (0, -5, 1.5)  # Transmitter position
grid_size = 128  # Use smaller grid for faster testing

print("=" * 60)
print("Testing 3D Reflection Tracing Implementation")
print("=" * 60)

# Create a simple cube mesh for testing
print("\n1. Creating test geometry (cube)...")
cube = box_geometry(center=(0, 0, 1.0), size=4.0)
cube_vertices, cube_faces = cube.to_mesh()
print(f"   Mesh: {cube_vertices.shape[0]} vertices, {cube_faces.shape[0]} faces")

# Initialize tracer with reflection parameters
print("\n2. Initializing tracer...")
scene = build_scene(cube)
tracer = Tracer(
    frequency=frequency,
    scene=scene,
    reflection_n_rays=360,  # Use fewer rays for faster testing
    reflection_max_bounces=2,
    reflection_coef=0.7,
    resolution_wavelength=0.25  # 1/4 resolution
)
print(f"   Frequency: {frequency/1e9:.1f} GHz")
print(f"   Wavelength: {tracer.wavelength:.4f} m")
print(f"   Resolution: 1/{1/tracer.resolution_wavelength:.0f} (cell_size={tracer.cell_size:.4f} m)")
print(f"   Reflection rays: {tracer.reflection_n_rays}")
print(f"   Max bounces: {tracer.reflection_max_bounces}")
monitor = FieldMonitor(
    "reflection_plane",
    axis="z",
    position=1.5,
    bounds=((-8.0, 8.0), (-8.0, 8.0)),
    grid_size=grid_size,
)

# Run simulation
print("\n3. Running simulation...")
print(f"   Tx position: {tx_pos}")
print(f"   Grid size: {grid_size}x{grid_size}")

result = tracer.trace(
    tx_pos=tx_pos,
    monitor=monitor,
)
payload = result.primary

# Check results
print("\n4. Checking results...")

# Compute dB from complex fields
los_power = to_numpy(to_power_db(payload.field.los))
ref_total_power = to_numpy(to_power_db(payload.field.reflection))

# Check LoS field
print(f"   LoS field shape: {los_power.shape}")
print(f"   LoS power range: [{np.nanmin(los_power):.1f}, {np.nanmax(los_power):.1f}] dB")

# Check reflection field
print(f"   Reflection total field shape: {ref_total_power.shape}")
print(f"   Reflection power range: [{np.nanmin(ref_total_power):.1f}, {np.nanmax(ref_total_power):.1f}] dB")

# Verification
print("\n5. Verification:")
valid_los = np.isfinite(los_power).all()
valid_ref = np.isfinite(ref_total_power).all()
print(f"   LoS field valid: {valid_los}")
print(f"   Reflection field valid: {valid_ref}")

# Check that reflection has some contribution
ref_has_contribution = np.any(np.isfinite(ref_total_power) & (ref_total_power > -100))
print(f"   Reflection has contribution: {ref_has_contribution}")

# Summary
print("\n" + "=" * 60)
if valid_los and valid_ref and ref_has_contribution:
    print("[OK] TEST PASSED: 3D reflection tracing working correctly!")
else:
    print("[FAIL] TEST FAILED: Issues detected")
    if not valid_los:
        print("  - LoS field has invalid values")
    if not valid_ref:
        print("  - Reflection field has invalid values")
    if not ref_has_contribution:
        print("  - Reflection field has no contribution")
print("=" * 60)

