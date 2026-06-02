"""Test sim.py step by step"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

print("Step 1: Importing matplotlib...")
sys.stdout.flush()
import matplotlib.pyplot as plt
print("OK")

print("Step 2: Importing backend...")
sys.stdout.flush()
import witwin as wt
print("OK")

print("Step 3: Importing scene...")
sys.stdout.flush()
from tests._scene_helpers import box_geometry, build_scene
print("OK")

print("Step 4: Importing visualization...")
sys.stdout.flush()
from witwin.channel import draw_scene
print("OK")

print("Step 5: Importing tracer...")
sys.stdout.flush()
from witwin.channel import Tracer
print("OK")

print("Step 6: Creating mesh...")
sys.stdout.flush()
cube = box_geometry(center=(0, 0, 2.0), size=4.0)
cube_vertices, cube_faces = cube.to_mesh()
print(f"Mesh: {cube_vertices.shape[0]} vertices, {cube_faces.shape[0]} faces")

print("Step 7: Creating tracer...")
sys.stdout.flush()
freq = 1e9
tx_pos = (-5, 5, 1.5)
scene = build_scene(cube)
tracer = Tracer(frequency=freq, scene=scene)
print("OK")

print("\n[SUCCESS] All steps completed!")


