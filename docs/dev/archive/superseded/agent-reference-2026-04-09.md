# Agent Reference

Status: Active
Category: Standard
Last reviewed: 2026-04-09

## Purpose

This document holds the detailed repository reference that does not need to stay in the always-on root agent context. `AGENTS.md` and `CLAUDE.md` keep only the high-priority repository rules and point here for expanded guidance.

## Project Overview

This repository provides GPU-accelerated wireless channel and radio-propagation simulation based on ray tracing and UTD-style diffraction.

The current implementation focuses on:

- Line-of-sight (LoS)
- Multi-bounce reflection
- Direct and mixed diffraction families
- Validation helpers for canonical wedge scenes

The declarative public scene model is:

- `Scene`: scene assembly from `witwin.core.Structure`, `Material`, and geometry objects
- `Tracer`: propagation solver and field synthesis entrypoint
- `Result`: result dictionaries containing field components, solver metadata, and optional validation audits

## Repository Layout

- `witwin/channel/__init__.py`: public package exports
- `witwin/channel/scene/`: declarative channel scene package plus mesh and edge compilation for the runtime
- `witwin/channel/trace/`: propagation solver package with `tracer.py`, `los.py`, `reflection/`, and `diffraction/`
- `witwin/channel/validation.py`: canonical validation scenes and reference-comparison helpers
- `tests/`: pytest-based regression, smoke, validation, and acceptance coverage
- `tests/support/bin/`: manual visualization, benchmark, and optimization support scripts that stay under `tests/` for shared imports while remaining outside the collected pytest tree
- `tests/output/`: generated plots for manual visual inspection
- `basic/`: lightweight reference or sandbox scripts
- `samples/`: maintained example scripts and analysis helpers
- `figures/`: generated plots and figure outputs
- `docs/dev/`: migration plans, acceptance matrices, standards, and design notes

## Shared Utilities

Before defining a local helper at the top of a file, check whether `witwin/channel/utils/` already provides it. Do not duplicate general-purpose helpers.

### Key modules

| Module | Contents |
| --- | --- |
| `utils/drjit_ops.py` | DrJit array helpers such as `empty_point3`, `zeros_vector3`, `concat_arrays`, `broadcast_point`, `repeat_float`, `complex_zero`, `eval_complex`, `complex_abs_sqr`, `gather_point3`, `safe_normalize`, `array_scalar`, and `mask_count` |
| `utils/geometry.py` | Pure geometric primitives such as `point_in_triangle_3d`, `reflect_point_across_plane`, `surface_contains_point`, `compute_face_normals`, and `extract_edges_with_adjacency` |
| `utils/angles.py` | `spherical_angles` |
| `utils/plane_axes.py` | Axis normalization and axis-aligned plane helpers |
| `utils/polarization.py` | Jones calculus, field vector transport, and polarization projection |
| `utils/material.py` | Fresnel reflection coefficients |
| `utils/raygen.py` | Ray direction generators for sphere, hemisphere, cone, and circle sampling |
| `utils/constants.py` | Shared numeric constants such as `EPS`, `SMALL_EPS`, and `RAY_ORIGIN_BIAS` |
| `utils/__init__.py` | `to_numpy`, `scalar`, and explicit torch or DrJit interop helpers |

### Utility rules

- Do not define `_zero_point`, `_complex_zero`, `_broadcast_point`, `_concat_arrays`, or similar one-line helpers locally. Use the canonical names from `utils/drjit_ops.py`.
- Do not duplicate geometry helpers such as `point_in_triangle`, `reflect_point`, or `face_normals` inside `trace/` or `scene/`. Import from `utils/geometry.py`.
- Place new generic helpers in the appropriate `utils/` module rather than in a domain-specific file.
- Do not create thin re-export shim files whose only purpose is to forward imports.
- Do not create wrapper methods on classes that only delegate to a module-level function with `self` as the first argument.

## Module Layering

The dependency direction is:

`utils/` -> `scene/` -> `trace/` -> `monitors/` -> `tracer.py`

Layering rules:

- `utils/` must not import from `scene/`, `trace/`, or `monitors/`
- `scene/` must not import from `monitors/`
- `trace/` solver modules such as `reflection/`, `diffraction/`, and `los.py` should not import from `monitors/`
- Grid accumulation helpers live in `monitors/field/` and are called by solver entrypoints, not the other way around
- `tracer.py` is the top-level orchestrator and may import from all layers

Naming rules:

- Public API entrypoints for each solver live in `api.py`
- Domain constants for diffraction live in `trace/diffraction/constants.py`, not a `common.py` re-export layer
- Monitor-specific grid accumulation belongs in `monitors/field/grid_reflection.py` and `monitors/field/grid_diffraction.py`

## Working Expectations

- Prefer package-style imports from `witwin.channel` for scripts and examples, and shared scene-building types from `witwin.core`
- Build scenes with `Scene(structures=[...])`, `Scene.add_structure(...)`, or `Scene.add_mesh(...)`
- Do not add raw `vertices/faces` scene constructors or `Scene.from_meshes(...)` compatibility helpers
- Use `witwin.core.Structure`, `Material`, `Mesh`, and geometry primitives as the primary scene-building model
- When DrJit-native mesh buffers are required, wrap them as declarative geometry objects rather than adding legacy scene construction paths
- When behavior changes in a way a user would notice, update `FEATURE_LIST.md`
- Keep new implementations aligned with the declarative-scene and compiled-runtime split
- When changing numerics, preserve visibility semantics, wedge-angle conventions, polarization transport, and GPU execution flow
- Keep exploratory or script-style gradient workflows under `tests/support/bin/`, not under `witwin/channel/`
- Do not introduce new `rfdt` import paths; user-facing code should import from `witwin.channel`

## Running Code

Activate the environment first:

```bash
conda activate witwin2
```

When installing the local package with `pip install .`, include `--no-deps`.

Common commands:

```bash
cd channel
python -m pytest tests
python -m pytest tests --gpu
python -m pytest tests --gpu --acceptance
python -m pytest tests/test_core_scene_migration.py --gpu --acceptance
python -m pytest tests/test_validation_references.py --gpu --acceptance
python -m pytest tests/test_validation_state_audit.py --gpu --acceptance
python -m tests.support.bin.run_all
python -m pytest tests/main/test_position_rotation_tx.py --gpu
python -m pytest tests/main/test_multipath_main.py --gpu
python -m pytest tests/main/test_optimize_main.py --gpu --run-optimize
```

Validation workflow:

- Use the regular suite for fast regression checks on scene compilation, tracing utilities, and diffraction-state bookkeeping
- Use `--acceptance` to run only the end-to-end validation and reference-comparison coverage
- Use `--gpu` for tests that require CUDA, RayD, or GPU-resident torch tensors
- Run performance benchmarks serially and avoid unrelated GPU-heavy workloads during collection
- If you add or modify Python code, prefer targeted pytest coverage first and broader tests second

## Windows And Codex Execution Notes

- In this environment, `rg` may exist but fail with `Access is denied`; switch to PowerShell-native search with `Get-ChildItem` and `Select-String` instead of retrying `rg`
- `conda run -n witwin2` may fail to pass stdin through to `python -` in this Codex and PowerShell setup; prefer `conda run -n witwin2 python -c ...` for short snippets or invoke `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe` directly
- Windows command-line length limits are easy to hit with `python -c` and large inline scripts; for long scripts, pass code through an environment variable such as `PYCODE` and execute `python -c "import os; exec(os.environ['PYCODE'])"`, or use the environment interpreter with stdin if that path is reliable
- Do not attempt to write a large file in one shot from the CLI on Windows; split large file creation or rewrites into smaller chunks or incremental patches
- Large single-shot `apply_patch` updates can also hit Windows path and command length limits; if a full-file replacement fails, rewrite it in smaller patch chunks
- When checking whether `AGENTS.md` and `CLAUDE.md` match, do not use bare `fc` in PowerShell because it resolves to `Format-Custom`; call `fc.exe` explicitly

## Dependencies

Primary dependencies include:

- PyTorch
- NumPy
- SciPy
- Matplotlib
- `tqdm`
- RayD
- DrJit
- `witwin`

An NVIDIA GPU with CUDA is expected for core solver workflows.
