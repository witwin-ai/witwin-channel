# Agent Reference

Status: Active
Category: Standard
Last reviewed: 2026-05-20

## Purpose

This document holds the detailed repository reference that does not need to stay in the always-on root context for `AGENTS.md` and `CLAUDE.md`. Agent-runtime-specific operating notes for Codex, Claude Code, and the Superpowers skill pack live next to this file at `11-codex-operating-guide.md`, `12-claude-code-operating-guide.md`, and `13-superpowers-operating-guide.md`.

## Project Overview

This repository (`channel/`) is the wireless channel and radio-propagation subproject of the `witwin-platform` monorepo. The solver is GPU-first and built around ray tracing plus UTD-style diffraction.

The current implementation covers:

- Line-of-sight (LoS)
- Multi-bounce reflection
- Direct and mixed diffraction families, including shadow-boundary correction
- Deterministic and Monte Carlo radiomaps
- Path-level solver for multi-transmitter scenes
- Closed-form validation helpers for canonical wedge scenes

The stable public architecture is:

- `Scene`: declarative scene assembly built from `witwin.core.Structure`, `Material`, geometry objects, and scene-owned endpoints (`Transmitter`, `Receiver`, `ReceiverGrid`).
- Solver entrypoints: each solver package exposes a `solve(scene, config)` function. There is no shared `Tracer` object.
- `Result`: typed result payloads. Radiomap solvers share `witwin.channel.core.results.RadioMapResult`. The path solver returns `witwin.channel.path.PathResult`.

## Repository Layout

| Path | Role |
| --- | --- |
| `witwin/channel/core/scene/` | Declarative `Scene`, `Mesh`, scene-owned `Transmitter`/`Receiver`/`ReceiverGrid`, Sionna and RayD adaptors, wedge runtime metadata |
| `witwin/channel/core/` | Shared utilities used across solver packages: geometry, materials, polarization, grids, raygen, kernels, native helpers, and the shared `RadioMapResult` payload |
| `witwin/channel/deterministic/` | Deterministic radiomap solver (`solve`), config, field solver, native and DrJit kernels |
| `witwin/channel/montecarlo/` | Monte Carlo radiomap solver (`solve`), config, filtering, integrators, sampler, native and DrJit kernels |
| `witwin/channel/path/` | Path-level multi-transmitter solver returning per-interaction path records |
| `witwin/channel/` | User-facing umbrella package that re-exports channel scene/core types and solver namespaces (`paths`, `deterministic`, `montecarlo`) |
| `tests/deterministic/` | Deterministic-solver regression and acceptance coverage |
| `tests/montecarlo/` | Monte Carlo-solver regression and acceptance coverage |
| `tests/scene/` | Scene, mesh, and edge-compilation coverage |
| `tests/support/bin/` | Manual visualization, benchmark, and optimization scripts kept under `tests/` for shared imports but outside the collected pytest tree |
| `docs/dev/` | Plans, optimization notes, standards, and bug inventories |

## Module Layering

The current dependency direction is:

`witwin/channel/core/` -> `witwin/channel/core/scene/` -> `{deterministic, montecarlo, path}/`

Layering rules:

- `witwin/channel/core/` must not import from any solver package or from `witwin/channel/core/scene/`.
- `witwin/channel/core/scene/` must not import from any solver package.
- Each solver package (`deterministic/`, `montecarlo/`, `path/`) owns its own runtime state, native bindings, and result construction. Solver packages must not import from each other.
- Shared payload types (currently `RadioMapResult`) live under `witwin/channel/core/` so deterministic and Monte Carlo can both produce them without cross-package coupling.

## Shared Utility Ownership

Before defining a local helper at the top of a file, check whether `witwin/channel/core/` already provides it. Do not duplicate generic helpers.

| Module | Contents |
| --- | --- |
| `channel/core/numerics/` | Shared constants, DrJit array constructors, shape helpers, and torch interop at API boundaries |
| `channel/core/geometry/` | Pure geometric primitives, diffraction geometry, mesh buffers, and ray-direction generators |
| `channel/core/physics/` | Material model, Jones calculus, field-vector transport, wave math, and boundary policies |
| `channel/core/runtime/` | Solve-time context objects, material assertions, and scene-aware runtime helpers |
| `channel/core/grid.py` | Receiver-grid sampling, including the canonical scene-owned grid contract |
| `channel/core/results/radiomap_result.py` | Shared `RadioMapResult` payload returned by deterministic and Monte Carlo solvers |
| `channel/core/kernels/`, `_native/` | DrJit and CUDA kernel helpers shared across solver backends |

## Public API Surface

Prefer the umbrella channel namespace for user-facing examples:

```python
import witwin.channel as wc

scene = wc.Scene(
    structures=[...],
    transmitters=[wc.Transmitter("tx", (...))],
    receivers=[wc.ReceiverGrid("rm", axis="z", position=1.5, bounds=..., grid_shape=...)],
)

result = wc.deterministic.solve(
    scene=scene,
    frequency=3.5e9,
    transmitter="tx",
    receiver="rm",
    config=wc.deterministic.Config(
        num_samples=256,
        max_bounces=1,
        max_diffraction_order=0,
        edge_policy=wc.EdgePolicy(edge_selection_mode="all_edges"),
    ),
)
path_result = wc.path.solve(
    scene=scene,
    frequency=3.5e9,
    transmitter="tx",
    receiver="rx0",
    config=wc.path.Config(num_samples=256, max_bounces=1, max_diffraction_order=0),
)
```

`witwin.channel.path` is the canonical path solver package. Deterministic and Monte Carlo radiomap public solves use scene-owned `Transmitter` and `ReceiverGrid` endpoints selected with `transmitter=` and `receiver=`. Public radiomap solver signatures do not accept `tx_pos=`, `grid=`, or `return_timing=`.

Solver configs keep the common user budget fields at the top level (
um_samples`, `max_bounces`, `max_diffraction_order`) and put advanced runtime controls under `Tuning`. Diffraction edge policy is only configured with `edge_policy=wc.EdgePolicy(...)`; do not add `edge_selection_mode`, `edge_diffraction`, `boundary_edge_policy`, or `vertical_ratio` as direct `Scene` or `Config` fields. Do not add public `ray_mode` fields to solver configs; standard solvers use 3D tracing, and shared result-level ray mode normalization belongs under `witwin.channel.core.results`. Monte Carlo integrator-specific controls live under `wc.montecarlo.IntegratorOptions`.

Scene construction is declarative. Do not add raw `vertices/faces` scene constructors or `Scene.from_meshes(...)` compatibility helpers. Build scenes from `witwin.core.Structure` and `witwin.channel.core.scene.Mesh` instances and attach structures or endpoints with the unified `Scene.add(obj)`, which dispatches over `Structure`, `Transmitter`, `Receiver`, and `ReceiverGrid`.

## Working Expectations

- Prefer package-style imports from `witwin.*` namespaces for scripts and examples.
- Keep runtime internals DrJit-native. Torch is allowed only at explicit public API or result-adapter boundaries.
- When behavior changes in a way a user would notice, update `FEATURE_LIST.md` in the same change.
- Keep exploratory or script-style gradient workflows under `tests/support/bin/`, not under `witwin/`.
- Do not introduce new `rfdt` import paths. User-facing code should import from the `witwin.*` namespace.

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
python -m pytest tests/scene
python -m pytest tests/deterministic
python -m pytest tests/montecarlo
python -m tests.support.bin.run_all
```

Validation workflow:

- Use the regular suite for fast regression checks on scene compilation, tracing utilities, and diffraction-state bookkeeping.
- Use `--acceptance` to run only the end-to-end validation and reference-comparison coverage.
- Use `--gpu` for tests that require CUDA, RayD, or GPU-resident torch tensors.
- Run performance benchmarks serially and avoid unrelated GPU-heavy workloads during collection.
- If you add or modify Python code, prefer targeted pytest coverage first and broader tests second.

## Dependencies

Primary dependencies:

- PyTorch (CUDA)
- NumPy, SciPy, Matplotlib, `tqdm`
- DrJit
- RayD
- `slangtorch` for native kernel hooks

An NVIDIA GPU with CUDA is expected for core solver workflows.
