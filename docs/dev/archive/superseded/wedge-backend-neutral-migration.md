# Wedge Backend-Neutral Migration Plan

## Goal

Create a standalone `witwin.channel.wedge` module that owns wedge geometry compilation, selection, anchor generation, triangle-wedge adjacency, and packed GPU buffers without binding the core implementation to Mitsuba-specific scene internals. The module must fit the current `Scene + Tracer + Result` architecture while preparing the codebase for a gradual Mitsuba-to-RayD backend migration.

## Motivation

The current channel runtime spreads wedge-related responsibilities across:

- `witwin/channel/scene/runtime.py`
- `witwin/channel/scene/topology.py`
- `witwin/channel/trace/diffraction/common.py`
- `witwin/channel/trace/diffraction/builders.py`

This makes wedge geometry, wedge selection policy, GPU packing, and diffraction-state construction too tightly coupled. It also hard-codes Mitsuba-oriented scene/runtime assumptions into logic that should eventually run on top of RayD topology/query backends.

## Target Architecture

### Wedge Core

`witwin.channel.wedge` should become the only owner of:

- Edge-to-wedge geometry compilation
- Ordered wedge normal convention (`n0`, `nn`)
- Wedge exterior angle and `wedge_n`
- Selection policies such as `vertical_only` and `all_edges`
- Boundary handling such as `exclude` and `half_plane`
- Height-plane anchor generation
- Triangle-to-wedge adjacency mapping
- GPU-first packed wedge buffers for downstream diffraction solvers

The wedge core must depend only on abstract backend contracts, not on Mitsuba `Scene` internals and not on RayD implementation details.

### Backend Adapters

Backend-specific code must live under adapter files:

- `witwin/channel/wedge/adapters/mitsuba_scene.py`
- `witwin/channel/wedge/adapters/rayd_scene.py`

The Mitsuba adapter is phase 1. It keeps the current channel implementation working.

The RayD adapter is phase 2. It should connect the wedge core to RayD scene edge/topology/query APIs once the diffraction builder path is ready to consume backend-neutral wedge packs.

### Diffraction Boundary

The wedge module must not own:

- UTD coefficient evaluation
- Diffraction state array recursion
- Reflection prefix/suffix expansion
- Mixed-path pruning and audit policy

Those remain in `witwin.channel.trace.diffraction`.

The planned contract is:

- `wedge` produces `WedgePack` and `TriangleWedgeMap`
- `diffraction` consumes those packed structures

## Package Layout

Planned package structure:

- `witwin/channel/wedge/__init__.py`
- `witwin/channel/wedge/array_api.py`
- `witwin/channel/wedge/config.py`
- `witwin/channel/wedge/contracts.py`
- `witwin/channel/wedge/types.py`
- `witwin/channel/wedge/compile.py`
- `witwin/channel/wedge/select.py`
- `witwin/channel/wedge/anchors.py`
- `witwin/channel/wedge/mapping.py`
- `witwin/channel/wedge/pack.py`
- `witwin/channel/wedge/cache.py`
- `witwin/channel/wedge/runtime.py`
- `witwin/channel/wedge/adapters/__init__.py`
- `witwin/channel/wedge/adapters/mitsuba_scene.py`
- `witwin/channel/wedge/adapters/rayd_scene.py`

## Backend-Neutral Contracts

The wedge core should target RayD-shaped contracts even when the active backend is Mitsuba.

Required topology/query contract:

- `version()`
- `edge_version()`
- `n_triangles()`
- `edge_info()`
- `edge_topology()`
- `triangle_edge_indices(prim_id, global_=True)`
- `edge_adjacent_faces(edge_id, global_=True)`
- `mesh_face_offsets()`
- `mesh_edge_offsets()`
- `shadow_test(ray, active=True)`
- `intersect(ray, active=True, flags=None)`

This keeps the wedge core ready for a future RayD adapter with minimal downstream changes.

## GPU Performance Rules

- Store core wedge data as structure-of-arrays DrJit buffers.
- Separate caches into geometry, selection, and anchor/pack layers.
- Key caches by backend `edge_version()` plus immutable config objects.
- Avoid Python-object edge lists on the hot path after compilation.
- Avoid CPU round trips except for debug summaries and transition-only topology compilation in the Mitsuba adapter.
- Use compressed index buffers (`dr.compress`) once per selection stage and reuse them for all downstream gathers.
- Keep `dr.eval()` calls at stage boundaries only.

## Phases

### Phase 1

Deliver a backend-neutral wedge package with:

- Immutable config types
- Abstract backend contracts
- Core wedge geometry compilation
- Selection and anchor generation
- Triangle-wedge mapping
- Packed wedge GPU buffers
- Mitsuba scene adapter
- Smoke tests for wedge geometry/pack construction

No solver math changes in phase 1.

### Phase 2

Switch `diffraction.builders` to consume `WedgePack` and `TriangleWedgeMap` instead of:

- `scene.get_edge_data(...)`
- `scene._tri_edge_indices`
- `scene._diffraction_edge_gpu`
- `scene.vertical_edges`

### Phase 3

Add `RayDSceneAdapter` and route wedge topology/query calls through RayD while keeping the current diffraction solver intact.

Current execution note:

- The first RayD-shaped adapter now targets the native `rayd.Scene` topology/query contract without binding the wedge core to RayD imports at module import time.
- `Scene` now builds a parallel native RayD 0.1.3 runtime mesh when `wedge_backend` or `query_backend` requests it, and both the wedge runtime and scene query path can consume that backend while the rest of the public API stays unchanged.
- The current default is `wedge_backend="rayd"` so channel scenes now build and consume native RayD wedge topology by default.
- The current query-path default is also `query_backend="rayd"` so scene visibility and intersection queries now route through native RayD by default, while Mitsuba remains available as an explicit opt-in backend for comparison and migration debugging.
- Mixed diffraction regression coverage now exercises both the default wedge auto path and the explicit native RayD wedge/query paths.

### Phase 4

Retire Mitsuba-owned edge topology/preload logic from `scene/runtime.py` once diffraction no longer depends on it.

## Immediate Execution Scope

This change set starts phase 1 by:

1. Adding this migration plan.
2. Creating the new `witwin.channel.wedge` package skeleton.
3. Implementing the backend-neutral runtime/cache/types/contracts.
4. Implementing the Mitsuba adapter needed by the current channel `Scene`.
5. Adding a smoke test for wedge geometry/pack construction.

## Open Constraints

- The current declarative `Scene` still owns a Mitsuba scene handle for ray queries.
- The current channel scene merge path does not preserve RayD-style per-structure edge ownership metadata.
- The Mitsuba adapter is therefore a transition backend. It should prioritize correctness, gradient preservation, and clean contracts over perfect parity with future RayD metadata richness.
