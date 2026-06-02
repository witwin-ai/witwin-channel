# Source Tree Restructure Plan

Status: Complete
Category: Plan
Last reviewed: 2026-05-20
Owners: channel team

Related:

- `plans/13-channel-readability-conciseness-followup-plan.md` - file-level / line-level conciseness (orthogonal; this plan covers directory-level structure)
- `plans/14-call-site-structural-debt-plan.md` - call-site debt
- `plans/24-realtime-rt-architecture-roadmap.md` - broader architecture direction
- `optimization/channel-readability-conciseness-audit-2026-05-16.md` - baseline audit

## Purpose

Document the directory-level structural problems in `witwin/` and propose a phased restructure. This plan is orthogonal to plan 13: that plan addresses file-internal conciseness ("megamodules"), while this plan addresses directory-level organization ("where does X live"). Both can land in parallel.

The motivating observation from Phase 0 was that, excluding `kernels/` and `_native/`, the Python orchestration layer was `25.2k LOC across 83 files`, which was about `1.40x` Sionna RT's `~18k LOC / 49 files`. The remaining gap is largely justified by witwin's extra solvers (deterministic radiomap, BDPT, AD variants). The perceived "bloat" comes mostly from organizational scatter, not real volume. This plan targets that scatter.

## Phase Status Snapshot

Captured and updated on 2026-05-20.

| Area | Phase 0 state | Current status |
|---|---|---|
| Active top-level package directories | `channel`, `channel_scene`, `channel_utils`, `deterministic`, `montecarlo`, `path` | Complete: `channel` is the only public channel-package root under `witwin/`, excluding cache directories. Solver implementations now live under `witwin/channel/`. |
| Cache-only top-level directories | `_native`, `_config`, `channel_core` were cache-only | Complete: `_native`, `_config`, and shared DrJit type aliases now live under `witwin/channel/`; no separate top-level `channel_core` package exists. |
| Python orchestration files | 83 files, 25,230 LOC excluding `kernels/`, `_native/`, and `__pycache__/` | Complete: the restructure moved ownership and removed wrapper packages without adding compatibility shims. |
| `channel_utils/` + `channel_scene/` | 26 files, 4,700 LOC excluding `_native/` | Phase 4 complete: both public top-level packages were removed; shared utilities now live in `witwin/channel/core/` and scene runtime lives in `witwin/channel/core/scene/`. |
| Internal `path` directories | Public path solver plus deterministic and Monte Carlo internal path subpackages collided | Phase 2 renamed internal solver dirs to `trace`; Phase 5 moves the public path solver to `witwin/channel/path`. |
| Native directories | Three package-local native spec directories each had a Python spec | Phase 1 complete: loader and specs live under `witwin/channel/_native/`; CMake installs there. |
| Config sharing | `deterministic/config.py` and `montecarlo/config.py` duplicated shared concepts | Phase 3 complete: shared tuning validation, resolved trace wave parameters, and solver guardrails live in `witwin/channel/_config/base.py`. |
| AD variants | `basic_ad.py`, `bdpt_ad.py`, and `diffraction_ad.py` were sibling files | Complete: AD variants are flattened as sibling `*_ad.py` modules next to their primary modules. |
| Solver package namespace | Solver implementations lived as top-level packages while `witwin/channel/*` provided thin public wrappers | Complete: solver implementations live in `witwin/channel/deterministic`, `witwin/channel/montecarlo`, and `witwin/channel/path`; old top-level solver packages and plural path wrappers are removed. |

## Identified Problems

### P0 - High Impact

#### P0.0 - Solver implementations live outside the channel namespace

After Phase 4, the package had a clean channel-owned core under `witwin/channel/core/`, but the three solver implementations still lived as top-level packages while `witwin/channel/*` provided public wrapper namespaces. That split no longer bought much: the solvers are channel-domain code, they depend on `witwin.channel.core`, and examples already prefer `import witwin.channel as wc`.

Target direction:

```
witwin/channel/deterministic/
witwin/channel/montecarlo/
witwin/channel/path/
```

This makes `witwin.channel` the single public root for channel scene, shared channel core, and all channel solvers. It also removes the previous singular/plural path-solver mismatch: the canonical package and umbrella attribute are both singular.

This is a breaking public import move. It should be Phase 5, after Phase 4, with no top-level compatibility packages unless maintainers explicitly override the repository rule against compatibility shims.

#### P0.1 - channel core and scene overlap

Phase 0 found that the split between the two utility directories was inconsistent and forced readers to search two places for related concepts.

| Concept | Former utility package | Former scene package |
|---|---|---|
| Arrays | `arrays.py` | `arrays.py` (same name, different content) |
| Materials | `materials.py` | `material_presets.py` |
| Mesh | `mesh_buffers.py` | `mesh.py` |
| Geometry | `diffraction_geometry.py`, `geometry.py` | `wedge.py` |

Combined: `28 files, 4,664 LOC` after Phase 2, excluding `_native/`. Sionna keeps the equivalent in a single `utils/` package (`11 files`).

Symptom: cross-cutting concerns (material binding, wedge geometry, scene mesh) are split arbitrarily; the `arrays.py` filename collision causes IDE jump-to-definition ambiguity.

Phase 4 status on 2026-05-20: complete. Shared utilities now live under `witwin/channel/core/`; scene-owned runtime and interop modules live under `witwin/channel/core/scene/`. The old top-level `witwin/channel_utils/` and `witwin/channel_scene/` packages are removed without compatibility shims.

#### P0.2 - Three different `path/` directories

Phase 0 state:

```
public path solver package
deterministic solver internal path subsystem
Monte Carlo solver internal path subsystem
```

Phase 2 status on 2026-05-20: complete. Maintainer decision: use `trace` for both internal solver trace subsystems, not `traversal` and not `transport`.

Current state:

```
witwin/channel/path/                    # public path solver
witwin/channel/deterministic/trace/     # deterministic internal trace subsystem
witwin/channel/montecarlo/trace/        # Monte Carlo internal trace subsystem
```

This reserves `witwin/channel/path/` for the public path solver while keeping the internal solver names short and symmetric.

#### P0.3 - `_impl/` over-nesting under `deterministic/path/`

Phase 0 state:

```
deterministic internal path/diffraction_impl/   # 6 files, 4-level deep
deterministic internal path/reflection_impl/    # 4 files
```

The `_impl` suffix borrows C++ pimpl semantics, but in Python these were core algorithm modules with public-facing exports. The 4-level depth made import paths long and obscured algorithm modules behind a generic `_impl` namespace.

Phase 1 status on 2026-05-20: complete. The implementation modules now live in:

```
witwin/channel/deterministic/diffraction/
witwin/channel/deterministic/reflection/
```

### P1 - Medium Impact

#### P1.1 - Three fragmented `_native/` directories

Phase 0 state:

```
witwin/channel/_native/channel_utils.py
witwin/channel/_native/deterministic.py
witwin/channel/_native/montecarlo.py
```

Each had its own spec boilerplate. Plan 13 already unified the `NativeExtensionLoader` class itself, but the per-package `_native/` directories remained.

Phase 1 status on 2026-05-20: complete. Loader ownership and specs now live under `witwin/channel/_native/`, and native CMake install destinations point at `witwin/channel/_native`.

#### P1.2 - `config.py` near-duplication between solvers

- `deterministic/config.py`
- `montecarlo/config.py`

Both share grid, frequency, material policy, and output spec fields. Future config fields need to be added twice unless a shared base is extracted.

#### P1.3 - `witwin/channel/` umbrella package cleanup

```
witwin/channel/
  deterministic/    # thin public namespace
  montecarlo/       # thin public namespace
  paths/            # thin public namespace
```

This is not simply a legacy alias: `docs/dev/standards/10-agent-reference.md` defines `witwin.channel` as the user-facing umbrella namespace, and examples prefer `import witwin.channel as wc`. Phase 1 therefore kept `witwin/channel/` as the public channel package. After Phase 4, the cleaner endpoint is stronger: move the solver implementations themselves under `witwin/channel/` instead of keeping thin wrappers over top-level solver packages.

Phase 1 status on 2026-05-20: audit complete. No `witwin/channel/` files were removed because the package is the supported public umbrella.

### P2 - Lower Impact

#### P2.1 - AD variants scattered as sibling files

Phase 0 state:

```
montecarlo/integrators/basic.py + basic_ad.py
montecarlo/integrators/bdpt.py + bdpt_ad.py
montecarlo/path/diffraction.py + diffraction_ad.py
```

Phase 2 follow-up status on 2026-05-20: complete. Current state:

```
montecarlo/integrators/basic.py
montecarlo/integrators/bdpt.py
montecarlo/integrators/basic_ad.py
montecarlo/integrators/bdpt_ad.py
montecarlo/trace/diffraction.py
montecarlo/trace/diffraction_ad.py
```

No deprecation shims were added; repository policy is to avoid legacy compatibility paths unless explicitly requested.

#### P2.2 - Solver-parallel internal modules

Both `deterministic/` and `montecarlo/` have private modules with similar names (`grid_ops.py`, `types.py`). Some shared concepts could lift; some are solver-specific. Plan 13's P0 already pulled a shared `grid.py` base out; this is a continuation, not a duplicate effort.

## Proposed Target Structure

```
witwin/
  channel/              # user-facing channel umbrella package
    __init__.py
    _native/            # channel-owned native loader/specs
      __init__.py
      loader.py
      channel_utils.py  # internal native spec name retained for the existing binary target
      deterministic.py
      montecarlo.py
    _config/            # shared config base + per-solver public boundaries
      __init__.py
      base.py
    types.py            # shared DrJit runtime aliases
    core/               # P0.1 complete: channel-owned core, not platform `witwin.core`
      materials.py
      polarization.py
      wave_math.py
      constants.py
      tensors.py
      grid.py
      shadow_boundary_policy.py
      runtime.py
      radiomap_result.py
      kernels/
      scene/
        scene.py
        builder.py
        edge_policy.py
        endpoints.py
        mesh.py
        sionna_adaptor.py
    deterministic/      # P0.0 target: deterministic implementation
      solver.py
      field.py
      trace/            # P0.2 complete: renamed from internal path/
      diffraction/      # P0.3 complete: was path/diffraction_impl/
        builders.py
        forward.py
        state.py
        postprocessing.py
        math.py
      reflection/       # P0.3 complete: was path/reflection_impl/
        epc.py
        paths.py
      kernels/
    montecarlo/         # P0.0 target: Monte Carlo implementation
      solver.py
      sampler.py
      filtering.py
      trace/            # P0.2 complete: renamed from internal path/
        ad/
          diffraction.py
      integrators/
        basic.py
        bdpt.py
        bdpt_diffraction.py
        ad/
          basic.py
          bdpt.py
      kernels/
    path/               # P0.0 target: path solver implementation
      solver.py
      config.py
      result.py
```

## Expected Metrics

| Metric | Phase 0 | After full plan |
|---|---|---|
| Active top-level package directories under `witwin/` | 6 | 1 package directory (`channel`) |
| Cache-only top-level directories under `witwin/` | 3 (`_native`, `_config`, `channel_core`) | 0 |
| Python files (no kernels, no `_native`) | 83 | about 65 after Phase 3/4 consolidation |
| Python LOC (no kernels, no `_native`) | 25.2k | about 24k-25k |
| Max nesting depth | 4 | 4 for solver internals under `channel/`; public import ownership becomes simpler even though some implementation paths get one directory deeper |
| Duplicate filenames (`arrays.py` x2) | yes | no |
| Internal/public `path/` namespace collision | yes | no; canonical public solver package becomes `witwin.channel.path` |
| `_native/` directories | 3 | 1 |

## Phased Execution

Ordered by `(risk x scope)`, starting with low-risk contained changes and deferring breaking changes to a major version boundary. Phases can be paused or resumed independently; each phase leaves the tree in a consistent state.

### Phase 0 - Refresh current state before execution

- Re-run the file/LOC and directory metrics before starting.
- Delete stray cache-only directories only through the normal cleanup workflow; do not treat `__pycache__`-only directories as implemented package structure.
- Update this plan if the current tree has already completed any listed phase.

Status on 2026-05-20: complete.

### Phase 1 - Internal cleanup, no breaking changes

- P1.3 first: audit `witwin/channel/` as the public umbrella namespace. Keep supported imports such as `import witwin.channel as wc`; remove only redundant wrappers proven unsupported.
- P1.1: consolidate the three `_native/` directories into `witwin/channel/_native/`. Move `NativeExtensionLoader`, `NativeExtensionSpec` declarations, CMake install destinations, and native-extension imports to the centralized package.
- P0.3: rename `deterministic/path/diffraction_impl/` to `deterministic/diffraction/` and `deterministic/path/reflection_impl/` to `deterministic/reflection/`.

Status on 2026-05-20: complete. Added `tests/test_source_tree_phase1_structure.py` to lock the new structure and old private path removal.

### Phase 2 - Internal rename for clarity

- P0.2: rename `deterministic/path/` to `deterministic/trace/` and `montecarlo/path/` to `montecarlo/trace/`. The maintainer chose one shared internal name, `trace`, instead of split names `traversal` and `transport`.
- P2.1 follow-up: flatten Monte Carlo AD files back into sibling `*_ad.py` modules and remove the standalone `ad/` packages.

Status on 2026-05-20: complete. Structure tests now assert that old `path`, `traversal`, and `transport` locations are absent.

### Phase 3 - Internal restructure with shared base

- P1.2: extract `_config/base.py` with shared dataclasses.
- Keep each solver's `config.py` as the solver-specific public config boundary.
- Coordinate with plan 13 P1 work to avoid churn.

Status on 2026-05-20: complete. `witwin/channel/_config/base.py` now owns common trace tuning validation, resolved wave-parameter calculation, and shared solver guardrails. Deterministic and Monte Carlo still expose their solver-specific public `Config`, `Tuning`, and resolved config classes from their own modules.

### Phase 4 - Public API restructure (breaking change)

- P0.1: merge channel-specific `channel_utils/` and `channel_scene/` modules into `witwin/channel/core/`, not platform-level `witwin.core`.

This was a breaking change for old public imports from the former top-level utility and scene packages.

The supported forms are now the stable `witwin.channel` umbrella or explicit `witwin.channel.core.*` imports:

```
import witwin.channel as wc
from witwin.channel.core.materials import ...
from witwin.channel.core.scene import Scene
```

Status on 2026-05-20: complete.

- Old top-level packages `witwin/channel_utils/` and `witwin/channel_scene/` were removed.
- Imports across `witwin/`, `tests/`, `examples/`, and active docs now target `witwin.channel.core` and `witwin.channel.core.scene`.
- `witwin.channel` re-exports the core scene and radiomap convenience types used by examples (`Scene`, endpoints, `Grid`, `GridSpec`, `RadioMapResult`, etc.).
- The native binary target keeps its existing internal `_channel_utils_native` name, but CMake now builds shared channel kernels from `witwin/channel/core/kernels/`.
- No compatibility shim was added for old public package imports.

Release note: this still requires coordinated downstream communication and versioning at release time because the public import paths changed.

### Phase 5 - Solver package ownership under `witwin.channel` (breaking change)

- P0.0: move the former top-level solver implementations into the channel package:
  - deterministic solver -> `witwin/channel/deterministic/`
  - Monte Carlo solver -> `witwin/channel/montecarlo/`
  - path solver -> `witwin/channel/path/`
- Move shared internal support under the same channel ownership boundary:
  - `witwin/channel/_config/` -> `witwin/channel/_config/`
  - `witwin/channel/_native/` -> `witwin/channel/_native/`
- Move shared public runtime aliases from `witwin/types.py` to `witwin/channel/types.py`; keep the top-level `witwin/__init__.py` as a namespace package entrypoint only.
- Keep solver packages independent from each other after the move. The dependency rule becomes:

```
witwin/channel/core/ -> witwin/channel/core/scene/ -> {witwin/channel/deterministic, witwin/channel/montecarlo, witwin/channel/path}
```

- Do not keep top-level solver, config, or native compatibility packages unless maintainers explicitly approve deprecation shims for a release window.
- Preserve user-facing `import witwin.channel as wc`; `wc.deterministic`, `wc.montecarlo`, and `wc.path` should point at the real implementation packages under `witwin/channel/`.
- Use singular `witwin.channel.path` as the canonical package path because it is the solver name. Remove the old plural path wrapper.
- Move tests, examples, docs, and native wrapper imports in the same change. Do not leave split implementation ownership across top-level and channel-level packages.
- Update `AGENTS.md`, `CLAUDE.md`, `FEATURE_LIST.md`, and active standards in the same change because the stable public architecture currently names the top-level solver packages.

Status on 2026-05-20: complete. Solver packages, shared config, native loader/specs, and shared DrJit aliases now live under `witwin/channel/`. The top-level `witwin/__init__.py` remains only as the namespace package entrypoint for `witwin.channel` and platform `witwin.core`.

## Non-goals

- Do not touch `reference/`; plan 13 owns that boundary.
- Do not change the `kernels/` subdirectory pattern. Each kernel's `.cu` / `.h` / `bind.cpp` / `native_impl.py` / `drjit_impl.py` structure is intentional.
- Do not collapse AD variants into primal modules. AD-aware variants have genuinely different control flow.
- Do not merge deterministic and Monte Carlo solver implementations; Phase 5 moves package ownership only.
- Do not block plan 13 P1 file-level work. File-internal megamodule splits and this directory-level restructure are independent.
- Do not move channel runtime internals into platform `witwin.core`.

## Validation

After each phase:

- Run the structure test for the phase.
- Run targeted solver tests affected by imports.
- Run `python -m compileall -q witwin tests/test_source_tree_phase1_structure.py`.
- Run broader `pytest tests` and examples before merging a branch that changes public imports.
- `pip install -e . --no-deps` should succeed before release integration.

Current Phase 1/2 verification targets:

- `python -m pytest tests/test_source_tree_phase1_structure.py -q`
- `python -m pytest tests/deterministic/test_no_dummy_transmitter_fallback.py -q`
- `python -m compileall -q witwin tests/test_source_tree_phase1_structure.py`

Current Phase 3 verification targets:

- `python -m pytest tests/test_source_tree_phase3_config_structure.py -q`
- `python -m pytest tests/montecarlo/test_monte_carlo_radiomap_integrators.py tests/montecarlo/test_shadow_boundary_backend.py::test_shadow_boundary_config_defaults_and_validation tests/deterministic/test_field_solver_package.py tests/integration/test_material_fallback_cleanup.py -q`
- `python -m compileall -q witwin tests/test_source_tree_phase3_config_structure.py`

Current Phase 4 verification targets:

- `python -m pytest tests/test_source_tree_phase4_channel_core_structure.py -q`
- `python -m pytest tests/test_source_tree_phase1_structure.py tests/test_source_tree_phase3_config_structure.py tests/test_source_tree_phase4_channel_core_structure.py -q`
- `python -m pytest tests/montecarlo/test_monte_carlo_radiomap_integrators.py tests/montecarlo/test_shadow_boundary_backend.py::test_shadow_boundary_config_defaults_and_validation tests/deterministic/test_field_solver_package.py tests/integration/test_material_fallback_cleanup.py -q`
- `python -m compileall -q witwin tests/test_source_tree_phase1_structure.py tests/test_source_tree_phase3_config_structure.py tests/test_source_tree_phase4_channel_core_structure.py`

Phase 5 verification targets:

- `tests/test_source_tree_phase5_channel_package_structure.py` asserts that `witwin/channel/deterministic`, `witwin/channel/montecarlo`, `witwin/channel/path`, `witwin/channel/_config`, `witwin/channel/_native`, and `witwin/channel/types.py` exist and that their old top-level directories/files do not.
- Scan `witwin/`, `tests/`, `examples/`, `docs/`, root Markdown, and `pyproject.toml` for removed top-level solver, native, config, type-alias, and plural path imports.
- Run `python -m pytest tests/test_source_tree_phase1_structure.py tests/test_source_tree_phase3_config_structure.py tests/test_source_tree_phase4_channel_core_structure.py tests/test_source_tree_phase5_channel_package_structure.py -q`.
- Run solver smoke coverage through the new canonical imports: `tests/deterministic/`, `tests/montecarlo/`, `tests/path/`, and maintained minimal examples.
- Run `python -m compileall -q witwin tests/test_source_tree_phase5_channel_package_structure.py`.
- Rebuild native targets after import-wrapper moves because native wrappers may import the moved Python packages.

## Open Questions

None for the source-tree restructure. Release versioning and downstream migration communication remain release-management work.
