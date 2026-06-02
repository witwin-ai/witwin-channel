# Channel Package Architecture Audit

Status: Active
Category: Optimization
Last reviewed: 2026-05-15

## Scope

This report audits the architecture, API shape, code elegance, and maintainability
of these packages:

- `witwin.channel.core.scene`
- `witwin.channel.core`
- `witwin.channel.deterministic`
- `witwin.channel.montecarlo`

The review uses the repository rules in `AGENTS.md`, especially the stable public
architecture target of `Scene + Tracer + Result`, DrJit-native runtime internals,
typed cross-module payloads, compact public APIs, and avoidance of thin wrappers,
duplicate helpers, and loose dict contracts.

## Summary

The public API direction is partly sound: `witwin.channel.montecarlo` exposes a small
package root, `witwin.channel.deterministic` has a clear `solve(...)` entrypoint, and
`channel_utils` centralizes shared math. The internal architecture is much weaker.
The worst issues are concentrated in Monte Carlo BDPT diffraction orchestration,
Monte Carlo metadata assembly, deterministic diffraction math/builders, and native
kernel wrapper boundaries.

Overall ratings:

| Package | Rating | Notes |
| --- | --- | --- |
| `channel_utils` | 6.5/10 | Correct ownership, weak typing and dict-shaped Jones/vector payloads |
| `channel_scene` | 5/10 | Good public direction, too much runtime/cache/material logic inside `Scene` |
| `deterministic` | 3/10 | Heavy procedural diffraction and kernel-wrapper debt |
| `montecarlo` | 3/10 | Small public API, but BDPT/diffraction/metadata internals are the main mess |

## Evidence

Static AST scanning found multiple files with extreme size and function shape:

- `witwin/channel/montecarlo/integrators/bdpt_diffraction.py`: 2085 lines.
- `witwin/channel/montecarlo/config.py`: 990 lines.
- `witwin/channel/montecarlo/integrators/basic.py`: 1078 lines.
- `witwin/channel/deterministic/path/diffraction_impl/math.py`: 1168 lines.
- `witwin/channel/deterministic/path/diffraction_impl/builders.py`: 699 lines.
- `witwin/channel/deterministic/kernels/radio_map_accumulate/native_impl.py`: 1587 lines.

The targeted ruff structural scan reported 479 findings with:

- missing public return annotations
- high cyclomatic complexity
- oversized functions
- long argument lists
- boolean-trap parameters
- unused imports
- unresolved type annotations

Command used:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m ruff check witwin\channel_scene witwin\channel_utils witwin\deterministic witwin\montecarlo --select F401,F821,ANN201,C901,PLR0913,PLR0915,FBT001 --output-format concise
```

## Severe Issues

### `witwin/channel/montecarlo/integrators/bdpt_diffraction.py`

Severity: Critical.

This is the clearest mess. The file combines MIS sampling, edge sampling,
direct/Keller/suffix strategies, tape storage, visibility checks, contribution
accumulation, result counters, and metadata support. Several functions are
150-260 lines with 15-21 parameters. The class is not an owner of one concept;
it is a procedural bucket for nearly all BDPT diffraction mechanics.

Recommended direction:

- Split strategy execution into typed strategy runners.
- Replace dict sample payloads with small dataclasses or slots classes.
- Move contribution storage and tape storage behind explicit owner objects.
- Keep `BDPTDiffractionMIS.trace(...)` as a thin orchestrator only after the
  owned components exist.

### `witwin/channel/montecarlo/integrators/bdpt.py`

Severity: Critical.

`BDPT.integrate(...)` mixes AD dispatch, setup, LoS, reflection, BDPT diffraction,
shadow-boundary correction, metadata construction, primal-state creation, and
result assembly. It has 15 parameters, high complexity, and a large inline
metadata block. It should orchestrate owned phases, not build every payload itself.

Recommended direction:

- Introduce typed metadata input objects.
- Extract BDPT-specific metadata augmentation.
- Extract primal-state assembly.
- Keep `integrate(...)` as phase ordering.

### `witwin/channel/montecarlo/integrators/metadata.py`

Severity: Critical.

`build_metadata(...)` takes 29 parameters and returns a large nested dict. The
schema mixes stable contract fields, diagnostics, backend identity, debug strings,
runtime reuse, and explanatory prose. This violates the repository rule that
cross-module payloads should be explicit typed structures and metadata should
stay compact and non-duplicative.

Recommended direction:

- Replace the 29-argument function with a `MetadataInput` dataclass.
- Keep metadata assembly focused on stable semantic facts.
- Move integrator-specific metadata additions back to integrator-owned helpers.

### `witwin/channel/deterministic/path/diffraction_impl/math.py`

Severity: Critical.

This file is a large mixed math/geometry/polarization module. It duplicates UTD,
Fresnel, cotangent, Jones operator, and geometry logic that already exists in
`channel_utils` and Monte Carlo diffraction code. Formatting suggests migration
or generated-code residue. It is hard to review and hard to validate.

Recommended direction:

- Move shared UTD/Fresnel helpers to `channel_utils.wave_math`.
- Keep deterministic-specific AD derivative math isolated.
- Replace dict-shaped Jones operators with typed helper payloads when feasible.

### `witwin/channel/deterministic/path/diffraction_impl/builders.py`

Severity: Critical.

`prepare(...)` is a procedural state-construction pipeline with a long signature
and many responsibilities: edge data lookup, backend choice, pruning policy,
lineage state, first-order state, higher-order state, and inserted-reflection
state. This should be an owning builder object with explicit state.

Recommended direction:

- Introduce a `DiffractionStateBuilder` or equivalent owner.
- Move budget/pruning policy into a typed policy object.
- Return a typed preparation result instead of tuple/dict payloads.

### `witwin/channel/deterministic/path/diffraction_impl/shadow_boundary_correction.py`

Severity: High.

Deterministic code imports `witwin.channel.montecarlo.kernels.shadow_boundary`. This
creates an ownership inversion: deterministic depends on Monte Carlo kernel
packaging for a deterministic correction path. Shared native kernels should live
under a shared kernel package or deterministic should own its deterministic
backend explicitly.

Recommended direction:

- Move shared shadow-boundary kernel ownership out of `witwin.channel.montecarlo`.
- Keep solver packages depending on shared infrastructure, not on each other.

### `witwin/channel/deterministic/kernels/radio_map_accumulate/native_impl.py`

Severity: High.

The native wrapper file is too large and contains many long signatures. It mixes
reference paths, native launch code, AD-safe behavior, and shadow-boundary support.
The Python boundary is not a clean kernel API.

Recommended direction:

- Split by kernel family and payload type.
- Introduce typed launch parameter objects.
- Keep reference implementations separate from native launch wrappers.

## Medium Issues

### `witwin/channel/core/scene/scene.py`

`Scene` is directionally right as the public object, but it owns too much runtime
state. Edge runtime, triangle runtime, material runtime, RayD bridge, cache
invalidation, and public scene mutation all live in one class. Runtime payloads
are mostly dicts and `SimpleNamespace`.

Recommended direction:

- Extract typed edge runtime and material runtime payloads.
- Keep public scene mutation methods small.
- Replace `get_*` runtime APIs with properties or typed domain methods where
  they are public.

### `witwin/channel/core/scene/sionna_adaptor.py`

The adapter is a valid boundary, but signatures are too long and ruff reports
unresolved `Scene` annotations. It also mutates `sys.path`, which should remain
strictly adapter-local and documented as a boundary behavior.

### `witwin/channel/montecarlo/config.py` and `witwin/channel/deterministic/config.py`

The packages duplicate guardrail and trace-control logic. Solver-specific fields
should remain separate, but execution intent, guardrail changes, and common
normalization patterns are strong candidates for shared typed policy helpers.

### `witwin/channel/core/polarization.py`

The module belongs in `channel_utils`, but it exposes many untyped functions and
dict-shaped Jones/vector contracts. This is acceptable for pure math in the short
term, but it blocks stronger static validation.

## Priority Refactor Plan

1. Refactor `witwin.channel.montecarlo.integrators.metadata` to use a typed input object.
2. Extract BDPT-specific metadata augmentation out of `BDPT.integrate(...)`.
3. Split the worst BDPT diffraction strategy loops into focused strategy helpers.
4. Move duplicated UTD/Fresnel math toward `channel_utils.wave_math`.
5. Break deterministic state construction out of `builders.prepare(...)` into an
   owning builder object.
6. Move shared shadow-boundary kernel ownership out of `witwin.channel.montecarlo`.

## Validation Expectations

Each refactor step should preserve behavior. Preferred validation order:

1. Add or update a focused regression test before production edits.
2. Run the targeted test and confirm the expected failure or baseline guard.
3. Implement the smallest behavior-preserving refactor.
4. Run the targeted test.
5. Run targeted ruff checks for the touched files.
6. Run relevant integration tests when crossing solver/package boundaries.
