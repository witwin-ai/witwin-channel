# gsplat Architecture Lessons

Status: Active
Category: Standard
Last reviewed: 2026-05-20

## Purpose

This document records architecture lessons from `nerfstudio-project/gsplat` for
use in `witwin.channel` design work. It is not a compatibility target and does
not override repository rules. The stable public architecture for this package
remains `Scene + Tracer + Result`, with DrJit-native runtime internals.

Reference snapshot:

- Repository: `https://github.com/nerfstudio-project/gsplat`
- Reviewed commit: `3d4f9027f36d70ac8e89536e14c247de377f74b5`
- Review date: 2026-04-16

## High-Level Philosophy

`gsplat` feels simple because it keeps the public surface narrow while allowing
the implementation boundary to be specialized and deep. The core user workflow is
not driven by a large global config object. Instead, the user passes explicit
tensors and execution flags into a small number of public functions, and receives
rendered tensors plus an explicit metadata dictionary.

The important design pattern is:

```text
public API
  -> pipeline orchestration
      -> backend wrapper and autograd boundary
          -> registered native operations
              -> native kernels
```

The library concentrates complexity at the backend boundary and keeps state
ownership explicit. Training parameters, optimizer state, strategy state, and
rendering metadata are separate concepts with separate lifetimes.

## What To Borrow

### Narrow Public Entrypoints

Prefer a small number of stable public calls over many convenience wrappers. In
`witwin.channel`, this means `Scene`, `Tracer`, and `Result` should remain the
primary public vocabulary. New features should extend those concepts through
clear inputs, result fields, or strategy objects rather than introducing parallel
top-level APIs.

Good direction:

```text
scene = Scene(...)
result = Tracer(...).trace(scene, ...)
```

Avoid turning every internal solver variant into a new public constructor or a
new public import path.

### Explicit Data Contracts

`gsplat` documents tensor shapes at function boundaries. Borrow the habit even
though this repository uses DrJit structures and core geometry objects rather
than raw tensor-only APIs.

For `witwin.channel`, public and package-internal boundaries should make these
contracts obvious:

- which structures are already device-placed
- which arrays are DrJit-native
- which dimensions represent paths, receivers, wedges, cells, or interactions
- which fields are differentiable
- which metadata is diagnostic only

Do not hide these contracts behind duck-typed dictionaries or implicit config
state.

### Explicit State Lifetimes

Borrow the separation between:

- model or scene state owned by the caller
- execution state owned by the tracer invocation
- strategy state owned by an optional strategy object
- diagnostic metadata owned by the result
- backend build or capability state owned by the backend layer

For this repository, a useful mapping is:

```text
Scene          long-lived public scene description
Tracer         execution policy and backend selection
Result         path fields, monitor outputs, diagnostics, and metadata
Strategy       optional refinement or sampling policy state
Backend layer  native capability checks and compiled operation boundaries
```

Avoid a central mutable runtime object that gradually accumulates all of these
responsibilities.

### Parameter-Selected Execution Paths

`gsplat` manages many execution paths through explicit parameters such as packed
mode, distributed mode, render mode, camera model, and backend feature flags. The
equivalent in this codebase should be explicit tracer or monitor options with
validated meanings.

Good direction:

- deterministic versus Monte Carlo tracing is explicit
- coherent versus incoherent accumulation is explicit
- monitor type and sampling grid are explicit
- native-kernel execution intent is explicit
- unsupported combinations fail early

Avoid hidden execution-path selection based on ambient globals, file-level
settings, or config defaults that are hard to inspect from the call site.

### Backend Boundary Concentration

`gsplat` has a clear Python-to-native boundary. Borrow that structure, but adapt
it to this repository's DrJit and native CUDA rules.

The backend boundary should:

- own native capability checks
- own conversion into native call signatures
- keep hot paths DrJit-native
- avoid leaking native implementation details into public scene APIs
- avoid spreading kernel-launch glue across unrelated modules

For the channel subproject, scene code should not know about solver execution
details, and each solver package should own its runtime, native bindings, and
result construction. Keep the existing layering rule:

```text
witwin/channel/core/ -> witwin/channel/core/scene/ -> {deterministic, montecarlo, path}/
```

### Strategy Objects For Mutable Algorithms

`gsplat` keeps densification and pruning as strategies instead of embedding that
logic in rasterization. Borrow this for features that mutate or refine execution
state across steps.

Candidate strategy-like concepts in this codebase include:

- Monte Carlo sampling policy
- receiver tiling policy
- path refinement policy
- validation or replay policy
- monitor accumulation policy

A strategy should have a small explicit state object or dataclass. It should not
become a hidden global registry.

### Metadata As A First-Class Result

`gsplat` returns `meta` for intermediate information consumed by training
strategies and tests. Borrow this through `Result` diagnostics and metadata.

Good metadata includes:

- selected execution path
- backend capability decisions
- path counts and interaction counts
- native kernel launch family
- monitor sampling shape
- culling or validity masks when useful for debugging

Metadata should be explicit and testable. It should not be required to reconstruct
primary physics outputs.

### Reference Implementations For Numerical Boundaries

`gsplat` keeps PyTorch reference implementations for validating CUDA kernels. The
same principle is useful here, but the implementation must respect repository
rules: runtime internals remain DrJit-native, and Torch/NumPy bridges are not
introduced into solver hot paths.

Good validation references for this repository can be:

- small DrJit-native reference paths
- closed-form analytical fixtures
- Sionna-aligned comparison tests where already established
- native versus baseline parity tests at stable public boundaries

## What To Avoid

### Avoid A Large Central `config.py`

Do not introduce a repository-wide configuration object that controls solver,
scene, monitor, backend, feature, and validation behavior at once. This creates
hidden control flow and makes test cases harder to reason about.

Prefer:

- explicit constructor arguments
- narrow dataclasses for a single responsibility
- result metadata that records decisions
- early validation of unsupported combinations

### Avoid Global Registries For Normal Execution

Do not add a registry or factory system unless it removes real duplication and
has a concrete owner. A registry is not a substitute for a stable public API.

Avoid patterns like:

```text
string name -> global registry -> factory -> wrapper -> backend branch
```

when a direct typed call or local dispatch table would be clearer.

### Avoid Parallel Compatibility APIs

`gsplat` has a few compatibility wrappers, but this repository explicitly avoids
preserving legacy code unless requested. Do not add old-style constructors,
fallback paths, or duplicate import surfaces to make migration feel easier.

In particular:

- do not add new `rfdt` import paths
- do not add raw `vertices/faces` public scene constructors
- do not add `Scene.from_meshes(...)` compatibility helpers
- do not keep duplicate implementations in parallel files

### Avoid CPU Fallback Paths In Core Computation

`gsplat` is Torch/CUDA-centered. This repository is DrJit-native and GPU-first.
Do not copy patterns that move data through CPU tensors for convenience in solver
internals or native-kernel paths.

Avoid:

- NumPy bridges in hot paths
- Torch bridges in solver internals
- DLPack bridges in native-kernel paths
- CPU fallback implementations for core computation unless explicitly requested

### Avoid Hidden Shape Or Layout Normalization

Some convenience normalization is useful at public boundaries, but internal
solver modules should not silently accept many unrelated layouts. Normalize once
at the boundary, then keep a strict internal representation.

This protects:

- symbolic DrJit behavior
- native kernel assumptions
- gradient correctness
- test reproducibility

### Avoid Wrapper Stacks That Add No Contract

Borrow `gsplat`'s boundary clarity, not its exact file sizes or wrapper count.
Every wrapper in this repository should add one of:

- API stability
- validation
- backend isolation
- lifecycle ownership
- a tested semantic conversion

Delete wrappers that only rename arguments, forward calls, or preserve an old
module shape.

### Avoid Strategy Objects That Own The Whole Pipeline

Strategies should mutate or select one focused part of execution. They should
not own scene construction, tracing, monitor scheduling, kernel selection, and
result formatting at the same time.

If a strategy needs many unrelated fields, split the ownership before adding more
configuration.

## Practical Checklist

Before adding a new architecture component, ask:

- Is this part of the stable `Scene + Tracer + Result` public surface?
- If not, which package layer owns it?
- Is its state long-lived, per-trace, per-strategy, or diagnostic?
- Can the execution path be read from explicit arguments?
- Does the backend boundary stay concentrated?
- Does the implementation remain DrJit-native in runtime internals?
- Does this add a real contract, or only another wrapper?
- Is there a focused validation path for the numerical behavior?

## Recommended Pattern For Future Work

Use this shape for new execution features:

```text
1. Add or extend a narrow public option on Scene, Tracer, or monitor API.
2. Validate the option at the public or package boundary.
3. Normalize into one internal DrJit-native representation.
4. Route through one backend boundary module.
5. Return primary outputs plus explicit Result metadata.
6. Add focused parity or analytical validation tests.
7. Document the public workflow and update FEATURE_LIST.md only when the change
   is user-visible.
```

This keeps the useful part of the `gsplat` design philosophy: simple calls,
explicit state, concentrated backend complexity, and testable execution paths.
