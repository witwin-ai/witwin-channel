# Scattering owner

## Ownership

`scattering` owns physical scattering models and their `eval`, `sample`, and
`pdf` semantics, including resident lookup tables and phase-screen behavior. It
does not own path enumeration, solver accumulation, scene lifetime, or raw
native tuple layouts.

## Public entry points

The existing modules `energy`, `phase_screen`, and `tables` remain the current
entries. This Phase 3 boundary adds no new public re-export and moves no
implementation.

## Dependency rules

Scattering may depend on material value contracts, physics utilities, and
domain kernel facades. It must not import a solver or acquire a mutable scene or
native handle. Solvers consume scattering contracts through their owning
pipeline stage.

## Numerical and AD contract

Evaluation, sampling, PDF normalization, seed consumption, dtype/device,
aliasing, and AD behavior remain bitwise identical during architectural moves.
Expression order or tolerance changes require a separate numerical change.

## Forbidden fallback

Missing native operations must fail loudly. CPU/PyTorch reference
recomputation, zero-result substitution, and silent model downgrade are
forbidden in production paths.
