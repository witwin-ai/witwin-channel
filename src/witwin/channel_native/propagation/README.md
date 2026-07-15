# Propagation owner contract

## Ownership

This package owns row-aligned topology, continuous geometry, RF fields, and
the enumerated propagation stages shared by Path and Deterministic solvers.
`EvaluatedPaths` is the internal composition boundary; solver accumulation and
public result types remain solver-owned.

## Public entry points

There are no top-level public API exports. Internal producers and consumers use
`PathTopology`, `PathGeometry`, `PathFields`, and `EvaluatedPaths`. During the
migration, `propagation.enumerated` owns the existing shared scattering-path
stage while the old deterministic import remains a compatibility re-export.

## Dependency rules

Topology cannot depend on continuous fields or solver policy. Geometry cannot
choose winners. Fields cannot discover topology. Enumerated propagation serves
only Path and Deterministic; Monte Carlo sampling, MIS, solver results, and
deterministic accumulation remain outside this package.

Raw native tuples may exist only inside domain `kernels` modules. A kernel
façade must validate and convert them to a named internal contract before any
solver or propagation pipeline can observe the result.

## Numerical and AD contract

Contract construction is metadata-only and zero-copy. Row order, row identity,
tensor object/storage aliasing, stride, dtype, device, and `requires_grad` must
remain exact. Fixed-winner geometry reevaluation cannot change discrete winner
selection or conceal a detach boundary.

## Forbidden fallback

Propagation code must fail loudly when a required native capability or contract
is missing. It cannot recompute geometry on CPU/Torch, load a global extension,
return a zero result, or silently switch algorithms.
