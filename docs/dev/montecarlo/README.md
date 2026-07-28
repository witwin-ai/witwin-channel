# Monte Carlo domain

This package contains the two Monte Carlo solver families. It is not itself a
solver API: callers select either `witwin.channel.montecarlo.basic` or
`witwin.channel.montecarlo.bdpt`.

## Ownership

- `basic.py` owns the incoherent Monte Carlo power-map solver, including its
  configuration, result type, orchestration, sampling, and metadata. It is one
  module: the former package's `__init__` re-export facade is gone with the
  package that held it, and its native facades live in
  `witwin.channel.kernels`.
  Its live ADR-027 wall-product primal/VJP/JVP facade consumes fixed RayD
  penetration storage and the solve-owned failure state after one flattened
  `MonteCarloTargetInset` traversal batch.
  The fixed-capacity product checks validity before every hit, uses only the
  canonical valid prefix bounded by the device `num_hits`, multiplies walls in
  ascending slot order, and never treats poisoned tail storage as an event.
  Basic metadata exposes only the host-known `contribution_capacity`; it does
  not perform a device nonzero-count read or publish a host actual-count key.
  That capacity label is not an actual visible-contribution count and must not
  be used to claim an improved capacity/active ratio. Changing it back to an
  actual-count contract, or removing the pre-existing carrier/edge-length host
  scalar reads, is measurement-required optimization debt: compare E2E
  latency, peak memory, steady throughput, synchronization cost, and result
  exactness before changing production semantics.
- `bdpt.py` owns bidirectional path tracing, including subpaths, connections,
  MIS, optional path-sample export, accumulation, and the per-solve workspace.
  It is one module: the former package's lazy `solve` re-export is gone with
  the `__init__` that held it, and its native facades live in
  `witwin.channel.kernels`.
- The event rules both solvers share are NOT owned here. The former `events/`
  package held specular-transmission and Kirchhoff scattering event physics
  that an enumerated caller consumed as well, so it was never a Monte Carlo
  concept; it now lives with its concepts, in
  `witwin.channel.interactions.transmission` and
  `witwin.channel.interactions.scattering`. Both solvers import those helpers
  read-only. Neither that package nor this one may grow a second solver
  pipeline.
- Scene compilation, material encoding, propagation primitives, and native
  symbol loading remain owned by their respective top-level domains.

## Public entry points

The stable, snapshotted entry points are:

- `witwin.channel.montecarlo.basic.Config`
- `witwin.channel.montecarlo.basic.Result`
- `witwin.channel.montecarlo.basic.solve(scene, config)`
- `witwin.channel.montecarlo.bdpt.BDPTPathSamples`
- `witwin.channel.montecarlo.bdpt.Config`
- `witwin.channel.montecarlo.bdpt.Result`
- `witwin.channel.montecarlo.bdpt.solve(scene, config)`

Everything else each solver defines is internal. `basic.py` and `bdpt.py` are
each one module now, so the former `pipeline.py`, `backend.py`, `endpoints.py`,
`connections.py`, `accumulation.py`, `subpaths.py`, `mis.py`, `sampling.py` and
per-solver `kernels/` packages no longer exist as import paths; the names they
defined are still there, as members of the collapsed module. Those names may be
used by focused contract tests, but are not compatibility promises. The parent
`montecarlo` package exports no solver symbols.

## Dependency rules

- Basic and BDPT are sibling owners. Neither solver may import the other or
  any other solver package.
- Basic must not call `witwin.channel.propagation.enumerated`. Under the
  narrow ADR-008 exception, `montecarlo.bdpt` may call only the public
  `evaluate_enumerated_paths` entry read-only as an opaque discrete-path oracle;
  it must not import enumerated internals or add BDPT policy to that engine.
- Orchestration may use public scene, material, geometry, topology, and field
  contracts. Solver-local kernel facades obtain required native symbols
  through the runtime symbol layer; pipelines must not import the raw
  extension or `core.kernels.ops`.
- A domain kernel must not depend on a solver or another domain's private
  kernel package. Cross-domain behavior belongs behind a public domain
  contract.
- Cross-domain imports are absolute. New edges must satisfy
  `ci/check_import_graph.py`; do not add an allowlist entry to establish a new
  architecture.

## Numerical and AD contract

- Both solvers require CUDA. `samples` is positive, `seed` is non-negative,
  and the selected components are drawn from `los`, `reflection`,
  `diffraction`, `transmission`, and `scattering`.
- Basic returns incoherent real power in `path_gain`, `component_power`, and,
  for receiver grids, `component_maps`. It does not claim complex path
  coefficients, polarization, or antenna-array synthesis.
- BDPT carries complex3/Jones fields for coherent events, but scalar throughput
  is only a sampling-probability proxy. Its public result reports power;
  `BDPTPathSamples` exposes topology, contribution, proposal PDF, MIS weight,
  endpoint IDs, depths, and path length when export is requested.
- BDPT proposal PDFs exclude geometry Jacobians. Endpoint connection is one
  strategy; diffraction uses direct and Keller strategies. The sensor subpath
  is the receiver endpoint, so its depth is zero.
- Shared rough-surface scattering is an ensemble-average, power-only
  Kirchhoff event. The stored phase is a placeholder used only through
  `|field|^2`; v1 scattering is single-bounce and terminates after receiver
  next-event estimation. Reflect/transmit/scatter event power is divided by
  the selected event probability to keep the estimator unbiased.
- SI units are used: positions and path lengths are metres, frequency is hertz,
  and the vacuum light speed is `299792458 m/s`.

### AD contract

- Basic accepts `ad_mode="none"`, `"jvp"`, and `"vjp"`. JVP and VJP
  differentiate the fixed sampled topology through native companion kernels;
  they do not estimate visibility transitions, path birth/death, or other
  discrete winner changes.
- Basic AD covers supported material leaves, frequency, endpoint positions,
  and geometry inputs. It rejects scattering before launch. Reflection AD also
  rejects depths above the limit reported by the native companion kernel, and
  frequency AD rejects material models outside its supported contract.
- The MC straight-penetration estimator differentiates base power, resident
  direction/normal geometry, CSR layer thickness/epsilon/conductivity, and
  frequency. Discrete validity/IDs, geometry mode, permeability, and transmitter
  polarization remain fixed. Wall products and shared-parameter reductions use
  deterministic ascending orders without floating-point atomics.
- The live `MonteCarloTargetInset` traversal is one flattened pair-major RayD
  batch, not one trace per transmitter. It preserves transmitter-major,
  receiver-minor order and routes the resident fixed-capacity hit block through
  the Channel wall-product primal/VJP/JVP family. Penetration and the
  estimator share one `CapacityFailureState`; failed work is sanitized before
  the Basic-owned native five-component-map primal/VJP/JVP sanitizer family
  runs before finalization, and the solve enqueues the unique terminal observer
  once after final result assembly.
- The former Python depth march, host Boolean breaks, active-row compaction,
  Torch restart/normalization, incident TE/TM, and wall-product expressions are
  not production backends or compatibility routes. Scattering retains only its
  independently owned event utilities.
- BDPT supports only `ad_mode="none"`. Any other mode is rejected by
  `Config`; detached or zero gradients are not a substitute.

## Forbidden fallback

Missing CUDA, the channel extension, required native symbols, or RayD
capability is a hard error. Do not add CPU, Python/Torch recomputation,
finite-difference, legacy RayD/DrJit, zero-result, or reference-oracle
fallbacks. A less capable accumulation strategy is valid only when it is an
implemented native strategy with the same contract, not an alternate compute
backend.

## Maintenance

- Adding, removing, renaming, or moving a stable basic/BDPT export requires an
  intentional update to `ci/public-api-snapshot.json` and a migration note.
- The completed wrapper migration ledger is immutable historical evidence at
  `docs/dev/audit/phase12-ops-migration-ledger.json`. Native binding changes
  must update the relevant binding/audit manifest.
- New dependencies must pass the import-graph checker; architecture debt is not
  extended by editing the allowlist.
- Keep solver capability metadata, contract tests, and this README synchronized
  with changes to component semantics, AD support, numerical measures, or
  fail-loud behavior.
