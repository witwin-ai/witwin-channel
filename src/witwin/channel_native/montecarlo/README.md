# Monte Carlo domain

This package contains the two Monte Carlo solver families and the event
contracts they share. It is not itself a solver API: callers select either
`witwin.channel_native.montecarlo.basic` or
`witwin.channel_native.montecarlo.bdpt`.

## Ownership

- `basic/` owns the incoherent Monte Carlo power-map solver, including its
  configuration, result type, orchestration, sampling, metadata, and
  solver-local kernel facades.
  Its dormant ADR-027 wall-product primal/VJP/JVP facade consumes fixed RayD
  penetration storage and a solve-owned failure state; it has no live solver
  caller until the MonteCarloTargetInset atomic switch/delete commit.
  The fixed-capacity product checks validity before every hit, uses only the
  canonical valid prefix bounded by the device `num_hits`, multiplies walls in
  ascending slot order, and never treats poisoned tail storage as an event.
- `bdpt/` owns bidirectional path tracing, including subpaths, connections,
  MIS, optional path-sample export, accumulation, and its solver-local kernel
  facades. Its package loads `solve` lazily to keep the public import light.
- `events/` (`transmission.py`, `scattering.py`) owns only event rules
  genuinely shared by both solvers. It must not become a second solver
  pipeline.
- Scene compilation, material encoding, propagation primitives, and native
  symbol loading remain owned by their respective top-level domains.

## Public entry points

The stable, snapshotted entry points are:

- `witwin.channel_native.montecarlo.basic.Config`
- `witwin.channel_native.montecarlo.basic.Result`
- `witwin.channel_native.montecarlo.basic.solve(scene, config)`
- `witwin.channel_native.montecarlo.bdpt.BDPTPathSamples`
- `witwin.channel_native.montecarlo.bdpt.Config`
- `witwin.channel_native.montecarlo.bdpt.Result`
- `witwin.channel_native.montecarlo.bdpt.solve(scene, config)`

`pipeline.py`, `backend.py`, `endpoints.py`, `connections.py`,
`accumulation.py`, `subpaths.py`, `mis.py`, `sampling.py`, and each `kernels/`
package are internal entry points. Their
names may be used by focused contract tests, but are not compatibility
promises. The parent `montecarlo` package exports no solver symbols.

## Dependency rules

- Basic and BDPT are sibling owners. Neither solver may import the other or
  any other solver package.
- Basic must not call `witwin.channel_native.propagation.enumerated`. Under the
  narrow ADR-008 exception, `montecarlo.bdpt.pipeline` may call only the public
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
- The future `MonteCarloTargetInset` traversal is one flattened RayD batch,
  not one trace per transmitter. Phase P exposes its dormant typed geometry
  and estimator contracts only; the current live solver is unchanged until
  the dedicated switch/delete commit.
- BDPT supports only `ad_mode="none"`. Any other mode is rejected by
  `Config`; detached or zero gradients are not a substitute.

## Forbidden fallback

Missing CUDA, the channel-native extension, required native symbols, or RayD
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
