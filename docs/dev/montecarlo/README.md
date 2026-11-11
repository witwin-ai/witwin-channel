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
- Basic returns incoherent real power in `path_gain`, `component_power` and
  `component_maps`. With exactly one `ReceiverGrid` each map is `[T, cols, rows]`;
  any other layout (point receivers, several grids, grids and points mixed) gives
  `[T, R]` matrices whose columns follow `scene.receivers` (grids in row-major
  receiver order), aligned with `path_gain`. It does not claim complex path
  coefficients, polarization, or antenna-array synthesis.
- Basic reflection launches a lattice rotated by a random rotation drawn from
  `seed` (different seeds give independent estimates), folded into the half-spaces
  that can reach the scene bounds, and bounds the final reflected segment by the
  farthest deposit any receiver needs. Grids take the cell flux estimator; a point
  receiver is a capture sphere of radius `Config.receiver_radius_m` (default 1% of
  its straight transmitter distance); diffraction at a point receiver uses the same
  capture sphere on the Keller-cone samples. `Config.scattering_samples` sets the scattering
  area-sample count independently of `samples` (default: `samples`); the receiver pairs run
  natively with fixed-order per-receiver sums. Migration:
  scenes that previously raised for point, multiple or mixed receivers now solve;
  the single-grid layout is unchanged.
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
  runs before finalization, and the solve reads the failure record once
  before metadata assembly (raising `CapacityError` on a set bit).
- The former Python depth march, host Boolean breaks, active-row compaction,
  Torch restart/normalization, incident TE/TM, and wall-product expressions are
  not production backends or compatibility routes. Scattering retains only its
  independently owned event utilities.
- BDPT accepts `ad_mode="none"`, `"jvp"`, and `"vjp"` (fixed topology). It
  differentiates layer/material parameters, roughness and BSDF tables,
  frequency, and transmitter power (power and coherent combine domains). It
  refuses transmitter/receiver position and mesh-vertex leaves before launch;
  `capabilities()` records exactly this scope.

## BDPT sampler and accumulation

- Reflection, diffraction, straight transmission and the coupled classes come
  from one enumerated-oracle call; each row's component id is its bucket.
- Mixed reflection+transmission chains are sampled by the fused wavefront
  (`bdpt_wavefront_transmission`): all transmitters in one wavefront, chunks of
  `2^18` samples, per bounce one RayD intersection and one native bounce step
  (event selection, only the selected branch, in-place update, device tallies),
  and per receiver tile of at most `2^20` pairs a visibility query and a fused
  connect-and-accumulate kernel that never stores a connection row. AD, rough
  (Kirchhoff) faces, `export_paths` and `diagnostics` use the event-composed
  sampler instead; both draw the same native counter-based random numbers
  (a pure function of seed, stream, global sample index and depth), so they
  give the same estimate. `metadata["sampler"]` names the one used.
- Metadata counts stay on the device and are read in one copy per solve.
- With only enumerated components (the default `los`, `reflection`,
  `diffraction`, and coupled paths) every row is a unit-mass discrete
  connection: the estimate does not depend on `samples` or `seed`, and `mis` /
  `power_heuristic_beta` do not change it (each connection has one strategy, so
  every MIS weight is 1). Only sampled transmission and scattering use samples.
- `accumulation_strategy="auto"` picks `atomic` (or `compact` for sparse grid
  estimates), never `staged`. `staged` and the accumulate JVP are a stable key
  sort plus a fixed-order reduce-by-key per `(tx, rx, component)`: bitwise
  reproducible run to run.
- Receivers: one `ReceiverGrid` keeps the grid layout (`component_maps`);
  several grids, or grids mixed with point receivers, are solved as one flat
  receiver list in scene order (`path_gain[tx, receiver]`), like the
  deterministic solver.
- Design record: ADR-049 (`docs/dev/standards/adr-049-m4-1-bdpt-wavefront.md`).

### BDPT wavefront evidence

Measured 2026-11-10 on the shared RTX GPU 1 under a slot lock while other jobs
held it at 94-100 % utilization (contended; counts are the primary evidence).
Scene: lossy front wall, PEC back wall, 1 transmitter, 4 point receivers,
`components={"transmission"}`, `max_depth=2`. Sampler = the mixed-chain sampler
call alone; peak = `max_memory_allocated` above the pre-call baseline.

| samples | sampler | kernel launches | host syncs | kernel busy | wall (min of 4) | peak |
|---:|---|---:|---:|---:|---:|---:|
| 4,096 | event-composed | 257 | 4 | 0.50 ms | 9.15 ms | 3.7 MiB |
| 65,536 | event-composed | 257 | 4 | 0.83 ms | 12.10 ms | 59.4 MiB |
| 1,048,576 | event-composed | 257 | 4 | 6.01 ms | 13.91 ms | 951.0 MiB |
| 4,096 | wavefront | 11 | 0 | 0.08 ms | 1.56 ms | 1.0 MiB |
| 65,536 | wavefront | 11 | 0 | 0.17 ms | 1.87 ms | 15.4 MiB |
| 1,048,576 | wavefront | 32 | 0 | 1.70 ms | 7.75 ms | 61.5 MiB |

Whole solve, 1 M samples, 4 receivers, depth 2: peak 61.5 MiB with the
wavefront versus 951.0 MiB through the event-composed sampler, for both
`{"transmission"}` and `{"los", "reflection", "diffraction", "transmission"}`.

Equivalence: for the same seed the wavefront and the event-composed sampler
return identical `path_gain` and event counts. Against the previous
`torch.Generator` event uniforms (reproduced in-process), the per-receiver
means over 32 seeds at 65,536 samples differ by |z| <= 0.78 (depth 2) and
<= 0.69 (depth 3); the sampled mixed share is about 0.7-0.8 % of the
transmission power here, the rest being the enumerated straight chains,
identical in both.

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
- The completed wrapper migration ledger is immutable archived in Git history
  (`docs/dev/archive.md`). Native binding changes
  regenerate `ci/native-binding-manifest.json` with `python ci/check_native_bindings.py --write`.
- New dependencies must pass the import-graph checker; architecture debt is not
  extended by editing the allowlist.
- Keep solver capability metadata, contract tests, and this README synchronized
  with changes to component semantics, AD support, numerical measures, or
  fail-loud behavior.
