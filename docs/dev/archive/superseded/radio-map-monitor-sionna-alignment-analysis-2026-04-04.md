# RadioMapMonitor Sionna 2.0.0 Alignment Analysis

Status: Active
Category: Optimization
Last reviewed: 2026-04-04

## Scope

This note compares the current `RadioMapMonitor` implementation against the
Sionna RT 2.0.0 radio-map design and records the concrete performance and
correctness gaps that block the next production phase.

Primary external references:

- Sionna RT technical report, Section 4: <https://nvlabs.github.io/sionna/rt/tech-report/S4.html>
- Sionna RT radio-map solver source docs: <https://nvlabs.github.io/sionna/_modules/sionna/rt/radio_map_solvers/radio_map_solver.html>

Local reference snapshot used for code-level comparison:

- `sionna-rt-reference-2.0.0/src/sionna/rt/radio_map_solvers/radio_map_solver.py`
- `sionna-rt-reference-2.0.0/src/sionna/rt/radio_map_solvers/planar_radio_map.py`
- `sionna-rt-reference-2.0.0/src/sionna/rt/radio_map_solvers/radio_map.py`

## Sionna 2.0.0 Design Points That Matter Here

The relevant Sionna behavior is not just "it has a radio map." The important
parts are the mathematical contract and the runtime structure:

1. Radio maps are defined as cell averages of path gain over the measurement
   cell, not as a point sample at the cell center.
2. The default radio-map metric is non-coherent. The technical report defines
   the integrand as the squared norm of the electric field, which corresponds
   to an isotropic receiver matched to the incident polarization.
3. Non-diffraction radio maps are computed with a single SBR loop and direct
   in-loop cell accumulation.
4. Diffraction radio maps are computed separately, again with direct
   accumulation into cells rather than exporting path records and reducing them
   later.
5. Compute cost is designed to stay effectively independent of the number of
   measurement cells. Storage still scales with the map resolution, but the
   solver work is sample-driven rather than cell-driven.
6. Sionna distinguishes the radio-map solver from coherent per-target path
   solving. The coherent fast-fading view is treated as a separate
   center-sample path-solver product rather than the default radio-map
   semantics.

The code reflects this directly:

- `RadioMapSolver` updates `radio_map.add_paths(...)` inline during traversal.
- `PlanarRadioMap.add_paths(...)` computes the contribution of a hit and
  `dr.scatter_reduce(...)` adds it straight into the cell buffer.

## Current Repository Behavior

The current implementation already has useful pieces:

1. `RadioMapMonitor` is a first-class monitor and result kind.
2. Axis-aligned coherent radio maps can reuse the native reflection and
   diffraction accumulation stack through `native_grid.py`.
3. Baseline incoherent and coherent reducers exist and are differentiable.
4. Result tooling for RSS, SINR, association, and sample-position extraction is
   already in place.
5. Axis-aligned incoherent radio maps now also expose an explicit
   `accumulation_backend="native_incoherent"` path that reuses reflection-family
   replay, receiver tiling, and reduced diffraction-state layouts to accumulate
   scalar power without exporting raw radio-map path payloads.

However, the current runtime splits into two very different paths:

### Baseline path

- `witwin/channel/monitors/radio_map/trace_radio_map.py`
- `collect_los_paths(...)`
- `collect_reflection_paths(...)`
- `collect_diffraction_state_paths(...)`

This path exports raw per-path data and then reduces it back onto the radio-map
grid with scatter-reduce.

### Native coherent path

- `accumulate_radio_map_los_coherent(...)`
- `accumulate_radio_map_reflection_coherent(...)`
- `accumulate_radio_map_diffraction_coherent(...)`
- `trace/diffraction/suffix.py`

This path still reuses the shared reflection and diffraction kernel
infrastructure, but radio-map orchestration now owns its coherent wrappers and
routes reflected-suffix replay through a monitor-neutral suffix module instead
of calling the field-only accumulation adapter directly.

## Performance Findings

### What Sionna optimizes that the current baseline does not

Sionna avoids path export for production radio maps. The current baseline
incoherent path does not. This is the main reason large radio maps can be
slower than `FieldMonitor` today.

The critical difference is:

- Sionna: sample-driven direct cell accumulation
- current baseline radiomap: target-driven path materialization followed by
  reduction

For dense maps this is a structural loss, not a tuning issue.

### Path-count explosion on dense baseline radio maps

A reproduced `512 x 512`, `quadrature_mode="2x2"` radiomap run produced:

- `los`: `604,913`
- `reflection`: `677,965`
- `diffraction`: `57,788,116`
- `total`: `59,070,994`

That workload exports almost 59.1 million path records before reduction. This
explains both the poor runtime and the OOM tendency.

### Why this can be slower than FieldMonitor

`FieldMonitor` already uses direct dense accumulation for reflection and
diffraction. `RadioMapMonitor` only has that optimization today for the
axis-aligned coherent path. The default incoherent contract still uses the
baseline exporter/reducer.

Therefore:

- coherent axis-aligned radiomap can be fast
- incoherent radiomap can still be much slower than `FieldMonitor`

That is expected from the current code structure and is the first gap to close.

### Update: explicit native incoherent parity path landed

The new explicit `native_incoherent` backend closes the first correctness step:

- it no longer materializes baseline radio-map path-export payloads,
- it matches the baseline wall parity fixture within the documented tolerance,
- it reports its reflection and diffraction scalar-power replay backends in
  monitor metadata.

However, the current implementation still executes part of the scalar-power
replay orchestration in Python/Dr.Jit. Small matched `8 x 8` wall runs can look
competitive depending on warmup and run order, but the broader dense-map
scaling story is still not closed well enough to let
`accumulation_backend="auto"` select it by default.

The current reliable implementation route is:

- native UTD pair replay for per-pair vector contributions,
- Python/Dr.Jit scalarization and final power scatter reduction.

That path is correct on the current wall parity fixture, but it is still not
the same structural endpoint as Sionna's direct in-loop cell accumulation.

The surrounding scheduler decisions are now also radio-map-owned:

- `monitors/radio_map/scheduler.py` resolves receiver tiles for radio-map
  surfaces,
- diffraction no longer blindly inherits field-style tiled replay on dense
  maps,
- reflection now resolves family tiling through the same radio-map scheduler
  module instead of embedding that policy directly inside the accumulation
  helper.

### Update: `512 x 512` hot benchmark exposed the real blocker

An opt-in large wall benchmark now runs through
`python -m tests.support.bin.benchmark_radio_map_monitor --include-large-wall-512`.

On the original reviewed hot run (`--repeats 1 --warmup 1`) the matched wall
workloads reported:

- `baseline_incoherent_wall_512_center`: about `27.04 ms`
- `native_incoherent_wall_512_center`: about `3688.89 ms`
- `baseline_incoherent_wall_512_2x2`: about `90.59 ms`
- `native_incoherent_wall_512_2x2`: about `15440.82 ms`

Deep instrumentation showed that the raw native UTD pair replay was not the
main culprit. The real regression came from the scheduler:

- the reviewed `512 x 512 center` wall case had only `3` diffraction states and
  `262,144` receivers (`786,432` dense cartesian pairs),
- the tiled planner reduced that to only `491,520` estimated pairs (`62.5%` of
  the full set),
- but the radio-map scalar-power path still chose `receiver_tiled`,
- that fragmented the replay into `1024` tiny tile calls with only
  `256 / 512 / 768` input pairs each,
- instrumented time inside those tiny scalar-power replay calls alone summed to
  about `3556 ms`,
- the matched baseline path evaluated the same wall case in one dense
  diffraction field-evaluation call over `494,592` visible pairs.

That scheduler choice has now been corrected with a radio-map-specific tiling
guard and a dedicated radio-map scheduler entrypoint. On the latest reviewed
strict-gate run after that fix, the same wall workloads reported:

- `baseline_incoherent_wall_512_center`: about `90.46 ms`
- `native_incoherent_wall_512_center`: about `81.89 ms`
- `baseline_incoherent_wall_512_2x2`: about `312.88 ms`
- `native_incoherent_wall_512_2x2`: about `281.54 ms`

So the large-wall regression was primarily an orchestration problem caused by
overreusing field-style receiver tiling in the radio-map scalar-power path, not
evidence that the native UTD replay itself was fundamentally slower than the
baseline collector/reducer.

## Correctness Findings

There are two separate correctness issues. They should not be conflated.

### 1. Receiver-model mismatch with Sionna

Sionna's default radio-map definition uses the squared norm of the electric
field, i.e. an isotropic matched receiver.

The current baseline path collectors do not do that. They scalarize each path
onto a user polarization through `scalarize_vector_to_polarization(...)`, which
is a path-local receive model.

Relevant local code paths:

- `witwin/channel/monitors/path/collectors.py`
- `witwin/channel/utils/polarization.py`

Implication:

- current baseline `incoherent` radiomaps are not yet semantically identical to
  Sionna's default radio-map definition
- reflected and diffracted energy can be understated or overstated depending on
  polarization rotation along the path

### 2. Coherent semantics are inconsistent between baseline and native paths

Baseline coherent radiomap currently means:

- sum path coefficients that were already projected in a per-path local receive
  basis

Native coherent radiomap currently means:

- accumulate vector fields on the monitor plane
- project the summed vector field onto the monitor-plane tangential basis

Those are not the same operation once paths arrive from different directions.
They only agree in simple scenes where the local receive frames align well.

This explains why small parity scenes can pass while richer multipath scenes can
diverge strongly.

## Reproduced Native-Coherent Diffraction Failure

The most important current blocker is a real parity failure on a cell-centered
multipath scene.

Using the multipath main scene with:

- `grid_size = 128`
- `reflection_n_rays = 256`
- `enable_rd_diffraction = True`
- `max_diffractions = 2`
- `quadrature_mode = "center"`

the following was reproduced:

- `FieldMonitor` total power max: about `6.60e-2`
- `RadioMapMonitor` coherent baseline `path_gain` max: about `2.32e-1`
- `RadioMapMonitor` coherent native `path_gain` max: about `3.54e5`

The inflated native value comes from diffraction:

- coherent native diffraction amplitude max: about `5.95e2`
- coherent baseline total amplitude max: about `4.82e-1`

The largest reproduced native diffraction spike occurred near:

- `x = 3.140625`
- `y = -0.609375`

which is close to the right cube edge in the multipath scene.

This was narrowed down as follows:

1. The problem reproduces in `compute_diffraction_field(...)` directly on the
   cell-centered radio-map receiver positions.
2. The same receiver positions do not match the baseline
   `collect_diffraction_state_paths(...)` coherent result.
3. The failure is therefore not introduced by `trace_radio_map.py` bookkeeping
   alone.
4. The divergence is scene-dependent. The simpler wall parity case still passes.

This is a real correctness blocker for the current native coherent path.

## Why Sionna Does Not Hit This Exact Failure Mode

Sionna avoids this exact combination of problems for two reasons:

1. Its default radio-map solver is non-coherent and cell-averaged by design.
   That removes the requirement that a single point-sampled coherent value be
   numerically stable at a cell center.
2. It accumulates into cells during traversal or wedge sampling instead of
   exporting and reinterpreting path-local coefficients across two different
   receiver models.

This is why the Sionna comparison pushes us toward a different production target
than "optimize the current exporter harder."

## Priority Conclusions

The next production work should follow this order.

### Priority 1: native incoherent accumulation

Implement the final axis-aligned native incoherent radio-map reducer that
directly accumulates per-family power into cells inside the production CUDA
kernels.

The explicit `native_incoherent` parity backend landed in this repository is a
correctness-first bridge, not the final performance answer. The remaining
Sionna-aligned performance fix is moving that scalar-power accumulation out of
Python/Dr.Jit replay and into the native kernels.

### Priority 2: explicit radio-map receiver model

Introduce an explicit receiver model for radiomaps, at least separating:

- Sionna-style matched-isotropic power accumulation
- projected-polarization accumulation

Without this, "correctness" will remain ambiguous across baseline, native, and
FieldMonitor comparisons.

### Priority 3: coherent-mode contract cleanup

Keep coherent combine as an opt-in fast-fading diagnostic/product, but stop
treating baseline and native coherent reducers as if they were already the same
physical quantity.

That contract needs to be made explicit in metadata, tests, and documentation.

### Priority 4: native diffraction parity on cell-centered receivers

The current native coherent diffraction path must be fixed or guarded before it
can be treated as a generally correct production backend for dense radio maps.

## Supporting Diagnostic Script

A reproducible local comparison script now lives at:

- `tests/support/bin/analyze_radiomap_monitor.py`

The script normalizes the comparison to `ray_mode="3d"` for both
`FieldMonitor` and `RadioMapMonitor` so the reported differences are not hidden
by the legacy `FieldMonitor` default of `ray_mode="2d"`.

It compares:

- `FieldMonitor`
- `RadioMapMonitor` incoherent baseline
- `RadioMapMonitor` coherent baseline
- `RadioMapMonitor` coherent native

and reports timing, path counts, backend selection, and native diffraction peak
locations.

The latest reproduced JSON report from this note was written to:

- `tests/output/radiomap_monitor_analysis_128.json`
