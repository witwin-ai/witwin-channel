# Deterministic Munich Scaling Optimization Plan

Status: In Progress
Category: Plan
Created: 2026-04-27

## Implementation Progress

- Completed Phase 0 profiler script with JSON output, GPU memory snapshots, kernel-history summary support, and finite/memory gates.
- Completed Phase 1 result metadata for standalone deterministic radiomaps, including phase timing, scene summary, solver controls, runtime backends, diffraction state counts, and builder reports.
- Completed Phase 2 first-order Munich baseline gate: local `512 x 512`, `max_diffractions=1`, `shadow_boundary_correction=False` run stayed below the 11 GiB process-memory gate and produced finite maps.
- Completed Phase 3 visibility for bounded second-order state growth: builder reports expose pre-expansion pruning, candidate backend, per-order state counts, and post-budget pruning.
- Completed Phase 4 receiver-side diffraction tiling for `memory_safe` runs, with tile metadata in `Result.metadata`.
- Completed Phase 5 candidate-pruned shadow-boundary correction routing for Munich-scale `shadow_boundary_correction=True`; large grids use the native candidate backend, while small reference grids can still request the dense native statistics path.

## Goal

Make the standalone deterministic radiomap solver reliable for the Munich scene at `256 x 256` and `512 x 512` receiver grids.

The practical target is:

- `512 x 512`, one-bounce reflection, first-order diffraction, `shadow_boundary_correction=False`: keep it comfortably under 16 GiB GPU memory and within interactive notebook runtime.
- `256 x 256` and `512 x 512`, second-order diffraction: make it an explicit bounded workload with predictable state counts and memory, not an accidental combinatorial expansion.
- Munich-scale deterministic shadow-boundary correction: replace the current dense all-edge correction path before enabling it by default.

## Current Evidence

Local probes on the RTX 5080 16 GiB workstation show:

- Munich scene load produces `38,936` triangles and `51,631` `all_edges` diffraction edges.
- `512 x 512`, `max_diffractions=1`, `shadow_boundary_correction=False`, `reflection_n_rays=256`, `reflection_max_bounces=1` completes in about `14.6 s`, with process GPU memory around `8.6-9.0 GiB`.
- `256 x 256` with the same first-order configuration completes comfortably.
- `max_diffractions=2`, even under `fast_approximate + memory_safe`, is state-preparation dominated and takes about one minute on small grids.
- `shadow_boundary_correction=True` now completes on `16 x 16`, `64 x 64`, `256 x 256`, and opt-in `512 x 512` first-order Munich runs by using candidate-pruned correction instead of dense all-edge replay. The local `512 x 512` correction run reported about `4.5M` candidate edge-cell pairs from a `13.5B` dense pair workload, finite output, and peak process GPU memory around `10.8 GiB`.

So the problem is not the `512 x 512` output tensor itself. The bottlenecks are:

1. diffraction state preparation for second-order and mixed diffraction families,
2. dense shadow-boundary correction over Munich-scale edge and cell counts when the candidate backend is unavailable or explicitly disabled,
3. lack of phase-level profiling in the standalone deterministic solver result, which makes future regressions hard to localize.

## Relevant Code Paths

- Public solve entry: `witwin/channel/deterministic/solver.py`
- Diffraction orchestration: `witwin/channel/deterministic/path/diffraction.py`
- Diffraction state builders: `witwin/channel/deterministic/path/diffraction_impl/builders.py`
- Receiver-side diffraction accumulation: `witwin/channel/deterministic/path/diffraction_impl/accumulation.py`
- Matched-ISB shadow-boundary correction: `witwin/channel/deterministic/path/diffraction_impl/shadow_boundary_correction.py`
- Native pruning helpers: `witwin/channel/deterministic/kernels/pruning_sort/`
- Native radio-map accumulation helpers: `witwin/channel/deterministic/kernels/radio_map_accumulate/`
- Munich notebook: `examples/deterministic_radiomap_munich.ipynb`

## Phase 0: Add A Reproducible Munich Profiler

Create `tests/support/bin/profile_deterministic_munich.py`.

The profiler should:

- load the bundled Munich XML with the same defaults as the notebook,
- expose `--grid-size`, `--max-diffractions`, `--shadow-boundary-correction`, `--edge-selection-mode`, `--reflection-n-rays`, `--solver-mode`, and `--memory-profile`,
- report wall time for scene load, solve, array materialization, and total runtime,
- capture `nvidia-smi` memory before scene load, after scene load, after solve, and after result materialization,
- optionally enable DrJit kernel history,
- write JSON under `tests/output/`.

Acceptance commands:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.profile_deterministic_munich --grid-size 64 --max-diffractions 1 --json
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m tests.support.bin.profile_deterministic_munich --grid-size 256 --max-diffractions 1 --json
```

This should be done first. Without this, later optimization work will be hard to verify.

## Phase 1: Surface Phase-Level Runtime Metadata

Add internal timing around the deterministic solve phases in `witwin/channel/deterministic/solver.py`:

- grid resolution,
- LoS trace,
- reflection trace,
- diffraction state preparation,
- diffraction accumulation,
- shadow-boundary correction,
- result tensor shaping.

Extend `witwin/channel/deterministic/result.py` with a `metadata: Mapping[str, object]` field.

Metadata should include:

- `performance_timing`,
- `scene_summary`,
- `solver_controls`,
- `runtime_backends`,
- `diffraction.state_counts`,
- `diffraction.builder_reports`,
- `shadow_boundary_correction.enabled`,
- `shadow_boundary_correction.backend`.

This mirrors the existing radio-map monitor metadata style without changing the numerical contract.

Acceptance:

- existing deterministic notebooks still run,
- `examples/deterministic_radiomap_munich.ipynb` can display timing metadata,
- a 64-grid Munich profile clearly identifies whether time is in state preparation, accumulation, or shadow-boundary correction.

## Phase 2: Keep First-Order 512 As The Stable Baseline

The first-order Munich case already runs at `512 x 512`. Make this the protected baseline:

- keep `shadow_boundary_correction=False` as the Munich notebook default,
- keep `max_diffractions=1` as the default high-resolution path,
- run `all_edges` by default for geometry completeness,
- record a benchmark gate for `512 x 512` first-order on a 16 GiB GPU.

Target gate:

- `512 x 512`, `max_diffractions=1`, `shadow_boundary_correction=False`, `reflection_n_rays=256`, `reflection_max_bounces=1`,
- peak process GPU memory below `11 GiB`,
- no Python abort,
- finite `path_gain` and component maps.

This is the configuration users should rely on while deeper diffraction and shadow-boundary correction are optimized.

## Phase 3: Bound Second-Order Diffraction State Growth

Second-order Munich is dominated by `builders.prepare()` rather than the final receiver grid size.

Improve `witwin/channel/deterministic/path/diffraction_impl/builders.py` in this order:

1. Add builder reports.
   Capture counts for `tx_first`, `prefix_first`, pre-expansion pruning, higher-order candidates, inserted reflection states, post-budget pruning, and final concatenated states.

2. Apply earlier source pruning for bounded modes.
   The current `fast_approximate + memory_safe` pruning exists, but the profiler should verify it happens before the expensive expansion stages. If expansion still materializes too many candidate pairs before pruning, move the budget boundary earlier.

3. Make `rayd_edge_bvh` the strict second-order candidate path for large scenes.
   Munich should not fall back to brute force edge-pair expansion when `n_edges` is large. If the auto selector chooses a dense path, change the selector threshold.

4. Add a large-scene guardrail preset.
   Prefer an explicit mode or helper over hidden heuristics. The effective controls should be visible in metadata:
   - `diffraction_state_budget`,
   - `inserted_reflection_state_budget`,
   - `max_inserted_reflections_per_path`,
   - candidate backend,
   - candidate probe count.

5. Cache receiver-independent state preparation.
   For a fixed scene, TX, frequency, reflection detail, edge set, and diffraction config, state arrays do not depend on `grid_shape` for the horizontal Munich plane. Cache this state preparation across repeated grid-resolution runs.

Acceptance:

- `64 x 64`, `max_diffractions=2`, `fast_approximate + memory_safe` reports bounded pre-expansion and final state counts.
- `256 x 256`, `max_diffractions=2`, bounded mode completes without OOM.
- `512 x 512`, `max_diffractions=2`, bounded mode is opt-in and has predictable memory, even if runtime remains notebook-unfriendly.

## Phase 4: Tile Receiver-Side Diffraction Accumulation

For first-order 512, memory is acceptable but still high. For second-order 512, receiver-side accumulation should not require one large all-receiver launch if state counts grow.

Update `baseline_matched_isotropic_diffraction_vector()` in `witwin/channel/deterministic/path/diffraction_impl/accumulation.py`:

- add a receiver tile size derived from memory profile,
- gather tile receiver positions,
- call `diffraction.accumulate_coherent()` per tile,
- scatter or write the tile result back into the full coherent vector buffers,
- preserve DrJit-native execution and avoid NumPy/Torch bridges in the hot path.

Target behavior:

- default mode can keep the current whole-grid path,
- `memory_safe` uses receiver tiling,
- metadata reports tile size and tile count.

Acceptance:

- first-order `512 x 512` still matches the untiled baseline within numerical tolerance,
- peak memory drops on `512 x 512`,
- no CPU fallback is introduced.

## Phase 5: Replace Dense Munich Shadow-Boundary Correction

Deterministic `shadow_boundary_correction=True` is now usable for Munich as an explicit opt-in workload. The high-resolution notebook default remains `shadow_boundary_correction=False` so the stable baseline is still the cheapest reliable path.

The current shadow-boundary correction path in `witwin/channel/deterministic/path/diffraction_impl/shadow_boundary_correction.py` computes scene-wide shadow-boundary incident statistics against all selected edges. On Munich, `51,631 edges x 262,144 cells` is too large for a dense correction strategy.

Implement a candidate-pruned native shadow-boundary correction path similar in shape to the Monte Carlo shadow-boundary native candidate backend:

1. Source-visible edge pre-cull.
   Compute and retain only edges visible from TX, with adjacent face/group ignore semantics preserved.

2. Per-cell candidate generation.
   Use projected edge bounds plus Fresnel/transition-band expansion to generate only relevant edge-cell candidates.

3. Native candidate accumulation.
   Route large deterministic shadow-boundary correction workloads through the shared native shadow-boundary candidate kernel. The large-grid candidate path uses bounded max-weight plus weighted-response-average aggregation over candidate cells; the exact dense sum-weight statistics path remains available for small reference grids through `shadow_boundary_backend="dense_native"`.

4. Metadata and strict fallback.
   For large scenes, `shadow_boundary_correction=True` should either use the candidate backend or raise a clear error. It should not silently enter the dense path.

5. Safety gate.
   Add a conservative `shadow_boundary_max_candidate_factor` or equivalent guard so pathological scenes fail with a useful message instead of aborting the process.

Acceptance:

- `16 x 16`, Munich, `shadow_boundary_correction=True` completes without process abort.
- `64 x 64`, Munich, `shadow_boundary_correction=True` completes and reports candidate counts.
- `256 x 256`, Munich, `shadow_boundary_correction=True` completes under 16 GiB.
- `512 x 512`, Munich, `shadow_boundary_correction=True` is opt-in and bounded by candidate statistics.

Verified local evidence:

- `16 x 16`, `shadow_boundary_correction=True`: completed with `native_candidate`, `4,416` candidate pairs, peak process GPU memory around `9.6 GiB`.
- `64 x 64`, `shadow_boundary_correction=True`: completed with `native_candidate`, `70,402` candidate pairs, peak process GPU memory around `9.9 GiB`.
- `256 x 256`, `shadow_boundary_correction=True`: completed with `native_candidate`, `1,126,712` candidate pairs, peak process GPU memory around `10.1 GiB`.
- `512 x 512`, `shadow_boundary_correction=True`: completed with `native_candidate`, `4,506,366` candidate pairs, peak process GPU memory around `10.8 GiB`.

## Phase 6: Notebook And Defaults

After the core changes:

- update `examples/deterministic_radiomap_munich.ipynb` with:
  - a profiler summary cell,
  - a first-order high-resolution default,
  - an opt-in second-order bounded cell,
- an opt-in shadow-boundary correction cell only after Phase 5,
  - clear printed effective solver controls.
- keep Sionna comparison opt-in because it is not needed to validate deterministic scaling.
- keep `shadow_boundary_correction=False` as the high-resolution default, with candidate shadow-boundary correction exposed as an explicit probe.

## Recommended Implementation Order

1. Profiler script and JSON output.
2. Deterministic `Result.metadata` and phase timing.
3. State-builder reports and large-scene guardrail visibility.
4. Receiver tiling for `memory_safe`.
5. Second-order bounded Munich validation.
6. Native candidate-pruned shadow-boundary correction.
7. Notebook update.

## Risk Notes

- Accuracy mode should remain exact within the currently implemented deterministic families. Any pruning must be explicit in config and metadata.
- Do not add CPU fallback paths for core solve stages.
- Do not use NumPy/Torch in solver hot paths. NumPy is acceptable in profiling scripts and result summaries only.
- Shadow-boundary correction optimization should preserve finite-edge semantics and existing adjacent-face ignore logic.
- The Munich default should optimize reliability first; second-order and shadow-boundary correction should remain opt-in until bounded by tests and profiler output.
