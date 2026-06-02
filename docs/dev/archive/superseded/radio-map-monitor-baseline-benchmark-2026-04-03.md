# Radio-Map Monitor Benchmark Snapshot 2026-04-03

Status: Active
Category: Optimization
Last reviewed: 2026-04-04

## Purpose

This document records the first frozen `RadioMapMonitor` benchmark matrix and
its follow-up native coherent rollout update plus the explicit native
incoherent parity add-on.

Use it together with:

- `docs/dev/plans/radio-map-monitor-plan.md`
- `tests/support/bin/benchmark_radio_map_monitor.py`

The plan remains the implementation narrative. This report is the point-in-time
benchmark snapshot for the current correctness-first rollout, the first native
coherent production closure, and the explicit native incoherent parity path.

## Canonical Command

Run the current baseline matrix with:

```bash
python -m tests.support.bin.benchmark_radio_map_monitor --json --strict-gates
```

Run the opt-in dense wall comparison with:

```bash
python -m tests.support.bin.benchmark_radio_map_monitor --json --strict-gates --include-large-wall-512
```

The command is intentionally small and repeatable:

- it runs the frozen benchmark matrix plus the matched baseline incoherent wall
  case, the explicit native incoherent wall case, and the native coherent wall
  case when the bundled native extension is available,
- it records timing plus summarized radio-map metadata,
- it exercises axis-aligned, quadrature, multi-TX, oriented, native
  incoherent, and native coherent parity cases,
- it exits nonzero when any lightweight correctness gate fails.

With `--include-large-wall-512`, it also runs opt-in `512 x 512` wall
comparisons for baseline vs explicit native incoherent center sampling and
baseline vs explicit native incoherent `2x2` sampling.

## Benchmark Matrix

### 1. `axis_aligned_center_baseline`

- Trace path: `Tracer.trace(...)`
- Surface: axis-aligned `z=1.5`
- Bounds: `(-6, 6) x (-6, 6)`
- Grid: `48 x 48`
- Metric: `path_gain`
- Reflection: disabled
- Diffraction: disabled
- Purpose: freeze the cell-center planar baseline

### 2. `quadrature_multipath`

- Trace path: `Tracer.trace(...)`
- Surface: axis-aligned `z=1.5`
- Bounds: `(-6, 6) x (-6, 6)`
- Grid: `20 x 20`
- Metric: `rss`
- Quadrature: `2x2`
- Reflection rays: `256`
- Reflection bounces: `1`
- Diffraction order: `1`
- Purpose: freeze the current multipath + quadrature workload

### 3. `multi_tx_sinr`

- Trace path: `Tracer.trace_many(...)`
- Surface: axis-aligned `z=1.5`
- Bounds: `(-4, 4) x (-2, 2)`
- Grid: `24 x 12`
- Metric: `sinr`
- TX count: `2`
- Reflection rays: `128`
- Reflection bounces: `1`
- Diffraction: disabled
- Explicit noise power: `1e-9`
- Purpose: freeze the current multi-transmitter aggregation path

### 4. `oriented_plane`

- Trace path: `Tracer.trace(...)`
- Surface: oriented plane
- Center: `(0, 0, 1.5)`
- Orientation: `(0.2, 0.1, 0.3)`
- Size: `(6, 4)`
- Grid: `18 x 12`
- Metric: `path_gain`
- Reflection: disabled
- Diffraction: disabled
- Purpose: freeze the rotated-plane baseline and result-space sampling path

### 5. `baseline_incoherent_wall`

- Trace path: `Tracer.trace(...)`
- Surface: axis-aligned `z=1.5`
- Bounds: `(-4, 4) x (-6, 6)`
- Grid: `8 x 8`
- Metric: `path_gain`
- Quadrature: `2x2`
- Combine mode: `incoherent`
- Accumulation backend: baseline
- Reflection rays: `4096`
- Reflection bounces: `1`
- Diffraction order: `1`
- Purpose: keep a matched baseline wall workload in the same matrix so the
  explicit native incoherent path has a fair same-scene timing comparison

### 6. `native_coherent_wall`

- Availability: included when the relevant `witwin.channel.{deterministic,montecarlo}.native_extension_available()` is `true`
- Trace path: `Tracer.trace(...)`
- Surface: axis-aligned `z=1.5`
- Bounds: `(-4, 4) x (-6, 6)`
- Grid: `8 x 8`
- Metric: `path_gain`
- Quadrature: `2x2`
- Combine mode: `coherent`
- Reflection rays: `4096`
- Reflection bounces: `1`
- Diffraction order: `1`
- Purpose: keep the axis-aligned coherent native reflection-plus-diffraction path under the same repeatable smoke/gate harness as the baseline cases

### 7. `native_incoherent_wall`

- Trace path: `Tracer.trace(...)`
- Surface: axis-aligned `z=1.5`
- Bounds: `(-4, 4) x (-6, 6)`
- Grid: `8 x 8`
- Metric: `path_gain`
- Quadrature: `2x2`
- Combine mode: `incoherent`
- Accumulation backend: explicit `native_incoherent`
- Reflection rays: `4096`
- Reflection bounces: `1`
- Diffraction order: `1`
- Purpose: keep the explicit axis-aligned incoherent parity backend under the
  same repeatable smoke/gate harness while it is still correctness-first and
  not auto-selected

## Correctness Gates

The current script enforces the following lightweight gates:

### Gate: `axis_aligned_center_metrics_finite`

- All axis-aligned baseline metric values must be finite.

### Gate: `quadrature_sample_count_matches`

- The quadrature multipath case must expose exactly four per-cell sample-position payloads for `2x2`.

### Gate: `multi_tx_association_labels_present`

- The multi-TX case must report `aggregate_tx_labels == ("left", "right")`.
- The transmitter-association map must span both association ids `0` and `1`.

### Gate: `oriented_samples_stay_on_plane`

- `RadioMapResult.sample_metric_positions(...)` samples for the oriented case must stay on the measurement plane within a small signed-distance tolerance.

### Gate: `native_coherent_backend_selected`

- The coherent wall case must resolve `metadata["accumulation_backend"]["resolved"] == "native_coherent"`.

### Gate: `native_coherent_reflection_and_diffraction_present`

- The coherent wall case must report nonzero reflected and diffracted coherent-power contributions so the parity check is not accidentally exercising a LoS-only workload.

### Gate: `native_coherent_matches_baseline`

- The coherent wall case must stay within the documented baseline-vs-native path-gain tolerance (`rtol=1e-3`, `atol=5e-6`).

### Gate: `native_runtime_backends_reported`

- The coherent wall case must report native reflection, diffraction, and suffix backends in result metadata.

### Gate: `baseline_incoherent_reflection_and_diffraction_present`

- The matched baseline incoherent wall case must report nonzero reflected and
  diffracted power contributions so the comparison case is not accidentally
  exercising a LoS-only workload.

### Gate: `native_incoherent_backend_selected`

- The incoherent wall case must resolve
  `metadata["accumulation_backend"]["resolved"] == "native_incoherent"`.

### Gate: `native_incoherent_reflection_and_diffraction_present`

- The incoherent wall case must report nonzero reflected and diffracted power
  contributions so the parity check is not accidentally exercising a LoS-only
  workload.

### Gate: `native_incoherent_matches_baseline`

- The incoherent wall case must stay within the documented baseline-vs-native
  path-gain tolerance (`rtol=1e-4`, `atol=1e-6`).

### Gate: `native_incoherent_runtime_backends_reported`

- The incoherent wall case must report the direct reflection EPC backend plus
  the current native UTD pair-vector replay backend in result metadata and must
  keep suffix accumulation disabled.

## Current Baseline Run

Reviewed command:

```bash
python -m tests.support.bin.benchmark_radio_map_monitor --repeats 1 --warmup 1 --strict-gates
```

Observed medians on the reviewed branch:

- `axis_aligned_center_baseline`: `40.09 ms`
- `quadrature_multipath`: `223.96 ms`
- `multi_tx_sinr`: `177.55 ms`
- `oriented_plane`: `18737.06 ms`
- `baseline_incoherent_wall`: `213.73 ms`
- `native_incoherent_wall`: `119.10 ms`
- `native_coherent_wall`: `102.19 ms`

Observed gate status:

- all strict gates passed

Runtime environment reported by the benchmark helper:

- Python: `C:\Users\Asixa\miniconda3\envs\witwin2\python.exe`
- backend variant: `cuda_ad_rgb`
- runtime backend: `rayd`
- native extension available: `true`
- CUDA runtime version: `12080`

## Notes

- This remains a correctness-first benchmark harness, not a full performance characterization of the final radio-map architecture.
- Reflection discovery is already shared across compatible radio-map and path-monitor traces.
- Direct-diffraction state preparation already reuses AD-safe persistent tracer caches for repeated radio-map traces.
- Native reflection- and diffraction-side radio-map accumulation are now landed
  for the axis-aligned coherent path; the benchmark harness includes a
  corresponding native coherent parity/gate case.
- The explicit `native_incoherent` backend is now covered by the same harness.
- The current explicit native incoherent diffraction path reports
  `radio_map_scalar_power_backend="native_utd_pair_vector_replay"` because the
  production route still uses native per-pair vector replay plus Python/Dr.Jit
  final scalar-power reduction.
- The matched small wall fixture is sensitive to warmup and run order, so it is
  useful as a correctness/parity smoke case but not as the sole decision point
  for default backend rollout.
- `native_incoherent` is still kept explicit because broader dense-map scaling,
  the final kernel ownership plan, and the Sionna-aligned receiver model are
  not closed yet; part of the replay orchestration still lives outside the
  production CUDA kernels.
- The current native incoherent diffraction path now uses a reliable native UTD
  pair-EPC scalar-power route plus a radio-map-specific scheduler guard that
  avoids field-style overtiling on dense maps.

## Optional `512 x 512` Wall Comparison

Reviewed command:

```bash
python -m tests.support.bin.benchmark_radio_map_monitor --repeats 1 --warmup 1 --strict-gates --include-large-wall-512
```

Observed medians on the reviewed branch after the scheduler fix:

- `baseline_incoherent_wall_512_center`: `90.46 ms`
- `native_incoherent_wall_512_center`: `81.89 ms`
- `baseline_incoherent_wall_512_2x2`: `312.88 ms`
- `native_incoherent_wall_512_2x2`: `281.54 ms`

Interpretation:

- the earlier multi-second dense-wall regression was caused by receiver-tiling
  overfragmentation in the radio-map scalar-power path rather than by the raw
  native UTD pair replay itself,
- the new scheduler guard removes that regression on the reviewed wall fixture
  and makes the explicit native incoherent backend faster than baseline on the
  matched `512 x 512` center and `2x2` cases,
- the coherent/native decoupling work also removed the last direct dependency
  on the field-only diffraction accumulation adapter by routing radio-map
  suffix replay through `trace/diffraction/suffix.py`,
- the backend is still not the default because the final scalar-power
  accumulation ownership plan is not finished and because broader scene classes
  still need benchmark coverage,
- this benchmark is opt-in because it is intentionally large and would be too
  expensive for the default smoke/gate matrix.
