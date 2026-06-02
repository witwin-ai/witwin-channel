# Multipath Scaling Stress Report

Date: 2026-04-01

## Scope

This report summarizes the currently completed JSON payloads under:

- `tests/output/multipath_scaling_stress_2026-04-01/`

Important scope note:

- The user clarified that the main stress-test metrics should be `forward` and `backward`.
- The currently completed JSONs are from the `trace + scalar-loss JVP` worker.
- Therefore, this report treats `trace_seconds` as the valid `forward` metric.
- `jvp_seconds` is intentionally excluded from the scaling conclusions below and should only be used as a gradient-correctness smoke signal.
- No valid `backward` curve exists yet in this dataset.

## Completed Coverage

Completed before the interrupted run:

- Resolution sweep: `6 / 6`
- Ray-count sweep: `5 / 5`
- Triangle sweep: `5 / 5`
- Interaction sweep: `5 / 8`

Missing interaction cases:

- `512 x 512`, `10000` rays, `16` motifs
- `256 x 256`, `40000` rays, `16` motifs
- `512 x 512`, `40000` rays, `16` motifs

## Method

- Each measured point was executed in an isolated Python subprocess.
- Each subprocess used `warmup_runs=1` and then recorded one measured pass.
- Forward time refers to `trace_seconds`.
- Forward memory refers to `phase_metrics.trace.memory_after.drjit_allocator.device_peak_bytes`.
- All runs resolved to the native benchmark path for reflection, diffraction, and suffix accumulation.

## Forward Scaling Summary

### 1. Resolution increase mainly hurts diffraction accumulation, not discovery

At fixed `10000` rays and the base triangle count:

| Grid | Receivers | Forward | Trace peak |
|---|---:|---:|---:|
| `64 x 64` | `4096` | `0.198 s` | `~1.07 GiB` |
| `256 x 256` | `65536` | `0.348 s` | `~2.76 GiB` |
| `512 x 512` | `262144` | `0.799 s` | `~7.92 GiB` |

Observed behavior:

- `64 -> 512` in linear resolution (`8x`) increased forward time by about `4.0x`.
- Reflection stayed nearly flat: about `0.031 s -> 0.038 s`.
- The growth was almost entirely inside diffraction:
  - `diffraction_total_seconds`: about `0.166 s -> 0.760 s`
  - `diffraction_state_preparation_seconds`: only about `0.115 s -> 0.133 s`
  - `diffraction_utd_accumulation_seconds`: about `0.143 s -> 0.527 s`
  - `diffraction_suffix_seconds`: about `0.053 s -> 0.100 s`

Interpretation:

- Raising monitor resolution does **not** materially create more path states.
- It mainly increases the per-receiver accumulation workload after state construction.
- The cost increase is concentrated in UTD accumulation and suffix tracing, which are the stages that directly touch receiver-wide field buffers.

### 2. Ray-count increase mainly hurts diffraction through path-count growth

At fixed `256 x 256` resolution and the base triangle count:

| Rays | Forward | Trace peak |
|---|---:|---:|
| `2500` | `0.240 s` | `~1.35 GiB` |
| `10000` | `0.339 s` | `~2.76 GiB` |
| `40000` | `0.833 s` | `~9.58 GiB` |

Observed behavior:

- `2500 -> 40000` rays (`16x`) increased forward time by about `3.5x`.
- Reflection stayed almost flat: about `0.033 s -> 0.037 s`.
- Diffraction carried essentially all of the growth: about `0.205 s -> 0.795 s`.
- The inserted-reflection second-order state count grew from about `3154` to `12382`.
- The inserted-reflection builder’s `total_rays_cast` grew from about `10013` to `40001`.

Interpretation:

- More rays do not meaningfully stress the current reflection field accumulation path on this setup.
- They do increase the number of useful diffraction-side candidate rays and surviving second-order states.
- That pushes both the UTD accumulation stage and the suffix stage upward.

### 3. Triangle increase hurts both sides, but it hits reflection much harder

At fixed `256 x 256` resolution and `10000` rays:

| Triangles | Motifs | Forward | Trace peak |
|---|---:|---:|---:|
| `36` | `1` | `0.348 s` | `~2.76 GiB` |
| `144` | `4` | `0.659 s` | `~4.29 GiB` |
| `324` | `9` | `0.948 s` | `~6.61 GiB` |
| `576` | `16` | `1.068 s` | `~6.98 GiB` |
| `900` | `25` | `1.169 s` | `~7.35 GiB` |

Observed behavior:

- `36 -> 900` triangles (`25x`) increased forward time by about `3.4x`.
- Reflection grew from about `0.035 s` to `0.434 s` (`~12.6x`).
- Diffraction grew from about `0.313 s` to `0.733 s` (`~2.3x`).
- Prefix states grew from `9` to `421`.
- Order-2 total states grew from `3154` to `12965`.
- Diffraction edges grew from `24` to `600`.

Interpretation:

- Triangle growth expands the surface-interaction search space and edge inventory.
- That makes reflection discovery / visibility work much heavier than in the base scene.
- Diffraction also grows because more edges and more valid prefixes create more higher-order candidates, but the forward curve starts to flatten after `16` motifs, which suggests visibility and state-pruning effects begin to cap the useful-state growth.

## Interaction Findings

The currently completed interaction data is enough to establish one strong result:

### Resolution and rays have a severe multiplicative interaction

Compared with the baseline `256 x 256`, `10000` rays, `1` motif:

| Case | Forward | Trace peak |
|---|---:|---:|
| Baseline | `0.339 s` | `~2.76 GiB` |
| `512 x 512`, `10000` rays | `0.813 s` | `~7.92 GiB` |
| `256 x 256`, `40000` rays | `0.822 s` | `~9.58 GiB` |
| `512 x 512`, `40000` rays | `41.174 s` | `~29.39 GiB` |

Key comparison:

- High resolution alone: `~2.40x` forward
- High rays alone: `~2.43x` forward
- If those factors were only multiplicative, the combined case should have been around `~5.8x`
- The actual combined case was `~121.6x`

Stage breakdown in the combined case:

- `reflection_total_seconds ~= 1.661 s`
- `diffraction_state_preparation_seconds ~= 0.150 s`
- `diffraction_utd_accumulation_seconds ~= 15.582 s`
- `diffraction_suffix_seconds ~= 23.737 s`

Interpretation:

- The blow-up is **not** caused by state construction.
- It is also **not** primarily a reflection-discovery problem.
- The cliff appears when both:
  - the number of receivers is high, and
  - the number of surviving path/ray contributions is also high
- That is exactly the regime where receiver-path accumulation starts behaving like a large cross-product workload.

The current data therefore strongly suggests:

- forward compute cost has a hidden multiplicative term close to `O(receiver_count * surviving_path_count)` in the accumulation-heavy stages
- once both axes are large, suffix and UTD accumulation fall off a performance cliff
- forward memory growth is much less pathological than forward time growth in this pairwise interaction:
  - the measured combined memory factor is about `10.7x`
  - the simple multiplicative expectation from the individual factors is about `10.0x`
  - so memory is close to multiplicative, but time becomes dramatically super-multiplicative

That pattern points more toward throughput collapse / working-set explosion in accumulation kernels than toward a pure path-discovery explosion.

## Complexity Characterization From The Current Forward Data

These are empirical curve descriptions over the completed range, not asymptotic proofs:

- Resolution:
  - forward time grows moderately when resolution rises alone
  - state-preparation cost stays almost flat
  - the dominant terms are receiver-wide UTD and suffix accumulation
  - practical interpretation: low-to-mid resolution scaling looks manageable, but high resolution becomes dangerous once ray count is also high

- Rays:
  - forward time grows moderately at fixed resolution
  - reflection cost is nearly flat in this setup
  - diffraction state count and accumulation cost rise with the usable ray budget
  - practical interpretation: rays are mostly a diffraction-side multiplier

- Triangles:
  - forward time grows slower than triangle count over the tested range
  - reflection grows much faster than diffraction as geometry density increases
  - practical interpretation: triangle growth mainly taxes discovery / visibility and reflection-side work, while diffraction growth is partially absorbed by pruning / visibility filtering

## What This Dataset Can And Cannot Answer

This dataset can already answer:

- which single factor moves forward time the most on this setup
- which factor mostly stresses reflection vs diffraction
- whether resolution and rays interact badly
- which internal stages are actually responsible for the forward-time cliff

This dataset cannot yet answer:

- the real `backward` scaling curve
- whether `resolution x triangles` has the same kind of superlinear interaction
- whether `rays x triangles` has the same kind of superlinear interaction
- whether the full `resolution x rays x triangles` corner fails by time, memory, or both

## Next Step Required For The User’s Updated Metric Definition

To finish the intended stress report under the correct metric scope:

- rerun the same isolated-process sweep with a `forward + backward` worker
- keep JVP only as a correctness guard
- regenerate the interaction section using `trace_seconds` and `backward_seconds`
- then update this report in place with the missing backward curves and the missing interaction points

