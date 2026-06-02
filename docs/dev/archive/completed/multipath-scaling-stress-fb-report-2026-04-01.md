# Multipath Scaling Stress Report (Forward + Backward)

Date: 2026-04-01

## Scope

This report summarizes the separate forward/backward scaling worker and the currently completed outputs under:

- `tests/output/multipath_scaling_stress_fb_2026-04-01/`

Worker entrypoints used for this run:

- `tests/support/bin/benchmark_multipath_scaling_fb.py`
- `tests/support/bin/run_multipath_scaling_stress_fb.py`

Metric definition:

- `trace_seconds`: forward cost
- `backward_seconds`: reverse-mode cost from `dr.backward(loss)`
- JVP is not part of this report

## Current Completion State

Completed before the OOM interruption:

- Resolution sweep: `6 / 6`
- Ray-count sweep: `5 / 5`
- Triangle sweep: `5 / 5`
- Interaction sweep: `4 / 8`

The interrupted worker was:

- `grid=512`, `n_rays=40000`, `motif_repeats=1`

That case never produced a JSON result and should be treated as an OOM / non-completing interaction point for the current setup.

## Baseline

Baseline case:

- `256 x 256`
- `10000` rays
- `36` triangles

Measured baseline:

| Metric | Value |
|---|---:|
| Forward | `0.349 s` |
| Backward | `0.531 s` |
| Forward peak | `~2.77 GiB` |
| Backward peak | `~2.77 GiB` |

Backward is already about `1.5x` the forward time at baseline, but peak memory is almost unchanged.

## Main Findings

### 1. Backward is much more sensitive to resolution than forward

At fixed `10000` rays and base geometry:

| Grid | Receivers | Forward | Backward | Trace peak | Backward peak |
|---|---:|---:|---:|---:|---:|
| `64 x 64` | `4096` | `0.209 s` | `0.048 s` | `~1.11 GiB` | `~1.12 GiB` |
| `256 x 256` | `65536` | `0.341 s` | `0.519 s` | `~2.77 GiB` | `~2.77 GiB` |
| `512 x 512` | `262144` | `0.776 s` | `2.048 s` | `~7.92 GiB` | `~7.93 GiB` |

Observed behavior:

- `64 -> 512` in linear resolution increased forward by about `3.7x`.
- The same change increased backward by about `42x`.
- Forward peak and backward peak remained almost identical at every completed point.

Interpretation:

- Resolution mainly expands receiver-wide field accumulation.
- Reverse-mode is more exposed to that receiver fan-out than forward is.
- The fact that backward peak memory is nearly the same as forward peak memory suggests the current OOM boundary is not caused by a large extra reverse-only memory reserve. The forward working set itself is already close to the limit.

### 2. Backward is also steeper than forward with ray count

At fixed `256 x 256` and base geometry:

| Rays | Forward | Backward | Trace peak | Backward peak |
|---|---:|---:|---:|---:|
| `2500` | `0.240 s` | `0.152 s` | `~1.36 GiB` | `~1.37 GiB` |
| `10000` | `0.343 s` | `0.528 s` | `~2.77 GiB` | `~2.77 GiB` |
| `40000` | `0.833 s` | `2.007 s` | `~9.58 GiB` | `~9.61 GiB` |

Observed behavior:

- `2500 -> 40000` rays (`16x`) increased forward by about `3.5x`.
- The same ray increase pushed backward by about `13.2x`.
- Peak memory again stayed almost the same between forward and backward for each completed point.

Interpretation:

- Rays grow the number of surviving diffraction-side contributions and second-order states.
- Reverse-mode pays more than forward once those path counts rise.
- This is consistent with the completed interaction data: high ray count is not catastrophic by itself, but it becomes dangerous once combined with a large receiver grid.

### 3. Triangle growth hurts forward more than it hurts backward

At fixed `256 x 256` and `10000` rays:

| Triangles | Forward | Backward | Trace peak | Backward peak |
|---|---:|---:|---:|---:|
| `36` | `0.336 s` | `0.533 s` | `~2.77 GiB` | `~2.77 GiB` |
| `144` | `0.699 s` | `0.933 s` | `~4.30 GiB` | `~4.30 GiB` |
| `324` | `0.975 s` | `1.379 s` | `~6.63 GiB` | `~6.64 GiB` |
| `576` | `1.084 s` | `1.189 s` | `~7.00 GiB` | `~7.02 GiB` |
| `900` | `1.204 s` | `1.216 s` | `~7.37 GiB` | `~7.38 GiB` |

Observed behavior:

- Triangle growth clearly increases forward cost.
- Backward rises too, but it starts to flatten after `324` triangles.
- Reflection was the main forward-side driver in the earlier forward-only report, and the same pattern is still visible here indirectly: triangle growth increases work, but it does not create the same explosive reverse-mode scaling that resolution and rays create.

Interpretation:

- More triangles expand the reflection and edge-discovery search space.
- That raises forward time and memory, but it does not create the same receiver-path cross-product explosion as `high resolution x high rays`.
- For this setup, triangles are a secondary scaling axis relative to receiver count and ray count.

## Empirical Curve Shape

Using simple log-log fits over the completed ranges:

| Axis | Forward slope | Backward slope | Forward peak slope | Backward peak slope |
|---|---:|---:|---:|---:|
| Receivers | `0.302` | `0.905` | `0.449` | `0.448` |
| Rays | `0.446` | `0.934` | `0.729` | `0.730` |
| Triangles | `0.399` | `0.269` | `0.323` | `0.323` |

These are empirical fits, not asymptotic proofs, but they support the practical picture:

- Forward scales moderately with each single axis over the tested range.
- Backward is close to linear in the receiver-count and ray-count dimensions on this workload.
- Triangle count is not the dominant reverse-mode risk over the tested range.

## Interaction Findings

Completed interaction points:

| Case | Forward | Backward | Trace peak | Backward peak |
|---|---:|---:|---:|---:|
| Baseline `256x256`, `10000`, `1` | `0.349 s` | `0.531 s` | `~2.77 GiB` | `~2.77 GiB` |
| `512x512`, `10000`, `1` | `0.755 s` | `2.049 s` | `~7.92 GiB` | `~7.93 GiB` |
| `256x256`, `40000`, `1` | `0.793 s` | `2.006 s` | `~9.58 GiB` | `~9.61 GiB` |
| `256x256`, `10000`, `16` | `0.987 s` | `1.191 s` | `~7.00 GiB` | `~7.02 GiB` |

Relative to baseline:

- High resolution alone:
  - forward `~2.16x`
  - backward `~3.86x`
- High rays alone:
  - forward `~2.27x`
  - backward `~3.78x`
- High triangles alone:
  - forward `~2.83x`
  - backward `~2.24x`

The failed next case was exactly:

- high resolution + high rays

That is consistent with both the earlier forward-only run and the partial forward/backward run:

- the dangerous interaction is not triangle-driven
- the dangerous interaction is `receiver_count x ray_count`

## Why `512x512 x 40000 rays` Is The First OOM Candidate

From the completed points:

- `512x512`, `10000` already needs about `7.93 GiB`
- `256x256`, `40000` already needs about `9.61 GiB`
- both points also show backward cost near `2.0 s`, much steeper than baseline

From the earlier forward-only run on the same setup:

- `512x512`, `40000`, `1` motif had already blown up to `41.174 s` forward
- its forward trace peak had reached about `29.39 GiB`

So the current failed forward/backward run is consistent with the same root cause:

- state preparation is not the main issue
- the dangerous term is the accumulation-heavy `receiver_count x surviving_path_count` regime
- once `512x512` receivers and `40000` rays meet, the workload stops behaving like a mild extrapolation of the single-axis curves

## Stage-Level Interpretation

The completed forward traces still show the same internal structure as before:

- raising resolution mostly increases:
  - `diffraction_utd_accumulation_seconds`
  - `diffraction_suffix_seconds`
- raising rays mostly increases:
  - second-order state count
  - inserted-reflection rays cast
  - downstream accumulation workload
- raising triangles mostly increases:
  - reflection-side work
  - prefix states
  - edge inventory

So the forward/backward picture is:

- `resolution` is a receiver-fanout multiplier
- `rays` is a surviving-path multiplier
- `triangles` is mostly a geometry/search multiplier

The current OOM boundary is reached when the first two are both high.

## Practical Conclusions

For this multipath setup, the scaling risk ranking is:

1. `resolution x rays`
2. `resolution`
3. `rays`
4. `triangles`

More specifically:

- If the goal is to avoid OOM, do not increase `512x512` and `40000` rays together on the current implementation.
- If the goal is to stress reflection and geometry discovery without immediately hitting the worst reverse-mode cliff, triangle growth is safer than raising rays at fixed high resolution.
- Backward time should be treated as the stricter runtime budget than forward time once either receivers or rays become large.
- Peak memory for completed cases is dominated by the same buffers in forward and backward; backward is primarily a time multiplier, not a separate large memory cliff, until the combined forward working set itself becomes too large to sustain.

## Remaining Missing Data

The following interaction points still do not have completed forward/backward JSON outputs:

- `512 x 512`, `40000` rays, `1` motif
- `512 x 512`, `10000` rays, `16` motifs
- `256 x 256`, `40000` rays, `16` motifs
- `512 x 512`, `40000` rays, `16` motifs

The first missing point already demonstrated non-completion. The remaining three should be assumed high risk unless additional mitigation is added before rerunning.

