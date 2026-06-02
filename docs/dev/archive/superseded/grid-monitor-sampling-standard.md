# Channel Grid / Monitor Sampling Audit

Status: Active
Category: Standard
Last reviewed: 2026-04-03

## Purpose

This note freezes the planar receiver sampling semantics used by `witwin.channel`
so future monitor work does not accidentally introduce a half-cell phase shift.

## Legacy Reference

The original sandbox/reference implementations in:

- `basic/basic.py`
- `basic/basic_rt.py`
- `basic/basic_drjit.py`

build receiver coordinates with:

```python
x = torch.linspace(range_x[0], range_x[1], grid_size)
y = torch.linspace(range_y[0], range_y[1], grid_size)
```

This is a boundary-aligned sample lattice:

- first sample sits exactly at `x_min` / `y_min`
- last sample sits exactly at `x_max` / `y_max`
- samples are **not** cell centers

For `n` samples over `[min, max]`, the sample spacing is:

```text
sample_step = (max - min) / (n - 1)
sample_i = min + i * sample_step
```

## Current Channel Implementation

`witwin/channel/monitors/field/field.py` intentionally matches the same coordinate lattice:

- `Field.get_coordinates()` uses `(max - min) / (n - 1)`
- `x_coords`, `y_coords`, `X`, and `Y` are boundary-aligned samples

This means:

- the channel field monitor samples are boundary points
- they are not cell centers

## Legacy Index Mapping That Must Also Stay Stable

The historical reflection/DDA accumulation path maps hit positions back to sample
indices using:

```text
cell_size = (max - min) / n
idx = floor((x - min) / cell_size)
```

This is not a pure cell-center model. It is also not the same spacing as the
boundary sample lattice.

So the legacy planar receiver model is:

- coordinate lattice: boundary-aligned samples
- DDA index partitioning: `span / n` bins

That combination already existed before monitorization and is preserved on
purpose for backward compatibility. Changing either side would shift reflection
sampling and can change phase.

## Compatibility Decision

The channel monitor API must preserve the old grid semantics exactly unless a
new, explicit sampling mode is introduced.

Current frozen behavior:

- `FieldMonitor` uses the same boundary-aligned sample positions as the legacy
  grid
- `result.monitor(name).metadata["receiver_sampling"]` records:
  - `sample_positions = "boundary_points"`
  - `index_partitioning = "legacy_span_over_n_bins"`

## Guardrail

Regression tests in `tests/scene/test_field_monitors.py` and `tests/scene/test_field_sampling_semantics.py`
protect:

- boundary-point coordinates
- non-center sample placement
- legacy `pos_to_idx()` binning

Do not switch to cell-centered sampling silently. If a centered monitor is ever
needed, add it as a new explicit mode and compare phase/output changes
separately.
