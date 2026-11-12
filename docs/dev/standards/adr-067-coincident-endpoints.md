# ADR-067: Coincident transmitter and receiver endpoints are refused

- **Status:** Accepted (0.6.0).
- **Date:** 2026-11-11.
- **Decision ID:** `coincident-endpoints`.
- **Scope:** endpoint validation of `path`, `deterministic`, `montecarlo.basic`,
  `montecarlo.bdpt` and the propagation consumer. Host validation only: no kernel,
  launch, synchronization or numerical order changes for any accepted input.

## Context

A point receiver exactly at a transmitter position was already refused (exact
equality only). A `ReceiverGrid` sample on a transmitter, a receiver a fraction
of a millimetre away from one, and a consumer source/sink pair at one position
were all accepted, and the answers were fabricated and solver-dependent. Measured
at 3.5 GHz, transmitter at (0, 0, 1.5), 8 x 8 grid with origin (-2, -2, 1.5) and
spacing 0.5 (cell (4, 4) is on the transmitter), wall at x = 2.5, `{los, reflection}`:

| Solver | cell (4, 4), d = 0 | cell (4, 4), d = 0.1 mm |
| --- | --- | --- |
| Deterministic | 4.7e-7 (LoS row dropped, reflection only) | 4646 |
| Path | LoS coefficient 0 (zero-length direction), reflection kept | 4646 |
| MC Basic | 4.6e7 (distance clamped to 1e-6 m) | 4646 |
| BDPT | 4.6e7 (distance clamped to 1e-6 m) | 4646 |
| consumer `evaluate` | LoS coefficient exactly 0 | 4646 |

A neighbouring cell 0.5 m away reads about 1.8e-4. None of these values means
anything: every solver evaluates far-field free-space propagation, whose LoS
term `(lambda / (4 pi d))^2` diverges at `d = 0`, and whose isotropic path gain
exceeds one (more power received than transmitted) for `d < lambda / (4 pi)`.

## Options considered

1. **Define the cell's value** (zero, NaN, or exclude the LoS term) in every
   solver. Rejected. It needs a native change in four solvers and the shared
   field stage, it produces a number that silently enters coverage statistics and
   dB conversions, and "exclude LoS" is discontinuous: a cell 1 um away would
   still receive the divergent term. There is no physically correct finite value
   to choose.
2. **Refuse by name before native work.** Chosen. It is the contract point
   receivers already had, it costs only host arithmetic, and it changes no result
   that is not refused.

## Decision

A receiver sample (point receiver or `ReceiverGrid` cell) and a transmitter are
**coincident** when their distance is at most

    tolerance = max(lambda / (4 pi), 2^-20 * M)

with `lambda = c / reference_frequency_hz` and `M` the largest absolute coordinate
of the two points (for a grid, a bound over the whole grid).

- `lambda / (4 pi)` is the physical part: inside it the far-field gain the
  solvers compute exceeds unity, so the result is not a propagation quantity.
  3.5 GHz: 6.8 mm; 28 GHz: 0.85 mm; 300 MHz: 8 cm.
- `2^-20 * M` (8 to 16 float32 ulps of the coordinates) is the numerical part:
  native code computes positions and distances in float32, so a smaller distance
  cannot be told from zero. It matters only for coordinates beyond about 10 km
  at mmWave frequencies.

Scene solvers (all four) refuse coincident endpoints with `ValueError` in
`scene.endpoints.bind_solver_scene`, before any endpoint upload or solve kernel
(after scene compilation, like the other endpoint value checks). The message
names the receiver, the grid cell `(row, column)`, its position, the transmitter
id and position, the distance, the tolerance, and how many other cells are as
close. Path's explicit (non-synthetic) arrays are checked element by element
after expansion.

The grid check is analytic and never materializes the grid: per transmitter, the
least-squares grid coordinates of the transmitter and the tolerance ellipse in
the grid plane bound an index box (normally 1 x 1, at most a few cells when the
spacing is below the tolerance), and only that box is measured. A degenerate grid
(parallel or zero axes) is scanned in chunks. Cost is O(transmitters x grids) host
work, independent of grid size.

Endpoint coordinates that live on a CUDA device are not read, exactly as for the
finiteness checks: reading them would synchronize every solve.

### Consumer

`EndpointBatch` coordinates are always CUDA tensors, and `evaluate` /
`reevaluate` never read them on the host: the LoS-only `evaluate` route has no
host read at all, and the replay routes carry a single four-byte validation read
(ADR-032 budget). Detecting coincidence inside `evaluate` would add a host
synchronization to every discovery request, which the consumer budget does not
have; reporting it through the capacity failure record would need that record on
the LoS-only route, which has no host read. The replay routes could fold a
coincidence bit into their existing validation read (extra Torch reductions on
the prepared route, a native validation-kernel change on the raw route), but a
check that covers replay and not discovery would make the contract depend on the
route, so that is left as a possible follow-up. The consumer states the same
tolerance as a **precondition** and publishes
`consumer.check_endpoint_separation(sources, sinks, reference_frequency_hz=...)`,
which applies the same rule to every source/sink pair with a few device
reductions and one one-byte read, and raises `ValueError` naming the first pair
(batch indices, stable ids, positions, distance, tolerance) and the count. A
caller runs it whenever its endpoint batches change, outside a replay loop.
Without it, a coincident pair publishes a zero LoS coefficient and a
near-coincident pair a far-field value that grows without bound; a test pins
that `evaluate` does not read endpoints.

### Near-coincidence that is accepted

Distances between `lambda / (4 pi)` and a few wavelengths are accepted. There the
receiver is in the transmitter's reactive or radiating near field, and the
far-field values all solvers return (path gain up to 1, Friis phase) are an
extrapolation with growing error, not a near-field solution. This is documented,
not refused: a hard refusal at, for example, `2 lambda` would reject ordinary
dense radio maps around a transmitter.

## Consequences

- A radio map whose grid has a sample on (or within `lambda / (4 pi)` of) a
  transmitter is refused; shift the grid origin by half a spacing, or move the
  transmitter. A transmitter on a grid row or column line between sample
  centres is accepted.
- A monostatic configuration (receiver co-located with the transmitter) must
  separate the two by more than the tolerance.
- The point-receiver refusal widens from exact equality to the tolerance, and its
  message changes from "is at the same position as" to "coincides with".
- Results for every accepted input are unchanged. Evidence: 219 result tensors
  (20 solves: four solvers on three wall-and-grid scenes with the transmitter
  between centres, on a cell line and in general position, and on the 2 x 2
  synthetic city with a 16 x 16 grid and with six point receivers) were compared
  bitwise before and after the change. 208 are bit-identical; the other 11
  (Deterministic diffraction and MC Basic reflection maps) differ at 1e-7
  relative, and differ by the same amount between two runs of the same code: they
  use unordered float atomics and are not run-to-run reproducible.
- Public API: one new consumer function, `check_endpoint_separation`; no schema
  or existing signature changes.
