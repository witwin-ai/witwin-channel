# Unified Radiomap Grid and Result Contract

Status: Draft

## Purpose

Unify deterministic and Monte Carlo radiomap public contracts around the same scene-owned receiver grid and a shared result shape, while preserving their different physical payloads.

The common user workflow should be:

```python
scene = Scene(
    structures=[...],
    transmitters=[Transmitter(name="tx", position=...)],
    receivers=[
        ReceiverGrid(
            name="rm",
            axis="z",
            position=1.5,
            bounds=((-10.0, 10.0), (-10.0, 10.0)),
            grid_shape=(256, 256),
        )
    ],
)

det = deterministic.solve(scene=scene, frequency=f, transmitter="tx", receiver="rm")
mc = montecarlo.solve(scene=scene, frequency=f, transmitter="tx", receiver="rm")
```

Both results should expose the same grid metadata and comparable scalar maps. Solver-specific data remains explicit instead of being coerced into a false common representation.

## Current State

`ReceiverGrid` already exists in `witwin.channel.core.scene` and can be attached to `Scene(receivers=[...])`.

Both standalone solvers already resolve `receiver=...` through the scene:

- `witwin.channel.deterministic.solve(...)` accepts a `ReceiverGrid` endpoint and builds a deterministic quadrature grid from it.
- `witwin.channel.montecarlo.solve(...)` accepts the same endpoint and builds a Monte Carlo scatter grid from it.

Both solvers share `witwin.channel.core.grid.GridSpec` and `Grid`, but their public result dataclasses are separate and use different naming for solver-specific data.

## Design

Add a small shared radiomap result layer under `witwin.channel.core`, not a new top-level package. This keeps shared data ownership in the existing utility layer and avoids solver-to-solver imports.

The shared layer should define:

- `RadioMapCoordinates`: common coordinate payload with `grid_x`, `grid_y`, `x`, `y`, `cell_centers`, `sample_positions`, `axis_x`, and `axis_y`.
- `RadioMapResult`: the single common result dataclass with fields for `name`, `kind`, `metric`, `solver`, `grid_shape`, `cell_size`, `surface`, `coords`, `path_gain`, `rss`, `sinr`, `tx_pos`, `tx_power`, `noise_power`, `metadata`, and optional solver-specific payload sections.
- `RadioMapFieldPayload`: deterministic coherent complex/vector payload container.
- `RadioMapPowerPayload`: Monte Carlo incoherent/coherent-power diagnostic payload container.

`deterministic.Result` and `montecarlo.Result` should be direct public aliases of the shared `RadioMapResult`, not package-local wrapper dataclasses. The public imports remain:

- `from witwin.channel.deterministic import Result`
- `from witwin.channel.montecarlo import Result`

No package-local `result.py` wrapper modules should be kept for standalone radiomap results.

## Physical Semantics

The unified fields are:

- `path_gain`: scalar power-like map used for comparison and downstream plotting.
- `rss`: `path_gain * tx_power`.
- `sinr`: single-transmitter SINR under the scene/config noise contract.
- `coords`, `surface`, `grid_shape`, and `cell_size`: resolved from the scene-owned `ReceiverGrid`.

The solver-specific fields are:

- Deterministic: `field` stores coherent complex/vector data. `components` can remain as scalar component power maps derived from field components.
- Monte Carlo: `power` stores incoherent, coherent-power, and optional diagnostic power maps. Existing Monte Carlo `incoherent`, `coherent`, and `coherent_power` fields can be preserved initially if tests or notebooks depend on them, but the preferred public access should be `result.power`.

Monte Carlo should not expose sampled power as if it were deterministic coherent field. Deterministic should not hide its coherent field behind a power-only contract.

## Data Flow

1. `Scene.receiver("rm")` returns a `ReceiverGrid`.
2. Each solver resolves the endpoint through its existing grid builder.
3. The resolved grid is converted into the shared `RadioMapCoordinates` payload through a common helper.
4. Each solver computes its native payload:
   - deterministic computes coherent field and derived scalar maps;
   - Monte Carlo computes sampled power diagnostics and scalar maps.
5. Each solver returns its package-specific `Result` with the common base fields populated consistently.

## Testing

Add focused integration tests that build one scene with a named `ReceiverGrid`, run both solvers, and assert:

- both results have identical `grid_shape`, `cell_size`, `surface`, and coordinate tensor shapes;
- both results expose `path_gain`, `rss`, and `sinr`;
- deterministic exposes `field`;
- Monte Carlo exposes `power`;
- no solver-specific result imports from the peer solver package.

Existing deterministic and Monte Carlo radiomap tests should continue to pass with current import paths.

## Non-Goals

- Do not create a mutable scene-owned accumulation buffer shared by both solvers.
- Do not make deterministic and Monte Carlo produce numerically identical component payloads.
- Do not introduce NumPy/Torch bridges in solver internals.
- Do not move solver-specific quadrature or scatter logic into `Scene`.
