# Migrating to Channel 0.6.0

Channel 0.6.0 is a single breaking release. The radiomap solvers move into one package,
`witwin.channel.radiomap`, whose estimator is chosen by the configuration type; BDPT moves
to a new experimental tier. Numbers do not change: every old configuration has an exact new
spelling that returns bit-identical results. There are no compatibility aliases.

## Import paths and names

| 0.5 | 0.6.0 |
| --- | --- |
| `from witwin.channel import deterministic` | `from witwin.channel import radiomap` |
| `deterministic.Config` | `radiomap.Coherent` |
| `deterministic.Config(coherent=True)` (default) | `radiomap.Coherent(combine="field")` (default) |
| `deterministic.Config(coherent=False)` | `radiomap.Coherent(combine="power")` |
| `deterministic.solve(scene, config, ...)` | `radiomap.solve(scene, radiomap.Coherent(...), ...)` |
| `deterministic.Result` | `radiomap.CoherentRadioMap` |
| `deterministic.Metadata` | `radiomap.CoherentRadioMapMetadata` |
| `deterministic.PathTable` | `radiomap.PathTable` |
| `from witwin.channel.montecarlo import basic` | `from witwin.channel import radiomap` |
| `montecarlo.basic.Config` | `radiomap.Stochastic` |
| `montecarlo.basic.solve(scene, config, ...)` | `radiomap.solve(scene, radiomap.Stochastic(...), ...)` |
| `montecarlo.basic.Result` | `radiomap.RadioMap` |
| `montecarlo.basic.Metadata` | `radiomap.RadioMapMetadata` |
| `witwin.channel.montecarlo.bdpt` (all names) | `witwin.channel.experimental.bdpt` (same names; experimental tier) |
| `witwin.channel.path` | unchanged |
| `witwin.channel.propagation.consumer` | unchanged |
| `metadata["estimator"]`: `"enumerated"` / `"sampled"` | `"coherent"` / `"stochastic"` (every solver; `path` is `"coherent"`, BDPT `"stochastic"`) |
| `metadata["combine"]`: `"coherent"` / `"incoherent"` / `"none"` | `"field"` / `"power"` / `"none"` (every solver) |
| `deterministic` `metadata["coherent"]` | removed; read `metadata["combine"]` |
| `deterministic` `diagnostics["coherent"]` | `diagnostics["combine"]` |
| `PathTable.interaction_position` `(K, 3)` | removed; use `PathTable.interaction_positions[:, 0]` |
| `PathTable.interaction_normal` `(K, 3)` | removed; use `PathTable.interaction_normals[:, 0]` |

`radiomap.solve(scene, config, *, reference_frequency_hz)` dispatches on the type of
`config`: a `Coherent` returns a `CoherentRadioMap`, a `Stochastic` returns a `RadioMap`, and
anything else raises `TypeError` before any work. The two estimators keep every other
option, result field and metadata key of the 0.5 solver they replace.
`Coherent.combine="field"` sums the complex path fields per (transmitter, receiver) and
squares the sum; `"power"` sums the per-path powers. The separate `scattering_coherent`
option is unchanged. BDPT keeps its `coherent` flag; only its `combine` metadata value moves
to the new vocabulary.

The removed `PathTable` fields were the first interaction slot of the full sequences, bit
for bit, so `[:, 0]` is an exact replacement (for a LoS row both are zeros). A table whose
interaction width `D` is 0 (a LoS-only solve, `max_depth=0`) has no slot to index; the
removed fields held zeros there, so use `torch.zeros((K, 3))` if you need that shape.

Messages that named the old path now name the new one. `capabilities()["solvers"]` keeps
its keys (`deterministic`, `montecarlo_basic`, `montecarlo_bdpt`): they name capability
records, not import paths.

## Removed paths raise `ImportError`

Importing a removed path fails immediately with an `ImportError` that names the
replacement, for example:

```text
>>> import witwin.channel.deterministic
ImportError: witwin.channel.deterministic was removed in Channel 0.6.0; use
witwin.channel.radiomap: Config -> radiomap.Coherent (coherent=True/False ->
combine='field'/'power'), Result -> radiomap.CoherentRadioMap, PathTable ->
radiomap.PathTable, and solve -> radiomap.solve. See docs/migration-0.6.md.
```

`witwin.channel.montecarlo`, `witwin.channel.montecarlo.basic` and
`witwin.channel.montecarlo.bdpt` raise one message naming `witwin.channel.radiomap`
(`Stochastic`) and `witwin.channel.experimental.bdpt`. These stub modules only raise; they
re-export nothing.

## The experimental tier

`witwin.channel.experimental.*` is importable and fully tested but is not part of the stable
API snapshot: its names, options and results may change in any release
(`docs/dev/standards/adr-0xx-experimental-api-tier.md`). BDPT is its first member; its
behaviour is unchanged by the move.

## Example

```python
# 0.5
from witwin.channel import deterministic
from witwin.channel.montecarlo import basic

coherent = deterministic.solve(scene, deterministic.Config(coherent=False, export_paths=True),
                               reference_frequency_hz=3.5e9)
first_hit = coherent.paths.interaction_position
sampled = basic.solve(scene, basic.Config(samples=1 << 16), reference_frequency_hz=3.5e9)

# 0.6.0
from witwin.channel import radiomap

coherent = radiomap.solve(scene, radiomap.Coherent(combine="power", export_paths=True),
                          reference_frequency_hz=3.5e9)
first_hit = coherent.paths.interaction_positions[:, 0]
sampled = radiomap.solve(scene, radiomap.Stochastic(samples=1 << 16), reference_frequency_hz=3.5e9)
```
