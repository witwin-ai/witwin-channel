# Cell-State Memory Phase 0/1 Implementation Notes

Date: 2026-04-01

This note records the concrete developer-facing entrypoints added for Phase 0 and Phase 1 of the cell-state memory plan.

## Phase 0: Repeatable Benchmark Entry Point

Use the new benchmark runner:

```bash
conda activate witwin2
cd channel
python -m tests.support.bin.benchmark_cell_state_memory --json
```

The default matrix covers:

- `dense_field`
- `high_diffractions`
- `high_reflection_rays`
- `path_export`

Useful subsets:

```bash
python -m tests.support.bin.benchmark_cell_state_memory --cases dense_field path_export --json
python -m tests.support.bin.benchmark_cell_state_memory --cases high_diffractions --memory-profile memory_safe --json
```

The payload now reports:

- packed-state `history_size`
- packed-state stride in floats and bytes
- estimated bytes per state
- per-order state peaks before/after pruning
- max Cartesian-pair pressure per chunk
- builder timing breakdowns
- path-collection timing for path-export runs
- outer allocator snapshots for setup/trace phases

## Phase 1: Memory-Safe Profile

`TraceConfig.memory_profile` now accepts:

- `"default"`
- `"memory_safe"`

`memory_safe` is an explicit developer guardrail mode. It does not change the public tracer/result architecture, but it applies bounded diffraction-expansion defaults/caps for stress runs:

- `diffraction_state_budget <= 2048`
- `inserted_reflection_state_budget <= 512`
- `max_inserted_reflections_per_path <= 1`

Example:

```python
from witwin.channel import Tracer

tracer = Tracer(
    frequency=1e9,
    scene=scene,
    reflection_n_rays=10000,
    reflection_max_bounces=3,
    enable_rd_diffraction=True,
    max_diffractions=4,
    solver_mode="accuracy",
    memory_profile="memory_safe",
)
```

## Execution Intents

Monitor tracing now records explicit execution intent metadata:

- `field`
- `field_scalar_only`
- `path_export`

This is meant to keep Phase 0 benchmarking and Phase 1 guardrails explicit:

- field traces stay off path-export collection paths
- scalar-loss / total-only benchmark traces can avoid unnecessary field payload expansion
- path-export traces report their own collection pressure separately

## Metadata To Inspect

For field traces:

- `result.primary.metadata["execution_intent"]`
- `result.primary.metadata["state_memory_profile"]`
- `result.primary.metadata["performance_guardrails"]`
- `result.primary.metadata["performance_memory"]`

For path traces:

- `result.paths(name).metadata["execution_intent"]`
- `result.paths(name).metadata["diffraction_groups"]`
- `result.paths(name).metadata["diffraction_groups"][i]["path_collection"]`
- `result.paths(name).metadata["performance_memory"]`
