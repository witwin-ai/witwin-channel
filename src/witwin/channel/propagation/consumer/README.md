# Propagation consumer boundary

## Ownership

`propagation.consumer` is the stable, solver-neutral contract that packages
outside Channel use to obtain propagation paths. It owns the request and result
value types, the declared conventions, the capability record, and the
orchestration that binds an explicit endpoint batch to the canonical enumerated
and compact owners.

It owns no physics. Discovery, geometry, field evaluation, and compaction stay
with `propagation.enumerated`, `propagation.geometry`, `propagation.fields`,
and `propagation.topology`. The consumer adds no second compaction, no count
observation, and no Torch gather; it reuses the ADR-032 native compact stage
and reports what that stage cost under `PropagationDiagnostics`.

The consumer is solver-neutral in both directions: it never imports `path`,
`deterministic`, `montecarlo.basic`, or `montecarlo.bdpt`, and those solvers
continue to use the internal contracts directly rather than routing through
here.

## Vocabulary

`contracts.py` is the single source of truth for the accepted values. Each
dimension is a `Literal` alias with a matching frozen set:

| Dimension | Alias | Values |
|---|---|---|
| Component | `PropagationComponent` | `los`, `reflection`, `transmission`, `diffraction` |
| Response | `PropagationResponse` | `scalar_transport`, `complex3_transport`, `polarimetric_transport` |
| Topology | `PropagationTopologyMode` | `discover` |
| AD | `PropagationAdMode` | `none`, `jvp`, `vjp` |

Scattering is deliberately absent. The existing scattering result is incoherent
power appended out of canonical order; publishing it here would misrepresent it
as field transport. It can only join once it has a coherent contract, a single
compact owner, and AD evidence.

## Discovering what is supported

```python
from witwin.channel.propagation import consumer

caps = consumer.capabilities()
assert "reflection" in caps.components_for("complex3_transport")
assert "vjp" in caps.ad_modes_for("scalar_transport")
```

`capabilities()` returns a frozen record and never runs a solve, so a caller can
check a component/response/AD combination before building a request rather than
discovering the answer from a rejected call.

## Where validation happens

`PropagationRequest` and `FixedTopologyRequest` validate their own structure at
construction: endpoint roles and device agreement, component membership, the
three vocabulary fields, `max_depth` range, and `max_paths` positivity.

Checks that need the compiled scene stay in `evaluate` and `reevaluate`:
reference-frequency match, response/component and response/AD compatibility,
polarimetric basis presence, fixed-topology interaction width, and fixed-LoS AD
input restrictions. All of them raise before any native work runs; there is no
partial result.

## Result shape

`PropagationPathBatch` publishes actual compact `K` rows, never provisioned
capacity. `pair_index` and `pair_offsets` are the CSR-style segmentation that
the owning native stage produces, in sink-major/source-minor order.
`path_count` is the host-visible `K`.

`PropagationGeometry` carries one interaction tensor pair,
`interaction_positions_m` and `interaction_normals`, both shaped
`(K, depth, 3)`. There is no separate first-interaction field; slice column 0.

## Conventions

`PropagationConvention` is returned with every result and states the pair
layout, units, phasor, time dependence, coefficient reference, Complex3 basis,
and Jones mapping. Its phasor and time-dependence strings come from
`witwin.channel.constants`, which is the package-wide owner of that convention.

Frequency offsets are not part of contract version 1. When they are
implemented, they arrive with a `CONTRACT_VERSION` bump rather than as a field
that exists but is always rejected.
