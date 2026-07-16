# ADR-007: Propagation data ownership

Status: accepted for the modular hardening migration.

## Decision

Internal enumerated propagation uses four shallow-immutable, slotted contracts:

- `PathTopology` owns existing discrete row fields: validity, endpoint and
  component IDs, winner IDs, material IDs, and interaction sequences.
- `PathGeometry` owns existing continuous geometry fields: lengths, delays,
  directions, interaction positions, and normals.
- `PathFields` owns existing RF fields: scalar gain, complex coefficient and
  field, and Complex3 field components.
- `EvaluatedPaths` composes the three contracts and requires them to share the
  exact same opaque row-identity object.

`PathTopology` creates the row identity. It records row count, sequence width,
and device without creating a tensor. Geometry and field producers must reuse
that object; equal counts alone are not sufficient proof of row alignment.
The row ordinal is the existing path identity and canonical order; this phase
does not invent a replacement path-ID or offset tensor absent from the source.

Constructors validate shape, dtype, device, and row identity using tensor
metadata only. They do not call `clone`, `contiguous`, `to`, or any native op,
so input tensor object identity, storage, stride, gradient state, and device
residence are preserved. Frozen dataclasses prevent attribute replacement;
tensor storage itself remains owned by the producing stage.

## Resource boundary

Propagation contracts cannot hold `Scene`, `CompiledScene`, native handles,
mutable caches, or workspaces. Required runtime resources are explicit pipeline
inputs and remain owned by `CompiledScene`. `EvaluatedPaths` is not a public
solver `Result` and does not own solver accumulation or execution metadata.

## Final migration state

Phase 12 removed `core.path_topology.TopologyBatch` and its mixed
topology/geometry/field adapters. Producers and consumers exchange the typed
contracts above directly, preserving tensor aliasing and row order. Imports of
the deleted compatibility module are rejected by the import-graph gate.
