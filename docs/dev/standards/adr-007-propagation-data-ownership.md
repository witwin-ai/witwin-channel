# ADR-007: Propagation data ownership

Status: Accepted; stable consumer boundary added by ADR-034.

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

## Stable consumer boundary

`witwin.channel.propagation.consumer` does not publish these internal
contracts. Its version-1 `PropagationPathBatch` is a distinct, compact public
contract with native-produced pair segmentation. Fields that are semantically
and layout-identical to the owning compact output alias the same tensor
object/storage/stride; fields absent from the internal schema are produced
once by the owning native producer. The consumer never exposes internal row
identity objects, failure state, solver tape, mutable workspace, or defining
module.

The public batch contains exactly `K` valid rows, a host
`path_count == K`, native-produced `pair_index`, and native-produced
`pair_offsets`. Its order is the owning compact stage order. A consumer adapter
may validate metadata and assemble immutable typed contracts, but it may not
clone, gather, compact, reorder, recompute physics, or transfer payload.

## Final migration state

Phase 12 removed `core.path_topology.TopologyBatch` and its mixed
topology/geometry/field adapters. Producers and consumers exchange the typed
contracts above directly, preserving tensor aliasing and row order. Imports of
the deleted compatibility module are rejected by the import-graph gate.
