# Propagation consumer contract

`witwin.channel.propagation.consumer` is the stable, solver-neutral Channel
consumer boundary. Version 1 is owned by Channel and contains no Radar
waveform, target, RCS, IQ, ADC, detection, or processing policy.

## Version and compatibility

- `CONTRACT_VERSION == 4`.
- The module's `__all__` is the complete stable public surface. Consumer names
  are not duplicated at `witwin.channel`.
- A breaking schema or semantic change increments the version and atomically
  updates the public API snapshot, capability matrix, migration note, and
  package-neutral conformance tests.
- Channel does not maintain parallel production schemas or compatibility
  adapters.

## Ownership

The public contract owns endpoint batches, requests, compact path batches,
transport response types, conventions, capabilities, diagnostics, evaluations,
and fixed-topology reevaluation. `CompiledScene` is an explicit runtime input
and remains Channel-owned; it is not retained by a result.

`PropagationDiagnostics.discovery_launch_count` reports only the canonical
discovery/topology sidecar count. It is zero for fixed-topology reevaluation.
It is intentionally not a total CUDA-launch estimate. Exact end-to-end CUDA
launch evidence is recorded by the concentrated profiler acceptance run.
Compact-cardinality and fixed-validation D2H copy, byte, and synchronization
fields remain exact operation-owned counters.

The following remain internal:

- `EvaluatedPaths`, `PathTopology`, `PathGeometry`, and `PathFields`;
- solver configuration, results, accumulation, and policy;
- failure state and the terminal observer;
- RayD resources, native handles, caches, stores, and solver tapes;
- native provider and binding helpers.

## Supported components

Contract version 1 supports `los`, `reflection`, `transmission`, and
`diffraction`. `scattering` is intentionally absent from the capability matrix
and is rejected during request preflight, before discovery or field compute.
The current enumerated scattering result is incoherent power-domain data and
does not preserve the coherent scalar/Complex3 transport or canonical
pair-major row order required by this contract.

## Compact rows and segmentation

A `PropagationPathBatch` contains exactly the `K` published rows. `path_count`
is a host integer equal to `K`; it is not a capacity. Rows retain the owning
native compact stage's pair-major stable order. `pair_index` and
`pair_offsets` are produced by that same native owner. Pair numbering is
sink-major/source-minor:

```text
pair_index = sink_index * source_count + source_index
```

Therefore `pair_offsets` has `sink_count * source_count + 1` entries,
including empty segments for endpoint pairs with no published path.

A reevaluation that declares `slot_count > 1` (ADR-041) keeps that law inside
one slot and adds a block-diagonal law across slots:

```text
pair_index = slot * slot_source_count * slot_sink_count
           + slot_sink_index * slot_source_count + slot_source_index
pair_count = slot_count * slot_source_count * slot_sink_count
```

`pair_count` is therefore linear in `slot_count`, not the `(T*S) x (T*K)`
outer product, and the per-slot segmentation - empty segments included - is
identical in every slot. The law is published verbatim as
`PropagationConvention.slot_pair_layout`; `pair_layout` is not redefined.

The enumerated source pipeline owns the ADR-032 cardinality observation.
Consumer projection receives exact rows plus the owner's pair sidecar and adds
zero count D2H copies, zero stream synchronizations, and zero segmentation
scans. Python/Torch must not use `nonzero`,
Boolean indexing, `index_select`, `gather`, `clone`, `contiguous`, `.item()`,
`.cpu()`, `.numpy()`, or `.tolist()` to compact or rebuild the batch.

For a nonempty general candidate batch, `enumerated_canonical_compact`
performs stable fixed-width radix selection, canonical deduplication,
`max_paths`, exact-K allocation/gather, and pair segmentation. It uses one
8-byte `int64` control-record D2H and one explicit
caller-current-stream synchronization. The record is either `K` or a negative
contract-error code. A zero-candidate request performs no D2H and no
synchronization. `pair_offsets` is produced by device histogram plus
inclusive-scan work in `O(K + P)` and creates no host observation.

The already-exact LoS fast path uses `enumerated_exact_pair_metadata`, derives
`K` from its tensor shape, and adds no count D2H or synchronization.

Semantically identical fields alias the native compact output with exact
tensor object, storage, stride, dtype, device, and gradient state. Consumer-only
fields are produced once by the same native operation. The field-level source
of truth is recorded in
`docs/dev/audit/stage1-phase3-consumer-provenance.json`.

## Transport and convention

The contract uses SI units and `torch.complex64` CUDA tensors:

- position and path length: metres;
- delay: seconds;
- frequency: hertz;
- directions and bases: unitless world-Cartesian vectors;
- time convention: `exp(+j 2*pi*f*t)`;
- propagation convention: `exp(-j k d)`.

Scalar transport is the complex source-to-sink coefficient excited by the
declared source amplitude `sqrt(sources.powers_w)`. Complex3 transport is the
world-Cartesian complex electric-field vector of the same excited transport,
with direction; projecting it onto the receive polarization reproduces the
scalar coefficient. Jones transport is a complete complex `2 x 2` linear
operator from an explicit source transverse basis to an explicit sink
transverse basis; it is not a renamed projected field or sidecar, and it
deliberately excludes source amplitude because a linear polarization-basis map
is not a transported field (ADR-039).

## Failure and AD

Unsupported components, response types, frequency offsets, topology tangents,
or AD modes fail before partial evaluation publication. Channel completes the
internal failure transaction before constructing the public result. No public
contract contains a failure bit/state, observer, native handle, resource,
cache, or tape.

Reevaluation is defined only for fixed topology, fixed compact row identity
and order, fixed pair segmentation, fixed winner selection, and advertised
continuous inputs. JVP/VJP do not rerun discovery, compaction, or selection.
For fixed LoS, endpoint positions and the reference frequency are
differentiable; transmitter power and endpoint polarization vectors are
primal-only and are rejected by preflight when marked for AD. The compiled
host frequency value is reused, so reevaluation does not add a frequency D2H
observation. A nonempty frozen-row request performs one 4-byte validation D2H
and one current-stream synchronization; an empty request performs neither. A
slot-batched request performs the same one copy and one synchronization for
the whole slot set, whatever `slot_count` is; a Python loop over instants pays
one of each per instant and is not a supported inner loop (ADR-041).
The versioned capability object is the source of truth for supported cells.

Forward mode on the prepared route accepts a forward-only dual: liveness is
decided in the caller-facing wrapper, where the dual is still visible, and
passed into the shared native field companions as an explicit trailing input
(ADR-038, commit `fb23078`). `requires_grad` is not required, and the older
`requires_grad`-plus-dual convention keeps working unchanged. Slot replication
preserves the tangent because it is a gather on the dual tensor itself. The raw
zero-interaction route keeps its version-1 acceptance rules.

Row validity applies to
`capabilities().fixed_topology_row_validity_components`, today
`{"los", "reflection"}`. A frozen line-of-sight row is re-tested with the same
native visibility gate discovery applies, so an occluded direct path publishes
`row_valid=False` with exact zeros. The raw zero-interaction route carries no
mask and keeps its version-1 behavior.

## Release boundary

The Stage-I Channel artifact is CPython 3.11 / Torch 2.10.0 only. `_channel`
uses the versioned LibTorch/Python extension ABI; it is not advertised as a
LibTorch Stable ABI binary. Release artifacts contain native SASS for the
repository-wide architecture set, including `sm_87`, plus `compute_120` PTX.
Linux wheels are compiled in a real `manylinux_2_28` environment.
