# ADR-041: Slot-batched reevaluation and the time-varying CIR consumer

- **Status:** Accepted
- **Date:** 2026-07-26
- **Kind:** Additive contract change. One new request field with a safe
  default, one new topology helper, one new result surface, two new capability
  fields, one new convention string. No native code, no ABI symbol, no kernel,
  no change to the number of launches per bucket, no change to reduction order,
  no change to any published number at `slot_count == 1`.
- **Related:** ADR-032 (the one-copy/one-synchronization budget this exists to
  preserve), ADR-037 (`row_valid` is the sole authority and an invalid row is
  inert at the input), ADR-038 (wrapper-level forward-AD liveness), ADR-039
  (the source amplitude is applied exactly once), ADR-040 (world provenance)
- **Closes:** Phase-7 work items 2 and 3 (the Channel contract half) and work
  item 8

## Context

`reevaluate` replays a frozen topology at new endpoint positions. It has no
time axis. A caller who needs the same rows at `T` world instants - a radar
frame of slow-time slots, a pulse train, a symbol block, a time-varying channel
impulse response - has exactly two options today, and both are wrong.

**Call it `T` times.** Every call stays inside its own ADR-032 budget, and the
frame pays `T` four-byte validation copies and `T` synchronizations. Measured on
the multi-endpoint fixture, `T = 64`:

```text
python per-slot loop T=64 : 147.6 ms, 64 validation copies, 64 synchronizations
one batched call  T=64    :   2.1 ms,  1 validation copy,   1 synchronization
```

That is the loop the Phase-7 plan forbids by name, and the reason is not the
70x: it is that a per-instant host observation in the inner loop gives up the
entire capability the frozen topology exists to provide.

**Stack the instants into the endpoint batches.** This works and is fast, but
`pair_count = sources.count * sinks.count` (`consumer/service.py`), so stacking
`T` instants into both ends makes the pair segmentation the full
`(T*S) x (T*K)` outer product. At `T = 1024`, `S = 4`, `K = 16` that is
67,108,864 pairs - 537 MB of `int64` CSR offsets - and every one of the
67 million cross-slot pairs describes a source and a sink that never coexist.
The layout is quadratic in the number of instants and almost entirely
meaningless.

## Decision

### 1. `slot_count`, a block-diagonal pairing law

`FixedTopologyRequest` gains `slot_count: int = 1`. Declaring `slot_count = T`
states that the frozen rows and both endpoint batches are `T` slots stacked
slot-major, and selects a block-diagonal pairing law published as
`PropagationConvention.slot_pair_layout`:

```text
row          = slot * frozen_row_count + frozen_row
source_index = slot * slot_source_count + slot_source_index
sink_index   = slot * slot_sink_count   + slot_sink_index
pair_index   = slot * slot_source_count * slot_sink_count
             + slot_sink_index * slot_source_count + slot_source_index
pair_count   = slot_count * slot_source_count * slot_sink_count
```

Slots never cross-pair, so `pair_count` is LINEAR in `slot_count`. Inside one
slot the existing sink-major/source-minor layout is preserved exactly, which is
why `pair_layout` is not redefined and gains a sibling instead.

`slot_count` is validated before any native work: it must be a positive `int`,
and the source count, the sink count, and the frozen row count must each be a
multiple of it. Each refusal names the offending input. The block law itself is
enforced on device, folded into the SAME single validation bitmask as the
existing bounds/order/identity checks (bit 32): a row whose sink does not live
in its source's slot fails the batch rather than landing in another slot's pair
segment.

`slot_count > 1` requires a `PreparedFixedTopology`. See "Consequences" for why
the raw zero-interaction route is refused rather than approximated.

### 2. `replicate_over_slots`, pure index arithmetic

```python
consumer.replicate_over_slots(prepared, slot_count, *, source_count, sink_count)
```

tiles a frozen topology into `slot_count` slots and re-partitions its buckets.
It shifts `source_index` and `sink_index` by the slot block offsets, tiles every
other row field, keeps `provenance` verbatim so ADR-040 still applies, and
returns the handle unchanged when `slot_count == 1`.

`source_count` and `sink_count` are the PER-SLOT endpoint counts and are
required rather than inferred. On the fixture world the second source publishes
no row at all, so the largest `source_index` a topology carries is 0 while the
per-slot source count is 2; inferring the count from the topology would
mislabel every slot after the first, silently.

The bucket COUNT is unchanged by replication - only bucket row counts grow - so
a batched frame runs exactly the same number of native launches as a single
instant.

### 3. `evaluate_time_varying`, the second dynamics consumer

`propagation/consumer/time_varying.py` publishes:

```python
consumer.evaluate_time_varying(compiled, TimeVaryingRequest) -> TimeVaryingEvaluation
```

The request carries the per-slot prepared topology, the slot-major stacked
endpoint batches, `times_s: float64[T]`, the reference frequency, the response,
the AD mode, and the ADR-040 `world_motion`. The result publishes `delay_s`,
`path_length_m`, the transport, and `row_valid` as `[T, K]` VIEWS over the
storage one replay produced, plus the frozen per-slot `pair_offsets`,
`times_s`, and the diagnostics of that one call.

The implementation is `replicate_over_slots` + one `reevaluate(slot_count=T)` +
`view`. It owns no physics, allocates no result, adds no compaction, and
introduces no native symbol. `times_s` labels the slots and is never differenced
or integrated: a delay RATE comes from the ADR-038 forward dual, never from a
finite difference across these samples.

`evaluate_time_varying` does NOT compile scenes. One `CompiledScene` covers one
structure-geometry epoch, which is exactly what makes a slot set one frame; a
world whose structures move is `T` epochs, `T` compiles, and a motion-event
cadence rather than an inner loop.

`PathResult.tau` is deliberately untouched. It has no time axis, its `a`
tensor's trailing `num_time_steps` axis is vestigial and always 1, and widening
a frozen public solver result to host a time axis its delays cannot express
would be a public-API break for no gain.

### 4. Capability disclosure

`PropagationCapabilities` gains `supports_slot_batching: bool` (true) and
`max_slot_count: int | None` (`None`: there is no contract bound, only device
memory, because nothing per-slot grows except the row and pair tensors).

## Evidence

All measurements on an SM120 device, CUDA-synchronized, 20 iterations after 3
warmups, `f_ref = 77 GHz`, one concrete wall at `x = 4`, `{los, reflection}`,
depth 1.

**Exactness.** `test_slot_batching_is_bitwise_identical_to_a_per_slot_loop`
asserts `torch.equal` on `delay_s`, `path_length_m`, the real and imaginary
parts of `coefficient`, and `row_valid`, slot by slot, at `T = 8`. Every row is
an independent evaluation of its own endpoints, so exact equality is the correct
assertion and a tolerance would hide a gather or reduction-order defect.

**Budget, flat in `T`** (`test_the_budget_is_flat_in_slot_count`, `T` in
`{1, 16, 64, 256, 1024}`): `validation_d2h_copies == 1`,
`validation_d2h_bytes == 4`, `validation_sync_count == 1`,
`compact_count_d2h_copies == 0`, `discovery_launch_count == 0`.

**Cost, on the survey's 4x16 / K=128 configuration:**

```text
     T      rows   pair_count        ms   ms/slot     peakMB
     1       128           64     2.199    2.1988        0.1
    16      2048         1024     2.481    0.1551        0.7
    64      8192         4096     2.154    0.0337        2.6
   256     32768        16384     2.237    0.0087       10.5
  1024    131072        65536     2.139    0.0021       42.6
```

Replay is launch-bound, not work-bound: 128 rows and 131072 rows both cost
~2.2 ms. Against the survey's naive stacked shape (2.758 / 2.993 / 3.165 /
3.141 / 2.717 ms and 0.4 / 1.1 / 3.2 / 11.9 / 46.6 MB) this is no slower and
somewhat lighter, and the pair count at `T = 1024` is 65,536 instead of
67,108,864.

**Forward AD** (`test_forward_duals_survive_slot_replication`): a dual built
inside `dual_level()` and gathered - never rebuilt from Python values - carries
its tangent through slot replication. Slot 0 recedes radially at 12 m/s and
publishes `d(delay)/dt = v/c` to better than 2e-3 relative, with
`f_D = -f_ref * tau_rate < 0`; slot 1 moves purely laterally and publishes
EXACTLY `0.0`. The lateral slot alone could never distinguish a dead tangent
from a correct zero, which is why the radial slot is mandatory.

**Amplitude** (`test_time_varying_cir_does_not_reapply_transmit_power`):
quadrupling `powers_w` scales the scalar coefficient by exactly 2.0
(`sqrt`, applied once, ADR-039) and leaves the Jones operator bit-identical.

## Consequences

**The raw zero-interaction route refuses `slot_count > 1`, by name.** Its
`pair_index` and `pair_offsets` are produced inside the native
`consumer_fixed_los_gather` symbol over the full source/sink outer product.
Expressing a block-diagonal layout there would be a native change, and
recomputing the segmentation in Python afterwards would create a second owner
of the pairing law and drop the native pair-order validation. Neither is
acceptable, so the request raises `NotImplementedError` and names
`prepare_fixed_topology` as the route. A LoS-only prepared topology reaches the
same physics through the same field owner, and additionally re-tests visibility,
so nothing is lost but one preparation.

**A slot set is one structure-geometry epoch.** All slots share one
`CompiledScene`, because the reflection re-solve reads vertices from the passed
compiled scene. Structure motion is therefore a new call, not another slot.

**Replay is still subtractive.** ADR-040's limitation holds per slot and across
the block: a row can die at instant `t` and publish exact zeros there while
staying alive elsewhere, but a path that comes into existence part way through
the block is not discovered and is absent from every slot. A caller whose scene
can gain paths owns the rediscovery cadence.

**Endpoint identity repeats per slot.** The frozen rows name endpoints by
stable id, so a stacked batch repeats the same ids in every slot: the slots are
the same physical endpoints observed at different instants. A caller who wants
genuinely different endpoints per slot is describing a different topology, not
a slot.

## Rediscovery cadence

Measured on the fixture geometry, the three tiers a caller chooses between:

| Tier | Operation | Cost | Cadence |
|---|---|---|---|
| 0 - session freeze | `compile` + `evaluate` + `prepare_fixed_topology` | 2.6 + 9.1 + 0.7 ms (2x2, K=3); 14.0 ms discovery at 4x16, 39.6 ms at 16x128 | once per topology epoch |
| 1 - motion event | `evaluate` + `prepare_fixed_topology`, on `rediscovery_required` or a declared birth-gap cadence | tier 0 minus the compile when only endpoints moved | per motion event, never per pulse |
| 2 - inner loop | one `evaluate_time_varying` / `reevaluate(slot_count=T)` per frame | 2.1-2.5 ms for the whole frame, flat in `T` | per frame, pulse train, or symbol block |

`prepare_fixed_topology` synchronizes and stays in tier 0/1. Tier 2 is never a
Python loop over instants.

## Alternatives rejected

- **A native slot-aware pairing kernel.** Slot batching is index arithmetic
  over host-known counts. A kernel would add an ABI symbol, a manifest entry,
  and an AD companion family for work that produces no physical quantity.
- **Redefining `pair_layout`.** Callers read that string to interpret
  `pair_index`. A sibling that states the additional law leaves the single-slot
  meaning exactly where it was.
- **Inferring the per-slot endpoint counts.** Rejected on the fixture's own
  evidence: an endpoint that publishes no row is invisible to the topology.
- **A time axis on `PathResult`.** `tau` has no time axis and the `a` tensor's
  trailing axis is vestigial. The time-varying CIR belongs in the consumer,
  which is where the solver-neutral impulse response already lives.
- **Post-masking dead rows per slot.** ADR-037 makes an invalid row inert at
  the input, and the batched form keeps that: the zeros come out of the kernel
  that owns the value.
