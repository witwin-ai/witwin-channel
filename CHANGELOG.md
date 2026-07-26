# Changelog

All notable changes to `witwin-channel` are documented in this file.

## [Unreleased]

### Changed

- **Breaking (ADR-039).** The propagation consumer publishes the declared
  source amplitude. `ScalarTransport.coefficient` and `Complex3Transport.field`
  now carry `sqrt(sources.powers_w)` of each row's own transmitting endpoint;
  `JonesTransport` stays excitation-free because a polarization-basis map is
  not a transported field. `CONTRACT_VERSION` moves from 2 to 3. A caller that
  already multiplied by `sqrt(powers_w)` itself must stop doing so.
- The Path solver's `phase_convention` metadata quotes the unit-excitation
  free-space amplitude, matching its own `coefficient_semantics`.
- **Breaking (ADR-040).** `reevaluate` refuses a frozen topology whose world
  moved since discovery. A moved `topology_version`, `material_version`, or
  `assignment_version` always raises; a moved `geometry_version` raises unless
  the request declares `world_motion="fixed_winner_replay"`. Previously a
  stale replay returned a full-strength old answer with no signal.
  `CONTRACT_VERSION` moves from 3 to 4.

### Added

- Native `field_source_amplitude_scale` and its backward/JVP companions, the
  owner of the excited complex3 field vector.
- **ADR-040.** `consumer.WorldProvenance`, stamped onto
  `PropagationTopology.provenance` by `evaluate` and forwarded verbatim by
  `prepare_fixed_topology`; `consumer.rediscovery_required(compiled, prepared)`,
  a host-only signal naming the version domain that moved;
  `FixedTopologyRequest.world_motion`; `CompiledScene.time_s`, the compiled
  snapshot instant, carried for reporting only. No call signature changes and
  no device work: the freshness check is four host integer comparisons.
- **ADR-041.** Slot-batched fixed-topology reevaluation and the time-varying
  channel impulse response. `FixedTopologyRequest.slot_count` (default `1`, so
  every existing call is unchanged) declares that the frozen rows and both
  endpoint batches are `slot_count` block-diagonal slots stacked slot-major,
  and `consumer.replicate_over_slots(prepared, slot_count, *, source_count,
  sink_count)` builds that topology by index arithmetic. `pair_count` becomes
  `slot_count * source_count * sink_count`, linear in the slot count, and a
  whole frame costs one launch per bucket, one four-byte validation copy, and
  one synchronization instead of one of each per instant.
  `consumer.evaluate_time_varying(compiled, TimeVaryingRequest)` publishes
  `delay_s`, `path_length_m`, the transport, and `row_valid` as `[T, K]` views
  over that single replay, with `times_s: float64[T]` and the frozen per-slot
  `pair_offsets`. `PropagationConvention.slot_pair_layout` states the pairing
  law without redefining `pair_layout`, and
  `PropagationCapabilities.supports_slot_batching` / `max_slot_count` disclose
  it. `slot_count > 1` requires a `PreparedFixedTopology`: the raw
  zero-interaction route builds its pair segmentation inside a native gather
  over the full outer product and refuses slot batching by name.
  `CONTRACT_VERSION` is unchanged at 4.
- **ADR-042.** Wideband frequency offsets on the fixed-topology route.
  `FixedTopologyRequest.frequency_offsets_hz` (default `None`, so every
  existing call is unchanged bit for bit) is a host tuple of propagation
  frequency offsets, and declaring it publishes the same frozen rows evaluated
  natively at each absolute frequency `reference_frequency_hz + df_j`.
  `ScalarTransport` gains `coefficient_offsets` (`[K, F]` complex64) and
  `Complex3Transport` gains `field_offsets` (`[K, F, 3]`), each paired with a
  `frequency_offsets_hz` echo of the grid it was evaluated on; a `0.0` entry
  reproduces the reference coefficient bitwise. `row_valid` stays `[K]`,
  geometry is published once, and the validation budget stays at one four-byte
  copy and one synchronization however large `F` is - the launch count is what
  grows, as `(1 + F) * buckets * launches_per_bucket`, reported by the new
  `PropagationDiagnostics.frequency_column_count`.
  `consumer.native_frequency_resolution_hz(f_ref)` publishes the float32 launch
  resolution a caller needs to build a resolvable grid (8192 Hz at 77 GHz).
  Seven new capability fields, two new convention strings, and
  `constants.NARROWBAND_FREQUENCY_OFFSET_ERROR_LAW` quantify what the
  narrowband law this replaces actually costs. Dispersive scenes, rough or
  phase-screen scenes, and unresolvable grids are three independent fail-loud
  refusals before any native work; `polarimetric_transport` and a tensor grid
  are refused structurally. No native code, no ABI symbol, no new production
  Torch physics. `CONTRACT_VERSION` moves from 4 to 5.

## [0.4.0] - 2026-07-23

### Added

- Native Windows and `manylinux_2_28_x86_64` wheels for CPython 3.11 and
  PyTorch 2.10.
- Native CUDA SASS for SM70, SM75, SM80, SM86, SM87, SM89, SM90, SM100,
  SM101, and SM120, plus `compute_120` PTX forward compatibility.
- GitHub-hosted, release-gated wheel builds with final binary architecture and
  package-layout verification.

### Changed

- Source-link the locked, clean RayD 0.7.0 checkout at
  `49c58c4cb8212f6babb920cc88fb937509826cc5`.
- Group compatible CUDA code-generation targets and cache C/C++ compilation
  to reduce hosted-runner time without reducing release architecture coverage.
- Restrict paid native CI to explicit manual dispatches and published GitHub
  Releases. Ordinary pushes, pull requests, and schedules do not trigger it.

### Removed

- Obsolete self-hosted quick, CUDA, nightly, and evidence-only release
  workflows.

[0.4.0]: https://github.com/witwin-ai/witwin-channel/releases/tag/witwin-channel-v0.4.0
