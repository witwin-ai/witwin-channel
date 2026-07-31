# `witwin.channel.propagation.consumer`

The solver-neutral consumer API discovers and evaluates batches containing
exactly the actual `K` propagation rows. It imports no Channel solver and does
not perform a second Python/Torch compaction.

```python
from witwin.channel.propagation import consumer
```

## Vocabulary and constants

| Name | Value or type | Description |
| --- | --- | --- |
| `CONTRACT_VERSION` | `6` | Consumer semantic contract version. |
| `MAX_DEPTH` | `5` | Maximum requested interaction depth. |
| `PropagationComponent` | `Literal["los", "reflection", "transmission", "diffraction"]` | Component vocabulary. |
| `PropagationResponse` | `Literal["scalar_transport", "complex3_transport", "polarimetric_transport"]` | Response vocabulary. |
| `PropagationTopologyMode` | `Literal["discover"]` | Discovery-mode vocabulary. |
| `PropagationAdMode` | `Literal["none", "jvp", "vjp"]` | AD-mode vocabulary. |
| `COMPONENTS` | `frozenset[str]` | Runtime values of `PropagationComponent`. |
| `RESPONSES` | `frozenset[str]` | Runtime values of `PropagationResponse`. |
| `TOPOLOGY_MODES` | `frozenset[str]` | Currently `{"discover"}`. |
| `AD_MODES` | `frozenset[str]` | `{"none", "jvp", "vjp"}`. |

```python
assert consumer.CONTRACT_VERSION == 6
assert "scalar_transport" in consumer.RESPONSES
```

Vocabulary membership does not prove that a component/response/AD combination
is supported. Query `capabilities()` before constructing dynamic requests.

## Quick start: discover and evaluate

```python
import torch

from witwin.channel.propagation import consumer
from witwin.channel.scene import compile

f_ref = 3.5e9
compiled = compile(scene, reference_frequency_hz=f_ref)

sources = consumer.EndpointBatch(
    stable_ids=torch.tensor([101], dtype=torch.int64, device="cuda"),
    positions_m=torch.tensor([[0.0, 0.0, 1.5]], device="cuda"),
    polarizations=torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
    powers_w=torch.tensor([1.0], device="cuda"),
)
sinks = consumer.EndpointBatch(
    stable_ids=torch.tensor([201], dtype=torch.int64, device="cuda"),
    positions_m=torch.tensor([[10.0, 0.0, 1.5]], device="cuda"),
    polarizations=torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
)

request = consumer.PropagationRequest(
    sources=sources,
    sinks=sinks,
    reference_frequency_hz=f_ref,
    components=frozenset({"los"}),
    max_depth=0,
    response="scalar_transport",
    topology_mode="discover",
    ad_mode="none",
)
result = consumer.evaluate(compiled, request)

print(result.paths.path_count)                  # K
print(result.paths.geometry.delay_s.shape)      # torch.Size([K])
print(result.paths.transport.coefficient.shape) # torch.Size([K])
print(result.paths.pair_offsets.shape)          # pair_count + 1
```

## Request types

### `EndpointBatch`

```text
EndpointBatch(
    stable_ids: torch.Tensor,
    positions_m: torch.Tensor,
    polarizations: torch.Tensor,
    polarization_basis: torch.Tensor | None = None,
    powers_w: torch.Tensor | None = None,
)
```

| Field | Shape and dtype | Description |
| --- | --- | --- |
| `stable_ids` | `(N,) int64` | Stable Core-world endpoint IDs. |
| `positions_m` | `(N, 3) float32` | World positions in metres. |
| `polarizations` | `(N, 3) float32` | Endpoint polarization vectors. |
| `polarization_basis` | `(N, 2, 3) float32` or `None` | Reference basis required for Jones responses. |
| `powers_w` | `(N,) float32` or `None` | Required for sources and forbidden for sinks. |

All tensors must be contiguous CUDA tensors on one device. Stable identity is
provided by the caller; this type validates dtype, shape, device, and
contiguity.

Properties:

- `count -> int`: number of endpoints `N`.
- `device -> torch.device`: shared CUDA device.

Construction raises `TypeError` or `ValueError` for a contract violation.

### `PropagationRequest`

```text
PropagationRequest(
    sources: EndpointBatch,
    sinks: EndpointBatch,
    reference_frequency_hz: float | torch.Tensor,
    components: frozenset[str],
    max_depth: int,
    response: PropagationResponse,
    topology_mode: PropagationTopologyMode,
    ad_mode: PropagationAdMode,
    max_paths: int | None = None,
)
```

| Parameter | Description |
| --- | --- |
| `sources`, `sinks` | Endpoint batches on one CUDA device. Sources require `powers_w`; sinks forbid it. |
| `reference_frequency_hz` | Positive frequency matching `CompiledScene`; may be a scalar tensor for supported AD. |
| `components` | Non-empty `frozenset` of component names. |
| `max_depth` | Integer in `0..MAX_DEPTH`. |
| `response` | Scalar, world-Complex3, or full Jones transport. |
| `topology_mode` | Currently `"discover"`. |
| `ad_mode` | `"none"`, `"jvp"`, or `"vjp"`. |
| `max_paths` | Optional positive limit on actual compact rows; overflow fails rather than truncates. |

The constructor validates structure and vocabulary. Scene-dependent
component/response/AD capability checks occur in `evaluate` before native work.

### `FixedTopologyRequest`

```text
FixedTopologyRequest(
    sources: EndpointBatch,
    sinks: EndpointBatch,
    reference_frequency_hz: float | torch.Tensor,
    topology: PropagationTopology | PreparedFixedTopology,
    response: PropagationResponse,
    ad_mode: PropagationAdMode,
    world_motion = "frozen_world",
    slot_count: int = 1,
    frequency_offsets_hz: tuple[float, ...] | None = None,
)
```

| Parameter | Description |
| --- | --- |
| `topology` | Raw zero-interaction LoS topology, or a prepared handle for any interaction-bearing route. |
| `world_motion` | `"frozen_world"` requires all versions to match; `"fixed_winner_replay"` permits geometry-version movement only. |
| `slot_count` | Positive block-diagonal slot count. Values above 1 require prepared topology. |
| `frequency_offsets_hz` | Optional host tuple; column `j` evaluates at `f_ref + df[j]`. |

Other fields have the same meaning as `PropagationRequest`.

Property:

- `frozen_topology -> PropagationTopology`: returns the underlying topology
  whether the request holds a raw or prepared object.

The frequency tuple must be non-empty, finite, unique, and distinguishable on
the native float32 frequency grid. Jones, dispersive-material, rough-material,
and phase-screen wideband combinations fail before native work.

### `TimeVaryingRequest`

```text
TimeVaryingRequest(
    sources: EndpointBatch,
    sinks: EndpointBatch,
    reference_frequency_hz: float | torch.Tensor,
    topology: PreparedFixedTopology,
    times_s: torch.Tensor,
    response: PropagationResponse,
    ad_mode: PropagationAdMode,
    world_motion = "frozen_world",
)
```

`times_s` must be a non-empty one-dimensional `float64` tensor. Its values
label slots; they do not drive physics, rates, integration, or finite
differences. Source and sink batches are stacked slot-major and their counts
must be divisible by the slot count.

Property:

- `slot_count -> int`: returns `len(times_s)`.

## Topology and geometry

### `WorldProvenance`

```text
WorldProvenance(
    topology_version: int,
    geometry_version: int,
    material_version: int,
    assignment_version: int,
    time_s: float | torch.Tensor | None = None,
)
```

| Field | Description |
| --- | --- |
| `topology_version` | Core topology version at discovery. |
| `geometry_version` | Core geometry version at discovery. |
| `material_version` | Core material version at discovery. |
| `assignment_version` | Core logical-assignment version at discovery. |
| `time_s` | Optional snapshot time for reporting only; never a freshness gate. |

Public methods:

```text
WorldProvenance.of(compiled: _WorldVersionSource) -> WorldProvenance
```

Captures the four versions and optional time label from a compiled scene.

```text
moved_domain(
    current: WorldProvenance,
    *,
    allow_geometry: bool = False,
) -> str | None
```

Performs four host-integer comparisons and returns the first changed version
domain, or `None`. With `allow_geometry=True`, `geometry_version` is ignored.
It performs no device work, allocation, or synchronization.

### `PropagationTopology`

```text
PropagationTopology(
    source_index: torch.Tensor,
    sink_index: torch.Tensor,
    source_id: torch.Tensor,
    sink_id: torch.Tensor,
    depth: torch.Tensor,
    component_id: torch.Tensor,
    primitive_id: torch.Tensor,
    edge_id: torch.Tensor,
    material_id: torch.Tensor,
    primitive_sequence: torch.Tensor,
    material_sequence: torch.Tensor,
    interaction_type: torch.Tensor,
    provenance: WorldProvenance | None = None,
)
```

All row fields have leading dimension `K`.

| Field | Shape and dtype | Description |
| --- | --- | --- |
| `source_index`, `sink_index` | `(K,) int32` | Rows in the endpoint batches. |
| `source_id`, `sink_id` | `(K,) int64` | Stable world IDs. |
| `depth`, `component_id` | `(K,) int32` | Actual depth and component code. |
| `primitive_id`, `edge_id`, `material_id` | `(K,) int32` | First-interaction convenience columns. |
| `primitive_sequence`, `material_sequence`, `interaction_type` | `(K, D) int32` | Complete padded sequences. |
| `provenance` | `WorldProvenance | None` | Discovery provenance; hand-built topology may use `None`. |

Properties:

- `row_count -> int`: returns `K`.
- `device -> torch.device`: returns the shared CUDA device.

### `PropagationGeometry`

```text
PropagationGeometry(
    path_length_m: torch.Tensor,
    delay_s: torch.Tensor,
    field_direction: torch.Tensor,
    interaction_positions_m: torch.Tensor,
    interaction_normals: torch.Tensor,
)
```

| Field | Shape and dtype |
| --- | --- |
| `path_length_m`, `delay_s` | `(K,) float32` |
| `field_direction` | `(K, 3) float32` |
| `interaction_positions_m`, `interaction_normals` | `(K, D, 3) float32` |

Properties `row_count -> int` and `device -> torch.device` return `K` and the
shared device.

### `PropagationPathBatch`

```text
PropagationPathBatch(
    pair_count: int,
    path_count: int,
    pair_index: torch.Tensor,
    pair_offsets: torch.Tensor,
    topology: PropagationTopology,
    geometry: PropagationGeometry,
    transport: ScalarTransport | Complex3Transport | JonesTransport,
)
```

| Field | Description |
| --- | --- |
| `pair_count` | Number of logical source/sink pairs. |
| `path_count` | Actual compact row count `K`. |
| `pair_index` | Pair index per row, shape `(K,)`. |
| `pair_offsets` | CSR prefix offsets, shape `(pair_count + 1,)`. |
| `topology` | Row-aligned discrete topology. |
| `geometry` | Row-aligned continuous geometry. |
| `transport` | Transport object selected by the requested response. |

The ordinary pair layout is:

```text
pair_index = sink_index * source_count + source_index
```

CSR segmentation:

```python
start = paths.pair_offsets[pair]
stop = paths.pair_offsets[pair + 1]
pair_rows = slice(start, stop)
```

The object publishes actual rows, never provisioned capacity.

## Transport types

### `ScalarTransport`

```text
ScalarTransport(
    coefficient: torch.Tensor,
    coefficient_offsets: torch.Tensor | None = None,
    frequency_offsets_hz: tuple[float, ...] | None = None,
)
```

| Field | Shape | Description |
| --- | --- | --- |
| `coefficient` | `(K,) complex64` | Excited scalar field at the reference frequency; includes source `sqrt(powers_w)`. |
| `coefficient_offsets` | `(K, F) complex64` or `None` | Native reevaluation at each frequency offset. |
| `frequency_offsets_hz` | host tuple or `None` | One-to-one column labels. |

Properties `row_count -> int` and `device -> torch.device` report row count
and device.

### `Complex3Transport`

```text
Complex3Transport(
    field: torch.Tensor,
    direction: torch.Tensor,
    field_offsets: torch.Tensor | None = None,
    frequency_offsets_hz: tuple[float, ...] | None = None,
)
```

| Field | Shape | Description |
| --- | --- | --- |
| `field` | `(K, 3) complex64` | Excited world-Cartesian field; includes source `sqrt(powers_w)`. |
| `direction` | `(K, 3) float32` | Propagation direction. |
| `field_offsets` | `(K, F, 3) complex64` or `None` | Wideband field columns. |
| `frequency_offsets_hz` | host tuple or `None` | Column labels. |

Direction and geometry are published once because they are frequency
independent. Properties `row_count` and `device` match `ScalarTransport`.

### `JonesTransport`

```text
JonesTransport(
    matrix: torch.Tensor,
    source_basis: torch.Tensor,
    sink_basis: torch.Tensor,
)
```

| Field | Shape | Description |
| --- | --- | --- |
| `matrix` | `(K, 2, 2) complex64` | Full source-basis to sink-basis linear operator. |
| `source_basis`, `sink_basis` | `(K, 2, 3) float32` | Native row-aligned transverse bases. |

`matrix[k, i, j]` maps source component `j` to sink component `i`. Jones
transport is an excitation-free basis map and therefore does not include
`sqrt(powers_w)`. Properties `row_count` and `device` report row count and
device. Jones has no wideband payload.

## Capability, convention, and diagnostics types

### `PropagationConvention`

Every evaluation returns this frozen convention record.

| Field | Description |
| --- | --- |
| `contract_version` | Defaults to `CONTRACT_VERSION`. |
| `pair_layout` | Sink-major/source-minor formula. |
| `slot_pair_layout` | Block-diagonal slot formula. |
| `distance_unit`, `delay_unit` | `"m"` and `"s"`. |
| `phasor`, `time_dependence` | Package-wide phase convention. |
| `coefficient_reference` | Coefficients include reference-frequency phase. |
| `narrowband_frequency_offset_law` | Delay-based narrowband offset formula. |
| `narrowband_frequency_offset_error_law` | Spreading, material-fringe, and dispersion error order. |
| `wideband_offset_layout` | Wideband column layout. |
| `wideband_frequency_quantization_law` | Float32 launch grid and phase-error bound. |
| `complex3_basis` | `"world_cartesian"`. |
| `jones_mapping` | Source transverse basis to sink transverse basis. |

### `PropagationCapabilities`

`capabilities()` returns this frozen record.

| Category | Fields |
| --- | --- |
| Vocabulary | `contract_version`, `components`, `responses`, `topology_modes`, `ad_modes`, `world_motions`, `world_version_domains` |
| Compatibility matrices | `response_components`, `response_ad_modes`, `component_ad_modes` |
| Fixed topology | `fixed_topology_components`, `fixed_topology_responses`, `supports_fixed_topology`, `supports_los_jones`, `fixed_topology_row_validity_components`, `polarimetric_frozen_ad_inputs` |
| Slot/wideband | `supports_slot_batching`, `max_slot_count`, `supports_wideband_offsets`, `wideband_responses`, `wideband_components`, `wideband_dispersive_materials`, `wideband_rough_materials`, `max_frequency_offset_count`, `native_frequency_resolution_law` |
| AD detail | `component_material_leaves`, `differentiable_geometry_outputs`, `direction_differentiable_components`, `primal_only_ad_inputs`, `supports_higher_order_ad`, `ad_accounting` |

Public lookup methods:

```text
components_for(response: str) -> frozenset[str]
ad_modes_for(response: str) -> frozenset[str]
ad_modes_for_component(component: str) -> frozenset[str]
material_leaves_for(component: str) -> tuple[str, ...]
differentiable_geometry_for(route: str) -> frozenset[str]
```

An unknown name raises `ValueError`.

```python
caps = consumer.capabilities()
print(caps.components_for("scalar_transport"))
print(caps.ad_modes_for_component("diffraction"))
```

### `PropagationDiagnostics`

```text
PropagationDiagnostics(
    discovery_launch_count: int,
    candidate_count: int,
    visibility_rejection_count: int,
    compact_count_d2h_copies: int,
    compact_count_d2h_bytes: int,
    compact_sync_count: int,
    validation_d2h_copies: int,
    validation_d2h_bytes: int,
    validation_sync_count: int,
    frequency_column_count: int = 1,
    ad_companion_launches: int = 0,
    ad_tape_bytes: int = 0,
)
```

| Field | Description |
| --- | --- |
| `discovery_launch_count` | Native topology-discovery launches. |
| `candidate_count` | Candidates considered. |
| `visibility_rejection_count` | Visibility rejections. |
| `compact_count_d2h_copies` | Compact-cardinality D2H copy count. |
| `compact_count_d2h_bytes` | Compact-cardinality D2H bytes. |
| `compact_sync_count` | Compact-allocation synchronization count. |
| `validation_d2h_copies` | Fixed-replay validation copy count. |
| `validation_d2h_bytes` | Fixed-replay validation bytes. |
| `validation_sync_count` | Fixed-replay validation synchronization count. |
| `frequency_column_count` | Published propagation-frequency columns. |
| `ad_companion_launches` | Native AD companion launches. |
| `ad_tape_bytes` | AD tape bytes retained by the request. |

These fields are auditable records, not performance guarantees.

## Evaluation types

### `PropagationEvaluation`

```text
PropagationEvaluation(
    paths: PropagationPathBatch,
    convention: PropagationConvention,
    capabilities: PropagationCapabilities,
    diagnostics: PropagationDiagnostics,
)
```

| Field | Description |
| --- | --- |
| `paths` | Compact path batch. |
| `convention` | Units, phase, and layout contract. |
| `capabilities` | Consumer capability record used by the result. |
| `diagnostics` | Per-call audit counters. |

### `FixedTopologyEvaluation`

```text
FixedTopologyEvaluation(
    paths: PropagationPathBatch,
    convention: PropagationConvention,
    capabilities: PropagationCapabilities,
    diagnostics: PropagationDiagnostics,
    row_valid: torch.Tensor | None = None,
)
```

`row_valid`, when present, is a `(K,) bool` CUDA mask and the sole authority
for whether a frozen-row payload is meaningful. Replay is subtractive: an
existing row may die, but replay never discovers a newly born path.

ABI, capacity, contract, and device failures still raise. They are never
encoded as `row_valid=False`.

### `TimeVaryingTransport`

```text
TimeVaryingTransport(
    response: PropagationResponse,
    coefficient: torch.Tensor | None = None,
    field: torch.Tensor | None = None,
    direction: torch.Tensor | None = None,
    matrix: torch.Tensor | None = None,
    source_basis: torch.Tensor | None = None,
    sink_basis: torch.Tensor | None = None,
)
```

Exactly the fields belonging to `response` are non-`None`. Their leading
layout is `(S, K, ...)`.

Class method:

```text
TimeVaryingTransport.from_transport(
    transport: ScalarTransport | Complex3Transport | JonesTransport,
    slot_count: int,
) -> TimeVaryingTransport
```

Creates zero-copy slot views. The row count must be divisible by `slot_count`.

### `TimeVaryingEvaluation`

```text
TimeVaryingEvaluation(
    slot_count: int,
    row_count: int,
    times_s: torch.Tensor,
    delay_s: torch.Tensor,
    path_length_m: torch.Tensor,
    transport: TimeVaryingTransport,
    pair_count: int,
    pair_offsets: torch.Tensor,
    convention: PropagationConvention,
    capabilities: PropagationCapabilities,
    diagnostics: PropagationDiagnostics,
    row_valid: torch.Tensor | None = None,
)
```

`delay_s` and `path_length_m` have shape `(S, K)`. `row_valid`, when present,
also has `(S, K)`. `pair_count` and `pair_offsets` describe one slot because
every slot repeats the same frozen segmentation.

Class method:

```text
TimeVaryingEvaluation.from_evaluation(
    evaluation: FixedTopologyEvaluation,
    times_s: torch.Tensor,
    slot_count: int,
) -> TimeVaryingEvaluation
```

Builds slot views over one batched fixed-topology evaluation. Normal callers
receive this through `evaluate_time_varying`.

## Fixed-topology handle types

### `FixedTopologyBucket`

```text
FixedTopologyBucket(
    component: str,
    depth: int,
    rows: torch.Tensor,
)
```

`rows` contains ascending indices into the frozen `K` rows. One bucket has a
uniform component and depth.

Property:

- `row_count -> int`: number of rows in the bucket.

### `PreparedFixedTopology`

```text
PreparedFixedTopology(
    topology: PropagationTopology,
    buckets: tuple[FixedTopologyBucket, ...],
    prepare_d2h_copies: int,
    prepare_d2h_bytes: int,
    prepare_synchronizations: int,
)
```

| Field | Description |
| --- | --- |
| `topology` | Underlying frozen topology. |
| `buckets` | `(component, depth)` bucket tuple. |
| `prepare_d2h_copies` | Preparation D2H copy count. |
| `prepare_d2h_bytes` | Preparation D2H bytes. |
| `prepare_synchronizations` | Preparation synchronization count. |

Properties:

- `row_count -> int`: frozen row count `K`.
- `device -> torch.device`: topology device.
- `provenance -> WorldProvenance | None`: forwards underlying provenance.

The `prepare_*` counters describe handle construction, not a later replay.

## Functions

### `capabilities`

```text
capabilities() -> PropagationCapabilities
```

Returns the current versioned consumer capability matrix. It takes no
parameters and launches no numerical work.

### `native_frequency_resolution_hz`

```text
native_frequency_resolution_hz(
    reference_frequency_hz: float,
) -> float
```

Returns one ULP of the native float32 launch grid at the reference frequency,
in hertz.

```python
print(consumer.native_frequency_resolution_hz(3e9))  # 256.0
```

A non-finite or non-positive frequency raises `ValueError`.

### `evaluate`

```text
evaluate(
    compiled_scene: CompiledScene,
    request: PropagationRequest,
) -> PropagationEvaluation
```

Discovers and evaluates one all-or-nothing compact batch. It validates the
scene, reference frequency, response/component/AD matrix, endpoint bases, and
native capabilities before returning any payload.

See [Quick start](#quick-start-discover-and-evaluate) for a complete example.

### `prepare_fixed_topology`

```text
prepare_fixed_topology(
    topology: PropagationTopology,
) -> PreparedFixedTopology
```

Performs the one host observation of a frozen topology, validates interaction
padding, and partitions rows by `(component, depth)`. It synchronizes. Call it
once per discovered topology and reuse the returned handle.

```python
prepared = consumer.prepare_fixed_topology(result.paths.topology)
print(prepared.row_count)
print([(b.component, b.depth) for b in prepared.buckets])
```

Unsupported components, invalid padding, or a malformed topology raise before
a usable handle is returned.

### `reevaluate`

```text
reevaluate(
    compiled_scene: CompiledScene,
    request: FixedTopologyRequest,
) -> FixedTopologyEvaluation
```

Reevaluates frozen rows without topology discovery or compaction.

```python
replayed = consumer.reevaluate(
    compiled,
    consumer.FixedTopologyRequest(
        sources=moved_sources,
        sinks=moved_sinks,
        reference_frequency_hz=f_ref,
        topology=prepared,
        response="scalar_transport",
        ad_mode="none",
        world_motion="fixed_winner_replay",
    ),
)
print(replayed.row_valid.shape)  # torch.Size([K]) when published
```

Topology, material, or assignment-version movement always raises. Geometry
movement requires `world_motion="fixed_winner_replay"`.

### `rediscovery_required`

```text
rediscovery_required(
    compiled_scene: CompiledScene,
    topology: PropagationTopology | PreparedFixedTopology,
    *,
    revalidate_source: bool = False,
) -> str | None
```

Returns the moved version-domain name or `None`. The default path compares four
host integers. `revalidate_source=True` recomputes the live Core-world versions
and is `O(scene)` host work; use it on a motion-event cadence, not per frame.

```python
domain = consumer.rediscovery_required(compiled, prepared)
if domain is not None:
    discovered = consumer.evaluate(compiled, request)
    prepared = consumer.prepare_fixed_topology(
        discovered.paths.topology
    )
```

This detects version changes. It does not detect paths born under fixed-winner
replay; callers must also choose an appropriate rediscovery cadence.

### `replicate_over_slots`

```text
replicate_over_slots(
    prepared: PreparedFixedTopology,
    slot_count: int,
    *,
    source_count: int,
    sink_count: int,
) -> PreparedFixedTopology
```

Replicates one per-slot frozen topology into block-diagonal slots using index
arithmetic and bucket repartitioning only.

| Parameter | Description |
| --- | --- |
| `prepared` | Per-slot prepared topology. |
| `slot_count` | Positive slot count. |
| `source_count` | Actual source count per slot. |
| `sink_count` | Actual sink count per slot. |

Endpoint counts cannot be inferred because an endpoint with no path row is
absent from topology. `slot_count == 1` returns the original handle.

```python
replicated = consumer.replicate_over_slots(
    prepared,
    8,
    source_count=2,
    sink_count=16,
)
assert replicated.row_count == 8 * prepared.row_count
```

### `evaluate_time_varying`

```text
evaluate_time_varying(
    compiled_scene: CompiledScene,
    request: TimeVaryingRequest,
) -> TimeVaryingEvaluation
```

Evaluates a complete time-labelled block in one slot-batched replay. It does
not compile scenes, derive velocity from `times_s`, or use finite differences.

```python
times = torch.arange(
    8, dtype=torch.float64, device="cuda"
) * 1e-3
cir = consumer.evaluate_time_varying(
    compiled,
    consumer.TimeVaryingRequest(
        sources=stacked_sources,
        sinks=stacked_sinks,
        reference_frequency_hz=f_ref,
        topology=prepared,
        times_s=times,
        response="scalar_transport",
        ad_mode="none",
    ),
)
print(cir.delay_s.shape)                # torch.Size([8, K])
print(cir.transport.coefficient.shape)  # torch.Size([8, K])
```

All slots share one `CompiledScene`, and therefore one structure-geometry
epoch. Structure motion requires a newly compiled scene and a new call.

## Wideband fixed replay

`FixedTopologyRequest.frequency_offsets_hz` evaluates the same frozen rows at
multiple absolute propagation frequencies.

```python
wide = consumer.reevaluate(
    compiled,
    consumer.FixedTopologyRequest(
        sources=sources,
        sinks=sinks,
        reference_frequency_hz=f_ref,
        topology=prepared,
        response="scalar_transport",
        ad_mode="none",
        frequency_offsets_hz=(-4e8, 0.0, 4e8),
    ),
)
transport = wide.paths.transport
print(transport.coefficient_offsets.shape)  # torch.Size([K, 3])
print(transport.frequency_offsets_hz)
# (-400000000.0, 0.0, 400000000.0)
```

Column `j` is evaluated at
`reference_frequency_hz + frequency_offsets_hz[j]`. A `0.0` column is
bit-identical to the reference coefficient. Geometry and `row_valid` have no
frequency axis.

The row gather runs once above the frequency loop. Launch count grows with
`(1 + F)`, but one validation copy and one synchronization cover the whole
request. The route refuses:

| Condition | Exception | Reason |
| --- | --- | --- |
| Dispersive compiled materials | `NotImplementedError` | Compile one scene per frequency. |
| Rough materials or phase screens | `NotImplementedError` | Resident resources are compile-frequency keyed. |
| Duplicate, non-finite, empty, or unresolvable offsets | `ValueError` | Labels must name distinct native launches. |
| Tensor offset grid | `TypeError` | The grid is a host declaration, not an AD input. |
| Jones response | `NotImplementedError` | Jones has no advertised wideband payload. |

## AD boundary

The capability matrix is authoritative:

- `component_ad_modes` gives primal/JVP/VJP support per component.
- `component_material_leaves` names supported material leaves.
- `differentiable_geometry_outputs` names differentiable route outputs.
- `direction_differentiable_components` controls whole-result direction
  liveness.
- `primal_only_ad_inputs` are rejected when they require gradients.
- `supports_higher_order_ad` is `False`; second-order composition fails before
  a partial result.

Torch autograd may dispatch registered native companions. It does not
reconstruct propagation math, use finite differences, or detach unsupported
inputs.
