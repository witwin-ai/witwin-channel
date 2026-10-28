# Plan 17: Converge the deterministic propagation entries

## Status

Proposed. Needs a decision on D1-D4 (section 8) and an accepted ADR before
Phase 1 lands. Nothing in this plan changes physics, numerical order, launch
configuration, or AD support.

## 1. Answer

Channel has three public entries for deterministic enumerated propagation:
`witwin.channel.path`, `witwin.channel.deterministic`, and
`witwin.channel.propagation.consumer`. All three call the same engine,
`propagation.enumerated.evaluate_enumerated_paths`. They differ only in how a
request is spelled and how the rows are shaped afterwards.

They can be reduced to one engine, one option owner, and one row table. They
should **not** be reduced to one public function. The recommended end state
has two entries, split by who the caller is:

| Entry | Caller | Input | Output |
|---|---|---|---|
| `propagation.consumer` | Radar and other programs that own their endpoints | `CompiledScene` + `EndpointBatch` | compact `K` rows, fixed-topology replay, AD contract |
| `deterministic` | people working from a `witwin.core.Scene` | `Scene` with transmitters and receivers | receiver-grid maps and per-link `PathResult` |

`path` merges into `deterministic` (Phase 1). The two remaining entries then
share one option type and one endpoint adapter (Phase 2). A literal single
entry is rejected in section 4.

## 2. Current state

### 2.1 One engine, three front ends

| | `path` | `deterministic` | `propagation.consumer` |
|---|---|---|---|
| Entry | `path.solve(scene, config, *, reference_frequency_hz)` | `deterministic.solve(scene, config, *, reference_frequency_hz)` | `consumer.evaluate(compiled_scene, request)` |
| Option type | `path.Config` (15 fields) | `deterministic.Config` (22 fields) | `PropagationRequest` (9 fields) plus private `_ConsumerConfig` |
| Endpoints | scene transmitters/receivers, antenna arrays | scene transmitters/receivers, `ReceiverGrid` | explicit `EndpointBatch` |
| Engine call | `evaluate_enumerated_paths` | `evaluate_enumerated_paths` | `evaluate_enumerated_paths` |
| Row table | `RaggedPathSoA` -> padded `PathResult` | padded `PathTable` (24 fields, optional) | compact `PropagationPathBatch` |
| Terminal stage | padding, array packing, CIR/CFR/taps/beamform | native coherent accumulation (`kernels.deterministic`) | native compact finalize, scalar/Complex3/Jones transport |

The engine reads its options through the `EnumeratedPathConfig` and
`TopologyConfig` protocols. Four concrete types satisfy them today:
`path.Config`, `deterministic.Config`, `consumer._ConsumerConfig`, and
`montecarlo.bdpt.Config`, which reaches the engine under ADR-008.
`montecarlo.basic` does not use the engine.

### 2.2 Duplicated options

Every one of the 15 fields of `path.Config` also appears in
`deterministic.Config`, with the same name, default, and validator from
`witwin.channel.components`:

`max_depth`, `components`, `max_paths`, `max_paths_scope`, `ad_mode`,
`coupled_paths`, `coupled_candidate_limit`, `scattering_samples_per_m2`,
`scattering_max_paths_per_pair`, `scattering_power_threshold`,
`scattering_chain_max_depth`, `scattering_chain_samples_per_m2`,
`scattering_chain_max_rows`, `isb_boundary_taper`, `isb_boundary_taper_width`.

The long comments on the scattering-chain and ISB-taper fields are copied
between the two files. The copies have already drifted:

- `max_paths_scope` defaults to `"per_pair"` in `path` and `"global"` in
  `deterministic`, and `path` rejects `"global"`.
- The depth cap for reflection/transmission is checked in `path.Config` and
  repeated in `deterministic.Config`, and stated a third time as
  `consumer.MAX_DEPTH`.
- `deterministic.Config.max_diffraction_order` accepts only `0` or `1` and
  duplicates the presence of `"diffraction"` in `components`.

### 2.3 Capability drift between the entries

| Capability | `path` | `deterministic` | `consumer` |
|---|---|---|---|
| `los`, `reflection`, `transmission` | yes | yes | yes |
| `diffraction` | yes | yes | primal raises `IndexError` (pinned defect, see `propagation-ad-capability-matrix.md` section 4) |
| `scattering` (incoherent power rows) | yes | yes, plus `scattering_coherent` | refused before compute |
| Coupled reflection-diffraction | yes | yes, with receiver-block streaming | no |
| `ReceiverGrid` | no | yes | no |
| Antenna arrays and directional patterns | yes | refused by `validate_scalar_endpoint_features` | no |
| Fixed-topology replay, slots, time-varying, wideband | no | no | yes |
| Jones transport, world provenance, published AD matrix | no | no | yes |
| ISB taper | yes | yes | no |

Every new capability so far has landed in exactly one entry. The consumer has
received all of the replay work (ADR-037 to ADR-043); `path` and
`deterministic` have received the scattering and taper work. A user who needs
two of these at once has no entry that offers both.

### 2.4 Who calls what

- **Radar** imports Channel only in `radar/witwin/radar/channel.py`. It uses
  ten consumer names: `evaluate`, `reevaluate`, `prepare_fixed_topology`,
  `replicate_over_slots`, `rediscovery_required`, `capabilities`,
  `EndpointBatch`, `PropagationRequest`, `FixedTopologyRequest`, `AD_MODES`.
  It does not pin `CONTRACT_VERSION`. `radar/tests/test_import_boundary.py`
  asserts that importing Radar never loads `witwin.channel.path`,
  `witwin.channel.deterministic`, or the engine modules.
- **Channel tests**: 46 files reference `path`, 65 reference `deterministic`,
  45 reference the consumer.
- **Benchmarks**: 5 files use `path`, 10 use `deterministic`, including the
  full-wave validation campaign that Plan 16 is running
  (`benchmarks/fullwave_validation/`).
- **Inside the package**: `path` imports `compact_evaluated_paths` from the
  consumer; `capabilities` embeds `consumer.capabilities()`. No other package
  in the monorepo imports any of the three.

## 3. Goals and non-goals

Goals:

- One option owner for the knobs the engine reads.
- One row table that a scene-level caller and an endpoint-level caller can
  both reach.
- A new capability lands once and is reachable from both remaining entries,
  or is refused by name in the entry that cannot carry it.

Non-goals:

- No change to `montecarlo.basic` or `montecarlo.bdpt`. They are stochastic
  solvers with their own estimators; ADR-008 already limits BDPT to the public
  engine entry.
- No change to physics, kernel fusion, launch counts, synchronization, or AD
  support. A refactor here that changes any of them fails its phase gate.
- No fix to the diffraction or scattering AD gaps. They stay `REF`.

## 4. Options

**A. Keep three entries.** No migration cost. Each new capability keeps
landing in one entry, and the 15 shared fields keep drifting. Rejected.

**B. One public entry.** Fold grids, antenna arrays, scattering sampling,
coupled streaming, and taper options into `PropagationRequest`, and delete
`path` and `deterministic`. Rejected for three reasons:

1. Radar needs an endpoint contract with no antenna, grid, or scene-receiver
   concepts, and its import boundary forbids loading solver modules. Every
   scene-level option added to the request becomes something Radar has to
   validate or ignore.
2. `consumer.py` is already the largest Channel module (4342 lines, contract
   version 6). Adding about 20 scene-level options would turn it into the
   option bag that the repository rules forbid.
3. A grid radiomap would have to go through the public compact finalize and
   publish `K` row views before accumulation. That stage does not exist on the
   grid path today. The repository rules forbid a refactor that adds a
   materialized intermediate or a launch, so B would need its own numerical
   ADR with memory and throughput evidence.

**C. One engine, one option owner, two entries (recommended).** Merge `path`
into `deterministic`. Move the shared options into one type that both
`deterministic.Config` and the consumer read. Keep two terminal stages:
compact publication for the consumer, and accumulation or `PathResult`
packing for `deterministic`. Section 5 describes the target surface.

## 5. Target surface (option C)

### 5.1 Option ownership

- The 15 shared fields become one frozen dataclass owned by
  `propagation.enumerated`, beside the `EnumeratedPathConfig` and
  `TopologyConfig` protocols it satisfies. Working name:
  `EnumeratedOptions`. It is the engine's own contract, not a generic bag.
- `deterministic.Config` holds an `EnumeratedOptions` plus only the fields the
  scene-level terminal stage reads: `coherent`, `return_field`,
  `export_paths`, `sort_key`, `diagnostics`, `scattering_coherent`.
- `max_diffraction_order` is deleted. `"diffraction" in components` is the
  only switch.
- `consumer._ConsumerConfig` is deleted. The consumer builds an
  `EnumeratedOptions` from `PropagationRequest`, and leaves every field it
  does not expose at its default.

### 5.2 Entries

- `deterministic.solve(scene, config, *, reference_frequency_hz) -> Result`
  keeps its signature. `Result.paths` becomes `PathResult | None`.
- `PathResult` and `InteractionType` move to `witwin.channel.deterministic`.
  `PathResult` keeps `cir`, `cfr`, `taps`, `beamform`, and `filter_by_type`.
  It also gains `PathResult.from_propagation(evaluation, ...)`, so a consumer
  caller can get the same padded antenna view without a second solve.
- Antenna arrays and directional patterns are accepted when
  `export_paths=True`, and stay refused for the accumulated maps, which is
  today's `deterministic` rule.
- `consumer.evaluate` and every replay entry keep their signatures.

### 5.3 Shared endpoint adapter (Phase 2)

Both entries reach the engine through one adapter from endpoints to the
engine's `SolverScene` and `EnumeratedEndpointTensors`. Today the consumer
builds a `SolverScene` with empty `transmitters` and `receivers`
(`consumer._solver_scene`). That is the cause of the pinned diffraction
`IndexError`, because `interactions.diffraction` indexes
`tx_polarizations[tx_index]` from the scene. One adapter removes the second
spelling and the defect together. Section 8, D4, makes this an explicit
decision rather than a side effect.

## 6. Phases

### Phase 0: Inventory and baselines

1. List every `path.Config` and `deterministic.Config` field with its
   readers in the engine, the terminal stage, metadata, and tests.
2. Record baselines for `path.solve` and `deterministic.solve` on the Munich
   grid, the three-way path benchmark (`bench_path_solver_threeway.py`), and
   the Plan 16 single-cube and three-cube scenes: outputs, launch count,
   synchronization count, D2H bytes, peak memory, and wall time.
3. Add a row-equivalence test: for a scene without scattering or coupled
   paths, `path.solve`, `deterministic.solve(export_paths=True)`, and
   `consumer.evaluate` must publish the same rows, delays, and coefficients.
   Any mismatch is a finding to resolve before Phase 1, not after.

Exit: baselines committed; the equivalence test passes or every difference
has a named owner.

### Phase 1: Merge `path` into `deterministic`

1. Introduce `EnumeratedOptions` and move the 15 shared fields and their
   validators into it.
2. Move `PathResult`, `InteractionType`, array packing, and the ragged-to-padded
   conversion into `deterministic`. `RaggedPathSoA` becomes private.
3. Replace `PathTable` with `PathResult` in `Result.paths`.
4. Delete `witwin.channel.path` in the same commit. Per the repository rules,
   it leaves no alias and no re-export.
5. Delete `max_diffraction_order`.

Exit:

- `path` scenes reproduce their Phase 0 outputs bit for bit through
  `deterministic.solve(export_paths=True)`.
- Launch count, synchronization count, D2H bytes, and peak memory are within
  the recorded budgets.
- A scene with antenna arrays or directional patterns and
  `export_paths=False` is still refused before compute.
- `quick` and `cuda` tiers pass.
- `ci/public-api-snapshot.json` is updated, and the migration note is in
  `docs/dev/replacement/channel-migration.md`.

### Phase 2: One option owner and one endpoint adapter

1. The consumer builds its engine options as an `EnumeratedOptions` and
   deletes `_ConsumerConfig`.
2. Both entries use the one endpoint adapter from 5.3.
3. The consumer diffraction primal starts working. Diffraction AD stays `REF`
   in `component_ad_modes`, and the pinned-defect test is replaced by a primal
   correctness test against `deterministic`.
4. Add `PathResult.from_propagation`.

Exit:

- Consumer outputs for `los`, `reflection`, and `transmission` are
  bit-identical to Phase 0.
- Every Radar test passes, including the bit-exact `echo(trace(...))` versus
  `simulate(...)` tests.
- `CONTRACT_VERSION` goes from 6 to 7 only if a consumer type changes; adding
  diffraction primal support is a capability change and is published through
  `capabilities()`.

### Phase 3 (conditional): Capability parity

Only after Phase 2, and only through its own ADR for each item:

- `scattering` in the consumer needs a power-domain response, because the
  current rows carry no coherent transport.
- Coupled reflection-diffraction in the consumer needs a pair-major
  segmentation for the streamed receiver blocks.
- The ISB taper in the consumer is a heuristic for visual continuity. It
  should stay out of the consumer unless a consumer asks for it.

Each of these lands once, in the shared engine or option owner, and becomes
reachable from both entries.

## 7. Impact

### 7.1 Radar

| Phase | Code change in Radar | What can break | Gate |
|---|---|---|---|
| 0 | none | nothing | — |
| 1 | remove `witwin.channel.path` from `SOLVER_AND_INTERNALS` in `radar/tests/test_import_boundary.py` | nothing; the entry would pass while pointing at a deleted module, so it is removed to keep the list accurate | Radar `quick` tier |
| 2 | none expected; the ten consumer names keep their signatures | consumer request validation or row order shifts; consumer starts importing a solver module at import time | full Radar test suite, including the bit-exact `echo(trace)` tests, and `test_import_boundary.py` |
| 3 | decide whether Radar exposes consumer `scattering` | the name `scattering` means environment scattering in Channel and target scattering in Radar | Radar capability record review |

The direction of imports is the main invariant. `deterministic` may import
the consumer, as `path` does today; the consumer must never import
`deterministic` at import time.

### 7.2 Public API

| Export | Today | After Phase 1 | After Phase 2 |
|---|---|---|---|
| `witwin.channel.path` (module) | public solver | deleted | — |
| `path.Config` | public | merged into `deterministic.Config` | — |
| `path.solve` | public | deleted; use `deterministic.solve(..., export_paths=True)` | — |
| `path.PathResult` | public | `deterministic.PathResult` | gains `from_propagation` |
| `path.InteractionType` | public | `deterministic.InteractionType` | — |
| `path.RaggedPathSoA` | public | private | — |
| `deterministic.Config` | 22 fields | `options: EnumeratedOptions` plus 6 terminal fields; `max_diffraction_order` deleted | — |
| `deterministic.PathTable` | public | deleted; `Result.paths` is `PathResult \| None` | — |
| `deterministic.Result`, `deterministic.solve` | public | signatures unchanged | — |
| `consumer.*` | contract v6 | unchanged | `_ConsumerConfig` deleted (private); diffraction primal works |
| Root package (`witwin.channel`) | 6 names | unchanged | unchanged |
| `capabilities()` solver list | 4 solvers | 3 solvers | — |

These changes are breaking. Under the repository rules they ship without
compatibility aliases, with a snapshot update and a migration note.

### 7.3 Files that change with the code

- `CLAUDE.md` and `AGENTS.md` (the solver owner list goes from four to three;
  both files change in the same commit).
- `FEATURE_LIST.md` and `README.md` (the solver table and the quick-start
  examples).
- `docs/dev/path/README.md` is deleted; `docs/dev/deterministic/README.md`
  absorbs its content.
- `docs/dev/ad-capability-boundary.md` (the `P` column merges into `D`).
- `docs/dev/propagation-ad-capability-matrix.md` (diffraction primal, Phase 2).
- `ci/public-api-snapshot.json`, `ci/check_import_graph.py` allowlists,
  `ci/check_orphan_modules.py` roots.
- 46 test files and 5 benchmarks that import `path`.

## 8. Decisions needed

- **D1. Which module survives.** Recommended: `deterministic`. It owns the
  native accumulation kernels, has more tests (65 files against 46) and
  benchmarks (10 against 5), and is what Plan 16 runs on. The alternative,
  `path`, matches Sionna's naming but would move the accumulation owner.
- **D2. Scene-level row table.** Recommended: `PathResult` only, with
  `PathTable` deleted. A caller who needs compact rows uses the consumer.
- **D3. Scattering in the consumer.** Recommended: keep it refused until a
  coherent or explicitly power-domain response has its own ADR (Phase 3).
  Until then, `deterministic` stays the only scattering entry.
- **D4. Consumer diffraction primal.** ADR-043 left the defect pinned so that
  a fix would be deliberate. Recommended: fix the primal in Phase 2 through
  the shared adapter, and keep diffraction AD `REF`.

## 9. Risks

- **Plan 16 is in flight on `deterministic.solve`.** Phase 1 keeps the
  `deterministic.solve` signature and every option name Plan 16 uses,
  including `isb_boundary_taper`. Only the nesting under `options` changes,
  and the full-wave validation scripts move with the same commit.
- **Existing host synchronizations on the grid path.** `deterministic`
  counts transmission and scattering rows with `.sum().item()` after the
  solve. These are recorded debt from before this plan. The merged
  solver must keep them unchanged, must not copy them into the consumer, and
  must report them in the Phase 0 baseline.
- **Row-order drift.** `max_paths_scope="global"` with `sort_key` selects
  rows differently from the per-pair cap used by `path`. Phase 1 keeps both
  scopes and must reproduce each old entry's selection exactly.
- **Hidden reliance on `PathTable` fields.** Its 24 fields include
  `field_real`, `field_imag`, and `phase_rad`, which `PathResult` does not
  carry. Phase 0 step 1 lists every reader before anything is deleted.

## 10. Validation

- Phase 0 and Phase 1: `conda run -n witwin2 python ci/run_ci_tier.py quick`
  and `cuda`, plus the row-equivalence and bit-exact reproduction tests.
- Phase 2: the same tiers in Channel, plus the full Radar suite with
  `--gpu`.
- Every phase: launch, synchronization, D2H, and peak-memory numbers compared
  with the Phase 0 baseline, recorded under `docs/dev/perf/`.
