# ADR-026: RayD ownership of generic scattering runtime operations

- **Status:** Accepted (2026-07-19); Phase 10A/10B implementation and Phase 11B
  frozen duplication acceptance complete; final release evidence pending
- **Date:** 2026-07-19
- **Kind:** Cross-repository native-owner move. This decision does not authorize
  a numerical, fusion, launch, resource-lifecycle, or public-Python-API change.
- **Related:** ADR-004 (numerical duplication), ADR-009 (native fusion
  ownership), ADR-010 (native scattering kernels), ADR-014/015 (scattering AD),
  ADR-021 (multi-bounce coherent scattering), ADR-022 (BDPT fixed-topology AD),
  ADR-023 (direct typed RayD integration), ADR-024 (shared RF ownership), and
  Plan 13 sections 7, 8, and 10.

## Context

ADR-010 replaced production Torch scattering physics with Channel-owned native
CUDA operations. ADR-014/015 added their native derivative companions, and
ADR-021 added two complete row-fused chain families. Those decisions fixed the
numerical contracts, but they did not decide the final cross-repository owner.

The resulting runtime operations are solver-neutral and consume caller-owned,
device-resident tables, phase-screen heights, geometry rows, and material
tensors. They do not build resources, choose topology, consume random numbers,
apply MIS, accumulate solver results, or own scene policy. RayD already owns the
shared RF/Jones dependency closure needed by the chain families. Keeping the
generic runtime implementations in Channel would therefore preserve an
unnecessary downstream numerical owner and keep the seven table-interpolation
helpers outside their reusable dependency owner.

At acceptance, this ADR fixed the final owner boundary without activating it.
The realized-state sections below record the completed Phase 10A and Phase 10B
switches. RayD is now the unique production numerical owner of all 17 contracts;
Channel retains the ABI/facade and explicitly retained lifecycle/policy owners.

## Decision

### Complete operation-family boundary

After the corresponding Phase 10 activation, RayD is the unique numerical
implementation owner of exactly these 17 Channel-facing contracts:

| Family | Complete contract set | Activation phase |
| --- | --- | --- |
| table evaluation AD | `scattering_table_eval`, `scattering_table_eval_backward`, `scattering_table_eval_jvp` | 10A |
| table sampling | `scattering_table_sample`, `scattering_table_pdf` | 10A |
| single-bounce ensemble | `scattering_ensemble_eval`, `scattering_ensemble_eval_backward`, `scattering_ensemble_eval_jvp` | 10A |
| phase-screen patch integral | `scattering_patch_integral_eval`, `scattering_patch_integral_eval_backward`, `scattering_patch_integral_eval_jvp` | 10A |
| chain ensemble | `scattering_chain_ensemble_eval`, `scattering_chain_ensemble_eval_backward`, `scattering_chain_ensemble_eval_jvp` | 10B |
| chain realization | `scattering_chain_realization_eval`, `scattering_chain_realization_eval_backward`, `scattering_chain_realization_eval_jvp` | 10B |

Primal, backward/VJP, and JVP companions move as complete families. Channel
continues to own the stable `_channel` symbol names and the Python domain
facades under `witwin.channel.scattering.kernels`. A Channel C++ adapter
packs named RayD requests and converts named RayD results to the existing
Python-facing dictionaries/tuples. It contains validation and packing only; it
must not reconstruct scattering math.

RayD declarations belong in a public typed header under
`backends/torch/include/rayd/torch/rf/`, included by
`rayd/torch/integration.h`. They use `at::Tensor`,
`std::optional<at::Tensor>`, scalar values, and named request/result structures.
They are a source-level interface for one CMake/LibTorch graph, not a stable
cross-build DSO ABI and not a RayD Python extension or second dispatcher.

### Frozen resident-resource and tensor ABI

RayD receives tensors by reference for the duration of each call. It neither
constructs nor retains a table, phase screen, quadrature resource, scene handle,
random seed, or solver tape. The caller remains responsible for tensor lifetime
and residency. All entries preserve the current CUDA-only shape, dtype,
contiguity, same-device, row-order, stride, empty-input, and current-stream
contracts and fail before partial computation on invalid state.

The typed requests/results preserve these exact logical field sets and their
current order at the Channel ABI adapter:

- **Table evaluation.** Primal inputs are `wi`, `wo`, `f_te`, and `f_tm`;
  outputs are `f_te` and `f_tm`. Backward adds optional output cotangents plus
  `need_grad_dirs` and `need_grad_tables`, returning optional `grad_wi`,
  `grad_wo`, `grad_f_te`, and `grad_f_tm`. JVP adds optional tangents for the
  four primal tensors and returns `tangent_f_te` and `tangent_f_tm`.
- **Table sampling.** `scattering_table_sample` consumes `wi`, `uniforms`,
  `marginal_cdf`, `conditional_cdf`, and `sample_density`, returning `wo`,
  `pdf_forward`, and `pdf_reverse`. `scattering_table_pdf` consumes `wi`, `wo`,
  `sample_density`, and `reverse`, returning one row-aligned density tensor.
  Sampling and PDFs remain fixed importance-distribution operations with no AD
  companions.
- **Single-bounce ensemble.** The primal request contains `wo_rows`, `r2_rows`,
  `cos_o_rows`, `n_o`, `t1r`, `t2r`, `wi_local`, `cos_i`, `r1`, `a_te2`,
  `a_tm2`, `weights`, `material_id`, `backup_axis`, `rx_pol`, `rc_idx`, `sc_idx`,
  `f_te_flat`, `f_tm_flat`, `table_offset`, `table_dims`, `material_slot`,
  `coef`, and `threshold`; results are `gain`, `amplitude`, `length`, and
  `keep`. Backward preserves the three optional output cotangents, four
  requested-gradient flags, and the 15 optional results `grad_wo_rows`,
  `grad_r2_rows`, `grad_cos_o_rows`, `grad_n_o`, `grad_t1r`, `grad_t2r`,
  `grad_wi_local`, `grad_cos_i`, `grad_r1`, `grad_a_te2`, `grad_a_tm2`,
  `grad_weights`, `grad_f_te`, `grad_f_tm`, and `grad_coef`. JVP preserves the
  14 optional tensor tangents plus scalar `tangent_coef` and returns tangents
  for `gain`, `amplitude`, and `length`; `keep` is non-differentiable.
- **Patch integral.** The primal request contains `patch_tris`, `patch_uvs`,
  `rows`, `d_i`, `d_o`, `n_rows`, `r_te`, `r_tm`, `pol_t`, `pol_r`, `r1_rows`,
  `r2_rows`, `centroids`, `heights`, `quad_a`, `quad_b`, `quad_w`, and `k0`;
  results are `total`, `integral`, and `row_value`. Backward preserves
  `grad_total`, the four requested-gradient flags, and optional gradients for
  `heights`, both Jones values, both directions, both distances, `centroids`,
  and `k0`. JVP preserves the eight optional tensor tangents plus scalar
  `tangent_k0` and returns `tangent_total`.
- **Chain ensemble.** The primal request preserves the current 39 tensors and
  three scalars in this order: `tx_pol`, `rx_pol`, `source`, `vertex`, `target`,
  the C1 block (`positions`, `normals`, `eps_r`, `sigma_e`, `mu_r`, `gain`,
  `thickness`, `depth`), the corresponding C2 block, `n_o`, `t1r`, `t2r`,
  `backup_axis`, `wi_local`, `cos_i`, `cos_o`, `d_i`, `d_o`, `l1`, `l2`,
  `weights`, `material_id`, `f_te_flat`, `f_tm_flat`, `table_offset`,
  `table_dims`, `material_slot`, `coef`, `threshold`, and `frequency_hz`.
  Results remain `gain`, `amplitude`, `length`, and `keep`. Backward preserves
  the three optional output cotangents, six flags, and 12 optional gradients
  for the C1/C2 electromagnetic blocks, both tables, `coef`, and `frequency`.
  JVP preserves its 21 optional tensor tangents plus scalar tangents for `coef`
  and `frequency`, returning tangents for `gain`, `amplitude`, and `length`.
- **Chain realization.** The primal request preserves its 44 tensors and two
  scalars: patch mesh/UV/rows and local directions, `n_rows`, source/vertex/
  target, the C1 and C2 padded blocks, tx/rx polarization, `l1`, `l2`, `sp1`,
  `sp2`, `centroids`, resident `heights`, `cos_spec`, material/layer CSR tensors,
  quadrature tensors, `k0`, and `frequency_hz`. Results remain `total`,
  `path_field`, `path_gain`, `integral`, and `row_value`. Backward preserves
  required `grad_total`, optional `grad_path_field`/`grad_path_gain`, seven
  flags, and its 25 optional result fields covering heights, layer parameters,
  both chain electromagnetic blocks, continuous geometry, `k0`, and frequency.
  JVP preserves 23 optional tensor tangents plus scalar tangents for `k0` and
  frequency, returning `tangent_total`, `tangent_path_field`, and
  `tangent_path_gain`.

The machine-readable parameter spelling and order remain additionally locked by
`ci/native-binding-manifest.json`; the named typed API may group a primal
request inside its backward/JVP request but may not add, remove, reorder, or
reinterpret a Channel-facing field.

### Shared scattering-table device header

RayD becomes the unique source owner, at Phase 10A activation, of a public
solver-neutral scattering-table device header under
`shared/include/rayd/shared/rf/`. It contains exactly the seven currently
audited helpers:

`positive_phi`, `linear_axis`, `nearest_axis`, `interp4`, `eval_te_tm`,
`linear_axis_grad`, and `eval_te_tm_grad`.

The normalized bodies, declaration order, `__device__ __forceinline__`
attributes, explicit `fmaf` interpolation order, horizon behavior, periodic
axis behavior, 16-corner index/weight order, and primal/dual relationship are
frozen. After Channel activates 10A, every remaining Channel consumer includes
the public RayD header and the private
`native/channel/kernels/scattering_table.cuh` is deleted. RayD never
includes a Channel-private header. `kirchhoff.cu` has no current
dependency on this header and must not gain a decorative or unused include.

### Fusion, launch, tape, reduction, and atomic contracts

The move preserves the non-empty active-call launch boundary:

| Entry kind | Raw CUDA launches |
| --- | ---: |
| table eval/backward/JVP, sample, or PDF | 1 each |
| ensemble primal/backward/JVP | 1 each |
| patch primal / backward / JVP | 2 / 1 / 2 |
| chain ensemble primal/backward/JVP | 1 each |
| chain realization primal/backward/JVP | 2 / 1 / 2 |

Patch and chain-realization primal/JVP retain their row kernel followed by the
fixed-order total reduction. No family gains a materialized inter-launch
physics intermediate, host synchronization, host/device copy, stream wait, or
persistent native tape. Torch autograd may retain the same primal tensors for
dispatch; native backward recomputes forward intermediates in primal expression
order.

`Dmax = 8`, padded C1/C2 slot validity and depth semantics, chain-1 Jones
transport, diffuse vertex, chain-2 transport, receiver projection, phase,
Jones-basis, CSR interpretation, `weights`, and output schemas remain one
complete row-fused contract. C1, scatter, and C2 may not be split into sibling
launches or persistent tensors.

Backward keeps the current direct per-row stores and shared-gradient
`atomicAdd` behavior, including table-corner, phase-height, layer, chain,
geometry, coefficient, `k0`, and frequency accumulation where currently live.
JVP and total reductions retain their deterministic fixed order and no new
atomics. Atomic nondeterminism already accepted for shared backward buffers is
not widened to a primal or JVP path.

### Family-specific geometry AD contract

The as-built chain families intentionally differ and the move preserves that
difference:

- chain-ensemble reverse-mode continuous geometry is unsupported and
  `need_grad_geometry=true` fails loudly; its forward-mode JVP accepts the
  existing geometry tangents;
- chain-realization backward and JVP both support their existing continuous
  geometry fields, including directions, C1/C2 positions and normals, lengths,
  spreading factors, and centroids.

This family-specific rule supersedes any blanket Plan-13 wording that implied
both chain families reject reverse geometry. Removing chain-realization
geometry VJP, enabling chain-ensemble geometry VJP, or changing either tangent
set is a separate numerical/AD decision, not an owner move.

All families retain the ADR-022 fixed-topology, fixed-sample,
fixed-visibility, fixed-PDF, and fixed-MIS contract. Discrete topology,
material ids, rows, sample selections, quadrature structure, masks, thresholds,
and keep decisions remain non-differentiable as currently documented.

### Translation-unit compile contract

Compile mode is owned per source translation unit and is frozen exactly:

- the table primal/sample/PDF kernels currently in `scattering.cu` use the
  target's default CUDA flags; they must not gain `--fmad=false`;
- `scattering_table_eval_ad.cu`, `scattering_ensemble.cu`,
  `scattering_ensemble_ad.cu`, `scattering_patch_integral.cu`,
  `scattering_patch_integral_ad.cu`, both chain primal TUs, and both chain AD
  TUs retain `--fmad=false`;
- the retained Channel `scattering_event_probabilities` kernel remains on its
  current default-flags Channel TU contract;
- no scattering TU inherits pure-wedge `--use_fast_math`, and no scattering
  flag is spread into transmission, coupled diffraction, table construction,
  or another family.

Explicit `fmaf` calls in the shared table helper remain explicit under either
mode. A file split required to leave event policy in Channel must prove that the
retained event kernel's generated instructions and resources are unchanged.

### Channel-owned resource, solver, and policy boundary

The following do not move:

- `scattering_event_probabilities`, because it owns MC/BDPT event-selection
  policy rather than a generic BSDF primitive;
- Kirchhoff table construction, `kirchhoff.cu`, CPU/NumPy test
  oracles, cache/version/validation, `KirchhoffTable`,
  `KirchhoffRuntimeResources`, and `KirchhoffTableStack` lifecycle;
- `PhaseScreenRuntime`, realization seed/generation, structure assignment,
  cache, and resident-height lifecycle;
- rough-reflection `C_r` composition in `propagation.fields`;
- chain discovery, join, row budgets, C1/C2 packing, and topology policy;
- deterministic coherent combine and all solver accumulation;
- BDPT continuation, NEE, MIS, RNG, event glue, and BDPT-owned AD companions;
- MC Basic's single-scatter estimator; and
- public result, capability, and metadata assembly.

RayD consumes the resident tensors opaquely. It does not acquire a Channel
scene/resource object, mutate/cache the tensors, construct a table or phase
screen, choose a random sample, or import solver policy.

### Default-off and unchanged solver behavior

The accepted ADR-021 defaults remain bitwise no-ops:

- `scattering_chain_max_depth = 0`;
- `scattering_coherent = False`;
- `max_scattering_order = 1`.

An owner move does not alter seed consumption, row discovery/selection,
reflection-only C1/C2 coverage, realization replacement, coherent combination,
MC continuation, result schema, or public metadata.

### Cross-repository activation and rollback

Activation is ordered and atomic by family group:

1. RayD merges the complete 10A typed candidate, public table header, exact
   compile flags, and direct contract tests while Channel still owns production.
2. Channel pins that reviewed commit, calls the typed 11-contract candidate,
   proves parity, changes remaining consumers to the RayD table header, and
   deletes the corresponding local source/header implementations in one 10A
   switch commit.
3. RayD then merges the complete dormant six-contract chain candidate with
   direct tests.
4. Channel pins it, switches all six contracts, proves parity, and deletes all
   four local chain TUs in one 10B switch commit.

A dormant RayD candidate is not compiled into or called by `_channel`
and is not a supported dual-owner state. Channel never pins a branch, dirty
worktree, or unmerged candidate. Rollback selects the previous accepted RayD
commit through the lock file; it never introduces a runtime feature flag,
dynamic lookup, copied implementation, or fallback.

### Phase 10A realized state

Phase 10A activated the first four complete families (eleven contracts) and the
seven shared table helpers at pushed RayD commit
`4577e744adfe8665f7817e3aff5e8e533ec896e7`. The locked typed scattering,
integration-v2, and shared table header SHA-256 values are respectively
`66d75a20be16057f03cdfb79e3b9dcc85cacec79b555cd73b019259aa510262a`,
`9f95ad9e8e3b790d00f8e762a3e6a09252d46afb65bfc3aba7c42325836cb1fb`, and
`38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38`.

Channel retains all eleven ABI symbols and typed Python facades but no longer
contains their numerical CUDA bodies or the private scattering-table header.
The retained event-policy kernel and all six Phase 10B chain contracts remain
complete Channel owners. The binding count stays 202 and the active numerical
owner split is RayD 37, layered Channel/RayD 2, and Channel 163. Exact
launch, current-stream, compile-flag, codegen/resource, direct-contract,
dependency, deletion, and no-fallback evidence is recorded in
`docs/dev/audit/phase13-scattering-phase10a-evidence.json`.

### Phase 10B realized state

Phase 10B activated the two complete fused chain families (six contracts) at
pushed RayD commit `768b96e42a95f70c32d55f98a72000085317e288`. The locked
typed scattering, integration-v2, and shared table header SHA-256 values are
respectively
`ac95c418860d109aeaa96623131592e4df8887992e5fc25ecab71b4ddbf1f55b`,
`0608bfbaf022379bc03442f9baa777ec05cfe3f6ab9b964e2385ec12a7b6c654`, and
`38ea9be424640301a88a97bccca9ab4bc599191ecfb0b259881ef6a300c96e38`.

Channel retains all six ABI symbols and typed Python/autograd facades but no
longer contains the four chain CUDA TUs. The binding count stays 202 and the
active numerical owner split is RayD 43, layered Channel/RayD 2, and Channel
Native 157. The as-built geometry AD split is unchanged: ensemble geometry is
JVP-only and rejects VJP loudly, while realization geometry supports VJP and
JVP. Exact launch/current-stream/compile-flag/codegen/resource/direct-contract,
dependency, deletion, and no-fallback evidence is recorded in
`docs/dev/audit/phase13-scattering-phase10b-evidence.json`.

The move-only implementation reduced exact-token duplication from 11.913070%
to 11.170566%, pruned all 12 stale chain-region entries, and classified the
three new typed-adapter packing regions. The frozen 10.211512% budget was not
relaxed and is still exceeded; this is an explicit Phase 11 nightly/release
acceptance blocker, not a reason to mix unrelated deduplication into Phase 10B.

Phase 11B subsequently reduced the metric to `7826/77821 = 10.056413%`, below
the unchanged `10.211512%` frozen budget, with 143 classified regions, zero
stale regions, and zero unclassified regions. This closes the duplication
blocker without changing scattering physics, signatures, launches, or numerical
order. Final clean-checkout nightly/release and wheel evidence remains tracked
separately in `docs/dev/audit/phase13-phase11-release-acceptance.json`.

## Consequences

After both activation phases, RayD is the only numerical source owner of the
17 generic runtime contracts and seven shared table helpers, while Channel
retains its stable extension ABI, domain facades, resident-resource lifecycle,
and solver/estimator policy. The wheel still contains one `_channel`
production extension and no RayD Python extension or undeclared DSO.

This decision narrows RayD's former statement that it contains no BSDFs: RayD
still provides no high-level material/BSDF framework, renderer, emitter, or
integrator, but it may own these low-level solver-neutral resident-table and
phase-integral CUDA primitives.

## Acceptance gates

Each Phase 10 switch requires, without tolerance or budget relaxation:

1. old/new exact or frozen-ULP primal parity for table boundary bins,
   sample/PDF normalization, ensemble energy/reciprocity/Jones behavior, patch
   phase convention, and chain depth 0/1/max plus single-bounce collapse;
2. backward/JVP lockstep, adjoint dot products, test-only finite differences,
   the family-specific geometry contract, and `ad_mode="none"` no-tape parity;
3. identical per-TU flags, normalized PTX/SASS or explained compiler-output
   differences, registers, stack/local/shared memory, occupancy, and atomics;
4. unchanged raw launch count, current-stream behavior, host/device-copy and
   explicit-sync count, persistent-tape bytes, materialized-intermediate bytes,
   and peak resident/temporary memory;
5. direct RayD typed tests for valid, empty, invalid shape/dtype/contiguity/
   device/optional-tensor/scalar contracts and CUDA error propagation;
6. Channel direct contract and end-to-end coverage through Path,
   Deterministic, MC Basic, and BDPT where each family is live;
7. updated binding/contract-coverage manifests, current-owner inventory,
   migration delta, helper dependency graph, duplication classification,
   launch/resource evidence, no-fallback tests, RayD lock/fingerprint,
   migration docs, and synchronized `AGENTS.md`/`CLAUDE.md`; and
8. no Channel CUDA/header duplicate, no RayD dependency on Channel private
   code, and no second extension, dispatcher, host numerical implementation,
   or fallback.

## Stop conditions

The corresponding current owner remains intact if any family is incomplete,
generated-code or output drift is unexplained, a compile flag spreads, the
event-policy or resource owner moves, chain fusion/tape/atomic/AD behavior
changes, a new launch/copy/synchronization/intermediate appears, a RayD source
needs a Channel-private header, or parity requires a fallback, detached
gradient, duplicate implementation, or relaxed gate.

Visibility-scattering fusion, cross-polar table channels, GPU table building,
new batching, CUDA Graph capture, stochastic-walk geometry AD, and any change
to the two chain families' geometry AD support require a separate accepted
numerical/fusion ADR.
