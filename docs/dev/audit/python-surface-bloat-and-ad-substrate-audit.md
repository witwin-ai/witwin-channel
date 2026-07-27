# Audit: Python Surface Bloat and the AD Substrate Trade

Status: **informational audit**. This document records measurements and analysis.
It is not an accepted ADR and mandates nothing. Four of its conclusions require
their own accepted ADRs before implementation: Class-B parameter unfreezing
(§II.3, Part IV Tier 0), higher-order derivatives (Part IV Tier 5), composable
module boundaries plus the `witwin.lab` differential layer (Part V), and the
interaction-indexed field stage (§V.7).

Two items need explicit resolution rather than reinterpretation, both against
ADR-009's fusion boundary: §V.5 R1 (modularization materializes intermediates)
and §V.7.d (per-interaction operator composition does the same).

**The highest-leverage single decision in this document is §V.7** — whether the
field stage stays indexed by path word or becomes indexed by interaction. It is
the common root of the arity explosion (§V.4), the `coupled_rd`/`coupled_dd`
special owners, and a large share of the Python bulk measured in §I.1, and it
gates Part IV Tier 1.

Date: 2026-07-27
Scope: `channel/src/witwin/channel` (49,917 lines, 196 `.py` files) plus the
native kernels it dispatches (`channel/native/channel/kernels`).

Contents:

- **Part 0** — baseline measurements and the pipeline architecture map
- **Part I** — where the Python bulk actually is (bloat, dead code, policy divergence)
- **Part II** — the AD substrate: hand-written JVP/VJP versus Dr.Jit
- **Part III** — are performance and developability incompatible?
- **Part IV** — recommended sequencing
- **Part V** — module boundaries, swappability, and the `witwin.lab` proposal
Method: static read of the package, `ci/native-binding-manifest.json`,
`ci/contract-coverage-manifest.json`, `ci/public-api-snapshot.json`, and targeted
reads of the CUDA translation units. Every claim below carries a `file:line`
anchor. Claims marked *(inferred)* were not confirmed against a running build.

---

## Part 0 — Baseline

| Region | Lines | Share |
| --- | --- | --- |
| Domain `kernels/` packages (native binding layer) | ~12,600 | 25% |
| `montecarlo/` (basic + bdpt + events) | ~11,600 | 23% |
| `propagation/` excluding `kernels/` | ~9,000 | 18% |
| `propagation/consumer/` (ADR-037…042 accretion) | 4,528 | 9% |
| Remainder (`scene`, `scattering`, `runtime`, four solvers) | ~12,000 | 24% |

The headline number is misleading: a large fraction is mechanically generated-
looking code that was typed by hand. Part I quantifies that fraction.

## 0.2 Pipeline architecture (enumerated solvers)

Stage order verified from `propagation/enumerated/engine.py:204` and its imports.

```
  scene.compile(reference_frequency_hz=...)
        └──▶ CompiledScene  [RayD SceneResource · material stores · resident tables]
                             built once per scene; not on the gradient path

════════════ DISCRETE STAGE (upstream) ══════════════════════════
  _los_topology / _reflection_topology_multibounce /
  _diffraction_topology_order1 / _transmission_topology / coupled_*
                          ↓
              PathTopology   depth · component_id · primitive_id · edge_id
                             material_id · primitive_sequence · material_sequence
                             interaction_type · valid · tx_id · rx_id
                             ── all int32/bool, zero gradient ──
                          ↓
              canonical_compact (ADR-032)  →  compact K rows + stable pair segmentation
                          ↓                    6 × 4-byte D2H, one synchronization

════════════ CONTINUOUS STAGE ═══════════════════════════════════
  evaluate_path_fields()
   │
   ├─ geometry   rayd_reflection_epc_paths_ad / rayd_face_normals_ad
   │             ↑ holds SceneResource ── NOT swappable
   │                  ↓
   │             PathGeometry   interaction_positions · interaction_normals
   │                            path_length · delay · direction
   │                  ↓
   └─ fields     field_free_space(6) / field_reflection_sequence(13)
                 field_transmission_sequence(17) / field_diffraction_wedge(22)
                 ↑ plain tensors only ── swappable
                      ↓
                 PathFields   field_vector · coefficient · path_field · path_gain

════════════ REDUCTION ══════════════════════════════════════════
                 accumulation → PathResult / PropagationEvaluation
```

| Stage | Owner | Output contract | Differentiable | Holds native state | Boundary arity |
| --- | --- | --- | --- | --- | --- |
| compile | `scene/compiler.py` | `CompiledScene` | n/a (once per scene) | yes | n/a |
| topology discovery | `propagation/topology`, `enumerated/*_topology` | `PathTopology` | **no** (int32/bool) | yes | n/a |
| compaction | `topology/kernels/canonical_compact.py` | compact rows + pair segmentation | no | no | n/a |
| geometry | `propagation/geometry` | `PathGeometry` | yes (vertices, endpoints) | **yes** | n/a |
| fields | `propagation/fields` | `PathFields` | yes (see II.3) | **no** | 6 / 13 / 17 / 22 |
| accumulation | `path`, `deterministic`, `consumer` | `PathResult`, `PropagationEvaluation` | yes | no | n/a |

Part V analyses these boundaries against the gsplat reference structure and
against the proposed `witwin.lab` composition layer.

---

# Part I — Where the bulk actually is

## I.1 The five-artifact template, instantiated 47 times

Every differentiable native op is expressed as **five separate hand-written
Python artifacts**, not three:

| Artifact | Example | Location |
| --- | --- | --- |
| primal facade | `field_free_space` | `propagation/fields/kernels/functional.py:40` |
| VJP facade | `field_free_space_backward` | `functional.py:532` |
| JVP facade | `field_free_space_jvp` | `functional.py:571` |
| `autograd.Function` (4 methods) | `_FieldFreeSpaceAdFunction` | `fields/kernels/autograd.py:35` |
| `_ad` wrapper | `field_free_space_ad` | `autograd.py:225` |

The package contains **47 `torch.autograd.Function` subclasses** across 27 files.

### Measured plumbing share

Counting lines consisting of nothing but one identifier and a comma (a parameter
declaration, or an argument being forwarded):

| File | Total | Plumbing | % |
| --- | --- | --- | --- |
| `propagation/fields/kernels/functional.py` | 1144 | 568 | **49%** |
| `propagation/fields/kernels/autograd.py` | 1790 | 699 | 39% |
| `propagation/geometry/kernels/bridge.py` | 792 | 202 | 25% |
| `propagation/geometry/kernels/autograd.py` | 1248 | 442 | 35% |
| `scattering/kernels/functional.py` | 759 | 381 | **50%** |
| `scattering/kernels/autograd.py` | 825 | 336 | 40% |
| `scattering/kernels/functional_chain.py` | 932 | 397 | 42% |
| `scattering/kernels/autograd_chain.py` | 1130 | 525 | **46%** |
| `materials/kernels/functional.py` | 268 | 94 | 35% |
| `materials/kernels/autograd.py` | 220 | 81 | 36% |
| **Total** | **9108** | **3725** | **41%** |

In `fields/kernels/autograd.py`, the six `Function` classes plus their wrappers
occupy ~1,740 of 1,790 lines (97%). Genuinely per-op information is the input-name
list, the fixed-index list, and the output-field tuple — roughly 20–30 declarative
lines per Function, ~150 lines total. **The other ~1,590 lines (89%) are the same
template re-typed.**

`setup_context` is the clearest case: semantically identical across all six
Functions, differing only in an index — and written in **two different styles in
the same file** (destructuring-by-name at `autograd.py:68`, indexing-by-position
at `autograd.py:299`). That is a copy-paste artifact, not a design choice.

### The worst instance

`scattering_chain_ensemble_eval` takes **40 positional tensor parameters**. That
40-name list appears verbatim four times:

- `scattering/kernels/functional_chain.py:206` (primal signature)
- `functional_chain.py:335` (backward signature)
- `functional_chain.py:439` (jvp signature)
- `functional_chain.py:96` (again, as the string tuple `_CHAIN_ENSEMBLE_PRIMAL_NAMES`)

The fourth copy exists to keep the other three in sync:

```python
# functional_chain.py:118
def _ordered_primal_args(scope: dict[str, object], names: tuple[str, ...]):
    """Pack caller locals in the frozen typed native ABI order."""
    return tuple(scope[name] for name in names)
# called as _ordered_primal_args(locals(), _CHAIN_ENSEMBLE_PRIMAL_NAMES) at :320, :414, :519
```

**This converts an import-time error into a runtime `KeyError`.** Renaming a
parameter produces no type error; it fails mid-solve. `scattering_chain_realization_eval`
repeats the pattern with a 42-name list (`_CHAIN_REALIZATION_PRIMAL_NAMES`, `:106`).

### Do the helpers exist, and are they used?

Yes, and mostly yes — but they abstract the wrong layer.

- `runtime/autograd_contracts.py` (283 lines, 18 `_ad_*` helpers) is imported by
  **32 of the 33 files** defining an `autograd.Function`. Only
  `propagation/topology/kernels/compaction.py` holds out.
- `runtime/tensor_contracts.py` is **32 lines total** — one function,
  `validate_cuda_tensor`, called **~700 times** across 36 files.

Both compress the *innermost* statements. **Nothing abstracts the shape** — the
five-artifact layout, the `setup_context` body, the `backward` need-flag /
positional-tuple dance, the `_ad` wrapper. That is where the ~3,700 plumbing
lines sit.

Where a helper **is** bypassed: `runtime/symbols.py:33` provides
`required_symbol`, yet **11 files hand-roll the identical lookup at 37 sites**.
`montecarlo/basic/kernels/maps.py` uses both forms twelve lines apart
(`:252` uses the helper, `:302` hand-rolls it).

## I.2 Cross-module clones

**`mc_` ↔ `bdpt_` maps facades.** Nine function names are shared between
`montecarlo/basic/kernels/maps.py` and `montecarlo/bdpt/kernels/maps.py`,
differing only by prefix. After prefix normalization, **140 of 200 lines are
identical (70%)**:

| Function | Identical after normalization |
| --- | --- |
| `los_component_maps_from_matrix` | 90% |
| `point_component_power` | 83% |
| `los_visibility_inputs` | 76% |
| `store_scaled_component_map` | 74% |
| `component_map_buffer` | 73% |
| `apply_los_visibility` | 71% |
| `store_component_map` | 60% |
| `finalize_component_maps` | 51% |
| `los_component_maps` | 50% |

The residual 30% divergence is **not semantic** — the `mc_` copy hand-rolls the
symbol lookup while the `bdpt_` copy uses `_required_native_op`, and the `bdpt_`
copy adds one output validation the `mc_` copy lacks. **The two copies have
drifted, and the drift is accidental.**

**`bridge.py` `*args: object` facades.** The same 14 lines, differing only in a
string literal, appear at `geometry/kernels/bridge.py:472, :488, :504, :644` and
again at `geometry/kernels/autograd.py:154`. They take `*args: object` (validating
no contract) and return `tuple[torch.Tensor, ...]` (converting to no typed
contract) — two of CLAUDE.md's four kernel-facade clauses are simply not
implemented. `coupled_rd_geometry_forward` (`bridge.py:520`) and
`coupled_dd_geometry_forward` (`bridge.py:582`) are 60-line near-clones sharing
~45 identical lines.

## I.3 Dead code — ADR-029/030 residue (~2,000 lines)

Verified by grepping all of `src/` for each symbol and confirming reachability
from the four solver entry points and `propagation/consumer`.

| Module / symbol | Lines | Status |
| --- | --- | --- |
| `propagation/enumerated/canonical_capacity.py` | 258 | **Dead** — self-referential only; docstring says "Dormant" |
| `path/capacity.py` | 284 | **Dead** — zero importers |
| `deterministic/capacity.py` | 315 | **Dead** — zero importers |
| `deterministic/kernels/diffraction_pair.py` | 366 | **Dead** — not even in its own `kernels/__init__.py` (ADR-030) |
| `montecarlo/bdpt/subpaths.py` | 50 | **Dead** — zero references repo-wide; owns a live ABI symbol |
| `enumerated/capacity.py` → `evaluated_paths_capacity_pack` | ~250 | **Dead**; sibling `sanitize_enumerated_capacity_transaction` (`:528`) is **live** |
| `topology/kernels/compaction.py` → 3 functions | ~200 | **Dead**: `deterministic_capacity_finalize` (`:899`), `enumerated_canonical_capacity_select` (`:829`), `deterministic_diffraction_order1_capacity_block` (`:602`) |
| `models/capacity.py` → 5 dataclasses | ~200 | **Dead** — referenced only by the above |
| `runtime/symbols.py` → `optional_symbol`, `has_symbol` | — | **Dead** — zero production callers |
| `deterministic/accumulation.py:49` `empty_field_like_power` | — | **Dead** |

Live and correctly wired, for contrast: `topology/kernels/canonical_compact.py`
(the ADR-032 production owner), `runtime/capacity.py`,
`montecarlo/basic/kernels/capacity.py`, and `models/capacity.py`'s
`CapacityExecutionCounts` / `_require_cuda_tensor`.

### CLAUDE.md's claim about this debt is inaccurate

CLAUDE.md states these artifacts "create no coverage, manifest, or release
requirement." In fact:

- `ci/native-binding-manifest.json` — 15 dormant symbols registered
  (`:3558, 3699, 4860, 5116, 5264, 6481, 7323, 7597, 7689, 7781, 8037, 8124, 15021, 15241, 15363`)
- `ci/contract-coverage-manifest.json` — 13 entries tagged `dormant_native_call_site`
- 7 dedicated test files

They **are** correctly absent from `ci/public-api-snapshot.json` and
`capabilities.py`, and the `dormant_native_call_site` tag is an honest
classification. This is **tracked debt, larger than the guide admits** — not a
hidden violation. The CLAUDE.md sentence should be corrected.

No ADR-031 (`Qr`, per-pair raw reflection EPC) Python artifact remains; that one
was fully removed.

## I.4 Fallback and backward-compatibility divergence

The policy holds better than expected in one important respect: **zero
`try/except ImportError` anywhere in the package**, and every `hasattr` /
`available` probe raises rather than selecting a second path (~40 sites).

Genuine divergences:

| # | Finding | Location |
| --- | --- | --- |
| 1 | Requested `transmission`/`scattering` components return `torch.zeros_like` as a *successful* result for point receivers | `montecarlo/basic/pipeline.py:449`, same shape at `:398, :403` |
| 2 | A second compaction pass existing solely to repair a self-described "legacy incoherent scattering owner" ordering — **live on the Path solver entry point** | `path/pipeline.py:162` |
| 3 | `kirchhoff_tables` property, docstring: "Compatibility view" | `scene/compiled.py:204` |
| 4 | `getattr(value, "require_resource", None)` — an object lacking the attribute is **passed through untyped** to the ABI rather than rejected | `runtime/native_resources.py:11` |
| 5 | Two live `transmitter_polarizations` owners with divergent bodies (one casts dtype and calls `.contiguous()`, the other does neither) | `field_state.py:67` vs `scene/tensors.py:110` |
| 6 | `# noqa: F401 - legacy reachable global` — an unused import kept so a name stays reachable | `montecarlo/events/scattering.py:82` |
| 7 | Stale READMEs describing deleted compatibility re-export layers | `materials/README.md:17`, `runtime/README.md:53` |
| 8 | Vestigial `_DEFAULT_BUILD_INFO` — unconditionally overwritten, unreachable in a degraded state | `runtime/extension.py:28` |

`scene/tensors.py:131` carries a comment admitting finding #5 is an unfinished
ownership migration.

## I.5 Production Torch/CPU physics — the policy's largest breach

**`propagation/enumerated/scattering.py` — `realization_coherent`.** Reachable
from `path/pipeline.py:155` and `deterministic/pipeline.py:192` whenever the
config includes `scattering`:

```python
r_min = float(endpoint_r.min().clamp_min(1.0e-3))          # :550  D2H
for local in range(resource.face_count):                    # :552  one D2H per face
    edge = float(torch.linalg.vector_norm(...).max())       # :553
patch_normal = _unit(torch.cross(...))                      # :568  normals in Torch
for tx_index in range(int(tx_positions.shape[0])):          # :580  host loop
    side  = torch.sign((wi_w * patch_normal).sum(-1))       # :590  incidence in Torch
    if not bool(front.any()): continue                      # :603  implicit sync per (face, tx)
    for rx_index in range(int(rx_positions.shape[0])):      # :605  nested host loop
```

The module docstring at `:8` labels this mode **"(reference)"**. CLAUDE.md
requires reference implementations to live under `tests/` as independent oracles
and forbids production packages from dispatching to them.

**`propagation/enumerated/scattering_chain*.py`** repeats the shape:
`scattering_chain.py:411` (`torch.cross` areas), `:506` (per-bounce Python loop),
`scattering_chain_append.py:125` (transverse basis via `torch.cross`), `:220`
(`for tx_index in torch.unique(tx_id).tolist()` — a D2H-driven host loop over
transmitters). Default-off (`scattering_chain_max_depth >= 1`), which limits
blast radius but not the policy breach.

**`scene/antenna.py` — on the Path solver main path** via `path/arrays.py:8`:

| Line | Content |
| --- | --- |
| `:58` | direction normalization via `torch.linalg.vector_norm` |
| `:68` | antenna pattern evaluation (`torch.sqrt(torch.clamp(1 - z², 0))`) |
| `:107` | steering phase `torch.einsum("...c,ac->...a", unit, positions)` |
| `:129` | precoding/combining `torch.einsum("...rt,t,r->...", coeffs, tx, rx.conj())` |

This is antenna-response physics in Torch, not structural packing.

**Documented deliberate exceptions** (not covered by any ADR found, but honestly
justified in-code): `montecarlo/basic/rayd_components.py:857` keeps a
per-transmitter `.item()` sync to preserve a double-rounded fill value bitwise.

**Accepted/benign**: offline table construction (`scattering/tables.py`,
`phase_screen.py`, `materials/evaluation.py`); metadata-only host reads in the
solver pipelines; the consumer's self-reported ADR-037/041 D2H budget
(`consumer/_prepared.py:75, :116`).

**Borderline, needs an owner decision**: `montecarlo/events/scattering.py:289`
`te_tm_incident_power` is polarization-basis physics, not "event glue".

## I.6 Consumer layer — two parallel replay pipelines

`service.reevaluate` branches at `:986`:

- **Path A** (`PreparedFixedTopology`): `_rows.prepared_row_gather` →
  `_fixed_reflection.evaluate_prepared` → `_reevaluate_prepared` (`service.py:857`)
- **Path B** (raw `PropagationTopology`, LoS only): `_fixed_los.fixed_los_gather`
  → an **inlined** replay at `service.py:990-1084`

Path B re-implements what Path A's `depth == 0` bucket already does:

| Duplicated element | Path A | Path B |
| --- | --- | --- |
| `ad_mode` → field-op selection | `_fixed_reflection.py:323` | `service.py:1000` |
| ADR-038 liveness rule | `_fixed_reflection.GeometryLiveness:55` | `_fixed_los.fixed_los_geometry_live:169` |
| Transport construction | `service.py:_fixed_transport:813` | `service.py:1045` |
| `FixedTopologyEvaluation` assembly | `service.py:946-963` | `service.py:1068-1084` |

**~95 lines of `service.py` are a second copy of a pipeline `_fixed_reflection.py`
already implements.**

Separately, `consumer/contracts.py` at 1,103 lines is a grab-bag: the wire
contracts (appropriate) plus the `_require_tensor` validator (duplicating
`models/topology.py:10`), the capabilities table (`:661-787`), three vocabulary
helpers (`:237`), and two slot-arithmetic helpers (`:890`). ~250 lines are not
contracts. The ADR-042 split into `_wideband.py` / `_prepared.py` addressed the
symptom; this root remains.

## I.7 Contract proliferation

~48 typed contracts for one subsystem (18 in `propagation/models/`, 17 in
`consumer/contracts.py`, 9 in the consumer's private modules).

Near-identical pairs:

- **`PathTopology` ↔ `PropagationTopology`** (`models/topology.py:42` vs
  `contracts.py:304`) — 8 shared field names, structurally identical
  `__post_init__`; differ only in `valid/tx_id/rx_id` vs
  `source_index/sink_index/source_id/sink_id`.
- **`PathGeometry` ↔ `PropagationGeometry`** (`models/geometry.py:13` vs
  `contracts.py:391`) — 5 shared core fields; `interaction_positions` renamed
  `interaction_positions_m`. `PathGeometry` additionally carries
  `interaction_position`/`interaction_normal`, which are `[:, 0]` slices of data
  already present.
- **`_require_tensor` defined three times**: `models/topology.py:10`,
  `contracts.py:79`, and near-clone `_require_cuda_tensor` at `models/capacity.py:25`.
- The `row_count` + `device` property pair is repeated verbatim in ~20 contracts.

Contracts with a producer and **zero** consumers: `SegmentPenetrationBackwardResult`
and `SegmentPenetrationJvpResult` (constructed by positional splat at
`bridge.py:315, :371`, never field-accessed), `CoupledCandidateCapacity`,
`CanonicalEvaluatedPaths`.

---

# Part II — The AD substrate: hand-written JVP/VJP vs Dr.Jit

## II.1 The mechanism difference

In Dr.Jit, AD is a **property of the trace**: primitives are recorded and
derivatives are derived mechanically. Here, AD is a **hand-authored artifact per
fused op**: `field_reflection_sequence` covers an entire multi-bounce transport
chain in one launch, so its adjoint must be derived by hand in CUDA and glued by
hand in Python.

**Framing note.** This is not "a compromise made because Dr.Jit was unavailable."
The rendering community reached the same conclusion from the Dr.Jit side —
Mitsuba 3's hot path abandoned naive trace AD for radiative backpropagation
(Nimier-David 2020) and path replay backpropagation (Vicini 2021), i.e.
hand-derived adjoint integrators. Channel starts where that line of work ended.

## II.2 Disadvantages

| Dimension | Dr.Jit JIT-AD | Hand-written JVP/VJP |
| --- | --- | --- |
| Marginal cost of a new differentiable parameter | ~0 (mark it) | CUDA ×1–3 sites + Python ×9 sites |
| Gradient w.r.t. an intermediate | anywhere in the trace | **impossible** — the fusion boundary is the differentiation boundary |
| Derivative order | re-traceable, higher order possible | **strictly first order** |
| primal/JVP/VJP consistency | structural | manual, via the ADR-004 duplication ledger |
| When differentiability is decided | per-trace, dynamic | `_ad_geometry_live()` at wrapper time, threaded in as a `bool` |

**Order is a hard wall.** The package has **49 `@once_differentiable`
decorators**, and `runtime/autograd_contracts.py:19` explicitly refuses composed
transforms:

```python
raise NotImplementedError(
    "rayd_*_ad entry points support a single forward-mode transform level;"
    " composed functorch transforms (e.g. torch.func.grad over forward-mode jvp)"
    " are not supported by the native geometry kernels (first-order only)")
```

No HVP, no Gauss-Newton, no second-order scene optimization, no meta-learning
through the solver. The surrounding comment notes the more dangerous property:
without this refusal, unwrapping would **silently return exact zeros**.

## II.3 Parameter differentiability — three distinct classes

### Class A — genuinely hard (research problem)

**Topology is discrete and frozen.** `scene/ad_geometry.py` docstring:

> "Topology discovery stays native and detached: RayD finds the winner (face
> sequence, validity, visibility) and channel freezes it."

Combined with ADR-037's "a row can die through `row_valid`, a row is never born",
the published geometry derivative is the derivative **under a fixed path set**.
Any perturbation that would create or destroy a path yields a discontinuous
result whose gradient does not see the discontinuity.

This is the root of the known radiomap FD-validation gap. **It is orthogonal to
the AD mechanism** — Dr.Jit does not solve visibility discontinuities either;
that requires reparameterization (Loubet et al. 2019) or warped-area sampling
(Bangaru et al. 2020).

### Class B — architecturally frozen, mathematically trivial

`_ad_reject_fixed_inputs` (`runtime/autograd_contracts.py:194`) refuses gradients
on `tx_power`, `tx_polarization`, `rx_polarization`, `mu_r`, material ids, and
valid masks, citing "the plan 07 fixed-topology contract."

Reading the kernel shows the machinery is **already present and deliberately
switched off**:

```cuda
// native/channel/kernels/field_transport_reflection.cu:474  (JVP section)
prepare(incident_pre, ad::df3_const(load3f(tx_polarization, index)));
// :576
project(final_direction, ad::df3_const(load3f(rx_polarization, index)));
```

`df3_const` constructs a dual with a **zero tangent**. The kernel already runs a
dual-number forward-AD framework; the polarizations are actively injected as
constants.

```cuda
// :123 — the entirety of tx_power's role
chain.amplitude_scale = sqrtf(fmaxf(tx_power[index], 0.0f));
```

A scalar multiply: `d(field)/dP = field / (2P)`.

```cuda
// :501 — mu_r already sits alongside the parameters that ARE differentiated
ad::slab_fresnel_dual(frame.cos_theta.v, eps_r[i], sigma_e[i],
                      mu_r[i], gain[i], thickness[i], frequency_hz, ...);
```

`eps_r`, `sigma_e`, `gain`, `thickness` are all differentiated through this same
call; `mu_r` is simply never seeded.

**Conclusion: the field is linear in `sqrt(tx_power)`, bilinear in the two
polarizations, and `mu_r` is in the same class as `eps_r`. These four are frozen
by a scoping decision, not by mathematics.** Unfreezing each is on the order of
10–30 lines of CUDA plus a seed vector. *(The bilinearity claim is read off the
kernel's structure at `:52`, `:107`, `:474`, `:576`; it was not verified against
a running gradient check.)*

### Class C — structural gaps

| Limit | Location | Cause |
| --- | --- | --- |
| Coupled RD/DD paths reject mesh-vertex gradients | `propagation/fields/evaluation.py:662` | `field_wedge_ad_coupled.cu` freezes normals, edge dirs, `n0`/`nn` via `dual_const3` |
| Grid receiver positions non-differentiable | `scene/ad_geometry.py:receiver_positions_ad` | no per-receiver tensor exists |
| Frequency is scalar-only | `_ad_frequency_value` → `float(...)`, one D2H per solve | frequency tangent is a Python float |
| Array steering frequency derivative is **exactly zero** | `scene/antenna.py:93` `float(frequency_hz.detach())` | "plan 07 AD-1 fixed-array contract" |
| `polarization_basis` non-differentiable | `consumer/service.py:_require_polarimetric_inputs` | ADR-036 frozen world-referenced basis |

The last two matter for the wideband direction: ADR-042 added
`frequency_offsets_hz`, but the frequency derivative is detached on the array side.

**Worth crediting**: every one of these is a named `NotImplementedError`. None
silently returns zero.

## II.4 Advantages

### (a) Memory: O(inputs) vs O(trace), and the former is near-free

`_FieldReflectionSequenceAdFunction.setup_context` (`autograd.py:300`) saves
**only the 12 input tensors — no intermediates**. `save_for_backward` stores
references to tensors the caller already owns, so the marginal tape cost is
approximately zero.

The adjoint recomputes instead. Dr.Jit reverse mode must record the whole trace;
a slab Fresnel evaluation (complex sqrt, exponentials, complex division) carries
tens of intermediates, multiplied by K paths × D bounces. At K=10⁶, D=3 that is
hundreds of MB of **pure new allocation** for the Fresnel portion alone.
*(Order-of-magnitude estimate from flop structure, not measured.)* This is
precisely why PRB exists.

### (b) Sparse differentiation: compute ∝ parameters actually requested

`field_transport_reflection.cu:283-320` is a **mixed-mode adjoint**: reverse to
the Fresnel coefficients, then a **one-hot forward dual seed per requested
parameter** to convert the cotangent into input gradients.

```cuda
if (grad_eps_r != nullptr) {
    ad::slab_fresnel_dual(cos_theta, b_eps, b_sigma, b_mu, b_gain, b_thickness,
                          frequency_hz, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,   // eps seed
                          r_te_dual, r_tm_dual);
    grad_eps_r[i] = adj_dot(g_r_te, r_te_dual.d) + adj_dot(g_r_tm, r_tm_dual.d);
}
if (grad_sigma_e != nullptr) { /* seed (0,0,1,0,0,0) */ }
// ... gain, thickness, frequency
```

Asking for one gradient costs one dual evaluation; asking for six costs six. The
switch propagates from Python's `need_grad_*` flags. **Trace-level AD cannot
reach this granularity** — graph pruning removes unused branches but cannot enter
a fused expression.

Same idea at the output side: `mark_non_differentiable(output[4], output[5],
output[6])` when `geometry_live` is false, so a materials-only graph pays nothing
for geometry adjoints (`autograd.py:317`).

**Net trade: compute ∝ #requested params with O(1) memory, versus O(1) compute
with memory ∝ trace depth × path count.** At RF path scales this strongly favors
the former.

### (c) Numerical determinism

ADR-009 ranks *tape lifetime* and *numerical evaluation order* as ownership
criteria #3 and #4. CLAUDE.md requires reductions "with fixed owners in ascending
pair/slot order."

Dr.Jit reverse-mode scatter-add uses atomics: **reduction order is
nondeterministic, so the same input yields bitwise-different gradients across
runs.** Irrelevant for rendering; not irrelevant here — FD gradient validation
presupposes reproducible gradients, otherwise a discrepancy cannot be attributed
to an implementation bug rather than reduction noise.
`montecarlo/basic/rayd_components.py:857` is the extreme expression of this:
a deliberate host sync to preserve double-rounding bitwise.

### (d) Silently-zero gradients are impossible

In Dr.Jit, an upstream `dr.detach()`, a non-differentiable primitive, or an
unmarked parameter all produce zero gradients with no signal. There is no place
to hang a check.

Because differentiability is hand-declared per op, the package knows what it can
differentiate and refuses the rest **by name**:

```python
# propagation/fields/evaluation.py:662
raise NotImplementedError(
    "coupled reflection-diffraction and double-diffraction paths do not support "
    "mesh vertex gradients: ... d(coupled)/d(vertices) would be silently missing.")
```

For a physics package this is arguably the single largest correctness advantage:
**a silently-zero gradient is indistinguishable from convergence.** The
`NotImplementedError` inventory in §II.3 reads as a feature-gap list; it is
better read as a **capability contract** that trace-based AD has no way to express.

### (e) No JIT — and "which parameters are differentiable" *is* the trace shape

Dr.Jit recompiles when the trace shape changes. In an optimization loop, toggling
`requires_grad`, enabling a component, or changing depth each produces a new
megakernel. Channel ships AOT-compiled SASS (the prebuild matrix mandates native
SM87), giving predictable first-iteration latency and **zero compile cost for
switching the differentiable parameter set**. The SM87 requirement also implies
edge deployment, where shipping an LLVM JIT runtime is a separate problem.

### (f) One tensor type across the platform

Dr.Jit tensors are not Torch tensors; `dr.wrap_ad` materializes at the boundary
and loses fusion. Channel's outputs feed Torch optimizers, networks, DDP,
checkpointing, and `torch.compile` directly. With `maxwell` and `radar` also on
Torch — and the planned T-matrix flow from Maxwell FDFD/FDTD into Channel
scattering — a single tensor type across the monorepo is a real asset.

### (g) Controlled fusion boundary and profilability

ADR-009: a native op "may intentionally fuse validation, geometry, field
evaluation, **AD recomputation**, reduction, and packing," and refactors must not
add launches, synchronizations, or materialized intermediates. The whole
reflection chain stays in registers. Dr.Jit megakernels also fuse, but under the
scheduler's control and with synthetic kernel names that are hard to map back to
source in Nsight.

## II.5 Where Dr.Jit wins back

Prototyping (change the physics without touching CUDA); parameter-count explosion
(zero marginal cost per differentiable quantity); intermediate gradients; higher
order.

Reasonable positioning: **Dr.Jit suits the "still searching for the physical
model" phase; this architecture suits the "physics is settled, make it a
reproducible research instrument" phase.**

---

# Part III — Are performance and developability incompatible?

**No.** They are coupled through one specific mechanism, and most of this
codebase's pain is not in that mechanism.

## III.1 Three things conflated under one label

1. **The fusion boundary — a genuine trade.** A fused kernel is fast, and its
   derivative must be hand-written. Larger fusion → better performance, coarser
   differentiation granularity, more hand-written adjoint. Real Pareto frontier.

2. **Boilerplate around the boundary — not a trade at all.** The 41% plumbing,
   the 47 copied templates, the `locals()` scrape. **These buy exactly zero
   performance.** Deleting them slows no kernel by one nanosecond.

3. **The numerical duplication ledger — a partial trade, and the source of the
   confusion.** ADR-009 says:

   > "**Numerical expressions** in primal/JVP/VJP paths remain duplicated by
   > default. They may be deduplicated only in a separate numerical change that
   > proves output exactness, unchanged evaluation order and inline attributes,
   > relevant **PTX/SASS parity**, and no performance regression."

   "Numerical expressions", "PTX/SASS parity" — **this rule is about CUDA and
   only CUDA.** It does not, and could not, require the Python dispatch layer to
   be triplicated.

**Primary diagnosis: ADR-004/009's "duplicate by default" discipline is correct
for CUDA and has leaked into Python, where it is pure cost. The leak is cultural,
not mandated by any accepted ADR.**

## III.2 Three-layer model

| Layer | Performance content | Correct strategy | Current state |
| --- | --- | --- | --- |
| **L1 · CUDA numerics** | decisive | hand-written, duplicate by default, SASS evidence | correct (the one-hot dual seed pattern is good) |
| **L2 · ABI / dispatch / AD registration** | **zero** — host code, once per launch | declarative, single source of truth | ~5,000 lines of boilerplate |
| **L3 · Contracts / orchestration** | **zero** | ordinary software engineering | 48 contracts, 2 parallel replay pipelines |

~90% of the pain is in L2 and L3, which have **no performance content**.
The project effectively pays for fusion three times: once for its real cost, twice
more by applying L1's discipline to L2 and L3.

## III.3 The codebase supplies evidence in both directions

**Abstraction at the right layer is free.** `runtime/tensor_contracts.py` is 32
lines and one function, called ~700 times across 36 files, at zero runtime cost.
`runtime/autograd_contracts.py` is 283 lines adopted by 32/33 Function-defining
files. Both eliminated thousands of lines without touching a kernel.

**Developability was sacrificed for nothing.** `_ordered_primal_args(locals(),
NAMES)` exists only to keep four copies of a 40-name list in sync, and it converts
an import-time error into a mid-solve `KeyError`. Strictly worse developability,
zero performance gained. Likewise the 37 hand-rolled symbol lookups that bypass
`required_symbol` — and which have already drifted (§I.2).

## III.4 The genuinely incompatible part

Only two things:

- **Intermediate gradients.** The fusion boundary forecloses them.
- **Higher-order derivatives.** 49 `once_differentiable`.

Both are **capability vs performance**, not **developability vs performance**.
Removing L2's 5,000 boilerplate lines neither worsens nor improves them.

## III.5 Cost model

Unfreezing one Class-B parameter today touches:

| Layer | Edit sites |
| --- | --- |
| CUDA | primal (maybe), backward, jvp — **2–3** |
| Python | 3 facade signatures, `Function.forward`, `setup_context`, `backward` (need flag + return index), `jvp`, `_ad` wrapper, `_ad_reject_fixed_inputs` tuple — **~9** |
| Other | binding manifest, contract-coverage manifest, tests |

**~15 edit sites, 9 of them with zero performance content.** With declarative
registration: ~4 (CUDA 3 + one declaration).

The compounding effect matters more than the line count: **because unfreezing a
parameter costs 15 edits, the frozen-input table never shrinks. The developability
debt has converted itself into a capability limit.**

## III.6 Where the two align

Some performance discipline actively *helps* developability:

- **Bitwise determinism** makes FD gradient validation meaningful.
- **Fail-loud capability contracts** exist *because* AD is hand-declared, and they
  prevent weeks lost to a gradient that was never connected.

Performance discipline does not inherently damage developability. The damage comes
from over-generalizing "do not deduplicate" from numerical expressions to all code.

---

# Part IV — Recommended sequencing

Ordered by risk × scope, not by size of payoff.

### Tier 0 — Unfreeze Class-B parameters (needs an ADR)

Start with `tx_power`: `d/dP = field/(2P)`, primal already computed, adjoint is one
scaling. Then the two polarizations (swap `df3_const` → a seeded dual; the
framework is present) and `mu_r` (add a seed to `slab_fresnel_dual`). ~10–30 lines
of CUDA each.

Value beyond the parameters themselves: it measures the true cost of an extension
and validates both the L1 dual wiring and the L2 registration design. If the
second parameter still costs 15 edits, the Tier-1 design is wrong.

### Tier 1 — Declarative registration for L2

Replace the 47 hand-written `autograd.Function` classes with a registry keyed on
(input names, fixed indices, output fields). No numerical change; no kernel
touched; ADR-009's duplication rule does not apply (it governs numerical
expressions). Expected reduction ~5,000 lines, and the Python side of a new
parameter drops from 9 edits to 1.

Risk to manage: a registry can hide ABI argument order. Mitigation — the registry
*becomes* the single source of that order, which is strictly better than four
hand-synced copies plus a `locals()` scrape.

**Ordering constraint (added after Part V):** §V.7's word-indexed vs
interaction-indexed decision must be settled **before** this tier starts. A
registry built over the current 47 word-indexed Functions would be built over a
set that interaction-indexing merges, and the registry would then be paid for
twice.

### Tier 2 — Delete dead code (~2,000 lines)

The §I.3 inventory, together with its 15 binding-manifest entries, 13 coverage
rows, and 7 test files. Pure subtraction. Correct the CLAUDE.md sentence claiming
these carry no manifest requirement.

### Tier 3 — Resolve the L1 policy breaches (needs a decision, possibly an ADR)

`enumerated/scattering.py:realization_coherent` and `scene/antenna.py` run Torch
physics and host loops on production paths. Two options: move them into native
kernels, or move them to `tests/` as oracles and make the production path fail
loud. **These positions currently get neither Dr.Jit's expressiveness nor native's
performance, determinism, or capability contract.** Higher priority than any
cleanup, because the failure mode is wrong results rather than excess lines.

### Tier 4 — Consumer and contract convergence

Merge the two replay pipelines (have the raw-topology route construct a
single-bucket `PreparedFixedTopology` internally), removing ~95 lines and one
duplicate ADR-038 liveness rule. Move the validator and capability table out of
`contracts.py`. Collapse the `PathTopology`/`PropagationTopology` and
`PathGeometry`/`PropagationGeometry` pairs, or document why both must exist.

### Tier 5 — Higher-order derivatives (needs an ADR; only if required)

Forward-over-reverse HVP does **not** require N² kernels — it requires the
backward kernel to itself support JVP, i.e. **one additional kernel per op**. This
is the only route to HVP that preserves fusion. If it is not pursued, the
`once_differentiable` limit should be stated in the capability record so callers
do not read it as an unimplemented detail.

### Not scheduled — topology discontinuity (research)

Class A. Requires reparameterization or warped-area sampling. Orthogonal to
everything above; Dr.Jit would not solve it either.

### Interaction with Part V

Part V's insertion interface (V.5, persona A) belongs immediately after Tier 0,
because Tier 0's Class-B unfreezing serves persona C and the two together cover
the common researcher cases at low cost. The `lab` replacement layer (persona B)
comes last and only under the constraints in V.6.

---

# Part V — Module boundaries, swappability, and the `witwin.lab` proposal

Context: a proposal to structure Channel as closed per-module kernels in the
gsplat style, with gradients crossing at tensor boundaries, plus a parallel
`witwin.lab.*` namespace offering the same modules as JIT/composable
implementations that a researcher can swap in one at a time.

This part assesses that proposal. It is analysis, not an accepted decision.

## V.1 The reference structure: gsplat

```
  means[N,3]  quats[N,4]  scales[N,3]  opacities[N]  viewmats[C,4,4]  Ks[C,3,3]
        └────────┴──────────┴──────────────┴──────────┬──────────┘
                                                       ▼
                                    ┌──────────────────────────────┐
                                    │  fully_fused_projection      │ fuses: world→cam,
                                    └──────────────────────────────┘ quat/scale→3D covar,
                                       │      │      │      │         projection+Jacobian,
                          radii ───────┘      │      │      │         2D covar, inverse,
                       (int, discrete)   means2d  depths  conics      radius, frustum cull
                                         [C,N,2]  [C,N]  [C,N,3]
     sh_coeffs ──┐                          │       │       │
          dirs ──┴──▶ spherical_harmonics ──┼───────┼───────┼──▶ colors[C,N,3]
                                            │       │       │          │
                  ┌─────────────────────────┴───────┴───┐   │          │
                  ▼                                     │   │          │
    ┌────────────────────────────────┐                  │   │          │
    │  isect_tiles / offset_encode   │ ◀── DISCRETE     │   │          │
    │  (tile binning + depth sort)   │                  │   │          │
    └────────────────────────────────┘                  │   │          │
                  │                                     │   │          │
      isect_offsets, flatten_ids  (int64 only)          │   │          │
                  └──────────┬──────────────────────────┴───┴──────────┘
                             ▼
               ┌──────────────────────────┐
               │   rasterize_to_pixels    │  consumes 4 differentiable tensors
               └──────────────────────────┘  + 2 integer metadata tensors
                             │
                  render_colors[C,H,W,3], render_alphas[C,H,W,1]
```

Five properties make this work:

1. **Low boundary arity with meaningful semantics.** Projection emits 3
   differentiable tensors; rasterization consumes 4. `means2d` and `conics` are
   quantities one wants to inspect or regularize anyway — they are not artifacts
   of the decomposition.
2. **The discrete stage emits indices only.** `isect_tiles` reads
   `means2d`/`radii`/`depths` but outputs `int64`. The differentiable values
   **bypass it** and flow directly from projection to rasterization.
3. **Each fused op has a hand-written backward.** The same trade Channel makes.
4. **Composition happens in Python at the tensor level.** `rasterization()` is a
   Python function calling four ops; users may call the ops directly.
5. **A pure-PyTorch reference implementation exists and is CI-enforced.**
   Verified against upstream (2026-07-27). `gsplat/cuda/_torch_impl.py` mirrors
   *every* stage, not a subset:

   ```
   _persp_proj / _fisheye_proj / _ortho_proj / _world_to_cam / _fully_fused_projection
   _isect_tiles / _isect_offset_encode / _isect_tiles_sparse     ← even the discrete stage
   _rasterize_to_pixels / _rasterize_to_pixels_sparse / accumulate
   _eval_sh_bases_fast / _spherical_harmonics
   ```

   `tests/test_basic.py` imports it and compares CUDA against Torch for **both
   forward and backward**.

   **The tolerance regime is the important finding**, because it is far looser
   than this repository's culture assumes:

   | Compared quantity | Tolerance |
   | --- | --- |
   | `quat_scale_to_covar_preci` gradients | `interior_rtol = 1.65e-2` (**1.65%**) |
   | projection forward `means2d` | `rtol = atol = 1e-4` |
   | projection backward | `interior_rtol = 7e-4` |
   | `rasterize_to_pixels` backward `v_means2d` | `rtol = 2.5e-4, atol = 1.6e-3` |

   They also maintain a bespoke `assert_close_with_boundary_band()` helper for
   FP32 cancellation noise, and their 2D-covariance comparison documents
   "~6% rel, ~0.12 abs" cancellation error.

   **Consequence for §V.5**: a differential oracle is a proven pattern, but it
   operates at ~1e-2 relative tolerance on some quantities and needs custom
   comparison machinery. It catches **structural** adjoint defects (missing
   terms, sign errors, index errors) — the common failure mode for hand-written
   adjoints — and does **not** substitute for ADR-004 lockstep on fine numerics.
   Plan for that tolerance regime rather than assuming bitwise agreement.

## V.2 Structural correspondence

| gsplat | Channel |
| --- | --- |
| `fully_fused_projection` | topology discovery + geometry |
| `isect_tiles` / `isect_offset_encode` (discrete) | winner selection + `canonical_compact` (discrete) |
| `spherical_harmonics` → colors | materials → `eps_r`/`sigma_e`/… (compile time) |
| `rasterize_to_pixels` | `field_*` transport + accumulation |

**The skeleton is already the same.** The proposal is not a redesign; it is a
formalization of boundaries that exist (§0.2).

## V.3 Two structural differences

### V.3.a The discrete stage's position and nature — the important one

```
gsplat:   differentiable ──▶ [DISCRETE: emits indices] ──▶ differentiable
                                 ╰── values bypass it ──╯
          Gaussians always exist; only tile assignment is discrete.

Channel:  [DISCRETE: decides which rows exist] ──▶ differentiable ──▶ …
                        ↑
                nothing can bypass it; every later stage
                operates on the rows this stage selected.
```

gsplat's discreteness is **benign** (indexing). Channel's is **existential**.
ADR-037's "a row can die through `row_valid`, a row is never born" states exactly
this.

Consequence: in gsplat a Gaussian leaving the frustum has its opacity fall
smoothly to zero, so the gradient is essentially complete. In Channel a specular
reflection whose stationary point slides off a facet edge is a **hard cutoff** —
the coefficient does not decay to zero, the row disappears. This is Class A
(§II.3) and the root of the radiomap FD-validation gap.

`propagation/geometry/silhouette_clearance.py` (128 lines) already touches this
boundary.

**Implication to record explicitly**: adopting the gsplat skeleton transfers its
structure, not its gradient completeness. A future reader must not infer that
matching gsplat's module layout confers gsplat's differentiability quality.

### V.3.b The shading stage is much heavier

gsplat's rasterizer consumes 4 differentiable tensors. Channel's field ops
consume 6 / 13 / 17 / 22.

The cause is physical, not a design defect: an RF material is
`(eps_r, sigma_e, mu_r, gain, thickness)` **per bounce**, whereas a Gaussian's
"material" is one RGB plus one opacity.

Practical consequence: **Channel's shading stage cannot be made into a thin
interface the way gsplat's is.** At 13–22 tensors, a reimplementation costs
nearly as much as the original and drifts immediately. This is the origin of the
arity threshold in V.4.

## V.4 Swappability matrix (verified)

**Criterion 1 — native state.** Every consumer of a RayD scene handle was
located by grepping `_rayd_scene_resource(`:

| Package | `_rayd_scene_resource` sites |
| --- | --- |
| `propagation/geometry/kernels/` (`autograd.py`, `bridge.py`, `penetration_autograd.py`) | 25 |
| `propagation/fields/kernels/` | **0** |
| `scattering/kernels/` | **0** |
| `materials/kernels/` | **0** |

The dividing line is clean and already exists. A JIT/Torch module cannot consume
a `SceneResource` RAII holder, so geometry and topology are **not swappable**.
This is semantically fortunate: those are the stages one least wants to swap, and
topology is discrete anyway.

**Criterion 2 — boundary arity.** Tensor parameters crossing each field-op
boundary:

| Boundary | Tensors | Verdict |
| --- | --- | --- |
| `field_free_space` | **6** | genuine module boundary (gsplat scale) |
| `field_reflection_sequence` | 13 | marginal |
| `field_transmission_sequence` | 17 | too wide |
| `field_diffraction_wedge` | 22 | not a module boundary |
| `scattering_chain_ensemble_eval` | **41** | a fused kernel with its internals exposed as an ABI |

**Candidate set = plain-tensor ∧ arity ≤ 10**: `field_free_space`, materials
evaluation, scattering evaluation. `field_diffraction_wedge` and
`scattering_chain_ensemble_eval` should be declared black-box kernels and
excluded from any composable module set.

## V.5 Assessment of the proposal

### What is sound

The core is sound and largely already built (V.2). The swappability boundary is
real, verified, and semantically well-placed (V.4). The performance claim holds
in direction.

### Four risks

**R1 — Modularization conflicts with ADR-009.** Making modules interchangeable
requires materializing intermediates between them, while ADR-009 forbids a
refactor that adds "kernel launches, synchronization, materialized intermediates."
`field_reflection_sequence` currently keeps a whole multi-bounce chain in
registers. **This requires an explicit ADR-009 amendment or a scoped exception,
not a silent reinterpretation.**

Corollary for the performance claim: the comparison is not "ours vs JIT" but
**"ours-after-modularization vs JIT"**. Still favourable, by a margin that depends
on where the boundaries are drawn. Finer granularity buys swappability and costs
bandwidth. That trade should be stated, not defaulted into.

**R2 — Invisible numerical divergence.** Two implementations cannot be bitwise
equal: reduction orders differ, and the native backward is *mixed-mode* (reverse
to the Fresnel coefficients, then one-hot forward dual seeds — §II.4.b) whereas a
JIT version would be pure reverse. Failure mode: a researcher swaps in the lab
module, sees a small delta, and **cannot distinguish better physics from a
numerical artifact.**

**R3 — Maintenance multiplier.** Unfreezing one parameter already costs ~15 edit
sites (§III.5), which is why the frozen-input table has never shrunk. A parallel
implementation multiplies every physics change. Adding this multiplier to a
system already stalled by it is the central danger.

**R4 — Rot.** This is not hypothetical. Part I.3 documents ~2,000 lines of
exactly this outcome already in the repository: carefully written, tested,
manifest-registered, docstring marked "Dormant", zero callers. A `lab` layer whose
only justification is "for researchers" has the same trajectory: written → main
line moves twice → drift → nobody trusts it → dead code.

### The mitigation that changes the outcome

Give `lab` a job the main line cannot drop:

> `witwin.lab.<module>` is an independent JIT reference implementation of
> `witwin.channel.<module>`. CI runs a primal + JVP + VJP differential comparison
> on small scenes every build. Researcher composition is its **second** purpose.

This yields three things: it cannot drift (drift reddens CI); the doubled
maintenance becomes an investment (independent verification of 47 hand-written
adjoints, many of whose VJPs have never been checked against anything
independent); and it supplies a cleaner instrument than finite differences for
the geometry-gradient validation gap.

Scope it honestly using the gsplat evidence (§V.1 property 5): at ~1e-2 relative
tolerance this oracle catches **structural** adjoint defects, not fine numerical
ones. That is still the common failure mode for hand-written adjoints, but the
claim must be stated at that strength — an oracle at 1.65% rtol is not a
substitute for ADR-004 lockstep, and presenting it as one would be worse than not
having it.

Boundary condition: CLAUDE.md currently permits Torch reference implementations
"only under `tests/`". Either widen that to "`witwin.lab` is a first-class test
asset that does not ship in the wheel", or place `lab` physically under `tests/`.
**It should not enter the production wheel** — that would create the first
exception to the single-backend policy, and exceptions grow.

### A cheaper 80%: insertion rather than replacement

Researcher needs decompose into three cases:

| Case | Need | Cost |
| --- | --- | --- |
| **A** "try a different scattering model" | **insertion** — take `PathGeometry`, compute coefficients in Torch, feed accumulation | near zero; needs only stable boundary contracts |
| **B** "modify reflection transport internals" | **replacement** — full lab implementation | highest |
| **C** "differentiate something you froze" | neither — unfreeze Class-B (§II.3) | 10–30 lines of CUDA each |

**A and C dominate; B is rare and is usually a prelude to contributing a kernel.**
The proposal's chosen mechanism (replacement) serves the rarest case at the
highest cost.

Case A is already ~80% built: ADR-036/037 publish `PropagationTopology`,
`PropagationGeometry`, and the transport contracts. What is missing is the return
path — feeding a **caller-computed** transport back into accumulation. That is an
order of magnitude cheaper than a parallel implementation of every module, and it
introduces no numerical divergence.

## V.6 Decisions required before implementation

Three decisions gate the work and should be settled first, because they change
how Tier 0, Tier 1, and the insertion interface are built:

1. **Whether the field stage stays word-indexed or becomes interaction-indexed**
   (§V.7). This is the most consequential of the three and gates Tier 1.
2. **Where the module boundaries sit.** Recommended rule: plain-tensor inputs and
   arity ≤ 10 (V.4). This determines whether ADR-009 must be amended and how much
   fusion is given up. Note that decision 1, if taken, collapses the arity
   problem and therefore changes this answer.
3. **Whether `lab` ships in the wheel.** This determines whether it is a test
   asset or the first exception to the single-backend policy.

A design ADR (provisionally "composable module boundaries and the lab
differential layer") is the right vehicle for 2 and 3. Decision 1 warrants its
own ADR because it touches the ADR-009 fusion boundary and the field-domain
ownership map.

## V.7 An interaction-indexed alternative for the field stage

### V.7.a The defect

A propagation path is a **word** over the interaction alphabet `{R, D, T, S}`.
The current architecture assigns one discovery function and one field kernel per
**word class**:

```
CURRENT — indexed by path word
  R…    → field_reflection_sequence        (13 tensors)
  D     → field_diffraction_wedge          (22 tensors)
  T…    → field_transmission_sequence      (17 tensors)
  RD    → field_coupled_rd                 ← one word, one complete owner
  DD    → coupled_dd                       ← one word, one complete owner
  S…    → scattering_chain_ensemble_eval   (41 tensors)

  adding one interaction type ⇒ a new kernel for its combination with every
  existing word.  Growth is O(N^k) in N interaction types and depth k.
```

This single choice is the common root of three separately-reported symptoms:

- the arity gradient 6 → 41 (§V.4) — each kernel must accept *every* parameter of
  *every* interaction in its word;
- `coupled_rd` / `coupled_dd` existing as complete owners with their own
  primal/JVP/VJP triples — they are merely the words "RD" and "DD";
- a large share of the 47 `autograd.Function` classes and the 41% plumbing
  measured in §I.1.

### V.7.b The physical form, and in-repo evidence that it already holds

```
E_rx = P_rx · G(s_n) · T_n · G(s_{n-1}) · T_{n-1} · … · T_1 · G(s_0) · P_tx
```

`T_i` is a 2×2 Jones interaction operator (Fresnel reflection, slab transmission,
UTD diffraction); `G(s)` is a scalar free-space propagator.

`propagation/consumer/_jones.py`'s module docstring states this structure
explicitly as its own design basis:

> "The native field transport is **linear in the transmit polarization and linear
> in the receive polarization** … a Fresnel bounce **scales the s and p
> components** by coefficients that depend on the incidence frame and the
> material, **never on the field itself** … the trailing free-space factor is a
> complex scalar … So the map from a source transverse component to a sink
> transverse component is **bilinear**, and the four entries of the operator are
> recovered exactly by exciting the SAME native transport twice."

Two consequences:

1. This **upgrades Appendix item 1** from an inference to an in-repository
   documented property. The Class-B polarization parameters (§II.3) are bilinear
   by the package's own stated design basis, not merely by my reading of the
   kernel.
2. The transport **is already structurally a 2×2 Jones operator**. The current
   code *recovers* that operator by exciting the transport twice and projecting —
   a workaround required precisely because the kernel is word-indexed and accepts
   concrete polarizations instead of emitting an operator.

### V.7.c The alternative

```
PROPOSED — indexed by interaction
  interaction_operator(type, primitive, material, w_in, w_out, f) → Jones 2×2   arity ≈ 7
  segment_propagator(length, f)                                   → complex     arity ≈ 2
  compose_chain(operators, segments, P_tx, P_rx)                  → transport   arity ≈ 4

  adding one interaction type ⇒ one new operator kernel.  Composition is free.
  Growth is O(N).
```

Effects:

- **Every boundary collapses to gsplat scale.** The composable-module candidate
  set (§V.4) widens from `field_free_space` alone to the whole field stage.
- **`coupled_rd` / `coupled_dd` stop being special cases** — subject to V.7.d.
- **The `lab` story becomes trivial**: a researcher writes *one* Torch operator
  and inserts it into the chain. No parallel implementation of anything, and no
  numerical-divergence risk (R2) for the modules they did not touch.
- **Class-B parameters unify**: `tx_power` and both polarizations appear once at
  the endpoints (`P_tx`, `P_rx`) instead of inside every kernel, which shrinks the
  Tier 0 surface substantially.

### V.7.d Infrastructure already present, and the genuine limits

Three prerequisites already exist:

| Requirement | Existing owner |
| --- | --- |
| Jones operator ABI contract | `field_state.py` — `JonesState` |
| Operator composition | `consumer/_jones.py` — `compose_jones`, `transverse_basis` |
| Word bucketing (removes chain-kernel branch divergence) | `consumer/_prepared.py:100` — `divmod(value, width + 1)` → `(component, depth)` buckets |

The third matters most. A per-interaction chain kernel would divergently branch
on interaction type inside its loop — except that `FixedTopologyBucket` already
partitions rows so that every row in a bucket shares a word. **That
infrastructure was built for ADR-037 and happens to be exactly what an
interaction-indexed chain requires.**

Genuine limits, stated plainly:

1. **Fusion loss.** Per-interaction operators require either materializing a 2×2
   complex per bounce per row, or a fused chain kernel. Bucketing solves
   divergence but not bandwidth. **This conflicts with ADR-009 and requires an
   explicit amendment, not a reinterpretation.**
2. **Coupled paths break the model.** The RD stationary re-solve couples the
   reflection point to the edge point; that is not a local operator. Coupled RD
   and DD would retain dedicated owners. Acceptable, because coupled has exactly
   two words and therefore does not combinatorially explode.
3. **Scattering breaks the model.** Rough-surface and phase-screen scattering is
   stochastic, not a deterministic 2×2.

So the model applies to `R`, `T`, and local-UTD `D` — which is precisely the part
that explodes combinatorially.

### V.7.e The decisive argument: the existing roadmap

Two planned work items are exactly the case that word-indexing penalizes and
interaction-indexing makes free:

| Planned work | Word-indexed cost | Interaction-indexed cost |
| --- | --- | --- |
| Multi-order diffraction rebuild (plan 32; ladder ends at Albani D12 + edge visibility graph) | new kernel + primal/JVP/VJP for `D`, `DD`, `DDD`, `RD`, `DR`, `RDR`, … | zero — the `D` operator exists; the chain supports arbitrary length |
| T-matrix scattering fed from Maxwell FDFD/FDTD | new kernel for `S`, `RS`, `SR`, `RSR`, `SRS`, … | **one new operator kernel** |

Every rung of plan 32's fix ladder currently pays the five-artifact cost again.

### V.7.f A near-free corollary

If a path value is a product of operators, then `row_valid ∈ {0,1}` can become
`row_weight ∈ [0,1]` — one more factor in the chain. This does **not** solve
Class A (computing a *correct* smooth weight is the Loubet/Bangaru problem), but
it means the architecture would not obstruct a future reparameterization: the
change becomes multiplying in a factor rather than reshaping the pipeline.
`propagation/geometry/silhouette_clearance.py` is the seed of this.

### V.7.g Sequencing consequence

**This decision must precede Part IV Tier 1.** A declarative registry built over
the current 47 word-indexed Functions would be built over a set that
interaction-indexing would merge. Deciding V.7 first avoids paying for the
registry twice.

Note also that V.7 reduces Tier 0's scope: with endpoint-level excitation, the
Class-B `tx_power` and polarization adjoints are written once at the chain
boundary rather than per word-kernel.

---

## Appendix — Claims requiring confirmation before action

1. ~~The bilinearity of the field in the two polarizations is read off kernel
   structure, not verified by a gradient check.~~ **Resolved (§V.7.b).**
   `propagation/consumer/_jones.py`'s module docstring states the bilinearity in
   the transmit and receive polarizations as the documented design basis of the
   composed Jones operator, and `consumer/_jones.py` depends on it in production.
   The kernel-structure reading (`field_transport_reflection.cu:52, :107, :474,
   :576`) agrees with it. An FD check remains advisable when the adjoint is
   written, but this is no longer an unsupported inference.
2. Dr.Jit tape-size estimates are order-of-magnitude arguments from flop
   structure, not measurements.
3. Dead-code verdicts were established by static grep over `src/`. Confirm against
   `tests/`, `benchmarks/`, and `tools/` before deletion — several dormant symbols
   have dedicated tests that must be removed in the same change.
4. `montecarlo/events/scattering.py:289` `te_tm_incident_power` sits on the
   boundary between the sanctioned "Monte Carlo event glue" exception and
   production polarization physics. This needs an owner decision, not a unilateral
   verdict.
5. ~~The gsplat structure in §V.1 must be confirmed upstream.~~ **Resolved
   (2026-07-27).** `gsplat/cuda/_torch_impl.py` and `tests/test_basic.py` were
   fetched and verified; the function inventory and tolerance table in §V.1
   property 5 come from that read. Remaining caveat: the tolerances were read
   from the current `main` branch and are not pinned to a release tag.
6. §V.5's "~80% built" claim for the insertion interface is based on the public
   consumer contracts (ADR-036/037) exposing topology, geometry, and transport.
   The missing return path — feeding a caller-computed transport into accumulation
   — was not prototyped and its cost is not measured.
7. §V.7's operator-chain proposal has **not** been prototyped. Three quantities
   are unmeasured and each could change the conclusion: (a) the bandwidth cost of
   materializing a 2×2 complex per bounce per row versus the current
   register-resident chain; (b) whether bucket counts stay small enough in
   realistic scenes that per-bucket launches do not dominate; (c) whether UTD
   diffraction is genuinely expressible as a local 2×2 operator once the
   spreading factor and transition function are included, or whether it joins
   coupled and scattering as an exception. Item (c) in particular should be
   settled before the ADR, because if UTD is non-local the proposal's benefit for
   plan 32 largely disappears.
