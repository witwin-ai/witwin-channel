# Channel Package Deep Audit (Concrete Findings)

Status: Active
Category: Optimization
Last reviewed: 2026-05-15
Companion to: `channel-package-architecture-audit-2026-05-15.md` (strategic / rating-focused).
This document is the **evidence + ordering** companion.

## Scope

Five packages, ~35,016 lines of Python total.

| Package | Lines | Role |
| --- | ---: | --- |
| `witwin/channel/core` | 1,896 | Pure math / DrJit helpers (foundational) |
| `witwin/channel/core/scene` | 2,178 | Scene + runtime / RayD adaptor (data layer) |
| `witwin/channel/deterministic` | 16,070 | Deterministic radiomap solver |
| `witwin/channel/montecarlo` | 14,872 | Monte Carlo radiomap solver |
| `witwin/channel/path` | 3,186 | Path-level collector solver (depends on `deterministic`) |

## Static-scan headline numbers

ruff under `F401,F821,ANN201,C901,PLR0913,PLR0915,FBT001`:

- 477 issues
- 225 too-many-arguments
- 167 missing public return annotations
- 26 too-many-statements
- 19 cyclomatic-complexity hotspots
- 16 boolean positional traps
- 6 undefined-name (`Scene`, `GridSpec`)

Dict-shaped public payloads:

- 81 functions return `dict / Dict / Mapping`
- 80 functions accept `dict / Dict` annotated parameters
- 32 functions accept `Mapping` annotated parameters
- 56 `'m00'|'m01'|'m10'|'m11'` literal accesses (Jones-as-dict)
- ~200 accesses of `scene._triangle_runtime` and similar private attributes from outside `Scene`

Largest files / worst function shapes:

| Lines | >=10-arg fns | >=100-line fns | Path |
| ---: | ---: | ---: | --- |
| 2085 | 9 | 7 | `montecarlo/integrators/bdpt_diffraction.py` |
| 1587 | 11 | 3 | `deterministic/kernels/radio_map_accumulate/native_impl.py` |
| 1317 | 13 | 3 | `deterministic/kernels/suffix_grid/native_impl.py` |
| 1168 | 6 | 0 | `deterministic/path/diffraction_impl/math.py` |
| 1146 | 5 | 2 | `montecarlo/integrators/basic.py` |
| 1021 | 0 | 1 | `deterministic/kernels/packed_state/native_impl.py` |
| 990 | 0 | 1 | `montecarlo/config.py` |
| 920 | 1 | 4 | `montecarlo/integrators/bdpt_ad.py` |
| 879 | 6 | 1 | `montecarlo/path/diffraction.py` |
| 873 | 7 | 1 | `deterministic/path/reflection_impl/epc.py` |

## Seven worst sites (with proof)

### 1. `montecarlo/integrators/bdpt.py:82` — `BDPT.integrate` is **476 lines / 16 params / complexity 17**

A single method that mixes:

- AD-mode dispatch
- LoS / reflection / BDPT diffraction phase calls
- shadow-boundary correction
- metadata assembly
- primal-state assembly
- result assembly

`witwin/channel/montecarlo/integrators/bdpt.py:340-470` contains a giant inline metadata literal where English explanatory strings are stored as dict values:

```
"pdf": "uniform_cell_times_edge_length_density",
"mis_weight": "balance_against_keller_cone_area_density",
```

Documentation is being smuggled in as runtime data — neither typecheckable nor greppable as code.

### 2. `montecarlo/integrators/bdpt_diffraction.py` — **2085 lines, 9 fns >=10 params, 7 fns >=100 lines**

Worst 5:

| Line | Args | Body | Function |
| ---: | ---: | ---: | --- |
| 1820 | 14 | 262 | `trace` |
| 1577 | 18 | 240 | `trace_suffix_reflection_batches` |
| 1372 | 21 | 201 | `trace_chain_keller_batches` |
| 1193 | 20 | 175 | `trace_chain_direct_batches` |
| 899 | 18 | 175 | `trace_keller_batches` |

`chain["..."]`, `sp["..."]`, `cell["..."]` dict-payload accesses occur **155 times** in this one file. The three strategies (`DIRECT` / `KELLER` / `SUFFIX_REFLECTION`) all have near-identical batch loops with 17-21 parameter lists — copy-paste with strategy-specific arithmetic.

### 3. `montecarlo/integrators/metadata.py:56` — refactor false positive

`MetadataInput` dataclass was added at `metadata.py:22` (was a 29-arg function). But `build_metadata` then unpacks all 31 fields back into local variables in the first 30 lines of its body:

```
def build_metadata(payload: MetadataInput) -> dict[str, object]:
    grid = payload.grid
    scene = payload.scene
    batch_plan = payload.batch_plan
    ...  # x30
```

Return type is still `dict[str, object]`. The dataclass moved the symptom; the underlying contract is still a loose dict.

### 4. `Scene` private attributes are the de-facto public API

| Accesses | Attribute |
| ---: | --- |
| 46 | `scene._triangle_runtime` |
| 15 | `scene._rayd_scene` |
| 13 | `scene._selected_edge_runtime` |
| 8 | `scene._structure_meshes` |
| 8 | `scene._triangle_surface_edge_groups` |

`tri_data["v0"]`, `tri_data["material_specified"]`, etc., dict-shaped runtime field access appears **~200 times across 24 files**. The repo's stated stable contract is `Scene + Tracer + Result`, but solvers reach through Scene's underscore-prefixed dict cache to do real work.

### 5. `deterministic/path/diffraction_impl/math.py` — **1168 lines + Jones operator as `dict[str, Complex2f]`**

40 `'m00'|'m01'|'m10'|'m11'` literal accesses in this file alone, 56 across the repo. The foundation in `channel_utils/polarization.py:44` (`vector_zero`) is already dict-shaped (`{axis: ... for axis in ("x","y","z")}`). Every consumer mirrors that shape. `matmul_op`, `detach_op`, `rotate_op`, `mask_op` all hand-write the 4-field dict literal.

### 6. `deterministic/path/diffraction_impl/shadow_boundary_correction.py:7` — package layering inversion

```
from witwin.channel.montecarlo.kernels.shadow_boundary import ShadowBoundaryKernel
```

`deterministic` imports from `montecarlo`. This is the one outright dependency reversal in the audited set. It breaks the documented `utils -> scene -> trace -> monitors` layering and makes the two solver packages non-independent.

### 7. `deterministic/path/diffraction_impl/builders.py` — 700 lines, four giant phase functions

| Line | Args | Body | Complexity | Function |
| ---: | ---: | ---: | ---: | --- |
| 118 | 20 | 114 | 15 | `prepare` |
| 279 | 9 | 72 | — | `(builder phase)` |
| 442 | 8 | — | 19 | `higher` |
| 572 | 12 | 122 | — | `inserted` |

`prepare` (line 118) takes two boolean positional arguments (ruff FBT001 col 201 / 459) — typical call site looks like `prepare(..., True, False, True, ...)`.

## Second tier (high but not blocking)

- `deterministic/kernels/radio_map_accumulate/native_impl.py` (1587 lines): reference path + native launch + shadow-boundary mixed in one wrapper; 11 fns >=10 args including one 23-arg launch.
- `deterministic/kernels/suffix_grid/native_impl.py` (1317 lines): **13 fns >=10 args**, with 27/23/22 arg `_launch_*_resume_batched` siblings. Top candidate for typed launch-parameter objects.
- `channel_scene/sionna_adaptor.py`: 6 `F821: Undefined Scene` (annotations without import). `sys.path.insert` mutation at `:38-39` leaks into the host interpreter.
- `montecarlo/config.py` (990) and `deterministic/config.py` (562): `__post_init__` complexity 17 / 14, `resolve_solver_controls` complexity 14. Guardrail / trace-control normalization is duplicated across the two packages.

## Cross-package duplication that belongs in `channel_utils`

Exact same function name in both `deterministic` and `montecarlo`:

| Function | det location | mc location |
| --- | --- | --- |
| `material_angular_frequency` | `deterministic/runtime.py` | `montecarlo/materials.py` |
| `assert_scene_materials_complete` | `deterministic/runtime.py` | `montecarlo/materials.py` |
| `tangential_axes` | `deterministic/{field,grid}.py` | `montecarlo/grid.py` |
| `pos_to_idx` / `get_coordinates` / `receiver_positions_3d` | `deterministic/{field,grid}.py` | (Grid class) |
| `point_grad_enabled` | `deterministic/path/...` | `montecarlo/integrators/basic_ad.py`, `montecarlo/path/diffraction_support.py` |
| `scene_geometry_grad_enabled` / `scene_material_grad_enabled` | `deterministic/path/reflection_impl/common.py` | `montecarlo/integrators/basic_ad.py`, `montecarlo/path/diffraction_support.py` |

## Architecture: does the 5-package shape need changing?

Current dependency picture (observed via imports):

```
channel_utils ──┬──> channel_scene ──┬──> deterministic ──┐
                │                    ├──> montecarlo     ─┴──> path
                │                    │            ^
                │                    │            │ (reversed dep — bug)
                └────────────────────┘            │
                                        montecarlo/kernels/shadow_boundary
```

**The 5-package shape is correct.** Do not split or merge packages. Three smaller boundary adjustments fix everything:

1. **Add a shared kernel home.** Either `witwin/channel/kernels/` or `witwin/channel/core/kernels/`. Move `shadow_boundary` (and any future cross-solver kernel) there. Both solvers depend on it; neither depends on the other.
2. **Promote `_triangle_runtime` / `_selected_edge_runtime` from dict-cache to typed runtime types** (`TriangleRuntime`, `EdgeRuntime` slots dataclasses in `channel_scene`). Expose as read-only Scene properties. Solvers stop reaching through the underscore layer.
3. **Promote dict-shaped Jones / Vector3 in `channel_utils/polarization.py` to typed `JonesOp2x2` / `VectorComplex3` slots dataclasses.** This is the foundation; the 56 literal accesses upstairs convert mechanically once it lands.

No new package. No reorganization of the 5-package boundary. Just three typed payloads + one kernel move.

## Refactor order (the order to actually do the work in)

Each step is unblocked by the previous ones. Validate each by running targeted regression tests before moving on (per `docs/dev/standards/50-test-and-acceptance-workflow.md`).

### Phase 1 — Foundation (`channel_utils` first)

Goal: stop the dict-bleed at the source.

1. **`JonesOp2x2` slots dataclass + ops** (matmul / mask / detach / rotate / scale / identity / diagonal / add) in `channel_utils/polarization.py`. Keep dict-returning aliases temporarily? No — per CLAUDE.md, do not preserve compat shims. Convert call sites in the same change.
2. **`VectorComplex3` slots dataclass + ops** (add / scale / select / zero / from_scalar_and_real_direction). Same call-site sweep.
3. **Move duplicated helpers into `channel_utils`**: `material_angular_frequency`, `assert_scene_materials_complete`, `point_grad_enabled`, `scene_geometry_grad_enabled`, `scene_material_grad_enabled`, `tangential_axes`. Delete the duplicates in `deterministic/runtime.py` and `montecarlo/materials.py`.

Expected diff: ~1.5k lines touched (mostly in `deterministic/path/diffraction_impl/math.py:1168` and consumers).

### Phase 2 — Scene boundary (`channel_scene` next)

Goal: remove the ~200 underscore-attribute reaches.

4. **`TriangleRuntime` slots dataclass** (`v0/v1/v2/material_specified/material_structure_idx/n_triangles/...`) and `EdgeRuntime` equivalent. `Scene._triangle_runtime()` becomes `Scene.triangle_runtime` property returning the typed object.
5. **Replace all `tri_data["..."]` accesses** with attribute access. Most call sites are one-line mechanical changes. The 46 callers of `scene._triangle_runtime` should drop the underscore in the same sweep.
6. **`channel_scene/sionna_adaptor.py`**: fix `F821: Scene` imports; isolate `sys.path.insert` behind a context manager or document explicitly that the adapter is import-time-only.

Expected diff: ~24 files touched; mostly trivial attribute renames after the type is in place.

### Phase 3 — Shared kernel home (cuts the package inversion)

7. **Decide on `witwin/channel/kernels/`** (preferred — it sits below both solvers) **or** `witwin/channel/core/kernels/`. Move `montecarlo/kernels/shadow_boundary` there. Update the one import in `deterministic/path/diffraction_impl/shadow_boundary_correction.py` and the existing montecarlo callers.

Expected diff: ~5 files. Small but unblocks Phase 4/5 cleanly.

### Phase 4 — Deterministic solver internals

Goal: tame `builders.py` and `radio_map_accumulate`.

8. **`DiffractionStateBuilder` class** owning `prepare / first / higher / inserted` as methods. Shared `BuilderContext` slots dataclass carries the parameters that "happen to be needed by every phase". Kill the 20-arg `prepare`.
9. **Typed launch-parameter object for `suffix_grid` and `radio_map_accumulate`** native launches. The 22-27 arg `_launch_*` siblings collapse into a `SuffixGridLaunchParams` dataclass + 1-arg launch.

### Phase 5 — Monte Carlo solver internals (the worst file, last)

Goal: split `bdpt_diffraction.py` and shrink `bdpt.integrate`.

10. **Strategy runner pattern**: `DiffractionChainSample`, `TargetCell`, `EdgeSample` slots dataclasses replace the three internal dicts. `trace_chain_{direct,keller,suffix_reflection}_batches` collapse into a single `_run_strategy(strategy: StrategyId, sample: DiffractionChainSample, ...)`.
11. **Extract BDPT metadata** out of `bdpt.integrate`. Use Enum values where the inline literal currently has prose. `bdpt_metadata.py` owns the assembly. `integrate` becomes phase ordering only.
12. **`BDPTDiffractionTapeStore` / `BDPTDiffractionEdgeUseStore`**: keep, they are already correctly owned. Just stop reaching into them from outside.

### Phase 6 — `path` package alignment

Goal: ride on the now-clean lower layers.

13. `witwin/channel/path` consumes `deterministic` already. Once `JonesOp2x2` / `TriangleRuntime` exist, sweep `path/collectors.py` (1155 lines) and `path/result.py` (1190 lines) for dict-payload smell. These two files have not yet been audited line-by-line but follow the same pattern — re-audit after Phase 1-2.

## Can `/simplify` skill do this?

**Short answer: no, not as a top-level driver.** `/simplify` is described as "review changed code for reuse, quality, and efficiency, then fix any issues found". It works on the **current diff / staged changes**, not on whole packages.

Useful pattern:

- After you finish a Phase step (say, the `JonesOp2x2` rollout) and stage the change, run `/simplify` to catch leftover dict-shaped fallbacks, redundant helpers, dead imports, and missed call sites within that diff.
- Do not expect `/simplify` to plan the staging order, do the cross-file dataclass introduction, or detect the architectural issues (kernel reversal, Scene private-API leak). Those are out of its scope.

For the planning and big mechanical edits use:

- `Plan` agent for the per-phase implementation plan (one phase at a time).
- Direct edits for the actual code change.
- `/simplify` as the post-stage QA pass.
- Targeted ruff scan (`F401,F821,ANN201,C901,PLR0913,PLR0915,FBT001`) before merging each phase to confirm the issue count drops monotonically.

## Validation

Each phase should:

1. Add or update a focused regression test before production edits.
2. Run the targeted test and confirm the expected failure or baseline guard.
3. Implement the smallest behavior-preserving refactor.
4. Run the targeted test.
5. Run `ruff check` on the touched files with the audit selectors.
6. Run relevant integration tests when crossing package boundaries (especially Phase 3 and Phase 5).
