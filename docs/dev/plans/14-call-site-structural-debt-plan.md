# Call-Site Structural Debt Reduction Plan

Status: Active  
Category: Plan  
Last reviewed: 2026-05-17  
Owners: channel team  

Related:

- `plans/13-channel-readability-conciseness-followup-plan.md` — megamodule splits and shared-module extraction
- `standards/20-python-package-architecture-standard.md` — dataclass ownership, thin-wrapper ban, ≤5 positional parameters
- `optimization/channel-readability-conciseness-audit-2026-05-16.md` — baseline conciseness audit

## Purpose

Cut inflated line counts and call-site noise by **reducing arity and repeated kwargs**, not by layout or formatter policy.

Vertical calls such as one keyword argument per line are usually a **symptom** of functions that take too many loosely related parameters. The fix is to group state into explicit types and pass one object through the pipeline.

Scope: `witwin/` only. **`reference/` is out of scope.**

## Problem

Typical bloated call site:

```python
ShadowBoundary.accumulate_into_diagnostics(
    weighted_diagnostics=weighted_diagnostics,
    scene=scene,
    tx_pos=tx_pos,
    grid=grid,
    config=config,
    edge_indices=edge_indices,
    ad_enabled=resolved_ad_mode,
)
```

This is hard to read and multiplies lines across integrators. Root causes:

1. **High arity** — six or more parameters where a single context would suffice.
2. **Repeated kwargs** — the same `scene`, `grid`, `config`, `tx_pos` tuple passed through every stage.
3. **Config re-expansion** — fields already on `Config` / `SolveSpec` / `ResolvedTraceConfig` / `TraceCtx` passed again as separate arguments.
4. **Python-side kernel launches** — long argument lists in `native_impl.py` instead of packed structs or existing packed-state layouts.

Layout style (one line vs many) is **out of scope** for this plan; treat it as a separate concern if needed later.

## Structural rules

| Situation | Action |
| --- | --- |
| ≥6 parameters at a public or cross-module boundary | Introduce a `@dataclass` context or pass an existing config/runtime object |
| Same kwargs at ≥3 call sites in one module | One context type owned by that module or `channel_utils` |
| Arguments are all fields of an existing config | Pass the config (or `TraceCtx`) only |
| Native / DrJit launcher with many scalar args | Extend packed state or kernel struct; keep Python launcher thin |
| Need a shorter call | **Never** add a rename-only wrapper; add a real type |

New APIs: **≤5 positional parameters**; group the rest in a frozen dataclass (per architecture standard).

### Target shape

```python
@dataclass(frozen=True, slots=True)
class ShadowAccumContext:
    diagnostics: dict
    scene: Scene
    tx_pos: wt.Point3f
    grid: Grid
    config: Config
    edge_indices: wt.UInt32
    ad_enabled: bool

ShadowBoundary.accumulate(ctx)
```

Prefer **one object per pipeline stage** (`TraceCtx`, integrator-local context, postprocessing context) over growing function signatures.

### Already good pattern in tree

`montecarlo/solver.py` passes a small set of objects into `integrate(...)` (`tx_pos`, `grid`, `mc_config`, `scene`, `resolved_trace`, `solver_controls`) instead of re-listing every trace flag. Extend that pattern downward into path and postprocessing layers.

## Priority refactors

Work in this order when touching related code (align with plan 13 megamodule edits):

| Priority | Area | Current smell | Structural target |
| --- | --- | --- | --- |
| 1 | `montecarlo/integrators/basic.py` → `ShadowBoundary.accumulate_into_diagnostics` | 7+ kwargs from integrator | `ShadowAccumContext` or integrator-owned `McTraceContext` |
| 2 | `montecarlo/integrators/bdpt.py`, `bdpt_diffraction.py` | Same kwargs repeated per phase | Shared context built once per `integrate()` |
| 3 | `deterministic/solver.py` trace stages | `scene`, `grid`, `spec`, `solver_controls` through los/reflection/diffraction | Widen `TraceCtx` usage; avoid parallel parameter lists |
| 4 | `deterministic/path/path_export.py` + `path_export_assembly.py` | Large materializer/collect signatures | Payload/collector context types (partially started by assembly split) |
| 5 | `native_impl.py` launchers | Long Python arg lists | Packed buffers / structs (coordinate with kernel owners) |

## Rollout phases

### Phase 1 — Inventory (read-only)

- [ ] List functions under `witwin/` with ≥6 parameters (public or called from ≥2 modules).
- [ ] Mark which already have a suitable config type (`Config`, `TraceCtx`, `ResolvedTraceConfig`, `Grid`, `Scene`).
- [ ] Record top 10 by call-site count in a short table in this plan (optional appendix commit).

### Phase 2 — Context types at integrator boundaries

- [ ] `ShadowBoundary.accumulate_into_diagnostics` → `accumulate(ctx: ShadowAccumContext)` (mc); update `basic.py` / `bdpt.py` call sites.
- [ ] If det postprocessing shares the same bundle, place the dataclass in `channel_utils` or a shared `path` submodule; do not duplicate two identical structs.

### Phase 3 — Pipeline objects

- [ ] Thread one `TraceCtx` (or deterministic equivalent) through los / reflection / diffraction without re-passing `scene`, `grid`, `spec` separately where already on the context.
- [ ] Path export: collector/materializer functions take a single `PathExportContext` where signatures exceed five logical inputs.

### Phase 4 — Native launchers (opportunistic)

- [ ] When editing a `native_impl.py` for other reasons, collapse new arguments into existing packed layouts instead of extending Python signatures.

## Expected impact

| Change | Effect |
| --- | --- |
| Context dataclass at a hot call site | ~20–40% fewer lines at those sites; easier reviews |
| `TraceCtx` / config threading | Fewer repeated kwargs across modules |
| Packed native args | Fewer Python launcher lines; less drift between det/mc |

Structural work complements plan 13 (module splits, shared geometry); it does not replace them.

## Explicit non-goals

- **No changes under `reference/`.**
- **No formatter or linter rollout** (no `ruff format`, line-length policy, or format-only PRs as part of this plan).
- **No** rename-only wrapper functions to shorten a call.
- **No** new `build_*` helpers when a constructor or dataclass is the right API.
- **No** merging det and mc reflection implementations.

## Acceptance criteria

- At least **three** high-churn call bundles use a context dataclass with pytest unchanged.
- New public functions in touched modules stay at **≤5 positional parameters**.
- No new parallel copies of the same six-field kwargs list without a shared type.
- Inventory (Phase 1) complete or explicitly deferred with owner note in this file.

## Tracking

Check off Phase 2–3 items as they land. When priority rows 1–4 are done or deferred, move this plan to `archive/completed/` or fold open items into plan 13.
