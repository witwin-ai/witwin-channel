# Compact Call Formatting Plan

Status: Active  
Category: Plan  
Last reviewed: 2026-05-17  
Owners: channel team  

Related:

- `plans/13-channel-readability-conciseness-followup-plan.md` — megamodule splits and `witwin/` line reduction (orthogonal to formatting)
- `standards/20-python-package-architecture-standard.md` — naming, thin wrappers, dataclass ownership, `ruff format` toolchain
- `optimization/channel-readability-conciseness-audit-2026-05-16.md` — baseline conciseness audit

## Purpose

Reduce inflated Python line counts caused by **manual one-argument-per-line** layout for function calls, constructors, and DrJit/native launches—without sacrificing readability or fighting the formatter.

Scope: production code under `witwin/` only. **`reference/` is out of scope** (no format or refactor passes there).

## Problem

Many call sites look like this even when each argument is short:

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

That pattern often adds **2–3× lines** compared to a compact layout, and it scales badly across integrators and kernel bindings. Root causes:

1. **Habit** — treating “one kwarg per line” as a style rule regardless of line width.
2. **Missing formatter config** — `pyproject.toml` has no `[tool.ruff]` section; `ruff format` is recommended in standards but not pinned for the repo.
3. **Oversized APIs** — functions with six or more parameters force vertical layouts even when a struct would be clearer (`PLR0913`-class signatures).

## Decision rules: one line vs multiple lines

| Situation | Layout |
| --- | --- |
| Entire call fits in **≤100 columns** (team default; see Ruff below) | **Single line** — do not break for alignment |
| Slightly over limit, 3–5 short arguments | **Two lines** max: open paren + args; or one line of positionals + one line of kwargs |
| Six or more arguments, or any argument is a multi-line expression | Multi-line is OK — **prefer reducing arity first** (dataclass/context) |
| Same long kwargs repeated at many call sites | **Dataclass / context object** — not repeated vertical boilerplate |

**Principle:** line breaks exist for **width** and **diff clarity**, not as decoration.

### Good compact example (already in tree)

```python
return integrator_cls().integrate(
    tx_pos, grid, mc_config, scene, resolved_trace, solver_controls,
    accumulation_backend=str(mc_config.accumulation_backend),
    return_timing=bool(return_timing),
    tx_power=tx_power,
)
```

## Structural fixes (higher ROI than “pressing lines together”)

Formatting alone cannot fix APIs that require a dozen keyword arguments.

### 1. Context objects for repeated call bundles

Replace long kwargs at integrator/postprocessing boundaries with a small frozen dataclass:

```python
@dataclass(slots=True)
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

Aligns with `20-python-package-architecture-standard.md`: fewer parameters, explicit types, no thin rename-only wrappers.

### 2. Keep configuration on `Config` / `SolveSpec` / `ResolvedTraceConfig`

Do not re-list the same ten fields at every call site. Pass one config object (or `TraceCtx`) through the pipeline.

### 3. Native / DrJit hot paths

Prefer **packed state** and kernel structs over growing Python-side argument lists. Python launchers should stay thin; vertical kwargs in `native_impl.py` are a smell to push into structs, not to pretty-print.

## Formatter toolchain

### Add Ruff format config to `channel/pyproject.toml`

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
src = ["witwin", "tests"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
```

- **`line-length = 100`** — slightly wider than Black’s 88; fewer artificial breaks for DrJit-heavy lines while staying reviewable.
- **`skip-magic-trailing-comma = false`** — a trailing comma after the last argument **forces** multi-line expansion (useful for diffs when intentional); omit the trailing comma when you want Ruff to collapse to one line.

### Validation order (unchanged from architecture standard)

1. `ruff format <paths>`
2. `ruff check <paths>`
3. `mypy` or `ty` (if configured)
4. targeted `pytest`

## Team conventions (human rules on top of Ruff)

1. If it fits in 100 columns, write it on **one line** (including `wt.Point3f(...)`, short `dr.select(...)`, small constructors).
2. When breaking lines, put **positionals first** on the first continuation line; reserve extra lines for long kwargs only.
3. **Do not** vertically align `=` across kwargs unless the formatter does it automatically.
4. **Do not** add a trailing comma solely to force vertical layout when a single line is readable.
5. **Do not** duplicate the same long kwargs block — introduce a dataclass or pass an existing config object.
6. New public APIs: target **≤5 positional parameters**; use `@dataclass` for grouped inputs (per architecture standard).
7. When editing a file for other reasons, run `ruff format` on **that file or package** only — no repo-wide format-only PR unless explicitly scheduled.

## Rollout phases

### Phase A — Policy and config (low risk)

- [ ] Land `[tool.ruff]` / `[tool.ruff.format]` in `channel/pyproject.toml`.
- [ ] Document this plan in `docs/dev/README.md` Active Plans table.
- [ ] Optional: add a one-line note in `standards/20-python-package-architecture-standard.md` pointing here for call-layout rules.

### Phase B — Format on touch (medium diff, controlled)

- [ ] While working on P1 megamodules from plan 13 (`bdpt_diffraction.py`, `basic.py`, `diffraction.py`, `reflection.py`), run `ruff format` on touched files only.
- [ ] Fix egregious manual vertical calls where a single line is obvious (no behavior change).

### Phase C — Context objects (behavior-preserving refactors)

- [ ] Identify top 5 call sites with ≥6 repeated kwargs (start: `ShadowBoundary.accumulate_into_diagnostics`, integrator finalize paths, path export materializers).
- [ ] Introduce context dataclasses; keep old function signatures as thin delegators only if needed for one release, then remove (no compatibility shims per repo rules).

### Phase D — Lint guardrails (optional)

- [ ] Enable `PLR0913` / `PLR0915` on new code paths via ruff `per-file-ignores` only where legacy blocks migration.
- [ ] Consider a custom or documented review checklist item: “no new 8+ kwarg calls without a context type.”

## Expected impact

| Approach | Typical line reduction at call sites | Risk |
| --- | --- | --- |
| Compact layout + `ruff format` only | ~5–15% in heavily vertical files | Low (mechanical) |
| `line-length = 100` | Fewer forced breaks vs 88 | Low |
| Context/dataclass for hot calls | ~20–40% at those sites; fewer args to read | Medium (refactor + tests) |

**Do not** expect a single formatting pass on all of `witwin/` to match the ~11% line drop from the MidMay-Refactor logic cleanup; formatting complements structural work in plan 13.

## Explicit non-goals

- **No changes under `reference/`** (formatting, line length, or refactors).
- **No repo-wide “format only” PR** that touches every file without functional reason.
- **No** replacing readable multi-line calls where arguments are genuinely long expressions.
- **No** new `build_*` helpers that only exist to avoid a long call — use constructors/dataclasses instead.

## Acceptance criteria

- `[tool.ruff]` present in `channel/pyproject.toml` with `line-length = 100`.
- New and touched `witwin/` code follows the decision table above; reviewers can cite this plan.
- At least **three** high-churn call bundles migrated to context dataclasses (Phase C) with pytest unchanged.
- No increase in `PLR0913`-style signatures in new APIs.
- Plan 13 megamodule work may proceed in parallel; formatting changes stay in the same PRs as those edits where possible.

## Tracking

Mark Phase A complete when Ruff config merges. Mark Phase C items in the checklist as each context type lands. When Phase B+C are done for the top integrator/path modules, move this plan to `archive/completed/` or fold remaining items into plan 13.
