# Channel Defensive-Code Reduction Plan

Status: Active
Category: Optimization
Last reviewed: 2026-05-15
Goal: Reduce line count by removing defensive programming, asserts, and runtime checks that violate the CLAUDE.md rule: *"Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs)."*

## Realistic ceiling

Conservative estimate from the static scan:

| Category | Count | Est. lines |
| --- | ---: | ---: |
| Internal `raise V/T/R` guards (non-boundary files) | 136 | ~270 |
| `isinstance` + `raise` paired (internal types) | 14 | ~40 |
| `safe_X` + `dr.select(valid, x, fallback)` redundant wrappers | ~60 | ~120 |
| `try/except` around trusted inputs | ~30 | ~120 |
| `max(0, ...)` / `max(1, ...)` clamps in scalar paths | ~50 | ~50 |
| `if width <= 0 / n_* <= 0` scalar early-outs | ~50 | ~100 |
| **Total** | | **~700-1100 lines (2-3% of 35,016)** |

**Verdict**: Pure defensive-code reduction is *not* where the bulk of the line savings live. The architectural refactor (dict-payload → typed dataclass) realistically saves 3-5k lines (~10%), but it is a code-shape change, not a deletion.

If line count is the *only* target, the highest-yield work is still architectural: the per-axis `{axis: ... for axis in ('x','y','z')}` patterns and the per-Jones-key `{'m00': ..., 'm01': ..., 'm10': ..., 'm11': ...}` literals collapse from 4-line blocks to 1-line operations once `JonesOp2x2` / `VectorComplex3` exist.

Recommendation: do the defensive sweep first (low risk, mechanical, 700-1100 lines), then continue with the architectural refactor for the rest.

## Reference: sionna-rt-2.0 comparison

Compared the audited packages against the bundled `sionna-rt-reference-2.0.0/src/sionna/rt` (50 files, 18,076 lines, same domain).

| Metric | sionna-rt 2.0 | witwin/channel | Ratio |
| --- | ---: | ---: | --- |
| Total lines | 18,076 | 35,016 | 1.94x |
| Files | 50 | 175 | 3.5x |
| `raise V/T/R` per 1k lines | 5.1 | 6.7 | similar |
| `isinstance` per 1k lines | 3.7 | 3.2 | similar |
| `try/except` blocks | 6 | 53 | **9x** |
| `safe_X` + `+EPS` patterns | 14 | 484 | **35x** |
| `assert` statements | 21 | 0 | sionna uses asserts, we don't |
| Largest single file | 1313 | 2085 | — |

Takeaways worth adopting:

1. **Drop the `safe_X` naming habit.** Sionna writes `idx = dr.maximum(idx, 0); val = dr.gather(..., idx)` directly, with no `safe_` prefix. The prefix inflates LOC and falsely implies an unsafe sibling. The 391 `safe_X` variable names in our code do not need that name; many can become unnamed inline expressions.
2. **Public APIs are keyword-only with defaults.** `PathSolver.__call__(scene, max_depth=3, los=True, specular_reflection=True, ...)` is 13 kwargs with defaults — readable at the call site. Our `BDPT.integrate` is 16 positional/keyword-mixed without defaults.
3. **Large files are allowed if they serve one class.** Sionna's 1313-line `scene.py` is all `class Scene`. Our 2085-line `bdpt_diffraction.py` is "3 strategies + tape store + edge-use store + result class + helpers" mixed. The line count is not the problem; the multi-concept mixing is.
4. **Sionna uses `assert` for internal invariants.** We have zero. Replacing `if cond: raise RuntimeError(...)` invariant guards with `assert cond` is a net win: shorter, semantically right (programmer-error, not user-error), and stripped under `python -O`.

What is NOT a problem on our side (matched levels):

- `raise` density per 1k lines is comparable (6.7 vs 5.1) — we are not over-raising at the boundary.
- `dr.select` density is comparable (15 / k vs 7.7 / k) — both are DrJit idioms, not defensiveness.
- Inner helpers in sionna also have 20+ arguments (`_shoot_and_bounce`, 271 lines / 20 args). The argument count itself is not the smell; the public-API contamination is.

Concrete rename / restructure targets implied by the comparison:

- Sweep `safe_*` variable names → drop the prefix where the variable is used only once or twice. Estimated 150-200 line reduction (1-line-per-site, names get inlined).
- Add a `tier-6` step: convert internal `raise RuntimeError(...)` invariant checks to `assert`. Estimated 30-50 line reduction (`raise` lines collapse from 2 lines to 1).

## Critical: which patterns are NOT defensive

Do NOT delete these. They look defensive but are functionally required:

- **`dr.select(valid, value, fallback)` in symbolic loops** — masks invalid lanes in vectorized DrJit code; required for symbolic-AD correctness. Removing breaks gradients.
- **`+ EPS` / `+ SMALL_EPS` in `dr.norm(...) + EPS`** — numerical stability in gradient computation through `1/x` and `sqrt`. Removing produces NaN gradients.
- **`safe_idx = dr.maximum(idx, 0)` before `dr.gather(..., safe_idx)`** — protects out-of-bounds gather; required to avoid GPU memory faults on invalid lanes inside symbolic loops.
- **`if int(value) <= 0: return wt.UInt32(0)` host-side early-outs before launching kernels** — required because DrJit symbolic loops on zero-width inputs can lock or produce undefined state.
- **`raise` inside `Config.__post_init__` / `Scene.__init__` / public `solve()`** — these *are* the system boundary.

A useful test: if removing the check would cause a silent NaN, GPU fault, or symbolic-loop hang under any *valid* input, keep it. If it would only fail on a programmer bug (wrong type passed from inside the package), delete it.

## Boundary files — keep all raises here

Treat as the API boundary; defensive checks here are correct:

- `witwin/channel/core/scene/scene.py` — public `Scene.__init__`, `from_sionna`, `load_mitsuba`
- `witwin/channel/core/scene/builder.py` — public constructor helpers
- `witwin/channel/core/scene/sionna_adaptor.py` — external-format adapter
- `witwin/channel/montecarlo/config.py` — `Config.__post_init__` (user-facing)
- `witwin/channel/deterministic/config.py` — `Config.__post_init__` (user-facing)
- `witwin/channel/path/config.py` — `Config` validation
- `witwin/channel/path/solver.py` — public `solve()`
- `witwin/channel/montecarlo/solver.py` — public `solve()`
- `witwin/channel/deterministic/solver.py` — public `solve()`
- Every `__init__.py` re-export

This is roughly **100 of the 236 `raise V/T/R` occurrences** that you should leave alone.

## Deletion targets (by package, ordered by safety)

### Tier 1 — Mechanical and reversible (do first)

These are pure deletions with no behavior change for valid callers. Risk: a buggy internal caller will now produce a different (later) error message instead of the explicit one. That's the point — the rule says "trust internal code."

#### `witwin/channel/core/scene/scene.py`

| Line | Pattern | Action |
| ---: | --- | --- |
| 636-637 | `if not isinstance(ray, rayd.Ray): raise TypeError(...)` in internal `ray_test` | delete (callers are solvers, type guaranteed) |
| 641-642 | same in `ray_intersect` | delete |
| 661-663 | `if vertices is None: raise ValueError` — `to_point3f(None)` already raises | delete |
| 668-669 | `if len(mesh_structures) != len(mesh_buffers): raise RuntimeError("inconsistent")` — invariant guard | delete |

Estimated reduction: ~12 lines in this file.

#### `witwin/channel/deterministic/path/los.py`, `reflection.py`, `diffraction.py`, `diffraction_impl/accumulation.py`

| Pattern | Sites | Action |
| --- | ---: | --- |
| `if runtime.rx is None: raise ValueError("trace requires receiver positions in runtime.rx.")` | 5 | delete — `runtime.rx` is dereferenced 2 lines later and would NPE on its own |
| `if scene is None: raise ValueError(...)` in non-boundary helpers | 4 | delete |

Estimated reduction: ~18 lines.

#### `witwin/channel/montecarlo/integrators/bdpt.py`, `basic.py`, `bdpt_ad.py`

| Pattern | Sites | Action |
| --- | ---: | --- |
| `raise NotImplementedError("BDPT does not return reflection detail in v1.")` plus the `if reflection_detail is not None:` check (`bdpt.py:102-103`) | 1 | delete — `reflection_detail` is plumbed through and unused; the parameter itself should go |
| `if requested_sampling not in {...}: raise ValueError` inside internal helper `_resolve_sampling()` | several | delete (resolved once at boundary) |

Estimated reduction: ~15 lines.

#### Across all internal files: paired `isinstance + raise`

```
if not isinstance(x, ExpectedType):
    raise TypeError(f"... expects {ExpectedType.__name__}, got {type(x).__name__}.")
```

When the function is called only from inside the package: delete. ~14 sites × 3 lines each.

### Tier 2 — Type-annotation cleanup (low risk, modest yield)

Replace `if x is None: raise` patterns with non-`Optional` annotations and let mypy enforce it statically. Where the parameter is genuinely optional but only one branch is ever used in the codebase, drop the parameter entirely.

Worth doing in:

- `witwin/channel/montecarlo/integrators/bdpt.py:102` — `reflection_detail` is `None` in every caller and raises if not. Drop the parameter.
- `witwin/channel/montecarlo/integrators/basic.py` — `loop_mode` parameter is passed everywhere as `"symbolic"`; check the call graph.
- Several `default_X=None` parameters in `montecarlo/config.py` resolver chains.

Estimated reduction: ~50 lines, mostly parameter list trimming on the giant 16-21-arg functions.

### Tier 3 — Scalar-path guards (review before deleting)

These need a 5-second check each:

- `if int(value) <= 0: return wt.UInt32(0)` — KEEP if it guards a `dr.hint(mode=symbolic, max_iterations=...)` loop. DELETE if it only protects against empty Python sequences that are already empty.
- `max(1, n_triangles)` — KEEP if it appears inside a `dr.hint(...)` exclude list or a divisor. DELETE if it's a Python-side branch that's already covered by an outer `if`.

Estimated reduction: ~80 lines (out of 111 occurrences).

### Tier 4 — `try/except` around config and conversion

53 `try:` + 52 `except` across the audited packages. Some are legitimate (`try: import ... except ImportError`). The cuttable ones:

- `try: return int(value); except TypeError: return int(value[0])` — `witwin/channel/core/scene/scene.py:54-57` `_scalar_int`. Replace with a typed parameter and require the caller to pass the right shape.
- `try: ... except KeyError: return default` blocks around dict-shaped runtime payloads — these go away naturally when the dict becomes a dataclass (architectural refactor).

Estimated reduction: ~80 lines reducible now, ~50 more after architectural refactor.

### Tier 5 — `safe_X` + `dr.select` patterns (HIGH RISK — read first)

391 `safe_X` variable usages and 527 `dr.select(...)` calls. Most of these are functionally required, not defensive.

**Only delete the wrapper, not the `dr.select`.** Pattern to look for:

```python
# Current (3 lines, but the middle line is sometimes redundant):
valid = mask & (prim_idx >= 0) & (prim_idx < n)
safe_idx = wt.UInt32(dr.select(valid, prim_idx, wt.Int32(0)))
result = dr.gather(T, buffer, safe_idx)

# Reducible to (still safe) when `valid` is already enforced upstream:
result = dr.gather(T, buffer, wt.UInt32(prim_idx))
```

But you can only do this when the caller has already enforced `prim_idx >= 0` and `prim_idx < n`. Many do; some don't. Each site needs review.

Estimated reduction: ~60 sites × 1-2 lines = ~80-120 lines. Do this last, with regression tests on each sweep.

## Order of operations

Lowest risk → highest yield. Stop and validate after each step.

| Step | Scope | Est. lines | Risk |
| ---: | --- | ---: | --- |
| 1 | Tier 1 sweep: internal `raise V/T/R` guards in `path/` and `deterministic/path/` | ~50 | very low |
| 2 | Tier 1 sweep: internal `raise V/T/R` in `montecarlo/integrators/` and `montecarlo/path/` | ~80 | very low |
| 3 | Tier 1 sweep: `channel_scene/scene.py` internal type checks | ~15 | very low |
| 4 | Tier 1 sweep: kernel-wrapper internal type checks (`*/kernels/*/native_impl.py`) | ~60 | very low |
| 5 | Tier 2: drop dead `Optional` parameters and unused branches | ~50 | low |
| 6 | Tier 4: `try/except` around trusted conversions | ~80 | low-medium |
| 7 | Tier 3: scalar-path `max(0/1, ...)` and `<= 0` guards (review each) | ~80 | medium |
| 8 | Tier 5: `safe_X` + `dr.select` wrapper reduction (review each) | ~100 | medium-high |
| | **Total** | **~515 lines** | |

For the remaining 200-600 lines of theoretical reduction, the work crosses into the architectural refactor (dict payload → typed dataclass). Once `TriangleRuntime` and `JonesOp2x2` exist, another ~1.5-3k lines of incidental savings fall out for free.

## Can `/simplify` skill drive this?

**No, not as the driver. Yes, as the post-stage QA.**

`/simplify` reviews **staged changes** for "reuse, quality, efficiency". It does not know:

- Which `raise` sites are at the API boundary vs internal
- Which `dr.select` calls are AD-required vs defensive
- Which `max(1, n)` calls protect a symbolic loop vs an already-bounded value

Effective pattern:

1. Use this document as the inventory.
2. Do one Tier of deletions per branch (or per file batch).
3. After staging the diff, run `/simplify` — it will catch leftover dead variables, redundant aliases, missed call sites, and dead imports in *that diff*.
4. Run targeted ruff with the audit selectors before merging.
5. Run targeted regression tests (per `docs/dev/standards/50-test-and-acceptance-workflow.md`).

Do NOT run `/simplify` over a fresh checkout expecting it to find the defensive sites. It works on diffs.

## Concrete first step

If you give me the go-ahead, I will:

1. Do Tier 1 sweep on `witwin/channel/deterministic/path/los.py`, `reflection.py`, `diffraction.py`, `diffraction_impl/accumulation.py` — pure deletion of `if runtime.rx is None: raise ...` and similar internal guards.
2. Run `pytest tests/` for the affected modules.
3. Run `ruff check witwin/channel/deterministic/path --select F401,F821,ANN201,C901,PLR0913,PLR0915,FBT001`.
4. Report the line delta.

This is roughly 20-30 lines deleted, 5 files touched, fully mechanical. If acceptable, continue with steps 2-4 in the order table above.
