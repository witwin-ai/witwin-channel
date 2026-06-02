# Python Package Architecture Standard

Status: Active
Category: Standard
Last reviewed: 2026-05-20

## Purpose

This document defines the package-architecture and naming rules for Python code in `witwin.channel`.

The goals are:

- keep module boundaries readable and stable
- stop name growth that turns files, functions, and variables into sentences
- eliminate duplicate math and utility helpers
- prevent thin one-line wrapper functions from obscuring the real call graph
- move stateful feature logic toward explicit object-oriented ownership instead of loose procedural pipelines
- replace loose dict payloads with explicit typed objects
- require explicit return types and short purpose comments on functions
- reduce stringly-typed state and accessor-style APIs
- keep imports module-qualified instead of flattening many functions into local scope
- prefer fail-fast behavior over defensive fallback paths
- prefer type construction over `build_*` helper sprawl

## Scope

This standard applies to production Python code under `witwin/channel/core/scene/`, `witwin/channel/core/`, `witwin/channel/deterministic/`, `witwin/channel/montecarlo/`, and `witwin/channel/path/`, and to new architecture work in closely related support modules.

Pure math utilities in `witwin/channel/core/` may remain function-oriented. Public scene and solver workflows, runtime orchestration, stateful builders, schedulers, compilers, and accumulators should follow the object-ownership rules in this document.

## Core Rules

### 1. Prefer Short, Context-Aware Names

- File names must use `snake_case.py` and should usually contain one to three domain words.
- Function and method names should usually be a compact verb-noun pair.
- Local variable names should be short and readable, not prose.
- Do not repeat context that is already obvious from the package path, module name, class name, or type.

Examples:

- In `grid_diffraction.py`, prefer `accumulate_cells()` over `accumulate_grid_diffraction_cells_for_monitor()`.
- In `ReflectionQueue`, prefer `from_scene()` or direct construction over `build_reflection_scheduler_execution_queue()`.
- Prefer `tx_pos`, `rx_pos`, `face_idx`, `edge_dir`, `num_hits` over long names such as `transmitter_position_coordinates` or `current_intersection_face_index_value`.

If a name only becomes clear when it is very long, the abstraction is usually too broad and should be split.

### 2. Do Not Encode Architecture Failures Into Names

- Do not lengthen names just to describe every special case, backend, monitor type, and algorithm branch.
- If a file or function name starts reading like a design note, move the distinction into module boundaries, class ownership, or typed configuration.
- Avoid prefixes such as `current_`, `computed_`, `temporary_`, `helper_`, and `internal_` unless they distinguish two otherwise ambiguous values.
- Do not encode execution mechanics such as `batch`, `symbolic`, `manual`, `native`, `drjit`, or similar planner/runtime details into the primary function or module name when the owned domain concept is simpler.
- Name the module and entrypoint for the domain action first, such as `reflection.py` with `trace_reflection(...)`. Put execution strategy details in a short purpose comment, runtime metadata, or a local helper name if they truly matter.

Long names are not a substitute for architecture.

### 3. Shared Math And Utility Logic Must Have One Home

- Before adding a helper, check `witwin/channel/core/` first.
- Do not redefine existing array, geometry, polarization, angle, or material helpers locally when the canonical implementation already exists in `witwin/channel/core/`.
- If a helper is generic and has no domain-specific imports, move it into the appropriate `witwin/channel/core/` module instead of leaving a copy in `witwin/channel/core/scene/`, `deterministic/`, `montecarlo/`, or `path/`.
- When touching a module that contains a local duplicate of an existing utility, remove the duplicate as part of the change unless there is a concrete blocker.

The rule is simple: generic helper logic gets one canonical implementation.

### 4. Ban Thin One-Line Wrapper Functions

- Do not add local helper functions whose only job is to rename another function call.
- Do not add wrapper methods that only forward `self` into a module-level function with no added invariant, caching, validation, or ownership.
- Do not create alias helpers just to shorten one call site for one file.
- Minimize single-line inline helpers aggressively. If a helper is only one obvious expression and does not create a real abstraction boundary, inline it at the call site or move it to a canonical utility module.
- Do not introduce wrapper helpers for trivial predicates, counts, or attribute selection such as `_count_true(mask)`, `_count_positive(x)`, `_axis_component(vec, axis)`, or similar one-expression aliases. Write the direct `dr.count(...)`, comparison, or `getattr(...)` at the call site.

Examples of code that should not be added:

```python
def _zero_point():
    return zeros_point3()


def _build_paths(self):
    return build_paths(self)


def _count_positive(value):
    return int(dr.count(value > 0))


def _axis_component(vec, axis):
    return getattr(vec, axis)
```

A one-line wrapper is allowed only if it creates a real boundary by:

- enforcing a class invariant
- normalizing input once at an API edge
- attaching caching, validation, logging, or lifecycle behavior
- providing a stable public method over intentionally private internals

If none of those apply, call the real function directly.

### 5. Stateful Feature Logic Belongs In Objects

- Prefer classes or dataclasses when a feature owns state, configuration, caches, lifecycle steps, or multiple operations over the same data.
- If several functions repeatedly take the same bundle of arguments, that is a signal to introduce an owning object.
- If a workflow has setup, compile, execute, and finalize phases, model it as an object with explicit responsibilities instead of a flat chain of free functions.
- Shared payloads should be represented by named dataclasses or small classes, not loose dicts passed through many helpers.
- If a function currently returns a large dict with many named fields, treat that as a refactor signal. Prefer a typed result class or dataclass unless the payload is truly ad hoc and local to one short scope.
- Cross-module return values should not be "bags of stuff". Give them a stable type with stable field names.

Typical object-owning concepts include:

- compilers
- schedulers
- accumulators
- replay contexts
- validation runners

### 5A. Prefer Constructors Over `build_*` Helper APIs

- Do not introduce free functions named `build_*` when the result is fundamentally an instance of a concrete type.
- Do not use plain `build()` methods for object construction when direct construction or a named alternate constructor is clearer.
- Prefer constructing the owning type directly through `ClassName(...)`, a well-scoped `@classmethod`, or a small explicit factory attached to the type.
- If object creation needs validation or normalization, keep that logic inside the type's constructor boundary or a type-owned alternate constructor such as `from_scene(...)`.
- Reserve `build_*` verbs for rare cases where the result is not a stable domain object and the operation is truly procedural.

Prefer:

```python
queue = ReflectionQueue(scene, config)
runtime = CompiledScene(scene)
monitor = RadioMapMonitor.from_scene(scene, grid)
```

Over:

```python
queue = build_reflection_queue(scene, config)
runtime = build_compiled_scene(scene)
monitor = build_radio_map_monitor(scene, grid)
```

If the code keeps inventing `build_*` helpers, that usually means object ownership is still in the wrong place.

### 5B. Keep Result Metadata Compact And Non-Duplicative

- Result metadata should report stable semantic facts, not a dump of every local runtime variable.
- Do not mirror the same value across several metadata sections just because different helper layers produced it.
- If a field already has a clear home, keep it there once. Do not repeat batch sizes, sampling choices, or state-pool counts in multiple sibling dictionaries.
- Keep implementation-debug details such as temporary batch plans, memory snapshots, or internal bookkeeping out of the stable result contract unless a concrete consumer requires them.
- When metadata grows large, cut duplicated implementation detail before inventing another metadata wrapper type.

Prefer compact structures such as:

- `receiver_sampling` for receiver/grid sampling semantics
- `ray_sampling` for emitted-ray batching and sampling strategy
- `runtime_backends` for backend identity
- `path_counts` for result counts
- `monte_carlo` for Monte Carlo-specific controls that are not already represented elsewhere

### 6. Return Types Must Be Explicit

- Every non-trivial function and method must declare its return type.
- Public functions, cross-module helpers, and class methods should always have an explicit return annotation, even when the type seems obvious.
- Do not leave return types implicit for functions that construct payloads, normalize data, or drive runtime control flow.
- If a return type becomes too complex to annotate clearly, that is usually a sign to introduce a named type.

Examples:

```python
def from_scene(cls, scene: Scene) -> ReflectionQueue:
    ...


def spherical_angles(direction: bk.Vector3f) -> tuple[bk.Float, bk.Float]:
    ...
```

### 7. Every Function Needs A Short Purpose Comment

- Every function and method should start with a short one-line comment that states what it does.
- Keep the comment factual and compact. It should describe purpose, not rephrase the implementation line by line.
- Prefer a normal comment directly under the signature when that matches the file style.
- If the file already uses docstrings consistently, a one-line docstring is acceptable instead of a comment.

Examples:

```python
def from_scene(cls, scene: Scene) -> ReflectionQueue:
    # Create a reflection queue from the compiled scene description.
    ...


def project_field(field: JonesField, basis: PolarBasis) -> JonesField:
    """Project a Jones field onto the requested polarization basis."""
    ...
```

Do not add noisy comments such as:

- `# Set value`
- `# Loop over paths`
- `# Return result`

The comment should explain the function's role, not the obvious syntax.

### 8. Keep Pure Functions Where They Actually Fit

This document is not a blanket rule to convert every helper into a class.

Pure functions are still the right choice for:

- math primitives
- stateless geometry transforms
- small deterministic conversions
- narrow utility helpers in `utils/`

Use OOP for domain ownership and runtime structure, not for basic arithmetic.

### 9. Prefer Cohesive Modules Over Function Dumps

- One module should own one primary concept.
- Do not let a module become a bag of unrelated helper functions.
- If a file contains many functions that all manipulate the same state or same payload shape, introduce an owning type and move the behavior onto it.
- When a helper cluster is strongly coupled but still belongs to one clear feature file, prefer introducing a local owning class in that file before splitting the code into several new micro-modules.
- Do not respond to procedural sprawl by creating several thin sibling files that each hold one or two tightly coupled helpers. Reduce the function sprawl first by giving the shared state a real owner.
- When a solver file contains a standalone LoS, reflection, or diffraction block with its own inputs and outputs, move that block into a domain module under the solver's package (`witwin/channel/deterministic/...` or `witwin/channel/montecarlo/...`) instead of leaving it inside a generic `solver.py`.
- Avoid parallel implementations split across multiple similarly named files. Extend the primary module unless there is a real architectural boundary.

### 9A. Generic Functional Code Must Live In `witwin/channel/core/`

- Any reusable math, geometry, array, vector, angle, polarization, material, or generic data-shaping logic belongs in `witwin/channel/core/`.
- Do not leave generic functional helpers inside `witwin/channel/core/scene/`, `deterministic/`, `montecarlo/`, or `path/` once they are useful beyond one tightly local scope.
- If a helper has no domain-specific imports and no ownership over runtime state, move it to the appropriate `witwin/channel/core/` module.
- When a module accumulates several pure helper functions for generic numeric manipulation, treat that as a signal that the code is misplaced.
- If the right `witwin/channel/core/` module does not exist yet, add one there instead of keeping the helper in a feature module.

This rule is mandatory for generic functional code. Feature modules should own domain behavior, not shared math toolboxes.

### 10. Public APIs Must Expose Stable Nouns

- User-facing workflows should continue to center on stable objects such as `Scene`, the per-package `solve()` entrypoints, and the typed result payloads (`RadioMapResult`, `PathResult`).
- New public capabilities should attach to the existing object model instead of introducing procedural side-channel entrypoints.
- Internal architecture should support the public object model rather than bypass it with free-function pipelines.

### 11. Follow PEP 8 And PEP 257 Where They Strengthen Readability

- Package names should be short and all-lowercase.
- Module names should be short, all-lowercase, and `snake_case` when separators improve readability.
- Class names should use `CapWords`.
- Exception classes should end with `Error`.
- Function, method, argument, and local variable names should use `snake_case`.
- Constants should use `UPPER_SNAKE_CASE`.
- If a function has a public or cross-module contract, prefer a one-line docstring over a plain comment so tooling can validate it.
- One-line docstrings should use triple double quotes and imperative mood, such as `"""Build the reflection queue."""`.

The repository-specific naming rules in this document take precedence when they are stricter than generic PEP guidance.

### 12. Avoid Signature Smells That Hide Missing Objects

- Do not let functions grow long argument lists just because the code is still procedural.
- If several arguments travel together, introduce a parameter object, config dataclass, or owning class.
- Avoid positional `bool` arguments for behavioral switches. Prefer separate methods, an enum-like mode, or a keyword-only argument when a boolean is unavoidable.
- A function that needs many locals, many branches, or many statements is usually doing too much and should be split or moved behind an object boundary.
- If several private functions repeatedly manipulate the same state schema such as `state_arrays`, `state_store`, or another shared payload contract, that schema should become a named owning type with methods instead of staying as free functions over dicts.
- If private function names keep repeating the same long domain prefix such as `_discover_direct_tx_*`, `_store_direct_tx_*`, `_build_discovered_tx_*`, or similar, treat that as a refactor signal that the shared context belongs on a class and the repeated prefix should become the class name.

### 13. Avoid Stringly-Typed State And Control Flow

- Do not use raw `str` values as the primary representation for internal states, modes, phases, or control branches when the allowed set is known.
- Prefer `Enum`, `Literal`, dedicated config types, or small result classes over free-form string flags such as `"ready"`, `"compile"`, `"reflection"`, or `"native"`.
- If a branch depends on a small closed set of options, model that set explicitly instead of comparing repeated string constants across the codebase.
- User-facing APIs may accept strings at the outer boundary for convenience, but normalize them immediately into the internal typed representation.

Examples:

```python
class TraceMode(Enum):
    DETERMINISTIC = "deterministic"
    MONTE_CARLO = "monte_carlo"
```

Prefer:

```python
def run(mode: TraceMode) -> TraceResult:
    ...
```

Over:

```python
def run(mode: str) -> dict[str, object]:
    ...
```

### 14. Avoid Accessor-Style APIs When Native Python Semantics Are Better

- Do not add `get_*` and `set_*` methods when direct attribute access, a property, or a domain verb is clearer.
- Use nouns for stable state exposed as attributes or read-only properties.
- Use verbs for operations that perform work, mutate state with intent, or trigger lifecycle transitions.
- Reserve explicit accessor methods for cases that truly need validation, lazy computation, compatibility boundaries, or side effects that should not be hidden behind plain attribute syntax.

Prefer:

```python
queue.size
result.path_count
scene.device
runtime.compile()
accumulator.reset()
```

Over:

```python
queue.get_size()
result.get_path_count()
scene.get_device()
scheduler.do_build()
accumulator.set_reset_state()
```

### 15. Prefer Module-Qualified Imports Over Flattened Function Imports

- Do not pull a large set of functions directly into local scope with `from module import a, b, c, ...`.
- Prefer importing the owning module with a short alias and calling functions through that module namespace.
- Keep the module path visible at the call site so the code makes ownership obvious.
- If an internal import block pulls several private helpers from the same sibling module, stop flattening it. Import the module once with a clear alias such as `reflection_queue`, `rm_diag`, or `mc_common` and qualify the calls.
- Use `from ... import ...` only when importing:
  - a small number of stable public classes
  - explicit type aliases or protocols
  - constants that are intentionally treated as local names
- Do not use wildcard imports.
- Do not hide ownership by aliasing many unrelated modules to ambiguous one-letter names.

Prefer:

```python
from witwin.channel.core.numerics import arrays as dj_arrays
from witwin.channel.deterministic import runtime as det_runtime

points = dj_arrays.zeros_point3(count)
state = det_runtime.RuntimeState.from_scene(scene)
```

Over:

```python
from witwin.channel.core.numerics.arrays import zeros_point3, concat_points, repeat_int
from witwin.channel.core.runtime import TraceContext

points = zeros_point3(count)
state = RuntimeState.from_scene(scene)
```

This rule is especially important in math-heavy modules where many helper names look generic and collide easily.

### 15A. Do Not Re-Export Names That Collide With Submodule Names

- Do not re-export a function, class, or constant from `package/__init__.py` using the same name as a real sibling submodule.
- This creates ambiguous imports and can cause `import package.solver as mod` style imports to resolve to the re-exported object instead of the submodule.
- If the canonical implementation lives in `solver.py`, `runtime.py`, `field.py`, or a similar submodule, callers should import from that submodule directly.
- Prefer leaving `__init__.py` empty or minimal over adding a thin package-level alias that hides the real module boundary.

### 16. Prefer Fail Fast Over Defensive Fallbacks

- Do not add defensive fallback paths just to keep execution alive when the primary contract is violated.
- Do not hide architecture or typing mistakes behind default values, silent recovery, alternate branches, or compatibility shims.
- Do not add backward-compatibility code, legacy adapters, or "best effort" behavior unless explicitly requested.
- If a required invariant, type contract, runtime dependency, or scene assumption is broken, raise a clear error at the boundary and stop.
- Tests must validate the intended architecture, not a watered-down fallback path that exists only to keep the suite green.

Prefer:

```python
if monitor is None:
    raise ValueError("PathMonitor is required for path replay.")
```

Over:

```python
if monitor is None:
    return {}
```

Prefer:

```python
if mode is not TraceMode.DETERMINISTIC:
    raise ValueError(f"Unsupported trace mode: {mode!r}")
```

Over:

```python
if mode not in {"deterministic", "monte_carlo"}:
    mode = "deterministic"
```

Allowed defensive checks are narrow and explicit:

- validating public inputs at the outer boundary
- failing early with a precise exception
- rejecting unsupported states before entering a hot path

Defensive code is not the same as correct validation. Validate once, then run the intended path without hidden rescue branches.

## Refactor Heuristics

Treat the current design as too procedural and refactor toward object ownership when one or more of the following is true:

1. Three or more functions share the same prefix or repeat the same long argument list.
2. A dict-like payload is created in one function, mutated in several others, and finally consumed elsewhere.
3. A module needs helper closures or many tiny wrappers just to pass shared context around.
4. A workflow has an obvious lifecycle such as build, validate, execute, collect, and summarize.
5. A name keeps growing because the code has no stable owner for the behavior.
6. A function returns a large dict because no result type was defined.
7. A function signature grows because related state was never promoted into an object.
8. The code compares the same small set of string status values in many places.
9. A class exposes mostly `get_*` and `set_*` methods instead of meaningful attributes and verbs.
10. A module starts with long `from ... import ...` lists because local names no longer show ownership clearly.
11. A function contains fallback branches that only exist to keep invalid states limping forward.
12. Object creation mainly happens through free `build_*` helpers instead of type constructors.
13. A feature module has started acting like an ad hoc math or array utility file.
14. Many single-line helpers exist only to rename obvious expressions.

## Review Checklist

Before merging Python architecture changes, verify:

1. Did the new file, function, and variable names stay compact and context-aware?
2. Did we avoid repeating package, module, class, or type context in the name?
3. Did we reuse existing helpers from `utils/` instead of redefining them?
4. Did we avoid new one-line wrapper helpers with no real behavioral value?
5. Did we replace loose cross-module dict payloads with classes or dataclasses where appropriate?
6. Does every non-trivial function and method declare an explicit return type?
7. Does every function start with a short purpose comment or one-line docstring?
8. Did we move stateful multi-step workflows toward classes or dataclasses where appropriate?
9. Did we keep truly stateless math and utility code as plain functions?
10. Did we avoid long positional argument lists and boolean-flag signatures where a real object boundary was needed?
11. Did public or cross-module functions use one-line docstrings where tooling should enforce the contract?
12. Did we avoid raw string state flags where a typed mode or enum was more appropriate?
13. Did we avoid accessor-style `get_*` / `set_*` APIs unless they added real behavior?
14. Did we keep function ownership visible by preferring module-qualified imports over flattened import lists?
15. Did we avoid free `build_*` helpers when a constructor or type-owned constructor was the right API?
16. Did we aggressively remove pointless single-line inline helpers?
17. Did reusable math, geometry, and array logic live in `utils/` instead of feature modules?
18. Did we fail fast on invalid states instead of adding silent fallback or backward-compatibility behavior?
19. Did the resulting module become easier to navigate than before?

## Suggested Validation Tooling

Use automated checks to enforce as much of this document as possible.

### Primary Toolchain

- `ruff format` for stable formatting.
- `ruff check` for naming, docstring, annotation, and structural lint checks.
- `mypy` or `ty` for return types, typed payloads, and annotation coverage.
- `pytest` for behavior-preserving refactors.

### Recommended Ruff Coverage

Prefer enabling these rule families:

- `N` for naming rules such as module, function, and variable style.
- `ANN` for missing annotations on function signatures.
- `D` for docstring shape and one-line docstring conventions.
- `F401` and `I` for import hygiene and stable import organization.
- `C90` for McCabe complexity.
- `PLR0913`, `PLR0914`, `PLR0915`, `PLR0912` for oversized function signatures and bodies.
- `FBT` for boolean-trap signatures and call sites.

Useful enforcement examples:

- `N999` catches invalid module names.
- `N802` catches invalid function names.
- `ANN201` and related `ANN` rules help enforce return annotations.
- `D401` checks that one-line docstrings use imperative mood.
- `C901` flags functions with excessive control-flow complexity.
- `PLR0913` flags too many function arguments.
- `PLR0915` flags too many statements.
- `FBT001` and `FBT003` flag boolean-trap APIs and call sites.

### Recommended Type-Checking Baseline

For `mypy`, a practical baseline is:

- `disallow_untyped_defs = true`
- `check_untyped_defs = true`
- `warn_return_any = true`
- `disallow_any_generics = true`
- `warn_unused_ignores = true`
- `warn_redundant_casts = true`

`ty` is also a viable project-wide type-checker if the team wants a faster integrated language-server workflow. Use one primary type checker consistently instead of mixing multiple conflicting gatekeepers in CI.

### Acceptance Workflow

For architecture and naming changes, the preferred validation order is:

1. `ruff format`
2. `ruff check`
3. `mypy` or `ty`
4. targeted `pytest`

Run broader pytest coverage after the targeted checks if the refactor crosses major runtime boundaries.

## Relationship To Other Standards

- `docs/dev/standards/21-python-runtime-typing-standard.md` owns runtime typing, normalization, and anti-duck-typing rules.
- This document owns package structure, naming discipline, utility ownership, thin-wrapper bans, and the OOP-vs-function placement rule for Python code.
