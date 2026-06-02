# Python Runtime Typing Standard

Status: Active
Category: Standard
Last reviewed: 2026-05-20

## Purpose

This document defines the Python-side typing and boundary rules for the `witwin.channel` runtime.

The goals are:

- keep runtime internals DrJit-native
- keep input and output contracts explicit and deterministic
- avoid architecture drift toward reflection-heavy or duck-typed code
- reduce hot-path overhead from repeated capability probing and dynamic transport conversion

This standard applies to `witwin/channel/core/scene/`, `witwin/channel/core/`, `deterministic/`, `montecarlo/`, `path/`, and the result adapters they expose, unless a narrower standard already overrides a topic.

## Non-Negotiable Rules

1. Public inputs may accept a small set of explicit `*Like` forms, but internal runtime code must normalize them once and then operate on a single concrete type.
2. Public outputs and internal cross-module payloads must use explicit typed structures. Do not use loose `object` payloads as the contract.
3. Runtime internals are DrJit-native by default. Do not use Torch, NumPy, or DLPack as an internal transport layer between Python runtime modules.
4. Do not use `hasattr()` or `getattr()` to drive normal runtime control flow once a value has crossed an internal boundary.
5. Do not rely on broad duck typing in core runtime code. Prefer concrete classes, typed dataclasses, or narrow `Protocol` definitions.
6. Do not use `object.__setattr__()` for normal runtime/config/monitor construction. Prefer normal assignment in `__post_init__()` or explicit factory normalization.
7. `isinstance()` is allowed only at explicit boundary normalization points, adapter edges, or versioned interop shims. It is not the default internal dispatch mechanism.
8. If a code path is performance-sensitive, do not repeatedly probe optional attributes or optional backends inside the hot loop. Resolve the path once before entering the loop.

## Boundary Rules

### Rule A: Normalize Once

At a public API boundary, convert accepted user input into the concrete internal form immediately.

Examples:

- transmitter positions become `bk.Point3f`
- path/result tensors become explicit wrapper types
- reflection-path payloads become typed dataclasses instead of free-form dicts

After normalization, downstream code should not continue checking whether the value is a Torch tensor, tuple, dict, or custom object.

### Rule B: Keep Boundary Checks Local

Reflection-style checks are allowed only in a small number of places:

- public API entrypoints
- backend adapters
- optional dependency detection during setup
- result conversion layers that intentionally expose Torch-facing data

Do not scatter the same capability checks through multiple runtime layers.

### Rule C: Prefer Explicit Types Over Open Payloads

Do not pass payloads around as:

- `object`
- `dict[str, object]` when a stable schema exists
- ad hoc objects that are later inspected with `hasattr()`

Prefer:

- dataclasses
- typed wrapper classes
- narrow immutable payload objects
- `Protocol` only when the boundary is intentionally polymorphic and the required surface is small and stable

## Reflection And Duck Typing Rules

### `hasattr()` / `getattr()`

Allowed:

- optional extension detection
- compatibility glue around third-party objects
- startup-time feature detection outside the hot path

Not allowed:

- deciding normal runtime payload shape after normalization
- selecting solver behavior based on loosely shaped objects
- probing for field names such as `"path_gain"` or `"tx_association_map"` instead of using typed result classes

### `isinstance()`

Allowed:

- converting public API input into a canonical internal representation
- adapter dispatch between a small number of known concrete types
- targeted validation of user-facing config payloads

Not allowed:

- repeated internal branching on representation after normalization
- using long `isinstance()` chains as a substitute for a stable interface design

### `object.__setattr__()`

Allowed:

- rare low-level metaprogramming cases where normal dataclass construction truly cannot work

Not allowed:

- normal immutable-config construction
- monitor/config setup that can be handled by `@dataclass(slots=True)` plus `__post_init__()`

## Internal Data Contract Rules

### Inputs

- Public functions should document accepted inputs explicitly.
- Internal helpers should take the normalized concrete type, not the original union of user-facing types.
- If a helper still needs many alternative input forms, that is a sign the normalization boundary is in the wrong place.

### Outputs

- Function return types must be concrete and documented.
- Result containers must have stable field names and stable value types.
- Do not return one of several unrelated payload shapes and expect downstream code to inspect attributes to figure out which one it received.

### Cross-Module Payloads

- If multiple modules share a payload, promote it to a named type.
- If the payload is performance-sensitive, keep the fields aligned with the actual runtime layout and avoid convenience dict wrappers.

## DrJit-First Runtime Rule

The repository already requires DrJit-native runtime internals. This standard makes the Python-side implication explicit:

- internal geometry, tracing, replay, and accumulation logic should consume and return DrJit-native structures
- Torch is acceptable at explicit public integration boundaries and result adapters only
- conversion to Torch inside reflection discovery, replay preparation, solver scheduling, or other hot runtime internals is not allowed

If a new implementation needs Torch in the middle of a runtime path to stay manageable, treat that as incomplete architecture and either:

1. move the conversion to an explicit boundary layer, or
2. redesign the internal contract so the runtime stays DrJit-native

## Review Checklist

Before merging runtime Python changes, verify:

1. Are accepted public input types listed explicitly?
2. Is normalization done once near the boundary?
3. Does the internal path use one concrete representation after normalization?
4. Did we avoid new `hasattr()` / `getattr()` / repeated `isinstance()` in runtime internals?
5. Did we avoid new Torch or NumPy transport inside the runtime hot path?
6. Are outputs represented by typed structures rather than loosely shaped payloads?
7. Did we avoid `object.__setattr__()` for ordinary config or monitor construction?

## Relationship To Other Standards

- `docs/dev/standards/30-cuda-kernel-development-guide.md` owns the kernel-side zero-copy and no-bridge rules.
- `docs/dev/standards/31-cuda-kernel-migration-workflow.md` owns the C++/CUDA migration sequence.
- This document owns the Python-side typing, normalization, and anti-duck-typing rules for the runtime.
