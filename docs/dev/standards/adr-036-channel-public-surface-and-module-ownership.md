# ADR-036: Channel public surface and module ownership

- **Status:** Accepted
- **Date:** 2026-07-24
- **Kind:** Public API, module ownership, and documentation decision
- **Supersedes:** the ADR-034 clause that had Channel's package root re-export
  the Core world model
- **Related:** ADR-003, ADR-007, ADR-033, ADR-034

## Context

Stage I moved the logical world model to `witwin.core` and switched every
Channel owner onto it. The Channel package root kept re-exporting the eleven
Core world types so existing call sites would keep working through the switch.

That left three problems behind, all of them about names rather than behavior.

`Scene`, `Structure`, `PhysicalMaterial`, and the rest had two import paths
each. Core additionally exported `Material` as a plain alias of
`PhysicalMaterial`, so the material contract had four spellings across the two
packages. ADR-033 and CLAUDE.md both forbid compatibility aliases, and this
qualified whether or not it originated as one.

`witwin.channel.core` collided by name with `witwin.core` while being neither a
domain owner nor a sibling of it. It was a 1206-line grab-bag imported from
thirty modules and absent from the CLAUDE.md ownership list. One of its
members, `core/kernels/extension.py`, described itself in its own docstring as
a compatibility facade and was the package root's only route to `build_info`;
it was carried as allowlisted import debt `boundary-001`.

`witwin.channel.physics` shipped a NumPy CPU reference oracle inside the
production wheel. CLAUDE.md allows CPU reference implementations only under
`tests/`. Its `physics/oracle.py` facade re-exported private helpers and
rewrote `__module__` on thirteen objects so they would report the facade as
their owner.

## Decision

### The package root exports only what Channel owns

`witwin.channel` exports `build_info`, `capabilities`, `pipeline_cache_key`,
`runtime_diagnostics`, `Complex3State`, and `JonesState`. Nothing else.

The world model is imported from `witwin.core`. Channel does not re-export it.
Each world type has exactly one import path, and the ADR-034 clause requiring
root world exports to resolve to Core is replaced by this stricter rule: the
root does not publish them at all.

Core drops the `Material` alias. `PhysicalMaterial` is the only public name.

This is breaking and intentional. Stage I is already a breaking replacement and
its artifacts are not yet published, so this is the cheapest moment to make the
surface honest.

### `witwin.channel.core` is dissolved

Every module moves to the owner that its dependencies already imply. Runtime
identity and budgets go to `runtime`; endpoint geometry, edge policy, and the
scene-leaf AD seam go to `scene`; diffraction edge state goes to
`propagation.geometry`. The compatibility extension shim is deleted, and
`deployment` publishes `build_info` as the same object the runtime loader
defines rather than as a second wrapper. That repays `boundary-001` and leaves
the package root with no internal import debt at all.

Four value modules that several domains and the public root all need stay at
the package root, because the `public_init_internal` boundary forbids the root
`__init__` from importing `runtime`, `propagation`, or any `kernels` package:
`constants`, `field_state`, `components`, and `tensor_math`.

The import-graph checker gains a `dissolved_module_dependency` rule so the
namespace cannot be recreated.

### `witwin.channel.physics` is dissolved

`physics.conventions` becomes `witwin.channel.constants`, which is also the
single owner of the phase convention that solver metadata and the consumer
contract quote. The oracle facade is deleted with its `__module__` rewriting.
The reference oracle itself moves to `tests/reference/em_oracle.py`.

Isolation becomes structural rather than gate-enforced: the import checker only
walks the shipped package, so production cannot reach the oracle at all. The
`oracle_production_dependency` rule is retired accordingly.

### Consumer contract vocabulary and validation

`propagation.consumer.contracts` is the single source of truth for accepted
values. Each dimension is a `Literal` alias with a matching frozen set, and
`capabilities()` is exported so a caller can check a combination before
building a request instead of learning it from a rejection.

`PropagationRequest` and `FixedTopologyRequest` validate their own structure at
construction, matching every other contract in that module. Scene-dependent
checks stay in `evaluate` and `reevaluate` and still run before any native work.

Contract version 1 drops two things it declared but never implemented,
`frequency_offsets_hz` and `PropagationConvention.frequency_offset_law`, and one
redundancy, the singular `interaction_position_m` and `interaction_normal`
fields that were column 0 of the plural tensors. Frequency offsets will arrive
with a `CONTRACT_VERSION` bump.

### One capability record per question

`witwin.channel.capabilities()` remains the solver-level manifest and keeps
`scattering` in its component list, because scattering is live in all four
solvers. It embeds the consumer contract record under `propagation_consumer`,
generated from `propagation.consumer.capabilities()` rather than restated, so
the broader and narrower answers cannot drift.

## Non-decisions

This ADR changes no physics, numerical order, launch configuration,
synchronization, reduction order, tape lifetime, or result schema. Every module
move preserves its implementation exactly.

Two things were found and deliberately left alone:

- Core's `VACUUM_PERMITTIVITY` (8.8541878128e-12, CODATA measured) and
  Channel's `EPS0` (derived from the pre-2019 exact definition,
  8.854187817620389e-12) differ in the ninth significant digit. Aligning them
  changes solver output and needs its own numerical ADR.
- `witwin.core.identity` allocates from a process-global counter, so
  unreserved IDs are not reproducible across processes. The behavior is
  unchanged here; the two allocation modes and their consequences are now
  documented, and the reproducible `reserve_*_id` path has direct tests.

## Consequences

Callers importing world types from `witwin.channel` must import them from
`witwin.core`, and callers using `witwin.core.Material` must use
`PhysicalMaterial`. Both are mechanical one-line changes; the migration note in
`docs/dev/replacement/channel-migration.md` lists them with every module move.

Radar has not started Stage II, so it absorbs these changes when it pins the
Stage-I artifacts rather than as a separate migration.
