# ADR-008: Enumerated propagation as a discrete-path oracle

Status: Accepted.

## Context

Plan 08 section 5.4 fixes the default role of enumerated propagation: it is a
path/deterministic-only, deterministic evaluation stage. Under that default the
Monte Carlo solvers, both `montecarlo.basic` and `montecarlo.bdpt`, must not call
`evaluate_enumerated_paths`, and the import-graph gate encodes that as the
`mc_enumerated_dependency` rule.

The plan-05 BDPT hybrid estimator predates section 5.4. Its delta-specular and
coupled discrete connection strategies need the exact deterministic enumerated
evaluation, including `fields.path_gain`, to build the discrete connection
samples that the estimator combines with its stochastic contributions. It obtains
them by calling `evaluate_enumerated_paths` from
`montecarlo.bdpt.pipeline`.

The dependency was invisible to the gate. The import edge targets the
`witwin.channel.propagation` package, whose `__init__` re-exports
`evaluate_enumerated_paths` from `propagation.enumerated.engine`. The
`mc_enumerated_dependency` rule matches `propagation.enumerated`, not the package
facade, so the edge never fired: the gate was green while the architectural
constraint was violated. The import-graph checker now resolves package re-exports
to the defining module, which makes the real edge
`montecarlo.bdpt.pipeline -> propagation.enumerated.engine` visible.

## Decision

BDPT may consume `evaluate_enumerated_paths` read-only as a black-box
discrete-path oracle, under the following restrictions:

- Only the public entry `evaluate_enumerated_paths` may be called. BDPT does not
  import enumerated submodules (`propagation.enumerated.*`) directly.
- The call is read-only. BDPT treats the returned `EvaluatedPaths` as an opaque
  discrete-path evaluation and does not mutate enumerated state.
- No BDPT-specific parameters are added to the enumerated engine. The oracle
  keeps its deterministic path/enumerated signature; BDPT adapts to it.
- `montecarlo.basic` retains zero enumerated dependency. This exception is bound
  to `montecarlo.bdpt.pipeline` alone.

## Rationale

Duplicating the enumerated evaluation inside BDPT would create a parallel
implementation of identical physics, including delta-specular and coupled
discrete field evaluation. That contradicts the monorepo rule against duplicate
implementations in parallel files and would split a single source of numerical
truth across two owners. Consuming the deterministic enumerated evaluation as a
black-box oracle keeps one implementation of that physics and confines the
architectural exception to a single, named consumer.

## Enforcement

The import-graph checker canonicalizes package re-exports to their defining
module before classification, so the BDPT edge is scored as
`montecarlo.bdpt.pipeline -> propagation.enumerated.engine` and fires
`mc_enumerated_dependency`. A single allowlist entry, `mc-enum-001` in
`ci/import_graph_allowlist.json`, admits exactly that source module and rule and
records this ADR as its justification. The allowlist keeps the monotonic-decrease
shape: the entry can only be removed when the dependency is resolved, and no
other source may inherit it.

## Revisit condition

Re-evaluate this decision if BDPT ever needs changes to the enumerated engine
itself, rather than read-only consumption of its public entry, or when a
dedicated discrete-connection service exists that BDPT can depend on without
reaching into enumerated propagation. Either event retires the exception and the
`mc-enum-001` allowance is removed.
