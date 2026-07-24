# ADR-003: Public and internal API

Status: Accepted; Stage-I owner migration amended by ADR-034.

## Decision

The package root and the four solver entry points are the stable public API.
Everything under `core.*` is classified by an explicit manifest and is not
guaranteed to stay compatible for every incidental import.

ADR-034 approves an intentional breaking owner migration. During Stage-I
Phase 2, root `Scene`, `Structure`, logical material, and logical endpoint
exports move directly to `witwin.core` contracts. Old Channel logical owners
and compatibility facades are deleted in the same change as the four solver
caller switch. The snapshot remains unchanged until that atomic cutover; it
must not advertise the new identity before the implementation is active.

Stage-I Phase 3 adds the solver-neutral propagation consumer module as an
explicit stable public module. Internal `EvaluatedPaths`, solver modules,
failure transactions, native handles, and compiled-scene resources remain
internal and are not re-exported through that consumer boundary.

## Context

A frozen snapshot pins the promised public surface so downstream code can rely
on it, while internal modules stay free to move as the architecture hardens.
The snapshot test fails on any unreviewed change to the root exports, forcing a
deliberate manifest update rather than silent public-API growth. This bounds the
compatibility promise to what the manifest declares.

## Implementing artifact

`ci/public-api-snapshot.json`, enforced by `tests/test_public_api_snapshot.py`.
ADR-034 is the owner, frequency, compact-cardinality, and migration decision
for the two approved Stage-I snapshot changes.
