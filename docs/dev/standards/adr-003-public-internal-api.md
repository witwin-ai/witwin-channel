# ADR-003: Public and internal API

Status: Accepted; Stage-I owner and consumer migration amended by ADR-034.

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

Stage-I Phase 3 adds `witwin.channel.propagation.consumer` as an explicit
stable public module with contract version 1. Its stable surface is the
module's declared `__all__`; it does not widen the package-root export set.
The consumer surface owns typed endpoint/request/result/convention/capability
contracts plus `evaluate` and fixed-topology `reevaluate`. Internal
`EvaluatedPaths`, solver modules, failure transactions, native handles,
compiled-scene resources, and native implementation helpers remain internal
and are not re-exported through that boundary.

Contract version 1 is intentionally narrow. A breaking schema or semantic
change increments `CONTRACT_VERSION`, updates the public snapshot, capability
matrix, migration note, and package-neutral conformance tests atomically, and
requires consumers to move to the new contract. Channel does not keep two
production schemas or add a compatibility shim.

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
