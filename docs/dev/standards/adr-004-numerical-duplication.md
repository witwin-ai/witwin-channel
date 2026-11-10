# ADR-004: Numerical duplication

Status: Accepted for the modular hardening migration.

## Decision

Without exactness evidence, primal and AD numerical expressions stay duplicated
and are maintained in lockstep; deduplication is not pursued for abstract
elegance alone.

## Context

Primal, JVP, and VJP paths often repeat the same floating-point arithmetic, but
merging them can silently change evaluation order, rounding, or inline
attributes. Retained duplicates are kept in lockstep by their primal/JVP/VJP
tests, which compare the companions numerically, so the pair is edited together
and cannot drift. (Until 0.6.0 each region was also recorded in a hand-kept
classification ledger keyed by content hash; that ledger is archived, see
`docs/dev/archive.md`.) A duplicate may only be collapsed by a separate change that
proves output exactness and unchanged evaluation order.

## Implementing artifact

`ci/check_duplication.py` regenerates the exact-token duplication report on every
run and fails when total duplicate coverage exceeds its tighten-only ceiling.
