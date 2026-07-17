# ADR-004: Numerical duplication

Status: Accepted for the modular hardening migration.

## Decision

Without exactness evidence, primal and AD numerical expressions stay duplicated
and are maintained in lockstep; deduplication is not pursued for abstract
elegance alone.

## Context

Primal, JVP, and VJP paths often repeat the same floating-point arithmetic, but
merging them can silently change evaluation order, rounding, or inline
attributes. Each retained duplicate is recorded in the classification ledger
with an owner, a reason, and its lockstep tests, so the pair is edited together
and cannot drift. A duplicate may only be collapsed by a separate change that
proves output exactness and unchanged evaluation order.

## Implementing artifact

The exemptions in `docs/dev/audit/duplication-classification.json`, enforced by
`ci/check_duplication.py`.
