# ADR-005: Git history

Status: Accepted for the modular hardening migration.

## Decision

Index cleanup (removing tracked build artifacts and oversized blobs) is
mandatory. Rewriting existing history is off by default and requires a
coordinated maintenance window before it is attempted.

## Context

Committed generated files and large blobs bloat the working tree and slow every
clone, so the hygiene gate rejects them going forward. Rewriting published
history, by contrast, invalidates every outstanding checkout and is disruptive
to all contributors, so it is treated as an exceptional, scheduled operation
rather than routine cleanup. The large-object report documents what history
already carries.

## Implementing artifact

`ci/check_repository_hygiene.py` for the ongoing gate and
`docs/dev/repository-large-object-report.md` for the historical inventory.
