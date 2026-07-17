# ADR-003: Public and internal API

Status: Accepted for the modular hardening migration.

## Decision

The package root and the four solver entry points are the stable public API.
Everything under `core.*` is classified by an explicit manifest and is not
guaranteed to stay compatible for every incidental import.

## Context

A frozen snapshot pins the promised public surface so downstream code can rely
on it, while internal modules stay free to move as the architecture hardens.
The snapshot test fails on any unreviewed change to the root exports, forcing a
deliberate manifest update rather than silent public-API growth. This bounds the
compatibility promise to what the manifest declares.

## Implementing artifact

`ci/public-api-snapshot.json`, enforced by `tests/test_public_api_snapshot.py`.
