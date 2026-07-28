# ADR-001: Python/native dispatch

Status: Accepted for the modular hardening migration.

## Decision

The project keeps the pybind11 binding registries and the current custom
autograd contracts, isolating only the Python/native boundary; it does not
migrate to `torch.library` dispatch during this cycle.

## Context

`torch.library` would standardize op registration but would also re-plumb every
kernel entry point and its tape ownership without changing numerical behavior.
The migration goal is a clean boundary, not a dispatch rewrite, so the existing
pybind11 modules stay the ABI surface and `runtime.py`
remains the single owner of dispatch-state validation. Revisiting `torch.library`
is deferred to a separate, evidence-backed change.

## Implementing artifact

The pybind11 binding registries and the `runtime.py` status
quo carry this decision; `tests/runtime/test_autograd_contracts.py` guards it.
