# ADR-006: Extension developer override

Status: Accepted for the modular hardening migration.

## Decision

The native extension loads only from inside the installed package by default.
Any developer override must be explicit and must validate the complete declared
ABI fingerprint before the extension is used.

## Context

Implicit global loading or artifact-directory search could bind the runtime to a
stale or mismatched build without notice. An override therefore has to name the
extension path explicitly and pass full build-identity validation, so a
developer build is used only when it exactly matches the declared fingerprint.
Silent ABI downgrade is rejected rather than tolerated.

## Implementing artifact

`runtime.py` performs the explicit override and fingerprint check,
guarded by `tests/kernels/test_extension_loading.py`.
