# Plan 13 Phase 0 audit summary

This directory now contains a reproducible, source-scanned Phase 0 baseline for
Channel Native `a741f8d2` and locked RayD `346416f8`.
The immutable ADR-009 file `phase9-native-owner-inventory.json` was hashed and
read, but not modified.

## Frozen counts

- `_channel_native` bindings: **211**.
- Current numerical ownership: **18 RayD**, **191 Channel**, and **2 layered Channel-operation/RayD-primitive** records.
- Legacy `RayDN/raydn/uses_raydn_native` scan: **73 files**, **946 matching lines** across production source/build/CI scope.
- RayD source integration header: **20 extern-C entries**, with no detected typed C++ v2 namespace, named result structs, or RAII scene type at the frozen baseline.
- Channel legacy indirection: **21 function-pointer getters** in `native/channel_native/rayd/common.cpp`; two point to Channel diffraction-discovery CUDA entries and the rest point to RayD extern-C entries.
- Candidate owner moves: **6 transmission**, **3 pure-wedge diffraction**, and **17 scattering runtime** bindings. `scattering_event_probabilities` remains Channel-owned.
- Shared RF helper scan: **129 inline/device helper declarations** across **8** headers.
- Runtime capture: **38 frozen cells** across all four solvers plus Path/
  Deterministic JVP/VJP, measured in two independent processes with one warmup
  and seven steady repeats. One deterministic rough-scattering-ensemble cell
  is explicitly excluded because its exact hash differed across processes;
  both fingerprints remain recorded instead of being hidden by a tolerance.

## Artifact map

- `phase13-current-native-owner-inventory.json`: all 211 symbols, separate ABI and numerical ownership, source definitions, callers, tests and disposition.
- `phase13-migration-delta.json`: empty Phase 0 transfer baseline.
- `phase13-symbol-delta-ledger.json`: planned renames/transfers plus unresolved four-part deletion audits.
- `phase13-rayd-integration-inventory.json`: legacy terms, bridge/getter map, RayD extern-C entries and typed-v2 absence evidence.
- `phase13-live-dead-public-internal-inventory.json`: conservative static/E2E/public classification; no-static is never treated as deletion authorization.
- `phase13-transmission-contracts.json`: the six complete-family contracts and frozen fusion/tape/compile constraints.
- `phase13-diffraction-family-matrix.json`: operation-family ownership, including layered geometry and MC tape producer/consumer distinctions.
- `phase13-scattering-bindings.json`: 17 conditional moves plus the one retained event-policy binding.
- `phase13-shared-rf-dependency-graph.json` and `phase13-shared-rf-helper-ledger.json`: include closure and helper-level owner/mirror/compiler evidence.
- `phase13-baseline-evidence.json`: toolchain/build identity and digest index
  for the immutable runtime baseline under
  `docs/dev/baselines/a741f8d2a0ff5ba353be60584f21ee7f910f03ad/runtime/`.

## Known Phase 0 limits

Static references establish positive liveness evidence but cannot prove a symbol
dead. Every unresolved diffraction legacy binding remains marked for the four
checks required by Plan 13: static caller, dynamic binding, public import and
real BDPT E2E. Runtime exact outputs, Nsight launch/sync/memcpy/peak-memory and
PTX/SASS evidence are separate executable baselines; this source-only generator
does not fabricate them.
