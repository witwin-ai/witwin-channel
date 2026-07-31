# ADR-047: RayD 0.8 source and integration boundary

- **Status:** Accepted.
- **Date:** 2026-07-30.
- **Decision ID:** `rayd-0.8-source-boundary`.
- **Scope:** Channel's RayD source lock, typed C++ boundary, shared device source,
  release build, and build identity.

## Context

RayD 0.8.0 intentionally removes the former `backends/torch` source layout and
single-header integration identity used by Channel 0.4.0. RayD ADR-0040 and
ADR-0041 replace it with a canonical `torch/`, `include/`, and `src/` source
tree. The typed Torch API is now an eight-header set rooted at
`include/rayd/integration.h`, with API version 8 and the unchanged identity
`rayd.torch.integration`. The Torch-only field-transport derivative source and
its shared transmission device contract are no longer advertised as public typed
headers; they are bundled at `src/field_transport_ad.cuh` and
`src/transmission_device.cuh` for same-graph native consumers.

Channel cannot adopt the 0.8 release by changing only a package version. Doing
so would either accept an incomplete identity, reach deleted paths, or compile
against an unpinned private source. A compatibility include root or forwarding
header would violate both RayD's hard break and Channel's single-owner policy.

## Decision

Channel 0.5.0 pins the released RayD 0.8.0 commit
`c7a99979d0fdcc67b2ec8a12246a7df597603409` and the exact metadata published in
the `rayd-torch==0.8.0` wheel:

- repository `https://github.com/Asixa/RayD`;
- source-manifest SHA-256
  `c4fb39b27eeea2588615d1493c6fe3f4fc0202017341fa320dafdcacb595b1c1`;
- integration kind `source-header-set-sha256`;
- entrypoint `include/rayd/integration.h`;
- API version 8 and identity `rayd.torch.integration`;
- aggregate normalized header-set SHA-256
  `db48cdb91b31c00a14259f912f8b504eb2485a031b036c6f79688cb5452670c4`.

The lock records all eight sorted header paths and their normalized SHA-256
values. Package discovery still validates distribution identity, RECORD
ownership, the complete source manifest, the absence of extra files, and every
source byte. Explicit Git checkout validation still validates commit, remote,
and clean state, and additionally validates every normalized integration header
against the same lock.

CMake consumes `torch/CMakeLists.txt` and the existing
`rayd_torch_native_core` target. Typed bridges include `rayd/integration.h`.
Channel-owned fused CUDA kernels may include exactly
`src/field_transport_ad.cuh` and `src/transmission_device.cuh` from the
lock-validated source root. These direct source dependencies are the
RayD-accepted downstream activation seam from ADR-0040/0041; they are not a
second typed boundary and do not authorize other private RayD includes.

No compatibility path, forwarding header, copied derivative implementation,
runtime fallback, second dispatcher, additional launch, synchronization,
host/device transfer, or numerical-order change is introduced.

## Release ordering

`witwin` 0.4.0 is released first. Channel's release workflow pins that release
commit and installs it from the checked-out Core source while building and
smoke-testing Channel 0.5.0. The public Channel dependency remains the exact
`witwin==0.4.0` distribution requirement.

## Superseded clauses

This decision supersedes only the old RayD version, include spelling,
single-header SHA, source-tree path, and API-version clauses in ADR-002,
ADR-023, ADR-024, ADR-026, ADR-027, ADR-029, ADR-030, ADR-034, and ADR-035.
Their ownership, failure, fusion, launch, and numerical contracts remain in
force. Historical audit records retain the RayD pins and paths they measured.

## Verification

The activation must pass the lock/schema, explicit-checkout CMake,
package-source mutation, typed-boundary, native-owner, release-policy, full
Python test, native build, wheel layout, and isolated wheel smoke suites. The
published release is complete only after both platform wheels and the PyPI
upload succeed and an isolated install reports the pinned RayD/Core identities.
