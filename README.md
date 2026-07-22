# witwin-channel

DrJit-free Torch/CUDA RF channel runtime under the `witwin.channel`
namespace. Python owns solver policy and typed contracts; one validated
`_channel` extension owns production CUDA/RayD execution.

## Public API

The curated surface is `witwin.channel` plus the `path`,
`deterministic`, `montecarlo.basic`, and `montecarlo.bdpt` solver packages.
Exact exports are frozen in `ci/public-api-snapshot.json`; other domain modules
are internal unless their README says otherwise.

## Architecture

- [`core`](src/witwin/channel/core/README.md): small shared value
  contracts and build metadata.
- [`scene`](src/witwin/channel/scene/README.md): models, compilation,
  stores, caches, and RayD lifetime.
- [`materials`](src/witwin/channel/materials/README.md): material models,
  ABI encoding, and electromagnetic evaluation.
- [`propagation`](src/witwin/channel/propagation/README.md): topology,
  geometry, fields, and enumerated stages.
- [`path`](src/witwin/channel/path/README.md) and
  [`deterministic`](src/witwin/channel/deterministic/README.md): coherent
  field solvers.
- [`montecarlo`](src/witwin/channel/montecarlo/README.md): basic and BDPT
  stochastic solvers.
- [`physics`](src/witwin/channel/physics/README.md) and
  [`scattering`](src/witwin/channel/scattering/README.md): reference
  conventions and rough-surface runtime.
- [`runtime`](src/witwin/channel/runtime/README.md): validated extension,
  symbols, tensor/AD contracts, buffers, and native handles.

Dependencies flow from solvers through typed owners to runtime; runtime never
imports scene or a solver. Production paths never silently fall back to
CPU/PyTorch, a Python ray tracer, a global extension, a zero result, or a
lower-fidelity algorithm.

## Native build

RayD (`backends/torch`) is built in the same CMake graph as `_channel`.

The default source location is `../../RayDi` relative to this repository. Set a
different checkout explicitly when configuring:

```powershell
cmake -S . -B build -DRAYD_SOURCE_DIR=E:/Code/RayDi
cmake --build build --config Release --target _channel
```

This integration builds and links `rayd_torch_native_core` directly. It does
not build/import RayD's Python module, use the Torch dispatcher, or load a
second DSO with `GetProcAddress`/`dlsym`. The former vendored `ext` snapshots
are no longer part of the repository; historical plans and audits may still
refer to them when describing the earlier architecture.

## Contract maintenance

Public export or signature changes update `ci/public-api-snapshot.json` and a
migration note. The completed ops migration ledger is immutable historical
evidence at `docs/dev/audit/phase12-ops-migration-ledger.json`; it is not an
active routing or compatibility mechanism. Cross-domain imports must satisfy
the import-graph contract; deleted legacy modules are hard failures.
