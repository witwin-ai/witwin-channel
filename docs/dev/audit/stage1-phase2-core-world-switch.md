# Stage-I Phase 2 Channel Core-world switch

Status: accepted, 2026-07-24.

## Contract

- Core baseline: `witwin.core` commit `0d6c6b5`, package version `0.4.0`.
- Channel dependency: exact `witwin==0.4.0`.
- Input: `witwin.core.Scene` or `SceneSnapshot`.
- Compile boundary:
  `witwin.channel.scene.compile(..., reference_frequency_hz=...)`.
- Solver boundary: Path, Deterministic, Monte Carlo Basic, and Monte Carlo
  BDPT all require the same keyword-only reference frequency.
- Runtime trace policy: RayD `TraceBackend::Auto` prefers OptiX and may select
  RayD's full-result pure-CUDA implementation. No CPU/Torch fallback is added.

## Ownership and invalidation

The old Channel logical Scene, Structure, materials, endpoints, loaders, pickle
rewrites, and `witwin.channel.core.runtime` facades are deleted. Root logical
exports resolve directly to Core. Channel retains only compiled resources,
stores, cache policy, material ABI, solver orchestration, and propagation.

The bounded 32-entry Channel registry keys source identity, all four Core
version domains, exact frequency identity/revision, CUDA device, and material
ABI. Topology/geometry reuse the RayD scene and GeometryStore; material and
frequency changes rebuild MaterialStore; assignment changes rebuild
AssignmentStore. Stable Core IDs are retained in signed `int64` maps and native
runtime rows use dense `int32` indices.

The compiler uses Structure assignment `MaterialId` values as the stable
MaterialStore identity, including when they differ from the PhysicalMaterial
object's own ID. ABI-v3 scalar Fresnel fields are the layer-0 view of the CSR
stack. Phase-screen face classification maps structure indices back to stable
Core `StructureId` values before suppressing realization-replaced delta rows.

## Concentrated adversarial audit

The large-module audit found and closed:

- cross-world stale registry reuse;
- Core antenna-pattern and endpoint tensor incompatibilities;
- mixed CPU/CUDA authored geometry at the compiler boundary;
- layer-0 scalar material ABI loss for multilayer materials;
- Structure assignment ID versus material-object ID confusion;
- phase-screen structure-index versus stable-StructureId confusion;
- BDPT enumerated source-power AD entering a fixed-power field kernel;
- package/submodule name collision that could replace the public
  `witwin.channel.scene.compile` function after an import;
- stale predecessor APIs, subprocess editable finders, wheel allowlists, and
  performance fixture identities.

The internal compiler implementation is consequently named
`witwin.channel.scene.compiler`; the stable public entry remains the callable
`witwin.channel.scene.compile`.

## Validation

All Python, tests, and build probes used the `witwin2` environment.

```text
Final full suite:
  2486 tests
  0 failures
  0 errors
  11 skipped/xfail
  443.106 s

Focused gates:
  BDPT AD                                  22 passed
  solver geometry AD                      22 passed, 1 xfailed
  RayD geometry AD                         18 passed
  RayD 0.7.0 locked boundary               29 passed
  material ABI + Phase-D material matrix   25 passed
  phase-screen adversarial set             30 passed
  contract/API cleanup                     72 passed
  solver/topology cleanup                  95 passed, 1 skipped
  performance/release/subprocess cleanup  211 passed
```

Static acceptance:

```text
ruff check src tests tools benchmarks
ci/check_import_graph.py
ci/check_contract_coverage.py
git diff --check
```

The import graph passes with the existing two boundaries and the single named
BDPT enumerated dependency. Contract coverage passes with 33 public exports and
241 native bindings.

The native build probe used `witwin2` Python, CMake, Ninja, Torch, and CUDA 12.9
against the clean RayD 0.7.0 worktree. Its developer wheel was
`witwin_channel-0.4.0-cp311-cp311-win_amd64.whl`, SHA-256
`9803E60241FC2E3A1B698F85C05711C9B28E36A2DDAC35099BF6AC717077D000`.
The final Python source was exercised against that validated packaged extension;
the clean Stage-I release wheel remains the Phase-3 release gate.
