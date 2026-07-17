# Phase 11 CI, coverage, documentation, and maintenance acceptance

Phase 11 was accepted on 2026-07-16 at implementation commit `4a8bc50`.
All Python, build, and runtime commands used the `witwin2` environment.

## Delivered gates

- Ruff debt in production, tests, benchmarks, and CI is zero.
- Mypy 1.18.2 checks ten critical runtime/config/deployment modules.
- The import graph, frozen ops migration contract, production dependency gate,
  repository hygiene, and import-without-native contract run in the quick tier.
- `ci/contract-coverage-manifest.json` maps all 37 curated public exports and all
  174 native bindings to a contract test and at least one E2E scenario. Native
  rows use 20 shape/device/dtype or negative-contract families and a unique
  Python owner.
- Python files have a 2,000-line hard limit and a 1,200-line recommended limit.
  Existing file and complexity debt has an owner, reason, expiry date, and
  non-growing ceiling.
- Quick, CUDA PR, nightly, and scheduled/per-release workflows use the only
  verified Windows/X64/Python 3.11/Torch CUDA 12.8/SM120 support row. CUDA tiers
  build against locked RayD `6047089` and load only the fingerprinted result.
- Ten top-level domain READMEs define ownership, public entries, dependencies,
  numerical/AD contracts, forbidden fallback, and manifest maintenance.

## Final evidence

- Quick tier: passed, including 31 manifest/audit tests and 55 CPU import/config
  tests.
- Native configure/build: Release, SM120-real, CUDA toolkit 12.9.41, Torch
  2.10.0 CUDA 12.8, clean Channel Native `4a8bc50`, clean locked RayD
  `6047089`; build fingerprint
  `fdca81d659443d8c8a492908cb145e07184dd5789ceea185b3a74924e4c6d36d`.
- CTest: 1/1 passed (`legacy_slab_lockstep`).
- Full branch-instrumented pytest: 1,751 passed, 1 skipped, 1 xfailed, 21
  warnings in 575.46 seconds.
- Coverage: statement 85.7341%, branch 66.0985%, core functional/pipeline
  aggregate 90.0558%. Phase 0 was statement 82.9597% and branch 64.4507%.
  The two sub-90 core files have explicit per-file floors expiring 2027-01-31.
- Reduced runtime matrix: result/metadata/scene fingerprints 8/8 exact; launch
  ledgers 8/8 exact. Worst current/baseline ratios were 0.3140x median wall,
  0.2726x p95 wall, and 1.0000x peak allocated memory.
- Repository hygiene: 593 tracked files; no forbidden artifact or local
  absolute runner path was introduced.

The first coverage attempt correctly rejected four README heading-contract
violations and two frozen-owner changes. Both were corrected without changing
the Phase 0 import or ops migration digests; the evidence above is the final
zero-failure rerun.
