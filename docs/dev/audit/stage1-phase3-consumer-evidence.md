# Stage-I Phase 3 consumer evidence

Status: local implementation, adversarial audit, native rebuild, full regression,
and performance acceptance passed. Publication-grade Windows and
manylinux_2_28 wheels remain a post-commit release workflow artifact and are
not claimed by this pre-commit record.

## Identity

- Channel commit: the Phase-3 implementation commit containing this record
- Core commit: `42b7b067b4512ebe05c462b79a75577458010b48`
- RayD lock: `49c58c4cb8212f6babb920cc88fb937509826cc5` (`0.7.0`)
- consumer contract: `witwin.channel.propagation.consumer`, version 1
- local build fingerprint:
  `f04b5629d5fba76fb283218ce14a53b3f75bbe4685d7a9f72ce5e2afbbb6a645`
- local environment: `witwin2`, CPython 3.11.14, Torch 2.10.0, CUDA
  runtime 12.8, RTX 5080 (SM120)
- build tools: `witwin2` CMake 4.3 and Ninja 1.13; MSVC 19.44 and the
  system CUDA 12.9.41 `nvcc` driver were used because the conda CUDA package
  does not contain `nvcc.exe`

## Concentrated gates

| Gate | Recorded result |
| --- | --- |
| package-neutral import/public snapshot | pass; 50 frozen exports |
| import graph/caller-free ADR-029/030/031 | pass |
| native owner/coverage manifest | pass; 250 bindings, 9 Phase-3 additions |
| scalar/Complex3/Jones E2E | pass |
| exact compact K/pair segmentation/order | pass |
| zero-copy object/storage/stride/device/grad | pass |
| fixed-row primal/JVP/VJP | pass |
| unsupported capability preflight/no partial result | pass |
| four Channel solver regression | pass |
| full Channel suite | 2515 passed, 10 skipped, 1 xfailed, 0 failed |
| adversarial audit | no P0/P1 after remediation |
| Ruff/import graph/contract coverage/diff check | pass |
| Windows hosted release workflow policy | pass by repository governance tests |
| real manylinux_2_28 wheel | pending GitHub Actions |
| full SASS including SM87 and compute_120 PTX | pending GitHub Actions |
| clean release-wheel install/native load | pending post-commit RC build |

The full suite used the rebuilt native extension and completed in 445.62
seconds. The focused remediation and CUDA/AD set completed with 155 passed.

## Compact and synchronization evidence

- Phase-2 comparison artifact:
  `docs/dev/audit/stage1-phase2-core-world-switch.md`
- general canonical owner: one 8-byte count D2H and one current-stream
  synchronization for non-empty candidates; zero for empty candidates
- exact LoS metadata owner: zero D2H and zero synchronization
- consumer assembly adds no count observation, D2H, synchronization, or second
  compact operation beyond the inherited canonical owner
- fixed LoS reevaluation: one 4-byte validation D2H and one current-stream
  synchronization for non-empty rows; zero discovery launches
- pair metadata, exact rows, pair offsets, and optional stable endpoint IDs are
  produced by the owning native boundary in sink-major/source-minor order
- production general compact complexity is `O(K + P)` after stable radix sort;
  no Torch sort/nonzero/boolean gather is present on the production path

The measured 32-source by 32-sink workload is frozen in
`stage1-phase3-consumer-performance.json`:

| Workload | Median | P95 | Throughput | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| general discover, 1024 paths | 3.533 ms | 3.837 ms | 289,810 paths/s | 1,147,392 B |
| fixed LoS reevaluate, 1024 paths | 0.748 ms | 0.868 ms | 1,369,541 paths/s | 476,672 B |

Each measurement used ten warmups and fifty CUDA-event-timed repeats. Event
synchronization belongs to the benchmark harness and is not counted as a
production synchronization.

## Trace backend evidence

RayD `TraceBackend::Auto` is accepted as resolving to OptiX when available and
to the RayD-owned pure-CUDA backend otherwise. Channel does not add a fallback
implementation. The local build reports `optix_available=true`, so the local
runtime result is not presented as independent pure-CUDA performance evidence.
The RayD 0.7.0 lock and source-linked ABI are validated; backend-specific RayD
performance remains RayD-owned release evidence.

## Release truth

Channel is CPython 3.11 / Torch 2.10.0 only. `_channel` is a versioned
LibTorch/Python extension, not a LibTorch Stable ABI binary. Linux release
evidence is valid only when the binary is compiled inside `manylinux_2_28`;
an Ubuntu build with a relabeled wheel is not accepted. The authoritative
workflow builds native SASS for SM70/75/80/86/87/89/90/100/101/120 and PTX for
compute_120, pins the Stage-I Core commit and RayD 0.7.0, and performs isolated
wheel import/native-symbol smoke tests on both Windows and Linux.
