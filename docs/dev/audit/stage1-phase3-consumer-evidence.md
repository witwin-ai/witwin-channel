# Stage-I Phase 3 consumer evidence

Status: Stage-I local implementation, concentrated adversarial audit, native
rebuild, full regression, performance acceptance, and a clean Windows RC wheel
smoke passed. The real manylinux_2_28 build remains a GitHub Actions release
gate and is not claimed by this local record.

## Identity

- Channel implementation commit:
  `88f8a35a102080b167211f6a6ec66d2724b4f92c`
- Channel isolated-wheel audit commit:
  `6a60e2f`
- Core commit: `42b7b067b4512ebe05c462b79a75577458010b48`
- RayD lock: `49c58c4cb8212f6babb920cc88fb937509826cc5` (`0.7.0`)
- consumer contract: `witwin.channel.propagation.consumer`, version 1
- local build fingerprint:
  `f04b5629d5fba76fb283218ce14a53b3f75bbe4685d7a9f72ce5e2afbbb6a645`
- Windows RC build fingerprint:
  `b955753f860fd50bbc7ae2ff7635f96c1b0e523b1d7814dc6f3d217e660139a7`
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
| clean Windows release-wheel install/native load | pass; Channel and Core wheels isolated together |

The full suite used the rebuilt native extension and completed in 445.62
seconds. The focused remediation and CUDA/AD set completed with 155 passed.
The final wheel-audit regression set completed with 58 passed.

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

The locally accepted Windows RC is:

- artifact:
  `witwin_channel-0.4.0-cp311-cp311-win_amd64.whl`
- size: 15,644,580 bytes
- SHA-256:
  `295adc07b82bae8472128cd8d378908fd2db32015b83a6be911f0aa698c965a5`
- paired Core wheel SHA-256:
  `24677c4902ca44e36bcef6933398d2e9afd3ec74fa9a246fbbacdf54e8ba1f62`
- smoke evidence SHA-256:
  `dfc50d5b5cb80d729002cf94da300a0af570fbc9d06abe0ed6a7d07f73aa6e32`
- build identity: clean Channel `88f8a35`, clean RayD `0.7.0`, Release,
  CPython 3.11, Torch 2.10, SM120 SASS plus compute_120 PTX

The strict smoke installs the locked Core wheel and Channel wheel into the
same disposable target, verifies both imports resolve inside that target,
audits the standard `dist-info/licenses/LICENSE` bytes against the repository
license, validates the PE dependency/export closure, and loads all nine
Phase-3 native symbols.
