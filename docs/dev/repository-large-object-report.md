# Repository Large-Object Report

This is the read-only Phase 1 inventory captured on 2026-07-14
(America/Los_Angeles), before removing `build-witwin3/` from the Git index.

## Snapshot

- HEAD: `51dcfcfde8de7f646d6ecfc9441ddf7c7d0b9003`
- Tracked paths under `build-witwin3/`: 316
- Tracked index blob bytes under `build-witwin3/`: 217,609,196 bytes
  (207.53 MiB)
- Checked-out bytes for those paths: 218,013,112 bytes (207.91 MiB). The
  checkout is larger than the index blobs because Git's Windows text checkout
  can expand line endings.
- Reachable objects reported by `git rev-list --objects --all`: 2,842, of
  which 1,579 are blobs.
- One pack index was verified successfully; `git verify-pack -v` reported
  3,193 packed object rows.
- `git count-objects -vH` reported a 58.61 MiB pack, 22.55 MiB of loose
  objects, and 20.83 MiB of garbage. The three garbage entries were temporary
  loose-object files. This report did not remove them.

## Largest reachable blobs

The sizes below are reconstructed blob sizes from `git cat-file`, not packed
delta sizes. Every listed path is under the tracked build tree.

| Bytes | MiB | Object | Last path reported by `rev-list` |
|---:|---:|---|---|
| 72,585,730 | 69.22 | `f94cbffefccceb564c054f8aa56b25c28c4def84` | `build-witwin3/ext/raydn/raydn_native_core.dir/Release/raydn_native_core.lib` |
| 16,044,153 | 15.30 | `209d87c436e558d5474d1f0b91a556d735c98940` | `build-witwin3/ext/raydn/generated/raydn/diffraction_accumulation_optix_ptx.h` |
| 13,602,559 | 12.97 | `c690c08c67f8ff89aeaf0fd1b565907f0cf90e65` | `build-witwin3/ext/raydn/_raydn.dir/Release/library.obj` |
| 13,090,332 | 12.48 | `d3053c9cf1d2266d1d9d15db506faf3c40e33fb2` | `build-witwin3/ext/raydn/raydn_native_core.dir/Release/raydn_na.BBBC4913.tlog/CL.read.1.tlog` |
| 8,030,092 | 7.66 | `305be1448e9d1ab4839c07d436774c8707622150` | `build-witwin3/ext/raydn/raydn_native_core.dir/Release/accum_ad.obj` |
| 6,367,447 | 6.07 | `faed922aec53ff4cf11cc336af08cf00bfb659e4` | `build-witwin3/ext/raydn/raydn_native_core.dir/Release/ops_intersect.obj` |
| 4,794,825 | 4.57 | `4f7a8fc8afb66604026d38f1018275909f898af5` | `build-witwin3/ext/raydn/raydn_native_core.dir/Release/src/torch_ext/diffraction/accum_reduce.cu.obj` |
| 3,998,526 | 3.81 | `48c9bf2f4530947571b90f6a4a4d36ea344db338` | `build-witwin3/CMakeFiles/CMakeConfigureLog.yaml` |
| 3,595,249 | 3.43 | `4b08bc5e2143abc66e8f5e121d5a4d111b710424` | `build-witwin3/ext/raydn/raydn_native_core.dir/Release/src/torch_ext/diffraction/pipeline.cpp.obj` |
| 3,288,096 | 3.14 | `cd15db187db73f8f25a66eff9e3e0e9a3e3f4377` | `build-witwin3/ext/raydn/raydn_native_core.dir/Release/src/torch_ext/diffraction/ops.cpp.obj` |

`git verify-pack -v` also found a 74,494,870-byte blob
(`5be9e63d8594b821b68c801d2f9ebf4d60b9c39b`) in the current pack. It is not
reachable from any ref enumerated by `git rev-list --objects --all`, so no
historical path is assigned to it here.

## Read-only commands used

```powershell
git rev-parse HEAD
git ls-files -s -- build-witwin3
git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize)'
git rev-list --objects --all
git verify-pack -v -- .git/objects/pack/pack-081acdf72c0a215b098e54d8e599292ad1d098db.idx
git count-objects -vH
```

## Decision

Phase 1 removes `build-witwin3/` only from the current index while preserving
developer files in the working directory. It does not rewrite history, prune
objects, run garbage collection, or force-push. A later history rewrite remains
an optional operations project and requires the coordination, mirror backup,
fresh-clone rehearsal, ref inventory, and recovery plan specified by the
architecture hardening plan.
