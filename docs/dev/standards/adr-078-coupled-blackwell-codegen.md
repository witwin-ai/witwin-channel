# ADR-078: Blackwell code for `coupled.cu` is built from compute_90

- **Status:** Accepted.
- **Date:** 2026-11-12.
- **Decision ID:** `coupled-blackwell-codegen`.
- **Scope:** the release (grouped gencode) build of `native/channel/kernels/coupled.cu`.
  Extends the per-file compile-flag rules of ADR-044.

## Context

The 0.6.0 publish workflow failed on every Linux and Windows wheel job: `cicc` was killed while
compiling `coupled.cu` on 16 GB runners. Measured with the release gencode on CUDA 12.8.93
(the CI toolkit is 12.8.1), each `cicc` and `ptxas` process of `coupled.cu` peaks at:

| process | compute_70 | compute_80 | compute_90 | compute_100 | compute_120 |
|---|---|---|---|---|---|
| `cicc` | 2.3 GB | 2.3 GB | 2.3 GB | 32.6 GB | 32.6 GB |
| `ptxas` (largest SASS target) | 3.0 GB | 3.0 GB | 2.9 GB | 2.5 GB | 2.7 GB |

CUDA 12.8 optimizes compute_100 and newer with a newer NVVM (the PTX reports "Based on NVVM
20.0.0"; compute_90 and older use NVVM 7.0.1). Its straight-line strength reduction needs memory
quadratic in the number of precise `sinf`/`cosf` calls inlined into one function: a kernel with
128 such pairs needs 1.7 GB for compute_120 and 0.17 GB for compute_90, and 256 pairs need
6.3 GB and 0.39 GB. `-opt-bisect-limit` places the growth at that pass.

In `coupled.cu` the trigonometry comes from RayD's wedge-pair evaluation instantiated on dual
numbers (`compute_pair_vector_contribution_faces<Dual>`, whose two `compute_op_terms_3d` calls
dominate). The double-diffraction dual row evaluates two pairs in one function and needs 32.6 GB
alone; the chain dual row evaluates one and needs 10.7 GB. Channel cannot split a pair: outlining
every pair evaluation into its own Channel function still needs 13.4 GB. Marking RayD's
`compute_op_terms_3d` not inlined (a RayD change, measured on a scratch copy) brings the whole
file to 1.9 GB.

## Decision

In the grouped-gencode (release) build, `coupled.cu` builds its SASS targets above compute_90
from compute_90 PTX and embeds compute_90 PTX in place of compute_120 PTX:

```
--generate-code=arch=compute_90,code=[sm_100,sm_120,compute_90]
```

`cmake/gencode.cmake` derives this group from the release gencode flags and refuses any gencode
option it cannot parse; every other translation unit keeps the release flags unchanged. The SASS
targets of the wheel are unchanged and the wheel still carries compute_120 PTX from the other
translation units, which is what `verify_cuda_binary_arches.py` checks. Developer builds that set
`CMAKE_CUDA_ARCHITECTURES` without grouped gencode flags are unaffected.

## Consequences

- The `cicc` and `ptxas` commands for the compute_70, compute_80 and compute_90 groups of
  `coupled.cu` are unchanged, so its sm_70 to sm_90 code is unchanged.
- On sm_100 and sm_120 the coupled kernels run code optimized by the same NVVM as on sm_90.
  Precise math, `--fmad` and every expression are unchanged; the generated code differs from what
  NVVM 20 would produce, as code for different architectures always may. No Blackwell device is
  available locally, so Blackwell results are not compared bit for bit.
- A GPU newer than sm_120 JIT-compiles the coupled kernels from compute_90 PTX.
- The sm_100 and sm_120 SASS of `coupled.cu` is about 9% larger; the extension grows by 0.9 MB,
  0.5 MiB compressed.
- Revisit when RayD outlines its operator-term evaluation or the CUDA toolkit fixes the pass;
  then `coupled.cu` returns to the shared release flags.

## Acceptance evidence

Release build of `_channel` with the release gencode, CUDA 12.8.93, RayD 0.10.3 and 24 parallel
jobs, every `cicc` and `ptxas` process sampled once a second:

| `coupled.cu` | compute_70 | compute_80 | compute_90 (also sm_100, sm_120) |
|---|---|---|---|
| `cicc` | 2.30 GB | 2.30 GB | 2.31 GB |
| `ptxas` | sm_70 2.95, sm_75 2.90 GB | sm_80 2.98, sm_86 2.99, sm_89 2.99 GB | sm_90 2.85, sm_100 2.86, sm_120 3.36 GB |

The heaviest process of the whole build is now `fields.cu`'s `cicc` for compute_100 and
compute_120 at 5.4 GB, then `diffraction.cu` and RayD's `multipath.cu` at 4.5 GB; every other
process stays under 1.8 GB. The build no longer has a process the 16 GB runners cannot hold.

Against a release build of the previous commit, the SASS of `coupled.cu` for sm_70, sm_75, sm_80,
sm_86, sm_89 and sm_90 and the SASS of every other Channel translation unit for all eight SASS
targets are identical function by function. On sm_86 the coupled primal, VJP and JVP outputs of
three coupled wall-wedge scenes at depths 2 and 3 (chain and double-diffraction rows; 72 tensors,
4,074 scalars) are bit-for-bit identical. The sm_100 and sm_120 SASS of `coupled.cu` differs, as
decided. All 35 CUDA objects in the extension carry every release SASS target; 34 carry compute_120
PTX and `coupled.cu` carries compute_90 PTX.
