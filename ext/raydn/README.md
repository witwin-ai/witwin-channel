# RayDN

RayDN is a Torch-native CUDA/OptiX package for RayD geometry primitives and
RayD-style multipath/diffraction kernels.

```python
import raydn as rt
```

The public package name is `raydn`; it does not provide `rayd` compatibility aliases, so it can coexist with the original RayD package in the same environment.

## Tensor ABI

RayDN APIs accept CUDA `torch.float32` tensors for vector data and CUDA `torch.int32` tensors for index data. Vector tensors are row-major `(N, 3)` unless otherwise documented, masks are `torch.bool`, and tensors should be contiguous. Outputs and AD tapes are Torch-owned tensors.

## Gradient Contract

Intersection, edge, reflection, EPC, and diffraction operators use a fixed-winner gradient contract where explicit native kernels exist. The discrete primitive, edge, visibility, or path decision selected in the forward pass is treated as non-differentiable; VJP and JVP propagate through the continuous values recomputed from the saved winner and live Torch tensors.

## Autograd

The native operators support Torch reverse-mode VJP and forward-mode JVP for the supported continuous inputs where explicit kernels have been implemented. CUDA work is launched on the current Torch CUDA stream.

## Current Status

RayDN now builds separate native scene, edge, reflection, and diffraction
Torch extension bindings. The native build includes OptiX PTX pipelines for
scene intersection, edge queries, reflection tracing/EPC/visibility/
accumulation, and diffraction path/accumulation/coherent direct execution.

Current opt-in RayD parity tests cover forward cases for scene intersection,
multi-mesh global ids, nearest-edge, visibility, reflection tracing,
diffraction paths, direct/Keller/suffix diffraction accumulation, order-2 and
order-3 diffraction chains, and coherent direct accumulation. Torch VJP/JVP
coverage exists for geometry, edge, reflection trace, EPC, and diffraction
accumulation under the fixed-winner contract.

The remaining completion risk is performance acceptance: the same-script
RayD/RayDN benchmark now exists, but current measurements still show
RayDN slower for build, reflection trace, and diffraction direct on the
recorded benchmark shape. See `docs/raydn_native_gap_analysis.md` and
`docs/raydn_native_performance.md`.

## Dependencies

RayDN depends on PyTorch, CUDA, and OptiX for native execution. The RayDN package path has no Dr.Jit dependency.
