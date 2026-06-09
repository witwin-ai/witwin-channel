# RayDN API Reference

Use RayDN as:

```python
import raydn as rt
```

## Core Types

- `rt.Mesh(vertices, faces, ...)`: CUDA mesh input. `vertices` must be contiguous `torch.float32` with shape `(N, 3)`, and `faces` must be contiguous `torch.int32` with shape `(M, 3)`.
- `rt.Scene()`: Native CUDA/OptiX scene. Build with `add_mesh()` and `build()`.
- `rt.Ray(o, d, tmax=None)`: Ray batch with CUDA `torch.float32` origin and direction tensors of shape `(N, 3)`.
- `rt.Camera(width, height, fov_x)`: Torch Python camera helper for primary ray generation.

## Geometry

- `Scene.intersect(ray, active=None)` returns `Intersection`.
- `Scene.nearest_edge(point)` returns `NearestPointEdge`.
- `Scene.nearest_edge(ray)` returns `NearestRayEdge`.
- `Scene.visible(start, end, active=None)` returns a `torch.bool` visibility tensor.

## Multipath

- `Scene.trace_reflections(ray, max_bounces, active=None)` returns `ReflectionChain`; the forward path uses a RayD-source-ported single OptiX launch with the bounce loop inside raygen.
- `Scene.trace_refl_epc_field(source, receiver, max_bounces, active=None)` returns `ReflEpcField`; the forward path uses RayD-source-ported reflection EPC plus EPC field kernels with simplified default material/options at the Python API boundary.
- `Scene.trace_dfr_paths(tx_positions=..., rx_positions=..., states=..., material=..., active=..., max_paths=..., wavelength=...)` returns `DfrPaths`.
- `Scene.accum_dfr_direct(states=..., grid=..., material=..., active=..., wavelength=..., direct_samples=..., keller_samples=..., suffix_samples=..., seed=...)` returns `DfrAccum`.
- `Scene.accum_dfr(initial_states=..., recursive_states=..., grid=..., material=..., active=..., recursive_active=..., wavelength=..., direct_samples=..., keller_samples=..., suffix_samples=..., seed=..., max_order=...)` returns `DfrAccum` for order-2/order-3 chain accumulation.
- `Scene.accum_dfr_coherent_direct(states=..., grid=..., material=..., active=..., wavelength=..., select_diffraction_point=..., prefilter_visibility=...)` returns `DfrCoherentAccum`.

These APIs use native CUDA/OptiX source ports for the reflection and diffraction
multipath execution paths. Current completion risk is performance parity, not
placeholder kernel coverage.

## Tensor ABI

All native geometry and multipath inputs are CUDA tensors. Continuous vector inputs use contiguous `torch.float32`; topology and primitive ids use `torch.int32`; masks and validity outputs use `torch.bool`. Native tapes are Torch-owned tensors and remain on the same CUDA device as the inputs.

## AD Contract

Native operators support VJP and JVP for continuous Torch inputs. Discrete choices such as primitive id, edge id, visibility, and fixed path sequence are non-differentiable and are held fixed from the forward pass. Continuous outputs are recomputed from the fixed winner and live geometry tensors during AD.

RayDN does not import or depend on Dr.Jit in the `raydn` package path.

See `docs/raydn_native_gap_analysis.md` for current acceptance status.
