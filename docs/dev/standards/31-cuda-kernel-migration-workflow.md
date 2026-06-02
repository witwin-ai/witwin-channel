# CUDA Kernel Migration Standard Workflow

Status: Active
Category: Standard
Last reviewed: 2026-04-03

This document defines the step-by-step process for migrating a DrJit Python computation to a native CUDA kernel with automatic differentiation support.

## Architecture Overview

The expected layering is:

```text
Python caller
  -> kernels/<module>/__init__.py
     -> drjit_impl.py      (pure DrJit reference path, AD implicit)
     -> native_impl.py     (dispatches to the C++ CustomOp)
        -> witwin/channel/<solver>/kernels/<module>/<name>.h
           -> eval()       (primal)
           -> forward()    (optional JVP)
           -> backward()   (reverse-mode VJP)
              -> CUDA kernels
```

Key principle: the C++ side exports one DrJit `CustomOp` class, not three separate function pointers. DrJit's AD system calls `eval()`, `forward()`, or `backward()` as needed. Python callers see a single function call; AD is transparent.

Boundary rule: Torch may exist at an explicit public API or result-adapter boundary, but the migration target itself must stay DrJit-native end to end. Do not insert NumPy staging, Torch transport, or DLPack handoff between DrJit arrays, nanobind bindings, and CUDA kernels.

## Step-by-Step Migration Process

### Step 1: Extract the kernel to `kernels/<module>/drjit_impl.py`

Identify the hot computation in the existing codebase and extract it into a clean function with a well-defined interface.

Rules:

- Input/output types: DrJit arrays (`bk.Float`, `bk.Point3f`, `bk.Complex2f`, etc.)
- No side effects and no scene object dependencies in the core computation
- The function signature is the contract that both DrJit and C++ backends implement
- AD in this path is handled by DrJit implicitly; no manual `forward()` or `backward()` is needed
- Do not redefine the runtime kernel contract in Torch or NumPy types

Example:

```python
# kernels/utd/drjit_impl.py
def utd_accumulate_forward(state_arrays, rx_pos, k, ...):
    """Pure-DrJit reference. AD is implicit via DrJit graph."""
    # ... extracted computation using DrJit ops ...
    return direct_vector, multi_vector
```

### Step 2: Write the C++ forward CUDA kernel

Create the fused mega-kernel under the relevant solver-owned `witwin/channel/<solver>/kernels/<module>/<name>.cu` directory.

Rules:

- Keep math in `__device__ __forceinline__` functions when practical
- Use SoA device pointers that match the DrJit array layout
- Write outputs directly to device buffers, including atomically accumulated outputs where needed
- Wrap kernel launch details in a host launcher function
- Do not add Torch tensor launches, DLPack handoff, or NumPy staging as part of the kernel path

### Step 3: Decide on differentiability

| Question | If YES | If NO |
|----------|--------|-------|
| Does this kernel participate in optimization? | Implement as `drjit::CustomOp` with backward and optionally forward | Export as a plain function |
| Example differentiable kernels | UTD accumulate, reflection accumulate | Not applicable |
| Example non-differentiable kernels | Not applicable | Coplanarity check, edge geometry, pruning sort, cartesian filter, packed-state gather/concat |

If the kernel is non-differentiable, skip to Step 6 and export the host launcher directly via nanobind.

If the kernel is differentiable, continue to Step 4.

### Step 4: Write the backward (VJP) kernel

The backward kernel recomputes forward intermediates and propagates adjoints in reverse.

Key insight: the backward kernel usually shares most of the forward kernel structure.

```cpp
__global__ void my_kernel_backward(...) {
    // 1. Reload inputs.
    // 2. Recompute forward intermediates.
    // 3. Load upstream gradients from output grad buffers.
    // 4. Apply the reverse chain rule.
    // 5. Accumulate gradients for the inputs.
}
```

The JVP (forward-mode) kernel is optional. If only reverse-mode AD is needed, you may omit `forward()`. DrJit will raise if someone calls `dr.forward()` without a `forward()` implementation, but `dr.backward()` will still work.

If you need JVP, implement it alongside backward using the dual-number pattern so tangent vectors propagate with the primal computation.

### Step 5: Register as a DrJit C++ CustomOp

This is the architectural step that makes AD integration automatic.

In C++ (`witwin/channel/<solver>/kernels/<module>/<name>.h`):

```cpp
#include <drjit/custom.h>
#include <drjit/autodiff.h>

using Float = drjit::CUDAArray<float>;
using DiffFloat = drjit::DiffArray<drjit::JitBackend::CUDA, float>;

class UTDAccumulateOp : public drjit::CustomOp<
    /* Output */ drjit::tuple<DiffFloat, DiffFloat, ...>,
    /* Inputs */ DiffFloat, DiffFloat, ...
> {
public:
    using Base = drjit::CustomOp<...>;
    using Output = typename Base::Output;

    Output eval(
        drjit::detached_t<DiffFloat> edge_pos_x,
        drjit::detached_t<DiffFloat> edge_pos_y,
        ...
    ) override {
        const float* d_edge_pos_x = edge_pos_x.data();
        utd_accumulate_forward_kernel<<<grid, block>>>(...);
        return { out_direct, out_multi, ... };
    }

    void backward() override {
        auto [grad_direct, grad_multi, ...] = this->grad_out();
        utd_accumulate_backward_kernel<<<grid, block>>>(...);
        this->set_grad_in("edge_pos_x", grad_edge_pos_x);
    }

    void forward() override {
        auto t_edge_pos_x = this->grad_in("edge_pos_x");
        utd_accumulate_jvp_kernel<<<grid, block>>>(...);
        this->set_grad_out({ t_direct, t_multi, ... });
    }

    const char* name() const override { return "UTDAccumulate"; }
};
```

In the nanobind module (`module.cpp`):

```cpp
m.def("utd_accumulate", [](DiffFloat edge_pos_x, ...) {
    return drjit::custom<UTDAccumulateOp>(edge_pos_x, ...);
}, "UTD diffraction accumulation with automatic differentiation.");
```

From Python:

```python
result = native.utd_accumulate(edge_pos_x, edge_pos_y, ...)

dr.backward(result)
grad_edge_pos_x = dr.grad(edge_pos_x)

dr.forward(edge_pos_x)
tangent_result = dr.grad(result)
```

### Step 6: Create `kernels/<module>/native_impl.py`

This is the Python dispatch layer that calls the C++ CustomOp.

```python
# kernels/utd/native_impl.py
from witwin.channel._native.<extension> import NativeExtension

def utd_accumulate_forward(state_arrays, rx_pos, k, ...):
    """Native CUDA path. AD is handled by the C++ CustomOp."""
    native = NativeExtension.require_functions(("utd_accumulate",), context="UTD backend")
    edge_pos_x = state_arrays["edge_pos"].x
    result = native.utd_accumulate(edge_pos_x, ...)
    return direct_vector, multi_vector
```

This dispatch layer may unpack DrJit structures into the kernel signature, but it must not route the runtime path through Torch, NumPy, or DLPack as an intermediate representation.

### Step 7: Wire up backend selection in `__init__.py`

```python
# kernels/utd/__init__.py
from witwin.channel._native.<extension> import NativeExtension

if NativeExtension.native_extension_available():
    from witwin.channel.kernels.utd.native_impl import utd_accumulate_forward
else:
    from witwin.channel.kernels.utd.drjit_impl import utd_accumulate_forward
```

## Summary: What Goes Where

| Layer | File | AD handling |
|-------|------|-------------|
| Caller | `trace/diffraction/field.py` | Calls `kernels.utd.utd_accumulate_forward(...)` |
| Dispatch | `kernels/utd/__init__.py` | Picks DrJit or native backend |
| DrJit reference | `kernels/utd/drjit_impl.py` | AD is implicit in the DrJit graph |
| Native dispatch | `kernels/utd/native_impl.py` | Calls the C++ CustomOp |
| C++ CustomOp | `witwin/channel/<solver>/kernels/utd/utd_accumulate.h` | `eval()` plus `backward()` and optional `forward()` |
| CUDA kernels | `witwin/channel/<solver>/kernels/utd/utd_accumulate.cu` | Forward, backward, and optional JVP kernels |

## Differentiability Decision Matrix

| Kernel | Differentiable? | Why | AD strategy |
|--------|-----------------|-----|-------------|
| UTD accumulate | Yes | Geometry optimization (`edge_pos`, `source_pos`, `rx_pos`) | C++ CustomOp with backward and JVP when needed |
| Reflection accumulate | Yes | Material and geometry optimization | C++ CustomOp with backward and JVP when needed |
| Packed-state gather/concat | No | Data movement only | Plain function export |
| Cartesian filter | No | Discrete selection (`compress` plus `compact`) | Plain function export |
| Pruning sort | No | Discrete ranking (`sort` plus `top-k`) | Plain function export |
| Coplanarity check | No | Scene compilation | Plain function export |
| Edge geometry | No | Scene compilation | Plain function export |

## FAQ

Q: Can DrJit auto-derive backward from forward, or vice versa?
A: No. `forward()` and `backward()` must be implemented independently on the C++ CustomOp. Most optimization workloads only need `backward()`. Implement `forward()` only when forward-mode AD is actually required.

Q: Why can DrJit not auto-differentiate a CUDA kernel?
A: DrJit's AD system traces element-wise operations on `DiffArray`. A fused CUDA kernel is opaque to that tracer. The CustomOp is how you define derivatives for that black box.

Q: Can I use Enzyme for CUDA kernel differentiation?
A: In principle yes, but it requires a custom LLVM toolchain and adds significant build complexity. The CustomOp approach remains the practical project standard.

Q: Does the DrJit Python path need a CustomOp?
A: No. The DrJit reference path uses native DrJit operations, so AD stays implicit in the DrJit graph.

Q: How do I validate the C++ backward kernel?
A: Compare against finite differences by perturbing each input by `eps`, running forward twice, and checking `(f(x + eps) - f(x - eps)) / (2 * eps)`. Also compare against the DrJit reference path with `dr.backward()` where possible.
