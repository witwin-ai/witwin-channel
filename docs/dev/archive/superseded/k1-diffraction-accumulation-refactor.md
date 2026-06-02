# K1 Fused Diffraction Accumulation Kernel: Problems & Refactoring Proposal

## 1. Background

K1 is the fused GPU kernel that replaces the inner loop of
`_accumulate_edge_states_to_receivers` in the diffraction field module. It
compresses **state gather -> UTD coefficient evaluation -> atomic accumulation to
receivers** into a single Slang kernel launch, eliminating ~50 intermediate DrJit
arrays per chunk.

### Current architecture

```
                     DrJit AD graph
                          |
                    dr.custom(CustomOp)
                          |
            _EdgeStateTotalsOp.eval()
                    /           \
           safe chunks        unsafe chunks
               |                   |
     Slang kernel            DrJit replay
    (utdAccumulate*)      (detach + forward_to
                           / backward_to)
               \                /
            _add_receiver_totals
                    |
              DrJit AD graph
```

### Current performance (fixed benchmark)

| Configuration | Forward | Backward | Peak Memory |
|---------------|---------|----------|-------------|
| K1=off (pure DrJit) | 0.429s | 0.867s | 2.495 GiB |
| K1 streaming scalar backward | 0.354s | 0.281s | 3.719 GiB |

K1 delivers **68% backward speedup** but increases peak memory by 1.2 GiB.

---

## 2. Current Problems

### P1. SoA tensor fragmentation at bridge boundary

The DrJit <-> Slang bridge decomposes every state field into separate scalar
tensors because slangtorch does not support passing structured buffers
(`StructuredBuffer<T>`) as kernel arguments. One `PairInputs` struct (10 logical
fields) becomes **32 separate `TensorView<float>` parameters**. With
dispatch/output tensors, a single forward kernel launch passes **~53 tensor
arguments**. For forward JVP, the count doubles to **~100+ tensors** (state +
tangent + output).

Consequences:
- Each tensor view calls `.contiguous()`, which may trigger a layout copy when
  the DrJit internal buffer is not C-contiguous.
- Kernel launch overhead scales linearly with argument count (driver-side
  parameter packing).
- The Python bridge (`_prepare_state_tensor_views`) is 60 lines of mechanical
  unpacking/repacking that must be kept in sync with the Slang struct definition.

Root cause: **slangtorch limitation** -- it only accepts flat `TensorView<T>` as
kernel parameters; there is no path to pass a GPU buffer interpreted as a
struct-of-arrays or array-of-structs.

### P2. Repeated safe/unsafe partition computation

`_partition_visible_pair_dispatch_chunks()` performs ray-cast visibility,
wedge-exterior validation, cotangent pole safety, and slope derivative safety
checks. This is expensive: it gathers state arrays, computes edge angles, and
evaluates multiple boolean masks per chunk.

The problem: **`eval()`, `forward()`, and `backward()` each call this function
independently**. The CustomOp stores `self.state_inputs` and `self.rx_pos` but
**does not cache the partition results**. The same geometry validation runs 2-3x
per AD pass.

```python
# In _EdgeStateTotalsOp:
def eval(self, ...):
    # _partition called inside _accumulate_..._hybrid()
    return _accumulate_edge_states_to_receivers_totals_hybrid(...)

def forward(self):
    # _partition called AGAIN via _strict_dispatch_chunks()
    _safe, _unsafe, dispatch = _strict_dispatch_chunks(...)

def backward(self):
    # _partition called AGAIN via _strict_dispatch_chunks()
    _safe, _unsafe, dispatch = _strict_dispatch_chunks(...)
```

Root cause: DrJit's `CustomOp` lifecycle does not guarantee that intermediate
Python objects survive between `eval()` and `forward()`/`backward()`. Caching
partition results on `self` may work in practice but has not been validated for
correctness under DrJit's enqueue/replay model.

### P3. Dual execution path (safe/unsafe split)

The safe/unsafe partition creates two fundamentally different AD paths:

- **Safe chunks**: Slang kernel with hand-written VJP/JVP.
- **Unsafe chunks**: Full DrJit replay -- `dr.detach` -> `dr.enable_grad` ->
  replay forward -> `dr.backward_to`.

This means:
1. The backward pass may run **two completely different gradient implementations**
   for the same mathematical operation.
2. Unsafe chunk replay rematerializes the entire DrJit accumulation loop,
   including all intermediate arrays that K1 was designed to eliminate.
3. If a scene has many pole-adjacent or slope-unsafe pairs, the unsafe path can
   dominate runtime and negate K1's memory savings.

The unsafe fraction is data-dependent and not bounded. In pathological wedge
configurations, a significant proportion of state-rx pairs can land in the
unsafe bucket.

Root cause: The Slang kernel's hand-written VJP does not handle cotangent pole
singularities or slope derivative edge cases. These require either:
(a) numerically stable reformulations inside the kernel, or
(b) kernel-side clamping with documented accuracy trade-offs.

### P4. First-launch CUDA forward instability

With K1 enabled (`accumulate_primal="custom_op_partitioned"`), the first forward
trace in a process produces different results from the second trace under
identical inputs. This manifests as horizontal/banded field artifacts.

Documented in `docs/dev/bugs/known-bugs.md`. The issue:
- Is specific to K1; disabling K1 restores repeatability.
- Is NOT caused by suffix DDA or reflection DDA toggles.
- Is NOT caused by the DrJit replay fallback (unsafe chunks).
- Persists even with a small warm-up trace workaround.

K1 remains **disabled by default** (`accumulate_primal="drjit"`) because of this
bug. This means the entire K1 optimization is opt-in and not benefiting default
users.

Root cause: Suspected to be CUDA kernel compilation/caching or slangtorch module
initialization state that differs between the first and subsequent launches. The
Slang module's JIT output or thread-block scheduling may not be deterministic on
first invocation. Not yet root-caused to a specific line.

### P5. Peak memory regression

Despite eliminating intermediate DrJit arrays, K1 **increases** peak memory from
2.495 GiB to 3.719 GiB (+49%). This is because:

1. The bridge allocates 16 output tensors + 32 state tensor views + gradient
   tensors per dispatch chunk per kernel launch.
2. The `_EdgeStateTotalsOp` captures the **entire state dictionary** on `self`
   for use in `forward()`/`backward()`, keeping it alive across the AD pass.
3. The backward path's scalar and vector kernels each allocate their own
   gradient output tensors, effectively doubling gradient memory when both are
   needed.

The K1 prototype validated that `dr.custom` reduces DrJit's intermediate array
pressure, but the bridge's own tensor allocations partially offset that gain.

### P6. `per_edge` output incompatibility

When `return_per_edge=True` is requested (for per-edge field decomposition),
K1 is completely bypassed:

```python
if execution.accumulate_primal == "custom_op_partitioned":
    if return_per_edge:
        raise ValueError("custom_op_partitioned does not support per-edge output")
```

This forces any workflow that needs per-edge diagnostics to fall back to the
pure DrJit path, losing all K1 benefits.

Root cause: The Slang kernel accumulates directly to receiver totals via atomic
adds. There is no output buffer for per-edge intermediate results because the
kernel's design goal was to eliminate exactly those intermediates.

### P7. Code duplication between DrJit and Slang paths

The UTD diffraction coefficient, Fresnel integral, slope diffraction, and vector
transport are implemented **twice**:
- In Python/DrJit (`utd.py`, `geometry.py`, `polarization.py`)
- In Slang (`utd_accumulate_math.slang`, `utd_accumulate_diffraction.slang`,
  `utd_accumulate_field.slang`)

Any numerical fix or formula change must be applied to both implementations and
validated for parity. The Slang backward kernels have **three variants**
(scalar, vector, full) with overlapping but not identical adjoint logic,
multiplying the maintenance surface further.

---

## 3. Refactoring Proposal

### Goal

Eliminate the DrJit <-> Slang bridge overhead, remove the safe/unsafe split,
and make K1 the unconditional default path with no fallback.

### R1. Replace slangtorch with DrJit C++ native + CUDA kernel

**Problem addressed**: P1 (tensor fragmentation), P4 (first-launch instability),
P5 (peak memory)

Instead of passing 53+ `TensorView<float>` through slangtorch's Python binding
layer, use DrJit's own C++ API to access GPU memory directly and launch CUDA
kernels with no framework intermediary:

```
Current:  DrJit arrays -> torch DLPack -> 32 TensorView<float> -> Slang kernel
Proposed: DrJit arrays -> jit_var_data() -> raw void*           -> CUDA kernel
```

**Recommended approach: DrJit C++ + nanobind + CUDA**

DrJit ships with full C++ headers (`drjit/custom.h`, `drjit-core/jit.h`),
link libraries (`drjit-core.lib`, `drjit-extra.lib`), and nanobind for Python
binding -- the same toolchain DrJit itself is built with. This enables a native
integration path that bypasses both slangtorch and PyTorch entirely:

1. **`jit_var_data(index, &ptr)`** -- zero-copy extraction of the raw CUDA
   device pointer from a DrJit JIT variable. Forces JIT evaluation if the
   variable is still lazy.
2. **`jit_malloc(AllocType::Device, size)`** -- allocate output buffers on
   DrJit's own CUDA memory pool, avoiding a second allocator competing for
   VRAM with DrJit's internal pool.
3. **`jit_var_mem_map(backend, type, ptr, size, free=1)`** -- zero-copy
   registration of a CUDA buffer as a DrJit variable. With `free=1`, DrJit
   automatically reclaims the memory when the variable's refcount drops to zero.
4. **`drjit::CustomOp<Output, Inputs...>`** -- C++-level CustomOp that hooks
   directly into the AD graph without Python dispatch overhead.
5. **nanobind** module binding to expose the fused operation to Python as a
   single callable that accepts and returns `dr.cuda.Float` arrays.

The CUDA kernels themselves are written in standard CUDA C++ (`__global__`
functions compiled by nvcc at build time) and launched via `cuLaunchKernel`
or the `<<<>>>` syntax.

Key advantages over alternative approaches:

| | slangtorch (current) | torch extension | **DrJit C++ native** |
|---|---|---|---|
| Bridge | DrJit→torch→TensorView→Slang | DrJit→DLPack→torch→data_ptr | **DrJit→jit_var_data→direct** |
| Conversions per launch | 32×2 (in+out) | 2 (DLPack in+out) | **0** |
| Extra dependency | slangtorch | torch cpp_extension | **none (DrJit ships headers+libs)** |
| Memory allocator | torch allocator | torch allocator | **DrJit allocator (unified)** |
| AD integration | Python dr.custom | Python dr.custom | **C++ dr::custom (no Python dispatch)** |
| CUDA intrinsics | not available | available | **available** |
| First-launch determinism | Slang JIT (suspect) | nvcc AOT (deterministic) | **nvcc AOT (deterministic)** |

Skeleton of the C++ bridge:

```cpp
#include <drjit/cuda.h>
#include <drjit/custom.h>
#include <nanobind/nanobind.h>

using Float  = drjit::CUDAArray<float>;
using UInt32 = drjit::CUDAArray<uint32_t>;

class UTDAccumulateOp : public drjit::CustomOp<UTDOutputs, Float, Float, Float> {
public:
    UTDOutputs eval(const Float& packed_state, const Float& rx_flat,
                    const Float& k_val) override {
        void *state_ptr, *rx_ptr;
        jit_var_data(packed_state.index(), &state_ptr);
        jit_var_data(rx_flat.index(), &rx_ptr);

        void* out_ptr = jit_malloc(AllocType::Device, n_rx * sizeof(PairOutput));

        launch_utd_forward_kernel(  // standard CUDA __global__ function
            (PackedState*)state_ptr, (float3*)rx_ptr,
            (PairOutput*)out_ptr, n_pairs, k);

        uint32_t idx = jit_var_mem_map(
            JitBackend::CUDA, VarType::Float32, out_ptr, n_rx, /*free=*/1);
        return UTDOutputs{ Float::borrow(idx), ... };
    }

    void forward() override { /* hand-written JVP, same pattern */ }
    void backward() override { /* hand-written VJP, same pattern */ }
};

NB_MODULE(utd_cuda_ext, m) {
    m.def("utd_accumulate", [](const Float& s, const Float& r, float k) {
        return drjit::custom<UTDAccumulateOp>(s, r, Float(k));
    });
}
```

Python side becomes trivial:

```python
import utd_cuda_ext
result = utd_cuda_ext.utd_accumulate(packed_state, rx_positions, k)
# result is dr.cuda.Float, AD graph is connected, no conversion needed
```

Why not other approaches:

- **Slang + custom PyBind wrapper**: keeps Slang JIT (first-launch suspect),
  still cannot use `__shfl_down_sync` / shared memory, gains little over
  current slangtorch.
- **torch extension**: works but introduces an unnecessary PyTorch allocator
  in the CUDA path, and the DLPack roundtrip (though cheap) is still an extra
  step that DrJit C++ native avoids entirely.
- **CuPy / PyCUDA**: adds a runtime dependency; DrJit already provides the
  same raw-pointer access natively.

### R2. Cache partition results in CustomOp

**Problem addressed**: P2 (repeated partition)

Cache the safe/unsafe chunk indices computed during `eval()` and reuse them in
`forward()` and `backward()`:

```python
class _EdgeStateTotalsOp(dr.CustomOp):
    def eval(self, state_inputs, rx_pos, k, ...):
        self._safe_chunks, self._unsafe_chunks = (
            _partition_visible_pair_dispatch_chunks(state_inputs, rx_pos, scene)
        )
        # Use cached chunks for forward computation
        return _hybrid_forward(state_inputs, rx_pos, k,
                               safe_chunks=self._safe_chunks,
                               unsafe_chunks=self._unsafe_chunks, ...)

    def forward(self):
        # Reuse cached partition -- no re-computation
        safe_chunks = self._safe_chunks
        unsafe_chunks = self._unsafe_chunks
        ...

    def backward(self):
        # Reuse cached partition -- no re-computation
        safe_chunks = self._safe_chunks
        unsafe_chunks = self._unsafe_chunks
        ...
```

Risk: DrJit may garbage-collect or invalidate Python-side objects attached to a
`CustomOp` between passes. This must be validated empirically and, if necessary,
the chunk indices can be stored as plain Python lists of integer arrays (not
DrJit-managed) to avoid GC interference.

### R3. Eliminate unsafe fallback with kernel-side pole handling

**Problem addressed**: P3 (dual execution path)

The unsafe path exists because the Slang VJP cannot handle cotangent poles and
slope derivative singularities. Two approaches to eliminate it:

**Approach A: Kernel-side numerical regularization**

Add guarded evaluation in the Slang kernel:

```slang
// Instead of raw cot(x):
float safeCot(float x) {
    float sinx = sin(x);
    float absSin = max(abs(sinx), POLE_GUARD);
    return cos(x) / copysign(absSin, sinx);
}
```

The forward value is already clamped this way in the DrJit path. The VJP for
`safeCot` is straightforward: `d/dx[-1/sin^2(x)]` clamped to the same guard
bound.

For slope derivatives, the finite-difference step can be widened near poles, or
the slope coefficient can be set to zero when the base coefficient itself is in
a singular region (physically, slope diffraction is negligible when the leading
UTD term dominates).

Trade-off: Introduces a small accuracy deviation near poles compared to the
DrJit replay path. This is acceptable because:
1. The pole region contributes negligible physical energy (cot -> inf means the
   observation direction is on the shadow/reflection boundary where GO dominates).
2. The current DrJit replay path also uses numerical guards (`pole_guard=1e-6`);
   matching the same guard in Slang produces identical results.

**Approach B: Analytical limit substitution**

At exact poles, the UTD transition function has known analytical limits. Replace
the singular `cot` evaluation with the limit form when `|sin(x)| < threshold`.
This is more accurate than clamping but requires implementing the limit
expressions for all four cotangent arguments and their derivatives.

**Recommendation**: Approach A first (minimal code change, matches existing DrJit
behavior), then Approach B for the `accuracy` solver mode if needed.

### R4. Packed state buffer with AoS layout

**Problem addressed**: P1 (fragmentation), P5 (peak memory)

Replace the 32 separate tensor allocations with a single contiguous buffer in
AoS (array-of-structs) layout:

```
struct PackedDiffractionState {  // 128 bytes aligned
    float3 edge_pos;             // 12
    float3 edge_dir;             // 12
    float3 n0;                   // 12
    float3 nn;                   // 12
    float  wedge_n;              //  4
    float3 source_pos;           // 12
    float2 incident_field;       //  8  (real, imag)
    float2 incident_derivative;  //  8
    float2 incident_vec_x;       //  8
    float2 incident_vec_y;       //  8
    float2 incident_vec_z;       //  8
    float2 derivative_vec_x;     //  8
    float2 derivative_vec_y;     //  8
    float2 derivative_vec_z;     //  8
    // Total: 120 bytes -> pad to 128 for alignment
};
```

Benefits:
- **One allocation** instead of 32 per chunk.
- **Better cache locality** -- all fields for one state are adjacent, improving
  L2 hit rate during kernel evaluation.
- **Simpler kernel signature** -- one pointer + stride instead of 32 TensorViews.
- Gradient output uses the same struct layout, halving gradient allocation count.

The packing can be done on the GPU with a simple copy kernel, or integrated into
the state-gathering phase that already exists in the diffraction pipeline.

### R5. Fuse partition into kernel with early-exit

**Problem addressed**: P2 (repeated partition), P3 (dual path)

Instead of pre-computing safe/unsafe masks on the Python side, move the safety
check into the kernel's per-thread prologue:

```slang
[CUDAKernel]
void utdAccumulateForward(..., int nPairs, float k, MaterialParams material) {
    uint idx = cyclic_thread_index();
    if (idx >= nPairs) return;

    PairInputs inp = loadPairInputs(state, stateIndex[idx]);
    float3 rx = loadRx(rxTensors, rxIndex[idx]);

    // In-kernel safety checks
    EdgeAngleCache cache = computeEdgeAngles(inp, rx);
    if (!isGeometryValid(cache, inp)) return;          // early exit
    bool poleSafe = isCotangentPoleSafe(cache, inp);

    // Use guarded evaluation regardless -- no safe/unsafe split
    PairOutputs out = computePairContributionGuarded(inp, rx, cache, k, material, poleSafe);

    atomicAddPairOutput(outputs, rxIndex[idx], out, ownershipCode[idx]);
}
```

This eliminates:
- The Python-side `_partition_visible_pair_dispatch_chunks()` entirely.
- The safe/unsafe chunk lists and their memory footprint.
- The dual execution path -- every pair goes through the same kernel.

The `poleSafe` flag is still computed per-thread but used to select between
the standard and guarded code path **within the same kernel**, not to route
between Slang and DrJit.

### R6. Streaming output accumulation for per-edge support

**Problem addressed**: P6 (per_edge incompatibility)

Add an optional per-edge output mode to the kernel:

```slang
// Mode 1: totals only (current behavior)
atomicAddPairOutput(totalOutputs, rxIndex[idx], out, ownershipCode[idx]);

// Mode 2: per-edge + totals
if (perEdgeEnabled) {
    int edgeSlot = edgeIndex[stateIndex[idx]];
    int perEdgeOffset = edgeSlot * nRx + rxIndex[idx];
    atomicAddComplex(perEdgeReal, perEdgeImag, perEdgeOffset, out.field);
}
atomicAddPairOutput(totalOutputs, rxIndex[idx], out, ownershipCode[idx]);
```

The per-edge buffer is allocated only when requested, so the default (totals-
only) path has zero extra cost.

### R7. Consolidate Slang backward variants

**Problem addressed**: P7 (code duplication)

The three backward kernels (`BackwardScalar`, `BackwardVector`, `BackwardFull`)
share ~70% of their adjoint code. Refactor into a single kernel with a runtime
mode flag:

```slang
[CUDAKernel]
void utdAccumulateBackward(
    ...,
    int backwardMode,  // 0=scalar, 1=vector, 2=full
    ...
) {
    // Shared: load inputs, compute forward cache
    PairInputs inp = loadPairInputs(...);
    EdgeAngleCache cache = computeEdgeAngles(...);

    // Shared: scalar adjoint (always needed)
    PairInputs.Differential scalarAdj = computeScalarAdjoint(...);

    // Conditional: vector transport adjoint
    if (backwardMode >= 1) {
        PairInputs.Differential vectorAdj = computeVectorTransportAdjoint(...);
        scalarAdj = addDifferentials(scalarAdj, vectorAdj);
    }

    // Shared: scatter gradients
    atomicAddPairInputDifferential(stateGrad, stateIndex[idx], scalarAdj);
}
```

The branch is uniform within a launch (all threads see the same mode), so there
is no warp divergence cost.

---

## 4. Implementation Phases

### Phase 1: Cache + consolidate (low risk, immediate gains)

1. Cache partition results in `_EdgeStateTotalsOp` (R2).
2. Consolidate backward kernels into one (R7).
3. Validate that cached partitions survive DrJit's AD lifecycle.

Expected impact: Eliminate 2x redundant partition computation; reduce code
maintenance surface.

### Phase 2: Kernel-side pole handling (medium risk, removes fallback)

1. Implement guarded cotangent and slope evaluation in Slang (R3-A).
2. Fuse safety checks into kernel prologue (R5).
3. Remove `_partition_visible_pair_dispatch_chunks()` and the DrJit replay path.
4. Validate numerical parity against the current safe-chunk Slang output.

Expected impact: Eliminate dual execution path; simplify the entire K1 flow to
a single kernel launch per forward/backward.

### Phase 3: DrJit C++ native + CUDA kernel (high impact, recommended path)

1. Set up nanobind + nvcc build infrastructure for the extension module.
2. Define `PackedDiffractionState` C++ struct matching CUDA kernel layout (R4).
3. Implement GPU packing kernel (DrJit SoA arrays -> AoS packed buffer).
4. Translate Slang math to CUDA `__device__` functions (mechanical translation
   of `utd_accumulate_math.slang`, `utd_accumulate_diffraction.slang`,
   `utd_accumulate_field.slang` -- standard float arithmetic, no Slang-specific
   features used).
5. Implement C++-level `drjit::CustomOp` with `jit_var_data` / `jit_malloc` /
   `jit_var_mem_map` for zero-copy DrJit integration (R1).
6. Add per-edge output mode (R6).
7. Validate numerical parity against current Slang kernel output.
8. Remove slangtorch dependency from the diffraction runtime.

Expected impact: Eliminate tensor fragmentation; unified DrJit memory allocator
reduces peak memory; AOT-compiled CUDA kernels fix first-launch determinism;
enable K1 as the unconditional default.

### Phase 4: Kernel-level performance optimization

1. Warp-level reduction (`__shfl_down_sync` / `__match_any_sync`) before
   `atomicAdd` to reduce atomic contention on popular receivers.
2. Shared memory tiling for receiver accumulation in high-density grids.
3. `__launch_bounds__` tuning for backward kernel register pressure.
4. Optional CUB-based scan/compact for dispatch index generation.

Expected impact: Further 20-35% throughput improvement on memory-bound
forward/backward kernels.

---

## 5. Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| R2 (cache partition) | Low | Test with existing parity suite; fall back to recompute if cache is invalidated |
| R3 (pole handling) | Medium | Per-pair regression test against DrJit reference; document accuracy bounds |
| R4 (packed buffer) | Medium | Alignment and padding must match across C++/CUDA; use `static_assert` on struct size |
| R5 (fused partition) | Medium | Must preserve visibility semantics exactly; test against current partition output |
| R1 (DrJit C++ native) | Medium | DrJit C++ API is stable and self-consistent (same API DrJit uses internally); nanobind build is well-documented; main risk is `jit_var_data` interaction with lazy evaluation -- must call `dr.eval()` before pointer extraction |
| Slang→CUDA translation | Medium | ~1500-2000 lines of standard float math; validate per-pair output against Slang reference before switching |
| R7 (consolidate backward) | Low | Existing backward parity tests cover all three variants |

---

## 6. Success Criteria

1. K1 becomes the **default** execution path (`accumulate_primal="partitioned_slang"` or better).
2. First-launch forward repeatability passes `test_default_trace_repeatability.py`.
3. Peak memory is at or below the current K1=off baseline (2.495 GiB).
4. Backward time stays at or below current K1 best (0.281s).
5. No DrJit replay fallback exists in the diffraction accumulation path.
6. `return_per_edge=True` works with K1 enabled.
7. Single source of truth for UTD math: CUDA `__device__` functions, replacing
   both Slang kernels and DrJit Python implementations.
8. Zero PyTorch dependency in the diffraction accumulation path (DrJit C++
   native bridge via `jit_var_data` / `jit_var_mem_map` / `jit_malloc`).
