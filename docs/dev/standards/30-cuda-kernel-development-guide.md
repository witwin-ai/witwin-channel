# CUDA Kernel Development Guide

Status: Active
Category: Standard
Last reviewed: 2026-05-17

Date: 2026-04-01

## Why This Note Exists

This note records the decisions, pitfalls, and implementation standards that came out of the multipath native + symbolic migration work during late March and early April 2026.

The goal is to prevent repeated mistakes when adding or migrating CUDA kernels for:

- diffraction builders
- reflection EPC
- packed-state helpers
- UTD accumulation and AD
- grid accumulation kernels

This is not a benchmark report. It is a development guide and a trap-avoidance document.

## Non-Negotiable Rules

These rules are now the default standard for new kernel work in this repository.

1. GPU zero-copy is required on the main runtime path.
2. Do not introduce Torch or DLPack bridges into kernel hot paths.
3. Do not introduce Python-side raw device-pointer bridges for new kernels.
4. C++ bindings must accept Dr.Jit arrays directly.
5. New runtime paths must not depend on `evaluated` loops.
6. Native is the default runtime target. Dr.Jit is a baseline and reference implementation.
7. Do not add CPU staging or CPU fallback logic to solve GPU performance issues.
8. Do not freeze discovery for AD-sensitive fresh traces.
9. Torch is allowed only at explicit public API boundaries or result-adapter layers. It must not become the internal transport between Dr.Jit arrays, C++ bindings, and CUDA kernels.
10. Do not introduce NumPy staging on runtime field, path, replay, accumulation, or AD-sensitive execution paths.

## Required Data Path

The required data flow for new kernels is:

1. Python builds Dr.Jit arrays.
2. Nanobind binding receives Dr.Jit arrays directly.
3. C++ binding obtains device pointers from the Dr.Jit arrays.
4. CUDA kernel runs directly on those device pointers.
5. Outputs are returned as Dr.Jit arrays.
6. If AD is needed, the Dr.Jit custom op must own `eval()`, `forward()`, and `backward()` in C++.

The path that should be avoided is:

1. Python converts Dr.Jit arrays to Torch.
2. Python exports DLPack or raw pointers.
3. Python passes integer pointers back into C++.
4. C++ launches the kernel from those Python-extracted pointers.

That path is fragile, harder to reason about, and breaks the zero-copy design intent.

The only acceptable Torch usage in this repository is at an explicit boundary layer where a public API or convenience helper intentionally adapts Dr.Jit-native results into torch tensors for downstream consumers. That boundary must stay optional and outside the core runtime path. It must not sit in the middle of kernel launch, replay, accumulation, or AD-sensitive execution.

## Preferred Binding Pattern

Use `witwin/channel/_native/shared/drjit_common.h` as the canonical base layer.

Key helpers:

- `drjit_data_ptr(...)`
- `drjit_data_ptr_mut(...)`
- `array_pointer_list(...)`
- `array_pointer_list_mut(...)`
- `WitwinCustomOp`
- `witwin_custom_op(...)`

For AD-capable kernels, prefer this pattern:

```cpp
struct FooInput {
    DiffFloat x;
    DiffFloat y;
    int n;

    DRJIT_STRUCT(FooInput, x, y, n);
};

struct FooOutput {
    DiffFloat out;

    DRJIT_STRUCT(FooOutput, out);
};

class FooOp final : public WitwinCustomOp<FooOutput, FooInput> {
public:
    explicit FooOp(const FooInput &input) : WitwinCustomOp<FooOutput, FooInput>(input) {}

    FooOutput eval(const drjit::detached_t<FooInput> &input) override {
        ...
    }

    void forward() override {
        ...
    }

    void backward() override {
        ...
    }
};
```

If `backward()` is missing, the op is not complete. A Python wrapper may temporarily hide that gap, but that is not the final architecture.

## When To Use Raw Array Bindings vs Custom Ops

Use raw array bindings when:

- the helper is primal-only
- the helper is a narrow builder-side utility
- the result is immediately consumed by Python-side symbolic logic
- AD is not required on that helper path

Use a C++ Dr.Jit custom op when:

- the kernel is on the main primal path and AD matters
- forward-mode JVP is required
- reverse-mode backward is required
- the kernel should be the runtime default, not just a benchmark helper

## Reference Patterns Already In The Tree

Use these as the primary reference implementations:

- `witwin/channel/deterministic/kernels/common/device.h` and `cuda_check.h` for shared SoA/device helpers
- `witwin/channel/deterministic/kernels/utd/bind.h`
- `witwin/channel/deterministic/kernels/reflection/bind.h`
- `witwin/channel/deterministic/kernels/radio_map_accumulate/bind.h`
- `witwin/channel/deterministic/kernels/packed_state/bind.h`

For builder-side narrow helpers, also use:

- `witwin/channel/deterministic/kernels/packed_state/native_impl.py`
- `witwin/channel/montecarlo/kernels/diffraction_builder/` for Monte Carlo-side diffraction builder kernels

For Monte Carlo replay and accumulation ownership, use:

- `witwin/channel/montecarlo/kernels/transport_grid/` and `witwin/channel/montecarlo/kernels/monte_carlo/`

## Major Pitfalls We Hit

### 1. Frozen EPC was incorrectly used on fresh AD-sensitive traces

Problem:

- EPC was designed for replaying a fixed discovered path set
- discovery was frozen
- fresh geometry/position AD workloads were still being routed into that replay path
- position and rotation gradients became zero

Fix:

- only use frozen replay for explicit `reflection_detail` reuse or path-audit workflows
- if geometry or TX position carries AD, preserve the requested native backend instead of silently freezing discovery
- fresh field traces now honor the requested backend by default, even when EPC would be technically eligible

Rule:

- never freeze discovery on a fresh AD-sensitive trace
- do not make frozen EPC the default field-monitor backend for fresh traces

### 2. UTD replay fallback caused OOM and unstable benchmark evaluation

Problem:

- native UTD primal existed
- JVP and backward still replayed large Dr.Jit reference paths
- full `benchmark_multipath` reverse ran into large memory pressure and could fill GPU memory

Fix:

- remove replay fallback from the active UTD JVP/VJP path
- implement `backward()` on the inner C++ custom op
- remove the outer Python backward wrapper after inner C++ backward is working

Rule:

- do not treat a native kernel as complete until its actual runtime AD path is also native

### 3. Python raw-pointer bridges are not acceptable as the long-term design

Problem:

- `float_ptr(...)`, `int_ptr(...)`, or similar Python-side pointer extraction is tempting
- it looks simple
- it is not the architecture we want
- it is easier to mis-own memory and easier to produce invalid or stale pointer behavior

Fix:

- bind Dr.Jit arrays directly in C++
- extract pointers in the nanobind/C++ layer

Rule:

- no new kernel should depend on Python-side pointer extraction as its main launch API
- no new kernel should depend on Torch tensors as its internal transport type

### 4. Torch and DLPack are not allowed in kernel hot paths

Problem:

- Torch interop is convenient for debugging and prototyping
- it is not acceptable on the hot runtime path
- it breaks the intended Dr.Jit-native zero-copy design

Fix:

- keep Torch interop out of new runtime kernels
- use Dr.Jit arrays end-to-end

Rule:

- no Torch view, no DLPack handoff, no Torch staging in kernel main paths
- no NumPy staging in kernel main paths

### 5. Generic packed-state subset was too wide for inserted-reflection

Problem:

- `subset_state_arrays(...)` is convenient
- it unpacks or gathers far more state than the inserted-reflection builder actually needs
- this makes builder-side packing slower than necessary

Fix:

- add narrow packed-state field gathers for the exact subset of fields needed
- keep the helper specialized to the builder use case

Rule:

- do not use a full-state subset helper when the builder only needs a narrow field slice

### 6. Scalar-loss optimization benchmarks should not build full monitor payloads

Problem:

- the AD benchmark historically reduced the field to a scalar loss
- the trace path still assembled the full `field`, `vector`, and `jones` monitor payload
- reflection detail also retained receiver-wide polarization/Jones arrays even when downstream code only needed replay metadata
- this inflated JVP/VJP graph size and allocator pressure for no optimization-workload benefit

Fix:

- add an explicit scalar-loss workload mode to the benchmark
- route that workload through a total-field-only trace helper
- keep runtime backend selection and timing metadata, but omit full receiver payloads that are not consumed by the scalar loss

Rule:

- if the workload ends at a scalar loss, do not keep a full monitor payload alive just for benchmark convenience

### 7. First benchmark run can be misleading

Problem:

- kernel compile and graph warmup can dominate the first run
- a path may look slower when it is only suffering compile noise

Fix:

- rerun the benchmark at least once
- compare steady-state numbers, not the first-run numbers

Rule:

- never make a migration decision from a single first-run benchmark

### 8. Unrelated CUDA compile errors can hide new work

Problem:

- a new feature may fail to rebuild because an unrelated file already has a compile ambiguity
- this happened with `reflection_grid_backward.cu` and `suffix_grid_backward.cu`

Fix:

- isolate the actual compiler error
- clear unrelated build blockers immediately
- then continue validating the new kernel

Rule:

- do not assume the newest edited file is the source of the rebuild failure

### 8. Builder kernels and replay kernels should be optimized in slices

Problem:

- trying to migrate an entire builder family in one step makes debugging too expensive

Fix:

- split the migration into narrow slices:
  - dedup
  - visibility
  - field gather
  - pack
  - replay

Rule:

- one measurable slice at a time, with parity tests for each slice

## Current Implementation Standards

### Standard A: No `evaluated` runtime path

The historical `evaluated` path must stay removed.

Implication:

- `suffix_dda` is symbolic-only
- do not reintroduce `evaluated` branches for debugging or fallback

### Standard B: Native first, Dr.Jit baseline second

Runtime intent:

- native path is the default production target
- Dr.Jit remains the reference baseline for correctness and parity checks

Implication:

- if a native path is incomplete, fix the native path
- do not expand the lifetime of Python-side wrapper fallback paths unless strictly temporary

### Standard C: C++ owns the device pointers

Implication:

- Python should pass arrays, not integer pointers
- C++ binding should be the place that turns arrays into device pointers

### Standard D: Custom op completeness matters

A custom op is only considered complete if:

- `eval()` exists
- `forward()` exists when JVP matters
- `backward()` exists when reverse AD matters

Anything else is transitional.

### Standard E: Builder helpers may be primal-only, but must still be zero-copy

This is acceptable:

- a narrow helper used only inside symbolic builder code
- direct Dr.Jit-array input
- direct CUDA gather
- no Torch, no DLPack, no Python raw pointer bridge

### Standard F: Kernel wrappers route extension lookup through `_native` helpers

Every Python-side native kernel wrapper must declare its required C++ symbol
names as a module-level tuple and call into the package's `_native` loader for
both presence checks and load-with-error-on-missing. Do not re-implement the
`ext = _extension()` + `hasattr` + `raise RuntimeError(...)` pattern in each
kernel file.

The centralized `witwin/channel/_native/{channel_utils,deterministic,montecarlo}.py` specs provide two helpers through `NativeExtension`:

- `has_functions(names) -> bool` — `True` when the extension is loadable
  AND every name in `names` is bound. Use this for runtime feature-detection
  (e.g. choosing between native and Dr.Jit fallback paths).
- `require_functions(names, *, context) -> ext` — load the extension and
  return it; if any name in `names` is missing, raise
  `RuntimeError(f"{context} requires <missing names>. Rebuild the witwin.<pkg> native extension.")`.

Wrappers import the appropriate centralized spec module and call these helpers
as static methods on `NativeExtension`.

Reference shape:

```python
# witwin/channel/<solver>/kernels/<name>/native_impl.py
from witwin.channel._native.<extension> import NativeExtension

_REQUIRED = (
    "<package>_<kernel>_forward_raw",
    "<package>_<kernel>_jvp_raw",
    "<package>_<kernel>_backward_raw",
)


def _require_native_kernel():
    return NativeExtension.require_functions(
        _REQUIRED,
        context="Native <kernel> backend",
    )


def _native_kernel_available() -> bool:
    return NativeExtension.has_functions(_REQUIRED)
```

For class-style wrappers (montecarlo / channel core):

```python
class <Name>Kernel:
    _REQUIRED = ("<symbol_1>", "<symbol_2>")

    @staticmethod
    def require():
        return NativeExtension.require_functions(
            <Name>Kernel._REQUIRED, context="<human-readable context>",
        )

    @staticmethod
    def available() -> bool:
        return NativeExtension.has_functions(<Name>Kernel._REQUIRED)
```

Rules:

- Required-symbol tuples are declared once at module top and reused by both
  `require_*` and `*_available` helpers; do not duplicate the list across the
  two helpers.
- The `context` string must name the kernel or operation in human-readable
  form; the loader appends the rebuild instruction. Do not include the words
  "requires" or "Rebuild..." in `context`.
- For one-off feature flags inside a larger function, prefer
  `has_functions((symbol,))` over an inline `ext = _extension(); hasattr(...)`
  expression.
- Never hand-roll the error message; `require_functions` owns the canonical
  rebuild-the-extension instruction so it stays consistent across packages.

This keeps each kernel wrapper focused on its launch logic and concentrates
loader semantics in one place per package.

### Standard G: Batch replay kernels should own the math-heavy replay body

For reflection-prefix replay, the math-heavy parts are:

- plane replay
- reflected TX reconstruction
- Jones transport through the chain
- replay hit-point generation

Visibility and mesh-surface validation may temporarily remain in Python if needed for correctness, but the replay body should move into the kernel.

## Implementation Checklist For New Kernels

Before coding:

1. Decide whether the kernel is primal-only or full AD.
2. Identify the minimal input field set.
3. Confirm that the kernel can stay fully on GPU.
4. Confirm that no Torch or DLPack bridge is needed.

While coding:

1. Add `.h` and `.cu` files if new CUDA code is needed.
2. Add nanobind registration in the appropriate `bind.h`.
3. Add the source file to the relevant `witwin/channel/_native/<extension>/CMakeLists.txt` if needed.
4. Add the binding include and registration call to the relevant `witwin/channel/_native/<extension>/module.cpp` if needed.
5. Prefer `DRJIT_STRUCT` plus `WitwinCustomOp` for AD-complete kernels.
6. Prefer raw array-returning bindings for narrow primal-only helpers.

After coding:

1. Run `py_compile` on the touched Python files.
2. Rebuild the editable package in the `witwin2` environment.
3. Run targeted parity tests first.
4. Run the relevant gradient regression tests.
5. Run the benchmark at least twice and look at steady-state.

## Validation Standard

For kernel work in this repository, the minimum validation set is:

1. Targeted parity tests against the Dr.Jit reference path.
2. At least one AD-sensitive regression test if the change affects gradients.
3. A small `benchmark_multipath_ad --workload scalar_loss` JVP run.
4. A small `benchmark_multipath_ad --workload scalar_loss` VJP run.
5. A steady-state `benchmark_multipath` rerun.

For geometry-sensitive work, always include:

- `tests/main/test_position_rotation_tx.py -k test_position_rotation_tx_visual_grid --gpu`

That test catches several classes of AD-routing mistakes.

## Specific Lessons From The Multipath Builder Work

### Inserted-reflection builder

What worked:

- moving from generic full-state subset to a dedicated packed-field gather

What did not fully solve the hotspot:

- `field_to_hit_seconds`
- remaining `state_pack_seconds`

Current conclusion:

- the dedicated gather is correct and should stay
- the next bottleneck is no longer the generic subset API itself
- the next work should target the builder body after the gather

### Reflection-prefix replay

What worked:

- adding a native batch replay-to-target kernel for primal batch work
- keeping the replay math body on the native forward path even when gradients are active
- adding direct JVP/VJP parity coverage for replay-to-target so the active benchmark path no longer depends on a Python reference fallback

What remains:

- the replay-to-target AD wrapper still lives in a Python custom op instead of a C++ `WitwinCustomOp`

Current conclusion:

- native-forward plus a parity-checked custom AD wrapper is acceptable for the current Phase 2 / Phase 3 closure
- moving that wrapper into a C++ custom op is now architecture cleanup, not the current benchmark critical path

## What To Avoid In Future Work

Do not do these again:

- routing fresh AD-sensitive reflection discovery through frozen EPC
- declaring a kernel migration complete while JVP/VJP still replay Dr.Jit internally
- adding Torch or DLPack to a kernel hot path
- using Python-side pointer extraction as the main interface for a new kernel
- reviving `evaluated` loops as a shortcut
- benchmarking only the first run
- changing multiple large builder stages at once without a parity test per slice

## Current Open Follow-Ups

As of 2026-04-01 after the latest Phase 5 reruns, the next logical work items are:

1. Add monitor tiling or chunked accumulation only if the scalar-loss workload is still too heavy at larger monitor scales.
2. Add a first-class symbolic graph-size counter if allocator snapshots stop being diagnostic enough.
3. Only revisit remaining builder-body fusion or replay custom-op ownership if fresh post-rebaseline data justifies it.

## One-Sentence Summary

The correct architecture for this repository is: Dr.Jit arrays in, C++ owns the pointers, CUDA does the work, Dr.Jit custom ops own AD, and Torch/DLPack/Python pointer bridges stay out of the runtime hot path.
