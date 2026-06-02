# Performance Bottleneck Analysis & CUDA Migration Plan

## 一、架构概览

当前计算管线：

```
Tracer.trace(tx_pos)
  ├── LoS:  ray_test → free-space propagation        (~77 行，轻量)
  ├── Reflection:
  │     ├── _trace_reflection_paths()               (射线发射 + 多次弹射)
  │     ├── _collect_unique_reflection_paths()       (路径去重)
  │     └── accumulate_reflection_paths_to_receivers() (replay + scatter_reduce)
  └── Diffraction:
        ├── State Builders:  TX / prefix / higher-order / inserted-reflection
        ├── Pruning:  power-based lexsort + budget cutoff
        ├── Field:  _accumulate_edge_states_to_receivers()
        │     ├── _edge_state_field_to_targets()     (UTD 系数 + Jones 传输)
        │     └── scatter_reduce per receiver
        └── Suffix reflection (optional)
```

---

## 二、现有 Native 基础设施

### 2.1 构建体系

项目已有 **DrJit + CUDA + nanobind** 原生扩展骨架：

```
channel/
├── CMakeLists.txt              # 根 CMake (scikit-build-core)
├── pyproject.toml              # build-requires: nanobind>=2.11, drjit>=1.2.0
├── witwin/channel/_native/
│   ├── CMakeLists.txt          # 两个 target:
│   │   │                       #   witwin_channel_utils_native_cuda (STATIC .cu)
│   │   │                       #   _channel_utils_native (nanobind module)
│   ├── shared/
│   │   └── sample_cuda.h
│   └── src/
│       ├── module.cpp          # nanobind 绑定 + DrJit type_caster
│       └── sample_cuda.cu      # 样例 CUDA kernel
├── witwin/channel/
│   └── _native/                # 公共 Python loader/specs + Windows DLL 路径配置
```

### 2.2 DrJit 类型桥接 (已就绪)

`module.cpp` 中已实现完整的 `drjit_type_caster<T>` 模板：

```cpp
using Float    = drjit::CUDAArray<float>;       // drjit.cuda.Float
using DiffFloat = drjit::CUDADiffArray<float>;  // drjit.cuda.ad.Float
```

- Python → C++: `drjit_try_load<T>()` 零拷贝 inst_ptr
- C++ → Python: `drjit_from_cpp<T>()` 直接写 inst_ptr
- 支持 Float / DiffFloat 重载分发
- 链接 `drjit`, `drjit-core`, `drjit-extra`, `nanothread`, `CUDA::cudart`

**结论**：新 kernel 只需在 C++ 侧操作 `drjit::CUDAArray<float>` / `drjit::CUDADiffArray<float>`，经 nanobind 导出后 Python 侧直接接收 `dr.cuda.Float` / `dr.cuda.ad.Float`，**无需经过 PyTorch 中间层**。

### 2.3 Reference Slang 原型 (已有 UTD forward/backward)

`reference/slang_kernels/.../diffraction/kernels/` 下已有完整的 Slang 实现：

| 文件 | 内容 |
|------|------|
| `utd_accumulate_base.slang` | Complex / Jones / PairInputs / PairOutputs 结构体 |
| `utd_accumulate_math.slang` | Boersma Fresnel 积分 (含一阶/二阶导数)、`diffractionBetaGroups3D`、slope 衍射系数 |
| `utd_accumulate_diffraction.slang` | Edge angle、face reflection、Jones operator 合成、vector transport |
| `utd_accumulate_field.slang` | 3D edge geometry、validity mask、pair field 计算 |
| `utd_accumulate_forward.slang` | `[CUDAKernel] utdAccumulateForward` — (state, rx) pair → atomic 累积 |
| `utd_accumulate_backward.slang` | `[CUDAKernel] utdAccumulateBackwardScalar` — reverse-mode VJP |
| `utd_accumulate_tensors.slang` | Tensor 读写 + `atomicAddComplex` 梯度散射 |

这些 Slang 原型已验证了数学正确性和 AD 结构，可作为 CUDA kernel 的直接参考蓝图。

### 2.4 迁移路径

```
.cu kernel (手写 CUDA)
  → witwin_channel_utils_native_cuda (static lib)
    → module.cpp (nanobind 导出, DrJit type_caster)
      → Python: dr.cuda.Float / dr.cuda.ad.Float
```

---

## 三、六大核心瓶颈

> **关于 DrJit JIT fusion 的说明**：DrJit 的 JIT 引擎会做 symbolic tracing，将连续的 element-wise 操作（算术、`dr.select`、`dr.fma` 等）融合成一个 kernel，只在遇到 **物化点** 时才编译并 launch。物化点包括：`dr.eval()`、`dr.scatter_reduce()`、`dr.compress()`、`dr.width()` 读值、ray intersection (`scene.ray_intersect`)、以及任何跨 Python loop 边界的依赖。
>
> 因此实际 kernel launch 数 **远少于** Python-level 的 DrJit 操作数。下文的分析聚焦于**物化点之间的 kernel 数**和**不可被 JIT 消除的结构性开销**。

### 瓶颈 1: Diffraction 内循环的物化点密度

`_accumulate_edge_states_to_receivers()` (`field.py:368`) 的内循环中，每个 state chunk 的物化点链：

```
dr.compress(visible)           ← 物化点 1: visibility filter (如果开启 scene occlusion)
_gather_state_arrays(...)      ← gather 操作本身不物化，但 53+ 字段的 gather 构成巨大 JIT graph
_edge_state_field_to_targets() ← 内部全部 element-wise，JIT 可融合...
                                  ...但 _diffraction_beta_groups_3d 调用 3 次 (3 组 R0/Rn)
                                  ...每次内部 4 个 _beta_term_state → 共 12 次 f_utd_with_derivatives
                                  ...整棵 expression tree 约 500-800 个 DrJit node
dr.scatter_reduce() × 12      ← 物化点 2-13: 每次 scatter_reduce 强制 eval 其输入子图
```

**核心问题不在于 kernel launch 数量，而在于**：
1. **scatter_reduce 不可融合**：12 次 `dr.scatter_reduce` 各自触发一次 kernel launch，且它们之间共享大量输入子图，但 DrJit 不会跨 scatter_reduce 边界做 CSE（公共子表达式消除）——每次 scatter_reduce 都会重新编译包含其输入的 mega-kernel
2. **JIT graph 规模过大**：500-800 node 的 expression tree 使得 DrJit JIT 编译时间本身成为开销（首次编译可达数十毫秒，后续命中缓存则快得多，但 width 变化会 invalidate cache）
3. **重复计算**：`operator_terms_3d` 被调用 3 次（不同 R0/Rn），DrJit 可以通过 CSE 优化其中的共用部分，但 dict-based Python 分发使得 JIT 无法看到整体结构

**实际 kernel launch 预估**：每 chunk 约 **13-17 次**（1 compress + 12 scatter_reduce + 可选的 visibility ray test + 可选的 per_edge scatter_reduce），而非之前估计的 200+。但每次 launch 的 kernel body 很大（数百条 PTX 指令）。

### 瓶颈 2: 衍射状态数爆炸 (State Explosion)

**增长公式**：

```
N(order=1) ≈ 0.5 × E                                    (TX→edge 直射)
           + Σ_bounce(n_paths_b × E × k_prefix)          (反射前缀)

N(order≥2) ≈ N(order-1) × E × k_distinct × k_vis × k_power
           + N(order-1) × n_rays × k_surface_edges       (插入反射)
```

**具体示例 (E=200 edges)**：

| 阶数 | 未剪枝状态数 | 剪枝后 (budget=2000) | 峰值中间内存 |
|------|-------------|---------------------|-------------|
| 1 | ~120 | 120 | <1 MB |
| 2 | ~3,900 | 2,000 | ~200 MB (含 gather 暂存) |
| 3 | ~69,000 | 2,000 | ~800 MB - 1.2 GB |
| 4 | ~69,000 | 2,000 | ~1.2 GB |

**关键问题**：
- 笛卡尔积 `prev_states × edges` 在 filter 之前就创建了完整数组
- `_gather_state_arrays()` 53+ 字段的 gather 虽然 JIT 可融合，但生成的 expression tree 非常宽
- `_concat_state_arrays()` 调用 `dr.concat` — 这是一个内存操作（memcpy），每个字段独立执行，无法与计算融合
- 每个 state 约 436 bytes，53+ 个字段的 SoA 布局

### 瓶颈 3: scatter_reduce 结构性限制

**Diffraction field 累积** (`field.py:419-448`)：
```python
for axis in ("x", "y", "z"):
    dr.scatter_reduce(..., direct_vector_total[axis].real, ..., rx_idx)   # 1
    dr.scatter_reduce(..., direct_vector_total[axis].imag, ...)           # 2
    dr.scatter_reduce(..., multi_vector_total[axis].real, ...)            # 3
    dr.scatter_reduce(..., multi_vector_total[axis].imag, ...)            # 4
```

- **每 chunk 12 次 scatter_reduce** — 这是 DrJit 无法融合的物化点
- 每次 scatter_reduce 独立触发 JIT compilation + kernel launch
- 12 次 launch 中的输入计算子图高度重叠，但无法跨 scatter 边界共享
- `rx_idx = pair_idx % n_rx` 产生跳跃写入，atomic contention

**Reflection 累积** (`accumulation.py:60-74`)：同样模式，6 次 scatter_reduce/chunk。

### 瓶颈 4: Reflection Chain Replay 的 Python 循环

`epc_reflection_chain_to_target()` (`epc.py:135-220`)：

```python
for slot in range(chain_depth - 1, -1, -1):     # 循环 1: 反向展开
    prim_idx = dr.gather(...)
    plane_point = dr.gather(...)
    # intersection test + reflect_point_across_plane

for slot, hit_p in enumerate(hit_points):         # 循环 2: visibility check
    visible = _segment_visibility_mask(...)         # ← 每次调用 scene.ray_test: 物化点

for hit_p, geom_n, prim_idx in zip(...):          # 循环 3: Fresnel + Jones
    reflect_field_vector(...)
```

- 循环 1 和 3 中的 element-wise 操作可以被 JIT 融合，但循环 2 中每次 `_segment_visibility_mask()` 都调用 ray intersection → **强制物化**
- chain_depth 次 ray intersection 只能串行执行
- 每条路径约 1500 sequential FLOPs + chain_depth 次 ray test launch

### 瓶颈 5: CPU-GPU 同步点

| 传输点 | 位置 | 方向 | 频率 |
|--------|------|------|------|
| Torch mesh → DrJit Point3f | `tracer.py:_coerce_vertices()` | CPU→GPU | 每帧 1 次 |
| DrJit → Torch (路径去重) | `paths.py:81-89` | GPU→GPU (DLPack view) | 每 trace 1 次 |
| Pruning: metric → torch lexsort | `pruning.py:55` | GPU→CPU→GPU | 每阶每次 |
| Higher-order dedup | `higher.py:118-138` | GPU→GPU (view) + torch.unique | 每阶每次 |
| `array_scalar()` (协面性检查) | `compile.py:452-473` | GPU→CPU sync | 每 edge pair |
| `dr.eval()` 强制同步 52 值 | `arrays.py:283` | pipeline flush | 每 state batch |
| `dr.compress()` → `dr.width()` | `higher.py` 多处 | 隐式 sync (读 width) | 每 filter stage |

**最严重的**：`array_scalar()` 在 scene 编译的 coplanarity loop 中逐 edge 调用；以及 `higher.py` 中多处 `dr.compress()` 后立即 `dr.width() == 0` 检查导致的 GPU pipeline stall。

### 瓶颈 6: operator_terms 重复计算

`assemble_material_diffraction_operators()` (`operator.py:131-195`) 中：

```python
# _diffraction_beta_groups_3d 被调用 3 次，R0/Rn 不同:
factor, dif_group, ... = _diffraction_beta_groups_3d(..., zero, zero)   # 直射项
_, _, _, _, sum_plus, ... = _diffraction_beta_groups_3d(..., one, zero)  # face0
_, _, _, _, sum_minus, ... = _diffraction_beta_groups_3d(..., zero, one) # face1
```

每次 `_diffraction_beta_groups_3d` 内部调用 4× `_beta_term_state` → 4× `f_utd_with_derivatives` → 4× `fresnel_integral_with_derivatives`。三次调用中，`phi, phi_prime, n, k, s, s_prime, sin_beta0` 完全相同，4 个 `_beta_term_state` 的 `cot`/Fresnel 积分也完全相同。DrJit JIT 的 CSE **理论上可以消除重复**，但：

- 三次调用在 Python 层面是独立的函数调用，返回不同的 dict
- JIT 需要在 ~300 node 的 graph 中做 CSE match，且 Complex2f 的 real/imag 分别追踪
- 实际效果取决于 DrJit 版本和 graph 规模，无法保证

在 fused kernel 中这些可以通过 C++ 变量复用显式消除。

---

## 四、CUDA Kernel 迁移优先级列表

按 **性能收益 × 可行性** 排序。所有 kernel 通过现有 nanobind 基础设施导出为 DrJit 原生类型。

### 收益来源说明

迁移到手写 CUDA 的收益**不主要来自** kernel launch 数的减少（DrJit JIT 已做了 fusion），而来自：

1. **消除 scatter_reduce 物化点边界**：在一个 kernel 内完成计算 + scatter，避免 12 次独立 launch 及输入子图的重复编译
2. **显式 CSE 和寄存器复用**：`_beta_term_state` 的共用中间值只算一次
3. **Warp-level reduction**：多个 state 写同一 rx 时可先做 warp shuffle reduce 再 atomic
4. **消除 Python 循环的 pipeline stall**：`replay` 的 chain_depth 循环、`higher.py` 的 chunk 循环
5. **紧凑内存布局**：替换 53-field SoA dict 为结构化 buffer

---

### P0 — 最高优先级

| # | 功能 | 迁移收益来源 | Slang 参考 | 预估提升 |
|---|------|------------|-----------|---------|
| **1** | **UTD field mega-kernel** (field evaluation + scatter_reduce) | 消除 12 次 scatter_reduce 物化点；显式 CSE _beta_term_state 共用值；warp-level reduction before atomic | `utd_accumulate_forward.slang` | **2-5x** |
| **2** | **UTD backward mega-kernel** | 反向传播同样受 scatter_reduce 碎片化影响；手写 backward 可与 forward 共享中间值缓存 | `utd_accumulate_backward.slang` | **2-5x** |

### P1 — 高优先级

| # | 功能 | 迁移收益来源 | Slang 参考 | 预估提升 |
|---|------|------------|-----------|---------|
| **3** | **Reflection accumulation** (replay + scatter_reduce) | 消除 chain_depth 次 Python 循环间的 pipeline stall；将 replay + field + scatter 融合为单 kernel | `adjointVectorTransport` | **2-4x** |
| **4** | **State array packed buffer** | 将 53-field SoA dict 替换为连续内存的 packed struct buffer；gather/concat 变为 single-stride memcpy | — | **1.5-3x** (减少 JIT graph 宽度 + memory bandwidth) |

### P2 — 中优先级

| # | 功能 | 迁移收益来源 | 预估提升 |
|---|------|------------|---------|
| **5** | **Higher-order Cartesian filter** | 消除 chunk 循环中每次 `dr.compress()` + `dr.width()` 的 sync；fused filter + compaction | **1.5-3x** |
| **6** | **Pruning sort** | 将 DrJit→Torch→lexsort→DrJit 的 round trip 替换为 GPU-native radix sort (CUB) | **1.5-2x** |

### P3 — 低优先级

| # | 功能 | 迁移收益来源 | 预估提升 |
|---|------|------------|---------|
| **7** | **Surface coplanarity** | 消灭 `array_scalar()` 逐 edge-pair 的 GPU→CPU sync | **场景初始化阶段** 3-10x |
| **8** | **Edge geometry** | 一次性操作，收益有限 | 小 |

---

## 五、VJP 梯度设计

### 5.1 AD 集成策略

推荐使用 **DrJit CustomOp**：

```python
class UTDAccumulateOp(dr.CustomOp):
    def eval(self, *args):
        return native.utd_accumulate_forward(...)

    def backward(self):
        grads = native.utd_accumulate_backward(..., self.grad_out())
        for name, grad in grads.items():
            self.set_grad_in(name, grad)
```

- 自动接入 DrJit AD graph
- Forward/backward kernel 均为手写 CUDA
- Python 侧只做调度

### 5.2 各 Kernel 的 VJP 参考

| Kernel | 需梯度参数 | VJP 难度 | Slang 参考 |
|--------|-----------|---------|-----------|
| UTD mega-kernel | `edge_pos, edge_dir, source_pos, n0, nn` (几何), `incident_jones` (场) | **高** | `utd_accumulate_backward.slang` 已有完整实现 |
| Fresnel 反射系数 | `cos_theta` | 低 (闭式) | `computeFaceReflectionCoefficients` |
| Reflection EPC | `image_src, plane_n/d` | **高** (N bounce 串行反向) | `adjointVectorTransport` |
| scatter_reduce (sum) | `values` | 低 (VJP = gather) | `atomicAddComplex3` |

### 5.3 Slang → CUDA 翻译对照

| Slang 概念 | CUDA 对应 |
|-----------|----------|
| `[CUDAKernel] void func(...)` | `__global__ void func(...)` |
| `[Differentiable] float func(...)` | 手写 `__device__` forward + backward pair |
| `TensorView<float>` | `const float* __restrict__` + stride/shape 参数 |
| `bwd_diff(computePairContribution)` | 手写 backward kernel |
| `Complex` struct | `float2` 或 `struct { float re, im; }` |
| `Jones2` struct | `float4` (u.re, u.im, v.re, v.im) |
| `atomicAdd` on Complex3 | 6× `atomicAdd` (或 warp-shuffle reduce 后再 atomic) |

---

## 六、分阶段实施方案

### Phase 1: UTD Forward Mega-Kernel

**目标**：将 `_edge_state_field_to_targets()` + `scatter_reduce` 融合为单个 CUDA kernel，消除内循环中 12 次 scatter_reduce 物化点。

**输入/输出接口**：
```
Input:
  state_arrays: packed float buffer [n_states × FIELDS_PER_STATE]
  rx_pos: float3 buffer [n_rx]
  pair_indices: uint32 buffer [n_pairs] (已经过 visibility filter)
  k, wavelength: scalar float

Output:
  direct_vector_total: Complex3 buffer [n_rx]  (atomic add)
  multi_vector_total:  Complex3 buffer [n_rx]  (atomic add)
  per_edge_vector:     Complex3 buffer [n_edges × n_rx] (optional)
```

**Kernel 结构** (参考 `utd_accumulate_forward.slang`)：
```cpp
__global__ void utd_accumulate_forward(
    const float* __restrict__ state_buf,    // packed states
    const float* __restrict__ rx_pos,       // [n_rx, 3]
    const uint32_t* __restrict__ pair_idx,  // [n_pairs]
    int n_rx, int n_states, int n_pairs,
    float k, float wavelength,
    float* __restrict__ direct_out,         // [n_rx, 6] (xyz × re/im)
    float* __restrict__ multi_out,          // [n_rx, 6]
    float* __restrict__ per_edge_out        // [n_edges * n_rx, 6] (optional)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    uint32_t pi = pair_idx[tid];
    int state_idx = pi / n_rx;
    int rx_idx = pi % n_rx;

    // 1. Load state (single coalesced read from packed buffer)
    PairInputs inputs = load_state(state_buf, state_idx);
    float3 target = load_rx(rx_pos, rx_idx);

    // 2. Compute edge geometry (phi, phi', s, s')
    EdgeGeometry geom = compute_edge_geometry(inputs, target);
    if (!geometry_valid(geom, inputs)) return;

    // 3. Face reflection coefficients (if material detail)
    FaceOperators face_ops = compute_face_operators(inputs, geom, wavelength);

    // 4. UTD diffraction coefficient — 核心: 3 组 beta_groups 共享中间值
    //    _beta_term_state 的 cot/Fresnel 只算一次，R0/Rn 系数在组装时分离
    UTDResult utd = compute_utd_coefficient(geom, face_ops, k);

    // 5. Jones operator assembly + incident field application
    Complex3 pair_vector = apply_jones_and_transport(utd, inputs, geom);

    // 6. Ownership classification
    uint32_t ownership = ownership_code(inputs);

    // 7. Warp-level partial reduction + atomic scatter
    //    同一 rx_idx 的线程先 warp shuffle reduce，再一次 atomicAdd
    scatter_accumulate(pair_vector, ownership, rx_idx, direct_out, multi_out);
}
```

**`__device__` 函数层次**：
```
compute_edge_geometry()
  └── dot, cross, atan2, norm (内联)

compute_face_operators()
  ├── complex_sqrt()        ← 整合 material.py:complex_sqrt
  └── fresnel_reflection()  ← 整合 material.py:fresnel_reflection

compute_utd_coefficient()
  ├── beta_term_metadata() × 4   ← 共用: 只算一次 cot, round, phase
  ├── fresnel_integral_with_derivatives() × 4
  │     └── boersma_polynomial()  ← 两分支，无 warp divergence (都算，select 输出)
  ├── assemble_beta_term_state() × 4
  └── group_assembly()            ← 3 组 (direct, face0, face1) 共享 dif/sum terms

apply_jones_and_transport()
  ├── jones_from_vector()
  ├── apply_jones_operator() × 2  (field + slope)
  ├── jones_scale()
  └── vector_from_jones()

scatter_accumulate()
  ├── __shfl_down_sync() for warp reduction
  └── atomicAdd() × 6 (or × 12 with per_edge)
```

**新增文件**：
```
witwin/channel/_native/
├── shared/
│   ├── utd_types.h           # PairInputs, EdgeGeometry, Complex, Jones 结构体
│   ├── utd_math.h            # Boersma, cot, f_utd device 函数
│   └── utd_accumulate.h      # forward/backward kernel 声明
└── src/
    ├── utd_math.cu           # Fresnel 积分、beta groups device 函数实现
    └── utd_accumulate.cu     # forward + backward global kernel
```

**Python 侧变更**：
- `field.py:_accumulate_edge_states_to_receivers()` 中，检测 `native_extension_available()` 后走 native path
- 将 state_arrays dict pack 为连续 buffer（或在 state 构建阶段就使用 packed layout）
- 保留 DrJit fallback path 用于调试和验证

**验证计划**：
1. 逐函数单元测试：`fresnel_integral`, `diffraction_coefficient`, `compute_edge_geometry` 与 Python 版对比
2. 端到端对比：整个 `_accumulate_edge_states_to_receivers` 的 native vs DrJit 输出
3. 梯度验证：有限差分 vs backward kernel
4. 性能基线：在标准测试场景上对比 native vs DrJit 的 wall-clock time

---

### Phase 2: UTD Backward Mega-Kernel

**目标**：手写 reverse-mode kernel，与 Phase 1 的 forward kernel 配对。

**依赖**：Phase 1 forward kernel 完成且验证通过。

**Kernel 结构** (参考 `utd_accumulate_backward.slang`)：
```cpp
__global__ void utd_accumulate_backward(
    // forward inputs (same as forward kernel)
    const float* __restrict__ state_buf,
    const float* __restrict__ rx_pos,
    const uint32_t* __restrict__ pair_idx,
    int n_rx, int n_states, int n_pairs,
    float k, float wavelength,
    // upstream gradients
    const float* __restrict__ grad_direct_out,  // [n_rx, 6]
    const float* __restrict__ grad_multi_out,   // [n_rx, 6]
    // output gradients (atomically accumulated)
    float* __restrict__ grad_state_buf,          // [n_states × GRAD_FIELDS]
    float* __restrict__ grad_rx_pos              // [n_rx, 3]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    // 1. Reload forward context (same load as forward)
    // 2. Recompute intermediate values (或从 saved buffer 读取)
    // 3. Gather upstream grad for this pair's rx_idx
    // 4. Reverse-mode chain: scatter_grad → jones → utd → geometry
    // 5. Atomic accumulate gradients to state_buf and rx_pos
}
```

**关键设计决策**：
- **Recompute vs Save**：forward kernel 的中间值（EdgeGeometry, FaceOperators, UTDResult）是 recompute 还是存到 buffer？Recompute 省内存但多算一次；Save 省计算但需要 `n_pairs × ~200 bytes` 的中间 buffer。**推荐 recompute**（中间值计算量约 2000 FLOPs，对 GPU 来说廉价，且避免了大 buffer 的内存带宽开销）。

**DrJit AD 集成**：
```python
class UTDAccumulateOp(dr.CustomOp):
    def eval(self, *inputs):
        # Pack state_arrays → contiguous buffer
        packed = pack_state_arrays(state_arrays)
        # Call forward kernel
        direct, multi = native.utd_accumulate_forward(packed, rx_pos, ...)
        # Save for backward
        self._packed = packed
        self._pair_idx = pair_idx
        return direct, multi

    def backward(self):
        grad_direct = self.grad_out(0)
        grad_multi = self.grad_out(1)
        grad_packed, grad_rx = native.utd_accumulate_backward(
            self._packed, ..., grad_direct, grad_multi
        )
        # Unpack gradients back to individual state fields
        unpack_and_set_grads(self, grad_packed)
```

**新增文件**：
```
witwin/channel/deterministic/kernels/utd/
    └── utd_accumulate.cu    # 增加 backward kernel (与 forward 同文件)
witwin/channel/trace/diffraction/
    └── native_field.py      # CustomOp wrapper + pack/unpack
```

**验证计划**：
1. 有限差分验证所有需梯度参数的 backward 正确性
2. 端到端优化测试：确认通过 native backward 的优化收敛行为与 DrJit AD path 一致

---

### Phase 3: Reflection Accumulation Kernel

**目标**：将 `accumulate_reflection_paths_to_receivers()` 的 replay + field + scatter 循环融合。

**依赖**：独立于 Phase 1/2，可并行开发。

**核心挑战**：`epc_reflection_chain_to_target` 中的 `scene.ray_test()` (visibility check) 依赖 Mitsuba/DrJit 的 BVH — 这部分无法直接在手写 CUDA kernel 中调用。

**方案选择**：

**方案 A: 分离 visibility，融合其余**
```
Step 1 (DrJit): batch visibility test for all (path, rx) pairs
Step 2 (CUDA kernel): replay chain + Fresnel + Jones + field + scatter_reduce
```
- visibility 仍走 DrJit `scene.ray_test()`，但只做一次 batch 调用
- 后续的 replay chain（3 个 Python 循环）全部融合到 CUDA kernel

**方案 B: 预计算 hit 几何，传入 kernel**
```
Step 1 (DrJit): 对每条 path 预计算 hit_points, normals, prim_idx (chain_depth 次 ray_test)
Step 2 (CUDA kernel): 接收预计算几何 → Fresnel + Jones chain + field + scatter
```
- 将 visibility 产生的几何数据打包传入 kernel
- kernel 内只做 Fresnel/Jones 链式计算和 scatter

**推荐方案 A**（减少数据拷贝，visibility 开销相对较小）。

**Kernel 结构**：
```cpp
__global__ void reflection_accumulate(
    const float* __restrict__ path_buf,     // packed path data per bounce
    const float* __restrict__ rx_pos,
    const uint8_t* __restrict__ valid_mask,  // from batch visibility
    int n_paths, int n_rx, int chain_depth,
    float k, float wavelength,
    float* __restrict__ total_vector         // [n_rx, 6] atomic add
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= n_paths * n_rx) return;
    if (!valid_mask[pair_idx]) return;

    int path_idx = pair_idx / n_rx;
    int rx_idx = pair_idx % n_rx;

    // Unroll chain replay
    Jones chain = jones_identity();
    for (int slot = chain_depth - 1; slot >= 0; slot--) {
        // plane intersection, reflect, Fresnel, Jones rotate
        chain = jones_multiply(bounce_operator, chain);
    }

    // Point-source field + Jones application
    Complex3 field = apply_chain_field(chain, path_buf, rx_pos, rx_idx, k);

    // Atomic scatter
    atomic_add_complex3(total_vector + rx_idx * 6, field);
}
```

**新增文件**：
```
witwin/channel/_native/
├── shared/
│   └── reflection_accumulate.h
└── src/
    └── reflection_accumulate.cu
witwin/channel/trace/reflection/
    └── native_accumulation.py   # Python 调度层
```

**验证**：与现有 DrJit path 逐 bounce 对比 Jones chain 输出。

---

### Phase 4: Packed State Buffer

**目标**：将 53-field SoA Python dict 替换为连续内存 packed buffer，使 gather/concat 变为 memory 操作。

**依赖**：Phase 1 的 kernel 已经需要 pack/unpack，这里是将 pack 格式固化为全局标准。

**设计**：

```
State Buffer Layout (per state, 连续内存):
  offset 0:    edge_idx          (uint32)
  offset 4:    edge_pos          (float3 = 12 bytes)
  offset 16:   edge_dir          (float3)
  offset 28:   source_pos        (float3)
  offset 40:   n0                (float3)
  offset 52:   n_face_n          (float3)
  offset 64:   wedge_n           (float)
  offset 68:   adjacent_face0    (int32)
  offset 72:   adjacent_face1    (int32)
  offset 76:   incident_jones_u  (complex = 8 bytes)
  offset 84:   incident_jones_v  (complex)
  ...
  offset ~420: history slots     (variable)

Total: STRIDE bytes per state (固定, pad to 16-byte alignment)
```

**Native API**：
```cpp
// 单次 gather: 从 packed buffer 取 N 个 state
void gather_packed_states(
    const char* __restrict__ src,    // [n_total × STRIDE]
    const uint32_t* __restrict__ indices,  // [n_gather]
    char* __restrict__ dst,          // [n_gather × STRIDE]
    int n_gather, int stride
);

// 多源 concat: 将 K 个 packed buffer 拼接为一个
void concat_packed_states(
    const char** __restrict__ srcs,  // K source pointers
    const int* __restrict__ lengths, // K source lengths
    char* __restrict__ dst,          // [sum(lengths) × STRIDE]
    int n_sources, int stride
);
```

**Python 侧**：
- `_make_state_arrays()` 改为直接写入 packed buffer
- `_gather_state_arrays()` / `_concat_state_arrays()` / `_subset_state_arrays()` 委托到 native
- 保留 dict 接口用于调试读取（lazy unpack on access）

**验证**：pack → unpack 的 round-trip 正确性；所有现有测试通过。

---

### Phase 5: Higher-Order Filter + Pruning

**目标**：消除 `higher.py` chunk 循环中的 `dr.compress()` + `dr.width()` sync 链；用 GPU-native sort 替代 lexsort round trip。

**依赖**：Phase 4 packed buffer（filter 输出直接写入 packed format）。

**5a: Fused Cartesian Filter**

```cpp
__global__ void cartesian_filter(
    const char* __restrict__ prev_states,   // packed [n_prev × STRIDE]
    const float* __restrict__ edge_data,    // edge positions/dirs [n_edges × ...]
    int n_prev, int n_edges,
    // output: compacted valid pairs
    uint32_t* __restrict__ out_prev_idx,
    uint32_t* __restrict__ out_edge_idx,
    uint32_t* __restrict__ out_count         // atomic counter
) {
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= n_prev * n_edges) return;

    int prev_idx = pair_id / n_edges;
    int edge_idx = pair_id % n_edges;

    // Fused checks: distinct edge + exterior + power threshold
    if (!is_valid_pair(prev_states, prev_idx, edge_data, edge_idx)) return;

    // Append to output (atomic increment)
    uint32_t slot = atomicAdd(out_count, 1);
    out_prev_idx[slot] = prev_idx;
    out_edge_idx[slot] = edge_idx;
}
```

- 不再需要多轮 compress + width check
- 输出的 `out_count` 只需最后读一次 → **1 次 GPU→CPU sync** 替代原来的 5+ 次

**5b: GPU Radix Sort for Pruning**

```cpp
// 使用 CUB DeviceRadixSort
void prune_states_by_budget(
    const char* __restrict__ states,     // packed
    int n_states, int budget,
    // composite sort key: [-power_bits | order | depths | edge_history]
    uint64_t* __restrict__ sort_keys,
    uint32_t* __restrict__ sort_indices,
    char* __restrict__ pruned_out         // packed [budget × STRIDE]
);
```

- 全程 GPU，无 DrJit→Torch→DrJit round trip
- CUB sort 已高度优化，远快于 Python 层的 torch.lexsort

**新增文件**：
```
witwin/channel/deterministic/kernels/
    ├── cartesian_filter.cu
    └── pruning_sort.cu     # 依赖 CUB (CUDA Toolkit 自带)
```

---

### Phase 6: 收尾优化

**6a: Surface Coplanarity 向量化** (P3 优先级)

将 `compile.py` 中逐 edge 的 `array_scalar()` 替换为单次向量化比较：

```cpp
__global__ void coplanarity_check(
    const float* __restrict__ face_normals, // [n_faces, 3]
    const uint32_t* __restrict__ edge_faces, // [n_edges, 2] (adjacent face pairs)
    float threshold,
    bool* __restrict__ is_coplanar          // [n_edges]
);
```

Python 侧读回 `is_coplanar` bool 数组，再做 CPU union-find。

**6b: Edge Geometry Batch** (低优先)

将 `compute_edge_geometry` 从逐 edge Python 循环改为 batch kernel。

---

## 七、时间线估算与依赖关系

```
                        ┌─────────────────┐
                        │  Phase 1        │
                        │  UTD Forward    │
                        │  Mega-Kernel    │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐     ┌─────────────────┐
                        │  Phase 2        │     │  Phase 3        │
                        │  UTD Backward   │     │  Reflection     │
                        │  + CustomOp     │     │  Accumulation   │
                        └────────┬────────┘     │  (independent)  │
                                 │              └────────┬────────┘
                        ┌────────▼────────┐              │
                        │  Phase 4        │◄─────────────┘
                        │  Packed State   │
                        │  Buffer         │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Phase 5        │
                        │  Filter + Sort  │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Phase 6        │
                        │  收尾优化       │
                        └─────────────────┘
```

Phase 3 (Reflection) 与 Phase 1/2 (UTD) **完全独立**，可并行开发。Phase 4 依赖 Phase 1/2 的 pack format 设计稳定。Phase 5/6 依赖 Phase 4 的 packed buffer 接口。
