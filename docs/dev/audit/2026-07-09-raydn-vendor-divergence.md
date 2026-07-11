# RayDN vendor 分叉评估：哪些回上游，哪些留 channel

**基线**：`ext/raydn` 导入自 commit `bf5e574`（"Add RayDN vendor snapshot"），上游为 `E:\Code\RayDTorch`。
**范围**：`git diff bf5e574 HEAD -- ext/raydn`，28 个文件，+2654 / −233 行。
**日期**：2026-07-09

---

## 结论

这些改动**不是**一组同质的通用正确性改进。按去向拆开：

| 去向 | 内容 | 规模 |
|---|---|---|
| **一、回上游 RayDN** | 上游自身的物理 bug、数值精度、未接线的能力 | ~55% |
| **二、留在 channel_native** | 应用层物理约定、专用算子、集成层 | ~40% |
| **三、待定 / 自身技术债** | 需先查清或清理 | ~5% |

核心结构问题：**这两类改动目前全部混在 `ext/raydn/` 这棵 vendor 树里**，
上游快照之后又自行前进了（`8e27838` Autograd dispatch keys + torch.compile、`f96273b` 热路径优化），
再想同步已无干净的合并基。迁移建议见第四节。

---

# 一、应回合上游 RayDN

## 1.1 可直接回合（纯修复，无需 API 讨论）

### U1 · Keller 锥半角用错入射方向
`diffraction/accum_optix.cu` · `keller_grid_hit()`

上游用 per-state 锚点的 `state_wi_at(state_idx)` 定锥角。长边的远端会射出偏离 Keller 锥的方向。
改为随采样边点变化的真实入射 `edge_point - state_src_at(state_idx)`。

### U2 · 多边场景的边测度权重系统性低估 S 倍
`diffraction/accum_optix.cu` · `diffraction_weight()`

上游用 `edge_length / N`。但 lane 按 `lane % state_count` 轮转分配到 S 个 state，
每个 state 只拿到 `N/S` 条 lane，每 lane 的边测度应为 `edge_length · S / N`。
任何 `S > 1` 的场景都低估 S 倍。

### U3 · 漏掉 `(λ/4π)²` 波长增益
`diffraction/accum_optix.cu` · `diffraction_weight()`、`diffraction/accum_ad.cu` · `common_no_src`

### U4 · 相位 float32 精度（审计 DF-3 / D-7）
新增 `utd_types.h` · `cplx_exp_neg_kd(k, d)`：`k·d` 在 double 下 `fmod 2π` 后再降 float 做 sincos。
f32 直乘丢失约 `k·d·2⁻²⁴` 的相位，mmWave 距离下相干求和会散。

**回合时必须一并修 `reflection/accum_optix.cu:444`** —— 见 [D1](#d1--df-3-相位修复漏了反射累加主内核)。

### U5 · 表面自交偏移在掠射角下失效
`reflection/accum_optix.cu`

`origin = hit_point + kRayBias * direction` → `offset_surface_point()`：
沿几何法线、按点坐标幅值做尺度自适应偏移。原式的偏移在法向上的投影随掠射角趋近 0。

### U6 · OptiX params buffer 过小时抛异常
`common/optix_pipeline.cpp` · `launch_impl()`

改为重新分配。代价是掩盖调用方的尺寸错误，可加一条 warning。

### U7 · 构建与卫生
- `CMakeLists.txt`：`torch_python` 增加 `${TORCH_PACKAGE_DIR}/lib` 搜索路径。
- `utd_math.h` 去 UTF-8 BOM；`utd_types.h` 修复 mojibake 注释。

---

## 1.2 需先参数化，再回合

这几条的**判据是通用的，但当前实现把 channel_native 的取值焊死了**。
回合前必须把常量提为参数；取值本身留在 channel（见 [C1](#c1--发射极化与工作频率)、[C4](#c4--共面容差取值)）。

### U8 · 反射累加的材质是伪造的
`reflection/ops.cpp` · `reflection_accumulation_forward_op()`

上游在函数体内直接 `at::ones` / `at::zeros` 构造材质，并把 `params.solid_angle_per_ray` 写死为 `1.0f`。
也就是说上游这个算子输出的功率图**在物理上没有意义**。

本地改为由调用方传入 `material_eta_r / sigma / mu_r / gain / valid` 与 `solid_angle_per_ray`，
并校验长度等于三角形数。这是纯上游 bug —— 但它破坏 schema（+5 个张量 +1 个 float），
需要作为 breaking change 提。

### U9 · EPC 点在三角形内判据缺出平面检查
`reflection/epc_optix.cu` · `point_inside_triangle()`

上游只做重心坐标测试，任何**投影**落进三角形的点都被接受 —— 平行的另一堵墙会被误判为命中。
本地增加相对出平面容差。

**判据上游，容差 `kEpcPlaneTolerance = 1e-3` 需提为参数**：它的注释明确写着依赖
channel_native 的 plane-group 量化尺度。

### U10 · 一阶衍射不是 UTD（审计 DF-1）
`diffraction/paths_optix.cu` · `trace_paths_order1_impl()`

上游取**边中点**作为绕射点，用一个各向同性标量 `path_weight` 当幅度，只写 x 分量场。改为：

- `utd::first_order_diffraction_parameter()` 求 Keller 驻点；时延、可见性、导出交点都跟着驻点走；
- `utd::compute_pair_contribution()` 得到矢量场三分量，按 `sqrt(P_tx)` 缩放写出 x/y/z。

配套给 `DfrPathParams` 加 `material_eta_r / sigma / mu_r`，才能构造 per-face Fresnel 算子。

**回合前必须把 `paths_material_params()` 里硬编码的发射极化与 `omega` 提为参数** —— 见 [C1](#c1--发射极化与工作频率)。

### U11 · Keller 采样缺 pdf 补偿（审计 DF-6）
`diffraction/accum_optix.cu` · `keller_grid_hit_from_incident()`

`(edge_t, φ)` 样本映射到测量平面的面元是 `J = |∂x/∂t × ∂x/∂φ|`，
沉积权重应携带 `2π·J` 而非固定 cell 面积。新增 `measure_scale`。

回合时需在文档中声明其受控近似：**忽略了锥轴沿边的变化**（源远离边时缓变）。

---

## 1.3 上游已有能力，本地只是接线

### U12 · wedge 事件收集
上游 `accum_params.h` 已有 20 个 `out_wedge_*` 字段，`accum_optix.cu` 已有 `store_wedge_event()`，
但 `ops.cpp` 把 `collect_wedges = 0; wedge_capacity = 0;` 硬接成关闭，且从不分配/返回缓冲。

本地把它接通（输出张量从 8 个变 18 个）。**这不是 channel 私货，是上游自己没写完的功能。**

### U13 · `scene_edge_records()`
一次性导出 12 个边表张量（顶点、面、面法线、边端点、全局面索引、边所属 shape/local id、对边）。
通用 scene introspection。

### U14 · 采样调度与策略的可选化
- `sample_state_index` / `sample_edge_weight` 两个可选张量：让调用方显式指定每条 lane 采哪个
  state 及其边测度，取代 `lane % state_count` 轮转。默认行为不变，为重要性采样留口。
- 无 suffix 时走 fused raygen（launch 3），省一次 launch。
- `ReflAccumStrategy` 枚举（auto / atomic / staged / compact / streaming_planar）：把原本写死的
  启发式变成可选策略。**枚举框架上游；`STREAMING_PLANAR` 这个具体策略留 channel**（见 [C2](#c2--streaming_planar-程序化射线与天线模型)）。
- 纯功率快路径：`out_field_x_re == nullptr` 时跳过相干场，只原子加功率。

---

## 1.4 上游基础设施，但当前惰性 —— 建议连同"激活"一起提

### U15 · astigmatic 边缘焦散 / exact-direct 拆分
`utd_math.h` · `edge_caustic_rho()`、`pair_state_at_stationary_point()`；`utd_types.h` · `PairInputs`

看起来是最深的物理修复：拆开"精确直接重算"与"沿边移动驻点后按球面波重标定"
（`incidentScale = exp(-jkΔd)·d_anchor/d_new`），并用边缘焦散半径 `ρ` 取代一律的球面假设
（`ls = sqrt(ρ/(s(s+ρ)))`，VJP 的解析导数同步改为对 ρ 求导）。

**但它在当前构建中不改变任何一个数字。** `pathLengthPrefix` 全仓库只有两处赋值，都等于
`|anchor − source|`：

- `paths_optix.cu:321` → `norm3(edge_pos - source)`
- `accum_optix.cu:501` → `safe_length(edgePos - sourcePos)`

于是 `prefixExtra = max(pathLengthPrefix − dAnchor, 0) ≡ 0`，`rhoPar ≡ sPrime`，`rhoExtra ≡ 0`，
扩散因子与所有 truncation factor 退化回原式。同理 `directFirstOrder` 两处赋值都令其等于
`selectStationaryPoint`，故 `exactDirect ≡ selectedStationary`、`incidentScale ≡ 1`，
`endpointContinuation` 判据也与原版等价。

`accum_optix.cu:497` 的注释自己承认了：

> The 84-slot table predates the reference header's exact-direct / edge-caustic fields;
> preserve the previous semantics **until the slot layout carries them explicitly**.

**处理**：接线完整、退化安全（无回归），是为级联衍射预留的正确基础设施。
但单独提上游没有价值 —— 应连同 **84-slot state 表的扩展**（携带真实 `pathLengthPrefix` 与
`directFirstOrder`）一起提，否则上游拿到的也是死代码。

在此之前，**不得把它计入"已生效的正确性改进"**，也应在 `FEATURE_LIST.md` 标注为未激活，
避免后续误以为级联衍射已具备焦散修正。

---

# 二、应保留在 channel_native

这些是应用层的约定与算子。它们不该活在 `ext/raydn/` 里 —— 目标是迁到 `native/channel_native/`，
或（对 U8–U11 那几条）以**参数传入**的方式留在 Python/C++ 调用侧。

## C1 · 发射极化与工作频率
`diffraction/paths_optix.cu` · `paths_material_params()`

```cpp
mat.txPolX = 1.f; mat.txPolY = 0.f; mat.txPolZ = 0.f;  // "deterministic solver convention"
mat.useFresnel = 1;
mat.omega = params.k * kSpeedOfLight;
```

全局 x̂ 极化是 channel_native deterministic solver 的约定，不是 RayDN 的通用契约。

**去向**：`DfrPathParams` 增加 `tx_pol` 与 `omega` 字段，值由 channel 侧填。

## C2 · `streaming_planar`：程序化射线与天线模型
`reflection/accum_optix.cu` · `__raygen__reflection_accumulation()`

raygen 内用 Fibonacci 球面直接生成射线（`fibonacci_sphere_direction`），
并对这些射线套用 `vertical_iso_polarization()` —— 一个硬编码的垂直极化各向同性天线。
TX 从 `tx[0]` 广播。同时新增 `procedural_rays` / `los_enabled` 两个 `AccumParams` 字段。

性能收益真实（不落地 ray buffer，适合大稀疏网格），但这是把**某个特定天线模型 + 某个特定采样模式**
焊死在通用反射内核里。

**去向**：留 channel。若要上游化，应改为回调式的 ray generator 或让调用方传入方向缓冲。

## C3 · silhouette 边选择
`diffraction/builder.cu`（新增 390 行）、`diffraction_discover_edges` / `_counted` 算子

把原先 Python 侧的边选择启发式搬到 GPU。带场景尺度魔数：

```cpp
constexpr float kHalfPiMinusOffset = 1.52079632679f;   // π/2 − 0.05
return add(hit_p, mul(d, 0.1f));                        // 绝对 0.1 米偏移
```

`0.1f` 是绝对长度，换个场景尺度就失效。这不是通用几何算子。

**去向**：留 channel；`0.1f` 应改为按包围盒对角线的相对量。

## C4 · 共面容差取值
`reflection/epc_optix.cu` · `kEpcPlaneTolerance = 1e-3f`

判据上游（[U9](#u9--epc-点在三角形内判据缺出平面检查)），**取值**依赖 channel 的 plane-group 量化尺度。

**去向**：作为 `ReflEpcParams` 的一个字段传入。

## C5 · `reflection_epc_paths_forward`
`reflection/ops.cpp`（+~300 行）

镜像法路径求解 + surface group 去重 + `visibility_ignore_mode`。
`surface_group_id / _size / _members` 这些概念上游不存在。

**去向**：留 channel。

## C6 · C ABI 直调层
`native_api.h`（新增，12 个 `extern "C"` 入口）、`module.cpp` · `native_module_handle()`、
`CMakeLists.txt` 的 `raydn_native_core` OBJECT 库拆分

`native/channel_native/raydn_bridge.cpp` 用 `GetProcAddress`/`dlsym` 拿函数指针直调，
绕开 Torch dispatcher、pybind 往返与 GIL。收益真实。

但**这不是 C ABI** —— 它跨 `extern "C"` 边界传递 `const at::Tensor *`，
要求 `_raydn.pyd` 与 `channel_native.pyd` 用完全相同的 libtorch 与 C++ ABI 编译。
这是一个耦合，不是正确性改进。

**去向**：留 channel。若要长期维护，要么退回 dispatcher（承担 GIL 代价），
要么设计真正的 POD C ABI（传裸指针 + shape/stride/dtype）。

## C7 · `diffraction_weight` 的半 cell 距离 clamp（已知偏差）
`diffraction/accum_optix.cu`

```cpp
const float half_cell = 0.5f * sqrtf(fmaxf(params.grid_cell_area, 0.f));
const float target_distance = fmaxf(norm3(target - edge_point), fmaxf(kDfrEps, half_cell));
```

注释坦承这是为了让边穿过接收平面时 `1/d²` 的方差可积。
**这是有意引入的偏差，不是正确性修复**：能量被抹平到沉积 cell 内，且偏差量随网格分辨率变化 ——
加密网格不会收敛到无偏解。

**去向**：留 channel，并在 `FEATURE_LIST.md` 记为已知偏差。
真正的解法是按 cell 立体角做解析积分，那可以上游。

---

# 三、待定 / 自身技术债

### D1 · DF-3 相位修复漏了反射累加主内核
`reflection/accum_optix.cu:444`

```cpp
const Complex phase = c_exp_neg_i(wave_k * unfolded_distance);   // 仍是纯 f32
```

`epc_field.cu`、衍射相干累加、`paths_optix` 的 target_export 都做了 double `fmod` 归约，
唯独漏了这里 —— 而它是 radiomap 的主内核。**这是真 bug，属 [U4](#u4--相位-float32-精度审计-df-3--d-7) 的一部分，应一起修一起上游。**

### D2 · `auto_staged` 新增的守卫未加解释
`reflection/ops.cpp:1491`

```cpp
max_bounces_i <= 1 && ray_count <= 10000000
```

若 staged 路径在 `max_bounces >= 2` 时只是慢，这是性能调优（留 channel）；
若是**结果不对**，那就是掩盖了一个 staged 累加 bug（应上游）。当前没有注释或测试区分这两种。
**需先查清再定去向。**

### D3 · `mp_int` 是死变量
`reflection/accum_optix.cu:608`

```cpp
const bool mp_int = accumulate_plane(...);   // 从未被读取
```

同时 `accumulate_plane_hit()` 在 `depth <= 0 && los_enabled == 0` 时的返回值语义
从上游的 `false` 改成了 `true`（"平面已到达"），但既然返回值无人使用，该语义变更未被任何东西验证。
**要么用起来，要么改回 `void`。** 属本地引入的债。

---

# 四、迁移建议

当前所有改动都直接落在 `ext/raydn/` 里，vendor 树已被污染，无法再从上游 rebase。建议：

1. **拉出上游 patch 系列**：把第一节的 U1–U14 整理成独立提交，提回 `E:\Code\RayDTorch`。
   其中 U8/U9/U10/U11 需先做参数化。
2. **把第二节迁出 vendor 树**：C1/C4 变成参数；C2/C3/C5/C6/C7 的代码移到
   `native/channel_native/`（builder.cu、epc_paths、raydn_bridge 已有天然归属）。
3. **先修 D1，先查 D2，先清 D3。**
4. **U15 暂缓**，直到 84-slot 表能承载 `pathLengthPrefix` / `directFirstOrder`；
   在此之前在 `FEATURE_LIST.md` 标注"级联衍射焦散修正未激活"。
5. 完成 1–2 后，`ext/raydn` 应能恢复为一个可重新导入的干净快照，
   届时再把上游的 `8e27838`（Autograd dispatch keys + torch.compile）与
   `f96273b`（热路径优化）同步进来。

顺带：`ext/raydtorch/` 是上游 `c16cd1d` 改名前的旧快照，
没有任何构建脚本或源码引用它，可删。
