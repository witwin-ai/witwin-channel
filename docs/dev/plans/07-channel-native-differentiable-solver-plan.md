# Channel Native 可微分求解器计划

**状态：** Proposed / 独立于前向仿真器发布
**基线日期：** 2026-07-12（v2：补齐 AD 架构与 RayD 集成边界）

## 1. 目标

在不污染 Channel Native 前向热路径和公共 API 的前提下，为**固定拓扑**的场计算提供可验证的 JVP、VJP 和 PyTorch autograd 能力。

本计划独立于 [06-channel-native-full-replacement-gap-plan.md](./06-channel-native-full-replacement-gap-plan.md)。前向仿真器不需要等待 AD 完成，也不应为了未来 AD 保留重型 tape、metadata 或兼容结构。

## 2. 当前状态（已核对代码）

- 四类公开 solver（`path` / `deterministic` / `montecarlo.basic` / `montecarlo.bdpt`）均以 `solve(scene, config)` 函数形式暴露；`Config.ad_mode` 只接受 `"none"`（`core/components.py:13` 的 `NO_AD_MODES`），`__post_init__` 拒绝其他值，`path`/`basic` 还在 solve 时二次拒绝；
- capability 全局与逐 solver 均声明 `supports_ad=False`、`ad_modes=["none"]`（`capabilities.py:39-50,93,103,112,129`）；
- 底层存在**两个实验性 LoS-only 解析导数 CUDA 原语**：`mc_los_path_gain_jvp`（`kernels/los.cu:128`）与 `mc_los_path_gain_backward`（`los.cu:81`），仅经 `ops.py` facade（`ops.py:4686`/`4618`）暴露，**无任何 solver 调用**，未接入 torch autograd；`test_ad_contract.py:143` 甚至断言 `native_public_solver_ad_callers == 0`；
- `psdr/solver.py` 是 `NotImplementedError` 桩，`__init__.py` 注释"预留给未来可微实验"；
- 尚无 solver 级 JVP/VJP、torch autograd registration 或端到端梯度；topology/visibility discontinuity 尚未定义梯度估计策略。

**关键新发现（决定架构）：**

1. **RayD（源码链接进 `_channel_native`）本身已有完整的 PyTorch-native JVP + VJP**。`rayd.torch` 用真正的 `torch.autograd.Function`（`backends/torch/python/rayd/torch/autograd.py`）实现 `intersect` / `trace_reflections` / `trace_refl_epc_field` / diffraction 的前向与反向，遵循**"fixed-winner gradient contract"**：离散命中图元、反射链、边、可见性判定被冻结并 detach，梯度只流经从命中图元重算的连续几何量（`t`、命中点 `p`、法线、路径长度、image sources、EPC 复场 `field_real/field_imag`），可微输入为 mesh 顶点、ray 原点/方向、source/receiver 位置。这与本计划 §3"不把不连续伪装成连续梯度"完全一致。
2. **但 channel_native 目前拿不到 RayD 的 autograd**。CMake 以 `RAYD_TORCH_BUILD_PYTHON_MODULE=OFF` 只链接 `rayd_torch_native_core`，`raydn_bridge.cpp` 调用的是 RayD 的 detached `rayd_torch_native_*_forward` C-ABI，返回的几何张量（`epc[4]` 命中点、`epc[5]` 法线、reflection 命中链）都是**已 detach 的普通张量**。RayD 的可微入口尚未跨过 C-ABI 边界。
3. **channel_native 的场求值 `ops.field_*` 是不透明 CUDA 核**。`field_free_space` / `field_reflection_sequence` / `field_transmission_sequence` / `field_coupled_rd` / `deterministic_reflection_field`（`ops.py:2292/2380/2466/2735/5582`）都通过 `_required_native_op(...)` 调原生核，torch 张量进出但**未注册 backward**，autograd 无法穿过。
4. **固定拓扑 AD 的挂载缝**是 `path_topology.py:_evaluate_shared_fields`（≈`:723`）：RayD 返回"拓扑（整数 id）+ 几何（positions/normals/path_length，float 张量）"，channel_native 在这里把几何 + 材料 + 频率喂给 `ops.field_*` 得到场。这个函数被 `deterministic` 与 `path` 共享，所以**在此处一次做对，多个 solver 同时获得 AD**。绕射是例外：UTD 复场在 RayD 内部算出（`rayd_torch_native_diffraction_paths_order1_forward` 返回 `x_re/x_im/...`）。

因此当前状态是：底层求解器已经是 torch 张量流、且 RayD 已具备可微能力，但两侧的可微入口都没打通到公开 solver。

## 3. AD 架构（本计划的核心决策）

固定拓扑 AD 分成两个正交层，通过标准 torch 计算图组合，避免复制前向物理：

### 层 1 — 几何 AD（委托给 RayD）

对"移动 tx/rx 或 mesh 顶点会使命中点移动"的参数，梯度来自命中几何对参数的雅可比。**不在 channel_native 里重推，直接源码链接调用 RayD 的 fixed-winner JVP/VJP 原生核**（不经 `rayd.torch` Python 层）：固定 RayD 发现的 `primitive_sequence`/`edge_id`/plane 组，让 RayD 的 backward/jvp 核返回可微的 `interaction_positions`/`interaction_normals`/`path_length`/`image_sources`。

前置工作（AD-A0）：把 RayD 的可微原生实现跨过 C-ABI 边界。**RayD 是源码编译进 `_channel_native` 的，因此直接链接调用 RayD 的原生可微函数，绝不走 `rayd.torch` 的 Python autograd 层**（那会引入第二个 Python 扩展 + torch dispatcher/ATen 派发开销，是性能瓶颈）：

- RayD 的可微逻辑本体在其 C++/CUDA 里（`ops_intersect.cpp` 的 `intersect_backward_optional_cuda` / `intersect_jvp_cuda`，`geometry_backward.cu` 的 backward/jvp 核，以及 reflection/EPC 的对应实现）。`rayd.torch/autograd.py` 只是这些原生核的薄 Python 封装，我们要的是被它封装的那层符号，不是它本身。
- 在 `rayd/torch/integration.h` 增补 backward/jvp 变体的 `extern "C"` 入口，直接 forward 到上述 `*_cuda` 实现；`raydn_bridge.cpp` 用与现有 `rayd_torch_native_*_forward` 完全相同的平凡 resolve 方式拿到符号地址（无 dlopen、无 module handle）；`RAYD_TORCH_BUILD_PYTHON_MODULE` 保持 OFF。
- `torch.autograd.Function` 的封装写在 **channel_native 侧的 `ops.py`**（薄封装，只做 save_for_backward / 调 native backward），不落在 RayD 的 Python 层。

### 层 2 — EM 响应 AD（在 channel_native 内）

对"改变材料 `eps_r`/`sigma_e`/`thickness` 或频率**不移动命中几何**"的参数，命中点是常量，只需 `ops.field_*` 对材料/频率可微。这些是不透明 CUDA 核，选定实现路径（**这是必须先拍板的分叉**）：

- **选定：为每个 `field_*` 核补写原生 `*_backward` / `*_jvp` CUDA 核，用 `torch.autograd.Function` 薄封装。** 这是 RayD 自己的房规（native forward + 手写 CUDA VJP/JVP + 薄 autograd.Function），单一前向实现不复制。LoS 已有 `mc_los_path_gain_backward/jvp` 可直接并入 `field_free_space` 的 autograd.Function（并补上缺失的 frequency 切向，见 §5）。
- 不采用"再用纯 torch 重写一份场公式"：虽然 autograd 免费，但违反 §4"不复制前向物理"，且 Fresnel/多层散射矩阵的纯 torch 版会显著变慢。

### 两层如何组合

`_evaluate_shared_fields` 里：层 1 产出的可微 `interaction_positions/normals`（requires_grad w.r.t. tx/rx/顶点）与层 2 的可微 `field_*`（w.r.t. 材料/频率/其输入几何）在同一张 torch 图上相乘/累加，`.backward()` 或 `torch.func.jvp` 自动贯通。绕射单独处理：要么在 RayD 内提供可微 UTD 场，要么在 channel_native 用返回的 `interaction_position`/`edge_id`/材料重推 Keller 系数（AD-4 决定）。

## 4. 设计边界

- topology discovery 与 differentiable field evaluation 分离；离散 winner 冻结（沿用 RayD 的 fixed-winner 契约）；
- 第一阶段只承诺 fixed-topology 导数；visibility、遮挡切换、路径出现/消失不伪装成普通连续梯度；
- AD 计算图按需创建，`ad_mode="none"` 时不得增加常驻显存或 kernel launch，RayD 亦走其 detached `*_noad`/`*_forward` 快路径；
- 不复制一套前向物理公式：几何导数复用 RayD，EM 导数复用同一 `field_*` 前向核（只加其 backward/jvp 伴生核）；
- 不继承旧 Channel 的 Dr.Jit AD API，只设计 Native 自身最小的 torch-native 接口。

## 5. 第一版可微参数（按"命中几何是否移动"重排优先级）

**关键洞察：材料与频率不移动命中几何，只需层 2；tx/rx/顶点移动几何，需层 1+层 2。** 因此把最易、ROI 最高的材料/频率放在最前，先在无需 RayD 几何 AD 的情况下打通端到端 autograd。

1. **材料 `eps_r`、`sigma_e`、`thickness` 与 frequency**（仅层 2，命中几何为 detach 常量）——最先做；
2. **TX/RX position**（层 1+2，命中点移动，需 AD-A0 的 RayD 可微几何）；
3. antenna orientation/pattern parameters；
4. **mesh vertices**（层 1+2，同时影响交点、法线、可见性）——延后，因为它同时触及拓扑，验收标准不同。

注意现有 `cn_mc_los_path_gain_jvp_cuda` 只有 tx/power/rx 三个切向、`frequency_hz` 是不可导 `double`。优先级 #1 的 frequency 需要先给 LoS/reflection 场核补 frequency 切向。

## 6. 承载位置与三个目标 solver

本轮**同时交付 `deterministic` / `path` / `montecarlo.basic` 三个 solver 的 AD**。`montecarlo.bdpt` 的梯度**本轮搁置**（见 §7 非目标：其复相干 contribution + 离散三路事件采样最重，单列后续计划）。三者的场求值架构不同，分成两条并行轨道：

### 轨道 T1 — Deterministic + Path（共享同一挂载缝，一起做）

两者的场求值**100% 共用** `path_topology.py:_evaluate_shared_fields`（`:723`）：`path.solve()` 经 `export_topology`（`path/solver.py:204`）走同一函数，`path/result.py:from_topology_result` 只是打包 `paths.coefficient`、不做物理。因此在这一层做一次 AD，两个 solver 同时可微。挂载点是这些 python 侧 torch 张量 op：

| 交互 | op | 位置 |
|---|---|---|
| LoS | `ops.field_free_space` | `path_topology.py:758` |
| 反射 | `ops.field_reflection_sequence` | `:786` |
| 透射 | `ops.field_transmission_sequence` | `:843` |
| 绕射 | `ops.field_project_complex3`（投影 RayD 已算的 `field_xyz`） | `:882` |
| 耦合 R↔D | `ops.field_coupled_rd` | 耦合块 ~`:897` |

做法：给每个 `field_*` 前向核补原生 `*_backward`/`*_jvp` 伴生核，在 `ops.py` 用薄 `autograd.Function` 封装；移除 `path/solver.py:179` 的 AD 硬拦截。

### 轨道 T2 — MC basic（标量非相干功率图，原生累加）

`montecarlo.basic` 输出**实数功率图**（非相干、无极化、无复系数/相位链），比 T1 的复场与 BDPT 的复相干都简单。它的场求值分两部分（`montecarlo/basic/solver.py`）：

- **LoS**：`los_path_gain`（`solver.py:128`，`.backend`）——正好对应现有的解析原语 `mc_los_path_gain_jvp/backward`（`los.cu:128/81`）。这是本轮 AD 的现成起点，只需补 frequency 切向、并入 `autograd.Function`。基础脚手架已在 `basic/metadata.py:36-53`（按 `ad_mode` 计 `jvp_launch_count`/`backward_launch_count`）预留。
- **反射/绕射/透射/散射**：走 `raydn_components` 的**原生功率图累加**（`reflection_component_maps_with_wedges` / `diffraction_component_map` / `transmission_component_map` / `scattering_component_map`），最后 `mc_finalize_component_maps` 求和。这些是原生核，需要对应 backward/jvp。反射/绕射累加可**直接复用 AD-A0 打通的 RayD 可微 accumulation 入口**（RayD 有 `reflection_accumulate.cu` 的 jvp）；透射/散射功率图另配原生 backward。

采样方面 basic 也有 seed（反射/绕射/散射的 `samples`+`config.seed`，`solver.py:197/231`），但它是**连续样本位置上的非相干功率累加**，没有 BDPT 那种依赖材料/频率的离散三路事件分支，所以固定 seed 后梯度远比 BDPT 稳定，重参数化按标准连续采样处理即可。

### 其他

- **`psdr` 桩本轮删除**：AD 直接进这三个现有 solver 的 `ad_mode`，不新建可微 solver（遵循 monorepo"避免重复实现"）。
- **`montecarlo.bdpt` 与几何不连续性（silhouette/visibility/路径 birth-death）本轮都不做**，见 §7 非目标与 §10。

## 7. 实施阶段

**顺序原则**：AD-A0（RayD 可微几何前置）→ AD-1（T1 材料/频率）→ AD-2（T1 tx/rx/几何）→ AD-3（T2 MC basic）→ AD-4（多交互收口）。T1 与 T2 可在 AD-A0 之后并行推进。

### AD-A0：打通 RayD 可微几何入口（前置，T1/T2 共用）

- [ ] 在 `rayd/torch/integration.h` 增补 RayD 可微 EPC/trace/intersect 的 backward/jvp `extern "C"` 入口，直接 forward 到 RayD 现有的 `intersect_backward_optional_cuda` / `intersect_jvp_cuda` 及 reflection/EPC 对应实现；**不开 `RAYD_TORCH_BUILD_PYTHON_MODULE`、不 import `rayd.torch`**；
- [ ] `raydn_bridge.cpp` 用与现有 `rayd_torch_native_*_forward` 相同的平凡方式 resolve 这些新符号；`ops.py` 在 channel_native 侧包 `torch.autograd.Function`，返回 requires_grad 的 `interaction_positions/normals/path_length`；
- [ ] 单测：固定拓扑下，RayD 命中点对 tx 位置与顶点的 JVP/VJP 与中心有限差分一致（可拿 RayD 自带 `test_intersect_grad.py` 的精确顶点梯度做数值交叉验证，但运行时路径只走 C-ABI）。

### AD-1：T1 材料/频率 JVP+VJP（Deterministic + Path，几何 detach）

- [ ] 定义 fixed-topology field-evaluation 输入契约（positions/normals 作为可 detach 的固定几何）；
- [ ] 为 `field_free_space` / `field_reflection_sequence` / `field_transmission_sequence` / `field_coupled_rd` / `field_project_complex3` 补原生 `*_jvp` 与 `*_backward` 核（LoS 复用并合并现有 `mc_los_path_gain_jvp/backward`，补 frequency 切向），薄 `autograd.Function` 封装；
- [ ] 打通对 **材料 `eps_r/sigma_e/thickness` 与 frequency** 的 JVP+VJP（LoS + 单反射 + 单透射多层）；
- [ ] 移除 `path/solver.py:179` 拦截，`deterministic`/`path` 的 `Config` 接受 `ad_mode="jvp"|"vjp"`；
- [ ] `ad_mode="none"` 与当前前向性能、显存保持一致（不注册图，RayD 走 `*_noad`/`*_forward`）。

### AD-2：T1 tx/rx position + mesh vertex（Deterministic + Path，几何移动）

- [ ] LoS 对 **TX/RX position** 的 JVP+VJP（LoS 场核直接吃 tx/rx 张量，无需 RayD）；
- [ ] 单反射对 **TX/RX position** 的 JVP+VJP（组合 AD-A0 的可微几何 + `field_reflection_sequence`）；
- [ ] mesh vertex 的连续部分导数（复用 RayD 顶点梯度，固定 winner）；
- [ ] `path` solver 端到端 torch autograd，`ctx.save_for_backward` 只存 backward 必需张量（positions/normals/materials/winner ids）；
- [ ] 支持 batch、多 TX/RX 与 complex loss（复场 → 实标量 loss）。

### AD-3：T2 MC basic 可微（标量功率图）

- [ ] LoS 功率增益：补 frequency 切向，把 `mc_los_path_gain_jvp/backward` 并入 `autograd.Function`（对 tx/rx 位置、材料、频率）；
- [ ] 反射/绕射功率图：复用 AD-A0 打通的 RayD 可微 accumulation 入口拿 JVP/VJP；透射/散射功率图配原生 backward/jvp；
- [ ] `montecarlo.basic` 的 `Config` 接受 `ad_mode`，移除 `basic/solver.py:41` 的 `_validate_ad_config` 拦截；seed 固定使 FD±h 与 AD 复用同一采样序列；
- [ ] 标准连续采样重参数化（basic 无离散事件分支，直接可做）；跨 seed 梯度方差/CI（见 §9）。

### AD-4：多交互复场收口（三 solver）

- [ ] 多反射链式导数（`field_reflection_sequence` 深度 > 1，Jones/complex3）；
- [ ] multilayer/transmission 散射矩阵对 `eps_r/sigma/thickness/freq` 导数；
- [ ] UTD fixed-topology 导数：决定"RayD 内提供可微 UTD 场" vs "channel_native 从 `interaction_position/edge_id/材料` 重推 Keller 系数"，含 shadow-boundary 排除区；
- [ ] coupled R-D / D-R 链式导数；
- [ ] metadata 记录 `ad_status`、tape bytes、`jvp_launch_count`/`backward_launch_count`、forward/backward 时间与峰值显存（schema 已在 `metadata.py` 预留）；
- [ ] 解析小场景通过后，进入 Munich/SF 场景。

### 本轮非目标（明确推迟）

- **`montecarlo.bdpt` 梯度**：其复相干 contribution 全在单体原生核（`bdpt_connect.cu`/`bdpt_accum.cu`），且 reflect/transmit/scatter 三路事件选择的概率 `p_event/p_scatter/p_transmit` 依赖材料/频率、离散分支即随机拓扑——最重的一档。单列后续计划，本轮 `bdpt.Config` 继续拒绝 `ad_mode != "none"`。
- **几何不连续性研究**：silhouette/visibility discontinuity estimator、路径 birth/death 邻域 bias/variance、边界采样项——本轮不做。沿用 RayD fixed-winner 契约：离散命中/分支冻结，只承诺连续部分梯度。落到不支持区域时返回显式错误或诊断，绝不返回误导梯度。

## 8. API 草案约束

最终接口应保持最小化：

- `ad_mode="none" | "jvp" | "vjp"`；
- 只有完成端到端实现的 solver 才接受 `jvp/vjp`；未实现的参数组合在 launch 前显式失败；
- 导数输入与输出使用正常 PyTorch Tensor（`requires_grad` / `torch.func.jvp`），不生成独立 JSON tape；
- metadata 只报告 `ad_mode`、tape bytes 和必要计时；
- 沿用 RayD fixed-winner 契约：离散 id/mask 永远 detach，不对外承诺其梯度。

## 9. 梯度测试套件（对标原版 channel 强度）

原版 channel 有约 25 个真正的梯度验证函数（AD-vs-中心有限差分、JVP-vs-VJP 自洽、解析-vs-数值），覆盖 (solver × 参数 × 交互) 的组合，用 `xfail(strict=True)` 显式记录已知缺口。本轮 channel_native 要建到**同等或更强**的密度（因为它多了透射/多层/散射且三个 solver 都要覆盖，且 T1 是纯 torch autograd，可额外上 `gradcheck`——原版没有）。

### 9.1 验证方法（每种都要有）

- **中心有限差分（2 点）**：`(f(x+h) − f(x−h)) / (2h)`，逐 (solver × 参数 × 交互) 单元；
- **JVP-vs-VJP 自洽**：`vjp ≈ sum(jvp)`（对标原版 `test_deterministic_material_gradients.py:195`）；再加**内积对偶** `⟨J v, u⟩ ≈ ⟨v, Jᵀ u⟩`（原版缺，本轮补上）；
- **`torch.autograd.gradcheck`**：仅 T1 的 `autograd.Function`（double 参数子集，小场景）——原版 Dr.Jit 做不到，是 channel_native 的强项；
- **解析-vs-数值 前向 parity**：native 场核对 torch 参考实现，`rtol=1e-5, atol=1e-12`；
- **梯度图相似度**（可选，用于 shadow-transition 类）：`SSIM ≥ 0.99, PSNR ≥ 30dB`；
- **优化收敛烟测**：Adam 恢复 tx 位置，最终误差 < 阈值（端到端可用性证据）。

### 9.2 容差与步长（起点，进 CI）

沿用原版量级：radiomap/deterministic `rel=5e-2`；path `rel=5e-3`；一般 `abs=1e-12`，材料/系数功率 `abs=1e-10`。FD 步长按参数量纲各自标定：位置 `1e-2`、几何 `1e-3`、材料按其尺度、MC basic `5e-3`。频率单列步长。所有常量集中在一个 `tests/ad/_tolerances.py`，不逐文件复制。

### 9.3 覆盖矩阵（(solver × 参数 × 交互) 每个可行单元一个测试）

| 参数 | LoS | 单反射 | 多反射 | 透射/多层 | 绕射 UTD | 耦合 R↔D |
|---|---|---|---|---|---|---|
| `eps_r` | — | D/P/M | D/P | D/P/M | D/P/M | D/P |
| `sigma_e` | — | D/P/M | D/P | D/P/M | D/P/M | D/P |
| `thickness` | — | — | — | D/P/M | — | — |
| frequency | D/P/M | D/P/M | D/P | D/P/M | D/P/M | D/P |
| TX/RX pos | D/P/M | D/P/M | D/P | D/P/M | D/P/M | D/P |
| mesh vertex | D/P | D/P | D/P | D/P | D/P | D/P |

D=deterministic, P=path, M=montecarlo.basic。**D/P 验证复系数（含相位/时延）导数；M 只验证实数功率增益导数**（basic 非相干、无复系数）。频率是原版**未**测的维度，本轮必测（channel_native 把它作为一等可微参数）。空格=物理上不适用。M 的耦合 R↔D、多反射不在 basic 组件集内故留空。BDPT 全部推迟（§7 非目标），不进本矩阵。

### 9.4 基础设施

- 共享 FD 引擎 + AD 清理 helper（`tests/ad/_fd.py`），不逐文件复制（原版的重复是反面教材）；
- AD 专用场景 fixture：cube / open-wall-reflection / double-slit / thin-sheet-transmission / munich-slice；
- **MC basic seed 固定**：FD±h 与 AD 复用同一采样序列（`config.seed`，对标原版 `IntegratorOptions(seed=...)`）；
- **MC basic 跨 seed 梯度方差 / 置信区间**：原版**没有**，但 MC 梯度是随机的，本轮必须加——多 seed 跑同一梯度，断言方差有界、均值落在 FD 的置信区间内；
- `pytestmark = pytest.mark.gpu`；已知缺口用 `@pytest.mark.xfail(strict=True)` 记录（如绕射运动 FD parity），不静默省略；
- `ad_mode="none"` 零额外 tape 与性能回归测试（含 RayD `*_noad` 快路径），断言 `tape_bytes==0`、launch 计数不变。

## 10. 完成定义

- [ ] AD-A0 打通，RayD 可微几何在 channel_native 内经 C-ABI 可用（不走 `rayd.torch` Python 层）；
- [ ] **`deterministic` / `path` / `montecarlo.basic` 三个 solver** 均支持材料、frequency、TX/RX position 的 JVP/VJP（D/P 复系数，M 实数功率），端到端 torch autograd 可用；
- [ ] §9 覆盖矩阵每个可行单元有测试；中心 FD、JVP-vs-VJP 自洽、内积对偶、T1 `gradcheck`、前向 parity 门禁全部通过；MC basic 有 seed 固定 + 跨 seed 方差/CI 测试；
- [ ] 已知缺口（绕射运动 FD parity 等）以 `xfail(strict=True)` 显式登记；
- [ ] 时间、显存与梯度误差预算固定（数字进 CI 门禁）；
- [ ] 前向模式（`ad_mode="none"`）不承担 AD 的额外常驻成本，RayD 走 detached 快路径；
- [ ] `psdr` 桩已删除；
- [ ] **`montecarlo.bdpt` 梯度与几何不连续性**明确标注为后续计划；`bdpt.Config` 仍拒绝 `ad_mode != "none"`；落到不支持区域时显式失败或诊断；
- [ ] 通过以上条件后，才把对应 solver 的 `supports_ad` 改为 `True`（本轮不含 bdpt）。
```