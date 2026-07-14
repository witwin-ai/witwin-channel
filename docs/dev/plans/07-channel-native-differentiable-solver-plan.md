# Channel Native 可微分求解器计划

**状态：** Complete（AD-4b 收口：矩阵补齐、metadata、CI 预算、Munich 烟测、capability 翻转；bdpt 与几何不连续性按 §7 明确推迟）
**基线日期：** 2026-07-12（v2：补齐 AD 架构与 RayD 集成边界）
**执行更新：** 2026-07-14（v4：AD-4b 收口，见 §10）

## 0. 执行进度（2026-07-14）

| 阶段 | 状态 | 提交 |
|---|---|---|
| AD-A0 RayD 可微几何 C-ABI | 已交付 + 已加固 | `1396181`, `0fdfed2`（RayDi: `d13499d`, `bb9d457`） |
| AD-1 材料/频率 JVP+VJP（T1） | 已交付 + 已加固 | `f6873d8`, `bc6dd5a` |
| AD-2 TX/RX 位置 + mesh 顶点（T1） | 已交付 | `b0d5823` |
| AD-3 MC basic 功率图 | 已交付（散射图显式拒绝；绕射图由 AD-4b 交付） | `760e2f4` |
| AD-4a UTD 绕射 + 耦合 R↔D（T1） | 已交付 | `9ae109b` |
| AD-4b 收口（矩阵补齐 + basic 绕射图 + metadata + CI 预算 + Munich/优化烟测 + capability 翻转） | 已交付（本次） | — |

**AD-4b 交付摘要（2026-07-14）**：

- **basic 绕射功率图 AD**：`sionna_diffraction_tape_accumulate` 的逐 lane 行模板化到标量 `T`（float 实例即 primal、Dual 实例即精确导数），backward 为逐输入种子化对偶探针（12 个/lane 上限）、jvp 为单次对偶；可微输入 = 楔面板材 4 参数 + 频率 + tx 位置（源经 state_src 广播，源梯度原生归并到 (3,)）。伪无限 ±1e5 截断因子按 AD-4a 政策冻结（否则 1e5 杠杆把端点纹波放大成梯度垃圾——实测修复前后源梯度差 ~50 倍）。**tx 单元的 FD 判据**：全解 FD 在 h=3e-4..3e-2 全程不收敛（楔事件发现、采样带与逐 lane Keller 锥接受判据 `phi<=exterior` 都随 tx 离散翻转），有效判据是固定带 + 接受稳定 lane 聚合 + 两步长一致性过滤（剔除 UTD 过渡区 1/h 尖峰 lane），聚合 FD 与对偶 ≤5e-2。
- **mesh 顶点 × {透射, 绕射}**：透射经 `raydn_face_normals_ad` 的活法线表（已验证与 torch 参考逐值一致；FD 用非相干步长——直线穿透路径长度与顶点无关，1e-3 步长下 2 点差分落在 float32 前向噪声底）；绕射经楔核内从 winner 顶点重建边表（边锚点/方向/界、符号对齐法线、外角；冻结发现表钉住平面指派，防 RayD 绕序约定漂移），新增 12 顶点分量探针。LoS × 顶点结构性为零（图不触及叶子，测试钉死）。**耦合 × 顶点不支持**：耦合伴随把墙平面/边表当冻结 winner，实现即静默缺失梯度，故 seam 在顶点叶子参与时显式 `NotImplementedError`，测试registered `xfail(strict=True)`。
- **多反射（深度 2）solver 级**：新 `tests/ad/test_solver_multibounce_ad.py`，双平行墙（z 不对称）夹 TX/RX，损失只取 depth==2 行，eps_r/sigma_e/频率/TX/RX/顶点全部过中心 FD + JVP-VJP 对偶。
- **metadata**：`AdLaunchLedger` 上移到 `core/kernels/metadata.py`（三 solver 一个形状）；deterministic/path 经 `TopologyBatch.ad_companion_launches/ad_tape_bytes` 上报真实注册伴随数与 save_for_backward 字节；`make_metadata` 增加 `forward_time_ms`（CUDA 同步壁钟；jvp 的对偶趟在其中）与 `peak_memory_bytes`（本次 solve 抬高的进程 CUDA 高水位）。vjp 的反向趟发生在 solve 之后、solve 无法观测，反向时间/显存预算由 CI 门禁承担（见下），不伪造 metadata 字段。
- **CI 预算门禁**：`tests/ad/test_ad_budgets.py` 冻结 §9.2 全部容差/步长常量（放松即红），并对 AD 前向时间、反向趟壁钟、tape 字节与 CUDA 峰值显存设上限（中位数测量 + 量级余量，抓回归不抓调度噪声）。
- **Munich 烟测**：`tests/ad/test_munich_ad_smoke.py`，缩减 Munich（16×16、depth1、los+reflection）vjp 求解后对 eps_r 与活 TX 位置各一次 backward，断言有限、非零、无 NaN。
- **优化烟测**：`tests/ad/test_optimization_smoke.py`，Adam 经 deterministic 可微 delay（ToA 多点定位，无 λ/2 相位缠绕局部极小）从 0.5 m 级扰动恢复 TX 至 <1e-2 m。
- **矩阵自查补漏（M×透射×TX）**：任务清单之外审计发现 basic 透射图的行进起点用的是 detach 的原生 tx 表——入射余弦（进而每面墙透射率）对 TX 的连续依赖会被静默丢失。已改走活 TX 起点，并顺带修掉层栈对偶在**严格法向入射**处的 NaN：k_par=k·sinθ 参数化在 sinθ=0 处 d sinθ/d cosθ=-∞，而物理量只依赖 k_par²（切向处处有限），对偶改携带 k_par²（primal 值按原运算次序逐位保持）。新增 FD 测试覆盖该单元。
- **capability 翻转**：`path`/`deterministic`/`montecarlo_basic` `supports_ad=True`、`ad_modes=["none","jvp","vjp"]` + 逐 solver `ad_excluded`；全局 `ad_contract` 诚实声明（fixed-topology JVP/VJP：是；visibility/几何不连续估计器：否；bdpt：否；LoS 低层原语自 AD-3 起已接线，不再"experimental"）。
- 测试规模：`tests/ad` 197 通过 + 1 strict xfail；全套（除 Munich parity 与 performance）903 通过、1 跳过、1 xfail。

AD-1 加固（`bc6dd5a`）修掉两个真实的钳位边界梯度 bug：`fmaxf` 次梯度门用 `>` 导致 `sigma_e = 0`（材料默认初值）梯度恒零；更隐蔽的是 `dc_layer_one_way_phase` 的衰减幅度门用 `exponent < 0`，而 passive 分支在 `sigma_e = 0` 或 `thickness = 0` 时把 `Im(kz)·d` 落在 **-0.0** 上，于是 `amplitude · d_exponent` 被整项吞掉——在 `sigma_e = 0` 处这一项就是层传播子对 sigma 的**全部**一阶效应（该处 dkz/dsigma 纯虚），透射 sigma 梯度因此小了约 20 倍且反号。同一提交还修好了 deterministic 累加在 AD 模式下截断计算图的问题，并让频率 AD 在遇到色散材料时显式失败。

## 3.1 两处路线修正（基于上游能力核对，2026-07-14）

**（一）层 1 几何 AD 联合修改 RayD，命中点导数由 RayD 自己给出。** §3 的原则不变，但上游能力不足，需要**在 RayD 内补齐**，而不是绕开它。

现状核对：RayD 的 C-ABI 只暴露了 EPC **path_length** 的导数，**没有任何 interaction point 的导数**（其 EPC field backward 更是在微分一个与前向无关的 toy contract，已在 `0fdfed2` 中禁用）。而 Fresnel 幅度依赖入射角，入射角**不**满足驻定性（只有总路径长度按 Fermat 驻定），所以命中点导数无法回避。

**硬约束（用户，2026-07-14）：几何导数必须走 RayD；不得在 channel_native 侧用 torch 重解一遍命中几何；一切热路径必须是 CUDA kernel，不允许 torch 逐算子编排或 CPU 计算。** 曾短暂存在过一版 `core/ad_geometry.py` 的 torch image-source 重建（提交 `bc6dd5a`），它重复了 RayD 前向已经算过的数学（`reflection_geometry.h` 的 `reflect_point_across_plane` + `intersect_segment_plane`），**已按此约束移除**。

落地形态（不重复任何计算）：

- **反射**：channel_native 走 EPC 的 `direct_plane_mode`——`deterministic_reflection_epc_input_batch` 先把 winner 面序列的 `(tri_a, face_normal)` 打成平面数组，RayD 的 `reflection_epc_paths_forward` 解镜像链（`reflect_point_across_plane`）并回溯出命中点（`intersect_line_plane`）与法线（`shared/include/rayd/shared/optix/reflection_epc_device.cuh:490-640`）。新增 `rayd_torch_native_reflection_epc_paths_backward/_jvp`：在**冻结 winner**（面序列、包含性检验、遮挡射线全部 detach）下，给出 `d(interaction_positions, interaction_normals, path_length) / d(vertices, source, receiver)`。**backward 不需要 OptiX**——winner 已冻结，只剩纯几何链。
- **共享同一份数学**：把 raygen 里内联的链求解抽成 header-only device 函数，前向与伴随共用；按 RayD 自己的房规（`utd_math.h` 已有 `adj_fresnel_reflection_face` / `adj_face_reflection_operator`）在 `reflection_geometry.h` 旁补 `adj_reflect_point_across_plane` / `adj_intersect_line_plane` / `adj_face_normal`。
- **透射**：墙面交点本来就来自 RayD 的 `intersect` 前向（`bdpt_intersect_forward`，flags=7 返回 t/p/geo_n/prim），而 AD-A0 已经桥好 `intersect_backward/_jvp`。只需把 `geo_n` 补成可微输出（法线是顶点的函数），并在 AD 模式下让透射行进循环调用可微入口。
- **channel_native 侧只做调度**：bridge + 薄 `autograd.Function`（`save_for_backward` + 调原生 backward），零数学。射线方向 `normalize(target - source)` 的伴随另配一个平凡的 CUDA 核。

推论：原定的 AD-1.5（修复 RayD 上游 EPC field backward）对本计划仍**不必要**——channel_native 用的是自己的 `field_*` 场核，从不消费 RayD 的 EPC 场；要修的是 EPC 的**几何**伴随。

**（二）绕射 AD 只能由 channel_native 自己重算 UTD 楔形场。** RayD 的 C-ABI **零个**绕射 backward/jvp：其内部的 `diffraction_accumulation_*_backward_op` 是 radiomap 累加的伴随，不是 channel_native 使用的 order-1 path export 的伴随；`diffraction_paths_order1_forward` 没有任何伴随。因此 `field_project_complex3` 的 backward 只能拿到 d/d(field_vector)，链条到常量即断。

可行路线（AD-4 采用）：channel_native 自己从固定拓扑（edge_id、边几何、楔面材料）重算 UTD 场并微分它。可行性已核实：

- `field_transport.cuh` 已经 include 了 `<rayd/shared/utd/utd_math.h>`，`field_coupled_rd` 本来就在 channel_native 内重算楔形场（构造 `PairInputs` → `compute_pair_contribution`，见 `kernels/field_transport.cu:445-490`）。纯绕射（component 2）就是同一段代码去掉反射腿：入射场换成 `free_space_complex3(source, edge, k, tx_pol)`。
- **驻定点由 `compute_pair_contribution` 内部求解**（`pair.selectStationaryPoint = 1.0f`，RayD 的约定），所以楔形场对几何的依赖整块落在这个函数里——它的伴随同时给出材料、频率与几何梯度，不需要在外面再解一次 Keller 锥。
- **该 header 是 header-only device 代码，自带 pair 的伴随**（`compute_pair_vector_contribution` 的 VJP、`PairInputsGrad`、`adj_face_reflection_operator`）。因此绕射 AD **不需要动 RayDi 的 C-ABI**，直接在 channel_native 的核里调这些 device 函数即可。注意核对 `pair_vector_output_jvp_completion` 是**有限差分**实现（eps=1e-3）而非解析对偶——若走 jvp 需要评估其精度，或自己写解析切向。
- **陷阱**：RayD 的绕射路径用**半空间** Fresnel（`face_reflection_operator`，无厚度，cos 由楔角给出），而 `field_coupled_rd` 用**有限厚板** Fresnel（`slab_face_operator`，有厚度，cos 由 |dot(n,dir)| 给出）；重算必须复现 RayD 的约定（`material.omega > 0`，让 `compute_pair_contribution` 自己算面算子）才能保持前向一致。另外 `raydn_bridge.cpp:1317` 把送进 RayD 的 tx 极化**硬编码成 +z**（场景的真实极化没进绕射核），重算要么复现这一约定，要么先单独修前向——**前向 parity 门禁（重算结果 vs `topology.field_xyz`）是这些细节的唯一保险**。

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

- [x] 在 `rayd/torch/integration.h` 增补 RayD 可微 EPC/trace/intersect 的 backward/jvp `extern "C"` 入口，直接 forward 到 RayD 现有的 `intersect_backward_optional_cuda` / `intersect_jvp_cuda` 及 reflection/EPC 对应实现；**不开 `RAYD_TORCH_BUILD_PYTHON_MODULE`、不 import `rayd.torch`**；
- [x] `raydn_bridge.cpp` 用与现有 `rayd_torch_native_*_forward` 相同的平凡方式 resolve 这些新符号；`ops.py` 在 channel_native 侧包 `torch.autograd.Function`；
- [x] 单测：固定拓扑下，RayD 命中点对 tx 位置与顶点的 JVP/VJP 与中心有限差分一致（`tests/ad/test_raydn_geometry_ad.py`，25 项）。
- 交付偏差：RayD 的 EPC 入口只给出可微 `path_length`，**不给 interaction point 导数**，其 EPC field backward 微分的是 toy contract（已禁用）。故 AD-2 的几何层改走 §3.1（一）。

### AD-1：T1 材料/频率 JVP+VJP（Deterministic + Path，几何 detach）

- [x] 定义 fixed-topology field-evaluation 输入契约（positions/normals 作为可 detach 的固定几何）；
- [x] 为 `field_free_space` / `field_reflection_sequence` / `field_transmission_sequence` 补原生 `*_jvp` 与 `*_backward` 核，薄 `autograd.Function` 封装；`field_coupled_rd` / `field_project_complex3` 移交 AD-4（前者的伴随与后者的 UTD 重算同属一块工作，见 §3.1（二））；
- [x] 打通对 **材料 `eps_r/sigma_e/thickness` 与 frequency** 的 JVP+VJP（LoS + 单反射 + 单透射多层）；
- [x] 移除 `path/solver.py` 拦截，`deterministic`/`path` 的 `Config` 接受 `ad_mode="jvp"|"vjp"`；
- [x] `ad_mode="none"` 与当前前向性能、显存保持一致（不注册图）；
- [x] 加固：钳位边界次梯度（`>=`）、`-0.0` 衰减门、可微累加、色散材料的频率 AD 显式失败。

### AD-2：T1 tx/rx position + mesh vertex（Deterministic + Path，几何移动）

层 2（EM 侧，channel_native）：（2026-07-14 交付）

- [x] `field_free_space` 对 **TX/RX position** 的 JVP+VJP（LoS 场核直接吃 tx/rx 张量）；
- [x] `field_reflection_sequence` / `field_transmission_sequence` 对 **source/target/interaction_positions/interaction_normals** 的伴随与切向（`slab_fresnel_dual` / `stack_rt_dual` 补了 cos_theta 切向；反射伴随为真反向模式，穿过 frames、cos_theta 与 propagation，法线翻转按冻结分支只取符号；透射的 interaction_positions 梯度恒为零——直线路径不依赖穿墙点）；
- [x] `path_length_m` / `delay_s` 在几何可微后转为可微输出（ToA 类损失；对材料/频率余切恒零；`direction` 输出保持不可微）。

层 1（几何侧，RayD，见 §3.1（一））：（`b0d5823` 交付）

- [x] Scene 几何叶子接线：`Transmitter.position` / `ReceiverPoint.position` / `Structure.vertices` 绕开原生 host-float 构造 op，保持 autograd 图（纯张量传递，无数学）；
- [x] RayD 新增 `reflection_epc_paths_backward/_jvp`（CUDA，无 OptiX）：固定 winner 下 `d(hits, normals, path_length)/d(vertices, source, receiver)`（RayDi `fc52233`）；
- [x] 透射法线走 `raydn_face_normals_ad` 的可微面法线表（顶点函数），channel_native bridge + 薄 `autograd.Function`；
- [x] 删除 `core/ad_geometry.py` 的 torch 链重建（现存文件只做纯张量传递，无数学）；
- [x] mesh vertex 的连续部分导数（固定 winner）：反射（任意深度，AD-4b 补 solver 级 depth-2 测试）、透射与楔形绕射（AD-4b，楔核内从 winner 顶点重建边表）；LoS 结构性为零（钉死）；耦合 R↔D 顶点梯度显式拒绝（xfail(strict) 登记）。

### AD-3：T2 MC basic 可微（标量功率图）

**路线修正**：§6 原假设"复用 RayD 可微 accumulation 入口"——**不成立**。RayD 既没有 reflection-accumulation jvp，也没有可用的绕射伴随；MC basic 真正做场求值的两个累加器是 channel_native **自己的**核（`cn_mc_sionna_reflection_accumulate_cuda` @ `kernels/reflection.cu:263`、`cn_mc_sionna_diffraction_tape_accumulate_cuda` @ `kernels/diffraction.cu:1298`），RayD 只负责求交与采样 tape（离散、冻结）。因此伴随核必须写在 channel_native 内。

- [x] 打通 basic 的材料/频率通路：`_host_material_tensors` 已删除，反射/绕射材料与厚度统一走编译材料仓（`face_material_field_bundle`，`ad_mode="none"` 与 AD 模式同一来源、同一数值；primal 在 no_grad 下读取）；`float(scene.frequency)` 在 AD 模式换成活的 0-d 张量；
- [x] LoS 功率增益：`mc_los_path_gain_jvp/backward` 补了 frequency 切向/余切（`d gain_scale/df = -2*gain_scale/f`），并入 `_McLosPathGainAdFunction`（tx/rx 位置、频率；tx_power 按固定拓扑契约显式拒绝）；
- [x] 反射功率图：`mc_sionna_reflection_accumulate_backward/_jvp`（`eta_r`/`sigma`/`gain`/`thickness`/`wavelength`->frequency；逐 bounce 反向场链 + `legacy_sionna_slab_fresnel_dual` 参数对偶）。**交付偏差：ray origin（tx 位置）的解析梯度恒等于零**——沉积权重 `|Gamma|^2*Omega*(lambda/4pi)^2/(A*|cos|)` 的每个因子只依赖冻结的采样方向、法线与材料，1/d^2 扩散由冻结的射线密度/分箱承载（离散 winner）；已实测验证（对 tx 的中心差分在 h=1e-2 只读到单射线换箱跳变 ~7e-7，h<=1e-3 只剩 float32 重排序噪声），梯度经活图返回精确零；
- [x] 绕射功率图：AD-4b 交付——逐 lane 行模板化（float=primal、Dual=导数），backward 逐输入种子化探针、jvp 单次对偶；材料/频率/tx 位置可微；伪无限截断因子冻结（AD-4a 政策）；
- [x] 透射功率图：`em_layer_stack_backward/_jvp`（`stack_rt_dual` 补 cap_r/cap_t 对偶；cos_theta/CSR 层参数/频率），`straight_transmission_chains` 在 AD 模式走 `em_layer_stack_ad`；`mc_finalize_component_maps` 的 VJP 是余切 view + 广播、JVP 复用前向核；LoS/透射图布局与可见性掩码经 `mc_los_grid_maps_ad`（伴随为一只掩码 gather 核）；
- [x] `montecarlo.basic` 的 `Config` 接受 `ad_mode`；散射与绕射分量 + AD 在 launch 前显式拒绝；
- [x] seed 固定使 FD±h 与 AD 复用同一采样序列（反射方向为确定性 Fibonacci 球，天然 seed 无关）；跨 seed 梯度稳定性测试就位；
- [x] `basic/metadata.py` 计数契约重定义为实际注册的伴随 launch（`AdLaunchLedger`：LoS 1 次、布局 Function 各 1 次、反射逐 tx 各 1 次、层栈逐次评估各 1 次；finalize 0 次），`tape_bytes` 为 vjp 实际 save_for_backward 字节，jvp 为 0。

### AD-4：多交互复场收口（三 solver）

**AD-4a 执行更新（2026-07-14，已交付）**：路线改为**模板化对偶**而非修复上游手写伴随。核对结论：RayD 的 `pair_vector_output_vjp` 不做驻定点重选、丢弃 finite-wedge 因子梯度（`(void) gFiniteFactor`）、恒走 stored-Jones 入射分支、且完全没有 k 依赖——它微分的是与前向不同的函数（宿主 FD 证实每个分量差 2-4 个量级）；其子伴随（`adj_fresnel_reflection_face` / `adj_face_reflection_operator` / `adj_cplx_sqrt` / `adj_compute_edge_geometry_3d`）反而是对的。因此把 `utd_types.h`/`utd_math.h` 整体模板化到标量类型 `T`（float | `Dual`）：float 实例化与原前向**逐位一致**（6 组配置位级基线核对），`Dual` 实例化即精确 JVP；`pair_vector_output_jvp` 为单次对偶（含 k 与 omega 独立切向，频率由调用方按 dk=2πdf/c、dω=2πdf 链接），`pair_vector_output_vjp` 为按输入种子化的对偶探针（复线性输入每个复标量一次探针），有限差分版 `pair_vector_output_jvp_completion` 与死代码 add_scaled 族已删除。宿主 FD 全绿（两种通道约定：stationary+omega>0 与 fixed-point+stored-ops，覆盖 source/target/边几何/双面 eta_r/sigma/gain/频率/JVP-VJP 对偶）。**已知边界**：耦合核的 ±1e5 伪无限边界使截断因子成为 float32 端点纹波，其对输入的导数被 1e5 杠杆放大为噪声（宿主 FD 永不收敛；真无限边导数为零），故耦合对偶把该因子**冻结**为微分常量（策略注释在 `kernels/field_wedge_ad.cu`）。

- [x] 多反射链式导数（`field_reflection_sequence` 深度 > 1，Jones/complex3；核级 depth-2 测试自 AD-1，solver 级 depth-2 矩阵单元由 AD-4b 的 `test_solver_multibounce_ad.py` 收口）；
- [x] multilayer/transmission 散射矩阵对 `eps_r/sigma/thickness/freq` 导数（AD-1/AD-3 交付，`stack_rt_dual`）；
- [x] **UTD fixed-topology 导数（channel_native 内重算）**：`cn_field_diffraction_wedge`（前向 = RayD 模板化 UTD 的 float 实例 + 驻定点内部重解 + 出射方向；backward/jvp = Dual 实例种子化），复现 order-1 export 约定（半空间 Fresnel `omega>0`、bridge 硬编码 +z 极化、sqrt(tx_power) 幅度）；前向 parity 门禁（重算 vs `topology.field_xyz`）双 solver × 双材料（介质/PEC）全绿（rel ≤ 1e-3）；
- [x] `field_project_complex3` backward/jvp（场向量与到达方向都在图上，方向链承载驻定点运动）；
- [x] `field_coupled_rd` 的 backward/jvp（12 材料标量 + 频率 + 端点/交点几何；楔面算子复用 `slab_fresnel_dual`，反射腿复用 `dual_reflect_frame`，pair 走 RayD Dual 模板）；耦合驻定几何在 geometry AD 下经 `coupled_rd_prepare` 的 fixed-winner 可微重解（镜像源 + 驻定边点 + 墙面交点），一致性门禁比对拓扑存储交点；解除 seam 中 component 2/3/4 的拦截，耦合块频率改走活张量；
- [x] metadata 记录 `ad_status`、tape bytes、`jvp_launch_count`/`backward_launch_count`、forward 时间与峰值显存（AD-4b：`AdLaunchLedger` 三 solver 共用；vjp 的反向趟发生在 solve 之后、无法在 metadata 里如实测量，反向时间/显存改由 `tests/ad/test_ad_budgets.py` 的 CI 门禁承担）；
- [x] 解析小场景通过后，进入 Munich 场景（`tests/ad/test_munich_ad_smoke.py`：材料与 TX 位置各一次 backward，有限/非零/无 NaN）。

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

## 10. 完成定义（AD-4b 全部满足，2026-07-14）

- [x] AD-A0 打通，RayD 可微几何在 channel_native 内经 C-ABI 可用（不走 `rayd.torch` Python 层）；
- [x] **`deterministic` / `path` / `montecarlo.basic` 三个 solver** 均支持材料、frequency、TX/RX position 的 JVP/VJP（D/P 复系数，M 实数功率），端到端 torch autograd 可用；
- [x] §9 覆盖矩阵每个可行单元有测试；中心 FD、JVP-vs-VJP 自洽、内积对偶、T1 `gradcheck`、前向 parity 门禁全部通过；MC basic 有 seed 固定 + 跨 seed 方差测试（含真正 seed 依赖的绕射图）。矩阵备注：LoS×顶点为结构性零（钉死）；M×绕射×TX 的 FD 判据是固定带 + 接受稳定 lane 聚合（全解 FD 因离散 winner 翻转在任何步长都不收敛，实测登记在测试 docstring）；M×绕射×RX 不适用（radiomap 的网格是输出不是参数）；
- [x] 已知缺口以 `xfail(strict=True)` 显式登记（mesh vertex × 耦合 R↔D：耦合伴随把墙平面/边表当冻结 winner，solver 先显式 `NotImplementedError`）；
- [x] 时间、显存与梯度误差预算固定（`tests/ad/test_ad_budgets.py`：§9.2 常量冻结 + AD 前向/反向时间、tape 字节、CUDA 峰值显存上限进 CI）；
- [x] 前向模式（`ad_mode="none"`）不承担 AD 的额外常驻成本，RayD 走 detached 快路径（零 tape、伴随计数为零、前向 launch 数与 AD 模式一致，metadata 测试钉死）；
- [x] `psdr` 桩已删除（`f6873d8`）；
- [x] **`montecarlo.bdpt` 梯度与几何不连续性**明确标注为后续计划；`bdpt.Config` 仍拒绝 `ad_mode != "none"`；落到不支持区域时显式失败（basic 散射、D/P 散射、耦合×顶点均 launch 前拒绝）；
- [x] 以上条件全部满足后，`path`/`deterministic`/`montecarlo_basic` 的 `supports_ad` 已改为 `True`（`ad_modes=["none","jvp","vjp"]` + 逐 solver `ad_excluded`）；bdpt 保持 `False`/`["none"]`。
```