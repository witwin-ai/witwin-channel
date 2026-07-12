# Channel Native 完整替代原 Channel：功能缺口与验收计划

**状态：** Active / No-Go for unconditional replacement
**基线日期：** 2026-07-12
**代码基线：** `b54f7dc`（`main`）
**目标：** 在不依赖原 Channel、Python RayD、DrJit、Mitsuba 或 Sionna 热路径回退的前提下，使 `witwin.channel_native` 在明确支持的场景中成为可验证、可维护、可迁移的主实现，并最终满足删除原 Channel 运行时依赖的条件。

---

## 0. 2026-07-12 状态审计与剩余执行板

本节是本计划的当前权威状态。后文保留完整设计背景；后文未勾选的原始任务不再等同于“尚未开始”，应以本节的状态和验收证据为准。

### 0.1 Phase 状态

| Phase | 状态 | 已落地 | 完成前仍需满足 |
|---|---|---|---|
| 0 契约与诚实失败 | 部分完成 | 公共 capability manifest；四类 Solver 明确 `supports_ad=False`；不支持的 AD mode 在配置阶段失败 | 完成原 Channel API 使用清单、replacement matrix JSON、所有 requested/effective metadata 审计 |
| 1 PathResultV2 | 大部分完成 | ragged SoA、padded `PathResultV2`、legacy/topology adapter、signal views、synthetic-array 展开和对应单元测试 | 与原 Channel 的完整 shape/mask/排序契约；真实 Solver 输出上的复系数、角度、交互序列和阵列端到端验收 |
| 2 共享多阶拓扑 | 部分完成 | Deterministic 多阶反射；公共 topology 和几何/张量工具已抽取 | Path 高阶反射以及 reflection-diffraction 组合拓扑；跨 Solver canonical ordering 与去重 |
| 3 复场与极化 | 部分完成 | 公共 native field transport；反射、衍射和 deterministic 场计算已统一部分约定 | BDPT Jones/complex3 状态；所有 Solver 对同一解析路径的幅度、相位和极化三方一致性门禁 |
| 4 天线、阵列与角度 | 部分完成 | `PathResultV2` synthetic-array 基础能力 | 真实方向图、orientation、polarization port、explicit array、precoding/combining 与多 TX/RX 验收 |
| 5 Path 三方验收 | 部分完成 | LOS/反射时延和候选覆盖抽样；PathResultV2 单元测试 | 对原 Channel/Sionna 的复系数、AoA/AoD、序列、CIR/CFR/taps 和大场景 CI 门禁 |
| 6 BDPT 物理状态与 MIS | 部分完成 | Native BDPT、基础 MIS/连接逻辑、Munich 与 single-plane 压力测试 | 极化场状态、PDF/测度逐项审计、可重复统计区间、冷启动和高深度稳定性门禁 |
| 7 材料、透射与散射 | 核心完成、覆盖未闭合 | Material ABI v3、稳定 multilayer slab、specular transmission、rough-surface scattering、UV/phase-screen/Kirchhoff 路线 | 完整材料目录与频散、复杂 XML/UV/instancing、跨 Solver 能量与互易性场景矩阵 |
| 8 可微分能力 | 未完成 | 仅有孤立的 `mc_los_path_gain_jvp` CUDA 原语和 AD 元数据结构 | Solver 级 fixed-topology JVP/VJP、PyTorch autograd 接入、参数梯度、有限差分/解析/原 Channel 三方验收 |
| 9 性能、显存与部署 | 部分完成 | 小中型场景 steady-state 基准和若干 Munich/SF 产物 | 统一 cold/steady/p95、峰值显存、100M sample、depth 3/5、多 GPU 架构和 wheel smoke 门禁 |
| 10 迁移与删除旧运行时 | 未完成 | Native 热路径已不依赖 Python fallback | 公共 API 迁移、生产调用方切换、双跑观察期及删除原 Channel runtime |

### 0.2 当前发布边界

- Channel Native 可以作为明确 capability 范围内的前向仿真后端；不能宣称无条件替代原 Channel。
- 可微分求解当前是产品级缺口。所有公开 Solver 的有效 `ad_mode` 仍只有 `none`。
- `mc_los_path_gain_jvp` 只证明一个 LOS path-gain 原语可计算方向导数；它没有接入任一 Solver，也不构成通用 JVP、VJP 或 PyTorch autograd 支持。
- 衍射采用物理正确、数值稳定优先的版本；由于 MC 衍射方差高，不以单 seed 像素级一致作为发布门槛，而采用解析极限、能量范围、跨 seed 置信区间和无 NaN/Inf 验收。

### 0.3 剩余缺口与改进方案

#### Milestone A：前向功能闭合（P0）

- [ ] 生成 machine-readable replacement matrix，逐项映射原 Channel API、Native API、capability、失败语义和测试证据。
- [ ] 使 Path 的 `requested_max_depth`、`effective_max_depth` 和逐 component 深度完全可审计；未实现组合拓扑必须在 GPU allocation 前失败。
- [ ] 将真实 Path Solver 输出完整接入 `PathResultV2`：complex field、AoA/AoD、interaction/material/primitive 序列、稳定排序和每 pair 上限。
- [ ] 对同一解析路径建立 Path、Deterministic、MC Basic、BDPT 的复场/相位/极化一致性测试。
- [ ] 补齐高阶 reflection 以及最多一次 diffraction 前后带反射的组合拓扑；定义 canonical ordering、去重和截断策略。

完成门槛：所有 P0 capability 都有正向测试和超范围失败测试；不存在配置被接受但热路径静默忽略的情况。

#### Milestone B：信道与阵列契约（P1）

- [ ] 在真实 Solver 结果上验收 `cir()`、`cfr()`、`taps()`、`filter_by_type()`，覆盖 0 path、padding、mask 和多 pair。
- [ ] 实现并验证 TX/RX orientation、天线方向图、极化端口、ULA/URA synthetic array 和 explicit array。
- [ ] 增加多 TX/RX、precoding/combining 和全局旋转/平移不变性测试。
- [ ] 明确移动信道是否进入替代范围；若进入，实现 velocity、Doppler、time axis 和动态场景 cache invalidation。

完成门槛：公共输出 shape、单位、mask、角度、阵列相位和时间语义与替代契约一致。

#### Milestone C：BDPT、材料与数值稳定性（P1）

- [ ] 将 BDPT scalar complex throughput 升级为 Jones/complex3 + local frame 状态，禁止场幅与功率权重混用。
- [ ] 审计每种连接策略的 forward/reverse PDF、Jacobian、delta 事件和 MIS measure；用可枚举小场景验证无偏性。
- [ ] 建立材料频散、multilayer、transmission、rough scattering 的跨 Solver 能量守恒、互易性和 grazing-angle 测试矩阵。
- [ ] 对 MC Basic/BDPT/衍射执行跨 seed 压力测试，报告均值、方差、置信区间、finite ratio 和极端几何失败率。

完成门槛：不以单 seed parity 作为结论；统计门槛预先固定，并在 Munich、SF 与解析小场景同时通过。

#### Milestone D：可微分求解（P1，替代原 Channel AD 的阻塞项）

- [ ] 冻结第一版可微参数：TX/RX position、frequency、material continuous parameters；vertices 延后到第二批。
- [ ] 将 topology discovery 与 fixed-topology field evaluation 分离，明确 visibility/topology discontinuity 不提供普通连续梯度。
- [ ] 先为 LOS、单反射、单透射实现 Solver 级 JVP，再实现 VJP/autograd；不得用 metadata launch count 代替真实导数路径。
- [ ] 增加解析导数、中心有限差分、JVP-VJP 对偶性和原 Channel AD 四类测试。
- [ ] 扩展到多反射、UTD 和 rough scattering 前，先定义不连续点、随机数重参数化和梯度方差政策。
- [ ] 只有在公开 Solver 接受 `ad_mode=jvp/vjp` 且端到端测试通过后，才将 capability manifest 的 `supports_ad` 改为 `True`。

完成门槛：至少一个生产 Solver 对声明的参数集合提供稳定 JVP/VJP；梯度误差、显存 tape 和运行时间均有门禁。

#### Milestone E：性能、部署与迁移（P0/P1）

- [ ] 固化 analytic、three-cube、Munich、SF planar、terrain 的 cold/steady/p95 与峰值显存基准。
- [ ] 覆盖 1x1、8x1k、16x1k、128²/512² grid，MC 1k/1M/10M/100M samples 和 depth 0/1/3/5。
- [ ] 增加支持矩阵内的 CUDA/driver/OptiX/PyTorch/Python/SM wheel smoke 和清晰 import diagnostics。
- [ ] 建立至少一个发布周期的原 Channel/Native 双跑观察；记录精度、性能、失败率和回滚条件。
- [ ] 全部门禁通过后迁移调用方，再删除原 Channel runtime、fallback 和仅供迁移使用的 adapter。

完成门槛：发布报告同时包含功能、物理、统计、性能、显存、冷启动和部署结果；任何一项缺失均不宣布完整替代。

### 0.4 推荐执行顺序

1. Milestone A：先消除错误成功和输出契约缺口。
2. Milestone C 中的 BDPT 场状态与统计门禁，与 Milestone B 阵列工作并行推进。
3. Milestone D：以 fixed-topology LOS/单反射为最小可验证 AD 垂直切片。
4. Milestone E：在接口和物理契约冻结后固化最终性能门槛。
5. Phase 10：只在 A-E 的发布门槛全部完成后执行。

---

## 1. 结论与边界

当前 Channel Native **不能作为原 Channel 的无条件、全功能替代品**。它已经适合以下受限范围：

- NVIDIA CUDA/OptiX 环境；
- LOS、镜面反射、一阶 UTD 衍射；
- ReceiverGrid radiomap；
- Deterministic 多阶镜面反射；
- Path Solver 一阶路径拓扑与时延导出；
- 不要求 AD、透射、粗糙面散射、RIS、移动信道或完整多天线复系数输出。

当前不允许做出的声明：

1. 不得声称 Path Solver 已支持 `max_depth > 1`。当前配置接受该值，但反射和衍射仍固定调用 `*_order1`。
2. 不得把 Path Solver 的标量 `path_gain` 当作原 `PathResult.a` 的完整替代。
3. 不得把单 seed 衍射图的逐像素相关作为物理正确性的唯一依据。
4. 不得因为 reduced/synthetic 场景通过，就声称 Munich、San Francisco、terrain、多 TX/RX 和深路径均完成验收。
5. 不得以静默降级、简化公式或原 Channel 回退补齐缺失功能。

完整替代的定义包含三层：

| 层级 | 定义 | 当前状态 |
|---|---|---|
| Kernel parity | 单个 CUDA/OptiX 几何与场计算满足解析或参考测试 | 部分完成 |
| Solver parity | 同一配置下，拓扑、复场、统计量、元数据和失败语义一致 | 部分完成 |
| Product replacement | API、场景、性能、显存、冷启动、部署与迁移均有门槛 | 未完成 |

---

## 2. 当前证据基线

### 2.1 自动化测试

- 完整测试：`340 passed, 1 skipped`。
- 最终衍射/加载/无回退重点回归：`31 passed`。
- Release CUDA 扩展构建成功。
- 生产源码静态约束禁止 legacy fallback、CPU tensor readback 和 ATen compute 热路径。

这些测试证明当前实现内部契约稳定，但不等于完成原 Channel 全功能覆盖。

### 2.2 Path Solver 三方抽样

场景：San Francisco，4096 samples，1 TX，2 RX，warmup 1，重复 3 次。

| 分量 | Channel Native | 原 Channel | Sionna | Native 相对原 Channel |
|---|---:|---:|---:|---:|
| all | 4.004 ms | 213.663 ms | 50.141 ms | 53.36x |
| LOS | 1.712 ms | 12.904 ms | 10.810 ms | 7.54x |
| reflection | 2.797 ms | 15.682 ms | 23.712 ms | 5.61x |
| diffraction | 2.696 ms | 199.016 ms | 54.897 ms | 73.82x |

正确性抽样：

- LOS、反射路径数量与参考一致；
- LOS、反射时延误差约 `1e-13 s`；
- Native 覆盖原 Channel 与 Sionna 的全部参考衍射时延；
- Native 导出 313 条一阶边路径，Sionna 25 条，原 Channel 1 条；这是候选集语义差异，不能视作路径数量 parity；
- 该基准只验证路径时延与数量/覆盖，不验证每条路径的复振幅、相位、极化、AoA/AoD 或几何序列。

### 2.3 Deterministic 与 BDPT

- Reduced Munich Deterministic：Native `25.95 ms`，原 Channel solve `809.24 ms`；LOS 中位误差 `0 dB`，反射中位误差 `0 dB`，衍射中位误差约 `1.706 dB`。
- Reduced Munich BDPT：Native `1.804 ms`，原 Channel `86.518 ms`，稳态加速约 `47.96x`。
- Single-plane BDPT：已有 strict shape/nonzero/speed/relative-sum gate。
- BDPT 冷启动曾测得 Native 约 `30.8 s`、原 Channel约 `6.1 s`；稳态快并不代表冷启动已达标。

### 2.4 MC Basic 物理基线

- Munich LOS 与 Sionna：总功率比约 `1.0009`，dB 相关约 `0.9826`，中位绝对误差约 `0.203 dB`。
- Munich 反射与 Sionna：总功率比约 `0.9606`，dB 相关约 `0.9905`，中位绝对误差约 `0.111 dB`。
- 衍射采用物理稳定版：完整 UTD、有限厚度材料、外部楔角、可见性和边长重要性采样；不以单 seed 逐像素一致作为验收门槛。
- MC Basic 的 CI 性能测试目前只验证 benchmark 输出字段，`performance_budget_ms` 仍为 `None`。

---

## 3. 完整功能缺口清单

优先级定义：

- **P0：** 阻止正确替代或会静默产生错误结果；
- **P1：** 原 Channel 公共能力或生产必需能力缺失；
- **P2：** 扩展物理能力、性能与维护性；
- **P3：** 长期研究能力，不阻塞首轮替代。

### 3.1 P0：Path Solver 深度配置与实际执行不一致

当前：

- `path.Config.max_depth` 接受任意非负值；
- `path.solver.solve()` 对反射固定调用 `reflection_paths_order1()`；
- 对衍射固定调用 `diffraction_paths_order1()`；
- `max_depth` 只进入 finalize/metadata，可能使用户误以为高阶路径已计算。

必须完成：

1. 在高阶实现完成前，对 `max_depth > 1` 且请求 reflection/diffraction 明确报错；
2. 实现反射序列深度 `1..max_depth`；
3. 明确定义衍射支持：首轮只支持单次衍射，但允许反射-衍射组合，或明确限制为纯一阶衍射；
4. metadata 必须报告 `requested_max_depth`、`effective_max_depth`、各 component 的实际最大深度；
5. 测试必须证明增加 `max_depth` 会增加可构造的路径集合，而不是只改变 tensor padding。

涉及文件：

- `src/witwin/channel_native/path/config.py`
- `src/witwin/channel_native/path/solver.py`
- `src/witwin/channel_native/path/raydn_export.py`
- `src/witwin/channel_native/deterministic/topology.py`
- `native/channel_native/raydn_bridge.cpp`
- `native/channel_native/kernels/deterministic_field.cu`

### 3.2 P0：Path Result 不是原 PathResult 的等价物

原 Channel `PathResult` 提供：

- 复系数 `a`；
- `tau`；
- `theta_t/phi_t`、`theta_r/phi_r`；
- `[rx, rx_ant, tx, tx_ant, path, time]` 维度；
- 每深度 `types`；
- `num_paths`；
- 可选 `vertices/normals/objects`；
- `cir()`、`cfr()`、`taps()`、`filter_by_type()`；
- 多时间步和 synthetic-array 语义。

Native 当前只提供扁平的一维路径表：

- `valid/tx_id/rx_id/depth/component_id/primitive_id/edge_id`；
- `path_length_m/delay_s/path_gain`；
- 没有复系数、角度、完整交互序列、时间维、天线维或 CIR/CFR/taps。

必须完成：

1. 新建版本化 `PathResultV2`，不要在旧扁平结果上隐式改变 shape；
2. 同时保留内部 SoA 扁平表示与公共 padded/ragged view；
3. 增加 `field: complex64` 或 `field_xyz: complex64[...,3]`；
4. 增加 `interaction_type/primitive/material/position/normal` 的深度序列；
5. 增加 AoD/AoA；
6. 增加每 TX/RX/antenna pair 的 `num_paths` 和稳定排序；
7. 实现 `cir/cfr/taps/filter_by_type`，并与原 Channel shape/掩码语义一致；
8. 对 `max_paths` 明确规定是全局上限还是每 pair 上限。推荐每 pair 上限，与原结果一致。

### 3.3 P0：复场、相位与极化契约未在所有求解器中统一

当前：

- Deterministic 已保存复场和多阶反射序列；
- Path Solver 只输出标量功率；
- BDPT 子路径使用 scalar complex throughput；
- scalar throughput 不能完整表示连续反射后的极化旋转；
- 不同求解器仍可能在事件处混用场幅系数与功率系数。

必须完成：

1. 定义统一 `complex3` 或 Jones+frame 路径状态；
2. 每个交互执行入射基投影、Jones 操作、出射基重组；
3. Fresnel/UTD 局部事件只更新复场，不重复加入自由空间因子；
4. 路径末端统一投影到接收天线并生成无量纲 channel coefficient；
5. BDPT 的 throughput、PDF、MIS 明确使用场域或功率域，不允许混合；
6. Path、Deterministic、MC/BDPT 对相同解析路径输出一致的复系数。

验收必须同时比较：

- 幅值；
- wrapped phase；
- 三维场或 Jones 分量；
- 旋转场景下的基变换不变性；
- 两次非共面反射后的交叉极化。

### 3.4 P1：多天线、方向图与 synthetic array 不完整

缺口：

- Native endpoint 主要表达位置和发射功率；
- 默认极化固定为垂直方向；
- 没有完整 TX/RX orientation、天线 pattern、阵元坐标、precoding/combining；
- Path Result 没有 `tx_ant/rx_ant` 维；
- `synthetic_array` 与显式阵列的公共语义未复刻。

必须完成：

1. 扩展 `Transmitter/Receiver`：orientation、pattern、polarization、array geometry；
2. 支持 synthetic array phase weighting；
3. 支持 explicit array 的逐阵元路径系数；
4. 输出 shape 与原 Channel 兼容；
5. 解析验证 ULA/URA 的 steering phase；
6. 测试全局旋转、阵列平移和极化正交抑制。

### 3.5 P1：路径类型与组合拓扑不完整

当前主要覆盖：

- LOS；
- 纯镜面反射；
- 一阶 UTD 衍射。

缺失或未完整验证：

- reflection → diffraction；
- diffraction → reflection；
- 多次反射后衍射；
- 多次衍射；
- 事件序列去重和 canonical ordering；
- shadow-boundary 邻域连续性；
- 相同几何路径由不同构造器重复生成时的合并策略。

首轮替代建议：

- 必须支持任意阶镜面反射；
- 必须支持最多一次衍射，并允许其前后插入受限数量的反射；
- 多次衍射作为 P2，除非现有生产调用明确依赖 `max_diffraction_order > 1`。

### 3.6 P1：材料和事件能力不足

当前材料模型：

- Dielectric / LossyDielectric / PerfectConductor；
- `eps_r/mu_r/sigma_e/gain/thickness_m`；
- 有限厚度 slab reflection；
- UTD face operators。

替代原 Channel 所需：

1. 明确频散材料参数如何随频率求值；
2. 保留 XML/Sionna material 的全部必要字段和单位；
3. 材料 ABI 版本化，停止依赖匿名 `model_params[:,0]`；
4. 每个结果可追踪 material id/model id；
5. PerfectConductor 使用明确模型，不仅依赖大 conductivity 近似；
6. 材料 cache 在 frequency/material 改变时正确失效。

物理路线扩展（不阻塞第一轮原 Channel 替代，除非生产需求确认）：

- transmission/refraction；
- absorption accounting；
- rough-surface scattering；
- layered/dispersive media；
- tabulated polarimetric BSDF；
- medium stack。

### 3.7 P1：场景与几何覆盖不完整

必须核对并补齐：

- Mitsuba XML transform、nested transform；
- PLY 法线、UV、face-UV；
- instancing 和共享 mesh；
- duplicate vertex 与 merge-shapes 语义；
- boundary/non-manifold edge；
- dynamic geometry cache invalidation；
- 多 structure 的 material/surface/shape id 稳定映射；
- ReceiverPoint、ReceiverGrid、planar map、terrain/mesh measurement surface；
- 大坐标场景的 float32 精度和局部坐标策略。

当前 `core/runtime/raydn.py` 向 RayD 传空 UV；在粗糙面、纹理材料或表格化 BSDF 前必须补齐 UV 通道。

### 3.8 P1：AD 与优化工作流缺失

当前 Native Path、Deterministic、MC Basic、BDPT 的有效 `ad_mode` 均为 `none`。

需要先做产品决策：

- 如果替代目标包含原 Channel 的 differentiable workflow，AD 是阻塞项；
- 如果生产只需要前向仿真，应在 public capability manifest 中明确 `ad=False`，并把 AD 移到 P2。

AD 路线：

1. 定义可微参数：TX/RX、vertices、material、frequency；
2. topology discovery 与 field evaluation 分离；
3. topology 在小扰动下固定时提供 JVP/VJP；
4. silhouette/visibility discontinuity 使用明确估计器，不伪装成普通连续梯度；
5. 与有限差分、原 Channel 和解析导数三方验证。

### 3.9 P1：移动信道与时间维缺失

原 `PathResult` 暴露 `num_time_steps`，Native 当前没有完整时间轴。

需要确认生产是否依赖：

- TX/RX velocity；
- Doppler shift；
- time-varying path coefficient；
- 几何运动和拓扑更新；
- CIR 时间序列。

如依赖，则必须增加统一时间采样、速度投影、多普勒相位和动态场景 cache 策略。

### 3.10 P1：性能验收矩阵不完整

当前有很好的小规模稳态数字，但缺少完整生产门槛：

- Path Solver 三方基准尚未进入 CI；
- MC Basic 没有正式 `performance_budget_ms`；
- 100M sample、depth 3/5、16 GB 显存门槛没有最终产物；
- Path point receiver 的大规模 TX×RX×edge 测试不足；
- 冷启动、scene compile、steady-state 被混合报告；
- 没有统一记录 peak GPU memory、temporary bytes 和 OptiX build bytes；
- 没有跨 GPU architecture 的性能基线。

必须建立：

| 维度 | 最少测试点 |
|---|---|
| 场景 | analytic、three-cube、Munich、San Francisco planar、terrain |
| TX/RX | 1×1、1×1k、16×1k、grid 128²/512² |
| 深度 | 0、1、2、3、5 |
| samples | 1k、1M、10M、100M（适用 MC） |
| 计时 | cold import、scene load、pipeline build、first solve、steady median/p95 |
| 内存 | persistent、peak temporary、output、tape |
| 数值 | finite、support、sum、dB distribution、cross-seed confidence interval |

### 3.11 P1：失败语义、元数据和 capability discovery 不完整

必须增加单一 capability manifest：

```python
capabilities = {
    "components": {"los", "reflection", "diffraction"},
    "max_reflection_depth": 5,
    "max_diffraction_order": 1,
    "supports_reflection_diffraction_coupling": True,
    "supports_complex_path_coefficients": True,
    "supports_polarization": True,
    "supports_arrays": True,
    "supports_ad": False,
    "receiver_types": {"point", "grid"},
}
```

规则：

- config 超出 capability 时必须在 launch 前失败；
- metadata 报告 requested/effective config；
- 不允许只在 metadata 中声称支持但热路径未执行；
- 不允许静默截断深度、路径数或事件类型；
- 结果必须记录 solver/build ABI、seed、edge policy、polarization convention。

### 3.12 P2：部署与跨平台

缺口：

- NVIDIA-only；
- OptiX/CUDA/RayD pipeline 冷启动较重；
- wheel/package ABI、CUDA version、SM architecture 兼容矩阵未固化；
- 无 CPU/ROCm 后端属于明确产品限制，不应通过慢速 Python 回退隐藏。

必须提供：

- 支持的 CUDA、driver、OptiX、PyTorch、Python、SM 列表；
- import/build diagnostics；
- wheel smoke；
- 缺少 CUDA/OptiX 时的清晰错误；
- pipeline cache 与版本失效规则。

---

## 4. 分阶段实施计划

### Phase 0：冻结替代契约与诚实失败（P0）

目标：消除“配置接受但功能未执行”的风险。

任务：

- [ ] 新建公共 capability manifest；
- [ ] Path Solver 在高阶实现前拒绝 `max_depth > 1`；
- [ ] 为每个 solver 增加 requested/effective metadata；
- [ ] 列出原 Channel 公共 API 使用情况，按真实调用频率分类；
- [ ] 建立 replacement matrix 文档与 machine-readable JSON；
- [ ] 禁止新增 fallback/unsupported silent branch。

验收：

- 所有超出支持范围的 config 在任何 GPU allocation 前报错；
- metadata 与实际 kernel launch 一致；
- API 合约测试覆盖所有 config 字段。

### Phase 1：Path Result V2 与公共信道输出（P0）

目标：先建立能承载完整物理量的结果结构。

任务：

- [ ] 定义内部 ragged SoA schema；
- [ ] 定义公共 padded `PathResultV2`；
- [ ] 增加 complex coefficient、AoA/AoD、interaction sequence；
- [ ] 增加 vertices/normals/objects/materials；
- [ ] 增加 TX/RX/antenna/time 维；
- [ ] 实现 `cir/cfr/taps/filter_by_type`；
- [ ] 提供旧扁平 Result 的显式 adapter，而非隐式 shape 改动。

建议新增文件：

- `src/witwin/channel_native/path/schema.py`
- `src/witwin/channel_native/path/result_v2.py`
- `native/channel_native/kernels/path_result_pack.cu`
- `tests/path/test_path_result_v2.py`
- `tests/path/test_path_signal_views.py`

验收：

- shape、mask、padding、排序与原 `PathResult` 一致；
- CIR/CFR/taps 对手工路径解析值一致；
- 0 path、1 path、超 `max_paths` 和多 antenna case 全覆盖。

### Phase 2：共享多阶拓扑（P0/P1）

目标：Path 与 Deterministic 使用同一套 canonical topology，而不是分别维护。

任务：

- [ ] 抽取 Deterministic 已有多阶 reflection topology；
- [ ] 输出 Path Solver 可消费的路径序列；
- [ ] 增加 per-pair pruning 与稳定 top-k；
- [ ] 实现 canonical sequence key 与去重；
- [ ] 支持 `max_depth=1..5`；
- [ ] 处理 coplanar face group、secondary visibility 和自相交 offset；
- [ ] 首轮支持最多一次衍射与反射组合。

验收场景：

- parallel mirrors；
- non-coplanar two-wall；
- three-cube；
- single wedge；
- reflection→diffraction 与 diffraction→reflection；
- Munich reduced depth 1/2/3。

验收指标：

- deterministic scenes 的 path count/sequence 精确一致；
- path length 绝对误差 `<= 1e-5 m`；
- delay 绝对误差 `<= 1e-12 s`；
- 不出现 duplicate canonical sequence。

### Phase 3：统一复场与极化（P0）

目标：所有 solver 共享同一物理事件运算。

任务：

- [ ] 设计 `complex3`/Jones state ABI；
- [ ] 统一 free-space phase 和 normalization；
- [ ] 统一 slab Fresnel 与 UTD operators；
- [ ] 修复 BDPT scalar throughput；
- [ ] 支持接收天线投影；
- [ ] 输出 per-path complex coefficient；
- [ ] 加入相位 convention metadata。

验收：

- free-space analytic；
- PEC 单反射；
- lossy slab 单反射；
- 两次非共面反射；
- simple wedge UTD；
- 全局旋转不变性；
- Path/Deterministic 对同一路径复场一致。

数值门槛：

- 解析场幅相对误差 `<= 1e-4`；
- wrapped phase 误差 `<= 1e-3 rad`（解析小场景）；
- Munich/SF 统计场景按分布报告，不对高方差衍射设逐像素硬门槛。

### Phase 4：天线、阵列与角度（P1）

任务：

- [ ] endpoint orientation；
- [ ] isotropic/vertical/horizontal/custom pattern；
- [ ] ULA/URA；
- [ ] synthetic array；
- [ ] explicit array；
- [ ] precoding/combining；
- [ ] AoA/AoD；
- [ ] 多 TX/RX/antenna result packing。

验收：

- steering-vector 解析测试；
- 与 Sionna synthetic array 小场景一致；
- 极化正交时接收功率抑制；
- 相同阵列整体平移只改变预期传播相位。

### Phase 5：Path Solver 三方完整验收（P1）

扩展 `benchmarks/bench_path_solver_threeway.py`：

- [ ] 比较 complex coefficient，而不只是 delay；
- [ ] 比较 AoA/AoD；
- [ ] 比较 interaction types/geometry；
- [ ] 比较 CIR/CFR；
- [ ] 区分 exact-count component 与 coverage component；
- [ ] 增加 multi-seed diffraction confidence interval；
- [ ] 增加 cold/steady/peak-memory；
- [ ] 输出版本化 JSON schema；
- [ ] 纳入 CI reduced gate。

场景矩阵：

- analytic empty space；
- one wall；
- two non-coplanar walls；
- simple wedge；
- three-cube；
- Munich；
- San Francisco；
- terrain（若 Path 支持 mesh receiver）。

完成门槛：

- LOS/镜面反射 deterministic case：路径数、序列和 delay 精确门槛通过；
- complex magnitude 中位误差 `< 0.25 dB`；
- phase 使用解析/同 convention 参考，不能混用原 Channel X 极化与 Sionna Z 极化；
- 衍射参考 delay coverage `>= 95%`，并报告置信区间；
- 无 NaN/Inf；
- `max_depth=1..3` 均有实际新增路径和参考覆盖。

### Phase 6：BDPT 物理状态与 MIS 完整性（P1）

任务：

- [ ] scalar throughput → complex3/Jones；
- [ ] 事件概率、forward/reverse PDF 与 throughput 域统一；
- [ ] delta reflection 与 continuous scattering MIS 分类；
- [ ] reflection-diffraction coupled path 的双向 PDF；
- [ ] point/grid result 一致；
- [ ] cross-seed unbiasedness 与 variance gate；
- [ ] 独立 point-receiver 性能基线。

完成门槛：

- 单平面和 reduced Munich strict gate；
- 两次非共面反射极化解析测试；
- sample count 4x 时均值置信区间收缩；
- MIS on/off 在可比配置下均值一致。

### Phase 7：材料、透射与散射路线（P1/P2）

此阶段分两层：

**7A 原 Channel 替代必需：**

- [ ] 频散材料；
- [ ] material ABI v2；
- [ ] XML/material parity；
- [ ] PEC 明确模型；
- [ ] material id 可追踪。

**7B 新物理能力：**

- [ ] transmission/refraction；
- [ ] absorption；
- [ ] layered medium；
- [ ] rough scattering；
- [ ] tabulated BSDF；
- [ ] medium stack；
- [ ] energy accounting。

7B 不应阻塞第一轮替代，除非调用清单证明生产已依赖对应事件。

### Phase 8：AD 决策与实现（P1/P2）

- [ ] 统计原 Channel AD 实际调用；
- [ ] 若无人使用，发布明确 `supports_ad=False` 的首轮替代版本；
- [ ] 若必须支持，按 topology-fixed JVP/VJP → visibility discontinuity estimator 分阶段实现；
- [ ] 增加 finite-difference、解析和原 Channel 三方测试。

### Phase 9：性能、显存与部署验收（P1）

任务：

- [ ] 统一 benchmark harness；
- [ ] CUDA event + wall clock 双计时；
- [ ] cold/first/steady 分离；
- [ ] peak GPU memory；
- [ ] TX/RX/depth/sample scaling；
- [ ] pipeline cache；
- [ ] SM 版本矩阵；
- [ ] wheel/import smoke；
- [ ] OOM 前预算检查与清晰错误。

建议门槛：

- maintained steady-state case 不慢于原 Channel；
- Path/Deterministic/BDPT 的代表场景目标至少 `2x`；
- cold start 单独报告，不能被 warmup 隐藏；
- 16 GB GPU 上 maintained memory-safe 配置不得 OOM；
- 100M MC 配置必须有明确 peak-memory 产物。

### Phase 10：迁移与删除原运行时依赖（P0）

迁移步骤：

1. 建立真实调用 inventory；
2. 按 capability 将调用迁移到 Native；
3. shadow run：原/Native 同时运行，仅记录差异；
4. canary：小比例以 Native 为主；
5. default-on：Native 主路径；
6. 保留离线 oracle benchmark，但生产运行时不依赖原 Channel；
7. 连续两个发布周期无回退需求后删除原运行时集成。

最终删除门槛：

- [ ] 所有 P0/P1 项关闭或经产品决策明确不支持；
- [ ] Path Result 公共契约完成；
- [ ] Path `max_depth` 真实生效；
- [ ] complex field/polarization 完成；
- [ ] 真实调用 inventory 100% 可路由；
- [ ] maintained 三方正确性 gate 全绿；
- [ ] maintained 性能/显存 gate 全绿；
- [ ] cold-start 和部署 gate 全绿；
- [ ] 无生产 fallback；
- [ ] 原 Channel 仅保留为离线参考或独立归档。

---

## 5. 必须新增的测试与基准

### Path

- `tests/path/test_path_max_depth_truthfulness.py`
- `tests/path/test_path_multibounce_reflection.py`
- `tests/path/test_path_reflection_diffraction_sequences.py`
- `tests/path/test_path_result_v2.py`
- `tests/path/test_path_complex_field.py`
- `tests/path/test_path_angles_geometry.py`
- `tests/path/test_path_cir_cfr_taps.py`
- `tests/path/test_path_array_shapes.py`
- `tests/path/test_path_munich_threeway.py`
- `tests/path/test_path_performance_gate.py`

### Physics

- `tests/physics/test_complex_field_convention.py`
- `tests/physics/test_polarization_rotation.py`
- `tests/physics/test_slab_energy_balance.py`
- `tests/physics/test_utd_simple_wedge_reference.py`
- `tests/physics/test_global_rotation_invariance.py`

### Performance

- `benchmarks/bench_path_solver_scaling.py`
- `benchmarks/bench_path_solver_threeway.py`（扩展）
- `benchmarks/bench_solver_cold_start.py`
- `benchmarks/bench_solver_peak_memory.py`
- `tests/performance/test_path_solver_budget.py`
- `tests/performance/test_mc_memory_budget.py`
- `tests/performance/test_deterministic_depth_scaling.py`

---

## 6. 推荐执行顺序

严格顺序：

1. **Phase 0：诚实 capability 与失败语义**；
2. **Phase 1：Path Result V2**；
3. **Phase 2：共享多阶拓扑**；
4. **Phase 3：复场与极化统一**；
5. **Phase 5：Path 三方完整验收**；
6. **Phase 4：阵列与角度**（可与 Phase 5 后半并行）；
7. **Phase 6：BDPT throughput/MIS**；
8. **Phase 7A：材料替代必需项**；
9. **Phase 9：规模性能与部署**；
10. **Phase 8 与 7B：按产品需求决定**；
11. **Phase 10：切换与删除原运行时依赖**。

第一里程碑不是“实现更多公式”，而是：

> Path Solver 对不支持的深度诚实失败，并输出能够承载复场、序列、角度和 CIR 的稳定结果契约。

第二里程碑是：

> Path 与 Deterministic 共享同一多阶拓扑，且对解析场景输出相同复路径系数。

只有在这两个里程碑完成后，才值得把 Path Solver 宣布为原 Channel 的替代实现。

---

## 7. 每阶段统一验证命令

```powershell
$env:PYTHONPATH='E:\Code\witwin-platform\channel_native\src;E:\Code\witwin-platform\channel_native\build-sionna-dev'
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pytest -q

C:\Users\Asixa\miniconda3\envs\witwin2\python.exe `
  benchmarks\bench_path_solver_threeway.py `
  --samples 4096 --warmup 1 --repeats 3 `
  --output artifacts\path_solver_threeway.json

C:\Users\Asixa\miniconda3\envs\witwin2\python.exe `
  -m tests.support.bin.benchmark_munich_deterministic_native_vs_original `
  --grid-size 32 --max-depth 2

C:\Users\Asixa\miniconda3\envs\witwin2\python.exe `
  -m tests.support.bin.benchmark_munich_bdpt_native_vs_original `
  --samples 16 --grid-size 4 --max-depth 1 --warmup-runs 1
```

每次验收必须保存：

- commit SHA；
- GPU/driver/CUDA/OptiX/PyTorch 版本；
- cold 与 steady timing；
- peak memory；
- 配置和 seed；
- correctness metrics；
- 是否满足每个 gate；
- 原始 JSON/NPZ，而不仅是截图。
