# Channel Native 前向仿真器完成计划

**状态：** Active
**基线日期：** 2026-07-12
**代码基线：** `11ee238`（`main`）

## 1. 目标与边界

目标是把 `witwin.channel_native` 建成一个独立、干净、稳定且高性能的前向信道仿真器。

本计划遵循以下边界：

- 不修改旧 `channel` 包；
- 不迁移或删除旧 `channel` 运行时；
- 不继承旧包的内部架构、重型 metadata、序列化 schema 或兼容层；
- 原 Channel 与 Sionna 只用于离线数值参考，不进入 Native 热路径；
- 不提供旧结果类型的 adapter，不维持向后兼容；
- 可微分能力不属于本计划，单独见 [07-channel-native-differentiable-solver-plan.md](./07-channel-native-differentiable-solver-plan.md)；
- 衍射以物理正确性和统计稳定性为准，不追求单 seed 像素级一致。

完成标准不是“覆盖旧包所有接口”，而是：声明支持的能力全部真实执行、结果物理可信、接口简洁、性能与数值稳定性有自动化门禁。

## 2. 当前状态

| 模块 | 状态 | 主要剩余工作 |
|---|---|---|
| Path Solver | 部分完成 | 统一结果类型、高阶反射、组合拓扑、真实复场输出 |
| Deterministic | 核心可用 | 跨场景复场/极化门禁、深度与显存压力测试 |
| MC Basic | 核心可用 | 跨 seed 统计门禁、大样本性能与稳定性 |
| MC BDPT | 部分完成 | Jones/complex3 状态、PDF/MIS 审计、高深度统计稳定性 |
| 材料/透射/散射 | 核心可用 | 频散材料、复杂场景覆盖、跨 Solver 能量测试 |
| 衍射 | 稳定版 | 解析极限、跨 seed 置信区间、极端几何 finite 测试 |
| 天线与阵列 | 部分完成 | 方向图、orientation、极化端口、explicit array |
| 性能与部署 | 部分完成 | cold/steady/p95、峰值显存、大规模矩阵、wheel smoke |

## 3. 设计原则

### 3.1 API 简洁

- 配置只保留会真实影响计算的字段；
- 不生成 replacement matrix JSON 或通用 schema registry；
- metadata 只保留调试和复现必需信息：solver、seed、有效 components、深度、样本数、设备和计时；
- 不记录能够由结果 shape 或配置直接推导出的重复字段；
- 不允许配置被接受后静默忽略。

### 3.2 单一结果类型

公共路径结果统一命名为 `PathResult`，不使用 `PathResultV2`。

必须直接完成以下清理：

- [x] 将旧版本化结果类型重命名为 `PathResult`；
- [x] 将版本化结果实现收敛到 `result.py`；
- [x] 删除旧 flat-result adapter 和兼容分支；
- [x] 更新公开导出、类型注解、测试和文档；
- [x] 不保留旧类型 alias；旧调用方应直接升级。

`PathResult` 的最小公共契约：

- `valid` / `num_paths`；
- `tx_id` / `rx_id`；
- complex coefficient 或明确的 vector/Jones field；
- delay、path length、AoA、AoD；
- interaction type、position、normal、primitive、material 序列；
- `cir()`、`cfr()`、`taps()`、`filter_by_type()`；
- 明确的 per-pair path 上限、padding、mask 和稳定排序。

### 3.3 物理契约统一

- 路径内部统一使用复场，不混用场幅系数与功率系数；
- 每次交互显式处理局部极化基、Jones 操作和出射基重组；
- 自由空间传播因子只在规定位置应用一次；
- Path、Deterministic、MC Basic、BDPT 对同一解析路径应给出一致的幅度、相位和极化；
- 所有 grazing、退化三角形、零长度边和非有限输入必须有明确稳定策略。

## 4. 实施阶段

### Phase A：结果与公共 API 收敛（P0）

- [x] 完成 `PathResult` 的破坏性重命名和兼容代码删除；
- [x] 让真实 Path Solver 直接输出复场、角度和完整交互序列；
- [x] 在真实求解结果上验收 `cir/cfr/taps/filter_by_type`；
- [x] 删除未使用配置、重复 metadata 和只为旧接口存在的字段；
- [x] 所有超出支持范围的配置在 GPU allocation 前失败。

验收：公开 API 中不再出现 `V2`、legacy adapter 或静默降级路径。

### Phase B：拓扑与场计算闭合（P0）

- [x] Path 支持 `1..max_depth` 高阶镜面反射；
- [x] 支持最多一次衍射及其前后的受限反射组合（当前边界为恰好 `1R1D`，覆盖 R-D 与 D-R）；
- [x] 定义组合路径的 canonical ordering、去重与截断；
- [x] 建立 LOS、单反射、双反射、单透射、单衍射解析场景；
- [x] Path/Deterministic 比较复场、wrapped phase、极化分量和 delay；MC Basic/BDPT 按公开 capability 验收共同可观测的 power、finite 与 convergence，不伪造复场能力。

验收：增加深度会增加真实可构造路径，而不是只改变 padding 或日志字段。

### Phase C：BDPT 与随机求解稳定性（P1）

- [x] 将 BDPT scalar throughput 改为 Jones/complex3 + local frame；
- [x] 审计 forward/reverse PDF、Jacobian、delta event 与 MIS measure；
- [x] 用可枚举小场景检查估计无偏性；
- [x] 对 MC Basic、BDPT 和衍射执行跨 seed 压力测试；
- [x] 固定报告 mean、variance、confidence interval、finite ratio 和失败率。

验收：统计阈值预先固定；不以单 seed parity 作为通过条件。

### Phase D：材料、场景与阵列（P1）

- [ ] 完成频散材料、multilayer、transmission 和 rough scattering 的能量/互易性测试；
- [ ] 补齐 XML transform、UV、instancing、boundary/non-manifold edge；
- [ ] 实现 TX/RX orientation、方向图和极化端口；
- [ ] 实现并验证 ULA/URA synthetic array 与 explicit array；
- [ ] 覆盖多 TX/RX、precoding/combining 和全局旋转/平移不变性。

验收：解析小场景和 Munich/San Francisco 场景同时通过，不依赖旧包运行时。

### Phase E：性能、显存与部署（P1）

- [ ] 固化 analytic、three-cube、Munich、SF planar 和 terrain 基准；
- [ ] 分开记录 import、scene load、pipeline build、first solve、steady median/p95；
- [ ] 记录 persistent、peak temporary、output 和 OptiX build memory；
- [ ] 覆盖 1x1、8x1k、16x1k、128²/512² grid；
- [ ] 覆盖 MC 1k/1M/10M/100M samples 与 depth 0/1/3/5；
- [ ] 增加支持环境内的 wheel/import smoke 和清晰诊断。

验收：每个公开 Solver 都有数值、时间和显存预算；超预算在 CI 或发布门禁中明确失败。

## 5. 测试矩阵

### 5.1 正确性

- 解析 LOS、镜面反射、有限厚度透射、UTD 极限；
- 双反射非共面极化旋转；
- reciprocity、能量范围、全局旋转/平移不变性；
- 0 path、遮挡、切线、退化几何、极端材料参数；
- Path/Deterministic/MC Basic/BDPT 同路径一致性。

### 5.2 随机稳定性

- 固定 seed 可复现；
- 多 seed 均值、方差和置信区间；
- 无 NaN/Inf；
- 样本数增加时估计稳定；
- 高深度下 throughput、PDF 和 MIS weight 有界。

### 5.3 性能

- cold 与 steady 分开；
- GPU 同步边界明确；
- 不包含场景下载、参考实现启动或绘图时间；
- 输出时间、显存和场景规模原始数据，图表只作为展示。

## 6. 完成定义

只有同时满足以下条件，前向仿真器计划才可关闭：

- [ ] `PathResult` 单一公共结果契约完成，旧兼容代码清零；
- [ ] 支持范围内的拓扑、复场和极化闭合；
- [ ] BDPT/MC 通过预定义统计门禁；
- [ ] 材料、阵列和目标场景矩阵通过；
- [ ] 性能、显存、冷启动和部署预算通过；
- [ ] 生产源码不存在原 Channel/Sionna/Python RayD 热路径 fallback；
- [ ] 文档只描述当前真实支持能力，不承诺旧包兼容或迁移。

本计划完成后，旧 `channel` 包仍保持独立且不被修改。是否继续维护或停用旧包不属于 Channel Native 项目范围。
