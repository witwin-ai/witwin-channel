# Channel Native 可微分求解器计划

**状态：** Proposed / 独立于前向仿真器发布
**基线日期：** 2026-07-12

## 1. 目标

在不污染 Channel Native 前向热路径和公共 API 的前提下，为固定拓扑的场计算提供可验证的 JVP、VJP 和 PyTorch autograd 能力。

本计划独立于 [06-channel-native-full-replacement-gap-plan.md](./06-channel-native-full-replacement-gap-plan.md)。前向仿真器不需要等待 AD 完成，也不应为了未来 AD 保留重型 tape、metadata 或兼容结构。

## 2. 当前状态

- 四类公开 Solver 当前只允许 `ad_mode="none"`；
- capability 正确声明 `supports_ad=False`；
- 底层已有 `mc_los_path_gain_jvp` CUDA 原语；
- 尚无 Solver 级 JVP/VJP、PyTorch autograd registration 或端到端梯度；
- topology/visibility discontinuity 尚未定义梯度估计策略。

因此当前状态是：存在局部导数实验原语，但没有可用的可微分 Solver。

## 3. 设计边界

- topology discovery 与 differentiable field evaluation 分离；
- 第一阶段只承诺 fixed-topology 导数；
- visibility、遮挡切换、路径出现/消失不伪装成普通连续梯度；
- AD tape 按需创建，`ad_mode="none"` 时不得增加常驻显存或 kernel launch；
- 不复制一套前向物理公式，primal/JVP/VJP 必须共享同一约定和基础函数；
- 不继承旧 Channel 的 AD API，只设计 Native 自身的最小接口。

## 4. 第一版可微参数

优先级顺序：

1. TX/RX position；
2. frequency；
3. 连续 material parameters：`eps_r`、`sigma_e`、thickness；
4. antenna orientation/pattern parameters；
5. mesh vertices。

Mesh vertices 延后，因为它同时影响交点、法线、可见性和拓扑，不能与简单场参数使用同一验收标准。

## 5. 实施阶段

### AD-A：最小 fixed-topology JVP

- [ ] 定义独立的 fixed-topology field-evaluation 输入；
- [ ] 完成 LOS 对 TX/RX position 与 frequency 的 Solver 级 JVP；
- [ ] 完成单反射和单透射 JVP；
- [ ] 复用现有 `mc_los_path_gain_jvp` 或在统一接口下替换它；
- [ ] `ad_mode="none"` 与当前前向性能、显存保持一致。

验收：解析导数与中心有限差分在预定义容差内一致。

### AD-B：VJP 与 PyTorch autograd

- [ ] 为相同参数集合实现 VJP；
- [ ] 验证 JVP/VJP 对偶性；
- [ ] 注册 PyTorch autograd，仅保存 backward 必需张量；
- [ ] 支持 batch、多 TX/RX 和 complex loss；
- [ ] 记录 tape bytes、forward/backward 时间和峰值显存。

验收：`gradcheck`、解析梯度、有限差分和 JVP/VJP 对偶测试全部通过。

### AD-C：多交互复场

- [ ] 多反射 Jones/complex3 链式导数；
- [ ] multilayer/transmission 材料导数；
- [ ] UTD fixed-topology 导数及 shadow-boundary 排除区；
- [ ] rough scattering 的随机数固定与重参数化策略；
- [ ] 梯度方差和跨 seed 置信区间。

验收：每种事件先通过解析小场景，再进入 Munich/SF 场景。

### AD-D：几何与不连续性研究

- [ ] Mesh vertex 对交点、法线和路径长度的连续部分导数；
- [ ] 明确 silhouette/visibility discontinuity 的 estimator；
- [ ] 检查路径 birth/death 邻域的 bias 与 variance；
- [ ] 将不支持区域变成显式错误或诊断，而不是返回误导梯度。

该阶段属于研究能力，不阻塞前向仿真器稳定版。

## 6. API 草案约束

最终接口应保持最小化：

- `ad_mode="none" | "jvp" | "vjp"`；
- 只有完成端到端实现的 Solver 才接受 `jvp/vjp`；
- 导数输入与输出使用正常 PyTorch Tensor，不生成独立 JSON tape；
- metadata 只报告 `ad_mode`、tape bytes 和必要计时；
- 不支持的参数组合在 launch 前失败。

## 7. 必须测试

- LOS 距离和相位解析导数；
- Fresnel TE/TM 对角度与材料参数导数；
- multilayer thickness/frequency 导数；
- 单反射 TX/RX position 导数；
- JVP/VJP 对偶性；
- complex loss backward；
- batch 和多 TX/RX；
- topology 固定范围内的中心有限差分；
- shadow boundary、grazing 和路径切换处的明确失败语义；
- `ad_mode="none"` 的零额外 tape 与性能回归。

## 8. 完成定义

- [ ] 至少一个生产 Solver 支持 TX/RX position、frequency 和材料参数的 JVP/VJP；
- [ ] PyTorch autograd 端到端可用；
- [ ] 解析、有限差分和对偶性门禁通过；
- [ ] discontinuity 边界有明确语义；
- [ ] 时间、显存与梯度误差预算固定；
- [ ] 前向模式不承担 AD 的额外常驻成本；
- [ ] 通过以上条件后，才把对应 Solver 的 `supports_ad` 改为 `True`。
