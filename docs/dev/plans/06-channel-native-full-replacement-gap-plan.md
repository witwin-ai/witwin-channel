# Channel Native 完整替代原 Channel：功能缺口与验收计划

**状态：** Active / No-Go for unconditional replacement
**基线日期：** 2026-07-11
**代码基线：** `d39da66`（`main` 与 `master` 同步）
**目标：** 在不依赖原 Channel、Python RayD、DrJit、Mitsuba 或 Sionna 热路径回退的前提下，使 `witwin.channel_native` 在明确支持的场景中成为可验证、可维护、可迁移的主实现，并最终满足删除原 Channel 运行时依赖的条件。

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
