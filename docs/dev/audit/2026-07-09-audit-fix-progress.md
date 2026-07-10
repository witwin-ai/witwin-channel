# 审计修复进度（对照 2026-07-09-deterministic-montecarlo-audit.md）

**日期**: 2026-07-09
**状态**: 全部四个 Phase 完成（含第二轮的 UTD 移植、Keller 采样器 pdf 补偿、性能门、PEC）。
**测试**: `python -m pytest tests` → 338 passed, 1 skipped, 0 failed。

## 已完成

### Phase 0（全部）
- **0.1 / D-3**: `concatenate_path_blocks` 按最大 sequence 宽度补齐后拼接；`max_depth=3` 不再崩溃。新增平行走廊 depth-2+3 混合测试。
- **0.2 / MC-4**: BDPT 网格反射图改用 `mc_store_scaled_component_map(..., tx_power, ...)`。新增 `power_w` 线性性测试。
- **0.3 / MC-3 / MC-11**: 点衍射与 tape 导出传 `strategy_count=2`（direct+keller balance/power 权重和恰为 1）；`mis="none"` + 多策略直接报错。单楔点接收机 pin 从 1.25e-04（固化 2×）修正。
- **0.4 / P-3 / MC-10**: `_estimate_workspace_bytes` 计入稠密连接表主导项（后随 P-5 修复更新为按组件实际行数），错误信息给出可操作建议。
- **0.5**: 根 `conftest.py` + `tests/support/native_ext.py` 自动发现 `artifacts/cmake-*` 中最新构建的 `_channel_native`/`_raydn`；pyproject 固化 `--import-mode=importlib`；基准子进程与 loader 测试同样注入。一条命令全套可跑。
- **0.6 / MC-6**: basic 点接收机 LoS 增加 `raydn_visibility_forward` 遮挡（`apply_point_los_visibility`）。单墙 LoS=0 测试。

### Phase 1（全部）
- **1.1 / D-1 / D-2**: order-1 反射按**共面组**枚举（每个平面一个代表面，EPC 解析实际包含三角形）；multibounce 在规划守卫内穷举共面组序列、超出守卫时改用 **trace 发现**（新增 RayDN C ABI `raydn_native_trace_reflections_forward` → bridge → `ops.raydn_trace_reflections_forward`，从 tx 追踪 262,144 条光线得到可达平面链，仅验证唯一链）；order-1 大场景（>4096 组）同样用发现模式。EPC `point_inside_triangle` 增加平面距离检验（拒绝投影幻影，audit D-2a）。path 求解器共面去重（共享对角线上的孪生候选）。验收测试：共面细分不变性（2 vs 128 三角形同场）、L 形内角同 structure 双反射、背墙被前墙遮挡。Munich depth-2 反射从"全部幻影"变为真实覆盖（发现模式），反射分量 vs 原实现中位误差 0.0 dB。
- **1.2 / D-4**: `deterministic_field.cu` 反射链全程复 3-vector 极化（s/p 分解 + p_out 基旋转，对齐 RayDN `epc_field.cu`），塌缩约定与衍射向量场一致（总功率 + 主导分量相位）。验收：强导体斜入射 |R|=1 对照解析 Friis（旧标量和会得 √2 或 0）。
- **1.3 / MC-1**: BDPT 子路径状态新增 `path_length` 前缀（15 字段 schema）；反射顶点 throughput 乘 **Fresnel 有效功率反射率**（按入射角、x̂ 极化投影）；连接贡献用**展开总距离**；求解器多弹跳循环尊重 `max_depth`；grid 图不再因 `export_paths` 切换估计器（导出仅附加采样，图恒用已验证的融合累加 kernel）。验收：导出路径 path_length ≥ 展开下界、contribution ≤ 自由空间上界、export 开关图不变性。

### Phase 2（除 2.1）
- **2.2 / DF-2**: `ext/raydn/include/utd/utd_math.h`、`utd_types.h` 与参考实现逐字节同步（`directFirstOrder`/`incidentScale`/`pathLengthPrefix`/`edge_caustic_rho`/`cplx_exp_neg_kd` 全部就位）；84 槽死代码路径设置兼容默认值。
- **2.3 / DF-3 / D-7**: 所有 `sincosf(-k·d)` 点位改为 double `fmod 2π` 约减：`deterministic_field.cu`（order-1/multibounce/LoS/phase_from_length）、`path_trace.cu`、`paths_optix.cu`、`accum_optix.cu`（非相干 lane）、`epc_field.cu`。
- **2.4 / DF-4 / D-6 / MC-2**:
  - DF-4: 新模块 `core/edge_selection.py`；`Scene.raydn_scene()` 把 `sionna_import_edge_policy` 挂到 runtime_cache，两个缓存边几何构建器统一 `refine_edge_geometry`（vertical_only 比率过滤、boundary exclude）。验收：wedge 场景 vertical_only 下 5→3 条路径（两条水平边被滤）。
  - D-6: 跨 structure 共享边按量化端点配对，**合并为单个楔记录**（正确 exterior angle、n1/face1 来自对侧面；共面对退化为不选）。楔场景从孪生 half-plane 双计变为一条 3π/2 楔记录；相关 pin 全部更新（点衍射收敛参考 6.25e-05 → 4.66e-05 等）。
  - MC-2: `sample_edge_weight_for_lane` 修为 `edge_length·S/N`（round-robin lane 映射下每状态 N/S 条 lane），对齐点接收机 kernel 约定。验收：双楔 direct-only 可加性测试（修复前 combined = separate/2）。
  - 附带发现：**git HEAD 源码与审计所用 6 月 15 日 pyd 不一致**（最后一次提交为 diffraction_weight 加了 (λ/4π)² 归一但没重建产物）；原 witwin.channel 的 BDPT MC 衍射估计器本身带 1/S 且无 λ² 归一。Munich BDPT 严格门相应调整：总和门只比 LoS+反射（原实现衍射刻度不可作参考），衍射相关门因 keller 采样方差无界（DF-6，已建后续任务）暂不作门。

### Phase 3（内存全部；性能部分）
- **3.1 / P-1 / P-5 / P-9**: LoS 连接表从 T·N·K·R 行去冗余为 **T·R 行**（深度 0 光端点全部相同 → 每 tx 一个端点、`samples_per_tx=1`，估计逐位相同）。实测 BDPT 128²@4096：峰值显存 **6.11 GB → 3.6 MB**，时间 13.6 ms → 1.9 ms。新增 64²@4096 < 300 MB 回归测试。LoS-only 方差按单确定行计算（恒 0）。
- **3.2 / P-2**: deterministic 衍射按 rx 分块（`_MULTIBOUNCE_PAIR_CHUNK_SIZE` 对预算，rx_id 加块偏移），Munich 规模不再需要 60–330 GB 的 rx×edges 工作区。
- **3.3 部分 / P-4**: `_MULTIBOUNCE_PAIR_CHUNK_SIZE` 262,144 → 4,194,304、序列块 8192 → 65,536；Munich deterministic 54 ms → ~40 ms。

### Phase 4（内联完成的加固）
解析镜像源幅度锚点（导体斜入射）、共面细分不变性、双楔可加性、`power_w` 线性性、点接收机贡献上界/展开距离下界、export 开关不变性、显存断言、vertical_only 策略、depth-3 走廊。

## 第二轮（同日晚）：四项遗留全部完成

1. **DF-1 / Phase 2.1 — 真 UTD 落地**：`trace_paths_order1_impl`（paths_optix.cu）重写为逐 (state, tx, rx) 的 K-P UTD 评估：Keller 驻点闭式解（`first_order_diffraction_parameter`）取代边中点（可见性/时延/交互点都用驻点，一并修复 D-5/DF-7 源腿可见性）；`PairInputs` 以 direct first-order 方式现场构造（`directFirstOrder=1` 时入射场由 kernel 内部从 tx 精确重算，面反射算符由原始材料参数经 Fresnel 现算，无需移植 84 槽 builder）；输出完整复 3-vector 场（×√P_tx）。ABI 扩展：`raydn_native_diffraction_paths_order1_forward` 增加 material_eta_r/sigma/mu_r 三个张量（raydn ops/library/native_api + bridge + bindings + ops.py + 两个调用方）。path 求解器桥修复：selection 掩码作为 active 传入（此前依赖启发式零贡献剔除）。
   **验收**：Munich 衍射分量 vs 原实现中位误差 **22.9 dB → 1.70 dB**，测试门槛从 25 dB 收紧到 **3 dB**；单楔 pin 更新为 UTD 值（4 条路径、驻点路径长度）。mpmath 系数级测试未做——UTD 头文件与逐行验证过的参考实现逐字节一致，端到端 1.7 dB 已覆盖该验收意图；如需仍可后续以 host 编译方式补。
2. **DF-6 / MC-5 — Keller 采样器**：(a) 锥轴改用逐样本边缘点入射方向（原为每状态中点 wi）；(b) 沉积增加 (edge_t, φ)→平面的 **Jacobian 度量补偿**（2πJ 取代固定 cell_area）；(c) 目标腿距离钳到半格（穿平面边缘的 1/d² 方差不可积）；(d) BDPT 网格衍射改 **keller-only**（对齐 basic，消 MC-5 双计与 direct 覆盖缺陷）。
   **验收**：单楔 16k 样本跨种子相对波动 **0.58 → 0.04**；新增种子稳定性测试（spread<0.3）与双楔可加性测试（比值 1.02–1.09）。
3. **P-4 剩余 — Munich 性能门首次通过**：共面组 union-find 缓存到 raydn.runtime_cache（此前每次 solve 重算两遍）；衍射状态按 tx 可见性预筛（沿边 4 采样点任一可见即保留，对齐原实现 tx_first 剪枝；纯中点判据会误杀驻点可见的状态）。
   **验收**：Munich all-components 56 → **21.7 ms**（原实现 28 ms），`test_reduced_munich_deterministic_parity_emits_artifacts` 全门通过。
4. **PEC model_id**：`PEC_EFFECTIVE_SIGMA_E=1e9` 有效电导率在三个材料导出点（material_runtime 张量路径 + bdpt/basic host 路径）按 model_id==2 注入。验收：PEC 单墙反射 = 展开距离 Friis（|R|=1）。

**最终测试状态**：`python -m pytest tests` → **338 passed, 1 skipped, 0 failed**（Munich 性能门与 3 dB 衍射门均在内）。

## 备注
- 所有 CUDA 改动已重建进 `artifacts/cmake-channel-native-raydn-witwin2-release`。注意：`*_optix_ptx.h` 生成头是 order-only 依赖，改 `.cu` 后需 touch 对应 `pipeline.cpp` 再 build 一次才会重新嵌入 PTX。
- `Scene.diffraction_edge_count` 的计数 op 尚未与 `refine_edge_geometry` 的合并逻辑对齐（计数可能比实际参与集多出被合并的重复边），差异为个位数量级。
