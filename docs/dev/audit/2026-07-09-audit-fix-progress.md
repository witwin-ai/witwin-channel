# 审计修复进度（对照 2026-07-09-deterministic-montecarlo-audit.md）

**日期**: 2026-07-09
**状态**: Phase 0 全部完成；Phase 1 全部完成；Phase 2 除 UTD 移植（2.1）外完成；Phase 3 内存项完成、性能门剩余项移交后续任务；Phase 4 验收测试随各修复内联完成。
**测试**: `python -m pytest tests` → 335 passed, 1 skipped, 1 failed（唯一失败 = Munich deterministic 性能门，native ~40 ms vs original ~28 ms，见"未完成"）。

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

## 未完成（已移交后续任务 chip）
1. **DF-1 / Phase 2.1**（最大剩余项）: UTD 状态表 builder 移植 + `trace_paths_order1_impl` 改 Keller 驻点 + `compute_pair_contribution`；随后收紧 Munich 衍射门 25 dB → 3 dB、mpmath 系数级测试。头文件已同步、路线已在 chip 中详细给出。
2. **DF-6 / MC-5**: keller 锥采样器无 pdf 补偿，方差无界（单楔 16k 样本跨种子 20× 摆动）。
3. **P-4 剩余**: Munich deterministic 性能门（40 vs 28 ms）：`<<<1,1>>>` 边缘 kernel、multibounce compact 的逐迭代同步、9 趟字典序排序、re/im 往返。
4. **PEC model_id**: 反射 Fresnel kernel 不识别 PerfectConductor（eps=1/σ=0 → 反射≈0），审计范围外的独立发现。

## 备注
- 所有 CUDA 改动已重建进 `artifacts/cmake-channel-native-raydn-witwin2-release`。注意：`*_optix_ptx.h` 生成头是 order-only 依赖，改 `.cu` 后需 touch 对应 `pipeline.cpp` 再 build 一次才会重新嵌入 PTX。
- `Scene.diffraction_edge_count` 的计数 op 尚未与 `refine_edge_geometry` 的合并逻辑对齐（计数可能比实际参与集多出被合并的重复边），差异为个位数量级。
