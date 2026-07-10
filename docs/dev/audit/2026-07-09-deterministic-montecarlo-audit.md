# Channel Native 审计报告：Deterministic 与 Monte Carlo 求解器

**日期**: 2026-07-09
**范围**: `witwin.channel_native.deterministic`、`witwin.channel_native.montecarlo.{basic,bdpt}`、`native/channel_native/kernels/*`、`ext/raydn`（衍射/反射相关部分）
**审计维度**: 数值正确性（重点：衍射路径）、求解时间、内存/显存消耗、测试体系有效性
**方法**: 4 路独立代码深读（deterministic 正确性 / 衍射专项 / MC 估计器 / 性能内存），逐公式对照原始 `witwin.channel` 实现与 Kouyoumjian–Pathak UTD 理论；另在 RTX 5080 (16 GB) 上实测求解时间与显存峰值，并运行完整测试套件。最高严重度发现均经过第二遍源码复核，若为实测复现会注明。

---

## 1. 结论摘要

| 维度 | Deterministic | Monte Carlo (BDPT) | Monte Carlo (basic) |
|---|---|---|---|
| LoS | ✅ 正确（Friis + e^{−jkd}，与原实现一致） | ✅ 正确（归一化无偏） | ⚠️ 点接收机忽略遮挡 |
| 反射 | ❌ 共面三角形重复计数(+3~6 dB)；surface group 语义错误 | ❌ 网格图漏乘 P_tx；点接收机/导出路径物理错误 | ✅ 网格路径正确 |
| **衍射** | ❌ **根本不是 UTD**——边缘中点各向同性启发式 | ❌ 多边缘低估 1/S；点接收机双计 2×；direct 策略覆盖残缺 | ❌ 同承 1/S 低估 |
| 求解时间 | ⚠️ 比原实现慢 ~2×（自家性能门失败）；多弹跳 launch 风暴 | ✅ 快（reduced 场景 1.6–15 ms） | ✅ 快 |
| 内存 | ❌ 衍射容量 = rx×edges 无分块，Munich 规模 60–330 GB | ❌ 稠密连接表 ≈91 B×samples×cells，128²@4096 实测 6.1 GB，65536 样本 OOM | ✅ ~1 MB 量级 |
| 鲁棒性 | ❌ `max_depth≥3` 直接崩溃（实测复现） | ⚠️ 显存 guardrail 低估 4 个数量级、从不触发 | ✅ |

**一句话结论**：两条求解路径的 LoS 与"骨架"（拓扑导出、排序、累加、网格布局、RNG、MIS 公式本身）质量很高、与原实现逐行对齐；但**衍射物理被系统性替换成了启发式**（且测试门槛被放宽到 25 dB 掩盖了这一点），反射存在多处会改变绝对幅度的缺陷，BDPT 的内存模型在真实规模不可用。当前测试为"自参照金标"（golden values 由同一份代码生成），结构上无法发现这些问题。

---

## 2. 实测数据

### 2.1 测试套件现状

```
pytest tests/deterministic tests/montecarlo tests/kernels --import-mode=importlib
→ 286 passed, 1 failed, 1 skipped (35.2 s)
```

- **失败项**: `test_munich_deterministic_parity` 的性能门 —— native 54.2 ms **慢于** original 27.9 ms（迁移计划要求 native 不慢于原实现，当前不达标）。
- 测试基建问题：各测试目录共享同名文件（`test_config.py` 等）且无 `__init__.py`，多目录合跑时 pytest 收集直接崩溃，必须 `--import-mode=importlib`。这意味着 CI 若分目录跑，从未有人一次性跑过全套。
- `_channel_native.pyd` 仅存在于 `artifacts/cmake-*` 目录，无 conftest 注入路径，环境未配好时整套测试静默不可运行。

### 2.2 求解时间与显存峰值（RTX 5080 16 GB，reduced Munich 合成场景，1 tx，warmup 后取 3 次中位）

**Deterministic（LoS+反射+衍射，max_depth=2）**

| 网格 | 时间 (ms) | 峰值显存 (MB) | 路径数 (其中衍射) |
|---|---|---|---|
| 32² | 20.7 | 3.5 | 7,793 (6,266) |
| 64² | 22.3 | 13.3 | 30,398 (24,746) |
| 128² | 23.2 | 54.4 | 122,868 (99,417) |
| 256² | 32.1 | 214.6 | 488,291 (396,355) |

单组件 @64²：LoS 1.2 ms / 反射 17.3 ms / 衍射 7.7 ms。时间几乎不随网格增长——**瓶颈在固定的 launch+同步开销，不在光线求交**（见 P-4/P-8）。

**max_depth=3：崩溃** —— `ValueError: blocks[1].primitive_sequence must have shape (1380, 2)`（见 D-3）。

**BDPT（LoS+反射+衍射，samples=4096，max_depth=3）**

| 网格 | 时间 (ms) | 峰值显存 |
|---|---|---|
| 32² | 1.6 | 382 MB |
| 64² | 4.4 | **1.53 GB** |
| 128² | 13.6 | **6.11 GB** |
| 64² @16384 样本 | 15.3 | **6.11 GB** |
| 64² @65536 样本 | — | **OOM**（申请超 20 GB） |

显存 ≈ **91 B × samples × 网格单元数**，随两者线性增长。外推 256² @4096 样本 ≈ 24 GB —— 消费级 GPU 全部不可用。而 `max_depth=1` 与 `=3` 的峰值显存、launch 数完全相同，印证 BDPT 点接收机/连接路径实际只生成 depth-1 反射（MC-1）。

**MC basic（samples=4096）**: 64² 2.3 ms / 0.67 MB；128² 2.4 ms / 1.36 MB —— 同等估计量下比 BDPT 省 3 个数量级显存，是内存安全的参照实现。

### 2.3 与项目自维护基线的关系

`docs/dev/perf/bdpt_native_baselines.md` 的所有数字都在 grid 4–32、samples 16–64 的微型规模上测得（且从不记录显存），因此 P-1/P-2 类内存爆炸在现有基准中不可见。冷启动 30.8 s（OptiX pipeline 编译）已知并单独报告，不计入本审计。

---

## 3. 正确性发现 —— 衍射（用户重点关注）

### DF-1 ｜ Critical ｜ deterministic 的"衍射"不是 UTD，而是各向同性功率启发式 【已复核源码】

`ext/raydn/src/torch_ext/diffraction/paths_optix.cu:161-193`（`path_weight`）：

```
contribution = P_src · material_gain · edge_length · wedge_scale · (λ/4π)² / (d_s² · d_r²)
```

- 无 D₁–D₄ 余切系数、无 Fresnel 过渡函数 F(x)、无 φ/φ′ 角度依赖、无极化、无阴影边界处理、无楔外侧检验（接收机在楔的"错误"一侧照样收到全额贡献）。
- 距离律是 `1/(d_s²·d_r²)`，而 UTD 是 `|D|²/(s·s′(s+s′))` —— 误差随距离在 dB 上近似线性增长。
- 衍射点固定取**边缘线段中点**（`paths_optix.cu:298-301`，另见 D-5），不解 Keller/Fermat 驻点：时延误差可达米级，且中点被遮挡 ≠ 真驻点被遮挡，可见性分类错误。
- 项目自己的奇偶校验测试印证了这一点：`test_munich_deterministic_parity.py:58-60` 对 LoS 要求 <1e-3 dB、反射 <1 dB，而**衍射只要求中位误差 <25 dB**。
- 关键事实：**完整的 K-P UTD 相干评估路径在 RayDN 中已经存在**（`accum_optix.cu:583-654` `run_coherent_utd_lane`），并且逐行对照原 DrJit 实现验证正确，但在 channel_native 里是死代码——没有任何调用方填充 `utd_*` 槽位。

### DF-2 ｜ High ｜ RayDN 的 UTD fork 相对参考内核已过期，高阶（带反射前缀）衍射态处理错误

`ext/raydn/include/utd/utd_math.h:1810-1854` 缺少参考实现（`channel/.../kernels/utd/utd_math.h`）中的 `directFirstOrder` 区分与 `incidentScale` 重标定：所有 `selectStationaryPoint>0.5` 的态被统一替换为源点球面波——反射前缀链的幅度/极化被静默丢弃；同时缺 `pathLengthPrefix`/`edge_caustic_rho` 象散扩展因子。一旦 DF-1 修复启用该路径，此问题立即成为高阶衍射的正确性阻塞。

### DF-3 / D-7 ｜ High ｜ 相位全程 float32，参考实现已明确修复过

原实现 `wave_math.py:26-37` 的 `unit_phase_neg_kd` 特意用 float64 做 `fmod 2π` 约减（注释写明"float32 乘积损失 ~k·d·2⁻²⁴ 相位，mmWave 距离上会移动相干零点"）；native 各处（`deterministic_field.cu:197,222,325`、`utd_math.h:424,1807` 等）直接 `sincosf(-k*d)`。28 GHz、500 m 时相位量化误差 ≈0.018 rad 且随距离线性增长——相干多径的零陷/峰值位置相对原实现漂移。`RAYDN_OPTIX_FAST_MATH=ON`（默认）进一步恶化。

### DF-4 ｜ High ｜ 边缘选择策略（vertical_only / boundary=exclude）未被任何求解器执行

`diffraction.cu:323-326` 的边缘几何 kernel 硬编码"boundary 恒选中、无垂直度过滤"，deterministic/basic/BDPT 三条路径全部直接消费该掩码；而 `Scene.diffraction_edge_count(policy)` 却诚实地执行策略过滤。后果：经 `scene_loader`（sionna 默认 `vertical_only`）导入的场景，报告的边缘数与实际参与衍射的边缘集不一致，策略配置静默失效。

### D-6 ｜ Medium ｜ 跨 structure 共享的几何边被导出两次，衍射贡献双计

两个 structure 共享一条几何边时，各自产生一条 boundary 半平面边（而非一条 interior 楔），单楔测试的期望值里就固化了这对孪生条目（entry 0 与 3 完全相同）。修复需按 (v0,v1) 容差去重并把两侧面配对成一个楔记录。

### MC 侧衍射（与 DF 互补，全部影响 BDPT/basic 的衍射输出）

- **MC-2 ｜ High ｜ 多边缘场景网格衍射图系统性低估 ~1/S** 【已复核源码】：`accum_optix.cu` 中 lane→state 映射为 `lane % state_count`（每个边缘态分到 N/S 个样本），但每 lane 的边缘度量权重是 `edge_length / N`（除以全预算 N，见 `sample_edge_weight_for_lane`）。每态总和 = f̄·L_s/S，全图 = (1/S)Σf̄L_s，正确值应为 Σf̄L_s。**加第二个完全相同的楔，衍射功率不增加。** 点接收机 kernel 用的是正确的 `edge_length·state_count/N`，印证约定本意。现有测试全是单楔场景，Munich 门槛是尺度不变的相关系数，全部无法发现。
- **MC-3 ｜ High ｜ BDPT 点接收机衍射双计 2×**：direct/keller 两个策略在点接收机 kernel 中是**完全相同的采样器**，各自完整归一化，solver 却传 `strategy_count=1` 使 MIS 权重恒 1，两份完整估计相加 = 2×。"maintained reference 1.25e-04" 把这个 2× 固化进了收敛测试。
- **MC-5 / DF-5 ｜ High/Medium ｜ BDPT 网格衍射的 direct 策略覆盖残缺 + 无 MIS 合并**：direct 策略确定性扫格 `cell = (lane/S) % C`，当 N_direct < S·C 时只覆盖前 ~N_d/S 个格子（例：N=4096、S=10、C=16384 → 覆盖 ~1%），且归一化按 (state,cell) 访问而非对预算；direct+keller 沉积**不带策略权重直接相加**，Keller 锥附近格子最多 2×。basic 通过 `direct=0, keller=N` 完全绕开了这些问题——BDPT 网格应同样 keller-only。
- **DF-6 ｜ Medium**：Keller 锥轴向从**边缘中点**算 β₀，长边远端采样点不满足锥条件，且 DF-5 的估计器不做 pdf 补偿，能量落点系统性偏移。
- **DF-7 ｜ Medium**：相干 UTD lane（修 DF-1 后会启用）不检验 tx→驻点段可见性，被部分遮挡的边会在 tx 不可见的点上被"点亮"。
- **DF-10 ｜ Info**：`_diffraction_sample_split` 硬编码 `suffix=0`，配置的样本预算有 1/3 被静默丢弃（无偏但浪费，且原实现的 D→R 后缀策略无配置面恢复）。

---

## 4. 正确性发现 —— 反射与 LoS

### D-1 ｜ Critical ｜ deterministic 一阶反射按共面三角形重复计数

≤4096 面时逐**面**枚举候选，EPC 有效性却按**组**包含判定：同一面墙的两个共面三角形产生同一个镜面点、双双通过校验，无任何去重。相干求和幅度 ×2 → **+6 dB**，随剖分密度线性恶化（N 个共面三角形 → ×N 幅度）。`test_reflection_multibounce.py:135-162` 把 4 条路径（2 条物理反射的孪生对）固化成了期望值。原实现有 `_canonicalize_prim_indices` 共面去重，这是 native 独有回归。

### D-2 ｜ Critical ｜ surface group 用 `surface_id`（每 structure 一个）而非共面组

原实现按"共面三角形 union-find"建组（`builder.py:256`）；native 用 structure 级 `surface_id`（loader 每合并网格一个 id；手工场景默认全部 `surface_id=0`）。连锁后果：
(a) `point_inside_triangle` 无平面距离检验 → 镜面点投影落在同 structure 另一面墙内也算有效 → **幻影反射**；
(b) `visibility_ignore_mode=1` 令路径段忽略端点所在**整个 structure** → 楼背面的"反射"不被正面墙遮挡；默认 `surface_id=0` 时反射段遮挡全场失效；
(c) >4096 面走分组导出：每 structure 只取一个代表面 → 多面楼的其余立面**一阶反射全部丢失**；
(d) 多弹跳 `adjacent_distinct=True` 禁止同 structure 不同墙的连续弹跳 → 内角双反射丢失。

### D-3 ｜ High ｜ `max_depth ≥ 3` 崩溃 【实测复现】

`concatenate_path_blocks`（`topology.py:63-69`）取第一个非空 block 的 `primitive_sequence` 宽度校验所有 block；depth-2（宽 2）与 depth-3（宽 3）块并存必然 `ValueError`。所有多弹跳测试只测 depth=2。修复：拼接前按 `max_depth` 填充。

### D-4 ｜ High ｜ 反射系数标量化：TE+TM 直接相加 + 全局硬编码 x̂ 极化

`deterministic_field.cu:143-156` 用 `coeff = r_te·e_s + r_tm·e_p`（标量和）代替向量合成 `√(|r_te e_s|²+|r_tm e_p|²)`：PEC 在绕射线 45° 方位可达 √2 倍幅度（**+3 dB 非物理增益**）或 −45° 伪相消，取决于任意全局轴。多弹跳每次弹跳重投影原始 x̂ 而非传播反射后极化，TE/TM 基旋转丢失。RayDN 自己的 `epc_field.cu:120-121` 就是正确的向量实现，可直接对齐。

### MC-1 ｜ Critical ｜ BDPT 点接收机/路径导出的反射估计器物理错误 【已复核源码】

连接贡献 = `bdpt_free_space_gain(P_tx, |rx−hit|, f)/N`：
- **tx→表面第一段距离完全不衰减**（throughput 恒 = P_tx）；
- **Fresnel 系数不存在**——`material.cu:75` 把 `face_gain` 硬编码 1.0，eps/sigma 解包后丢弃；有损材料 = 完美镜面；
- 只生成 depth-1，`max_depth>1` 静默忽略（与实测"深度不改变显存/launch 数"吻合）；
- 网格场景一旦 `export_paths=True`，反射从正确的累加 kernel **切换到这个错误估计器**，radiomap 随之改变。
1 km 外、距 rx 1 m 的反射体贡献 `P(λ/4π·1m)²` 而非 `|R|²P(λ/4π·1001m)²`——错多个数量级。测试只断言正值。

### MC-4 ｜ High ｜ BDPT 网格反射图漏乘发射功率 【已复核源码】

`bdpt/solver.py:387` 用 `mc_store_component_map`（无缩放），basic 用 `mc_store_scaled_component_map(..., tx_power, ...)`；BDPT 的 LoS/衍射分量含 P_tx，反射不含 → `power_w ≠ 1` 时 path_gain 混合了带功率与不带功率的分量。所有反射测试场景都用 `power_w=1.0`，故不可见。

### 其余

- **MC-6 ｜ Medium**：basic 点接收机 LoS 不做遮挡检验（穿墙）；网格路径和 BDPT 都做。
- **MC-7 ｜ Medium**：`streaming_planar` 累加策略偷换物理量（极化替换为 `vertical_iso`、LoS 折进反射图）——切"累加策略"开关不应改变被估计量。
- **MC-9 ｜ Medium**：basic 反射方向是无种子 Fibonacci 晶格：`seed` 无效（`make_cuda_generator(config.seed)` 创建即弃）、多次运行无法平均降误差、两极点权重有 O(1/N) 系统偏差；元数据却报告为 seeded MC。
- **MC-8 ｜ Medium**：方差估计器对异质样本块用全局 `samples_per_tx` 重构（含衍射的运行方差错误）；网格+反射无导出时方差静默全零而 `metadata["variance"]=True`。
- **MC-12 ｜ Low**：反射 `compact` 累加策略无分支处理，静默退化为 atomic（数值相同，配置项是空操作）。
- **D-8 ｜ Low**：`max_paths` 截断按 (rx,tx,…) 排序取前 N —— 丢的是"高编号接收机的全部路径"而非最弱路径；多弹跳还按 launch 顺序预截断。`guardrail_count` 永远为 0。
- **D-9 ｜ Low**：tx==rx 重合时 LoS 注入 `Pt·(λ/4π·1e−6)²` ≈ 6e9·Pt 而非跳过。
- **D-5 参见 DF-1**（衍射中点问题并入衍射章节）。

**已验证正确的部分**（两个 agent 逐公式对照原实现均通过）：LoS Friis 幅度与 e^{−jkd} 相位约定；Fresnel TE/TM 公式与主分支复根；多弹跳展开总长扩展因子与时延；镜像源 EPC 回代与逐段遮挡（在组正确的前提下）；相干/非相干累加语义与分量-总量一致性；网格 (tx,cols,rows) 布局与 cell 索引；稳定 LSD 基数排序；RNG splitmix64 分流（light/sensor/connection/diffraction 常量独立、无维度复用）；MIS balance/power 公式本身；BDPT 三种累加变体数值一致；按请求样本数 N 而非有效数归一化（无经典 valid-count 偏差）。RayDN UTD 核心数学（D₁–D₄、F(x) Boersma 12 项、阴影边界极限、L 参数、Keller 驻点闭式解、极点守卫）与原实现逐行一致——问题在于 deterministic 根本没接它（DF-1），以及 fork 落后于参考（DF-2）。

---

## 5. 性能与内存发现

### P-1 ｜ Critical ｜ BDPT LoS/连接路径物化全量 light×sensor 稠密表 【实测复现：6.1 GB @128²，OOM @65536 样本】

`bdpt_endpoint_connection_samples` 以 `max_paths=None` 分配 `light_count(=T·N·K) × R` 行 × 57 B（12 个字段）+ 可见性输入 25 B/行。默认配置（N=4096）+ 256² Munich radiomap = 2.68 亿行 ≈ **15.3 GB + 6.7 GB**。维护基准跑 grid 8–32 所以从未暴露。

### P-5 ｜ High ｜ 且这份开销是 N 倍冗余的

所有 light 端点都是**同一个 tx 位置**：LoS 对同一 tx→rx 段追了 N·K 次完全相同的 OptiX 光线（268M 条 vs MC basic 的 65,536 条），对一个本质确定性的 LoS 项。depth-0 只需连接 T 个唯一端点、MIS 权重乘 N·K，或直接复用 MC basic 的 LoS 图。

### P-2 ｜ Critical ｜ deterministic 衍射分配 `capacity = n_rx × state_count` 无分块

`topology.py:870-882` + RayDN op 内 17 个 capacity 尺寸缓冲 ≈ 97 B/行。Munich（51,650 边）× 65,536 rx ≈ **60–330 GB**。同文件的反射路径有 262,144 对/launch 的分块，衍射没有。

### P-3 / MC-10 ｜ Critical ｜ 显存 guardrail 公式漏掉主导项，1 GB 上限从不触发

`_estimate_workspace_bytes` 只算 `launch_entries·32 + maps`，P-1 场景估出 ~0.9 MB vs 实际 ~22 GB——差 4 个数量级，guardrail 形同虚设，用户看到的是 OOM 而不是可操作的错误。

### P-4 ｜ High ｜ deterministic 多弹跳 = 数万次 launch，每次强制流水线停顿

三重循环（depth × 8192 序列块 × rx 块 × tx），每次迭代 `deterministic_reflection_sequence_compact` 内含 thrust scan + D2H 拷贝 + `cudaStreamSynchronize`（`path_trace.cu:2609-2633`）+ thrust 临时 cudaMalloc/Free。100k 序列上限 + 65,536 rx → ~25,000 次迭代，纯 launch/stall 开销 >12 s。这（连同 P-8）解释了实测"时间不随网格涨、被固定开销钉死"以及 Munich 性能门失败（native 54 ms vs original 28 ms）。缓解：块尺寸 262,144→4M（工作区仍只 ~480 MB）、tx 折入批次、容量上界写出 + 每 depth 只读一次计数、cub::DeviceSelect + 预分配工作区。

### P-6 ｜ High ｜ BDPT "staged" 累加 kernel 是 O(cells × samples) 串行扫描

每输出 cell 一个线程遍历整个样本数组（`bdpt_connect.cu:859-905`）：65,536 cells × 2.68 亿样本 = 1.7e13 次顺序全局读——小时级。MC basic 的 staged 是真正的 sort-reduce，可直接搬。所幸 P-7 的 auto 启发式量纲错误（`samples/grid_cells` 而非每 cell 实际样本数）+ 硬编码 valid ratio，使 auto 永远选 atomic——两个 bug 相互掩护。

### 其余（Medium/Low）

- **P-8**：`<<<1,1>>>` 单线程 GPU kernel 做边缘选择计数/填充 + 每 tx 循环内调用（`diffraction.cu:551-633`）——毫秒级串行 + 每 tx 一次 stall；改 flags→scan→compact 并把 tx 不变量提出循环（BDPT 已这么做）。
- **P-9**：`export_paths=False` 时全量连接表仍保活到 solve 结束（只为方差诊断和一个 int 计数），`bdpt_concat_connection_samples` 还额外整表复制一次（2× 瞬时峰值）。
- **P-10**：字典序排序 = (6+W) 趟完整 `thrust::stable_sort_by_key`（W=3 时 9 趟 + face_groups 5 趟），每趟隐式同步 + thrust 临时分配；(tx,rx,component,depth) 都是小整数，可打包 64-bit 单键。
- **P-11**：field 数据 re/im→complex64→`.real/.imag.contiguous()` 往返 ≥3 次全量拷贝；`TopologyBatch` 内部保持平面 re/im、仅对外打包一次。
- **P-12**：EPC op 每 launch ~28 个临时分配 + SoA↔AoS 转换，在 P-4 循环里放大 ~25,000 倍。
- **P-13**：surface-group union-find 每迭代同步（≤512 次 D2H），計入 30.8 s 冷启动；每 16 迭代查一次 `changed` 即可。
- **P-14**：facade 逐参数 Python 校验在热循环内 ~1 s 级纯 CPU 开销；`component_counts`/`selected_edge_count` 等元数据 op 在 `diagnostics=False` 时也强制同步。
- **P-15**：基准从不记录峰值显存、规模停留在 grid 8–32 —— 这正是 P-1/P-2 长期不可见的原因。

**已存在且有效的防护**（不重复建议）：多弹跳 100k 序列上限与 262,144 对分块、`max_paths` 截断、共面组规划守卫；MC basic 完整的累加策略族（含 warp 聚合同 cell 原子与真 sort-reduce）；边缘几何按场景缓存；BDPT 网格反射/衍射（无导出时）走融合累加无逐对表；全库无 `torch.cat` 循环二次方拷贝。

---

## 6. 测试体系为何没接住这些问题

1. **自参照金标**：单楔/单墙测试的期望值由被测代码自己生成（D-1 的孪生路径、D-6 的孪生边、MC-3 的 2× 都被固化为"正确答案"）。
2. **尺度不变门槛**：Munich 奇偶校验用 Pearson 相关 ≥0.85 + 总和相对误差 ≤0.35（由 LoS 主导）——统一 ×0.5 或 ×1/S 的分量错误全部通过。
3. **量身放宽**：衍射中位误差门槛 25 dB（LoS 是 0.001 dB），`max_abs_delta_db < 250` 形同虚设。
4. **场景太特殊**：全部 `power_w=1`（藏 MC-4）、单楔单墙（藏 MC-2/D-1）、镜面点恰好落在共享对角线上（藏 D-1 的组/面差异）、只测 depth≤2（藏 D-3）。
5. **只测形状/正值**：BDPT 点反射（MC-1）只有正值断言；方差测试的关键断言包在 `if torch.any(positive)` 里，可空转通过。
6. **无系数级测试**：D₁–D₄、F(x)、a±/N±、阴影边界连续性、Keller 点解均无数值 pin —— UTD fork 的回归（DF-2/DF-3）不可检测。

---

## 7. 改进方案

按"先止血、再校物理、然后接真 UTD、最后提性能"排序。每项标注对应发现与建议验收方式。

### Phase 0 —— 立即修复（天级，互相独立可并行）

| # | 修复 | 对应 | 验收 |
|---|---|---|---|
| 0.1 | `concatenate_path_blocks` 前按 `max_depth` 填充 `primitive_sequence`/`material_sequence`/交互缓冲 | D-3 | 新增 `max_depth=3` 多弹跳测试（两墙走廊场景，depth 2+3 并存） |
| 0.2 | `bdpt/solver.py:387` 改用 `mc_store_scaled_component_map(..., tx_power, ...)` | MC-4 | `power_w=2.0` 的反射分量测试：期望值 = power_w × 单位功率值 |
| 0.3 | 点衍射与 tape 导出传 `strategy_count=2`（或点接收机放弃 split） | MC-3 | 单楔点接收机期望值减半后 pin 住；`mis="none"` + 多策略组合直接报错（MC-11） |
| 0.4 | `_estimate_workspace_bytes` 加入 `light×rx×57B + 25B` 可见性项 | P-3/MC-10 | 现有 1 GB 上限在 64²@16384 样本时提前报可操作错误而非 OOM |
| 0.5 | 各测试目录加 `__init__.py`（或 pytest.ini 固化 `--import-mode=importlib`）；conftest 注入 `_channel_native` 构建路径或在缺失时显式 skip 并给出指引 | 2.1 | 一条命令全套绿 |
| 0.6 | basic 点接收机 LoS 补 `raydn_visibility_forward` | MC-6 | 单墙点接收机 LoS=0 测试 |

### Phase 1 —— 反射物理正确性（1–2 周）

1. **共面组取代 surface_id**（D-1/D-2，最大的一项）：EPC 的组构建改用已有的 `_coplanar_face_groups`（union-find 结果），`point_inside_surface_group` 增加平面距离检验，`visibility_ignore_mode` 只忽略端点共面组而非整 structure；分组导出的代表面按共面组取；恢复内角连续弹跳（`adjacent_distinct` 以共面组为单位）。
   验收：两共面三角形墙 = 恰一条反射路径（对照镜像源解析解 pin 绝对幅度）；L 形楼内角双反射存在；背墙反射被前墙遮挡。
2. **向量极化贯穿反射链**（D-4）：`deterministic_reflection_sequence_field_kernel` 内部携带复 3-vector（对齐 RayDN `epc_field.cu` 的现成实现），末端取模或投影 RX 极化。
   验收：PEC 45° 方位单反射 |R|=1（当前 √2）；与原实现单平面场景 <0.1 dB。
3. **重写 BDPT 点接收机/导出反射估计器**（MC-1）：反射顶点携带传播幅度（入射扩展 × r_te/r_tm，复用 `reflect_field_vector`），连接用展开总距离，尊重 `max_depth`；或短期方案——点接收机路由到网格累加 kernel 的单 cell 特例，并禁止 `export_paths` 切换估计器。
   验收：点接收机 vs 同位置 1×1 网格一致性测试；1 km 反射体量级对照解析镜像源。
4. `streaming_planar` 对齐其余策略的极化与分量归属（MC-7）；`compact` 补分支或从配置移除（MC-12）；basic 方向晶格显式文档化为 quadrature 或按 seed 随机旋转（MC-9）。
   验收：跨策略数值一致性测试（当前缺失）。

### Phase 2 —— 衍射接真 UTD（2–4 周，用户最关心，依赖 Phase 1 的组修复）

1. **deterministic 衍射路由到相干 UTD lane**（DF-1/D-5）：用 `builders.py` 同构逻辑填充 `utd_*` 状态槽，启用 `run_coherent_utd_lane`；Keller 驻点闭式解已在 `utd_math.h:329-361` 验证正确，直接产出逐 (tx,rx) 驻点、时延与可见性。
   验收：单楔场景对照原 `witwin.channel` UTD 场 <0.5 dB；Munich 衍射门槛从 25 dB 收紧到 ≤3 dB。
2. **同步 UTD fork 与参考头**（DF-2）：补 `directFirstOrder`、`incidentScale`、`pathLengthPrefix`、`edge_caustic_rho`。
   验收：系数级测试——`diff_coeff_3d`/`f_utd_value` 对照 mpmath 高精度 Fresnel（rtol 1e-5），阴影边界两侧总场连续性数值测试。
3. **双精度相位约减**（DF-3/D-7）：移植 `cplx_exp_neg_kd`（double fmod 2π）到所有 `sincosf(-k*d)` 点位；评估 FAST_MATH 对衍射路径的开关。
   验收：28 GHz / 1 km 双径零陷位置对照 float64 参考 <1e-3 rad。
4. **边缘策略贯通**（DF-4）+ **跨 structure 边去重**（D-6）+ **per-sample Keller wi**（DF-6）+ **源腿驻点可见性**（DF-7）。
   验收：两 structure 共享边 = 恰一条衍射路径；`vertical_only` 下水平边不产出路径；两个相同楔 = 2× 单楔功率（同时验收 MC-2 的 `edge_length·S/N` 权重修复）。
5. **BDPT 网格衍射改 keller-only**（MC-5/DF-5，对齐 basic），或给 direct 策略随机化 cell + S·C 感知权重 + 真 MIS 合并；恢复或删除 suffix 预算（DF-10）。

### Phase 3 —— 内存与性能（2–3 周，与 Phase 2 可并行）

1. **BDPT 连接分块**（P-1）：sensor 维按 ~262k 行分块、逐块累加后释放；`export_paths=False` 且 `diagnostics=False` 时逐块丢弃（P-9），有效计数在累加时顺带算。
   目标：128²@4096 从 6.1 GB → <300 MB；256²@4096 可跑。
2. **LoS 去冗余**（P-5）：depth-0 只连 T 个唯一端点。目标：LoS 光线数 ÷4096。
3. **deterministic 衍射分块**（P-2）：复用反射的 rx 分块循环。目标：Munich 全规模 <2 GB。
4. **多弹跳 launch 合并**（P-4/P-12/P-14）：块尺寸提到 4M 对、tx 折入批次、容量上界写出 + 每 depth 单次计数读回、cub 预分配工作区、EPC 缓冲按恒定块容量复用、热循环 facade 校验瘦身。
   目标：Munich 性能门恢复（native < original，即 <28 ms）。
5. **staged 累加改 sort-reduce**（P-6）+ 修 auto 启发式量纲（P-7）；`<<<1,1>>>` kernel 改并行 compact 并提出 tx 循环（P-8）；排序打包 64-bit 键（P-10）；planar re/im（P-11）；union-find 批量迭代（P-13）。
6. **基准补内存维度**（P-15）：所有 bench 记录 `max_memory_allocated`，增加 256² 与 samples 65536 配置，基线文档同步。

### Phase 4 —— 测试与门槛加固（持续）

- 解析解锚点：镜像源单平面绝对幅度、双缝/单楔 UTD 对照 mpmath、自由空间双径零陷位置。
- 对称性/守恒测试：两相同楔 = 2×、共面细分不变性（把墙从 2 三角形剖成 200 个，反射场不变——直接守护 D-1）、点 vs 1×1 网格一致性、`power_w` 线性性。
- 收紧 Munich 门槛：衍射 25 dB → 3 dB（Phase 2 完成后）、反射 median → p95、删除 `<250 dB` 这类空门槛。
- 每分量非自参照 golden：由原 `witwin.channel` 在固定场景离线生成、入库为常量。
- 性能门增加显存断言（如 64²@4096 BDPT <300 MB）。

### 建议的优先顺序与量级

`0.1–0.6`（≈2 人日）→ `1.1/1.2/1.3`（反射可信，≈1.5 周）→ `2.1–2.3`（衍射真 UTD，≈2–3 周）→ `3.1–3.3`（真实规模可用，≈1 周）→ 其余按需。若只有一周预算：做 Phase 0 全部 + 1.1 + 3.1——分别消除崩溃、最大的幅度错误和 OOM。

---

## 8. 审计覆盖说明

- 未覆盖：`path.solver` 独立正确性（仅作为 deterministic 拓扑来源间接审计）、AD/VJP 路径（当前 primal-only）、OptiX BVH 构建正确性、`ext/raydtorch`（未被 channel_native 使用）。
- BDPT 反射顶点→rx 可见性段的 `kRayTMin` 自遮挡行为未追进 `visibility_optix.cu`（标注为未验证，风险低）。
- 实测环境：RTX 5080 16 GB / torch 2.10 / CUDA 可用 / Windows 11；测试与基准命令见 §2；基准脚本为一次性脚本（reduced Munich 合成场景），数字用于量级判断而非精确基线。
