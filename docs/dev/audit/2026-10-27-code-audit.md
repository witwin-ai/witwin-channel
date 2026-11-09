# witwin.channel 代码审计报告（2026-10-27，HEAD 33cce0e）

## 1. 结论

代码的骨架是好的：原生 facade 的 autograd 元数（arity）一致，容量失败事务（`CapacityFailureState`）设计严密，consumer 合同的主要 ADR 机制（溯源检查、slot 批处理、宽带打包）在探针下都成立。
但有三类问题需要优先处理：

1. **物理结果静默出错**。至少 10 处在受支持的配置下给出错误数值且不报错，例如 Deterministic 相干求和的偏振塌缩、`gain` 在部分求解器上被丢弃、AD 模式下绕射用固定 +z 偏振、MC Basic 点接收机丢掉反射/绕射。
2. **梯度静默断开**。BDPT 网格接收机、Deterministic 网格绕射覆盖、若干 `detach()` 让梯度消失，而元数据仍声称可微。
3. **防护机制没有真正生效**。没有任何 CI 自动跑分层门禁；HEAD 已经让 quick 档（mypy）和 nightly 档（duplication）失败；测试夹具按修改时间挑选旧的 `.pyd`，本机全量测试 1071 个失败中 1069 个是 RayD 锁不匹配；contract-coverage 门禁只核对清单，不核对真实调用点，所以有 20 多个无生产调用者的绑定通过了门禁。

| 切片 | 健康分 | 主要问题 |
|---|---|---|
| 运行时 / 构建 / 打包 | 66 | SM 声明与发布 wheel 不一致；构建身份可被缓存误导 |
| 传播 consumer 合同 | 68 | 边界拒绝顺序、公共句柄不校验、能力声明与实现不符 |
| 枚举 CUDA 内核 | 58 | 偏振塌缩、第二套反射物理、非确定性 |
| 场景 / 材料 | 52 | 形变被抵消、凹角误判为绕射楔、gain 丢失 |
| 枚举管线 Python | 52 | AD 绕射偏振错误、网格绕射断梯度、大量隐式同步 |
| Monte Carlo CUDA | 52 | int32 溢出、紧凑输出非确定且静默截断 |
| 测试 / CI / 文档 | 48 | 门禁不自动运行、旧扩展被静默加载、历史测试占比高 |
| Monte Carlo Python | 42 | 网格/混合接收机行为错误、能力记录错误、无调用者 facade |

加权整体约 **55/100**。

## 2. 方法与覆盖范围

- 8 个子代理并行静态审读全部生产代码（Python 约 4.6 万行，原生约 4.1 万行）及 owner 文档和 ADR。部分发现用纯 Torch 探针或标量脚本复现（标为“已运行确认”）。
- 我本人逐行复核了下表中标 ✔ 的发现。
- 运行了 `ci/run_ci_tier.py quick` 的全部 17 个门禁（不中断）：16 个通过，`quick.mypy` 失败。
- **没有做的**：没有构建扩展，没有跑 GPU 测试套件；本机的测试夹具会加载一个过期的 `.pyd`，所以 GPU 端到端结果在本次审计中不可用。所有标为“推断”的发现需要在 GPU 上复现后再修。

置信度记号：✔ 我已复核源码；R 子代理运行确认；C 读代码确认；P 推断（依赖输入或部署条件）。

## 3. 优先修复清单（功能缺陷，按严重度）

| # | 严重度 | 位置 | 缺陷 | 后果 | 置信 |
|---|---|---|---|---|---|
| 1 | 高 | `native/channel/kernels/deterministic.cu:143` `collapse_field_vector` | 相干求和前把场矢量塌缩为“总幅度 + 最大分量的相位”的标量 | 正交偏振路径错误干涉，功率在 0 到 4\|E\|² 之间摆动；接收机移动 1 mm 可能翻转主分量，地图和梯度出现跳变 | ✔ |
| 2 | 高 | `witwin/channel/propagation/fields.py:613`，RayD `src/diffraction/wedge.cu:284` | `ad_mode` 为 jvp/vjp 时绕射场由 RayD 内核重算，内核硬编码 +z 发射偏振 | 非 z 偏振发射机：AD 模式与 `none` 模式的绕射原值不同，梯度是错误场的梯度 | ✔ |
| 3 | 高 | `witwin/channel/deterministic.py:870` | 单个 `ReceiverGrid` 时用 Torch 侧 `\|E\|²` 覆盖绕射功率；该张量无梯度、未做接收偏振投影、包含 `max_paths` 已丢弃的行 | 非相干 + vjp 下绕射梯度被抵消；网格与等价点接收机结果不同 | ✔ |
| 4 | 高 | `witwin/channel/materials.py:58`，`native/channel/kernels/montecarlo.cu:821` | `face_material_tensors` 不传 `gain`，内核写死 `face_gain=1` | 确定性反射、绕射、散射忽略 `gain`；Path 场与 MC Basic 使用 `gain`，求解器之间静默不一致 | ✔ |
| 5 | 高 | `witwin/channel/scene/compiler.py:784` | 有形变时，按形变后的包围盒重新居中 `recenter=True` 的网格 | 整体平移的形变被完全抵消；只抬高屋顶会让地基下沉一半 | ✔（R） |
| 6 | 高 | `witwin/channel/scene/resources.py:381` | 合并两条边界边时外角 `2π − arccos(−n_a·n_b)` 恒 ≥ π，`> π` 的拒绝条件永不触发 | 分开建模的两面墙的内角被当成 270° 绕射楔，UTD 加入虚假绕射场 | ✔ |
| 7 | 高 | `witwin/channel/montecarlo/basic.py:1575` | 点接收机只把 LoS 交给 `mc_point_component_power`，原生侧把反射/绕射置零 | `components` 含反射/绕射时点接收机结果只有 LoS，元数据仍声称可用 | ✔ |
| 8 | 高 | `witwin/channel/montecarlo/bdpt.py:4116` | 网格累积矩阵先经过无 autograd 的原生 `bdpt_los_component_maps_from_matrix` | BDPT + 网格 + vjp：`path_gain.requires_grad=False`，`backward()` 报错；jvp 切向量静默为 None | ✔ |
| 9 | 高 | `basic.py:1374`，`bdpt.py:3760` `first_receiver_grid` | 接收机集合只取第一个网格 | 网格 + 点或两个网格时，其余接收机从结果中消失；BDPT 的混合拒绝分支不可达 | ✔ |
| 10 | 高 | `native/channel/kernels/bdpt_connect.cu:368` | 拓扑行偏移 `static_cast<int>(index * 4)` | 连接行数 > 2²⁹（如 1e5 光子路径 × 100×100 地图）时越界写 | ✔ |
| 11 | 高 | `bdpt_connect.cu:680, 935` | 紧凑输出用 `atomicAdd` 取槽位；`max_paths` 时取 `min(max_paths, valid)` | 行顺序每次运行不同；`max_exported_paths` 静默截断成“抢到原子的那些行”，违反全有或全无与稳定顺序 | ✔ |
| 12 | 高 | `witwin/channel/capabilities.py:69` 及 `docs/dev/montecarlo/README.md` | 能力记录写 BDPT 只有 `none`、MC Basic 排除 scattering；代码接受 BDPT jvp/vjp 和 MC Basic scattering AD | 每个 Result 的 `metadata["semantic_capabilities"]` 都错误；按能力做门控的调用方会做错决定 | ✔ |
| 13 | 高 | `witwin/channel/deployment.py:17` | `DECLARED_SM_ARCHITECTURES=(75,80,86,89,120)`，发布 wheel 含 70/87/90/100/101 SASS | Jetson Orin（SM87）、H100、B200 上 `require_supported_runtime()` 拒绝实际可用的 wheel | ✔ |
| 14 | 中 | `native/channel/kernels/bdpt_paths.cu:136-205` | 散射后吞吐量的镜面缩放用单界面、固定 x̂ 偏振的反射率，不计板厚，gain 按功率线性 | `max_scattering_order > 1` 时散射 NEE 功率与场输运不一致（gain=0.5 时差 2 倍） | P |
| 15 | 中 | `bdpt_connect.cu:1437`；`montecarlo.cu:817`；`kirchhoff.cu:887` | 方差核不查 `tx` 上界；设备端面材料 id 不查范围；各向异性 Kirchhoff 表只校验 `nti==nto` 未校验 `npi==npo` | 越界原子写 / 越界读，发布带 `valid=true` 的垃圾材料 | ✔（前两项） |
| 16 | 中 | `witwin/channel/scene/compiler.py:813` vs `scene/endpoints.py:367` | 结构的 `RigidMotion.rotation` 绕世界原点旋转，端点的旋转绕自身 | 同一个运动作用于车和车载天线后二者分离 | R |
| 17 | 中 | `compiler.py:980` | 材料有 `layers` 时材料级 `dispersion` 被丢弃，但仍标记为频率相关 | 4 GHz 下应为 2.0 的 eps_r 存成 4.0，不报错 | R |
| 18 | 中 | `compiler.py:1025`，`1122` | `scattering_coefficient`、`xpd_coefficient` 写入 store 后无任何消费者 | 用户设置后结果与 0 相同，属于被禁止的静默降级 | C |
| 19 | 中 | `compiler.py` 编译缓存 + `_kirchhoff_resources_cache` | 缓存的 CompiledScene 持有一次性 autograd 图 | 第一次 `backward()` 后再次 solve 同一世界，报 “backward through the graph a second time” | R（store 级） |
| 20 | 中 | `witwin/channel/interactions/scattering.py:446` vs `:2920` | 枚举散射与 MC 求解器对各向异性粗糙度轴的切向量定义相差 90° | 各向异性粗糙地面：Deterministic 与 MC/BDPT 的散射瓣互相旋转 90° | R |
| 21 | 中 | `witwin/channel/propagation/consumer.py:3630` | 宽带分辨率检查只用 f_ref 处的 float32 ULP | 跨 2 的幂边界的频率网格通过检查，两列在同一个 float32 频率上发射 | R |
| 22 | 中 | `consumer.py:3630-3717` | 不拒绝 `f_ref + df ≤ 0` | 原值路径在原生工作后才报错；AD 路径以 0 或负频率启动原生 | R |
| 23 | 中 | `consumer.py:1227` `FixedTopologyBucket` / `PreparedFixedTopology` | 公共句柄不校验分量与行覆盖 | `component="transmission"` 的桶被当成反射重放；未覆盖的行以 `row_valid=True` 和零载荷发布 | R |
| 24 | 中 | `consumer.py:302, 362` | `ad_mode="none"` 时带 `requires_grad` 或对偶的输入被静默剥离（离散极化路径会拒绝同样情况） | 求和损失静默缺项 | C |
| 25 | 中 | `native/channel/kernels/coupled.cu:2635` | D-D 二分法的证书失败用 `CUDA_KERNEL_ASSERT`；`evaluated_paths.cu:488, 1860` 用断言检查 id 范围 | 开启断言时整个进程 CUDA 上下文被毒化；关闭断言时静默丢行或写错 `pair_offsets`，都不经过失败状态 | P |
| 26 | 中 | `witwin/channel/path.py:734` `filter_by_type` | 过滤后的结果丢失 `field_xyz`、`field_direction`（被填零） | 过滤后 `a` 正确但场为零 | C |
| 27 | 中 | `native/channel/kernels/paths.cu:3878, 4009` | 面组成员数组按 `group_count × max_group_size` 稠密分配，RayD 侧 int 索引 | 城市场景单个 10 万面地面 + 3 万小组 → 约 12 GB，或 int32 溢出越界 | P |
| 28 | 中 | `native/channel/tensor_checks.h` 及 Python 校验 | 不拒绝也不解析复数张量的惰性共轭位 | 损失经 `h.conj()` 回传时梯度虚部符号错误 | P |

另有若干低严重度的功能问题，见第 4 节各切片。

## 4. 各切片补充发现

### 4.1 运行时 / 构建 / 打包（`runtime.py`、`deployment.py`、`CMakeLists.txt`）

- `capacity.cu:26-66`：`capacity_failure_state_create` 与 `capacity_failure_terminal_check` 没有 `CUDAGuard`，终端检查也不查启动错误。多 GPU 下终端检查可能静默不运行，溢出变成“成功的零结果”（P）。
- `CMakeLists.txt:126-145`：`CHANNEL_CUDA_GENCODE_FLAGS` 决定真实编译，但写入 `build_info` 的是 `CHANNEL_CUDA_ARCHITECTURES`，二者不做一致性校验，且 gencode 缓存项是粘性的。
- `CMakeLists.txt:39-47`：自己 FORCE 写入的 `CMAKE_CUDA_ARCHITECTURES`，下次配置时被当成用户显式值；改 `TORCH_CUDA_ARCH_LIST` 后重配置直接 FATAL。
- `CMakeLists.txt:405-463`（commit f8575a0）：cxx_abi 与 torch_version 取自 Python 的 torch，链接的却是 `find_package(Torch)` 找到的 libtorch，二者不校验相同。
- 脏工作树的构建指纹只记录 HEAD SHA + dirty 布尔值，不按内容寻址；同一提交上两次不同的脏构建指纹相同，开发者覆盖无法识别旧构建。
- 导入时不检查当前 GPU 的 SM 是否在 `build_info` 的架构列表内（只有性能测试调用 `require_supported_runtime`），不支持的 SM 会在第一个内核处才以通用错误失败。
- `build_info.cpp:41`：`optix_available` 编译期恒为 True，而 RayD 可能在场景构建时选纯 CUDA 后端，结果元数据会误报 OptiX。
- `deployment.py:191` `pipeline_cache_key` 缺少 `topology_version`（四个版本域之一），且 NaN/inf 频率能通过检查。
- `runtime.py:214`（mypy 报错）：zip 形式的分发会让 `Path(zipfile.Path)` 抛 `TypeError`，逃出 `ExtensionLoadError` 合同（R）。
- `rayd/resource.h:11`：static_assert 检查 API 8，报错文字写 6。
- `witwin/__init__.py`：Channel wheel 用 pkgutil 风格的命名空间 `__init__`，其他 witwin 子包是 PEP 420，混用命名空间风格不受 PyPA 支持。

### 4.2 场景 / 材料（`scene/`、`materials.py`）

- `resources.py:645`：可微 Kirchhoff 表的活性门控漏掉 `mu_r`，只有 `mu_r` 需要梯度时被静默剥离。
- `resources.py:871` vs `:1270`：两个都叫 `MAX_RMS_SLOPE=0.5` 的门限度量的是不同量（单轴 vs 总斜率），同一统计量一个通过一个拒绝。
- `resources.py:1523`：`height_scale_m` 为张量时 `float()` 导致主机读取并静默断开梯度。
- `compiler.py:572, 580`；`resources.py:181`：派生缓存键写死 `torch.device("cuda")`，多 GPU 下表建在当前设备而不是场景所在设备（P）。
- 编译注册表保留 32 个 CompiledScene（各带 BVH）；每次未命中扫描 32 个候选 5 次，构造包含逐面 `primitive_ids` 的身份元组，为 O(32×F) 主机工作。

### 4.3 枚举管线 Python（`interactions/`、`propagation/`、`path.py`、`deterministic.py`）

- `path.py:928-967` `from_flat`：按 `rx*num_tx+tx` 重排，但 `field_xyz[result.valid]=...` 按原顺序散布。现在所有生产者都已是接收机优先顺序，所以暂时正确；没有任何断言保护这一不变量（P）。
- `path.py:879` `endpoint_angles`：Torch 的 acos/atan2 在天顶方向导数为 inf/NaN，vjp 下 NaN 梯度流入位置（P）。
- `deterministic.py:927`：`truncated` 诊断与理论候选数比较，`max_paths=10**9` 时也报告截断。
- `enumerated.py:110`：`EnumeratedPathConfig` 声称是四字段视图，引擎实际通过带默认值的 `getattr` 再读 7 个字段，属性拼错时静默取默认。

### 4.4 Consumer 合同（`propagation/consumer.py`）

- 准备好的固定拓扑路线上，`interaction_normals` 携带活的梯度图，但能力矩阵声明它不可微（`consumer.py:2987`）。
- `fixed_topology_row_validity_components` 含 `los`，原始 `PropagationTopology` 路线却不做可见性重测且返回 `row_valid=None`（“全部有效”），调用方无法从按分量的记录里发现按路线的差异。
- AD 账本漏记 `field_project_complex3_ad` 等启动，离散路线与准备路线计数不一致（`consumer.py:2523-2572`）。
- `prepare_fixed_topology` 报告的同步次数少于实际（`torch.unique`、`nonzero` 内部同步），R：报告 5 次，实际 7 次。
- `TimeVaryingRequest` 是 `frozen=True` 默认 `eq=True`，含一维张量字段，比较即抛异常（`consumer.py:4124`）。
- `evaluate_time_varying` 在 `reevaluate` 的预检前先做 O(K·T) 的 slot 复制，削弱了“任何原生工作前拒绝已移动世界”的承诺。
- 生产合同检查用裸 `assert`（约 11 处），`python -O` 下被去掉。

### 4.5 Monte Carlo Python（`montecarlo/`、`interactions/scattering.py`）

- `bdpt.py:4164`：诊断方差把单位质量的确定性行当成 N 次 MC 抽样，没有 MC 噪声时也报告方差约等于均值²。
- `basic.py:637`：MC Basic 反射用无种子的 Fibonacci 格点，`seed` 无效；格点含两极点且等权，有 O(1/N) 求积偏差。
- `bdpt.py:144, 225`：`receiver_strategy="point_sphere"`、`mis`、`power_heuristic_beta` 实际不起作用（每个连接都是单策略，权重恒为 1）；能力记录与 README 声称的 `diffraction_mis_strategies` 不存在。
- 四处静默断开梯度：`basic.py:1576` 的 `los.detach()`；`bdpt.py:3609/3632` 相干块不重新挂接 `tx_power`；`bdpt.py:2876` 声称 `phase_screen_heights` 可微但从不使用相位屏；`scattering.py:522` `float(tx_power[i])`。
- `bdpt.py:4274`：`metadata["scattering"]["max_scattering_order"]` 写死为 1。
- 种子复用：BDPT 方向种子 `seed + 65537*tx` 在 (65537,0) 与 (0,1) 碰撞；`bdpt_support.cu:53` 的种子键是线性积异或，不是单射。

### 4.6 CUDA 内核（两片合并）

- 第二套反射物理：`deterministic.cu:50-133` 用固定世界 +z 偏振、归一化、无板厚的 Fresnel；Path/consumer 走 RayD `field_transport.cuh` 的 slab Fresnel。同一场景两条路线反射场不同（C）。
- 非确定性：面组成员槽位用 `atomicAdd`（`paths.cu:3271`），共享边上的命中在不同运行中落到不同图元，进而改变规范排序与去重；主值相干格求和（`deterministic.cu:891`）与多处 VJP 用 float 原子加。成员已在 `order` 中连续，槽位可直接取 `index − group_start`。
- `evaluated_paths.cu:275` `shortest_winner_kernel`：只有一个有效候选时 NaN 行成为胜者；两个以上时含 NaN 的组被静默丢弃，不置失败位。
- `mc_transmission.cu:453`：墙透射率为 NaN 时行仍以 `penetrated=true` 发布 NaN 功率，不置失败位。
- `bdpt_paths.cu:595`：BDPT 透射子路径不检查薄板几何模式和 CSR 范围；`layer_count=0` 的普通材料在 BDPT 中透明、在 MC Basic 中阻挡。
- int32 扫描：`paths.cu` 中多处 `thrust::exclusive_scan<int>` 缺少 `evaluated_paths.cu:574` 那样的 `count <= INT_MAX` 守卫（P）。
- 多 GPU：除少数入口外几乎所有 host 入口都不设 `CUDAGuard`（P）。
- `los.cu:599`：在默认流上同步 `cudaMemcpy`，不用调用方当前流。
- `kirchhoff.cu:824`：每次求解创建/销毁 cuSOLVER 句柄，忽略 `info`，gesvd 不收敛时静默返回垃圾梯度。
- 性能：`mc_transmission.cu:644` 共享层 VJP 只开 3L+1 个线程串行遍历所有行；`bdpt_connect.cu:1363, 2010, 2079` 每个输出格扫描全部样本，O(tx·rx·samples)。

## 5. 架构策略违规（对照 CLAUDE.md）

CLAUDE.md 的规则是“热路径只在原生 CUDA；除 ADR-032 紧凑计数外不得有隐式同步；`kernels` 是所有原生 facade 的唯一归属；不得有无生产调用者的绑定”。实际偏离如下。

**隐式主机同步**。生产代码中有 13 处 `.item()`、30 处 `nonzero`、4 处 `synchronize()`、多处布尔掩码索引，大部分在 ADR-032 预算之外：
- `propagation/topology.py:98, 105` 每次枚举求解都做布尔掩码赋值，R：繁忙 GPU 上阻塞主机 714 ms，换成 `masked_fill_` 只要 0.1 ms。这是最便宜的修复。
- `path.py:556` 第二次计数 D2H（`counts.max().item()`）以及 argsort/bincount/`repeat_interleave` 在 Torch 里重做原生已给出的 `pair_offsets`。
- `deterministic.py:638, 641, 845` 只为元数据计数而同步，`diagnostics=False` 时也执行。
- `bdpt_connect.cu:871-933`、`montecarlo.cu:518` 原生侧有不在账本上的 `cudaStreamSynchronize`。
- BDPT 每个发射机每次反弹若干次同步（`bdpt.py:2271, 2339, 2384, 2764`）。
- `kernels/topology.py:2110` 丢弃原生报告的 D2H 计数并写死 `1` 和 `8`，所以“实测”的 ADR-032 遥测其实不是测量值。

**Torch 热路径数值**：
- `propagation/fields.py:736` 在 Torch 里除去源幅度，而原生已有 `field_source_amplitude_scale`。
- `fields.py:189, 849` 的边轴归一化与 Keller 投影、`path.py:879` 的角度、`geometry.py:294` 的遮挡盒都是 Torch 几何。
- `interactions/scattering.py:3367-3547` 的 `scattering_map_matrix` 在 Torch 中做三角形面积、`torch.multinomial` 采样、cos/r² 辐射度量，并由 Torch autograd 求导，超出了 ADR-010 允许的 MC 事件胶水范围。
- `scene/resources.py:333-404` `refine_edge_geometry` 在 Torch 中计算绕射楔几何（且含第 3 节 #6 的数值错误）。
- `scene/resources.py:1515-1719` 生产包里提供只有测试使用的相位屏 Torch 物理实现，并列入 `__all__`。

**facade 归属**：
- `montecarlo/bdpt.py:410-852` 直接通过 `_required_native_op` 调用 9 个原生符号，其中 `bdpt_accumulate_connection_samples` 有两个 Python 所有者且校验不同。
- `consumer.py:1738-2224` 直接持有 `evaluated_paths_compact_finalize` 等符号的前向，而反向与 JVP 在 `kernels/topology.py`；Path 求解器为了自己的紧凑化导入“求解器无关”的 consumer。

**无生产调用者的绑定**（违反“没有 caller-free binding”）：
- `kernels/montecarlo.py` 中 12 个 facade（如 `bdpt_mis_weights`、`bdpt_los_export`）；
- `paths.cu` 中 7 个原生 op（`path_concat_vec3`、`path_filter_block` 等）；
- `bdpt_face_material_tensors(_from_host)`；
- `deterministic_sort_order` 只由死代码 `propagation/topology.py:70-189` 调用。

contract-coverage 清单仍给这些符号列出端到端调用者，门禁只核对清单，所以全部通过。

**主机侧每次求解的 H2D**：`MaterialStore`、`AssignmentStore` 在编译时建在主机上，每次求解多次 `.to(device)` 上传，包括 O(F) 的 `face_material_id`（`materials.py:45-57` 等）。

## 6. 可维护性

- **超大文件**：`consumer.py` 4342 行、`bdpt.py` 4405 行、`kernels/fields.py` 3756 行、`kernels/montecarlo.py` 3753 行、`interactions/scattering.py` 3546 行；原生 `paths.cu` 4126 行、`fields.cu` 3523 行。超长函数如 `bdpt._transmission_sampled_connection_samples` 309 行、`basic.solve_pipeline` 292 行；`basic.py:1063` 有一个 41 个位置参数、含 11 个裸 `None` 的原生调用。
- **拼接的翻译单元**：多个 `.cu` 由旧 TU 拼接而成，靠宏改名避免冲突（`#define kBlockSize kMcSanitizeBlockSize`、`#define kPi kCompactionPi`、`#define zero_filled ...`）。宏会静默改写后续头文件里的同名记号，这是潜在的错误源。
- **平行重复**：`mc_*` 与 `bdpt_*` 两套 facade 指向同一个内核；三个发射机打包函数；RNG 辅助函数两份；x̂ 横向代理权重四份（`bdpt_paths.cu`）；光速三种写法；`compiler.py` 私有的 `require_tensor` 与 `runtime.require_tensor` 重复；`orientation_matrix` 是 Core 函数的手抄副本。
- **ABI 形状**：BDPT 以“张量字典 + 12/16 元位置元组”跨边界，每个桥接文件手写原型；返回类型不参与名字修饰，元组元数漂移时仍能链接，行为未定义。
- **死代码**：`GeometryStore.edge_param_range`、`kirchhoff_tables`（标注 “Compatibility view”，违反无兼容层规则）、`_validate_ad_readiness` 空函数体、`basic.py:817, 1022` 不可达分支、`propagation/geometry.py:120, 433, 441` 只被测试保活、生产代码中检查 `uses_rayd_native` 的死分支（该值编译期恒为 1）。
- **源注释被机械改写损坏**：如 “zero-the penetration implementationower rows”（`deterministic.py:142`、`scattering.py:147`）、“forward-mode liveness liveness”、截断的文件头句（`enumerated.py:2`）、`:func:` 前缺空格、2273 行文档字符串续行被压成一个空格缩进（commit c60367f）。同时仍有 52 处 ADR/plan 编号残留在字符串字面量里（门禁只查注释和文档字符串）。

## 7. 测试、CI 与文档

**CI 不自动运行任何分层门禁**（高，✔）。Channel 仓库只有 `publish-witwin-channel.yml`，触发器是 `release` 和 `workflow_dispatch`，不调用 `run_ci_tier.py`；monorepo 的 `pytest.yml` 不测 Channel。结果是 HEAD 提交 33cce0e 引入的 mypy 错误和过期的 duplication 账本无人发现。

**测试夹具静默加载旧扩展**（高，✔）。`tests/support/native_ext.py:27` 按修改时间挑选 `artifacts/cmake-*/_channel*.pyd`，并自行设置三个开发者覆盖环境变量。本机挑中了 2026-07-29 的构建（RayD 0.8 切换之前），全量测试 1071 失败 / 1449 通过，其中 1069 个是 RayD 锁不匹配。若构建稍新一点，会通过锁检查而静默测试旧的原生代码。这与 CLAUDE.md 的“永不搜索或静默加载全局/过期扩展”相矛盾。

**测试测的是历史而不是行为**。
- 73 个文件 / 428 个测试只读源码文本、不做数值断言；做数值断言的是 115 个文件 / 890 个测试。
- 229 个测试位于以 phase/stage 命名的文件，135 个测试读取 `docs/dev/audit` 下的 JSON，`test_phase13_*_governance.py` 等约 3000 行比较的是两侧都手工冻结的常量（如 `len(renames) == 21`）。
- 这些测试让历史证据文件成为承重结构，无法清理。

**跳过即通过**。
- 612 个 CUDA 跳过点，cuda 档不对跳过报错；cuda 档排除 `tests/ad`（25 个文件，11732 行），完整的 FD/VJP/JVP 套件只在 nightly 跑，而 nightly 不会自动运行。
- `tests/scattering/test_table_build_ad.py:28` 把扩展导入失败变成跳过，本次 13 个原生对比 AD 测试因此被跳过而不是失败。

**门禁强度**。
- mypy 只检查 7/41 个模块（约 23% 行数），且 `follow_imports="skip"`。
- `public-api-snapshot.json` 中 60 个导出有 11 个是 `unresolved`、未哈希，包括 `CONTRACT_VERSION`、`AD_MODES`。
- `maintenance-budgets.json` 中 27 条豁免全部在 2027-01-31 到期，届时 quick 档会整体失败。
- duplication 账本按内容哈希键入，改动任一重复区域就失败，目前 nightly 已失败（2 条）。

**独立参考解缺口**。`tests/reference/` 只覆盖 Fresnel、层叠、Kirchhoff 和相位屏，没有 UTD/楔绕射、耦合 R-D/D-D 或 BDPT 估计量的独立 oracle。风险最高的物理只靠跨求解器一致性（同一个内核）或冻结的黄金数值验证。

**文档漂移**。
- CLAUDE.md 链接的 `docs/dev/plans/08-channel-modular-architecture-hardening-plan.md` 不存在（实际文件名含 `-native-`），并漏列已接受的 ADR-044 到 ADR-047。
- ADR-011/012 仍为 Proposed，但 `coupled_paths` 已发布。
- README 写 0.4（pyproject 为 0.5.0），称 owner 文档在包内（实际不在）。
- propagation/path/deterministic 的 README 描述约 10 个已删除的模块和函数。
- 23 个 Markdown 文件中有 64 个指向不存在文件的路径。
- `docs/dev/kernels/` 与 `docs/dev/interactions/` 没有 owner README。

**仓库体积**。跟踪文件 24.1 MB，其中 `docs/dev/baselines` 占 12.3 MB；5 个基线目录中 3 个（约 5.4 MB）没有代码引用。约 9875 行一次性迁移证据工具（`phase13_phase12*` 系列）主要由自身测试保活。

## 8. 建议的修复顺序

按“风险 × 修复成本”排序：

1. **先让防护机制生效**（成本低，收益覆盖后续所有修复）：
   - 修 `runtime.py:214`，更新 duplication 账本；
   - 给 Channel 仓库加 push/PR 工作流，至少跑 quick 档；
   - 改 `tests/support/native_ext.py`：不按 mtime 自动挑选，或挑选后校验与当前源码的指纹；
   - cuda 档对非预期跳过报错；
   - `check_contract_coverage.py` 改为扫描真实调用点。
2. **修静默错误物理**（第 3 节 #1-#7、#14、#16-#18、#20）。每项先写一个能复现的失败测试，再修。#1 和 #6 属于 CLAUDE.md 定义的“数值变更”，需要独立 ADR 与验收证据。
3. **修断开的梯度**（#3、#8、4.5 节的四处 detach），并让元数据与实际可微性一致。
4. **修越界与溢出**（#10、#15、#27 与 int32 扫描守卫），补 `CUDAGuard`。
5. **修合同不一致**：能力记录（#12）、SM 声明（#13）、consumer 边界拒绝（#21-#24）、`max_exported_paths` 的静默截断（#11）。
6. **架构清理**（第 5、6 节）：
   - 先做 `topology.py:98` 的 `masked_fill_` 这类零风险替换；
   - 再把 `bdpt.py` 与 `consumer.py` 中的原生调用迁回 `kernels`；
   - 删除无调用者的绑定（需同步改清单与 owner 清单）；
   - 让材料 store 常驻设备。
7. **测试与文档减负**：把历史治理测试与审计 JSON 归档，补 UTD/BDPT 独立参考解，修正失效链接与过期 README。
