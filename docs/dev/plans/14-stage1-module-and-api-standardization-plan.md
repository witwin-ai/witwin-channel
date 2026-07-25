# Stage-I 模块与公共 API 标准化计划

本计划修复 Stage-I（ADR-034）落地后遗留的模块归属与公共 API 问题。它是一次**纯结构性
重构**：不改物理、不改数值、不改 launch 配置、不改 reduction 顺序、不改 result schema。
任何发现的数值问题只登记，不在本计划内修改。

- 上游基线：Channel `codex/radar-architecture-stage1` @ `6e805c2`，Core @ `42b7b06`
- 工作分支：两个仓库均为 `claude/stage1-api-cleanup`
- 前置：Stage-I 全量套件已在本地复现通过（2372 + 367 补跑，0 failed）

## 0. 决策

| 议题 | 决策 |
|---|---|
| Channel root 的 Core 世界类型 re-export | **全部移除**，`witwin.core` 是唯一导入路径 |
| `witwin.channel.core` | **完全拆解**到现有 domain owner，包整体删除 |
| `witwin.core.identity` 全局计数器 | **本次不改语义**，只补文档与显式 `reserve_*` 路径测试 |

## 1. 问题清单与归属

| # | 问题 | 归属 | 阶段 |
|---|---|---|---|
| P1 | `Material = PhysicalMaterial` 别名，两个公共名字指向一个类 | Core | A |
| P2 | `Structure` / `MaterialAssignment` 定义在 `material.py` 里 | Core | A |
| P3 | Core 公共签名大量 `Any`，与 Channel 的严格 `torch.Tensor` 不一致 | Core | A |
| P4 | ID 全局计数器跨进程不可复现 | Core | A（仅文档+测试） |
| P5 | Channel root re-export Core 世界类型，一个类型两条导入路径 | Channel | B |
| P6 | `witwin.channel.core` 与 `witwin.core` 撞名，且不在 owner 清单 | Channel | C |
| P7 | root 经 `core/kernels/extension.py` 兼容 shim 取 `build_info` | Channel | C |
| P8 | consumer 枚举全裸字符串，合法值不可查，无 `capabilities()` 入口 | Channel | D |
| P9 | `PropagationRequest` / `FixedTopologyRequest` 无构造期校验 | Channel | D |
| P10 | `PropagationGeometry` 冗余单数字段（`[:, 0]` 的重复） | Channel | D |
| P11 | v1 声明了未实现的 frequency-offset 契约 | Channel | D |
| P12 | 相位约定在三处重复声明 | Channel | D |
| P13 | 两套 capability 系统语义打架 | Channel | E |
| P14 | `physics/oracle.py` 兼容 facade 转发私有名 | Channel | F |
| P15 | numpy CPU reference oracle 打进生产 wheel（违反 CLAUDE.md） | Channel | F |
| P16 | 门禁与 allowlist 存在指向已删除模块的陈旧条目 | Channel | G |
| P17 | 文档缺口：consumer 无 README，owner 清单缺四个包 | 两侧 | G |

## 2. 阶段

### 阶段 A — Core 契约整理

1. 删除 `Material` 别名，`PhysicalMaterial` 为唯一公共名。
2. `Structure`、`MaterialAssignment` 迁出 `material.py` → 新建 `witwin/core/structure.py`。
   `material.py` 此后只含材料规格本身。
3. 收紧公共签名类型标注：`evaluate_at_frequency`、`Trajectory.at`、`DynamicScene.at`、
   `Scene.snapshot` 的 `Any` → `float | torch.Tensor`。
4. `identity.py` 补模块级文档说明分配语义与可复现性边界；新增显式 `reserve_*_id`
   路径的确定性测试。
5. 更新 `README.md` 的 0.4 迁移段。

**验收**：Core `tests/` 全绿；`witwin.core.__all__` 无别名；`grep` 确认无遗留 `Material` 引用。

### 阶段 B — Channel root 公共 API 收敛

1. `witwin/channel/__init__.py` 移除全部 `witwin.core` re-export。
   保留 root 真正拥有的：`build_info`、`pipeline_cache_key`、`runtime_diagnostics`、
   `capabilities`、`Complex3State`、`JonesState`。
2. 包内所有 `from witwin.channel import Scene/...` 形式改为 `from witwin.core import ...`。
3. 更新 `ci/public-api-snapshot.json`；写迁移说明。
4. 新增 ADR-036 记录该 breaking 决策并修订 ADR-034 中"root 世界导出"条款。

**验收**：public API snapshot 测试通过；无模块经 channel root 取世界类型。

### 阶段 C — 拆解 `witwin.channel.core`

逐模块搬迁，每步只移动、不改实现：

| 现位置 | 目标 | 依据 |
|---|---|---|
| `core/kernels/extension.py` | **删除** | 唯一使用者是 root；`build_info` 改由 `deployment` 导出 |
| `core/kernels/metadata.py` | `runtime/kernel_metadata.py` | 纯 torch，无跨域依赖 |
| `core/memory_budget.py` | `runtime/memory_budget.py` | 纯值类型 |
| `core/edge_policy.py` | `scene/edge_policy.py` | scene 策略 |
| `core/edge_selection.py` | `scene/edge_selection.py` | 文档自述为 scene-policy 过滤 |
| `core/ad_geometry.py` | `scene/ad_geometry.py` | scene-leaf AD 接缝 |
| `core/antenna.py` | `scene/antenna.py` | 端点几何 |
| `core/receiver_geometry.py` | `scene/receiver_geometry.py` | 已依赖 `scene.endpoints` |
| `core/diffraction_geometry.py` | `propagation/geometry/edge_state.py` | 已依赖 `propagation.geometry.kernels` |
| `core/components.py` | `components.py`（顶层） | 跨域组件标识，root/solver 需可见 |
| `core/field_state.py`（状态类型） | `field_state.py`（顶层） | 公共 API，不能落在 `propagation.*` 下 |
| `core/field_state.py`（端点极化函数） | `scene/endpoints.py` | 依赖 `scene.endpoints` |
| `core/tensor_math.py` | 内联到两个使用点 | 9 行 |
| `core/` 包 | **删除** | |

`build_info` 由 `deployment.py` 导出后，root 只保留 `deployment` 一条内部边，
allowlist 的 `boundary-001` 从 `allowed` 中移除（`baseline` 保持不动以维持
`FROZEN_BASELINE_DIGEST`）。

顶层落 `components.py` / `field_state.py` 是因为门禁 `public_init_internal` 禁止
root `__init__` 导入 `*.kernels.*`、`runtime.*`、`propagation.*`——这两个是 root 公共
API 的实际依赖，必须落在这三类之外。

**验收**：`ci/check_import_graph.py` 通过且 allowed 债务从 2 条降到 1 条；
`witwin.channel.core` 加入门禁的 `deleted_modules` 集合防止重建。

### 阶段 D — consumer 契约标准化

1. 组件 / response / topology / AD 模式的合法值常量从 `service.py` 移入 `contracts.py`
   作为唯一来源，并给出 `Literal` 类型别名。
2. `propagation.consumer` 导出 `capabilities()`，可在发请求前查询。
3. `PropagationRequest`、`FixedTopologyRequest` 补 `__post_init__` 结构校验，与同文件
   其它契约风格一致。语义级校验（设备一致、能力匹配）仍留在 `service`。
4. 删除 `PropagationGeometry.interaction_position_m` / `interaction_normal`
   单数字段——它们就是复数字段的 `[:, 0]`。
5. 删除 v1 未实现的 `frequency_offsets_hz` 与 `PropagationConvention.frequency_offset_law`。
   将来实现时通过 `CONTRACT_VERSION` 提版引入。
6. `PHASE_CONVENTION` 字典删除，相位约定以 `PropagationConvention` 为唯一来源。

**验收**：consumer 契约与 E2E 测试全绿；`capabilities()` 有直接测试；
公共 API snapshot 同步更新。

### 阶段 E — capability 语义统一

root `capabilities()` 保持 solver 级事实描述，新增 `propagation_consumer` 段，
其内容由 consumer 的 typed capabilities 生成，不再各写一份。README 写明两者分工：
root = solver 能力，consumer = 跨包契约能力。

**验收**：两处 components 列表不再各自硬编码；capability 测试覆盖派生关系。

### 阶段 F — reference oracle 归位

1. 删除 `physics/oracle.py` 兼容 facade（它转发了 `_admittances` 等私有名）；
   11 个测试文件改为直接引用规范实现。
2. `physics/reference/` 迁至 `tests/support/reference_oracle/`，符合 CLAUDE.md
   "CPU/Torch reference 实现只允许在 tests/ 下"。
3. `physics/conventions.py`（纯常量，7 个生产使用点）迁至顶层 `constants.py`，
   `physics/` 包删除。门禁的 `oracle_production_dependency` 规则随之由结构保证取代。

**不做**：Core 的 `VACUUM_PERMITTIVITY = 8.8541878128e-12` 与 Channel 的
`EPS0 = 1/(MU0*C0^2) = 8.854187817620389e-12` 是两个不同的值。这是数值问题，
按 CLAUDE.md 必须走独立 ADR，本计划只登记不修改。

**验收**：生产 wheel 不再包含 numpy reference oracle；oracle 测试全绿。

### 阶段 G — 门禁与文档收尾

1. `check_import_graph.py`：`_COMPILED_SCENE_MODULES` 移除已不存在的
   `channel.core.runtime.compiled_scene`；`deleted_modules` 增加 `channel.core`、
   `channel.physics`。
2. 新增 `propagation/consumer/README.md`；`propagation/README.md` 补 consumer 段。
3. `CLAUDE.md` / `AGENTS.md` 的 domain owner 清单补齐 `propagation.consumer`、
   `propagation.models`、顶层 `components` / `field_state` / `constants`，删除 `core`
   与 `physics` 条目。两文件逐字一致。
4. `docs/dev/replacement/channel-migration.md` 补本次全部 breaking 变更。

## 3. 登记但不修改

| 项 | 原因 |
|---|---|
| Core 与 Channel 的真空介电常数取值不同（第 9 位有效数字） | 数值变更，需独立 ADR |
| `witwin.core.identity` 全局计数器 | 已决定本次不改语义 |
| Python / Torch 发布矩阵仅验证 3.11 | 与本计划无关，独立议题 |

## 4. 验收总门禁

```bash
conda run -n witwin2 python ci/run_ci_tier.py quick
conda run -n witwin2 python ci/run_ci_tier.py cuda
```

外加两仓库全量 pytest。因为本计划不含数值变更，`nightly`/`release` 的数值与性能
证据沿用 Stage-I 已冻结基线，不重新采集。

**回归判定**：Channel 全量必须仍为 0 failed，且通过数不得低于基线 2739（2372 + 367），
差额只允许来自本计划显式删除的契约字段测试。
