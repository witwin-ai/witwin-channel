# Plan 14 — RayD 构建源获取简化（ADR-023 §4 修订）

**状态：** DRAFT / 待 ADR-023 修订接受

**计划日期：** 2026-07-24

**Channel 基线：** `main@bff9b21`（`Restore standard release runners`）

**RayD lock 基线：** `49c58c4cb8212f6babb920cc88fb937509826cc5`，
`rayd-torch 0.7.0` source bundle，manifest
`e2eb1a7577f906b3ab52e6345b039837228771c8f1582c9f821d0f2bb07d41b4`

**范围：** 仅 RayD **构建源获取（build-source acquisition）** 机制。本计划
不改变源码级链接决策本身、不改变 typed integration 边界、不改变物理模型、
数值顺序、fusion 边界、launch 配置、solver 行为或公开 Python API，也不引入
任何 runtime fallback。

**关联记录：** ADR-023 §4（native boundary and build/package boundary）、
ADR-006（extension developer override）、
[Plan 13](./13-direct-rayd-integration-and-rf-runtime-ownership-plan.md)

---

## 1. 背景：源码链接是必要的，获取机制不是

### 1.1 源码链接必须保留（本计划不触碰）

Channel 的 CUDA kernel 直接 `#include` RayD 的 device 头，并在自己的
kernel body 内展开 `__device__` 内联代码：

| RayD device 头 | Channel 引用点数 |
|---|---|
| `rayd/shared/rf/field_transport.cuh` | 7 |
| `rayd/torch/rf/field_transport_ad.cuh` | 6 |
| `rayd/shared/rf/layer_stack.cuh` | 4 |
| `rayd/shared/utd/utd_math.h` | 2 |
| `rayd/torch/integration.h` | 3 |

共 13 个 `.cu`/`.cuh` TU。这些 device 内联代码无法跨任何二进制边界传递：
即使改为预编译产物，这些头仍必须在 Channel 的 TU 内、以 Channel 的
gencode 与 `--fmad=false` 策略重新编译。源码在构建期必须在场，没有选择余地。

同时 typed 边界使用 `at::Tensor`、`std::optional<at::Tensor>`、RAII
`SceneResource` 与具名 result struct。改为 DSO 边界会强制退回 ADR-023 已
退役的 `extern "C"` + `int64_t` handle + capacity 输出数组设计，而 libtorch
自身 ABI 已强制编译器/Torch 精确匹配，预编译换不到任何真实解耦。

**结论：`add_subdirectory` + `rayd_torch_native_core` 静态链接保持不变。**

### 1.2 需要修订的是「源码从哪来」

`RAYD_SOURCE_DIR` 未提供时，[CMakeLists.txt:315-344](../../../CMakeLists.txt)
走 package-discovery 分支：从已安装的 `rayd-torch` distribution 中定位
passive `rayd/torch/_source/rayd-source.json`，校验 distribution
RECORD 归属与逐文件 source manifest digest，然后 `add_subdirectory`。

审计事实：

1. **该路径当前无任何真实构建在使用。**
   - [publish-witwin-channel.yml:273](../../../../.github/workflows/publish-witwin-channel.yml)
     与 `:451` 都从独立 git checkout 传 `RAYD_SOURCE_DIR`。
   - `benchmarks/phase13_phase12/builds.py:234,496` 与 `release.py:287,1115`
     同样传 `RAYD_SOURCE_DIR`。
   - [README.md:83-88](../../../README.md) 自述：已发布的 `rayd-torch 0.7.0`
     bundle **不满足 release guard**，Channel 0.4.0 源码构建必须设置
     `RAYD_SOURCE_DIR`。

2. **该路径从干净环境不可达。** `rayd-torch` 既不在
   [pyproject.toml:15](../../../pyproject.toml) 的 `dependencies`，也不在
   `build-system.requires`。sdist 构建不会安装它。

3. **它在 git 路径下不提供任何校验价值。**
   [CMakeLists.txt:245](../../../CMakeLists.txt) 把
   `CHANNEL_RAYD_SOURCE_MANIFEST_SHA256` 初始化为 lock 中的
   `source_bundle.manifest_sha256`，git-checkout 分支从不覆盖它。因此
   git 构建的 build fingerprint 中该字段只是 lock 文件的常量回显，
   不校验任何实际 checkout 内容。

4. **成本可量化。**

   | 文件 | 行数 | 处置 |
   |---|---|---|
   | `cmake/resolve_rayd_source.py` | 332 | 删除 |
   | `tests/kernels/test_rayd_package_discovery.py` | 177 | 删除 |
   | `CMakeLists.txt` discovery 分支 + lock 读取 | ~35 | 删除 |
   | `cmake/validate_build_identity.cmake` `python-package` 分支 | ~18 | 删除 |
   | `ci/wheel_smoke.py` source_bundle 校验 | ~45 | 删除 |
   | `tests/kernels/test_rayd_lock_cmake.py` 相关用例 | 部分 | 收缩 |

   合计约 600+ 行专用于一条无人使用、且从干净环境不可达的路径。

5. **副作用。** 装有 `rayd-torch` 的环境中，同一份 RayD native core 被编译
   两次：一次进入 wheel 自带的 `_raydtorch`（Channel 从不加载），一次进入
   `_channel`。

---

## 2. 决策草案

### Phase A — 删除 package discovery，`RAYD_SOURCE_DIR` 成为唯一源输入

`RAYD_SOURCE_DIR` 从「最高优先级可选输入」升格为**必填输入**。未提供时
configure 立即 fail-loud，并在错误信息中给出获取锁定 checkout 的命令。

保留 **不变** 的现有 git 校验（[CMakeLists.txt:246-314](../../../CMakeLists.txt)）：
`rev-parse HEAD` 对齐 lock commit、`remote get-url origin` 对齐 lock
repository、integration 头 SHA256 对齐 lock、`status --porcelain` 判定
dirty 并在 `CHANNEL_RELEASE_BUILD` 下拒绝。

这是纯删除，不引入任何新机制。

### Phase B（可选，Phase A 之后独立评估）— submodule 作为默认来源

在 `dependencies/rayd/` 加入 pin 到 lock commit 的 git submodule。
`RAYD_SOURCE_DIR` 未提供时默认指向它，并**复用 Phase A 保留的同一段 git
校验逻辑**，不新增任何校验代码。

选择 submodule 而非 `FetchContent` 的理由：configure 期不需要网络、
`git status` dirty 语义与现有校验一致、离线/受限 CI 可用。

**Phase B 的已知代价（决策点，不在本计划中预先接受）：** scikit-build-core
sdist 默认按 `git ls-files` 收集，不含 submodule 内容；若 Channel 需要可用的
sdist，须显式配置 `tool.scikit-build.sdist.include`。当前 Channel 只发布
wheel，因此 Phase B 可以延后到确有 sdist 需求时再做。

---

## 3. 不变量（任何阶段都不得改变）

- RayD 以源码形式进入同一 CMake 图，`rayd_torch_native_core` 静态链接进
  `_channel`；`_channel` 仍是唯一生产扩展。
- Channel `.cu` 对 RayD device 头的 include 集合与编译 flag 不变。
- build fingerprint 仍包含 `rayd_commit`、`rayd_repository_url`、
  `rayd_dirty`、`rayd_integration_abi_kind/path/sha256`，仍不含机器相关绝对路径。
- 不新增 runtime fallback、动态符号查找、第二 dispatcher、第二 scene registry、
  或对 `rayd.torch` 的 Python import。
- 不改变物理、数值顺序、fusion 边界、launch 配置、stream、同步、结果 schema
  或 AD 支持。
- 不放宽任何 test、tolerance、manifest、allowlist 或 maintenance budget。

---

## 4. 逐文件变更清单（Phase A）

| 文件 | 变更 |
|---|---|
| `dependencies/rayd.lock.json` | 删除 `source_bundle` 块；`schema_version` → 3 |
| `CMakeLists.txt` | 删除 `:18-21` source_bundle 读取、`:245` manifest 初始化、`:315-344` discovery 分支；`RAYD_SOURCE_DIR` 为空时 FATAL_ERROR；fingerprint JSON 移除 `rayd_source_kind`、`rayd_source_manifest_sha256` |
| `cmake/resolve_rayd_source.py` | 删除 |
| `cmake/validate_build_identity.cmake` | 删除 `CHANNEL_RAYD_SOURCE_KIND` / `CHANNEL_RAYD_RESOLVER` 参数与 `python-package` 分支（`:74-105`） |
| `native/channel/build_info.cpp` | 删除 `:52-53` 两个字段与对应编译定义 |
| `ci/wheel_smoke.py` | 删除 `:67-68` 期望键、`:237-275` source_bundle 校验、`:592-594` source_kind 断言 |
| `benchmarks/phase13_phase12/release.py` | 删除 `:55-56` 两个 fingerprint 键 |
| `tests/kernels/test_rayd_package_discovery.py` | 删除 |
| `tests/kernels/test_rayd_lock_cmake.py` | 收缩到 schema 3；新增「缺 `RAYD_SOURCE_DIR` 必须 fail-loud」负例 |
| `tests/kernels/test_build_identity_cmake.py` | 移除 source_kind 维度 |
| `tests/kernels/test_build_info.py` | 移除两个字段断言 |
| `README.md` | 安装章节改为 `RAYD_SOURCE_DIR` 必填，删除 bundle guard 说明 |
| `FEATURE_LIST.md` | 更新 `:21` 构建源描述 |
| `AGENTS.md` / `CLAUDE.md` | 更新「RayD build-source discovery」段落，两文件同一 commit 保持一致 |
| `docs/dev/standards/adr-023-*.md` | 按附录 A 修订 §4 |
| `docs/dev/replacement/channel-migration.md` | 更新 `:588` 描述 |

---

## 5. 验收证据

本变更为构建边界变更，不触及数值/fusion，因此不需要 Munich 数值重跑。

1. **静态门。** `conda run -n witwin2 python ci/run_ci_tier.py quick` 通过，
   含 `check_import_graph.py`、`check_repository_hygiene.py`、
   `check_production_dependencies.py`、`check_maintenance_budgets.py`。
2. **构建门。** 以 `RAYD_SOURCE_DIR` 指向 lock commit 的干净 checkout 完成
   本机 Release 构建；`cuda` tier 通过。
3. **负例。** 三条新增 fail-loud 用例：未设 `RAYD_SOURCE_DIR`、
   `RAYD_SOURCE_DIR` 指向非 RayD 树、checkout commit 与 lock 不符。
   三者都必须在 configure 期失败，且失败信息不提示任何回退路径。
4. **fingerprint 变更说明。** 记录旧/新 fingerprint JSON 与 hash，说明
   输入集合合法收缩（两个字段在 git 路径下本就是 lock 常量回显）。
   `CHANNEL_ABI_VERSION` **不**变更：这不是 ABI 变更。
5. **打包门。** wheel 检查显示恰好一个生产扩展 `_channel`、无 RayD Python
   扩展、无未声明 DSO；`ci/wheel_smoke.py` 与 `ci/audit_windows_pe.py` 通过。
6. **manifest 一致性。** `ci/check_contract_coverage.py`、
   `ci/check_coverage.py` 通过；`native-binding-manifest.json` 与
   `public-api-snapshot.json` **不应有变更**（本计划不改 ABI 与公开 API），
   若有变更即为越界，停止。

---

## 6. 风险与回滚

| 风险 | 处置 |
|---|---|
| 现存 wheel 的 fingerprint 与新构建不可比 | 预期行为；ADR-006 override 本就校验完整 fingerprint。在 CHANGELOG 与 migration note 中明示。 |
| 第三方从 sdist 构建变得更不方便 | 现状即已不可行（`rayd-torch` 不在 build requires，且 0.7.0 bundle 不过 guard）。Phase A 只是把隐性失败变成显式失败。Phase B 解决可用性。 |
| `benchmarks/phase13_phase12` 证据 schema 引用旧字段 | 与 `benchmarks/schemas/*.json` 同一 commit 更新；历史已归档证据不重写。 |

**回滚：** 单 commit revert。本计划不产生任何需要迁移的持久状态。

---

## 附录 A — ADR-023 §4 修订文本草案

将 §4 中以下段落：

> An explicit `RAYD_SOURCE_DIR` remains the highest-priority build input and is
> validated as a Git checkout. When it is absent, Channel may locate only the
> selected Python interpreter's unique `rayd-torch` distribution and read its
> passive `rayd/torch/_source/rayd-source.json` resource without importing
> `rayd.torch`. Before source-linking, Channel validates the lock-pinned
> distribution/version, repository, commit, stable API/identity/header,
> distribution RECORD ownership, and every file in the complete source manifest.
> Missing, duplicate, dirty, escaped, or mutated package sources fail loudly.
> This is build-source discovery, not a second runtime backend or dispatcher.

替换为：

> RayD enters the build only as a validated Git checkout named by
> `RAYD_SOURCE_DIR`. Before source-linking, Channel validates the lock-pinned
> repository URL, commit, and stable integration API/identity/header hash, and
> records worktree dirtiness in the build fingerprint;
> `CHANNEL_RELEASE_BUILD` rejects a dirty checkout. A missing, invalid, or
> mismatched `RAYD_SOURCE_DIR` fails loudly at configure time. Channel never
> discovers RayD source from an installed distribution, a Conda prefix,
> site-packages, or a CMake registry, and never imports `rayd.torch`. A
> repository-pinned submodule may supply the default checkout path; it is
> subject to the same validation and adds no second acquisition mechanism.

并在 §4 第二段的禁止清单中保留原有四条，补充：

> - read RayD source out of an installed Python distribution.
