# RFDT-nano 性能审计（optimize.py + witwin/channel/*）

## 范围与假设
- 路径：`optimize.py`、`witwin/channel/trace/tracer.py`、`witwin/channel/scene/` 等核心运行时。
- 典型场景：`RadioFieldOptimizer.optimize()` 每个迭代都会更新网格并调用 `Tracer.trace()`，网格大小 256²、CUDA 变体。
- 关注点：CPU 侧开销、重复场景/边重建、GPU↔CPU 同步。

## 主要瓶颈
- **双重 update_scene 与缓存失效**  
  - `optimize.py:324-339` 先调用 `self._tracer.update_scene(..., recompute_edges=True)`，紧接着 `Tracer.trace()` 内部再次调用 `update_scene`（`witwin/channel/tracer.py:258-334`）。  
  - 第二次调用会重复 `scene_params.update()`、重新预加载三角形数据、清空 `_edge_cache`，等同于每迭代做两次场景更新与缓存失效。
- **边缓存每次 trace 都被清空，导致投影/衍射重复 CPU 路径**  
  - `Tracer.update_scene()` 无论 `recompute_edges` 是否为 `True` 都清空 `_edge_cache`（`witwin/channel/tracer.py:145-172`），而 `trace()` 每次都会调用它。结果是 `_get_edge_data()` 每次 trace 都重新跑 `project_to_2d` 和 `preload_diffraction_edges`。  
  - 这些函数包含大量 Python/NumPy 操作与标量提取（当前位于 `witwin/channel/scene/topology.py` 与 `witwin/channel/scene/projection.py`），强制 GPU→CPU 同步并在每迭代反复执行，边缓存形同虚设。
- **边/角生成管线过度依赖 CPU**  
  - `filter_vertical_edges` 把所有顶点拉到 NumPy 后做几何判定；`project_to_2d` 多次调用 `scalar()`/`np.array` 获取高度和排序角度。  
  - 对于刚性变换（平移 + Z 轴旋转），拓扑与 wedge 角度不变，可以在 GPU 端对“基准边集”做仿射变换，而无需每次 Python 循环与 CPU 拷贝。
- **不必要的反射/衍射输出**  
  - `Tracer.trace()` 始终请求反射的逐 bounce 结果并转为 dB（`witwin/channel/tracer.py:338-379`），即使优化循环并未使用；`dr.eval` 也会把这些中间结果同步。  
  - 虽然衍射设置了 `return_per_edge=False`，但反射部分仍保留全部列表，增加内存与同步开销。
- **频繁标量提取触发同步**（次要）  
  - 每迭代使用 `scalar()` 拉取 7 个标量用于日志（`optimize.py:366-373`）。虽成本较小，但仍会触发 GPU→CPU 读取，可按需降采样日志频率。

## 修复方案（优先级从高到低）
1) **避免双重场景更新**  
   - 给 `Tracer.trace()` 增加 `update_scene=False` 选项，或在优化循环内只调用一次 `update_scene`；确保只在顶点变更时重建。  
   - 预期：减少一次 `scene_params.update()`、一次 `_preload_triangle_data()` 和一次缓存清空/重建，迭代 CPU 同步显著下降。
2) **让边缓存真正可复用**  
   - 在 `Tracer.update_scene()` 中只在需要时清空 `_edge_cache`/重建竖直边；否则保留已投影的 2D 边与衍射数据。  
   - 为 mesh 维护一个 `version_id`：顶点更新时自增；`_get_edge_data` 使用 `(version_id, calculation_height)` 作为缓存键，防止重复重建。
3) **把边/角投影管线迁移到 GPU（去除 NumPy 回落）**  
   - 预计算基准竖直边的拓扑、wedge_n、法向；运行时用 DrJit 变换得到当前 `p0/p1/normal_2d`，避免 `np.array`/`scalar()`。  
   - 对刚性变换可直接应用 4x4 变换矩阵到基准边，衍射点的 `face_normals_3d` 也可同样变换。  
   - 预期：消除每迭代的 GPU→CPU 拷贝与 Python 循环，衍射准备阶段完全驻留 GPU。
4) **按需返回反射/衍射细粒度结果**  
   - 为 `Tracer.trace()` 添加参数控制 `return_per_bounce`，优化路径默认关闭，同时在调用方跳过对应的 dB 计算与 `dr.eval`。  
   - 若后续需要调试再打开，以避免热路径的多余同步。
5) **轻量化日志与准备阶段**（可选）  
   - 日志改为每 N 次迭代采样；`prepare()` 阶段的目标/初始场可复用同一个 `Tracer`，避免两次场景构建。

## 验证建议
- 在应用以上改动后，记录每迭代 wall-clock 与 CPU profile（如 py-spy/line_profiler，关注 `Tracer.trace` 前的 Python 部分）对比基线。  
- 利用 DrJit 的 `dr.flush_kernel_cache()` 前后观测 GPU 同步次数，确认 `np.array` 路径已被移除。  
- 检查优化收敛是否一致（误差、loss 曲线），确认几何与梯度链路未受影响。

---

## 追加审计（当前代码状态，Scene/Tracer 重构后）
- **ensure_variant 影响**  
  - 现有实现：几乎所有入口函数都调用 `ensure_variant()` 做一次 `mi.variant()` 检查，只有首次未设置时才 `set_variant`。检查开销在 µs 级，无 GPU/CPU 拷贝，不是主要瓶颈。可在应用入口（如 `optimize.main`）提前设置后，逐步移除热路径里的 `ensure_variant()`，或在 `ensure_variant` 内部加全局缓存以跳过重复调用。
- **仍存在的 CPU/GPU 同步点**  
  - `filter_vertical_edges`：每条边都在 Python 循环内做 `bool(is_valid)`，强制同步；可向量化计算 mask 后用 `dr.compress` 一次性收集有效边。  
  - `project_to_2d`：大量 `scalar()`/Python 分支（高度过滤、角度排序、corner 构造），每个 corner/edge 都在 CPU 侧循环，反复 GPU→CPU 取标量。  
  - `compute_edge_geometry`：按 edge 遍历（通常 2 面），同步点较少，可接受，但仍有 Python 循环。  
  - `extract_edges_with_adjacency`：仍是 Python over faces，执行一次；对小网格影响有限。
- **缓存与重建**  
  - `Scene.update_vertices` 每次都清空 `_edge_cache` 并重跑 `_build_vertical_edges` + `project_to_2d`。在优化中为每迭代必做。若仅做平移/绕 Z 旋转，可缓存基准 wedge/法线/拓扑，迭代时仅用 DrJit 4x4 变换更新 `p0/p1/normal_2d`，避免每轮重建与 Python 循环。
- **并行/向量化改进方向**  
  1) **边筛选与几何**：对全部边构建 DrJit 向量（p0/p1/edge_vec/length），用 mask 过滤竖直边并通过 `dr.compress` 收集索引；`compute_edge_geometry` 亦可批量处理成数组，减少 Python 循环与 bool 同步。  
  2) **投影与 corner**：高度过滤与 2D 投影可批量进行，边名称/索引可由 `dr.arange` 生成；corner 可用 scatter 收集端点，再在 CPU 做一次性拓扑排序（数量较少），避免 per-edge `scalar()`。  
  3) **Rigid 变换重用**：对于仅平移+Z 旋转的优化，`face_normals_3d`、`wedge_n` 不变，可只对顶点/法线做矩阵变换，省去重新计算角排序。  
  4) **日志**：仍有少量 `scalar()` 用于进度显示；可降频或仅在必要时同步。
- **建议的行动顺序**  
  1) 在应用入口设置 variant，删除/缓存热路径中的 `ensure_variant()` 调用。  
  2) 向量化 `filter_vertical_edges`（mask + compress），消除 `bool()` 同步。  
  3) 批量化 `project_to_2d` 高度过滤与投影，减少 `scalar()` 次数；corner 生成可使用端点表+有限 CPU 排序。  
  4) 若优化仅做刚性变换，引入“基准边几何 + 矩阵变换”路径，避免每迭代重建边几何。


Cache edge data across iterations 
Vectorize edge processing 
Pre-allocate result buffers 
Split reflection phases
