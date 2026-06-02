# Transmission / Refraction Implementation Analysis

## 1. Current State Summary

### What exists today

| Component | Status | Details |
|-----------|--------|---------|
| Material model | Partial | `eps_r`, `sigma_e` per triangle; complex permittivity computed; Fresnel TE/TM reflection coefficients |
| LoS occlusion | Binary shadow test | `ray_intersect` → blocked/unblocked, no partial transmission |
| Reflection | Full | Multi-bounce, Fresnel or scalar, polarization transport, image-method path collection |
| Diffraction | Full | UTD wedge diffraction, slope term, mixed R-D/D-R families |
| Transmission | **None** | Rays terminate or reflect; no refracted ray spawning, no transmission coefficient |

### Energy accounting gaps

当前架构中的能量核算：

1. **反射**：Fresnel 系数 `|R|² < 1`，但反射未从入射能量中扣除 `1 - |R|²` 的透射部分 — 这部分能量直接被丢弃。
2. **LoS**：二元遮挡测试 — 物体后方 LoS 直接归零。对于薄墙/玻璃/介质板场景，这不正确：应衰减而非完全遮挡。
3. **Diffraction**：UTD 系数自身满足局部能量守恒，但入射场未考虑前序材料透射衰减。

结论：**增加透射并非仅为"更完整"，而是让现有反射计算的物理含义自洽。**

---

## 2. Is Transmission Worth Adding Now?

### Yes — three reasons

1. **物理自洽**：Fresnel 反射系数已经计算了；透射系数是免费的 (`T = 1 + R` for TE, `T = (1+R)/η` for TM`)。不使用它是浪费已有信息。
2. **Indoor/Urban 刚需**：室内场景中，穿墙路径经常是主要贡献。没有透射，室内覆盖预测基本不可用。
3. **差异化**：加入可微分的透射后，材料参数（`eps_r`, `sigma`）可以通过梯度优化从测量数据反演 — 这是 Sionna RT 目前也在做的方向。

### Scope control

不需要一步到位实现体积折射 (Snell's law bending)。分两个阶段：

- **Phase 1**：Thin-slab approximation（薄板近似） — 只计算透射衰减，不弯折光线方向。适用于大多数建筑墙体/楼板。
- **Phase 2**：Full Snell refraction — 光线穿过厚介质体时发生方向偏折。仅对玻璃幕墙、棱镜等场景需要。

---

## 3. Design Question: Spawn Transmitted Ray vs. Probabilistic Selection

### Option A: Deterministic split — spawn both reflected and transmitted rays

每次击中表面时，同时生成反射光线和透射光线，各自携带对应的 Fresnel 权重。

```
Incident ray (weight=w)
  ├── Reflected ray (weight = w * R)
  └── Transmitted ray (weight = w * T)
```

**Pros:**
- 物理精确，确定性结果
- 适合 image-method path collection（每条路径是确定的几何序列）
- 梯度干净：`dR/d(eps_r)` 和 `dT/d(eps_r)` 各自独立

**Cons:**
- 状态指数爆炸：每次反射/透射 split，光线数量翻倍。`N` 次交互后有 `2^N` 条路径
- 需要 path budget 裁剪

### Option B: Probabilistic selection — Russian roulette

每次击中时，按 `|R|²` / `|T|²` 的比例随机选择反射或透射。

**Pros:**
- 光线数量不膨胀
- Monte Carlo 收敛性好

**Cons:**
- 引入随机噪声，不适合确定性 image-method
- 梯度需要 reparameterization trick 或 importance sampling 修正
- 与当前架构（确定性反射链 + DDA 网格累积）不兼容

### Option C (recommended): Deterministic split with budget control

采用 Option A，但加入路径预算控制：

1. **Thin-slab fast path**：透射不改变方向，只衰减。不需要额外光线 — 等价于在原始光线上乘以 `T` 系数继续传播，与反射光线并行。
2. **Full refraction path**：方向偏折时 spawn 新光线，但受全局路径预算限制（类似现有 `diffraction_state_budget`）。
3. **Pruning**：低权重（`|weight| < threshold`）路径直接终止。

这与当前架构最匹配：当前反射已经是确定性 multi-bounce，增加一个透射分支自然。

---

## 4. Implementation Plan

### Phase 1: Thin-Slab Transmission (Recommended first step)

**核心思想**：将墙体视为平行平面薄板。光线穿过后方向不变，仅衰减。

#### 4.1 Fresnel Transmission Coefficient

在 `material.py` 中增加：

```python
def fresnel_transmission(cos_theta, eta):
    """Returns (t_te, t_tm) Fresnel transmission coefficients."""
    r_te, r_tm = fresnel_reflection(cos_theta, eta)
    t_te = 1.0 + r_te          # interface 1: air → material
    t_tm = (1.0 + r_tm) / eta  # normalized by impedance ratio
    return t_te, t_tm
```

薄板透射系数（两次界面穿透 + 内部传播衰减）：

```
T_slab = T_12 * exp(-j*k_z*d) * T_21
```

其中 `d` 是板厚，`k_z` 是材料内纵向波矢。

**难度**：**低**。纯数学运算，不涉及光线拓扑变化。

#### 4.2 LoS Transmission Through Obstacles

修改 `los.py`：当前的二元遮挡改为**累积透射衰减**。

```python
# Current (binary):
a_los = select(blocked, 0, free_space_field)

# Proposed (transmission):
# For each obstacle the LoS ray intersects:
#   accumulate T_slab(material, thickness, angle)
a_los = free_space_field * product(T_slab_i for each intersected obstacle)
```

**Implementation detail:**
- 使用 multi-hit ray tracing：循环调用 `ray_intersect`，每次将 origin 推进到上一个 hit 点之后
- 每次 hit 查找 triangle material → 计算 `T_slab`
- 累积所有 `T_slab` 乘积作为最终 LoS 衰减

**难度**：**中**。需要多次 ray-mesh 交点查询（目前只做一次），需要估计 slab 厚度。

**Slab thickness estimation:**
- 简单方案：使用固定默认厚度（如 0.2m for concrete wall）
- 中等方案：每个 `Material` 增加 `thickness` 属性
- 完整方案：对每个 hit，cast 反向 ray 找到 exit point，`thickness = exit.t - entry.t`

#### 4.3 Reflection Bounce Loop: Spawn Transmitted Ray

修改 `reflection/field.py` 的 bounce loop：

```python
for bounce in range(max_reflections + 1):
    si = scene.ray_intersect(ray, active)
    hit = si.is_valid() & active

    # --- Existing: reflected ray ---
    weight_reflected = weight * R
    # ... continue reflection chain ...

    # --- New: transmitted ray ---
    weight_transmitted = weight * T_slab
    # Continue the transmitted ray with same direction, attenuated weight
    # These transmitted rays enter a separate "transmission chain"
    # or are accumulated as additional paths
```

**Architecture choice:**

对于 thin-slab（不改变方向），最简单的实现方式：

1. 在 bounce loop 中，每个 hit 同时更新两组状态：
   - `reflected_*`: 原有反射链（方向改变，权重乘 R）
   - `transmitted_*`: 原有方向继续（方向不变，权重乘 T_slab）
2. 透射光线继续参与后续 bounce（可能再次反射或再次透射）
3. DDA 累积对两类光线都执行

**难度**：**中-高**。涉及 bounce loop 核心逻辑改动，需要谨慎处理状态分裂和路径收集。

#### 4.4 Diffraction Shadow Test

当前 diffraction 中的 shadow test 也是二元的。需要与 4.2 类似的改动：被遮挡的衍射路径不应直接归零，而应乘以透射衰减。

**难度**：**中**。改动模式与 4.2 相同。

### Phase 2: Full Snell Refraction (Future)

当 Phase 1 稳定后，可选实现完整折射：

#### 4.5 Snell's Law Direction Change

```python
def snell_refract(incident_dir, normal, eta_ratio):
    """Compute refracted direction via Snell's law."""
    cos_i = -dot(incident_dir, normal)
    sin2_t = eta_ratio**2 * (1 - cos_i**2)
    # Total internal reflection check
    total_reflection = sin2_t > 1.0
    cos_t = sqrt(1 - sin2_t)
    refracted = eta_ratio * incident_dir + (eta_ratio * cos_i - cos_t) * normal
    return refracted, total_reflection
```

**难度**：**高**。

原因：
- 光线方向改变 → 不能再使用 image method（image method 依赖光线直线传播假设）
- 需要全新的 path tracing 分支来追踪折射光线
- 介质内部的多次内反射需要单独处理
- 全内反射 (TIR) 判断和处理
- 从介质内部出射时需要第二次 Snell's law

#### 4.6 Volumetric Absorption

穿过厚介质体时的体吸收：

```
attenuation = exp(-alpha * path_length_inside_medium)
alpha = omega * sqrt(mu * epsilon) * sqrt(0.5 * (sqrt(1 + (sigma/omega*eps)^2) - 1))
```

**难度**：**中**。数学简单，但需要跟踪光线在介质内的传播距离。

---

## 5. Difficulty Assessment

| Task | Difficulty | LOC Estimate | Dependencies |
|------|-----------|-------------|--------------|
| 4.1 Fresnel transmission coefficients | **Low** | ~40 lines | None |
| 4.2 LoS multi-hit transmission | **Medium** | ~80-120 lines | 4.1, slab thickness |
| 4.3 Reflection loop: thin-slab split | **Medium-High** | ~150-250 lines | 4.1, bounce loop redesign |
| 4.4 Diffraction shadow with transmission | **Medium** | ~60-100 lines | 4.1, same pattern as 4.2 |
| 4.5 Full Snell refraction | **High** | ~300-500 lines | New path tracing branch |
| 4.6 Volumetric absorption | **Medium** | ~50-80 lines | Medium tracking |

### Overall Phase 1 estimate: Medium difficulty

Phase 1 (4.1 + 4.2 + 4.3 + 4.4) 总体难度为**中等**。最复杂的部分是 4.3 — 修改反射 bounce loop 的核心逻辑。其余都是模式化改动。

### Risk areas

1. **Slab thickness**：thin-slab 模型需要知道材料厚度。如果 mesh 是 single-sided（只有一个面），需要假设默认厚度或要求用户指定。
2. **Path explosion**：每个 hit 产生两条路径，`N` 次交互后 `2^N`。需要严格的权重阈值裁剪。
3. **Gradient stability**：透射系数在掠射角附近可能产生梯度尖峰。需要与反射一样的 `clip(cos_theta, eps, 1.0)` 保护。
4. **DDA grid 兼容性**：透射光线方向不变，可以直接使用现有 DDA。但如果加入 Snell 折射（Phase 2），DDA 可能需要修改。
5. **Multi-hit performance**：LoS multi-hit 需要循环调用 `ray_intersect`。对大型场景可能需要限制最大穿透次数。

---

## 6. Recommended Execution Order

```
Step 1: Fresnel transmission coefficients in material.py
        └── Unit test: verify R + T = 1 at normal incidence for lossless dielectric

Step 2: Material.thickness attribute (or per-structure default)
        └── Update witwin.core.Material if needed

Step 3: LoS multi-hit with thin-slab transmission
        └── Test: LoS through single wall → known analytical attenuation
        └── Test: LoS through two walls → product of attenuations
        └── Regression: existing LoS-blocked tests still pass

Step 4: Reflection bounce loop: transmitted ray branch
        └── Test: single-wall scene, verify reflected + transmitted power ≈ incident
        └── Test: two-room scene, verify coverage behind wall
        └── Gradient test: d(field)/d(eps_r) through transmission

Step 5: Diffraction shadow paths with transmission attenuation
        └── Regression: existing diffraction tests
        └── Test: diffraction behind a thin wall

Step 6 (optional): Config controls
        └── TraceConfig.enable_transmission: bool (default True)
        └── TraceConfig.max_transmissions: int (default 3)
        └── TraceConfig.transmission_weight_threshold: float (default 1e-4)
```

---

## 7. Architecture Impact

### Files to modify

| File | Change |
|------|--------|
| `witwin/channel/material.py` | Add `fresnel_transmission()`, `thin_slab_transmission()` |
| `witwin/channel/trace/materials.py` | Add `bounce_transmission_weight()`, `resolve_slab_thickness()` |
| `witwin/channel/trace/los.py` | Multi-hit loop, cumulative transmission attenuation |
| `witwin/channel/trace/reflection/field.py` | Bounce loop: transmitted ray branch, path collection for T paths |
| `witwin/channel/trace/reflection/materials.py` | Possibly merge into `trace/materials.py` |
| `witwin/channel/trace/diffraction/field.py` | Shadow test → transmission attenuation |
| `witwin/channel/trace/diffraction/geometry.py` | Shadow mask → transmission weight |
| `witwin/channel/trace/tracer.py` | Config plumbing, result assembly for T component |
| `witwin/channel/trace/config.py` | Transmission-related config fields |
| `witwin/channel/scene/runtime.py` | Per-triangle thickness data (if supported) |

### New files (possibly)

| File | Purpose |
|------|---------|
| `witwin/channel/trace/transmission.py` | Thin-slab transmission model, multi-hit utilities |

### Result schema extension

```python
Result.field = {
    'los': ...,
    'reflection': ...,
    'diffraction': ...,
    'transmission': ...,   # NEW: paths that include at least one transmission
    'total': ...,
}
```

---

## 8. Comparison: Thin-Slab vs Full Refraction

| Aspect | Thin-Slab (Phase 1) | Full Refraction (Phase 2) |
|--------|---------------------|--------------------------|
| Ray direction | Unchanged | Bent by Snell's law |
| Image method compatible | Yes | No |
| DDA compatible | Yes | Needs modification |
| Applicable to | Walls, floors, ceilings | Glass curtain walls, prisms, lens |
| Industry standard | Yes (ITU-R P.2040) | Rarely used in channel sims |
| Differentiable | Straightforward | Complex (TIR discontinuity) |
| Implementation effort | ~2-3 days | ~1-2 weeks |

**Recommendation**: Phase 1 (thin-slab) covers 90%+ of practical use cases. Phase 2 only needed for special scenarios.

---

## 9. References

- ITU-R P.2040: Effects of building materials and structures on radiowave propagation
- 3GPP TR 38.901: Channel model for frequencies from 0.5 to 100 GHz (Table 7.4.3-1: material properties)
- Degli-Esposti et al., "Ray tracing including diffraction and rough surface scattering" (thin-slab model)
- Sionna RT: `compute_paths()` with transmission enabled via `max_depth` parameter
