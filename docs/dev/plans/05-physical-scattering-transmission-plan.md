# Channel Native 物理散射与透射实现方案

## 1. 目标与范围

本文给出一套可在 `channel_native` 当前 CUDA/RayD 路径框架内落地的散射与透射方案。下文将需求中的“投射”解释为无线电传播中的**透射（transmission）**。

目标不是沿用指向性散射瓣或单一经验系数，而是从 Maxwell 边界条件出发，在几何光学可用的尺度上同时满足：

1. 复数幅度、相位与极化一致；
2. 反射、粗糙面散射、透射和吸收不重复计能；
3. 有限厚度墙体包含折射、材料吸收和多次内反射；
4. 散射方向由表面统计量（RMS 高度、相关长度、各向异性）决定；
5. 与现有确定性求解器、MC basic、BDPT 和 RayD 可见性查询兼容；
6. 每个近似都有明确适用范围，并可用测量或全波结果替换局部模型。

本方案不声称在所有尺度上“全波精确”。当结构细节、粗糙相关长度或层内横向变化与波长同量级时，单纯射线法无法恢复绕射、表面波、近场耦合和相干多体效应。此时应离线用 FDTD/FEM/MoM 或测量生成双站散射表，再作为本求解器的表格化 BSDF 输入。本文的主模型适用于：传播距离和物体曲率半径远大于波长、表面可局部近似为平面、介质线性且被动。

### 1.1 最终模型选择

实现不是一个“万能散射公式”，而是由尺度判据选择模型：

| 当前条件 | 采用模型 | 输出 |
|---|---|---|
| 平滑、均匀、有限厚度界面 | Fresnel + 稳定层栈 S-matrix | 相干 Jones 反射/透射 |
| 给定一次具体高度实现 | 相位屏 + Kirchhoff patch integral | 可复现相干复场/speckle |
| 只给相关函数或 PSD | 集合平均 Kirchhoff-BSDF | 极化双站平均功率与采样 PDF |
| 超出 Kirchhoff/相位屏适用域 | 明确报错或测量/全波表格 | `TabulatedPolarimetricBSDF` |

`auto` 模式只有在某一解析模型满足其适用条件时才选择它；落入模型空白区时必须报错或要求表格数据，不能静默退化成余弦瓣。这样“更物理准确”具体指：材料频散、有限厚度、极化、相位、表面 PSD 和适用尺度都进入模型，而不是仅更换一条经验方向函数。

**当前实现范围进一步收敛：散射只实现相位屏 + Kirchhoff-BSDF。** SPM、Beckmann 微表面、POM/relief、位移求交和 tessellation 均不进入当前阶段；前两者只保留为未来模型对照，后三者明确不修改 RayD。

---

## 2. 当前工程基础与必须先修正的问题

当前工程已经具备：

- `core/materials.py` 中的 `eps_r`、`mu_r`、`sigma_e`、`thickness_m`；
- `MaterialStore` 和逐 face 材质展开；
- CUDA 中的复 Fresnel 系数、有限厚度 slab 反射和确定性复矢量场；
- RayD 交点、法线、primitive id、多次反射链和可见性；
- BDPT 的子路径状态、PDF、MIS 与连接累积框架。

但新增物理事件前必须解决以下语义问题：

1. `deterministic_field.cu` 使用复电场矢量，而 `bdpt_subpaths.cu` 只保存一个标量复 throughput；两者不能正确表示连续反射后的极化旋转。
2. 当前 BDPT 反射处计算的是功率反射率 `R=|r|^2`，却直接乘到复 throughput 上。如果 throughput 表示场幅，应乘 `r`；如果表示功率，就不应保留复相位。散射与透射加入前必须统一。
3. 现有 `components` 只有 `los/reflection/diffraction`，需增加 `transmission` 与 `scattering`，吸收只作为能量损失和诊断项，不生成传播路径。
4. 当前厚度位于 `model_params[:,0]`，四个匿名槽不足以可靠表达粗糙面、分层介质和频散参数；需要显式、版本化的材质 ABI。
5. RayD Mesh 支持 UV，但 `core/runtime/raydn.py` 当前向 native scene 传入空 UV；高度图前必须保留并传递 structure 的 UV/face-UV。

建议统一为两条清晰的数据通道：

- **相干路径**：携带 Jones 复电场，最终按相位相干叠加；适用于 LoS、镜面反射、平滑界面/平行板透射和确定性绕射。
- **非相干粗糙散射**：携带功率或 Stokes 向量，样本之间按功率累加。只有在输出复 CIR 且给定“表面实现编号”时，才给散射样本分配可复现随机相位；不同表面实现不能误作相干路径叠加。

---

## 3. 统一电磁约定

### 3.1 相量和复材料参数

全文采用时间因子

$$
\mathbf E(\mathbf x,t)=\Re\{\tilde{\mathbf E}(\mathbf x)e^{j\omega t}\},
$$

因此传播因子为 $e^{-jkr}$，有损介质的复相对介电常数为

$$
\tilde\epsilon_r(\omega)
=\epsilon_r'(\omega)-j\epsilon_r''(\omega)
=\epsilon_r'(\omega)-j\frac{\sigma_e(\omega)}{\omega\epsilon_0}.
$$

类似地允许 $\tilde\mu_r=\mu_r'-j\mu_r''$。定义

$$
k_m=k_0\sqrt{\tilde\epsilon_{r,m}\tilde\mu_{r,m}},\qquad
\eta_m=\eta_0\sqrt{\frac{\tilde\mu_{r,m}}{\tilde\epsilon_{r,m}}},\qquad
k_0=\frac{\omega}{c_0}.
$$

平方根选择满足被动介质衰减的分支：$\Re(k_m)\ge0,\ \Im(k_m)\le0$，使 $e^{-jk_m z}$ 随传播距离衰减。所有 CPU 参考实现和 CUDA 实现必须使用同一分支规则。

### 3.2 局部极化基

射线方向为 $\hat{\mathbf d}$，界面朝入射介质的法线为 $\hat{\mathbf n}$。定义

$$
\hat{\mathbf s}=\frac{\hat{\mathbf n}\times\hat{\mathbf d}}
{\|\hat{\mathbf n}\times\hat{\mathbf d}\|},\qquad
\hat{\mathbf p}=\hat{\mathbf s}\times\hat{\mathbf d}.
$$

法向入射附近用确定性的备用轴构造 $s$，不能依线程或三角形顶点顺序随机翻转。路径状态保存全局横向复矢量 $\tilde{\mathbf E}\in\mathbb C^3$，或保存 Jones 向量 $(E_s,E_p)$ 加可重建的局部 frame。前者占用更大但最不易出现跨事件基变换错误，建议第一版使用 `complex3`。

每次事件均执行：投影到入射 $(s,p)$ 基、施加 $2\times2$ Jones 矩阵、在出射基重组。各向同性平滑介质的 Jones 矩阵为对角阵；各向异性粗糙面或测量表格可以产生交叉极化非对角项。

### 3.3 场量、功率和自由空间因子

必须固定以下契约：

- 路径 throughput 是无自由空间 $1/r$ 因子的复电场传递量；每个局部事件只乘 Jones 系数。
- 完整路径长度 $L$ 统一乘 $e^{-jk_0L}$。
- 发射功率、发射/接收天线方向图、有效孔径 $A_e=G_r\lambda^2/(4\pi)$ 和球面扩散只在路径端点/连接处处理一次。
- 若公共输出是无量纲 `path_gain=P_r/P_t`，应由接收端场或功率估计器统一转换，不能在事件内混入 `gain` 经验倍率。

---

## 4. 平滑界面的反射与透射

### 4.1 用波导纳统一 TE/TM Fresnel 公式

设入射介质为 1、透射介质为 2，界面切向波数保持不变：

$$
k_\parallel=k_1\sin\theta_i,
\qquad
k_{z,m}=\sqrt{k_m^2-k_\parallel^2}.
$$

同样选择向介质内部传播或衰减的 $k_z$ 分支。两种极化的波导纳为

$$
Y_m^{\mathrm{TE}}=\frac{k_{z,m}}{\omega\mu_m},
\qquad
Y_m^{\mathrm{TM}}=\frac{\omega\epsilon_m}{k_{z,m}}.
$$

以同一切向电场方向定义反射和透射幅度：

$$
r_q=\frac{Y_1^q-Y_2^q}{Y_1^q+Y_2^q},
\qquad
t_q=\frac{2Y_1^q}{Y_1^q+Y_2^q},
\qquad q\in\{\mathrm{TE},\mathrm{TM}\}.
$$

功率系数为

$$
R_q=|r_q|^2,
\qquad
T_q=\frac{\Re(Y_2^q)}{\Re(Y_1^q)}|t_q|^2,
\qquad
A_q=1-R_q-T_q.
$$

对无损界面应有 $A_q=0$；被动介质数值上允许 $10^{-6}$ 级误差，超出后应报告而不是静默 clamp。全反射时 $T=0$，但 $r$ 的相位必须保留。

注：不同文献对 TM 出射基方向的定义不同，可能导致 $r_{TM}$ 相差一个负号。代码只要始终使用上述局部 frame 与边界条件，不能为了匹配某个标量公式单独翻转 TM 符号。

### 4.2 几何折射方向

无损或弱损耗介质中，用相位折射率 $n_p=\Re(k)/k_0$ 求几何方向：

$$
n_{p,1}\sin\theta_i=n_{p,2}\sin\theta_t.
$$

向量形式令 $\eta=n_{p,1}/n_{p,2}$、$c_i=-\hat{\mathbf n}\cdot\hat{\mathbf d}_i$：

$$
c_t=\sqrt{1-\eta^2(1-c_i^2)},
\qquad
\hat{\mathbf d}_t=\eta\hat{\mathbf d}_i+(\eta c_i-c_t)\hat{\mathbf n}.
$$

若根号内小于零则为全反射。强损耗介质中的折射波一般是非均匀平面波，单一实射线方向不能完整表示；第一版仍以 $\Re(k)$ 构造方向，以复 $k_z$ 精确处理相位和衰减，并在 metadata 标记 `inhomogeneous_wave_approximation=true`。

---

## 5. 有限厚度墙体与多层透射

### 5.1 传输矩阵模型

对入射介质 0、$N$ 个均匀层、出射介质 $s=N+1$，第 $\ell$ 层厚度 $d_\ell$，对每种极化定义

$$
\delta_\ell=k_{z,\ell}d_\ell,
$$

$$
\mathbf M_\ell^q=
\begin{bmatrix}
\cos\delta_\ell & j\sin\delta_\ell/Y_\ell^q\\
jY_\ell^q\sin\delta_\ell & \cos\delta_\ell
\end{bmatrix},
\qquad
\mathbf M^q=\prod_{\ell=1}^{N}\mathbf M_\ell^q.
$$

令

$$
B_q=M_{11}^q+M_{12}^qY_s^q,\qquad
C_q=M_{21}^q+M_{22}^qY_s^q,
$$

则整个层栈的复幅度为

$$
r_q^{\mathrm{stack}}=\frac{Y_0^qB_q-C_q}{Y_0^qB_q+C_q},
\qquad
t_q^{\mathrm{stack}}=\frac{2Y_0^q}{Y_0^qB_q+C_q}.
$$

这会自然包含两界面 Fresnel、Fabry–Pérot 多次内反射和层内吸收。功率仍按

$$
R_q=|r_q^{\mathrm{stack}}|^2,\qquad
T_q=\frac{\Re(Y_s^q)}{\Re(Y_0^q)}|t_q^{\mathrm{stack}}|^2,
\qquad A_q=1-R_q-T_q
$$

计算。

厚层或高损耗层会使普通传输矩阵中的指数条件数很差。生产 CUDA 版本应使用递归散射矩阵（S-matrix/Redheffer star product）或缩放后的导纳递推；上式只作为清晰的数学定义和 CPU double oracle。

### 5.2 两种几何表达模式

#### A. `closed_volume`

建筑几何确实包含墙体前后两个面。射线在入口折射，在介质中继续追踪，在出口再次折射。路径状态必须维护 `medium_stack`，用 face 的 front/back 和闭合体 id 判断入射、出射介质。这是几何上最准确的模式，并能表示斜墙、楔形介质和非平行边界。

#### B. `thin_sheet`

场景只有零厚度单面，但材质给出真实厚度。局部假设为平行板，直接施加层栈 $r/t$。若入口点为 $\mathbf x_i$，墙体总法向厚度为 $d$，层内方向的单位切向分量为 $\hat{\mathbf u}_\parallel$，等效出口点必须同时包含法向移动和层内切向移动：

$$
\mathbf x_e=\mathbf x_i-d\hat{\mathbf n}
+d\tan\theta_t\,\hat{\mathbf u}_\parallel.
$$

这里 $-\hat{\mathbf n}$ 指向墙内；代码应由 front/back side 决定符号，不能假定网格 winding 永远一致。若与“把入射射线直接延伸到背面”的交点比较，平行出射线的横向位移为

$$
\delta\mathbf x_\parallel
=d(\tan\theta_t-\tan\theta_i)\hat{\mathbf u}_\parallel.
$$

并增加层内相位/吸收对应的光程。射线起点移动到等效出口点后沿出射方向继续。必须使用 primitive ignore id 和尺度相关的 epsilon，避免立刻再次击中同一面。多层板中的切向移动为逐层求和 $\sum_\ell d_\ell\tan\theta_\ell\hat{\mathbf u}_\parallel$，不能用单一平均折射率替代。

`thin_sheet` 对有限墙边缘存在局部平行板近似：若偏移后的出口投影越出该 surface group，应判为无效，或退化为显式几何模式，不能从墙边“瞬移”出去。

### 5.3 透射路径拓扑

新增事件 `TRANSMIT_SPECULAR`，并让 path component mask 独立包含 `transmission`。路径长度需要同时保存：

- `geometric_length_m`：时延定位和几何可见性；
- `phase_length_rad` 或直接累计复传播因子：包含各介质 $k_zd$；
- `group_delay_s`：宽带时用 $\partial\arg H/\partial\omega$ 计算，不能一律用几何长度除以 $c_0$。

窄带第一版可以只输出载频复传递函数；宽带版本在每个频点计算材料参数和层栈响应，或对频段内 $H(f)$ 进行矢量拟合后生成 CIR。

---

## 6. 粗糙表面散射

### 6.1 表面统计参数

不能只给一个无量纲 `scattering_coefficient`。最小物理参数为：

- RMS 高度 $\sigma_h$（m）；
- 两个主方向的相关长度 $l_x,l_y$（m）；
- 各向异性主轴角 $\psi$；
- 高度自相关模型或二维功率谱密度 PSD；
- 可选的测量/全波 BSDF 表。

第一版采用高斯相关函数

$$
C(x,y)=\sigma_h^2\exp\left[-\left(\frac{x}{l_x}\right)^2
-\left(\frac{y}{l_y}\right)^2\right].
$$

其 RMS 坡度为

$$
\alpha_x=\sqrt{2}\frac{\sigma_h}{l_x},\qquad
\alpha_y=\sqrt{2}\frac{\sigma_h}{l_y}.
$$

明确写出相关函数约定很重要；如果输入数据使用 $e^{-r^2/(2l^2)}$，上式中的 $\sqrt2$ 会改变。

### 6.2 相干镜面分量

高斯高度分布下，平均高度引入的镜面反射相位差为 $2k_{z,1}h$，相干场衰减因子为

$$
C_r=\mathbb E[e^{-j2k_{z,1}h}]
=\exp[-2(k_{z,1}\sigma_h)^2].
$$

因此相干镜面 Jones 系数为

$$
r_{q,\mathrm{coh}}=r_q C_r,
$$

相干功率比例为 $|C_r|^2$。透射的相干衰减同理由入口、出口几何产生的高度相位差计算，不能直接复用反射的 $2k_z$；对于 `thin_sheet`，应从完整层栈相位对界面高度的一阶变化推导，或第一版仅对外表面粗糙、内表面平滑的模型求值。

该分解使光滑极限 $\sigma_h\to0$ 自动回到 Fresnel 镜面反射。

### 6.3 弱粗糙面：PSD 驱动的小扰动法

> **非当前实现项。** 本节只保留理论边界，当前阶段不实现 SPM kernel，也不参与 `auto` 选择。

当 $k_0\sigma_h\ll1$ 时，首选 SPM，而不是几何光学微表面。定义二维 PSD 的 Fourier 约定

$$
W(\mathbf q)=\int_{\mathbb R^2}C(\boldsymbol\rho)
e^{-j\mathbf q\cdot\boldsymbol\rho}\,d^2\boldsymbol\rho,
\qquad
\frac{1}{(2\pi)^2}\int W(\mathbf q)d^2\mathbf q=\sigma_h^2.
$$

对 6.1 的各向异性高斯相关函数，

$$
W(q_x,q_y)=\pi\sigma_h^2l_xl_y
\exp\left[-\frac{q_x^2l_x^2+q_y^2l_y^2}{4}\right].
$$

散射只从表面获得切向动量差

$$
\mathbf q=\mathbf k_{s,\parallel}-\mathbf k_{i,\parallel}.
$$

将粗糙边界写成 $z=h(x,y)$，在平均平面 $z=0$ 对 Maxwell 切向边界条件作一阶展开。对入射极化 $q$ 和散射极化 $p$，一阶散射振幅可写为

$$
E_{s,p}^{(1)}(\mathbf k_s)
=K_{pq}(\mathbf k_i,\mathbf k_s;
\tilde\epsilon_1,\tilde\mu_1,\tilde\epsilon_2,\tilde\mu_2)
\,\tilde h(\mathbf q)E_{i,q},
$$

其中 $K_{pq}$ 不是拟合系数，而是由平滑界面的 TE/TM 场和四个切向边界条件解出的 $2\times2$ 极化核。集合平均后的单位面积双站散射系数为

$$
\gamma_{pq}(\omega_i\rightarrow\omega_s)
=\mathcal G_{pq}(\omega_i,\omega_s)
\,W(\mathbf k_{s,\parallel}-\mathbf k_{i,\parallel}),
$$

$\mathcal G_{pq}$ 包含波导纳、$k_z$、传播立体角 Jacobian 和 $|K_{pq}|^2$。实现时应由 CPU `complex128` 边界方程生成 $K$，再把同一代数移植到 CUDA；不要从文献中混拼不同相量、PSD 或散射系数约定的常数因子。数值积分

$$
R_{q,\mathrm{diff}}^{SPM}=\sum_p\int_{\Omega_r}\gamma_{pq}\,d\Omega,
\qquad
T_{q,\mathrm{diff}}^{SPM}=\sum_p\int_{\Omega_t}\gamma_{pq}\,d\Omega
$$

给出非相干能量预算，同时为方向采样构造归一化 PDF $p(\omega_s)\propto\sum_p\gamma_{pq}$。二阶 SPM 用于修正相干反射率和能量守恒；若二阶校正不再小于一阶项，说明已超出 SPM 有效域，必须切换模型而不是继续 clamp。

### 6.4 大相关长度粗糙面：各向异性 Beckmann–Smith 微表面

> **非当前实现项。** 当前阶段不实现 Beckmann eval/sample、VNDF 或 rough BTDF；生产散射统一使用第 6.7 节的 Kirchhoff-BSDF。

对局部微表面法线 $\hat{\mathbf m}$，各向异性 Beckmann 法线分布为

$$
D(\mathbf m)=
\frac{\exp\left[-\tan^2\theta_m
\left(\frac{\cos^2\phi_m}{\alpha_x^2}+
\frac{\sin^2\phi_m}{\alpha_y^2}\right)\right]}
{\pi\alpha_x\alpha_y\cos^4\theta_m}.
$$

采用与 Beckmann 分布匹配的 Smith masking-shadowing：

$$
G_2(\omega_i,\omega_o,\mathbf m)
=\frac{\chi^+(\omega_i\cdot\mathbf m)\chi^+(\omega_o\cdot\mathbf m)}
{1+\Lambda(\omega_i)+\Lambda(\omega_o)},
$$

其中 $\Lambda$ 使用 Beckmann 的精确式或稳定有理近似。不能用任意 cosine lobe 替换 $D\,G_2$，因为那会丢失掠射遮蔽、后向增强和各向异性。

单次微表面反射的每极化功率 BRDF 为

$$
f_{r,q}(\omega_i,\omega_o)=
\frac{D(\mathbf m)G_2(\omega_i,\omega_o,\mathbf m)
|r_q(\omega_i\cdot\mathbf m)|^2}
{4|\mathbf n\cdot\omega_i||\mathbf n\cdot\omega_o|},
$$

其中镜面半向量 $\mathbf m\parallel\omega_i+\omega_o$。Jones 版本保留复 $r_q$ 并执行入射/出射局部基变换。

粗糙透射使用折射半向量

$$
\mathbf m\parallel n_i\omega_i+n_o\omega_o
$$

以及从微表面法线到出射方向的折射 Jacobian。实现应直接采用 Walter 等人的 rough-surface BTDF 推导，而不是把反射方向改成 Snell 方向后沿用反射 PDF；二者 Jacobian 不同。该模型已针对真实粗糙透射数据进行过比较，适合作为工程基线（见文末参考资料）。

### 6.5 避免相干项与散射项重复计能

不能同时计算完整 realization 相位积分、Kirchhoff 集合平均 BSDF 和衰减后的 Fresnel delta，否则镜面附近能量会重复。当前 `ensemble_bsdf` 模式采用以下分配：

1. 由平滑层栈得到可用于反射的总预算 $\bar R_q$ 及 $\bar T_q,A_q$；Kirchhoff 核中的局部 Fresnel 项使用相同的 $r_q^{stack}$。
2. 显式 delta 路径分配
   $$R_{q,coh}=|r_qC_r|^2.$$
3. 非相干反射预算
   $$R_{q,diff}=\max(0,\bar R_q-R_{q,coh}).$$
4. 数值积分原始 Kirchhoff lobe。只有当其积分与 $R_{q,diff}$ 的差异小于已声明理论/数值容差时，才做小幅归一化；超出容差说明模型或适用域错误，直接报告失败。
5. 当前高度相位屏不处理粗糙透射，因此 $\bar T_q,A_q$ 保持平滑层栈预算，不另造 diffuse transmission。

积分表按 `(material, frequency bin, cos_theta_i bin, phi_i bin, polarization)` 预计算并缓存。`realization_coherent` 模式直接积分完整相位屏，不再叠加上述 delta/diffuse 分解。强制通过半球能量测试，不能简单把所有分量除以总和来掩盖公式错误。

### 6.6 模型适用性与回退

定义无量纲指标

$$
g=k_0\sigma_h,\qquad c_x=k_0l_x,\qquad c_y=k_0l_y.
$$

- $c_x,c_y\gg1$、局部曲率半径远大于 $\lambda$ 且 RMS 坡度不过大：当前 Kirchhoff 切平面近似适用。
- 高度足以显著改变遮挡/轮廓、出现空腔多次反射或强 shadowing：超出相位屏范围，当前版本报错而不做几何近似。
- $l\sim\lambda$、周期纹理、金属网、复合墙或强交叉极化：当前解析模型不适用，只接受测量/全波 `TabulatedPolarimetricBSDF`。
- 参数缺失时默认 `SmoothSurface`，即只产生 Fresnel 反射/透射；禁止静默猜测粗糙度。

### 6.7 高度图只作为相位屏 / Kirchhoff-BSDF 输入

现阶段明确不做 tessellation、POM/relief 或任何位移几何求交。RayD 始终与平均基础面求交；高度图 $h(u,v)$ 只描述该平均面附近的一次具体粗糙表面实现，用于修正复相位和构造 Kirchhoff 散射，不改变交点、可见性、轮廓或绕射拓扑。

#### 6.7.1 相位屏

RayD 返回平均面交点 $\mathbf x_0$ 和 UV 后，Channel 采样米制高度 $h(u,v)$。令

$$
q_n=(\mathbf k_s-\mathbf k_i)\cdot\mathbf n,
$$

则表面高度引入的相位屏为

$$
P_h(u,v;\mathbf k_i,\mathbf k_s)
=\exp[-j q_n h(u,v)].
$$

符号由第 3 节的 $e^{j\omega t}$、$e^{-jkr}$ 约定固定；镜面反射极限必须得到相位变化幅值

$$
|\Delta\phi|=2k_0|h|\cos\theta_i.
$$

平滑反射的局部 Jones 场因此变为

$$
\mathbf E_r(u,v)=\mathbf R^{stack}(\theta_i)
\mathbf E_i\,P_h(u,v).
$$

透射相位屏不能直接复用反射的 $2k_z h$，而应使用入口/出口介质对应的法向波数差；第一版只对反射散射启用显式高度相位屏，透射仍使用平均平面层栈，避免错误相位。

高度图必须代表固定 `surface_realization_id`。同一表面、相邻射线和阵列单元从同一连续高度场采样；禁止每个 hit 独立随机高度。这样才能保持空间相关、阵列相位、Doppler 和重复求解稳定。

#### 6.7.2 Kirchhoff 相位积分与 BSDF

对表面 patch $A$，极化 $q\rightarrow p$ 的 Kirchhoff 远场振幅写成

$$
E_{s,p}(\mathbf k_s)=
\frac{e^{-jk_0R}}{R}
\int_A K_{pq}(\mathbf x;\mathbf k_i,\mathbf k_s)
E_{i,q}(\mathbf x)
e^{-j(\mathbf k_s-\mathbf k_i)\cdot\mathbf x_0}
e^{-jq_nh(\mathbf x)}\,dA,
$$

其中 $K_{pq}$ 由平均面局部 Fresnel/Jones 边界场和 Kirchhoff 切平面近似给出。高度进入指数相位，而不是通过 normal map 替换路径几何；第一版用平均面法线计算 $K_{pq}$，仅在小/中等 RMS 坡度下启用。

对零均值高斯高度场，集合平均所需的二点相位相关为

$$
\left\langle
e^{-jq_n[h(\mathbf x)-h(\mathbf x')]}
\right\rangle
=\exp\{-q_n^2[\sigma_h^2-C(\mathbf x-\mathbf x')]\}.
$$

将它代回双重面积积分即可得到 Kirchhoff 双站平均强度；数值上在 scene compile 时按 `(material, frequency, cos_theta_i, phi_i)` 预计算角度表：

$$
f^{K}_{pq}(\omega_i,\omega_o)
=\frac{1}{|\mathbf n\cdot\omega_i|}
\frac{dR_{pq}}{d\Omega_o}.
$$

运行时 `eval` 查表并做极化基旋转，`sample` 从按 $f^K|\mathbf n\cdot\omega_o|$ 归一化的二维角度分布采样。表格同时保存正向/反向 PDF、积分能量和适用域标志，供 BDPT MIS 与能量诊断使用。

#### 6.7.3 两种输出模式不能混算

- `realization_coherent`：使用给定高度图的复相位屏，对 patch/路径复场相干积分，输出一个可复现的 speckle/CIR realization。
- `ensemble_bsdf`：使用 $C$ 或 PSD 推导的 Kirchhoff-BSDF，按功率/Stokes 累加，输出集合平均功率。

同一次结果不能把完整 realization 相干积分和同一高度谱的 ensemble BSDF 再相加，否则散射功率重复。若确实需要多尺度混合，必须用互补滤波器把 $h=h_L+h_H$ 分带，并只让 $h_L$ 进入 realization phase screen、$W_H$ 进入 ensemble BSDF，同时在 metadata 报告 cutoff。

高度纹理的低通也必须在**复相位域**完成：一般有

$$
\mathbb E[e^{-jq_nh}]\ne e^{-jq_n\mathbb E[h]}.
$$

因此不能先普通 mip-map 平均高度再指数化。应按频率和 $q_n$ 对复 phasor $e^{-jq_nh}$ 做 footprint 积分，或直接执行 patch quadrature。

#### 6.7.4 归属与适用边界

这一版全部放在 **Channel**：高度图/UV 绑定、纹理采样、相位屏、Kirchhoff 积分、BSDF/PDF 和 realization 缓存都属于电磁材料与积分器逻辑。RayD 无需新增 primitive，只需把现有求交已经支持的 UV 通过 Channel bridge 传出；当前 `core/runtime/raydn.py` 传入空 UV，必须先贯通 structure UV/face-UV。

该模型明确不表示高度导致的遮挡、轮廓变化、真实交点移动、空腔多次反射或位移后的绕射边。当最大高度/坡度使这些效应不可忽略，返回 `phase_screen_geometry_limit_exceeded`，不能继续宣称结果是几何准确的。

---

## 7. Monte Carlo / BDPT 采样与权重

### 7.1 事件选择

在交点按当前 Jones/Stokes 入射态计算标量能量预算：

$$
p_r+p_s+p_t+p_a=1,
$$

分别对应相干反射、非相干散射、透射和吸收。事件概率可取对应功率比例，并设置最小概率保护。选中事件后 throughput 除以事件选择概率，保证估计无偏；吸收事件终止路径。

delta 反射/透射与连续散射必须使用不同 measure：delta 事件的方向 PDF 不是一个有限立体角密度，MIS 中用离散概率处理，不能把它伪装成极窄 cosine lobe。

### 7.2 Kirchhoff-BSDF 方向采样

对当前入射极化状态，将预计算极化 BSDF 收缩成非负功率密度 $f_K(\omega_i,\omega_o)$。连续散射事件的方向 PDF 为

$$
p_K(\omega_o\mid\omega_i)=
\frac{f_K(\omega_i,\omega_o)|\mathbf n\cdot\omega_o|}
{R_{diff}(\omega_i)},
\qquad
R_{diff}=\int_{\Omega_r}f_K|\mathbf n\cdot\omega_o|d\Omega_o.
$$

实现按出射半球的 $(\cos\theta_o,\phi_o)$ 二维表构造 marginal/conditional CDF 或 alias table；采样后仍用原始高精度 BSDF 表求值，避免采样表量化改变物理值。BDPT 的反向 PDF 用交换入射/出射后的同一互易 Kirchhoff 表计算，不能假定与正向 PDF 数值相等。连接策略、BSDF 采样策略和接收端 next-event estimation 使用 balance 或 power heuristic MIS。

### 7.3 相位策略

- 相干事件：累乘精确复 Jones 系数和传播相位。
- 非相干功率图：只累加非负功率贡献，不分配随机相位。
- 随机信道实现：从 `(scene_seed, surface_id, realization_id)` 选择固定高度图，并按连续 UV 采样相位；同一空间相关长度内的射线必须相关，不能每次 hit 独立随机。否则阵列信道、时间连续性和 Doppler 会不物理。

---

## 8. 数据结构与 API 方案

### 8.1 Python 材质

建议新增而不是继续堆叠 `Dielectric` 的匿名参数：

```python
Layer(
    thickness_m: float,
    eps_model: ComplexPermittivityModel,
    mu_model: ComplexPermeabilityModel | None = None,
)

Roughness(
    rms_height_m: float,
    corr_length_x_m: float,
    corr_length_y_m: float,
    principal_axis_rad: float = 0.0,
    correlation: Literal["gaussian", "tabulated_psd"] = "gaussian",
)

PhysicalSurface(
    layers: tuple[Layer, ...],
    outside_medium_id: int = 0,
    backing_medium_id: int = 0,
    geometry_mode: Literal["thin_sheet", "closed_volume"] = "thin_sheet",
    roughness_front: Roughness | None = None,
    roughness_back: Roughness | None = None,
    bsdf_table: PolarimetricBSDFTable | None = None,
)
```

高度图不放进可复用 `PhysicalSurface` 材质本体，而挂在具体 surface 的电磁绑定上，因为它依赖 UV 和具体表面 realization，但不修改几何：

```python
PhaseScreen(
    height: Tensor | HeightTexture,
    height_scale_m: float,
    height_offset_m: float = 0.0,
    realization_id: int = 0,
    mode: Literal["realization_coherent", "ensemble_bsdf"] = "realization_coherent",
    correlation: Roughness | TabulatedPSD | None = None,
    quadrature_tolerance: float = 1e-4,
)

SurfaceAssignment(
    material: PhysicalSurface,
    phase_screen: PhaseScreen | None = None,
)
```

scene compile 负责校验 UV、把高度转成米制 GPU 纹理、计算 $C/PSD$、预计算 Kirchhoff 角度表并登记有效频段。高度纹理保持独立资源；不能烘焙进 RayD 顶点，也不能改变 scene topology。

保留 `Dielectric` 作为单层 `PhysicalSurface` 的便捷构造器。`gain` 不再作为任意能量倍率；若为校准保留，命名为 `calibration_db`，仅允许在最终输出层使用，并在 metadata 显式报告破坏物理归一化。

频散至少支持：常数 $\epsilon_r'+\sigma$、Debye、按频率插值的复 $\epsilon_r(f)$。宽带求解禁止只在载频求一次材料参数后复用。

### 8.2 编译后材质 ABI

将 `MaterialStore.model_params[N,4]` 替换为版本化 struct-of-arrays：

- `material_eps_real/imag`, `material_mu_real/imag`；
- `material_layer_offset/count`；
- `layer_thickness`, `layer_eps_real/imag`, `layer_mu_real/imag`；
- `rough_sigma_h`, `rough_lx`, `rough_ly`, `rough_axis`；
- `scatter_model_id`, `geometry_mode_id`, `bsdf_table_id`；
- `surface_phase_screen_id`, `surface_realization_id`；
- phase texture descriptor、Kirchhoff table offset、PDF/CDF offset 与有效频段；
- face 到 material、surface group、front/back side 的映射。

层表使用扁平 CSR 布局，既支持单层快速路径，也支持可变层数。材质 ABI 增加显式版本号，native binding 在版本不符时立即报错。

### 8.3 路径状态

相干子路径新增：

- `field_real/imag: float32[N,3]`；
- `medium_stack_top` 与当前 `medium_id`；
- `geometric_length_m`；
- `phase_real/imag` 或介质段列表；
- `event_type`, `side`, `surface_id`；
- `pdf_forward_discrete`, `pdf_forward_solid_angle`；
- 对应的 reverse PDF。

第一版限制 medium stack 深度（例如 8），溢出时使路径无效并计数；不要在 GPU 热路径动态分配。

公共 `component_power` 新增 `transmission`、`scattering`；混合路径按事件 mask 统计，同时提供互斥的 `path_class`（例如最后一个非 LoS 事件或完整事件序列），避免 `reflection+transmission` 路径在总功率中被加两次。

---

## 9. CUDA/RayD 实现分层

建议建立单一设备端电磁核心，避免确定性、MC basic 和 BDPT 各自复制公式：

```text
native/channel_native/em/
  complex.cuh
  polarization.cuh
  medium.cuh
  fresnel.cuh
  layer_stack.cuh
  phase_screen.cuh
  kirchhoff_bsdf.cuh
  kirchhoff_table.cuh
  event_sample.cuh
```

核心接口：

```cpp
eval_smooth_interface(material, wi, normal, frequency) -> JonesR, JonesT, RTA
eval_layer_stack(material, wi, normal, frequency) -> JonesR, JonesT, RTA
eval_phase_screen(screen, uv, wi, wo, frequency) -> complex phasor
eval_kirchhoff_bsdf(surface, wi, wo, frequency) -> Mueller/power, pdf_fwd, pdf_rev
sample_kirchhoff_bsdf(surface, wi, frequency, rng) -> wo, weight, pdf_fwd, pdf_rev
sample_surface_event(material, state, rng) -> event, wo, weight, pdf_fwd, pdf_rev
```

### 9.1 与当前仓库的落点

| 层 | 当前落点 | 必要改动 |
|---|---|---|
| 公共材质 | `core/materials.py`, `core/scene.py` | 增加 `Layer/Roughness/PhysicalSurface`，编译为 ABI v2；旧 `Dielectric` 显式转换 |
| face 展开 | `core/material_runtime.py`, `kernels/material.cu` | 从固定 `(eps_r,sigma_e,mu_r,thickness)` 改为 face material id + CSR layer/roughness view，避免复制可变层数组 |
| 相位屏 | `core/runtime/raydn.py`, Channel 新增 phase-screen runtime | 只贯通 RayD 已支持的 UV；高度纹理、phasor、Kirchhoff 表和 realization cache 全部留在 Channel，不修改 RayD 几何 |
| 共享电磁核心 | `native/channel_native/em/*.cuh` | 唯一实现 Fresnel、S-matrix、Jones frame、phase-screen phasor 与 Kirchhoff-BSDF eval/sample |
| 确定性场 | `kernels/deterministic_field.cu`, `deterministic/topology.py` | 镜面透射生成离散拓扑；连续散射用可见 surface patch 的面积求积，不能伪造成唯一驻相路径 |
| MC basic | `montecarlo/basic/solver.py`, `raydn_components.py` | 命中点做接收面 next-event estimation；独立累计 `transmission/scattering`，混合事件用 path mask 去重 |
| BDPT | `kernels/bdpt_subpaths.cu`, `bdpt_connect.cu`, `bdpt/mis.py` | throughput 改 Jones/功率契约；事件采样、正反 PDF、delta/continuous MIS、medium stack |
| 路径导出 | `path/solver.py`, `path/result.py` | 导出事件序列、复系数、几何长度、群时延、正反 PDF 和模型 id；不只导出最终 component id |
| binding/metadata | `bindings.cpp`, `core/kernels/ops.py` | ABI 版本检查、模型适用域/回退、能量误差和无效路径计数 |

确定性粗糙散射的 patch 求积可写成

$$
P_r\approx\sum_{a\in\mathcal V}
P_t\,G_t(\omega_{i,a})\,
\frac{\gamma_a(\omega_{i,a}\to\omega_{o,a})A_a
|\mathbf n_a\cdot\omega_{i,a}|
|\mathbf n_a\cdot\omega_{o,a}|}{(4\pi)^2r_{ta}^2r_{ar}^2}
\,A_{e,r}(\omega_{o,a}),
$$

其中 $\mathcal V$ 只包含 Tx→patch 与 patch→Rx 均可见的面元，$A_a$ 为实际面积。该式用于功率型非相干输出；相干随机表面实现必须对每个面元使用空间相关相位后再累加复场。

执行策略：

1. 单层、无粗糙度走专用 fast path；
2. 层栈角度表、Kirchhoff-BSDF、正反 PDF/CDF 在 scene compile 时生成并上传 GPU；
3. UV 高度采样、复 phasor、Jones frame 变换融合，ensemble 模式独立走 Kirchhoff 表查询；
4. 按 event type 对路径做 stream compaction，再分别追踪反射、透射、散射射线，减少 warp divergence；
5. `closed_volume` 透射进入下一次 RayD trace；`thin_sheet` 直接计算等效出口并做出口/边界有效性检查；
6. 所有 hit offset 使用 `max(abs_position*ulp_scale, scene_diagonal*relative_eps, 1e-6 m)`，不能固定一个对所有场景都相同的 epsilon。

---

## 10. 分阶段落地计划

### Phase 0：物理契约和 CPU oracle

- 固定相量、法线、TE/TM 基、场/功率、路径增益和时延约定；
- 用 Python/CPU `complex128` 实现 Fresnel、层栈、Jones frame 和数值半球积分；
- 建立 JSON/NPZ golden vectors，CUDA 只对 oracle 做一致性测试；
- 修正 BDPT 中功率反射率乘场幅的语义错误。

**完成标准：** 正入射、Brewster 角、全反射、PEC 极限、无损界面 $R+T=1$ 全部通过。

### Phase 1：平滑有限厚度透射

- 新增 `transmission` component 和材质 ABI v2；
- 先实现 `thin_sheet` 单层 S-matrix，再扩展多层；
- 确定性 solver 输出复透射场、相位、时延；
- 增加出口偏移、边缘检查和透射后的可见性。

**完成标准：** 与 CPU oracle 的复 $r/t$ 相对误差小于 `2e-5`（float32 正常条件区间），高损耗/厚层不产生 NaN/Inf。

### Phase 2：全路径 Jones 极化

- 将 BDPT scalar complex throughput 迁移为 `complex3`；
- 统一反射、透射、绕射后的基变换；
- 公共结果可选输出 $2\times2$ MIMO/Jones channel，而不仅是标量 path gain。

**完成标准：** 多次事件后始终满足 $\mathbf E\cdot\hat{\mathbf d}\approx0$；互易场景交换 Tx/Rx 后 Jones 矩阵满足相应转置关系。

### Phase 3：相位屏与 Kirchhoff-BSDF

- Channel 贯通 structure UV/face-UV，RayD 仍只与平均面求交；
- 增加米制 `PhaseScreen`、稳定 `surface_realization_id` 和 GPU 高度纹理缓存；
- 实现 $e^{-jq_nh(u,v)}$ 复相位采样与确定性 patch quadrature；
- 由高度相关函数/PSD 预计算极化 Kirchhoff-BSDF、积分能量和正反 PDF 表；
- 分离 `realization_coherent` 与 `ensemble_bsdf` 输出，禁止重复计能；
- 加入高度/坡度适用域检查，超限时报告而不是切换到位移求交。

**完成标准：** 常量高度给出解析相位移；正弦/高斯高度屏与 CPU complex128 面积分一致；固定 realization 可复现且空间相关；集合平均 realization 收敛到 Kirchhoff-BSDF；每角度/极化满足 $R+T+A\le1+10^{-4}$。

### Phase 4：BDPT/MIS 与空间相关随机信道

- delta/continuous 混合 measure 的 MIS；
- 双向透射 reverse PDF 和 $\eta^2$ measure 变换；
- 引入 realization id、空间相关表面相位和稳定 RNG；
- 导出完整事件序列、每事件 PDF 和能量诊断。

**完成标准：** 不同合法采样策略均收敛到同一结果；固定 seed/realization 可复现；增加样本数后方差按 MC 规律下降。

### Phase 5：高精度扩展

- Kirchhoff patch 自适应复积分与误差估计；
- 空间/时间相关的动态 phase-screen realization；
- 多频率 phasor footprint 积分和宽带 speckle/CIR；
- `closed_volume` medium stack；
- 测量/全波 polarimetric BSDF 表；
- Debye/表格频散与宽带群时延。

---

## 11. 验证矩阵

### 11.1 解析单元测试

1. 相同介质：$r=0,t=1$。
2. 无损介质正入射：与阻抗公式一致，$R+T=1$。
3. TM Brewster 角：无损非磁介质 $R_{TM}\to0$。
4. 全反射：$T=0,|r|=1$ 且相位正确。
5. PEC 极限：$R\to1,T\to0$。
6. 零厚度 slab：回到无界面结果；单层厚度扫描出现正确 Fabry–Pérot 周期。
7. 高损耗厚层：$T\to0$，无上溢/下溢。
8. Jones 横向性：出射场与出射方向正交。

### 11.2 物理不变量

- 被动性：$R,T,A\ge-\epsilon$ 且 $R+T+A=1\pm\epsilon$；
- Helmholtz 互易性：交换入射/出射与极化后满足对应 BSDF/Jones 关系；
- 旋转不变性：同时旋转场景、极化和粗糙主轴，结果只发生坐标变换；
- 基础网格不变性：保持同一平均面和 UV 参数化时，改变基础三角剖分不改变相位屏结果；
- 相位积分收敛性：增加 patch quadrature 样本后复场收敛到 CPU complex128 参考；
- 频谱分带不变性：在有效重叠区移动 cutoff 不应造成可见的能量跳变或重复散射；
- 光滑极限和各向同性极限连续；
- 半球积分验证 Kirchhoff-BSDF 的归一化、被动性和互易性。

### 11.3 数值与统计测试

- CPU complex128 对 CUDA float32，覆盖入射角、频率、损耗和厚度的对数网格；
- Kirchhoff-BSDF 出射方向采样对预计算 PDF 做 chi-square/KS 或分箱置信区间测试；
- MC 均值对数值半球积分，报告偏差和置信区间，而非只比较单一 seed；
- grazing incidence、接近临界角、$\alpha\to0$、极端各向异性作为专项稳定性测试；
- Munich 场景做性能/回归测试，但不把任何现有近似实现当物理真值。

### 11.4 外部验证

优先级依次为：

1. 可解析 Fresnel/平行板；
2. 独立 CPU 电磁 oracle；
3. 公开或自测的材料复介电常数和双站散射测量；
4. 小场景 FDTD/FEM/MoM 扫频结果；
5. 实际 Tx/Rx 测量的幅度、相位、极化和时延联合对比。

只拟合总接收功率不足以区分材料损耗、粗糙散射和几何误差。校准至少要覆盖多个入射角、两种线极化和多个频点。

---

## 12. 性能预算与降级档位

建议提供三个明确档位：

- `physical_fast`：单层平滑 slab、Jones 相干路径、无粗糙散射；适合作为默认替代当前近似。
- `physical_rough`：集合平均 Kirchhoff-BSDF、角度/PDF 表和非相干散射；用于生产级 radiomap。
- `physical_reference`：给定 realization 的相位屏 patch 复积分、多层 S-matrix、双精度 CPU/CUDA 校验或表格 BSDF；用于验证和小规模高精度计算。

性能监控至少记录：各 event 数量、吸收终止数、TIR 数、medium-stack 溢出、无效出口、Kirchhoff 表命中率、每种 event 的方差、`R+T+A` 最大误差、phase texture/角度表显存、phasor 采样时间和 patch quadrature 时间。RayD 的 GAS 和求交路径保持不变，不纳入高度图实现成本。

---

## 13. 推荐的第一批代码改动

按风险和依赖顺序，第一批提交应只做：

1. 新建共享 `em/*.cuh`，迁移而不改变现有 Fresnel 行为；
2. 建立 CPU complex128 oracle 与 golden tests；
3. 明确并修复 BDPT 场幅/功率 throughput 契约；
4. 材质 ABI v2 和 `PhysicalSurface`，保留旧 `Dielectric` 兼容转换；
5. 实现平滑单层 `thin_sheet` 透射及解析测试；
6. 贯通 UV，加入 `PhaseScreen` 和常量/正弦高度的 complex128 相位 oracle；
7. 实现 realization patch 复积分，再实现集合平均 Kirchhoff-BSDF/PDF 表；
8. 最后接入连续散射事件与 BDPT MIS。

不要在同一个提交中同时更换 Fresnel、极化状态、MIS 和粗糙分布，否则结果变化无法归因。

---

## 14. 参考资料

1. P. Beckmann, A. Spizzichino, *The Scattering of Electromagnetic Waves from Rough Surfaces*, Pergamon Press, 1963. Kirchhoff 粗糙面相干/非相干分解的经典来源。
2. M. Sylvain, “Diffuse reflection by rough surfaces: an introduction,” Comptes Rendus Physique 6(6), 2005. 对 Kirchhoff 切平面近似、SPM 及 shadowing/multiple-scattering 局限作了清楚区分。<https://doi.org/10.1016/j.crhy.2005.06.014>
3. M. A. Karam, R. S. McDonough, “Analytic Models for Bistatic Scattering from a Randomly Rough Surface with Complex Relative Permittivity,” ITU Journal 2(1), 2019. 给出复介电常数下的极化 Kirchhoff/physical-optics 双站系数。<https://pmc.ncbi.nlm.nih.gov/articles/PMC7323588/>
4. M. Franco et al., “Validity of the Kirchhoff approximation for the scattering of electromagnetic waves from dielectric, doubly periodic surfaces,” JOSA A 34(12), 2017. 用全数值解评估三维全极化 Kirchhoff 近似的适用范围。<https://doi.org/10.1364/JOSAA.34.002266>
5. 本方案的平滑层栈部分直接来自 Maxwell 边界条件的波导纳/传输矩阵形式；生产实现采用数值更稳定的散射矩阵递推，但必须与本文的 complex128 传输矩阵 oracle 等价。

---

## 15. 最终验收定义

只有同时满足以下条件，才可称为“散射和透射已物理实现”，而不仅是增加了两个 component 名称：

- 复场、功率、极化和相位约定在全部 solver 中一致；
- 平滑界面和有限厚度层栈通过解析复数幅度测试；
- 粗糙散射由米制表面统计量或测量表驱动；
- 反射、散射、透射、吸收逐角度逐极化守恒；
- delta 与连续分布的 PDF/MIS measure 正确；
- 非相干散射不会被错误地相干累加；
- 结果通过互易性、极限、统计收敛和至少一种独立外部参考验证；
- 给定高度图 realization 的复相位可复现且保持空间相关，集合平均与 Kirchhoff-BSDF 一致；
- metadata 明确声明相位屏不修改交点、遮挡、轮廓与绕射拓扑；
- 所有近似与回退都在 metadata 中可见。
