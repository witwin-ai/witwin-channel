# Channel — Differentiable Solving: Capability Boundary

Status as of plan 07 completion (2026-07-14). This document is the authoritative
statement of what the automatic-differentiation (AD) support does and does not
cover. It is written to be honest about the edges: where a gradient is exact,
where it is a documented structural zero, and where the solver refuses to
differentiate rather than return a misleading value.

The machine-readable form of this boundary lives in
`src/witwin/channel/capabilities.py` (`ad_contract` and each solver's
`ad_excluded`). If this document and that file disagree, the file is the source
of truth and this document is stale — reconcile them.

## 1. Differentiable solvers and modes

| Solver | `ad_mode` accepted | Field semantics under AD |
|---|---|---|
| `path` | `none`, `jvp`, `vjp` | complex path coefficients (magnitude, phase, delay) |
| `deterministic` | `none`, `jvp`, `vjp` | complex path coefficients |
| `montecarlo.basic` | `none`, `jvp`, `vjp` | real incoherent power map only |
| `montecarlo.bdpt` | `none` only | — (AD refused by design) |

`jvp` is forward mode (directional derivative / tangent); `vjp` is reverse mode
(`.backward()`). The two agree through the inner-product duality, which is
tested.

## 2. Differentiable parameters

Reverse mode: mark the leaf `requires_grad_(True)` and call `.backward()` on a
loss.

- Materials: `scene.compile().materials.<eps_r | sigma_e | gain | layer_thickness_m | ...>`
- Frequency: `Scene(..., frequency=torch.tensor(f0, device="cuda", requires_grad=True))`
  (a 0-d tensor; the scene reads it once per solve).
- TX/RX position: `Transmitter(position=leaf)` / `ReceiverPoint(position=leaf)`.
- Mesh vertices: `Structure(vertices=leaf, ...)`.

Forward mode: `torch.autograd.forward_ad.dual_level` (make the leaf a dual), or
`torch.func.jvp` at the Function level.

### 2.1 Coverage matrix (plan 07 section 9.3, as delivered)

D = deterministic, P = path, M = montecarlo.basic. A cell lists the solvers that
differentiate that (parameter x interaction). Blank = physically inapplicable.

| parameter | LoS | 1 reflection | multi-reflection | transmission / multilayer | UTD diffraction | coupled R-D |
|---|---|---|---|---|---|---|
| `eps_r` | — | D / P / M | D / P | D / P / M | D / P / M | P |
| `sigma_e` | — | D / P / M | D / P | D / P / M | D / P / M | P |
| `thickness` | — | — | — | D / P / M | — | — |
| frequency | D / P / M | D / P / M | D / P | D / P / M | D / P / M | P |
| TX / RX position | D / P / M | D / P (see 2.2) | D / P | D / P (M-TX see 2.2) | D / P (M-TX see 2.2) | P |
| mesh vertex | D / P (zero, 2.2) | D / P | D / P | D / P | D / P | refused (2.3) |

- D / P differentiate complex coefficients (including phase and delay). `path_length_m`
  and `delay_s` are differentiable outputs of a geometry-AD solve, so a
  time-of-arrival loss works.
- M differentiates the real power gain only.
- Coupled reflection-diffraction paths are a path-solver feature, so the coupled
  column is P only.

### 2.2 Structural zeros (delivered as exact zeros through a live graph)

These are not missing gradients: the leaf reaches a loss and receives a value of
exactly zero, because the continuous dependence is genuinely zero. They are
pinned by tests so they cannot silently become nonzero-but-wrong.

- **LoS x mesh vertex**: a line-of-sight path touches no face, so no vertex moves it.
- **montecarlo.basic reflection map x TX position**: the Sionna-style radiomap
  deposit weight `|Gamma|^2 * solid_angle * (lambda/4pi)^2 / (A_cell * |cos|)`
  depends only on the frozen sampled direction, the face normal and the
  materials; the `1/d^2` spreading lives in the frozen ray density and the cell
  binning. A moving transmitter only re-bins deposits (discrete, frozen), so the
  continuous part is identically zero.

### 2.3 Refused combinations (fail loudly before any launch)

The contract forbids returning a misleading gradient, so these raise (with a
named reason) instead of silently degrading:

- **`montecarlo.bdpt` + any AD**: the coherent BDPT contribution and its
  three-way discrete event sampling (reflect / transmit / scatter, probabilities
  that depend on material and frequency) are random topology, deferred to a
  future plan.
- **scattering component + AD** (all solvers): Kirchhoff rough-surface scattering
  is not differentiated this cycle.
- **coupled reflection-diffraction x mesh vertex** (path): the coupled stationary
  re-solve treats the wall plane and edge tables as frozen winners, so a vertex
  gradient would be incomplete. Registered as `xfail(strict=True)` and raised at
  the seam.
- **frequency AD over frequency-dependent (dispersive) materials**: `Scene.compile()`
  freezes material records at the primal frequency, so a frequency gradient would
  miss `d(material)/d(frequency)`. Refused; use a constant-material scene for
  frequency AD, or drop the frequency leaf for materials-only AD.
- **`mu_r`**: held fixed (not a differentiable input this cycle).
- **Receiver-grid positions**: a `ReceiverGrid` is generated natively from
  origin / axes / spacing and exposes no per-receiver position leaf. For a
  radiomap the grid is the output, not a parameter, so there is nothing to
  differentiate (this is not a refusal, just an absence of a leaf).

## 3. Contract semantics

- **Fixed-winner.** Topology discovery stays native and detached: the discrete
  winner — path topology, sampling and trace tapes, validity masks, primitive
  ids, surface-group ids, polarization frames, and the normal-flip branch — is
  frozen. Gradients flow only through the continuous geometry and EM response of
  that already-selected path.
- **No discontinuity estimator.** Visibility and topology discontinuities (a path
  appearing or disappearing, a shadow-boundary transition) are explicitly out of
  contract. There is no silhouette / edge-sampling estimator. A solve that lands
  on such a boundary does not attempt a continuous gradient; the boundary is a
  discrete event outside the fixed-winner contract.
- **`ad_mode="none"` is byte-identical to the pre-AD primal.** No autograd graph,
  no companion launches, zero tape bytes, and — since the AD-4b instrumentation
  fix — no timing synchronize. The metadata for a none-mode solve reports
  `forward_time_ms == 0.0` and `peak_memory_bytes == 0` precisely because it is
  not instrumented; a nonzero value there would mean the primal was paying an AD
  cost.
- **AD-mode forward values for diffraction / coupled rows.** Pure diffraction and
  coupled fields are re-evaluated inside channel under AD (RayD's order-1
  export is detached and has no adjoint). The re-evaluation is gated against the
  export to a tight tolerance, but it is a re-evaluation, so `ad_mode="jvp"/"vjp"`
  is not guaranteed bit-identical to `none` for those two components (it is for
  LoS / reflection / transmission).

## 4. How gradients are computed (architecture)

Two orthogonal layers composed on one torch graph. Both are GPU-first: every hot
path is a CUDA kernel and the `torch.autograd.Function` layer is dispatch only
(save-for-backward plus one native call per direction).

- **Layer 1 — geometry (RayD).** Hit points, interaction normals and path length
  come from RayD's own fixed-winner EPC chain adjoint (`reflection_epc_paths_backward/_jvp`),
  a plain CUDA kernel with no OptiX (the winner is frozen, so only continuous
  geometry remains). The discovery raygen and the adjoint share one chain
  implementation (`shared/reflection/epc_chain.h`). channel never
  re-solves hit geometry in torch.
- **Layer 2 — EM response (channel).** Native CUDA backward/jvp companions
  of the field kernels (`field_free_space`, `field_reflection_sequence`,
  `field_transmission_sequence`) and the accumulators. UTD diffraction uses
  RayD's UTD pair forward, templated over its scalar type so the float
  instantiation is the production forward and the dual instantiation is the exact
  derivative.

## 5. Known performance characteristics

The AD path is correct and CUDA-native, but it is not free, and a couple of
choices trade FLOPs for a single-forward-implementation guarantee. Documented so
nobody assumes the reverse pass is as cheap as the forward:

- **Seeded-dual backwards re-evaluate the forward per parameter.** The wedge and
  coupled backwards run one dual pass per differentiable input (bounded structural
  constants, ~25 per row for wedge/coupled; ~12 per lane for the MC diffraction
  tape; the transmission layer stack is O(L^2) in layer count). None scale with
  path/ray count, but the per-row cost is tens of times the forward. This is the
  documented AD-4a choice (one forward implementation, exact derivative) and is
  the main remaining optimization headroom.
- **Scalar-gradient atomics.** `grad_frequency` (1 address) and the MC
  `grad_source` (3 addresses) accumulate with `atomicAdd` from every row/lane; a
  block reduction would cut the serialization on large radiomaps.

Measured on small analytic scenes (frequency gradient, tensor-frequency): the AD
forward is within ~1.0-1.15x of the none-mode forward and the reverse pass is a
fraction of it. Device-to-host synchronizations during a solve: `ad_mode="none"`
takes 0; a tensor-frequency AD solve takes 2 (one for the compiled material
cache token, one solve-level read shared by discovery, fields and accumulation)
rather than one per field kernel; the reverse pass takes 0. Material and geometry
reverse mode carries additional per-parameter dual cost (the seeded-dual choice
above) beyond the frequency figure. None of this affects `ad_mode="none"`: the
none-mode primal is byte-identical and uninstrumented.

## 6. Verification

Gradients are validated against, in order of strength: central finite differences
of the primal (per solver x parameter x interaction), JVP-vs-VJP inner-product
duality, `torch.autograd.gradcheck` on the float64 Function subset, forward parity
against a complex128 reference oracle, and — for the RayD geometry and UTD
adjoints — standalone host finite-difference programs that validate the analytic
adjoint before any CUDA integration. Monte Carlo gradients additionally get a
cross-seed variance check. Tolerances and finite-difference steps are centralized
in `tests/ad/_tolerances.py`. End-to-end usability is shown by Adam recovering a
material and a transmitter position from a perturbed start.
