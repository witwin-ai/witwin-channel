# Sionna 2.0.1 RadioMapSolver — Gradient Reality Check

Date: 2026-05-20
Status: V1.1 / V1.2 probe complete; closes the "does Sionna's RadioMapSolver flow geometry gradients" question raised in [`docs/dev/plans/26-mc-sionna-parity-acceleration-plan.md`](../plans/26-mc-sionna-parity-acceleration-plan.md) W1.

## Setup

- Source: `e:\Code\witwin-platform\sionna-rt-reference-2.0.1\` (in-tree)
- Runtime: `witwin2` conda env — **drjit 1.2.0, mitsuba 3.7.1** (NOT the pyproject-pinned drjit 1.3.1 / mitsuba 3.8.0). Sionna imports and runs cleanly under the older substrate; trajectory replay (Dr.Jit 1.3 feature) is therefore NOT available in these measurements.
- Variant: `cuda_ad_mono_polarized`
- Scene: 3 separated cubes (concrete material), 1 TX at (-12, -12, 6) looking at (0, 0, 1.5)
- Radio map: 40 × 40 m planar grid at 0.5 m cell size centred at (0, 0, 1.5)
- Solver config: `samples_per_tx=200_000` (AD probes), `1_000_000` (FD-vs-AD comparison), `max_depth=3`, specular+los only.
- Metric: `dr.sum(radio_map.path_gain.array)`
- Probe script: [`.codex_tmp/sionna_radiomap_grad_probe.py`](../../../.codex_tmp/sionna_radiomap_grad_probe.py)

## Result Matrix

| Probe | Symbolic (default) | Evaluated |
|---|---|---|
| Material `eps_r` AD | **AD chain BROKEN**: `dr.grad_enabled(metric) = False`, `dr.backward(metric)` raises `RuntimeError: ... the argument does not depend on the input variable(s) being differentiated` | **AD works**: `grad_eps = 1.96e-6`, FD `central_diff = 1.40e-6` (Δε=0.5). Same sign, magnitude ratio ≈ 1.4 — within MC noise at 200k samples. |
| Geometry `vertex_positions` AD | AD chain claims to exist (`grad_enabled_on_metric = True`) but `dr.grad(vp)` is identically zero across all 72 components | **AD reports 24 / 72 nonzero**, with a *structural pattern*: exactly 1 axis per vertex carries gradient (`n_nonzero_x = n_nonzero_y = n_nonzero_z = 8`; cube has 8 face-vertices per cube × 3 cubes = 24). `grad_l2_norm = 5.54e-6`, `grad_max_abs = 3.57e-6`. |

### FD vs AD per-component check (evaluated mode, 1 M samples)

Picked the top-5 components by `|AD grad|` plus one near-zero control. Three FD step sizes per component to assess MC noise sensitivity:

| comp | vertex | axis | AD grad | FD δ=0.01 | FD δ=0.05 | FD δ=0.2 | Verdict |
|------|--------|------|---------|-----------|-----------|----------|---------|
| 4    | 1  | y | +3.58e-6 | -5.73e-5 | -1.08e-5 | -1.79e-6 | sign mismatch |
| 51   | 17 | x | +2.72e-6 | -5.76e-5 | -1.11e-5 | -1.80e-6 | sign mismatch |
| 1    | 0  | y | -2.69e-6 | -6.43e-5 | -1.70e-5 | -6.19e-6 | sign match, magnitude 2.3× off (δ=0.2) |
| 54   | 18 | x | -1.16e-6 | -5.76e-5 | -1.50e-5 | -4.37e-6 | sign match, 3.8× off |
| 10   | 3  | y | -9.81e-7 | -1.27e-5 | -3.76e-6 | -1.56e-6 | sign match, 1.6× off |
| 27   | 9  | x (control) | 0.0 | 0.0 | 0.0 | 0.0 | both zero ✓ |

FD reduces by ~30× as δ grows from 0.01 to 0.2, which is the classic MC-noise floor signature: smaller perturbations get drowned out. At δ=0.2 the FD is the cleanest available, and even there the agreement with AD is poor on 2 of 5 components.

### `detach` audit (source-level)

```
path_solvers/paths_buffer.py:873-887  detach_geometry()  — PathSolver only
path_solvers/path_solver.py:219       paths_buffer.detach_geometry()
sliced_integrator.py:83, 179, 305     dr.detach(ray) / dr.detach(si)  — render path, not RadioMapSolver
radio_materials/radio_material.py:460 dr.detach(dr.rsqrt(probs))      — Russian-roulette weight normalization (correct)
```

**RadioMapSolver does not call `detach_geometry()`** — the V1.1 inference that radio-map inherits the PathSolver detach was wrong. RadioMapSolver does not construct a `PathsBuffer`; it scatters directly into a tensor.

## What this actually tells us

1. **Sionna's docstring is accurate for symbolic mode**: "Symbolic mode is the fastest mode but does not currently support backpropagation of gradients." Empirically confirmed at the material level (chain literally broken) and effectively confirmed at the geometry level (chain present but flat-zero). Dr.Jit 1.3 trajectory replay would not flip this automatically — Sionna would have to wire it up, and 2.0.1 does not.
2. **Evaluated mode for materials is real AD**: AD vs FD within MC noise at 200k samples. Slow path, correct.
3. **Evaluated mode for geometry is not what V1.1 thought**: gradients DO flow — 24 of 72 components nonzero — but the pattern (exactly one axis per face-vertex) is suspicious. Two interpretations:
   - **Charitable**: Sionna is differentiating the face-plane equation, which mathematically gives nonzero derivative only along the face normal — in-plane vertex motion doesn't change the plane. This would be correct geometrically.
   - **Uncharitable**: the AD chain is partially detached, capturing only a subset of the dependence.
4. **FD at 1 M samples cannot adjudicate** between the two interpretations. Per-vertex contribution to the global sum is ≈ 1e-6, MC noise floor is ≈ 1e-5. Need 100 M+ samples for the FD to resolve individual components — or a much smaller ROI metric to amplify per-vertex sensitivity.

## Implication for plan 26 W1

The previous W1 revision (2026-05-20 morning) claimed Sionna "cuts geometry gradient on either mode" because of `detach_geometry`. That was wrong for the radio-map path. The honest story:

| Sionna 2.0.1 capability | Symbolic (fast) | Evaluated (slow) |
|---|---|---|
| Material AD | ❌ broken (docstring + empirical) | ✅ works |
| Geometry AD via RadioMapSolver | ❌ chain present, gradients all zero | ⚠️ partial — 1 axis per face-vertex, FD-correctness unverified at 1M samples |
| Geometry AD via PathSolver | (same disclaimer) | ❌ explicitly `detach_geometry()` at `path_solver.py:219` — the `PathsBuffer` strips vertex/wedge gradients before image-method/field-calculator run |

The witwin differentiator that **does** hold up against this evidence:

- **In fast mode, witwin can flow material AND geometry gradients; Sionna cannot.** This is the dominant practical claim.
- **In slow / AD mode**, both can flow material grads. For geometry grads via radio map, Sionna technically reports something but its correctness is unverified at reasonable sample counts; witwin's tape-replay representation is at minimum auditable (integer topology + sparse VJP scatter is straightforward to FD-check).
- **For PathSolver-equivalent functionality** (the closer analogue to witwin's path-list mode), Sionna's `detach_geometry` cut is real and structural — that part of V1.1 stands.

## Follow-ups

- **V1.2** (still pending): re-run on drjit 1.3.1 substrate (would require installing into an isolated env — not done here per "don't pollute witwin2" instruction) to confirm trajectory-replay does *not* change the symbolic-mode results without Sionna wiring it up explicitly.
- **V1.3** (still pending): same probes against witwin's MC path for direct comparison. Numbers go into the pitch.
- **Verification at higher sample count**: re-run FD-vs-AD on geometry at samples_per_tx ≥ 10 M, with a small ROI metric that focuses on a single reflection focal point, to determine whether evaluated-mode geometry AD is "geometrically correct but face-normal-only" or "partially detached". This decides whether the witwin claim is "Sionna's geometry AD is incomplete" or "Sionna's geometry AD is restricted in scope".

## Raw artefact

Full JSON output of the final probe run is captured by the script — re-run with:

```
"C:/Users/Asixa/miniconda3/envs/witwin2/python.exe" \
    e:/Code/witwin-platform/channel/.codex_tmp/sionna_radiomap_grad_probe.py
```
