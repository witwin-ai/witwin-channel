# Deterministic/full-wave boundary audit — main migration

- **Audit date:** 2026-07-29
- **Channel target:** `main`, rebased implementation on the post-refactor owner layout
- **RayD target:** `codex/boundary-physics-channel-pin` at `9ab3bf6326efe6ff22f079638d65be76f4b08fc8`
- **Runtime environments:** Channel `witwin2`; RayD `witwin3`
- **Boundary taper:** disabled (`isb_boundary_taper=False`) in every deterministic benchmark
- **Ground truth:** read-only archived FDTD arrays; hashes are recorded below

## Outcome

The pre-refactor corrections were reimplemented in the current owners instead of
replaying obsolete file paths. The migration keeps full-wave data unchanged,
removes the known numerical/topological confounders, and leaves the remaining
ISB/RSB/vertex deficit visible. No smoothing is part of this change.

The implemented corrections are:

1. RayD stationary wedge geometry uses the local-coordinate weighted-axial
   line-Fermat minimizer, including translation/axis-reversal contracts and
   rejection of the non-unique double-axial family.
2. RayD direct diffraction preserves the unnormalised transverse short-dipole
   projection, hence the physical `sin(theta)` amplitude and axial null.
3. RayD finite-edge visibility offsets only the visibility probe. The true
   Fermat point remains the field, delay, and exported interaction point.
4. The Mend locality weight is evaluated until true exponential underflow; the
   former hard `w=1e-3` cutoff is removed.
5. Channel coupled D-D discovery replaces the uncertified fixed-16 alternating
   iteration with a local-coordinate convex reduced root bracket, FP32 position
   certificate, and the ordinary UTD three-leg distance domain.
6. Benchmark configuration explicitly disables the legacy visual taper and
   supplies the current explicit reference-frequency contract.

## Ownership after the main refactor

| Concern | Current owner |
|---|---|
| Pure-wedge source state and line-Fermat primitive | RayD `shared/include/rayd/shared/utd/utd_math.h` |
| Pure-wedge finite-edge visibility probe | RayD `shared/include/rayd/shared/multipath/diffraction_paths_algo.h` |
| Coupled D-D stationary topology | Channel `native/channel/kernels/coupled.cu` |
| Native facade wording/contracts | `witwin/channel/kernels/geometry.py` and `native/channel/binding/rayd.cpp` |
| Full-wave validation and rendering | `benchmarks/fullwave_validation` |

## Direct regression evidence

RayD:

- `rayd_torch_diffraction_wedge`: pass. It covers the line-Fermat analytic
  value, translation, axis reversal, non-unique axial rejection, short-dipole
  `sin(theta)`, axial null, Dual `cos(theta)` tangent, the removed Mend cutoff,
  and the existing native JVP/VJP contraction test.
- `test_cuda_multipath_parity`: 5/5 pass.
- Correctly isolated API6 Torch unittest discovery: 146 run, 19 explicit external
  parity skips, zero failures/errors.

Channel regression evidence from the clean implementation lineage:

- coupled D-D topology + diffraction/coupled AD + validation: 60/60 pass;
- full validation module after renderer tests: 27/27 pass;
- native CUDA unit-contract suite: 2,190 passed, 2 skipped;
- four-solver E2E: 14 passed; no-fallback: 26 passed; AD core: 121 passed;
- nightly full coverage: 2,686 passed, 8 skipped, 1 expected failure, with the
  coverage policy passing;
- Munich parity: 10 passed, 6 explicit skips; full AD: 329 passed, 2 skipped,
  1 expected failure; the statistics acceptance gate passed;
- Python compileall, Ruff, mypy, import graph, binding coverage, orphan-module,
  single-definition, shared-math, source-prose, and compact-signature checks pass;
- nightly multi-architecture wheel build, independent `witwin.core` wheel,
  installed-wheel native identity/PE smoke, and duplication gate pass;
- generated-scene Phase-E performance, peak-memory preflight, four-solver
  cold-start, full solver-scaling axes, and 256/1024-structure compile scaling
  pass.

The monolithic release wrapper cannot complete its full external-city Phase-E
case on this refactored `main`: `benchmarks/phase_e_scenarios.py` explicitly
fails before numerical work because the Core-owned Munich/SF XML importer has
not yet been restored, and the full assets require an external root. The
reduced generated profile passes. This pre-existing release-infrastructure gap
is recorded rather than bypassed with a synthetic fallback; it is not evidence
for or against this boundary-physics change.

## Immutable full-wave references

| Reference | SHA256 after rerender |
|---|---|
| single cube metal | `D2B87B6C4AD6A693C7F4A35C84B44D5C504AD789CB5BB3C2B60FBF4887B51AB0` |
| single cube empty | `AC0DB283958F8034BE3E1EFB64E28C6B3A6730ABC74976483EE3402218A2AAAB` |
| three cube metal | `E6296DD0C6F12F91DF80565C118041BFC2566D9031CC1CCCB2ABCC86932A7944` |
| three cube empty | `26C17A39577BA601C17771982B235218E0C137B5745CBC5A7B5536D16BEC8172` |

The historical files use the predecessor schema name. For the benchmark only,
new artifact copies were written with the current schema while preserving every
numeric array and metadata field exactly. No archive file was rewritten.

## Fresh benchmark results

### Single cube

- deterministic grid: `256 x 256`; FDTD source grid: `320 x 320`;
- calibrated magnitude NMSE: `0.0350500961`;
- magnitude correlation: `0.8761177011`;
- ISB deterministic jump p95: `2.1035169725 dB`;
- RSB deterministic jump p95: `3.3675912373 dB`;
- the single-cube result metadata records zero boundary taper.

### Three cubes (`three_cube_320`)

| Metric | Coupled OFF | Coupled ON |
|---|---:|---:|
| Path count | 710,523 | 3,563,755 |
| Coupled rows | 0 | 127,117 |
| Envelope NMSE | 0.0890 | 0.0912 |
| Magnitude correlation | 0.8040 | 0.7997 |
| Magnitude RMSE | 4.040 dB | 3.989 dB |
| ISB p95 excess | -0.326 dB | 1.842 dB |
| RSB p95 excess | 3.933 dB | 0.848 dB |
| Deterministic-only max jump | 48.106 dB | 30.225 dB |

The coupled term materially reduces the RSB excess and worst deterministic-only
jump. It does not close ISB or vertex/shadow physics: the ISB p95 gets worse in
this case while the global magnitude fit changes only slightly. Compared with
the 2026-07-27 coupled-ON artifact, the new worst jump improves while its ISB
and RSB p95 excess values are both somewhat worse; therefore this audit does
not claim every localized metric regressed monotonically in the good direction.
This is the expected honest outcome of removing numerical defects without
introducing a heuristic taper.

The three-cube benchmark command and renderer set `isb_boundary_taper=False`
explicitly, but the current coupled ON/OFF NPZ schema does not serialize that
field. The audit records the executable configuration instead of asserting a
nonexistent artifact key.

## Rendering artifacts

- `artifacts/fullwave/main-port-regression-20260729/single-cube/`
  `single-cube-metal-centered-tx-z042-5ghz-256-empty-calibrated-6k.png`
- `artifacts/fullwave/main-port-regression-20260729/three-cube-320/`
  `three_cube_320_fullwave_vs_deterministic.png`

Both field renderers use the requested orange `inferno` colormap. Synthetic
renderer regressions lock the colormap and real PNG output for single- and
three-cube comparisons.

## Remaining physical work

The result is not evidence that deterministic RT is uniformly correct at all
boundaries. The remaining work is the accepted plan in
`docs/dev/plans/16-deterministic-uniform-boundary-and-surface-current-correctness-plan.md`:
matched uniform ISB/RSB terms, finite-edge endpoint and vertex diffraction,
inter-edge coupling/overlap ownership, then a surface-current residual where
its cost and double-counting rules are explicit. Smoothing remains the final
visualisation-only option, not a physics acceptance mechanism.