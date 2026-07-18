# Plan 09 — Fullwave continuity: state handoff and next-step execution plan

Date: 2026-07-17. Written as a session handoff for the next orchestrating
agent (Fable main thread + Opus subagents/workflows). Read this file, the
two audit docs, and the memory index before doing anything.

Authoritative companion records:
- `docs/dev/audit/fullwave-deterministic-discontinuity-audit.md` (root causes R1–R6 + round-2 addendum)
- `docs/dev/audit/utd-continuity-fix-design.md` (F1–F6, F5c/d/e/f — the as-built math)
- `docs/dev/standards/adr-011-deterministic-coupled-paths.md`, `adr-012-stationary-coupled-diffraction.md`
- Runbook: `docs/dev/fullwave-validation.md`

---

## 1. Repository / build state

All of this work landed on `main` and the temporary worktree was removed —
**future work happens directly on `main` in the main checkout**, off fresh
short-lived branches, not in a `.codex/worktrees/*` copy.

| Item | State |
| --- | --- |
| Working copy | `E:/Code/witwin-platform/channel_native`, branch `main`. Push remote `origin` = `https://github.com/witwin-ai/channel_native.git` (private). Branch off `main` for each new phase; the old `.codex/worktrees/fullwave-ground-truth` worktree and its `codex/fullwave-ground-truth` branch are gone (fully merged). |
| RayD | `E:/Code/RayDi`, branch `main` = `origin/main` = `408a086` (pushed to `github.com/Asixa/RayD.git`), = UTD continuity + the merged RT backend. `dependencies/rayd.lock.json` pins `408a086`. |
| Commit chain | UTD continuity `bb4b092` → 3-cube harness `ea04351` → G1 `6cbb6a9`+RayD `545f1bc` → G2 `d93d0b2`+RayD `6d6b212` → G3/G4 coupled paths `53af8e4`+RayD `9495a70` → plan `863657b` → merge-in-main `73bc724` → RT-integ governance `3ef6ede`+RayD `408a086`. (SHAs before the build-artifact history purge; re-read `git log` for current values.) |
| Preserved data | The irreplaceable gitignored artifacts from the deleted worktree (Maxwell/Tidy3D ground-truth npzs, deterministic baselines, comparison figures, `fable_*`/`render_pil` diagnostic scripts) were copied to `artifacts/fullwave-refs/` in the main checkout (gitignored). Paths in §2/§9 that read `.test-tmp/fullwave-smoke/...` or `artifacts/fullwave-fix/...` now live under `artifacts/fullwave-refs/{fullwave-smoke,fullwave-fix}/...`. |
| Build | Configure a fresh `artifacts/cmake-<name>` in the main checkout (conftest auto-discovers `artifacts/cmake-*`). Recipe: vcvars64 + `cmake -S . -B artifacts/cmake-<name> ...` then `cmake --build <dir> -j 4` (cl.exe OOM → just rerun). Identity guard trips after commits → run `cmake <dir>` once to reconfigure. There is no pre-built pyd anymore — first phase rebuilds. |
| Run env | `PYTHONPATH=<main-checkout>/src`, `WITWIN_CHANNEL_NATIVE_DEVELOPER_OVERRIDE=1`, `WITWIN_CHANNEL_NATIVE_EXTENSION_PATH=<pyd>`, `WITWIN_CHANNEL_NATIVE_EXPECTED_FINGERPRINT=<trimmed sidecar>`. Redirect `$env:TEMP/$env:TMP` to a fresh dir before `ci/run_ci_tier.py` (the shared Windows `pytest-of-Asixa` temp dir permission-locks and yields phantom ERRORs). |
| Fingerprint semantics | The build fingerprint hashes **git SHAs + ABI metadata, not binary bytes** — it does NOT distinguish builds from uncommitted source edits. Use behavior probes / mtimes when in doubt. |

## 2. Single-cube state (DONE, all gates green)

Benchmark vs recorded Maxwell reference (`.test-tmp/fullwave-smoke/visual-maxwell-metal-centered-5ghz-256.npz`, s_empty re-derived = 62.2108):

| Metric | before | after (HEAD) |
| --- | ---: | ---: |
| Envelope NMSE | 0.217 | **0.0358** |
| Magnitude correlation | 0.447 | **0.874** |
| Coherence (one global phase) | 0.807 | **0.890** |
| ISB p95 excess | +6.96 dB | **−0.73 dB** |
| RSB p95 excess | +1.27 dB | **−0.19 dB** |
| Shadow gap median | +3.8 dB | **+1.36 dB** |
| Max adjacent jump | 156.9 dB | 29.8 dB (physical null) |

Fixed defect inventory: 5 cm UTD gate (R1), System-B normalized-basis
polarization (R5; R2 was a dead-code misdiagnosis — the collapse kernel in
`deterministic_field.cu` never fed the solver output), endpointContinuation
branch → angle clamp (R3a), visibility bisector bias (R3b), Fresnel endpoint
truncation + even/odd corner mend + monotone even part + boundary-distance
odd blend (R3c/R4 → F5c/d/e/f, C_BLEND=0.35 Maxwell-calibrated), tx-pol
threading everywhere incl. MC-basic. Regression tests:
`tests/deterministic/test_field_continuity.py` (5 tests, provenance-commented
thresholds). Figures: `artifacts/fullwave-fix/final/single_cube_final_comparison.png`
(rendered by `render_pil.py` — **matplotlib's native draw path crashes on
this box**; only numpy+PIL rendering is reliable, see §7 traps).

Remaining single-cube residuals (documented, not blocking):
- Corner-zone ISB toggle median 1.09 dB vs 0.29 pre-fix baseline (single-variable
  mend limit; p90/frac already better than baseline). Exact fix = P4 below.
- Shadow gap p10/p90 ±4 dB spread (same limitation).

## 3. Three-cube state (coupled paths landed; net budget OPEN)

Machinery landed and verified (ADR-011/012): deterministic coupled R→D/D→R
(config `coupled_paths`, RX-chunked streaming — 86 blocks under the 1M
candidate budget, 6th coherent accumulator slot for cid 3/4, stationary-leg
semantics with external-incident spherical re-extrapolation + real edge
bounds). Coupled-off is bit-identical to pre-change; all suites green;
three-cube solve 1.6 s warm.

Measured effect (256², `artifacts/fullwave-fix/verify-g4/three_cube_coupled_{off,on}.npz`):

| Metric | coupled OFF | coupled ON (G4) |
| --- | ---: | ---: |
| Flagship occlusion-RSB (y=0.457) | 33.2 dB | **11.6 dB** |
| Occlusion pairs improved | — | 19/24 (median 11.9→4.4 dB) |
| >15 dB pairs | 237 | **219** |
| >3 dB pairs | 4,928 | 5,487 |
| Amplitude-weighted budget | — | fixed +6,496 dB vs new −12,741 dB (**net negative**) |
| Coupled component | — | support 38.9%, 2,763 support-toggle pairs, 4,429 internal >6 dB jumps |

Interpretation (verified): the coupled compensator removes the largest
steps but its own **enumeration-level existence boundaries** (specular
point exiting the finite face → needs D(face-edge)→D; segment occlusion by
a third cube → needs R→D→D; order-2 RR boundaries → needs R→R→D) are
order-3 responsibilities — the standard UTD hierarchy: each order heals the
previous order's seams and introduces weaker ones of its own.
IMPORTANT NUANCE: the net-dB budget is a *continuity* metric, not
accuracy-vs-truth. OFF is "smoothly wrong" (whole reflected-shadow regions
missing real energy); ON is "raggedly closer". Only the full-wave arbiter
(P1) can decide the benchmark default. Do not flip `coupled_paths` in
`benchmarks/fullwave_validation/backends.py` before P1 concludes.

## 4. Known residuals (complete list)

1. **Flagship residual 11.6 dB** sits at receivers ~λ/5 (1.2 cm) from the
   compensating PEC edge — kL≈1.3, UTD outside its asymptotic regime; and
   three_cube has **no full-wave reference** (receiver pitch 7.8125 mm not
   Yee-coincident). → P1.
2. **R→R→D missing everywhere** (order-2 RSB 48.6% >3 dB, audit M2) and
   **D→D missing** (pure-D blockage boundaries + coupled sector edges). → P2/P3.
3. **Two-variable corner transition** (generalized Fresnel / complex-pole
   truncated transition integral) — recorded refinement for the corner-zone
   residual and the λ/5 near-field. → P4.
4. Coupled ON global budget net-negative (see §3) — resolved by P1 decision
   + P2 healing.
5. Chores: RayD branch push; drjit committed-PTX regeneration (device
   behavior changed); MC-basic tape keeps γ≡1 semantics (same shared-plane
   disease in its integrand, power-domain so milder — decide after P1);
   `deterministic_field.cu` dead-kernel deprecation cycle (recipe in the
   Phase-F cleanup agent output); Munich perf pin relaxed to 1.5× (watch it).

---

## 5. Execution plan

### P1 — Three-cube full-wave arbiter (FIRST; cheap; decides everything else)

Goal: a versioned, Yee-grid-coincident three-cube FDTD reference and the
ON-vs-truth / OFF-vs-truth comparison.

Steps:
1. Add a versioned `three_cube_320` case to
   `benchmarks/scenarios/fullwave_validation.v1.json` (320×320 receivers,
   6.25 mm pitch, origin phase-locked so receiver cells coincide with
   Maxwell Ez Yee nodes exactly like the single-cube case; keep the old
   256 case untouched — the runbook forbids silent interpolation).
   Update `models.py`/`scenarios.py` fingerprint plumbing + tests.
2. Run witwin-maxwell FDTD (pattern:
   `benchmarks/fullwave_validation/experiments/run_maxwell_single_cube.py`,
   env `WITWIN_MAXWELL_SOURCE=E:\Code\witwin-maxwell`; single-cube took
   ~70 s, three-cube domain is larger — expect a few minutes). Also run the
   empty-scene calibration at the same grid.
3. Deterministic solves at 320²: coupled OFF and ON (same pyd).
4. Metrics both ways (reuse `artifacts/fullwave-fix/final/compute_metrics.py`
   + `fable_shadow_diag.py` patterns): NMSE, magnitude corr, coherence
   (global phase), ISB/RSB p95 excess, per-region gap maps, and the
   flagship line-scan vs truth (does FDTD show a step there too? the λ/5
   near-field may genuinely look like ours).
5. DECIDE: benchmark `coupled_paths` default; whether the +17% small seams
   are visible vs truth; freeze thresholds for the three-cube case in the
   runbook.

Acceptance: reference NPZs versioned + fingerprinted; a comparison table in
the runbook; a decision paragraph in ADR-011 (default ON or OFF, with data).

### P2 — D→D double diffraction (cheapest big win)

Goal: heal (a) pure-D blockage boundaries (diffracted ray occluded by
another body), (b) the coupled component's sector edges from the R-leg's
finite-face termination.

Design notes (verified numbers): candidates/pair = edges² = 36² = 1,296 ≈
same scale as R→D (648×2) → the EXISTING RX-chunked streaming handles it
(~86 blocks, expect ~2× coupled runtime). Enumerate edge-pair candidates
(exclude same-edge; order matters — TX→e1→e2→RX), native discovery mirrors
`discovery/coupled.py` (new cid, e.g. 7). Field evaluation: two wedge
operators in series; the FIRST leg's diffracted field arriving at the
second edge is NOT a spherical wave — use the same frozen-Jones +
external-incident stationary mode on the second leg (ADR-012 machinery)
with the first leg evaluated at the second leg's re-anchored Q via the
spherical re-extrapolation approximation from the first edge's Q (document
the approximation; exact double-diffraction transition functions are P4+
territory). BOTH legs get stationary+mend semantics from day one — the G3
lesson: a compensator without the continuity machinery injects its own
seams and can be net-negative.
Accumulate into the coupled slot 5 (they are the same "higher-order
compensator" family; avoids another public-API change) or a 7th slot —
decide in a short ADR-013 (same tradeoff analysis as ADR-011 D3).
Governance: same checklist as ADR-011 (binding manifest if signatures
change, 7-slot oracle if a new slot, coupled-off bitwise test, no-fallback
negatives, AD lockstep via the shared templates).

Acceptance gates: flagship-class occlusion boundaries where the *diffracted*
field is blocked (find 2-3 from the audit dumps) drop below the coupled-ON
residual; three-cube amplitude-weighted budget (fixed-minus-new) turns
net-positive vs coupled-OFF; single-cube bitwise with everything off; all
suites green; runtime < 10 s warm.

### P3 — R→R→D (targets audit M2)

Candidates/pair = groups²×edges = 18²×36 = 11,664 → ~765 M for 65k RX →
~9× more chunks (same machinery, ~40–60 s solve; consider making it opt-in
`coupled_reflection_depth=2`). Enumeration = extend
`_coupled_reflection_diffraction_topology` with a two-bounce reflection
prefix (image-of-image sources); the D leg reuses ADR-012 semantics with
the double-image source. Only build it if P1 shows the order-2 RSB
boundaries matter vs truth (they are 168 pairs, median 5.4 dB — visible,
but truth may smooth them at this geometry).

### P4 — Two-variable corner transition (the principled residual killer)

Replaces the single-variable γ(u)·B(δ) stand-ins with the complex-pole
truncated transition integral (generalized Fresnel / incomplete transition
function): the odd part's weight becomes T evaluated at the complex-shifted
pole parameter, giving exact behavior through corner rays AND correct
finite-edge transition decay (removes the shadow-gap ±4 dB spread, the
corner-zone ISB median gap, and improves the λ/5 near field). Needs a
device float32 Faddeeva/complex-Fresnel implementation + AD templates.
Validate first in fp64 numpy against (a) brute-force line integrals with
per-sample TRUE off-cone angular arguments, (b) the P1 fullwave reference
at corner rays. This is a research-grade change: oracle-first, host probes,
then the header. Budget: a full session.

### P5 — Chores (fold into whichever phase ships next)

RayD push + PTX regen story; MC-basic γ decision; dead-kernel deprecation;
keep `docs/dev/audit/*` updated per phase; commit per phase (the standing
user requirement).

---

## 6. Working protocol that worked (keep it)

- **Fable main thread**: math derivations, dose-response experiment design,
  diff audits, per-path debugging, gate calibration, commits. Never
  delegate a load-bearing attribution without verifying the numbers
  yourself — two subagent misattributions were caught this way (the R2
  dead-code misdiagnosis; the V2 ripple-test flaw), and one subagent
  correctly caught a Fable analysis error (even/odd shadow dominance).
- **Opus subagents via Workflow**: implementation chunks, builds, sweeps,
  test repairs, audits. Pattern: `phase('Implement'|'Build'|'Verify')`
  with typed schemas; verify agents get explicit numeric gates + the
  instruction to per-path-decompose the worst failure before returning.
- **Instrument first**: every fix started from per-path dumps
  (`export_paths=True` → PathTable rx_id/edge_id/field_xyz) + offline
  numpy geometry (Fermat scans, gate classification). Diagnostic script
  library: `artifacts/fullwave-fix/{instrument,verify-*}/**`, esp.
  `fable_vertex_step.py`, `fable_shadow_diag.py`, `fable_ripple_test.py`.
- **Dose-response before implementation**: candidate fixes are validated as
  offline reweightings of per-path dumps vs Maxwell BEFORE touching CUDA —
  but model the kernel's ACTUAL even/odd structure (the V2 lesson).
- Commit per verified phase; commit docs EARLY (an agent once wiped an
  untracked audit doc).

## 7. Traps (all hit this session; do not rediscover)

1. matplotlib native draw (savefig/tight_layout/add_patch) crashes with a
   delay-load fault on this box — render figures with numpy+PIL only
   (`artifacts/fullwave-fix/final/render_pil.py` is the template).
2. cl.exe transient OOM at `-j` high: "ninja: build stopped" with only
   C4267 warnings → rerun; real failures have an `error C` line.
3. Build-identity guard trips after ANY commit (dirty-state change) →
   `cmake <build-dir>` reconfigure once.
4. Fingerprint = metadata hash; identical across uncommitted source edits.
5. Parallel agents sharing `artifacts/cmake-fw`: snapshot the pyd + sidecar
   to a private dir and pin the env override there.
6. pytest `-k "los or reflection or diffraction"` filters TEST NAMES — it
   silently skipped the forward-parity tests for two whole phases.
7. Experiment grids must be phase-locked to `−0.796875 + k·0.00625` to hit
   Maxwell Ez nodes (tolerance 1e-4; the Maxwell npz coords carry float32
   noise).
8. The welded-wedge AD fixture must stay OFF exact mirror symmetry (branch
   cut + off-edge Fermat → codegen-dependent near-null).
9. `fast_math` is ONLY for `field_wedge_ad_diffraction.cu` (matches the
   OptiX raygen); the other AD kernels mirror precise-math primals.
10. Munich strict-parity subprocess test: known-flaky timing race.

## 8. Opus prompt skeleton (copy-adapt per phase)

```
const COMMON = `
Context docs (READ FIRST): <worktree>/docs/dev/plans/09-fullwave-continuity-next-steps.md,
docs/dev/audit/utd-continuity-fix-design.md, docs/dev/standards/adr-011..013.
RayD E:/Code/RayDi branch fix/utd-continuity HEAD <sha> (no commits, no
git-restore/checkout/clean of tracked files). Worktree <path> branch
codex/fullwave-ground-truth HEAD <sha> (same rules; main thread commits).
English comments stating constraints. Python C:/Users/Asixa/miniconda3/envs/witwin2/python.exe.
pyd env: PYTHONPATH=<worktree>/src, WITWIN_CHANNEL_NATIVE_DEVELOPER_OVERRIDE=1,
WITWIN_CHANNEL_NATIVE_EXTENSION_PATH=<pyd>, WITWIN_CHANNEL_NATIVE_EXPECTED_FINGERPRINT=<sidecar>.
CLAUDE.md governance in full force (binding manifest / contract coverage /
owner inventory move together; never weaken tests; no fallbacks).`
```

Implement-agent prompts: exact file:line spec from the main thread's design;
mandatory host probe (float+Dual) with numeric pass criteria; "deviations
must be justified in notes"; structured output schema
{success, files_changed, <domain fields>, notes}.
Build-agent: the vcvars64+ninja recipe + the two traps.
Verify-agent: numbered gates with numeric thresholds and BEFORE columns from
this file; "if a gate fails, per-path decompose the worst pair and name the
mechanism precisely before returning"; artifact list required.

## 9. Key artifacts index

- Single-cube: baseline `artifacts/fullwave-fix/baseline-deterministic.npz`;
  current `verify-g2/winner/after.npz`; metrics `final/final-metrics.json`;
  figures `final/{single_cube_final_comparison,three_cube_final_components}.png`.
- Three-cube: audit `artifacts/fullwave-fix/threecube/`; coupled off/on +
  path tables `verify-g4/three_cube_*.npz`; scaling measurements
  `threecube-g3/coupled_scaling_results.json`.
- Maxwell references (single-cube only): `.test-tmp/fullwave-smoke/visual-maxwell-*.npz`.
- Diagnostic scripts: `final/fable_*.py`, `final/render_pil.py`,
  `instrument/_env.py` (scene builder + env), `verify-e1/decomp1b_v2.py`
  (seam/null splitter).
