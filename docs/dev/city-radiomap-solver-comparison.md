# City radiomap solver comparison - Munich and San Francisco vs Sionna 2.0.1

Date: 2026-07-18. Records the four-way solver comparisons (Deterministic /
MC-basic / BDPT / Sionna RT 2.0.1) produced after the ADR-018/019/020 model
alignment and the path-solver performance fix. Sionna runs use the local
copy at `channel/reference/sionna-rt-reference-2.0.1/src` (drjit 1.3.1 +
mitsuba 3.8.0, witwin2, separate process, npz handoff). All artifacts,
scripts, raw npz, and verification JSONs live under
`artifacts/four-way/` (gitignored; regenerate with the scripts there).

## Orientation convention (bug record)

The first sheets rendered the witwin panels TRANSPOSED: native `path_gain`
is laid out `[tx, worldY, worldX]` (same as Sionna's RadioMap), but the
renderer assumed `[worldX, worldY]`; a square 256x256 grid hid the
transpose from every shape check. Fixed via a per-solver `canonical()`
mapping (det/BDPT `[1,NY,NX] -> .T`; MC-basic flat y-major
`reshape(NY,NX).T`), verified by dihedral NCC against the Sionna panel
(identity 0.67-0.80 vs transpose ~0.35;
`artifacts/four-way/orientation_verification.json`). All sheets are
"N up / E right" (Sionna convention). Lesson: never validate raster
orientation on a square grid without an asymmetric-landmark check.

## Munich - diffraction ON, like-for-like (`munich_4way.png`)

Config: sionna `munich.xml`, TX=(8.5, 21, 27), extent x[-120,120]
y[-120,140], 256x256, z=1.5 m, 2.4 GHz, depth 3,
{los, reflection, diffraction}; Sionna with `diffraction=True` +
`edge_diffraction=True`; iso 1x1 V-pol, unit tx power; per-cell
path gain in dBW, shared 60 dB scale.

| Solver | time | nonzero cells | notes |
| --- | ---: | ---: | --- |
| Deterministic | 834 ms | 42,629 | coherent (interference fringes) |
| MC-basic (1.05M smp) | 7 ms | 19,217 | diffraction sampling speckle |
| BDPT (1.05M smp) | 884 ms | 42,629 | power-domain, matches det support exactly |
| Sionna (20M smp) | 46 ms | 29,582 | incoherent radiomap, smooth |

Verdict: footprint, main lobe, street-canyon diffraction fill, and shadow
gradients agree closely across Deterministic/BDPT/Sionna; the witwin
solvers resolve ~44% more diffraction-lit cells than Sionna at these
sample counts. Diffraction OFF baseline (earlier sheet): det 16,420 vs
Sionna 16,736 nonzero (1.9% apart) - near-parity on the pure
LoS+reflection problem. Known semantic difference: det is coherent,
Sionna/MC/BDPT power-domain.

## San Francisco - flat plane at the official extent (`sf_4way.png`)

Config: `benchmarks/bench_sf_planar_radiomap.py` (TX=(468,106,70), full
scene x[-520,720] y[-480,470], 256x256, z=1.5 m flat, 3.5 GHz).

- Depth 1, {los, reflection}: det 7 / MC 2 / BDPT 7 nonzero vs Sionna
  142 (100M rays). The live Sionna run reproduces the plan-02 recorded
  142 / 2.93e-10 exactly, closing that parity question: the gap is
  image-method exact-sparse specular vs ray-launch grazing spread; power
  sums agree within one order (0.145x).
- The flat z=1.5 plane is BURIED for most of the scene: SF terrain spans
  0-170 m and 99.5% of vertices sit above z=1.5. A flat-plane SF map is
  therefore near-empty by construction for every engine - this was the
  root cause of the earlier "SF result totally disagrees with Sionna"
  observation, not a solver defect.
- Diffraction ON, depth 3 (bonus): det/BDPT 38,984 nonzero cells vs the
  7-cell depth-1 baseline (`artifacts/sf-sionna/`).

## San Francisco - terrain-following (authoritative tutorial config, `sf_terrain_4way.png`)

The only Sionna-SF radiomap code anywhere in the channel package is the
upstream NVIDIA tutorial `tutorials/Radio-Maps.ipynb` (cells 48/50/52):
measurement surface = Terrain clone +1.5 m (192,000 triangle cells,
z 1.5-105 m), TX=(468,106,70), 3.5 GHz, depth 5, 1e8 samples, refraction
ON, diffraction OFF.

All three witwin solvers accept the terrain surface directly: the 192k
Sionna cell centers were fed as arbitrary receiver points (PATH natively;
Deterministic per-point; BDPT via `receiver_strategy="point_sphere"`).
For the enumerated component set {los, reflection, diffraction} with
coupled off, the three witwin maps are bit-identical by construction (one
shared enumerated engine) - labeled honestly on the sheet.

| Solver | time | nonzero / 192k | p50 / p95 (dB) | power sum |
| --- | ---: | ---: | --- | ---: |
| witwin PATH (depth 3) | 150 s | 125,455 | -133.5 / -97.5 | 2.15e-5 |
| witwin Deterministic | 35 s | 125,455 | same (bit-identical) | 2.15e-5 |
| witwin BDPT | 72 s | 125,455 | same (bit-identical) | 2.15e-5 |
| Sionna default (depth 5, refraction ON) | 0.2 s | 105,221 | -107.8 / -91.4 | 5.52e-5 |
| Sionna +diffraction | 0.7 s | 105,270 | -107.8 / -91.4 | - |

Findings:

- Orientation identity verified for all panels; the TX street-canyon LoS
  beam, the diagonal avenue band, and hillside coverage co-register.
- witwin covers ~20k MORE weak shadow cells (enumerated diffraction);
  Sionna is ~25 dB brighter at the median. Known contributors, in order:
  (1) component mismatch - the tutorial default has REFRACTION
  (building transmission) ON, which floods interior-adjacent terrain
  cells; the witwin runs had transmission off; (2) depth 5 vs 3 (witwin
  depth-3 PATH already 150 s at 192k receivers); (3) coherent vs
  power-domain semantics. A transmission-on, depth-matched rerun is the
  recorded next step for closing the gap.
- Sionna's own diffraction adds only +49 cells on the terrain surface -
  negligible there, whereas Munich shows the witwin enumerated
  diffraction advantage clearly (42,629 vs 29,582 lit cells).
- Sionna's 1e8-sample solves in 0.2-0.7 s reflect drjit's fused
  megakernel; witwin's terrain-surface solves are receiver-streamed
  (35-150 s at 192k arbitrary points). City-scale receiver throughput is
  part of the recorded deterministic acceleration follow-up.
